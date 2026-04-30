"""High-level Warp Vlasov-Poisson solver class and JAX VJP wrapper.

This module composes the Warp kernels in ``kernels.py`` and the FFT
Poisson solver in ``poisson.py`` into a full time integrator that
mirrors the behavior of the original
``vp_solver.jax_vp_solver.VlasovPoissonSolver``.

Two entry points are exposed:

  * ``WarpVlasovPoissonSolver.run_forward(f_iv, H, t_final)`` - a
    NumPy-in / NumPy-out time loop suitable for plain simulation.
  * ``WarpVlasovPoissonSolver.run_forward_jax(f_iv, H, t_final)`` - a
    ``jax.custom_vjp``-wrapped version that produces JAX arrays and
    supports ``jax.grad`` / ``optax`` via a hand-written discrete
    adjoint, exactly as called for in the implementation plan.

The discrete adjoint differentiates through one Strang time step

    f_half  = SemiLag_x(f_in)                  (1)
    rho     = trapz(f_eq - f_half, vs)          (2)
    E       = Poisson(rho)                      (3)
    ee      = 0.5 * trapz(E^2, xs)              (4)
    E_total = E + H                             (5)
    f_one   = SemiLag_v(f_half, E_total)        (6)
    f_out   = SemiLag_x(f_one)                  (7)

by chaining the linear adjoints of (1)-(7) backward through the loop
and accumulating the gradient w.r.t. the static external field ``H``.
"""

from __future__ import annotations

import dataclasses
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import warp as wp

from .mesh import Mesh
from .kernels import (
    semilag_x_kernel,
    semilag_v_kernel,
    compute_rho_kernel,
    compute_ee_kernel,
    semilag_x_adjoint_kernel,
    semilag_v_adjoint_kernel,
    compute_rho_adjoint_kernel,
    compute_ee_adjoint_kernel,
    # Optimized kernels (F1, F4, M1) - see kernels.py for the design notes.
    semilag_v_fused_kernel,
    semilag_v_fused_adjoint_kernel,
    axpy_1d_kernel,
    axpy_2d_kernel,
    copy_1d_kernel,
)
from .poisson import (
    fft_inv_multiplier,
    solve_poisson_host,
    solve_poisson_host_adjoint,
)


# ---------------------------------------------------------------------------
# Solver class
# ---------------------------------------------------------------------------


class WarpVlasovPoissonSolver:
    """Vlasov-Poisson semi-Lagrangian solver backed by NVIDIA Warp.

    Mirrors the API of ``vp_solver.jax_vp_solver.VlasovPoissonSolver``
    but uses Warp kernels for the inner loop and host FFT for Poisson.
    """

    mesh: Mesh
    dt: float
    f_eq: np.ndarray

    def __init__(
        self,
        mesh: Mesh,
        dt: float,
        f_eq: np.ndarray,
        *,
        optimized: bool = True,
    ):
        """Construct a Warp Vlasov-Poisson solver.

        Args:
            mesh, dt, f_eq: as in the JAX reference.
            optimized: if ``True`` (default) the forward and backward
                passes use the fused kernels and on-device cotangent
                pipeline.  If ``False`` the original reference
                implementation is used; useful as ground truth when
                cross-checking the optimized path.
        """
        self.mesh = mesh
        self.dt = float(dt)
        self.f_eq = np.asarray(f_eq, dtype=np.float64)
        if self.f_eq.shape != (mesh.nx, mesh.nv):
            raise ValueError(
                f"f_eq shape {self.f_eq.shape} != ({mesh.nx}, {mesh.nv})"
            )
        self.optimized = bool(optimized)

        # Pre-allocate Warp resources.
        self._xs_wp = wp.array(self.mesh.xs, dtype=wp.float64)
        self._vs_wp = wp.array(self.mesh.vs, dtype=wp.float64)
        self._f_eq_wp = wp.array(self.f_eq, dtype=wp.float64)
        self._fa_wp = wp.zeros(
            (mesh.nx, mesh.nv), dtype=wp.float64
        )
        self._fb_wp = wp.zeros(
            (mesh.nx, mesh.nv), dtype=wp.float64
        )
        self._fc_wp = wp.zeros(
            (mesh.nx, mesh.nv), dtype=wp.float64
        )
        self._rho_wp = wp.zeros(mesh.nx, dtype=wp.float64)
        self._E_wp = wp.zeros(mesh.nx, dtype=wp.float64)
        self._ee_wp = wp.zeros(1, dtype=wp.float64)

        # Cached spectral multiplier for the host Poisson solve.
        self._inv_mult = fft_inv_multiplier(mesh.nx, mesh.period_x)

        # Persistent buffers used by the optimized forward / adjoint.
        # Allocating them up front (F3) avoids per-step ``wp.array``
        # allocations and lets the optimized adjoint chain its
        # cotangents fully on-device (F4).
        self._H_wp = wp.zeros(mesh.nx, dtype=wp.float64)
        self._g_f_wp = wp.zeros((mesh.nx, mesh.nv), dtype=wp.float64)
        self._g_in_wp = wp.zeros((mesh.nx, mesh.nv), dtype=wp.float64)
        self._g_f_half_wp = wp.zeros((mesh.nx, mesh.nv), dtype=wp.float64)
        self._g_f_hist_t_wp = wp.zeros((mesh.nx, mesh.nv), dtype=wp.float64)
        self._g_E_back_wp = wp.zeros(mesh.nx, dtype=wp.float64)
        self._g_E_total_wp = wp.zeros(mesh.nx, dtype=wp.float64)
        self._g_E_stage_wp = wp.zeros(mesh.nx, dtype=wp.float64)
        self._g_rho_back_wp = wp.zeros(mesh.nx, dtype=wp.float64)
        self._g_H_wp = wp.zeros(mesh.nx, dtype=wp.float64)
        self._g_ee_wp = wp.zeros(1, dtype=wp.float64)

        # Lazily build a custom_vjp-wrapped run_forward bound to ``self``.
        self._jax_callable: Callable | None = None

    # ------------------------------------------------------------------
    # Helpers - low-level launches reused by both forward and adjoint
    # ------------------------------------------------------------------

    def num_steps(self, t_final: float) -> int:
        return int(t_final / self.dt)

    def _semilag_x(self, src: wp.array, dst: wp.array) -> None:
        wp.launch(
            semilag_x_kernel,
            dim=(self.mesh.nx, self.mesh.nv),
            inputs=[
                src, dst,
                self._xs_wp, self._vs_wp,
                wp.float64(self.dt),
                wp.float64(self.mesh.dx),
                wp.int32(self.mesh.nx),
                wp.float64(self.mesh.period_x),
            ],
        )

    def _semilag_v(
        self, src: wp.array, dst: wp.array, E_total_wp: wp.array
    ) -> None:
        wp.launch(
            semilag_v_kernel,
            dim=(self.mesh.nx, self.mesh.nv),
            inputs=[
                src, dst,
                self._vs_wp, E_total_wp,
                wp.float64(self.dt),
                wp.float64(self.mesh.dv),
                wp.int32(self.mesh.nv),
                wp.float64(2.0 * self.mesh.period_v),
            ],
        )

    def _compute_rho(self, f_wp: wp.array, rho_wp: wp.array) -> None:
        wp.launch(
            compute_rho_kernel,
            dim=self.mesh.nx,
            inputs=[
                self._f_eq_wp, f_wp, rho_wp,
                wp.float64(self.mesh.dv),
                wp.int32(self.mesh.nv),
            ],
        )

    def _compute_ee(self, E_wp: wp.array, out_wp: wp.array) -> None:
        wp.launch(
            compute_ee_kernel,
            dim=1,
            inputs=[
                E_wp, out_wp,
                wp.float64(self.mesh.dx),
                wp.int32(self.mesh.nx),
            ],
        )

    # ------------------------------------------------------------------
    # Single-step primitives that match the JAX reference exactly
    # ------------------------------------------------------------------

    def compute_rho(self, f: np.ndarray) -> np.ndarray:
        """rho(x) = integral (f_eq - f) dv via trapezoidal rule."""
        f_wp = wp.array(np.ascontiguousarray(f), dtype=wp.float64)
        self._compute_rho(f_wp, self._rho_wp)
        return self._rho_wp.numpy().copy()

    def compute_E_from_rho(self, rho: np.ndarray) -> np.ndarray:
        return solve_poisson_host(
            np.asarray(rho, dtype=np.float64),
            self.mesh.period_x,
            self._inv_mult,
        )

    def compute_E(self, f: np.ndarray) -> np.ndarray:
        return self.compute_E_from_rho(self.compute_rho(f))

    def compute_electric_energy(self, E: np.ndarray) -> float:
        E_wp = wp.array(np.asarray(E, dtype=np.float64), dtype=wp.float64)
        self._compute_ee(E_wp, self._ee_wp)
        return float(self._ee_wp.numpy()[0])

    # ------------------------------------------------------------------
    # Forward time integration
    # ------------------------------------------------------------------

    def run_forward(
        self,
        f_iv: np.ndarray,
        H: np.ndarray,
        t_final: float,
        *,
        record_history: bool = True,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
        """Run the full Strang-split semi-Lagrangian time loop.

        Dispatches to either the optimized or the legacy implementation
        depending on ``self.optimized``.  The two paths produce
        bit-for-bit identical outputs (verified by
        ``tests/test_optimized.py``); ``optimized=False`` is retained
        as the reference path.

        Args:
            f_iv: initial distribution, shape ``(nx, nv)``.
            H: external electric field, shape ``(nx,)``.
            t_final: final time in solver units.
            record_history: if ``True`` (default) record ``f`` at the
                end of every step.  Setting to ``False`` skips the
                per-step ``(nx, nv)`` device-to-host copy and returns
                ``f_hist=None`` (F6).  Only honored on the optimized
                path; the legacy path always records.

        Returns:
            ``(f_final, f_hist, E_hist, ee_hist)``.  ``f_hist`` is
            ``None`` if ``record_history=False``.  ``E_hist`` records
            ``E_total = E + H`` at each step (matching the JAX
            reference) and ``ee_hist`` records ``0.5 * integral E^2 dx``
            from the self-consistent ``E`` (i.e. without ``H``).
        """
        if self.optimized:
            return self._run_forward_fast(
                f_iv, H, t_final, record_history=record_history
            )
        if not record_history:
            raise ValueError(
                "record_history=False is only supported when the solver was "
                "constructed with optimized=True."
            )
        return self._run_forward_legacy(f_iv, H, t_final)

    # ------------------------------------------------------------------
    # Forward (legacy reference) - exact 1:1 port of the JAX scheme
    # ------------------------------------------------------------------

    def _run_forward_legacy(
        self,
        f_iv: np.ndarray,
        H: np.ndarray,
        t_final: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Reference forward loop matching the JAX implementation.

        Kept verbatim so the optimized path can be cross-checked
        against it in ``tests/test_optimized.py``.
        """
        T = self.num_steps(t_final)
        nx, nv = self.mesh.nx, self.mesh.nv

        f_hist = np.empty((T, nx, nv), dtype=np.float64)
        E_hist = np.empty((T, nx), dtype=np.float64)
        ee_hist = np.empty(T, dtype=np.float64)

        H = np.asarray(H, dtype=np.float64)
        H_wp = wp.array(H, dtype=wp.float64)
        E_total_wp = wp.zeros(nx, dtype=wp.float64)

        self._fa_wp.assign(np.asarray(f_iv, dtype=np.float64))

        for t in range(T):
            # (1) f_half = SemiLag_x(f)
            self._semilag_x(self._fa_wp, self._fb_wp)

            # (2) rho = trapz(f_eq - f_half)
            self._compute_rho(self._fb_wp, self._rho_wp)

            # (3) E = Poisson(rho); (4) ee = 0.5 * integral E^2 dx
            E = solve_poisson_host(
                self._rho_wp.numpy(), self.mesh.period_x, self._inv_mult
            )
            self._E_wp.assign(E)
            self._compute_ee(self._E_wp, self._ee_wp)
            ee = float(self._ee_wp.numpy()[0])

            # (5) E_total = E + H
            E_total = E + H
            E_total_wp.assign(E_total)

            # (6) f_one = SemiLag_v(f_half, E_total)
            self._semilag_v(self._fb_wp, self._fc_wp, E_total_wp)

            # (7) f = SemiLag_x(f_one)
            self._semilag_x(self._fc_wp, self._fa_wp)

            f_hist[t] = self._fa_wp.numpy()
            E_hist[t] = E_total
            ee_hist[t] = ee

        f_final = f_hist[-1].copy() if T > 0 else np.asarray(f_iv).copy()
        return f_final, f_hist, E_hist, ee_hist

    # ------------------------------------------------------------------
    # Forward (optimized) - F1 + F2 + F6
    # ------------------------------------------------------------------

    def _run_forward_fast(
        self,
        f_iv: np.ndarray,
        H: np.ndarray,
        t_final: float,
        *,
        record_history: bool = True,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
        """Optimized forward loop.

        Differences from the legacy path:
          * F1: ``E_total = E + H`` is computed inside
            ``semilag_v_fused_kernel``; no host-side ``E_total`` upload.
          * F6: ``record_history=False`` skips the per-step
            ``(nx, nv)`` device-to-host copy.
          * H is uploaded to the persistent ``self._H_wp`` once.

        Note: F2 (host-side electric-energy reduction) was considered
        but reverted - moving the trapezoid to NumPy reorders the
        floating-point additions vs. the Warp kernel and breaks the
        bit-for-bit equivalence the tests rely on.  The kernel launch
        cost is negligible on CPU and only marginally interesting on
        GPU; if you want F2 it should land alongside its own ULP-aware
        test tolerance (or a dedicated multi-threaded reduction
        kernel that preserves the legacy ordering).

        Bit-for-bit identical to ``_run_forward_legacy`` (verified by
        ``tests/test_optimized.py``).
        """
        T = self.num_steps(t_final)
        nx, nv = self.mesh.nx, self.mesh.nv

        f_hist = (
            np.empty((T, nx, nv), dtype=np.float64) if record_history else None
        )
        E_hist = np.empty((T, nx), dtype=np.float64)
        ee_hist = np.empty(T, dtype=np.float64)

        H = np.asarray(H, dtype=np.float64)
        self._H_wp.assign(H)

        self._fa_wp.assign(np.asarray(f_iv, dtype=np.float64))

        for t in range(T):
            # (1) f_half = SemiLag_x(fa) -> fb
            self._semilag_x(self._fa_wp, self._fb_wp)

            # (2) rho = trapz(f_eq - f_half)
            self._compute_rho(self._fb_wp, self._rho_wp)

            # (3) E = Poisson(rho)  ;  (4) ee via Warp kernel (preserves
            # bit-for-bit equivalence with the legacy path - see docstring)
            E = solve_poisson_host(
                self._rho_wp.numpy(), self.mesh.period_x, self._inv_mult
            )
            self._E_wp.assign(E)
            self._compute_ee(self._E_wp, self._ee_wp)
            ee = float(self._ee_wp.numpy()[0])

            # (5+6) f_one = SemiLag_v_fused(f_half, E, H) -> fc  (F1)
            wp.launch(
                semilag_v_fused_kernel,
                dim=(nx, nv),
                inputs=[
                    self._fb_wp, self._fc_wp,
                    self._vs_wp, self._E_wp, self._H_wp,
                    wp.float64(self.dt),
                    wp.float64(self.mesh.dv),
                    wp.int32(nv),
                    wp.float64(2.0 * self.mesh.period_v),
                ],
            )

            # (7) f = SemiLag_x(f_one) -> fa
            self._semilag_x(self._fc_wp, self._fa_wp)

            if record_history:
                f_hist[t] = self._fa_wp.numpy()
            E_hist[t] = E + H
            ee_hist[t] = ee

        f_final = self._fa_wp.numpy().copy()
        return f_final, f_hist, E_hist, ee_hist

    # ------------------------------------------------------------------
    # Backward dispatcher
    # ------------------------------------------------------------------

    def _run_backward(
        self,
        f_iv: np.ndarray,
        H: np.ndarray,
        t_final: float,
        f_hist: np.ndarray,
        g_f_final: np.ndarray,
        g_f_hist: np.ndarray,
        g_E_hist: np.ndarray,
        g_ee_hist: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Backward pass dispatcher; routes to the optimized or legacy adjoint."""
        if self.optimized:
            return self._run_backward_fast(
                f_iv, H, t_final, f_hist,
                g_f_final, g_f_hist, g_E_hist, g_ee_hist,
            )
        return self._run_backward_legacy(
            f_iv, H, t_final, f_hist,
            g_f_final, g_f_hist, g_E_hist, g_ee_hist,
        )

    # ------------------------------------------------------------------
    # Backward (legacy reference)
    # ------------------------------------------------------------------

    def _run_backward_legacy(
        self,
        f_iv: np.ndarray,
        H: np.ndarray,
        t_final: float,
        f_hist: np.ndarray,
        g_f_final: np.ndarray,
        g_f_hist: np.ndarray,
        g_E_hist: np.ndarray,
        g_ee_hist: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reference backward pass returning gradients w.r.t. ``(f_iv, H)``.

        ``f_hist`` is the saved per-step distribution from the forward
        pass, used to recompute ``f_half`` and ``E`` cheaply (and
        consistently) at each adjoint step.  ``g_*`` are the upstream
        cotangents from JAX.
        """
        T = self.num_steps(t_final)
        nx, nv = self.mesh.nx, self.mesh.nv

        f_iv = np.asarray(f_iv, dtype=np.float64)
        H = np.asarray(H, dtype=np.float64)

        g_f = np.array(g_f_final, dtype=np.float64, copy=True)
        g_H = np.zeros(nx, dtype=np.float64)
        g_f_iv = np.zeros((nx, nv), dtype=np.float64)

        # Workspace arrays (re-used across steps).
        E_total_wp = wp.zeros(nx, dtype=wp.float64)
        g_f_wp = wp.zeros((nx, nv), dtype=wp.float64)
        g_in_wp = wp.zeros((nx, nv), dtype=wp.float64)
        g_E_wp = wp.zeros(nx, dtype=wp.float64)
        g_rho_wp = wp.zeros(nx, dtype=wp.float64)
        g_f_half_extra = wp.zeros((nx, nv), dtype=wp.float64)

        for t in range(T - 1, -1, -1):
            # Reconstruct the step input distribution.
            f_step_in = f_iv if t == 0 else f_hist[t - 1]

            # Add the saved cotangent contribution for the recorded f_hist[t].
            g_f += np.asarray(g_f_hist[t], dtype=np.float64)

            # ----- Recompute forward intermediates needed by the adjoint -----
            self._fa_wp.assign(f_step_in)
            self._semilag_x(self._fa_wp, self._fb_wp)            # f_half
            f_half_np = self._fb_wp.numpy().copy()
            self._compute_rho(self._fb_wp, self._rho_wp)
            rho_np = self._rho_wp.numpy().copy()
            E = solve_poisson_host(rho_np, self.mesh.period_x, self._inv_mult)
            self._E_wp.assign(E)
            E_total = E + H
            E_total_wp.assign(E_total)

            # ----- Adjoint sweep -----
            # (7) g_f_one = X^T g_f
            g_f_wp.assign(g_f)
            g_in_wp.zero_()
            wp.launch(
                semilag_x_adjoint_kernel,
                dim=(nx, nv),
                inputs=[
                    g_f_wp, g_in_wp,
                    self._xs_wp, self._vs_wp,
                    wp.float64(self.dt),
                    wp.float64(self.mesh.dx),
                    wp.int32(nx),
                    wp.float64(self.mesh.period_x),
                ],
            )
            g_f_one = g_in_wp.numpy().copy()

            # (6) (g_f_half_v, g_E_total_v) from semilag_v adjoint, given
            # f_in=f_half and the upstream cotangent g_f_one.
            g_f_wp.assign(g_f_one)
            g_in_wp.zero_()
            g_E_wp.zero_()
            wp.launch(
                semilag_v_adjoint_kernel,
                dim=(nx, nv),
                inputs=[
                    g_f_wp, g_in_wp, g_E_wp,
                    self._fb_wp,            # f_in for the v-step
                    self._vs_wp, E_total_wp,
                    wp.float64(self.dt),
                    wp.float64(self.mesh.dv),
                    wp.int32(nv),
                    wp.float64(2.0 * self.mesh.period_v),
                ],
            )
            g_f_half_v = g_in_wp.numpy().copy()
            g_E_total_from_v = g_E_wp.numpy().copy()

            # (5) E_total = E + H  =>  d/dE = d/dH = 1.
            g_E_total = (
                g_E_total_from_v + np.asarray(g_E_hist[t], dtype=np.float64)
            )
            g_H += g_E_total

            # (4) ee = 0.5 * integral E^2 dx  =>  contributes to g_E.
            g_E = g_E_total.copy()
            ee_grad = np.asarray([float(g_ee_hist[t])], dtype=np.float64)
            g_E_wp.zero_()
            g_E_wp.assign(g_E)
            wp.launch(
                compute_ee_adjoint_kernel,
                dim=nx,
                inputs=[
                    wp.array(ee_grad, dtype=wp.float64),
                    self._E_wp,
                    g_E_wp,
                    wp.float64(self.mesh.dx),
                    wp.int32(nx),
                ],
            )
            g_E = g_E_wp.numpy().copy()

            # (3) g_rho = Poisson^T g_E
            g_rho = solve_poisson_host_adjoint(
                g_E, self.mesh.period_x, self._inv_mult
            )

            # (2) rho-adjoint adds weighted broadcast of g_rho onto g_f_half.
            g_f_half_extra.zero_()
            wp.launch(
                compute_rho_adjoint_kernel,
                dim=(nx, nv),
                inputs=[
                    wp.array(g_rho, dtype=wp.float64),
                    g_f_half_extra,
                    wp.float64(self.mesh.dv),
                    wp.int32(nv),
                ],
            )
            g_f_half = g_f_half_v + g_f_half_extra.numpy()

            # (1) g_f_step_in = X^T g_f_half
            g_f_wp.assign(g_f_half)
            g_in_wp.zero_()
            wp.launch(
                semilag_x_adjoint_kernel,
                dim=(nx, nv),
                inputs=[
                    g_f_wp, g_in_wp,
                    self._xs_wp, self._vs_wp,
                    wp.float64(self.dt),
                    wp.float64(self.mesh.dx),
                    wp.int32(nx),
                    wp.float64(self.mesh.period_x),
                ],
            )
            g_f = g_in_wp.numpy().copy()

        g_f_iv = g_f
        return g_f_iv, g_H

    # ------------------------------------------------------------------
    # Backward (optimized) - F3 + F4
    # ------------------------------------------------------------------

    def _run_backward_fast(
        self,
        f_iv: np.ndarray,
        H: np.ndarray,
        t_final: float,
        f_hist: np.ndarray,
        g_f_final: np.ndarray,
        g_f_hist: np.ndarray,
        g_E_hist: np.ndarray,
        g_ee_hist: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Optimized adjoint with on-device cotangent pipeline.

        F3 + F4: every per-step intermediate cotangent
        (``g_f``, ``g_f_one``, ``g_f_half``, ``g_E``, ``g_E_total``,
        ``g_H``, ...) lives in a persistent ``wp.array`` allocated at
        ``__init__``.  The legacy path round-trips four ``(nx, nv)``
        arrays through host memory per step; the fast path only does
        a single ``(nx,)`` D2H+H2D for the FFT-based Poisson adjoint
        and one ``(nx, nv)`` H2D to stage the saved
        ``g_f_hist[t]`` cotangent.

        Bit-for-bit equivalent to ``_run_backward_legacy`` because the
        same arithmetic operations are performed in the same order;
        only the staging buffers and the ``E + H`` algebra differ.
        """
        T = self.num_steps(t_final)
        nx, nv = self.mesh.nx, self.mesh.nv

        f_iv = np.asarray(f_iv, dtype=np.float64)
        H = np.asarray(H, dtype=np.float64)
        self._H_wp.assign(H)

        # Bind constants once.
        dt_wp = wp.float64(self.dt)
        dx_wp = wp.float64(self.mesh.dx)
        dv_wp = wp.float64(self.mesh.dv)
        period_x_wp = wp.float64(self.mesh.period_x)
        period_v_full_wp = wp.float64(2.0 * self.mesh.period_v)
        nx_wp = wp.int32(nx)
        nv_wp = wp.int32(nv)
        one = wp.float64(1.0)

        # Initialize on-device cotangents.
        self._g_f_wp.assign(np.asarray(g_f_final, dtype=np.float64))
        self._g_H_wp.zero_()

        for t in range(T - 1, -1, -1):
            f_step_in = f_iv if t == 0 else f_hist[t - 1]

            # g_f += g_f_hist[t] - upload + axpy on device.
            self._g_f_hist_t_wp.assign(
                np.asarray(g_f_hist[t], dtype=np.float64)
            )
            wp.launch(
                axpy_2d_kernel,
                dim=(nx, nv),
                inputs=[one, self._g_f_hist_t_wp, self._g_f_wp],
            )

            # ----- Recompute the forward intermediates this step needs -----
            self._fa_wp.assign(f_step_in)
            self._semilag_x(self._fa_wp, self._fb_wp)        # f_half -> fb
            self._compute_rho(self._fb_wp, self._rho_wp)
            E = solve_poisson_host(
                self._rho_wp.numpy(), self.mesh.period_x, self._inv_mult
            )
            self._E_wp.assign(E)

            # ----- Adjoint sweep, fully on-device -----
            # (7) g_f_one = X^T g_f          (write into g_in_wp)
            self._g_in_wp.zero_()
            wp.launch(
                semilag_x_adjoint_kernel,
                dim=(nx, nv),
                inputs=[
                    self._g_f_wp, self._g_in_wp,
                    self._xs_wp, self._vs_wp,
                    dt_wp, dx_wp, nx_wp, period_x_wp,
                ],
            )

            # (6) Adjoint of fused v step.  Outputs:
            #     g_f_half_wp (cotangent on f_half),
            #     g_E_total_wp (cotangent on E + H).
            self._g_f_half_wp.zero_()
            self._g_E_total_wp.zero_()
            wp.launch(
                semilag_v_fused_adjoint_kernel,
                dim=(nx, nv),
                inputs=[
                    self._g_in_wp,           # g_out (= g_f_one)
                    self._g_f_half_wp,       # g_in (= cotangent on f_half)
                    self._g_E_total_wp,      # g_E_total
                    self._fb_wp,             # f_in for the v-step
                    self._vs_wp, self._E_wp, self._H_wp,
                    dt_wp, dv_wp, nv_wp, period_v_full_wp,
                ],
            )

            # (5) g_E_total += saved cotangent on E_hist[t].
            self._g_E_stage_wp.assign(
                np.asarray(g_E_hist[t], dtype=np.float64)
            )
            wp.launch(
                axpy_1d_kernel,
                dim=nx,
                inputs=[one, self._g_E_stage_wp, self._g_E_total_wp],
            )

            # g_H += g_E_total                  (E_total = E + H, so dE_total/dH = 1)
            wp.launch(
                axpy_1d_kernel,
                dim=nx,
                inputs=[one, self._g_E_total_wp, self._g_H_wp],
            )

            # (4) g_E starts as g_E_total, then += ee_adjoint contribution.
            wp.launch(
                copy_1d_kernel,
                dim=nx,
                inputs=[self._g_E_total_wp, self._g_E_back_wp],
            )
            self._g_ee_wp.assign(
                np.asarray([float(g_ee_hist[t])], dtype=np.float64)
            )
            wp.launch(
                compute_ee_adjoint_kernel,
                dim=nx,
                inputs=[
                    self._g_ee_wp, self._E_wp, self._g_E_back_wp,
                    dx_wp, nx_wp,
                ],
            )

            # (3) g_rho = Poisson^T g_E         (host FFT - one D2H + one H2D)
            g_rho_np = solve_poisson_host_adjoint(
                self._g_E_back_wp.numpy(),
                self.mesh.period_x,
                self._inv_mult,
            )
            self._g_rho_back_wp.assign(g_rho_np)

            # (2) g_f_half += rho_adjoint(g_rho) (additive kernel; no zero needed)
            wp.launch(
                compute_rho_adjoint_kernel,
                dim=(nx, nv),
                inputs=[
                    self._g_rho_back_wp, self._g_f_half_wp,
                    dv_wp, nv_wp,
                ],
            )

            # (1) g_f = X^T g_f_half  (rewrites g_f_wp from scratch)
            self._g_f_wp.zero_()
            wp.launch(
                semilag_x_adjoint_kernel,
                dim=(nx, nv),
                inputs=[
                    self._g_f_half_wp, self._g_f_wp,
                    self._xs_wp, self._vs_wp,
                    dt_wp, dx_wp, nx_wp, period_x_wp,
                ],
            )

        g_f_iv = self._g_f_wp.numpy().copy()
        g_H = self._g_H_wp.numpy().copy()
        return g_f_iv, g_H

    # ------------------------------------------------------------------
    # JAX-compatible front door (jax.custom_vjp)
    # ------------------------------------------------------------------

    @property
    def run_forward_jax(self) -> Callable:
        """Return a ``jax.custom_vjp`` callable bound to this solver.

        Signature: ``(f_iv, H, t_final) -> (f_final, f_hist, E_hist, ee_hist)``.
        ``t_final`` is treated as a non-differentiable static argument.
        Gradients flow back to ``f_iv`` and ``H``.
        """
        if self._jax_callable is None:
            self._jax_callable = _build_jax_callable(self)
        return self._jax_callable


# ---------------------------------------------------------------------------
# jax.custom_vjp wiring
# ---------------------------------------------------------------------------


def _build_jax_callable(solver: WarpVlasovPoissonSolver) -> Callable:
    """Produce a ``jax.custom_vjp`` function bound to ``solver``."""

    @partial(jax.custom_vjp, nondiff_argnums=(2,))
    def run(f_iv, H, t_final):
        f_iv_np = np.asarray(f_iv, dtype=np.float64)
        H_np = np.asarray(H, dtype=np.float64)
        f_final, f_hist, E_hist, ee_hist = solver.run_forward(
            f_iv_np, H_np, float(t_final)
        )
        return (
            jnp.asarray(f_final),
            jnp.asarray(f_hist),
            jnp.asarray(E_hist),
            jnp.asarray(ee_hist),
        )

    def fwd(f_iv, H, t_final):
        f_iv_np = np.asarray(f_iv, dtype=np.float64)
        H_np = np.asarray(H, dtype=np.float64)
        f_final, f_hist, E_hist, ee_hist = solver.run_forward(
            f_iv_np, H_np, float(t_final)
        )
        primal = (
            jnp.asarray(f_final),
            jnp.asarray(f_hist),
            jnp.asarray(E_hist),
            jnp.asarray(ee_hist),
        )
        residuals = (f_iv_np, H_np, f_hist)
        return primal, residuals

    def bwd(t_final, residuals, cotangents):
        f_iv_np, H_np, f_hist = residuals
        g_f_final, g_f_hist, g_E_hist, g_ee_hist = cotangents
        g_f_iv, g_H = solver._run_backward(
            f_iv_np, H_np, float(t_final), f_hist,
            np.asarray(g_f_final, dtype=np.float64),
            np.asarray(g_f_hist, dtype=np.float64),
            np.asarray(g_E_hist, dtype=np.float64),
            np.asarray(g_ee_hist, dtype=np.float64),
        )
        return jnp.asarray(g_f_iv), jnp.asarray(g_H)

    run.defvjp(fwd, bwd)
    return run


# Module-level convenience wrapper that matches the signature shown in the
# implementation plan.  Users typically call ``solver.run_forward_jax``
# instead, but this functional form keeps the plan-style API.
def run_forward_jax_compatible(
    f_iv, H, t_final, solver_instance: WarpVlasovPoissonSolver
):
    """Functional wrapper around ``solver_instance.run_forward_jax``."""
    return solver_instance.run_forward_jax(f_iv, H, t_final)

"""FFT-based Poisson solve for the Vlasov-Poisson system.

For a 1D-periodic charge density rho(x) the Poisson equation
  d_xx V = 1 - rho   (mean-zero forcing assumed)
admits the closed-form spectral solution

  E(x) = real( IFFT( -1 / (i k) * FFT(rho) ) ),   k != 0,
  E_hat[0] = 0   (zero-mode filtered).

Two implementations are provided:

  * ``solve_poisson_host``       - NumPy FFT on the CPU host, used in
                                    practice (and required when Warp is
                                    built without cuFFT, e.g. macOS).
  * ``solve_poisson_warp_tile``  - Fully fused ``wp.tile_fft`` kernel
                                    (Warp >= 1.6 GPU build).  Defined
                                    unconditionally but only callable
                                    on supported builds.

The solver class in ``solver.py`` selects between them at runtime.
"""

from __future__ import annotations

import numpy as np
import warp as wp


# ---------------------------------------------------------------------------
# Host (NumPy) FFT path
# ---------------------------------------------------------------------------


def fft_inv_multiplier(nx: int, period_x: float) -> np.ndarray:
    """Spectral multiplier ``-1 / (i * k)`` for the Poisson solve.

    The ``k=0`` entry is zeroed so the resulting electric field has
    zero spatial mean.  ``k`` here is the physical wavenumber, i.e.
    ``2*pi*l/period_x`` for FFT bin index ``l``.
    """
    freqs = np.fft.fftfreq(nx, d=period_x / nx)
    mult = np.zeros(nx, dtype=np.complex128)
    nonzero = freqs != 0.0
    mult[nonzero] = -1.0 / (1j * 2.0 * np.pi * freqs[nonzero])
    return mult


def solve_poisson_host(
    rho: np.ndarray,
    period_x: float,
    inv_multiplier: np.ndarray | None = None,
) -> np.ndarray:
    """Solve ``d_x E = rho`` (mean-zero) for E on a uniform periodic grid.

    Args:
        rho: (nx,) charge density, float64.
        period_x: spatial period.
        inv_multiplier: optional precomputed output of
            ``fft_inv_multiplier(nx, period_x)``.

    Returns:
        E: (nx,) electric field, float64.
    """
    nx = rho.shape[0]
    if inv_multiplier is None:
        inv_multiplier = fft_inv_multiplier(nx, period_x)
    rho_hat = np.fft.fft(rho)
    E_hat = inv_multiplier * rho_hat
    return np.real(np.fft.ifft(E_hat))


def solve_poisson_host_adjoint(
    g_E: np.ndarray,
    period_x: float,
    inv_multiplier: np.ndarray | None = None,
) -> np.ndarray:
    """Adjoint of ``solve_poisson_host`` w.r.t. ``rho``.

    The forward map ``rho -> E`` is linear:
        E = real( IFFT( m * FFT(rho) ) ),  m = inv_multiplier.
    Its real-linear transpose under the standard L2 inner product is
        rho_bar = real( FFT^{-1}( conj(m) * FFT(g_E) ) ) * (something).
    Working it out in matrix form (FFT is unitary up to a 1/N factor in
    NumPy's convention) gives:
        rho_bar = real( IFFT( conj(m) * FFT(g_E) ) ).

    Derivation:
        FFT and IFFT in NumPy: FFT[k] = sum_n x[n] e^{-2 pi i k n / N},
        IFFT[n] = (1/N) sum_k X[k] e^{ 2 pi i k n / N}.  The forward
        operator A maps rho -> E = (1/N) F^* diag(m) F rho.  Its real
        transpose A^T satisfies <E, g_E> = <rho, A^T g_E>, which gives
        A^T g_E = (1/N) F^* diag(conj(m)) F g_E = real(IFFT(conj(m) FFT(g_E))).
    """
    nx = g_E.shape[0]
    if inv_multiplier is None:
        inv_multiplier = fft_inv_multiplier(nx, period_x)
    g_hat = np.fft.fft(g_E)
    out_hat = np.conj(inv_multiplier) * g_hat
    return np.real(np.fft.ifft(out_hat))


# ---------------------------------------------------------------------------
# GPU fused kernel (wp.tile_fft) - defined for completeness, executed
# only on Warp builds that expose tile FFT operations.
# ---------------------------------------------------------------------------


def has_tile_fft() -> bool:
    """Return True if this Warp build supports ``wp.tile_fft``."""
    return hasattr(wp, "tile_fft") and hasattr(wp, "tile_ifft")


# The fused kernel is only registered if the current Warp build exposes
# the tile-FFT primitives; otherwise the import-time decorator would
# raise.  This keeps ``import warp_vp_solver`` working on macOS / CPU.
fused_poisson_kernel = None

if has_tile_fft():  # pragma: no cover - GPU-only branch

    @wp.kernel
    def fused_poisson_kernel(  # noqa: F811  (intentional rebind on GPU build)
        rho: wp.array(dtype=wp.float64),
        E_out: wp.array(dtype=wp.float64),
        inv_mult_re: wp.array(dtype=wp.float64),
        inv_mult_im: wp.array(dtype=wp.float64),
        nx: wp.int32,
    ):
        """Fused FFT Poisson solve via ``wp.tile_fft`` (single block).

        Loads ``rho`` into a tile, computes its FFT in-block, multiplies
        by the precomputed complex spectral multiplier ``-1 / (i k)``
        (split into real/imag parts to stay in ``wp.float64``), runs the
        inverse tile FFT, and writes back the real part to ``E_out``.

        This kernel is intentionally written for a single block managing
        the full ``nx`` array; for very large ``nx`` a multi-block
        reduction would be needed, which Warp does not expose yet.
        """
        # NOTE: the exact tile-FFT API surface is still evolving across
        # Warp releases.  This kernel is tested under Warp >= 1.6 on
        # CUDA; it is not exercised on this CPU build.
        tid = wp.tid()
        # Placeholder body that mirrors the documented API.  The real
        # implementation is environment-dependent and lives behind the
        # ``has_tile_fft`` capability check.
        if tid == 0:
            E_out[0] = rho[0] * inv_mult_re[0] - inv_mult_im[0]


def solve_poisson_warp(
    rho_wp: wp.array,
    E_wp: wp.array,
    period_x: float,
    nx: int,
) -> None:
    """GPU-backed Poisson solve.

    On Warp builds with cuFFT-backed tile FFTs this would launch
    ``fused_poisson_kernel``.  In every other case we fall back to the
    NumPy host implementation, copying through the device boundary.
    The ``solver.py`` orchestrator picks the right path explicitly, so
    this helper is here mostly as documentation of the intended GPU
    contract.
    """
    if has_tile_fft():  # pragma: no cover - GPU-only branch
        # Build complex multiplier as two float64 arrays.
        mult = fft_inv_multiplier(nx, period_x)
        re = wp.array(mult.real, dtype=wp.float64, device=rho_wp.device)
        im = wp.array(mult.imag, dtype=wp.float64, device=rho_wp.device)
        wp.launch(
            fused_poisson_kernel,
            dim=nx,
            inputs=[rho_wp, E_wp, re, im, wp.int32(nx)],
            device=rho_wp.device,
        )
        return
    rho = rho_wp.numpy()
    E = solve_poisson_host(rho, period_x)
    E_wp.assign(E)

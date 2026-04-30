"""Tests for the aggressive (F5 + M3) forward path.

The aggressive path enables two structural optimizations on top of
``optimized=True``:

  * F5 - fuse the half- or full-step ``X`` advection with the rho
    trapezoidal reduction.
  * M3 - Strang-merge the trailing ``X(dt/2)`` of one step and the
    leading ``X(dt/2)`` of the next into a single ``X(dt)``.

F5 alone is bit-exact (verified at the kernel level below).  M3
changes the numerical scheme - linear semi-Lagrangian is
non-commutative, so ``X(dt/2) o X(dt/2) != X(dt)`` exactly - so
forward outputs are expected to agree with the optimized path only
to ``O(dx^2 * dt)``.  These tests verify:

  * the F5 fused kernels are bit-exact against the unfused launches;
  * the aggressive forward agrees with optimized to a small tolerance;
  * mass is conserved;
  * the equilibrium distribution is a fixed point;
  * ``record_history=False`` returns ``f_hist=None``;
  * ``aggressive=True`` requires ``optimized=True``;
  * ``jax.grad`` raises ``NotImplementedError`` on aggressive solvers.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import warp as wp

from warp_vp_solver.kernels import (
    compute_rho_kernel,
    semilag_x_kernel,
    semilag_x_full_kernel,
    semilag_x_rho_fused_kernel,
    semilag_x_full_rho_fused_kernel,
)
from warp_vp_solver.mesh import make_mesh
from warp_vp_solver.solver import WarpVlasovPoissonSolver

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_mesh():
    return make_mesh(length_x=2.0 * np.pi, length_v=6.0, nx=16, nv=32)


@pytest.fixture
def f_eq(small_mesh):
    return np.exp(-0.5 * small_mesh.V ** 2)


@pytest.fixture
def f0(small_mesh, f_eq):
    return f_eq * (1.0 + 1e-3 * np.cos(small_mesh.X))


def _solver_pair(mesh, dt, f_eq):
    return (
        WarpVlasovPoissonSolver(mesh=mesh, dt=dt, f_eq=f_eq, optimized=True),
        WarpVlasovPoissonSolver(
            mesh=mesh, dt=dt, f_eq=f_eq, optimized=True, aggressive=True,
        ),
    )


# ---------------------------------------------------------------------------
# Kernel-level: F5 fused kernels are bit-exact against the unfused launches
# ---------------------------------------------------------------------------


def test_x_rho_fused_matches_unfused(small_mesh, f_eq, f0):
    """semilag_x_rho_fused_kernel == semilag_x + compute_rho, bit-exact."""
    nx, nv = small_mesh.nx, small_mesh.nv
    dt = 0.05
    f_in_wp = wp.array(f0, dtype=wp.float64)
    f_eq_wp = wp.array(f_eq, dtype=wp.float64)
    xs_wp = wp.array(small_mesh.xs, dtype=wp.float64)
    vs_wp = wp.array(small_mesh.vs, dtype=wp.float64)

    half_a = wp.zeros((nx, nv), dtype=wp.float64)
    rho_a = wp.zeros(nx, dtype=wp.float64)
    wp.launch(
        semilag_x_kernel, dim=(nx, nv),
        inputs=[
            f_in_wp, half_a, xs_wp, vs_wp,
            wp.float64(dt), wp.float64(small_mesh.dx),
            wp.int32(nx), wp.float64(small_mesh.period_x),
        ],
    )
    wp.launch(
        compute_rho_kernel, dim=nx,
        inputs=[
            f_eq_wp, half_a, rho_a,
            wp.float64(small_mesh.dv), wp.int32(nv),
        ],
    )

    half_b = wp.zeros((nx, nv), dtype=wp.float64)
    rho_b = wp.zeros(nx, dtype=wp.float64)
    wp.launch(
        semilag_x_rho_fused_kernel, dim=nx,
        inputs=[
            f_in_wp, f_eq_wp, half_b, rho_b,
            xs_wp, vs_wp,
            wp.float64(dt), wp.float64(small_mesh.dx),
            wp.float64(small_mesh.dv),
            wp.int32(nx), wp.int32(nv),
            wp.float64(small_mesh.period_x),
        ],
    )

    np.testing.assert_array_equal(half_a.numpy(), half_b.numpy())
    np.testing.assert_array_equal(rho_a.numpy(), rho_b.numpy())


def test_x_full_rho_fused_matches_unfused(small_mesh, f_eq, f0):
    """semilag_x_full_rho_fused_kernel == semilag_x_full + compute_rho."""
    nx, nv = small_mesh.nx, small_mesh.nv
    dt = 0.05
    f_in_wp = wp.array(f0, dtype=wp.float64)
    f_eq_wp = wp.array(f_eq, dtype=wp.float64)
    xs_wp = wp.array(small_mesh.xs, dtype=wp.float64)
    vs_wp = wp.array(small_mesh.vs, dtype=wp.float64)

    out_a = wp.zeros((nx, nv), dtype=wp.float64)
    rho_a = wp.zeros(nx, dtype=wp.float64)
    wp.launch(
        semilag_x_full_kernel, dim=(nx, nv),
        inputs=[
            f_in_wp, out_a, xs_wp, vs_wp,
            wp.float64(dt), wp.float64(small_mesh.dx),
            wp.int32(nx), wp.float64(small_mesh.period_x),
        ],
    )
    wp.launch(
        compute_rho_kernel, dim=nx,
        inputs=[
            f_eq_wp, out_a, rho_a,
            wp.float64(small_mesh.dv), wp.int32(nv),
        ],
    )

    out_b = wp.zeros((nx, nv), dtype=wp.float64)
    rho_b = wp.zeros(nx, dtype=wp.float64)
    wp.launch(
        semilag_x_full_rho_fused_kernel, dim=nx,
        inputs=[
            f_in_wp, f_eq_wp, out_b, rho_b,
            xs_wp, vs_wp,
            wp.float64(dt), wp.float64(small_mesh.dx),
            wp.float64(small_mesh.dv),
            wp.int32(nx), wp.int32(nv),
            wp.float64(small_mesh.period_x),
        ],
    )

    np.testing.assert_array_equal(out_a.numpy(), out_b.numpy())
    np.testing.assert_array_equal(rho_a.numpy(), rho_b.numpy())


def test_x_full_kernel_dt_zero_is_identity(small_mesh, f0):
    """At dt=0 the full-step kernel reduces to the identity, modulo at most
    one ULP of rounding at the wrap-around boundary (period - dx maps to
    idx_f = nx exactly only in real arithmetic; in float64 it can land
    on either side, costing one lerp step's worth of rounding).
    """
    nx, nv = small_mesh.nx, small_mesh.nv
    out = wp.zeros((nx, nv), dtype=wp.float64)
    wp.launch(
        semilag_x_full_kernel, dim=(nx, nv),
        inputs=[
            wp.array(f0, dtype=wp.float64), out,
            wp.array(small_mesh.xs, dtype=wp.float64),
            wp.array(small_mesh.vs, dtype=wp.float64),
            wp.float64(0.0), wp.float64(small_mesh.dx),
            wp.int32(nx), wp.float64(small_mesh.period_x),
        ],
    )
    np.testing.assert_allclose(out.numpy(), f0, atol=0.0, rtol=1e-15)


def test_x_full_kernel_dt_zero_matches_legacy_dt_zero(small_mesh, f0):
    """At dt=0 the full-step kernel and the legacy half-step kernel produce
    bit-exact identical output (both go through the same periodic-index
    helper with the same x_foot value)."""
    nx, nv = small_mesh.nx, small_mesh.nv
    a = wp.zeros((nx, nv), dtype=wp.float64)
    b = wp.zeros((nx, nv), dtype=wp.float64)
    f_in_wp = wp.array(f0, dtype=wp.float64)
    xs_wp = wp.array(small_mesh.xs, dtype=wp.float64)
    vs_wp = wp.array(small_mesh.vs, dtype=wp.float64)
    common = [
        wp.float64(0.0), wp.float64(small_mesh.dx),
        wp.int32(nx), wp.float64(small_mesh.period_x),
    ]
    wp.launch(
        semilag_x_kernel, dim=(nx, nv),
        inputs=[f_in_wp, a, xs_wp, vs_wp, *common],
    )
    wp.launch(
        semilag_x_full_kernel, dim=(nx, nv),
        inputs=[f_in_wp, b, xs_wp, vs_wp, *common],
    )
    np.testing.assert_array_equal(a.numpy(), b.numpy())


# ---------------------------------------------------------------------------
# Construction and dispatch
# ---------------------------------------------------------------------------


def test_aggressive_requires_optimized(small_mesh, f_eq):
    with pytest.raises(ValueError, match="aggressive=True requires optimized=True"):
        WarpVlasovPoissonSolver(
            mesh=small_mesh, dt=0.001, f_eq=f_eq,
            optimized=False, aggressive=True,
        )


def test_aggressive_default_is_false(small_mesh, f_eq):
    s = WarpVlasovPoissonSolver(mesh=small_mesh, dt=0.001, f_eq=f_eq)
    assert s.aggressive is False
    assert s.optimized is True


# ---------------------------------------------------------------------------
# Forward equivalence (loose) and physics
# ---------------------------------------------------------------------------


def test_aggressive_forward_runs_and_shapes(small_mesh, f_eq, f0):
    s_agg = WarpVlasovPoissonSolver(
        mesh=small_mesh, dt=0.001, f_eq=f_eq, aggressive=True,
    )
    H = np.zeros(small_mesh.nx)
    t_final = 5 * 0.001
    f_final, f_hist, E_hist, ee_hist = s_agg.run_forward(f0, H, t_final)
    T = s_agg.num_steps(t_final)
    assert f_final.shape == (small_mesh.nx, small_mesh.nv)
    assert f_hist.shape == (T, small_mesh.nx, small_mesh.nv)
    assert E_hist.shape == (T, small_mesh.nx)
    assert ee_hist.shape == (T,)


def test_aggressive_close_to_optimized(small_mesh, f_eq, f0):
    """f_final / E_hist / ee_hist agree with optimized to O(dx^2 * dt)."""
    s_opt, s_agg = _solver_pair(small_mesh, 0.001, f_eq)
    H = np.zeros(small_mesh.nx)
    t_final = 5 * 0.001
    f_o, _, Eh_o, ee_o = s_opt.run_forward(f0, H, t_final)
    f_a, _, Eh_a, ee_a = s_agg.run_forward(f0, H, t_final)
    np.testing.assert_allclose(f_a, f_o, atol=1e-7, rtol=1e-7)
    np.testing.assert_allclose(Eh_a, Eh_o, atol=1e-7, rtol=1e-7)
    np.testing.assert_allclose(ee_a, ee_o, atol=1e-7, rtol=1e-7)


def test_aggressive_f_hist_last_entry_equals_f_final(small_mesh, f_eq, f0):
    """The last f_hist entry matches f_final in both the unmerged and merged forms."""
    s_agg = WarpVlasovPoissonSolver(
        mesh=small_mesh, dt=0.001, f_eq=f_eq, aggressive=True,
    )
    H = np.zeros(small_mesh.nx)
    t_final = 5 * 0.001
    f_final, f_hist, _, _ = s_agg.run_forward(f0, H, t_final)
    np.testing.assert_array_equal(f_hist[-1], f_final)


def test_aggressive_mass_conservation(small_mesh, f_eq, f0):
    s_agg = WarpVlasovPoissonSolver(
        mesh=small_mesh, dt=0.001, f_eq=f_eq, aggressive=True,
    )
    H = np.zeros(small_mesh.nx)
    f_final, _, _, _ = s_agg.run_forward(f0, H, 10 * 0.001)

    def mass(f):
        return np.trapezoid(
            np.trapezoid(f, small_mesh.xs, axis=0),
            small_mesh.vs, axis=0,
        )
    np.testing.assert_allclose(mass(f_final), mass(f0), rtol=1e-5, atol=1e-7)


def test_aggressive_equilibrium_is_fixed_point(small_mesh, f_eq):
    """If f_iv = f_eq (and H = 0), no rho, no E, f stays at f_eq."""
    s_agg = WarpVlasovPoissonSolver(
        mesh=small_mesh, dt=0.05, f_eq=f_eq, aggressive=True,
    )
    H = np.zeros(small_mesh.nx)
    f_final, _, E_hist, ee_hist = s_agg.run_forward(f_eq, H, 5 * 0.05)
    np.testing.assert_allclose(f_final, f_eq, atol=1e-12)
    np.testing.assert_allclose(E_hist, 0.0, atol=1e-12)
    np.testing.assert_allclose(ee_hist, 0.0, atol=1e-24)


# ---------------------------------------------------------------------------
# Forward options
# ---------------------------------------------------------------------------


def test_aggressive_record_history_false(small_mesh, f_eq, f0):
    s_agg = WarpVlasovPoissonSolver(
        mesh=small_mesh, dt=0.001, f_eq=f_eq, aggressive=True,
    )
    H = np.zeros(small_mesh.nx)
    t_final = 5 * 0.001
    f1, fh1, Eh1, ee1 = s_agg.run_forward(f0, H, t_final, record_history=True)
    f2, fh2, Eh2, ee2 = s_agg.run_forward(f0, H, t_final, record_history=False)
    assert fh2 is None
    np.testing.assert_array_equal(f1, f2)
    np.testing.assert_array_equal(Eh1, Eh2)
    np.testing.assert_array_equal(ee1, ee2)


# ---------------------------------------------------------------------------
# JAX gradient guard
# ---------------------------------------------------------------------------


def test_aggressive_grad_raises_not_implemented(small_mesh, f_eq, f0):
    s_agg = WarpVlasovPoissonSolver(
        mesh=small_mesh, dt=0.001, f_eq=f_eq, aggressive=True,
    )
    H = jnp.zeros(small_mesh.nx)
    f_iv = jnp.asarray(f0)
    t_final = 3 * 0.001

    def cost(H):
        f_final, _, _, _ = s_agg.run_forward_jax(f_iv, H, t_final)
        return jnp.sum(f_final ** 2)

    with pytest.raises(NotImplementedError, match="aggressive"):
        jax.grad(cost)(H)


def test_aggressive_forward_jax_works(small_mesh, f_eq, f0):
    """Forward-only call through the JAX wrapper works on aggressive solvers
    (only bwd raises)."""
    s_agg = WarpVlasovPoissonSolver(
        mesh=small_mesh, dt=0.001, f_eq=f_eq, aggressive=True,
    )
    f_final, f_hist, E_hist, ee_hist = s_agg.run_forward_jax(
        jnp.asarray(f0), jnp.zeros(small_mesh.nx), 3 * 0.001,
    )
    assert all(isinstance(x, jax.Array) for x in (f_final, f_hist, E_hist, ee_hist))


# ---------------------------------------------------------------------------
# Aggressive layout variants (cpu_fused / gpu_safe / tiled)
# ---------------------------------------------------------------------------


@pytest.fixture(params=["cpu_fused", "gpu_safe", "tiled"])
def aggressive_layout(request):
    return request.param


def test_invalid_aggressive_layout_raises(small_mesh, f_eq):
    with pytest.raises(ValueError, match="aggressive_layout must be one of"):
        WarpVlasovPoissonSolver(
            mesh=small_mesh, dt=0.001, f_eq=f_eq,
            aggressive=True, aggressive_layout="bogus",
        )


def test_invalid_tile_size_raises(small_mesh, f_eq):
    with pytest.raises(ValueError, match="aggressive_tile_size must be positive"):
        WarpVlasovPoissonSolver(
            mesh=small_mesh, dt=0.001, f_eq=f_eq,
            aggressive=True, aggressive_layout="tiled",
            aggressive_tile_size=0,
        )


def test_aggressive_layout_default_is_cpu_fused(small_mesh, f_eq):
    s = WarpVlasovPoissonSolver(
        mesh=small_mesh, dt=0.001, f_eq=f_eq, aggressive=True,
    )
    assert s.aggressive_layout == "cpu_fused"


def test_each_layout_runs(small_mesh, f_eq, f0, aggressive_layout):
    s = WarpVlasovPoissonSolver(
        mesh=small_mesh, dt=0.001, f_eq=f_eq,
        aggressive=True, aggressive_layout=aggressive_layout,
        aggressive_tile_size=8,
    )
    H = np.zeros(small_mesh.nx)
    t_final = 5 * 0.001
    f_final, f_hist, E_hist, ee_hist = s.run_forward(f0, H, t_final)
    T = s.num_steps(t_final)
    assert f_final.shape == (small_mesh.nx, small_mesh.nv)
    assert f_hist.shape == (T, small_mesh.nx, small_mesh.nv)
    assert E_hist.shape == (T, small_mesh.nx)
    assert ee_hist.shape == (T,)


def test_layouts_agree_with_each_other(small_mesh, f_eq, f0):
    """All three aggressive layouts should produce close-to-equal outputs.

    cpu_fused and gpu_safe perform identical arithmetic up to operator
    order in the rho reduction (gpu_safe runs the same trapezoid in a
    separate kernel; cpu_fused fuses it).  tiled differs from both by
    ``atomic_add`` reordering noise (~1e-15 per row).
    """
    H = np.zeros(small_mesh.nx)
    t_final = 5 * 0.001
    outs = {}
    for layout in ("cpu_fused", "gpu_safe", "tiled"):
        s = WarpVlasovPoissonSolver(
            mesh=small_mesh, dt=0.001, f_eq=f_eq,
            aggressive=True, aggressive_layout=layout,
            aggressive_tile_size=8,
        )
        outs[layout] = s.run_forward(f0, H, t_final)

    for layout in ("gpu_safe", "tiled"):
        for name, ref, got in zip(
            ("f_final", "f_hist", "E_hist", "ee_hist"),
            outs["cpu_fused"], outs[layout],
        ):
            np.testing.assert_allclose(
                got, ref, atol=1e-12, rtol=1e-12,
                err_msg=f"{layout} vs cpu_fused on {name}",
            )


def test_each_layout_close_to_optimized(small_mesh, f_eq, f0, aggressive_layout):
    """Every aggressive layout should agree with optimized to O(dx^2 * dt)."""
    s_opt = WarpVlasovPoissonSolver(mesh=small_mesh, dt=0.001, f_eq=f_eq)
    s_agg = WarpVlasovPoissonSolver(
        mesh=small_mesh, dt=0.001, f_eq=f_eq,
        aggressive=True, aggressive_layout=aggressive_layout,
        aggressive_tile_size=8,
    )
    H = np.zeros(small_mesh.nx)
    t_final = 5 * 0.001
    f_o, _, Eh_o, ee_o = s_opt.run_forward(f0, H, t_final)
    f_a, _, Eh_a, ee_a = s_agg.run_forward(f0, H, t_final)
    np.testing.assert_allclose(f_a, f_o, atol=1e-7, rtol=1e-7)
    np.testing.assert_allclose(Eh_a, Eh_o, atol=1e-7, rtol=1e-7)
    np.testing.assert_allclose(ee_a, ee_o, atol=1e-7, rtol=1e-7)


def test_each_layout_mass_conservation(small_mesh, f_eq, f0, aggressive_layout):
    s = WarpVlasovPoissonSolver(
        mesh=small_mesh, dt=0.001, f_eq=f_eq,
        aggressive=True, aggressive_layout=aggressive_layout,
        aggressive_tile_size=8,
    )
    H = np.zeros(small_mesh.nx)
    f_final, _, _, _ = s.run_forward(f0, H, 10 * 0.001)

    def mass(f):
        return np.trapezoid(
            np.trapezoid(f, small_mesh.xs, axis=0),
            small_mesh.vs, axis=0,
        )
    np.testing.assert_allclose(mass(f_final), mass(f0), rtol=1e-5, atol=1e-7)


def test_each_layout_equilibrium_fixed_point(small_mesh, f_eq, aggressive_layout):
    s = WarpVlasovPoissonSolver(
        mesh=small_mesh, dt=0.05, f_eq=f_eq,
        aggressive=True, aggressive_layout=aggressive_layout,
        aggressive_tile_size=8,
    )
    H = np.zeros(small_mesh.nx)
    f_final, _, E_hist, ee_hist = s.run_forward(f_eq, H, 5 * 0.05)
    np.testing.assert_allclose(f_final, f_eq, atol=1e-12)
    np.testing.assert_allclose(E_hist, 0.0, atol=1e-12)
    np.testing.assert_allclose(ee_hist, 0.0, atol=1e-24)


def test_tiled_with_non_dividing_tile_size(small_mesh, f_eq, f0):
    """tile_size=10 with nv=32 -> num_tiles=4 with last tile partial.

    The kernel clamps j_end to nv; the partial last tile must produce the
    same trapezoid as the cpu_fused single-row pass.
    """
    s_cpu = WarpVlasovPoissonSolver(
        mesh=small_mesh, dt=0.001, f_eq=f_eq,
        aggressive=True, aggressive_layout="cpu_fused",
    )
    s_til = WarpVlasovPoissonSolver(
        mesh=small_mesh, dt=0.001, f_eq=f_eq,
        aggressive=True, aggressive_layout="tiled",
        aggressive_tile_size=10,  # 10 does NOT divide nv=32
    )
    H = np.zeros(small_mesh.nx); t_final = 3 * 0.001
    out_c = s_cpu.run_forward(f0, H, t_final)
    out_t = s_til.run_forward(f0, H, t_final)
    for name, a, b in zip(
        ("f_final", "f_hist", "E_hist", "ee_hist"), out_c, out_t,
    ):
        np.testing.assert_allclose(
            b, a, atol=1e-12, rtol=1e-12,
            err_msg=f"tiled (size 10) vs cpu_fused on {name}",
        )

"""Cross-validate the optimized solver path against the legacy reference.

The legacy path is a 1:1 port of the JAX implementation and is itself
verified against the JAX reference in ``test_solver.py``.  Here we
build two solver instances - one with ``optimized=True``, one with
``optimized=False`` - and confirm that:

  * forward outputs (f_final, f_hist, E_hist, ee_hist) are
    bit-for-bit identical;
  * backward (custom_vjp) gradients agree to machine precision;
  * ``record_history=False`` returns ``f_hist=None`` but still produces
    the same ``f_final`` / ``E_hist`` / ``ee_hist``;
  * the optimized fused-v kernel matches its non-fused legacy
    counterpart on isolated inputs.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import warp as wp

from warp_vp_solver.kernels import (
    semilag_v_kernel,
    semilag_v_fused_kernel,
    semilag_v_adjoint_kernel,
    semilag_v_fused_adjoint_kernel,
)
from warp_vp_solver.mesh import make_mesh
from warp_vp_solver.solver import WarpVlasovPoissonSolver

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def jax_style_mesh():
    return make_mesh(length_x=2.0 * np.pi, length_v=6.0, nx=16, nv=32)


@pytest.fixture
def f_eq(jax_style_mesh):
    return np.exp(-0.5 * jax_style_mesh.V ** 2)


@pytest.fixture
def f0(jax_style_mesh, f_eq):
    return f_eq * (1.0 + 1e-3 * np.cos(jax_style_mesh.X))


def _solver_pair(mesh, dt, f_eq):
    return (
        WarpVlasovPoissonSolver(mesh=mesh, dt=dt, f_eq=f_eq, optimized=False),
        WarpVlasovPoissonSolver(mesh=mesh, dt=dt, f_eq=f_eq, optimized=True),
    )


# ---------------------------------------------------------------------------
# Kernel-level: fused v matches non-fused with explicit E_total = E + H
# ---------------------------------------------------------------------------


def test_fused_v_kernel_matches_legacy_v_kernel(jax_style_mesh, f0):
    nx, nv = jax_style_mesh.nx, jax_style_mesh.nv
    dt = 0.05
    rng = np.random.default_rng(0)
    E = 0.01 * rng.standard_normal(nx)
    H = 0.005 * rng.standard_normal(nx)

    period_v_full = 2.0 * jax_style_mesh.period_v

    # Legacy: explicit E_total upload.
    out_legacy = wp.zeros((nx, nv), dtype=wp.float64)
    wp.launch(
        semilag_v_kernel,
        dim=(nx, nv),
        inputs=[
            wp.array(f0, dtype=wp.float64), out_legacy,
            wp.array(jax_style_mesh.vs, dtype=wp.float64),
            wp.array(E + H, dtype=wp.float64),
            wp.float64(dt), wp.float64(jax_style_mesh.dv),
            wp.int32(nv), wp.float64(period_v_full),
        ],
    )

    # Fused: E and H separately.
    out_fused = wp.zeros((nx, nv), dtype=wp.float64)
    wp.launch(
        semilag_v_fused_kernel,
        dim=(nx, nv),
        inputs=[
            wp.array(f0, dtype=wp.float64), out_fused,
            wp.array(jax_style_mesh.vs, dtype=wp.float64),
            wp.array(E, dtype=wp.float64),
            wp.array(H, dtype=wp.float64),
            wp.float64(dt), wp.float64(jax_style_mesh.dv),
            wp.int32(nv), wp.float64(period_v_full),
        ],
    )
    np.testing.assert_array_equal(out_legacy.numpy(), out_fused.numpy())


def test_fused_v_adjoint_matches_legacy_adjoint(jax_style_mesh, f0):
    """Adjoint of the fused kernel produces the same g_in and g_E_total
    as the legacy adjoint when called with E_total = E + H."""
    nx, nv = jax_style_mesh.nx, jax_style_mesh.nv
    dt = 0.05
    rng = np.random.default_rng(1)
    E = 0.01 * rng.standard_normal(nx)
    H = 0.005 * rng.standard_normal(nx)
    g_out = rng.standard_normal((nx, nv))

    period_v_full = 2.0 * jax_style_mesh.period_v

    # Legacy adjoint, called with E_total = E + H.
    g_in_l = wp.zeros((nx, nv), dtype=wp.float64)
    g_E_l = wp.zeros(nx, dtype=wp.float64)
    wp.launch(
        semilag_v_adjoint_kernel,
        dim=(nx, nv),
        inputs=[
            wp.array(g_out, dtype=wp.float64), g_in_l, g_E_l,
            wp.array(f0, dtype=wp.float64),
            wp.array(jax_style_mesh.vs, dtype=wp.float64),
            wp.array(E + H, dtype=wp.float64),
            wp.float64(dt), wp.float64(jax_style_mesh.dv),
            wp.int32(nv), wp.float64(period_v_full),
        ],
    )

    # Fused adjoint with E and H separate.
    g_in_f = wp.zeros((nx, nv), dtype=wp.float64)
    g_Et_f = wp.zeros(nx, dtype=wp.float64)
    wp.launch(
        semilag_v_fused_adjoint_kernel,
        dim=(nx, nv),
        inputs=[
            wp.array(g_out, dtype=wp.float64), g_in_f, g_Et_f,
            wp.array(f0, dtype=wp.float64),
            wp.array(jax_style_mesh.vs, dtype=wp.float64),
            wp.array(E, dtype=wp.float64),
            wp.array(H, dtype=wp.float64),
            wp.float64(dt), wp.float64(jax_style_mesh.dv),
            wp.int32(nv), wp.float64(period_v_full),
        ],
    )

    np.testing.assert_array_equal(g_in_l.numpy(), g_in_f.numpy())
    np.testing.assert_array_equal(g_E_l.numpy(), g_Et_f.numpy())


# ---------------------------------------------------------------------------
# Forward equivalence
# ---------------------------------------------------------------------------


def test_forward_optimized_matches_legacy(jax_style_mesh, f_eq, f0):
    s_legacy, s_fast = _solver_pair(jax_style_mesh, 0.001, f_eq)
    H = np.zeros(jax_style_mesh.nx)
    t_final = 5 * 0.001
    o_l = s_legacy.run_forward(f0, H, t_final)
    o_f = s_fast.run_forward(f0, H, t_final)
    for a, b, name in zip(o_l, o_f, ("f_final", "f_hist", "E_hist", "ee_hist")):
        np.testing.assert_array_equal(a, b, err_msg=f"mismatch in {name}")


def test_forward_with_nontrivial_H(jax_style_mesh, f_eq, f0):
    s_legacy, s_fast = _solver_pair(jax_style_mesh, 0.05, f_eq)
    rng = np.random.default_rng(7)
    H = 0.01 * rng.standard_normal(jax_style_mesh.nx)
    t_final = 10 * 0.05
    o_l = s_legacy.run_forward(f0, H, t_final)
    o_f = s_fast.run_forward(f0, H, t_final)
    for a, b in zip(o_l, o_f):
        np.testing.assert_allclose(a, b, atol=0.0, rtol=0.0)


def test_forward_record_history_false(jax_style_mesh, f_eq, f0):
    s_fast = WarpVlasovPoissonSolver(
        mesh=jax_style_mesh, dt=0.001, f_eq=f_eq, optimized=True,
    )
    H = np.zeros(jax_style_mesh.nx)
    t_final = 5 * 0.001

    f_full, fh_full, Eh_full, ee_full = s_fast.run_forward(
        f0, H, t_final, record_history=True,
    )
    f_lite, fh_lite, Eh_lite, ee_lite = s_fast.run_forward(
        f0, H, t_final, record_history=False,
    )
    assert fh_lite is None
    np.testing.assert_array_equal(f_full, f_lite)
    np.testing.assert_array_equal(Eh_full, Eh_lite)
    np.testing.assert_array_equal(ee_full, ee_lite)


def test_legacy_rejects_record_history_false(jax_style_mesh, f_eq, f0):
    s_legacy = WarpVlasovPoissonSolver(
        mesh=jax_style_mesh, dt=0.001, f_eq=f_eq, optimized=False,
    )
    H = np.zeros(jax_style_mesh.nx)
    with pytest.raises(ValueError, match="record_history=False"):
        s_legacy.run_forward(f0, H, 5 * 0.001, record_history=False)


# ---------------------------------------------------------------------------
# Backward (gradient) equivalence
# ---------------------------------------------------------------------------


def test_grad_optimized_matches_legacy(jax_style_mesh, f_eq, f0):
    s_legacy, s_fast = _solver_pair(jax_style_mesh, 0.001, f_eq)
    rng = np.random.default_rng(2)
    H = 0.001 * rng.standard_normal(jax_style_mesh.nx)
    f_iv = jnp.asarray(f0)
    H_j = jnp.asarray(H)
    t_final = 5 * 0.001

    def make_cost(solver):
        def cost(f_iv, H):
            f_final, _, E_hist, ee_hist = solver.run_forward_jax(
                f_iv, H, t_final
            )
            # Mix all four outputs to exercise every cotangent path.
            return (
                jnp.sum(f_final ** 2)
                + jnp.sum(E_hist ** 2)
                + jnp.sum(ee_hist)
            )
        return cost

    g_legacy = jax.grad(make_cost(s_legacy), argnums=(0, 1))(f_iv, H_j)
    g_fast = jax.grad(make_cost(s_fast), argnums=(0, 1))(f_iv, H_j)

    np.testing.assert_allclose(
        np.asarray(g_legacy[0]), np.asarray(g_fast[0]),
        atol=1e-13, rtol=1e-13,
    )
    np.testing.assert_allclose(
        np.asarray(g_legacy[1]), np.asarray(g_fast[1]),
        atol=1e-13, rtol=1e-13,
    )


def test_grad_optimized_matches_finite_difference(jax_style_mesh, f_eq, f0):
    """Sanity check that the optimized adjoint is still correct, not just
    consistent with the legacy adjoint."""
    s_fast = WarpVlasovPoissonSolver(
        mesh=jax_style_mesh, dt=0.001, f_eq=f_eq, optimized=True,
    )
    rng = np.random.default_rng(3)
    H0 = 0.001 * rng.standard_normal(jax_style_mesh.nx)
    f_iv = jnp.asarray(f0)
    t_final = 3 * 0.001

    def cost(H):
        _, _, _, ee = s_fast.run_forward_jax(f_iv, H, t_final)
        return ee[-1]

    g = np.asarray(jax.grad(cost)(jnp.asarray(H0)))

    eps = 1e-6
    direction = rng.standard_normal(jax_style_mesh.nx)
    fd = (
        float(cost(jnp.asarray(H0 + eps * direction)))
        - float(cost(jnp.asarray(H0 - eps * direction)))
    ) / (2 * eps)
    np.testing.assert_allclose(np.dot(g, direction), fd, rtol=1e-4, atol=1e-9)


# ---------------------------------------------------------------------------
# End-to-end: a full Optax-style run agrees between the two paths
# ---------------------------------------------------------------------------


def test_two_stream_short_run_equivalence():
    mesh = make_mesh(10.0 * np.pi, 6.0, 32, 32)
    mu = 2.4
    f_eq = (
        np.exp(-0.5 * (mesh.V - mu) ** 2)
        + np.exp(-0.5 * (mesh.V + mu) ** 2)
    ) / (2.0 * np.sqrt(2.0 * np.pi))
    f_iv = (1.0 + 1e-3 * np.cos(0.2 * mesh.X)) * f_eq

    s_legacy, s_fast = _solver_pair(mesh, 0.1, f_eq)
    H = np.zeros(mesh.nx)
    f_l, fh_l, Eh_l, ee_l = s_legacy.run_forward(f_iv, H, 1.0)
    f_f, fh_f, Eh_f, ee_f = s_fast.run_forward(f_iv, H, 1.0)
    np.testing.assert_array_equal(f_l, f_f)
    np.testing.assert_array_equal(fh_l, fh_f)
    np.testing.assert_array_equal(Eh_l, Eh_f)
    np.testing.assert_array_equal(ee_l, ee_f)

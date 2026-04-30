"""Phase 4 tests: full solver (forward + custom_vjp adjoint)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from warp_vp_solver.mesh import make_mesh
from warp_vp_solver.solver import WarpVlasovPoissonSolver

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Fixtures (JAX-style; dt=0.001 like the reference test suite)
# ---------------------------------------------------------------------------


@pytest.fixture
def jax_style_mesh():
    return make_mesh(
        length_x=2.0 * np.pi, length_v=6.0, nx=16, nv=32,
    )


@pytest.fixture
def f_eq(jax_style_mesh):
    return np.exp(-0.5 * jax_style_mesh.V ** 2)


@pytest.fixture
def solver(jax_style_mesh, f_eq):
    return WarpVlasovPoissonSolver(
        mesh=jax_style_mesh, dt=0.001, f_eq=f_eq,
    )


@pytest.fixture
def f0(jax_style_mesh, f_eq):
    perturb = 0.001 * np.cos(jax_style_mesh.X)
    return f_eq * (1.0 + perturb)


# ---------------------------------------------------------------------------
# Ports of the original JAX tests (12 total)
# ---------------------------------------------------------------------------


def test_make_mesh_shapes(jax_style_mesh):
    assert jax_style_mesh.X.shape == (
        jax_style_mesh.nx, jax_style_mesh.nv,
    )
    assert jax_style_mesh.V.shape == (
        jax_style_mesh.nx, jax_style_mesh.nv,
    )


def test_compute_rho_shape(solver, f0):
    rho = solver.compute_rho(f0)
    assert rho.shape == (solver.mesh.nx,)


def test_compute_E_shape(solver, f0):
    E = solver.compute_E(f0)
    assert E.shape == (solver.mesh.nx,)


def test_electric_energy_positive(solver, f0):
    E = solver.compute_E(f0)
    ee = solver.compute_electric_energy(E)
    assert ee >= 0.0


def test_zero_density_gives_zero_field(solver, f_eq):
    rho = solver.compute_rho(f_eq)
    E = solver.compute_E_from_rho(rho)
    np.testing.assert_allclose(E, 0.0, atol=1e-12)


def test_one_time_step_forward(solver, f0):
    H = np.zeros(solver.mesh.nx)
    t_final = solver.dt
    num_steps = int(t_final / solver.dt)

    f_array, f_total, E_array, ee_array = solver.run_forward(
        f0, H, t_final
    )

    assert f_array.shape == (solver.mesh.nx, solver.mesh.nv)
    assert f_total.shape == (num_steps, solver.mesh.nx, solver.mesh.nv)
    assert E_array.shape == (num_steps, solver.mesh.nx)
    assert ee_array.shape == (num_steps,)


def test_mass_conservation(solver, f0):
    H = np.zeros(solver.mesh.nx)
    t_final = 10 * solver.dt
    f_final, _, _, _ = solver.run_forward(f0, H, t_final)

    def mass(f):
        return np.trapezoid(
            np.trapezoid(f, solver.mesh.xs, axis=0),
            solver.mesh.vs,
            axis=0,
        )

    np.testing.assert_allclose(mass(f0), mass(f_final), rtol=1e-5, atol=1e-7)


def test_zero_mode_filtered(solver, f0):
    rho = solver.compute_rho(f0)
    E = solver.compute_E_from_rho(rho)
    np.testing.assert_allclose(np.fft.fft(E)[0], 0.0, atol=1e-12)


def test_linear_landau_damping(solver, f0):
    """Qualitative Landau damping: energy should decay early on."""
    H = np.zeros(solver.mesh.nx)
    t_final = 20 * solver.dt
    _, _, _, ee_array = solver.run_forward(f0, H, t_final)
    ee = ee_array[1:]
    assert ee[-1] < ee[0]
    assert np.max(ee) < 2.0 * ee[0]


# ---------------------------------------------------------------------------
# JAX VJP / gradient tests
# ---------------------------------------------------------------------------


def test_jax_forward_returns_jnp(solver, f0):
    H = np.zeros(solver.mesh.nx)
    out = solver.run_forward_jax(jnp.asarray(f0), jnp.asarray(H), 5 * solver.dt)
    assert all(isinstance(x, jax.Array) for x in out)


def test_grad_kl_runs(solver, f0):
    """Gradients of KL(f_T || f_eq) w.r.t. H should evaluate without error."""
    H0 = jnp.zeros(solver.mesh.nx)
    f_iv = jnp.asarray(f0)
    t_final = 5 * solver.dt

    def cost(H):
        f_array, _, _, _ = solver.run_forward_jax(f_iv, H, t_final)
        # Use a simple quadratic surrogate to keep cotangents nontrivial.
        return jnp.sum(f_array ** 2)

    g = jax.grad(cost)(H0)
    assert g.shape == H0.shape
    assert np.all(np.isfinite(np.asarray(g)))


def test_grad_finite_difference(solver, f0):
    """Compare custom_vjp gradient to a finite difference probe."""
    rng = np.random.default_rng(42)
    H0 = 0.001 * rng.standard_normal(solver.mesh.nx)
    f_iv = jnp.asarray(f0)
    t_final = 3 * solver.dt

    def cost(H):
        _, _, _, ee = solver.run_forward_jax(f_iv, H, t_final)
        return ee[-1]

    g = np.asarray(jax.grad(cost)(jnp.asarray(H0)))

    eps = 1e-6
    direction = rng.standard_normal(solver.mesh.nx)
    H_plus = H0 + eps * direction
    H_minus = H0 - eps * direction
    c_plus = float(cost(jnp.asarray(H_plus)))
    c_minus = float(cost(jnp.asarray(H_minus)))
    fd = (c_plus - c_minus) / (2 * eps)
    ad = float(np.dot(g, direction))
    np.testing.assert_allclose(ad, fd, rtol=1e-4, atol=1e-9)


def test_grad_wrt_initial_condition(solver, f0):
    """jax.grad w.r.t. f_iv via the custom_vjp finite-difference cross-check."""
    rng = np.random.default_rng(0)
    H = jnp.zeros(solver.mesh.nx)
    t_final = 2 * solver.dt
    f_iv = jnp.asarray(f0)

    def cost(f_iv):
        _, _, _, ee = solver.run_forward_jax(f_iv, H, t_final)
        return ee[-1]

    g = np.asarray(jax.grad(cost)(f_iv))
    assert g.shape == f0.shape
    direction = rng.standard_normal(f0.shape)
    eps = 1e-6
    f_plus = f_iv + eps * direction
    f_minus = f_iv - eps * direction
    fd = (float(cost(f_plus)) - float(cost(f_minus))) / (2 * eps)
    ad = float(np.sum(g * direction))
    np.testing.assert_allclose(ad, fd, rtol=1e-4, atol=1e-9)


# Historical note: at v0.1.0 the legacy forward path was cross-validated
# bit-for-bit against the original JAX reference implementation
# (vlasov-poisson, JAX scan), to atol=1e-10.  That reference has since
# been removed from the repository; the optimized path remains anchored
# to that original behavior via tests/test_optimized.py, which checks
# the optimized vs. legacy paths are bit-for-bit equivalent.

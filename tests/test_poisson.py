"""Phase 3 tests: FFT-based Poisson solve."""

from __future__ import annotations

import numpy as np

from warp_vp_solver.poisson import (
    fft_inv_multiplier,
    solve_poisson_host,
    solve_poisson_host_adjoint,
)


def test_zero_density_gives_zero_field():
    rho = np.zeros(64)
    E = solve_poisson_host(rho, period_x=2.0 * np.pi)
    np.testing.assert_allclose(E, 0.0, atol=1e-15)


def test_zero_mode_filtered():
    rng = np.random.default_rng(0)
    rho = rng.standard_normal(64)
    E = solve_poisson_host(rho, period_x=2.0 * np.pi)
    np.testing.assert_allclose(np.fft.fft(E)[0], 0.0, atol=1e-12)


def test_recovers_single_mode():
    """For rho(x) = sin(k0 x) the analytical E is cos(k0 x) / k0."""
    nx = 64
    period = 2.0 * np.pi
    xs = np.linspace(0.0, period, nx, endpoint=False)
    k0 = 2.0  # integer multiple of fundamental
    rho = np.sin(k0 * xs)
    E = solve_poisson_host(rho, period_x=period)
    expected = np.cos(k0 * xs) / k0
    np.testing.assert_allclose(E, expected, atol=1e-12)


def test_matches_jax_reference():
    """Validate against the original JAX expression for the spectral mult."""
    nx = 32
    period = 5.0
    xs = np.linspace(0.0, period, nx, endpoint=False)
    rho = np.exp(-((xs - period / 2.0) ** 2)) - np.mean(
        np.exp(-((xs - period / 2.0) ** 2))
    )

    rho_hat = np.fft.fft(rho)
    inv_mult = -1.0 / (
        1j
        * 2.0
        * np.pi
        * np.fft.fftfreq(nx, d=period / nx)[1:]
    )
    E_hat = np.zeros_like(rho_hat)
    E_hat[1:] = inv_mult * rho_hat[1:]
    expected = np.real(np.fft.ifft(E_hat))

    got = solve_poisson_host(rho, period_x=period)
    np.testing.assert_allclose(got, expected, atol=1e-13)


def test_adjoint_inner_product():
    """<A x, y> == <x, A^T y> for the linear forward map A: rho -> E."""
    nx = 64
    period = 2.0 * np.pi
    rng = np.random.default_rng(0)
    rho = rng.standard_normal(nx)
    g_E = rng.standard_normal(nx)

    E = solve_poisson_host(rho, period_x=period)
    rho_bar = solve_poisson_host_adjoint(g_E, period_x=period)

    np.testing.assert_allclose(
        np.dot(E, g_E), np.dot(rho, rho_bar), rtol=1e-12, atol=1e-12
    )


def test_inv_multiplier_zero_dc():
    mult = fft_inv_multiplier(32, period_x=4.0)
    assert mult[0] == 0.0

"""Shared pytest fixtures for the Warp Vlasov-Poisson solver."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from warp_vp_solver.mesh import make_mesh


@pytest.fixture(scope="session", autouse=True)
def _wp_init():
    wp.init()
    yield


@pytest.fixture
def small_mesh():
    return make_mesh(length_x=2.0 * np.pi, length_v=6.0, nx=16, nv=32)


@pytest.fixture
def two_stream_mesh():
    return make_mesh(length_x=10.0 * np.pi, length_v=6.0, nx=64, nv=64)


@pytest.fixture
def two_stream_state(two_stream_mesh):
    """Two-stream equilibrium and a small perturbed initial condition."""
    mu1 = 2.4
    V = two_stream_mesh.V
    f_eq = (
        np.exp(-0.5 * (V - mu1) ** 2) + np.exp(-0.5 * (V + mu1) ** 2)
    ) / (2.0 * np.sqrt(2.0 * np.pi))
    f_iv = (1.0 + 1e-3 * np.cos(0.2 * two_stream_mesh.X)) * f_eq
    return f_eq, f_iv


@pytest.fixture
def maxwell_state(small_mesh):
    """Maxwellian equilibrium and small cosine-perturbed IC.

    Mirrors the fixture in the original JAX test suite.
    """
    f_eq = np.exp(-0.5 * small_mesh.V ** 2)
    perturb = 1e-3 * np.cos(small_mesh.X)
    f_iv = f_eq * (1.0 + perturb)
    return f_eq, f_iv

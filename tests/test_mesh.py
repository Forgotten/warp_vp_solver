"""Phase 1 tests: Mesh construction."""

import numpy as np

from warp_vp_solver.mesh import Mesh, make_mesh


def test_make_mesh_shapes(small_mesh):
    assert small_mesh.X.shape == (small_mesh.nx, small_mesh.nv)
    assert small_mesh.V.shape == (small_mesh.nx, small_mesh.nv)
    assert small_mesh.xs.shape == (small_mesh.nx,)
    assert small_mesh.vs.shape == (small_mesh.nv,)


def test_make_mesh_periodic_endpoints(small_mesh):
    assert small_mesh.xs[0] == 0.0
    np.testing.assert_allclose(
        small_mesh.xs[-1] + small_mesh.dx, small_mesh.period_x
    )
    np.testing.assert_allclose(
        small_mesh.vs[-1] + small_mesh.dv, small_mesh.period_v
    )


def test_make_mesh_dx_dv():
    mesh = make_mesh(length_x=2.0, length_v=3.0, nx=4, nv=6)
    np.testing.assert_allclose(mesh.dx, 0.5)
    np.testing.assert_allclose(mesh.dv, 1.0)


def test_meshgrid_consistency(small_mesh):
    np.testing.assert_array_equal(small_mesh.X[:, 0], small_mesh.xs)
    np.testing.assert_array_equal(small_mesh.V[0, :], small_mesh.vs)


def test_dataclass_type():
    mesh = make_mesh(length_x=1.0, length_v=1.0, nx=8, nv=8)
    assert isinstance(mesh, Mesh)

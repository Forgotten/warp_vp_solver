"""Phase-space mesh for the 1D-1V Vlasov-Poisson solver."""

from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass
class Mesh:
    """Mesh object for the 1D-1V phase space."""

    xs: np.ndarray
    dx: float
    vs: np.ndarray
    dv: float
    V: np.ndarray
    X: np.ndarray
    period_x: float
    period_v: float
    nx: int
    nv: int


def make_mesh(
    length_x: float,
    length_v: float,
    nx: int,
    nv: int,
) -> Mesh:
    """Generate a phase-space mesh.

    Args:
        length_x: spatial domain length, x in [0, length_x).
        length_v: half velocity domain width, v in [-length_v, length_v).
        nx: number of spatial grid points.
        nv: number of velocity grid points.
    """
    xs = np.linspace(0.0, length_x, nx, endpoint=False, dtype=np.float64)
    dx = float(xs[1] - xs[0])
    vs = np.linspace(-length_v, length_v, nv, endpoint=False, dtype=np.float64)
    dv = float(vs[1] - vs[0])
    V, X = np.meshgrid(vs, xs)
    return Mesh(
        xs=xs,
        dx=dx,
        vs=vs,
        dv=dv,
        X=X,
        V=V,
        period_x=float(length_x),
        period_v=float(length_v),
        nx=int(nx),
        nv=int(nv),
    )

"""NVIDIA Warp kernels for the 1D-1V Vlasov-Poisson solver.

The kernels here implement the building blocks of the operator-split
semi-Lagrangian scheme.  Two parallel sets are provided:

* **Legacy / reference kernels** - one-to-one ports of the original
  JAX primitives.  These are the ground truth used to cross-check
  every optimized variant.

  * ``semilag_x_kernel``           - half-step advection in x
  * ``semilag_v_kernel``           - full-step advection in v
  * ``compute_rho_kernel``         - charge density via trapezoidal rule
  * ``compute_ee_kernel``          - electric energy via trapezoidal rule
  * ``semilag_x_adjoint_kernel``   - transpose of the x advection
  * ``semilag_v_adjoint_kernel``   - transpose of the v advection
  * ``compute_rho_adjoint_kernel`` - transpose of the v-direction reduction
  * ``compute_ee_adjoint_kernel``  - transpose of the x-direction reduction

* **Optimized kernels** (used when ``WarpVlasovPoissonSolver`` is built
  with ``optimized=True``):

  * ``semilag_v_fused_kernel``         - F1: ``E_total = E + H`` is
    computed inside the kernel, eliminating one host-to-device upload
    per time step.
  * ``semilag_v_fused_adjoint_kernel`` - matching adjoint that produces
    the cotangent on ``E_total`` so the caller can scatter it onto
    both ``E`` and ``H`` (since ``E_total = E + H``).
  * ``axpy_1d_kernel`` / ``axpy_2d_kernel`` / ``copy_1d_kernel`` -
    F4: device-side helpers that let the optimized adjoint pipe
    cotangents through the time loop without round-tripping each
    intermediate through host memory.
  * ``_periodic_indices_fast``         - M1: drops the dead
    ``if i0 < 0`` defensive branch that the wrap math already
    guarantees can never fire.
"""

from __future__ import annotations

import warp as wp


# ---------------------------------------------------------------------------
# Helper functions (@wp.func)
# ---------------------------------------------------------------------------


@wp.func
def _periodic_indices(
    x_query: wp.float64,
    x0: wp.float64,
    h: wp.float64,
    n: wp.int32,
    period: wp.float64,
):
    """Wrap ``x_query`` and return the bracketing indices and lerp weight.

    Returns a tuple ``(i0, i1, t)`` such that the interpolated value at
    ``x_query`` is ``(1 - t) * data[i0] + t * data[i1]`` for any
    periodic 1D array sampled on ``x0 + k * h`` with period ``period``.
    """
    x_rel = x_query - x0
    x_wrapped = x_rel - wp.floor(x_rel / period) * period
    idx_f = x_wrapped / h
    i0 = wp.int32(wp.floor(idx_f)) % n
    if i0 < wp.int32(0):
        i0 = i0 + n
    i1 = (i0 + wp.int32(1)) % n
    t = idx_f - wp.floor(idx_f)
    return i0, i1, t


@wp.func
def periodic_interp_col(
    data_2d: wp.array2d(dtype=wp.float64),
    col: wp.int32,
    x_query: wp.float64,
    x0: wp.float64,
    h: wp.float64,
    n: wp.int32,
    period: wp.float64,
) -> wp.float64:
    """Periodic linear interpolation along axis 0 of a 2D array."""
    i0, i1, t = _periodic_indices(x_query, x0, h, n, period)
    one = wp.float64(1.0)
    return (one - t) * data_2d[i0, col] + t * data_2d[i1, col]


@wp.func
def periodic_interp_row(
    data_2d: wp.array2d(dtype=wp.float64),
    row: wp.int32,
    x_query: wp.float64,
    x0: wp.float64,
    h: wp.float64,
    n: wp.int32,
    period: wp.float64,
) -> wp.float64:
    """Periodic linear interpolation along axis 1 of a 2D array."""
    i0, i1, t = _periodic_indices(x_query, x0, h, n, period)
    one = wp.float64(1.0)
    return (one - t) * data_2d[row, i0] + t * data_2d[row, i1]


# ---------------------------------------------------------------------------
# Forward kernels
# ---------------------------------------------------------------------------


@wp.kernel
def semilag_x_kernel(
    f_in: wp.array2d(dtype=wp.float64),
    f_out: wp.array2d(dtype=wp.float64),
    xs: wp.array(dtype=wp.float64),
    vs: wp.array(dtype=wp.float64),
    dt: wp.float64,
    dx: wp.float64,
    nx: wp.int32,
    period_x: wp.float64,
):
    """Half-step semi-Lagrangian advection in x.

    Solves d_t f + v d_x f = 0 over a half time step ``0.5 * dt`` by
    tracing characteristics backwards: ``x_foot = x_i - 0.5 * v_j * dt``.
    """
    i, j = wp.tid()
    half = wp.float64(0.5)
    x_foot = xs[i] - half * vs[j] * dt
    f_out[i, j] = periodic_interp_col(
        f_in, j, x_foot, xs[0], dx, nx, period_x
    )


@wp.kernel
def semilag_v_kernel(
    f_in: wp.array2d(dtype=wp.float64),
    f_out: wp.array2d(dtype=wp.float64),
    vs: wp.array(dtype=wp.float64),
    E_total: wp.array(dtype=wp.float64),
    dt: wp.float64,
    dv: wp.float64,
    nv: wp.int32,
    period_v_full: wp.float64,
):
    """Full-step semi-Lagrangian advection in v.

    Solves d_t f - (E + H) d_v f = 0 over a full time step ``dt``.
    ``period_v_full`` is the full velocity period (``2 * length_v``).
    """
    i, j = wp.tid()
    v_foot = vs[j] - E_total[i] * dt
    f_out[i, j] = periodic_interp_row(
        f_in, i, v_foot, vs[0], dv, nv, period_v_full
    )


@wp.kernel
def compute_rho_kernel(
    f_eq: wp.array2d(dtype=wp.float64),
    f: wp.array2d(dtype=wp.float64),
    rho: wp.array(dtype=wp.float64),
    dv: wp.float64,
    nv: wp.int32,
):
    """Charge density rho(x) = integral (f_eq - f) dv via trapezoidal rule."""
    i = wp.tid()
    half = wp.float64(0.5)
    acc = half * (f_eq[i, 0] - f[i, 0])
    for k in range(1, nv - 1):
        acc = acc + (f_eq[i, k] - f[i, k])
    last = nv - 1
    acc = acc + half * (f_eq[i, last] - f[i, last])
    rho[i] = acc * dv


@wp.kernel
def compute_ee_kernel(
    E: wp.array(dtype=wp.float64),
    out: wp.array(dtype=wp.float64),
    dx: wp.float64,
    nx: wp.int32,
):
    """Electric energy 0.5 * integral E(x)**2 dx via trapezoidal rule.

    Launched with ``dim=1`` - the kernel reduces sequentially.  The size
    of the time loop dwarfs the cost of the reduction, so we keep this
    simple instead of going parallel.
    """
    tid = wp.tid()
    if tid != 0:
        return
    half = wp.float64(0.5)
    acc = half * E[0] * E[0]
    for k in range(1, nx - 1):
        acc = acc + E[k] * E[k]
    last = nx - 1
    acc = acc + half * E[last] * E[last]
    out[0] = half * acc * dx


# ---------------------------------------------------------------------------
# Adjoint (transpose) kernels - used by the custom VJP
# ---------------------------------------------------------------------------
#
# The forward semi-Lagrangian kernels apply a sparse linear operator
# ``L`` to the distribution: each output cell is a convex combination of
# two input cells.  The adjoint ``L^T`` therefore scatters the upstream
# cotangent ``g_out`` back to the same two input cells with the same
# weights.  Because multiple output threads can scatter to the same
# input cell concurrently we use ``wp.atomic_add``.
#
# These kernels are independent of any Warp-native autodiff and work on
# CPU and CUDA backends alike.


@wp.kernel
def semilag_x_adjoint_kernel(
    g_out: wp.array2d(dtype=wp.float64),
    g_in: wp.array2d(dtype=wp.float64),
    xs: wp.array(dtype=wp.float64),
    vs: wp.array(dtype=wp.float64),
    dt: wp.float64,
    dx: wp.float64,
    nx: wp.int32,
    period_x: wp.float64,
):
    """Adjoint of ``semilag_x_kernel``.

    For each ``(i, j)`` thread the forward pass set
    ``g_out[i, j] = (1 - t) * g_in[i0, j] + t * g_in[i1, j]``.
    The transpose scatters ``g_out[i, j]`` to ``g_in[i0, j]`` and
    ``g_in[i1, j]`` weighted by ``(1 - t)`` and ``t``.
    """
    i, j = wp.tid()
    half = wp.float64(0.5)
    x_foot = xs[i] - half * vs[j] * dt
    i0, i1, t = _periodic_indices(x_foot, xs[0], dx, nx, period_x)
    one = wp.float64(1.0)
    w0 = (one - t) * g_out[i, j]
    w1 = t * g_out[i, j]
    wp.atomic_add(g_in, i0, j, w0)
    wp.atomic_add(g_in, i1, j, w1)


@wp.kernel
def semilag_v_adjoint_kernel(
    g_out: wp.array2d(dtype=wp.float64),
    g_in: wp.array2d(dtype=wp.float64),
    g_E: wp.array(dtype=wp.float64),
    f_in: wp.array2d(dtype=wp.float64),
    vs: wp.array(dtype=wp.float64),
    E_total: wp.array(dtype=wp.float64),
    dt: wp.float64,
    dv: wp.float64,
    nv: wp.int32,
    period_v_full: wp.float64,
):
    """Adjoint of ``semilag_v_kernel``.

    Scatters the upstream cotangent into ``g_in`` and additionally
    accumulates the gradient w.r.t. ``E_total`` into ``g_E``.  The
    derivative of ``v_foot = v_j - E[i] * dt`` w.r.t. ``E[i]`` is
    ``-dt``, and the derivative of the linear-interp output w.r.t.
    ``v_foot`` is ``(f_in[i, i1] - f_in[i, i0]) / dv``.
    """
    i, j = wp.tid()
    v_foot = vs[j] - E_total[i] * dt
    i0, i1, t = _periodic_indices(v_foot, vs[0], dv, nv, period_v_full)
    one = wp.float64(1.0)
    g = g_out[i, j]
    wp.atomic_add(g_in, i, i0, (one - t) * g)
    wp.atomic_add(g_in, i, i1, t * g)
    df_dvfoot = (f_in[i, i1] - f_in[i, i0]) / dv
    wp.atomic_add(g_E, i, -dt * df_dvfoot * g)


@wp.kernel
def compute_rho_adjoint_kernel(
    g_rho: wp.array(dtype=wp.float64),
    g_f: wp.array2d(dtype=wp.float64),
    dv: wp.float64,
    nv: wp.int32,
):
    """Adjoint of ``compute_rho_kernel`` w.r.t. ``f``.

    rho[i] = dv * (0.5*(f_eq[i,0]-f[i,0]) + sum_k (f_eq[i,k]-f[i,k])
                   + 0.5*(f_eq[i,nv-1]-f[i,nv-1])).

    d rho[i] / d f[i,k] is ``-dv`` interior, ``-0.5*dv`` at the ends.
    The adjoint thus *subtracts* a broadcast of ``g_rho[i]`` weighted by
    those factors.
    """
    i, j = wp.tid()
    half = wp.float64(0.5)
    last = nv - 1
    if j == 0 or j == last:
        w = -half * dv
    else:
        w = -dv
    g_f[i, j] = g_f[i, j] + w * g_rho[i]


@wp.kernel
def compute_ee_adjoint_kernel(
    g_ee: wp.array(dtype=wp.float64),
    E: wp.array(dtype=wp.float64),
    g_E: wp.array(dtype=wp.float64),
    dx: wp.float64,
    nx: wp.int32,
):
    """Adjoint of ``compute_ee_kernel`` w.r.t. ``E``.

    ee = 0.5 * dx * (0.5*E[0]^2 + sum E[k]^2 + 0.5*E[-1]^2).
    d ee / d E[k] = dx * E[k] interior, 0.5*dx*E[k] at ends.
    """
    k = wp.tid()
    half = wp.float64(0.5)
    last = nx - 1
    if k == 0 or k == last:
        w = half * dx
    else:
        w = dx
    g_E[k] = g_E[k] + w * E[k] * g_ee[0]


# ===========================================================================
# OPTIMIZED KERNELS
# ===========================================================================
#
# These live next to the legacy kernels above so the two paths can be
# unit-tested against each other.  The optimized solver path
# (``WarpVlasovPoissonSolver(..., optimized=True)``) calls the kernels
# below; the legacy path (``optimized=False``) calls the originals.
#
# Each optimization is annotated with the F-/M- tag from the design
# review:
#   F1 - fuse E + H addition into the v-step kernel
#   F4 - keep adjoint cotangents on-device with axpy / copy helpers
#   M1 - remove the dead negative-index branch from periodic indexing


@wp.func
def _periodic_indices_fast(
    x_query: wp.float64,
    x0: wp.float64,
    h: wp.float64,
    n: wp.int32,
    period: wp.float64,
):
    """Same as ``_periodic_indices`` minus the dead ``i0 < 0`` branch (M1).

    Because ``x_wrapped = x_rel - floor(x_rel/period)*period`` produces
    a value in ``[0, period)`` for every finite real, ``floor(idx_f)``
    is always non-negative and ``i0 = floor(idx_f) % n`` stays in
    ``[0, n-1]``.  The defensive correction in ``_periodic_indices``
    therefore never fires; this fast variant drops it to remove a
    per-thread compare+branch from the innermost interpolation step.
    """
    x_rel = x_query - x0
    x_wrapped = x_rel - wp.floor(x_rel / period) * period
    idx_f = x_wrapped / h
    i0 = wp.int32(wp.floor(idx_f)) % n
    i1 = (i0 + wp.int32(1)) % n
    t = idx_f - wp.floor(idx_f)
    return i0, i1, t


@wp.kernel
def semilag_v_fused_kernel(
    f_in: wp.array2d(dtype=wp.float64),
    f_out: wp.array2d(dtype=wp.float64),
    vs: wp.array(dtype=wp.float64),
    E: wp.array(dtype=wp.float64),
    H: wp.array(dtype=wp.float64),
    dt: wp.float64,
    dv: wp.float64,
    nv: wp.int32,
    period_v_full: wp.float64,
):
    """F1: semi-Lagrangian step in v with ``E_total = E + H`` fused inline.

    Identical numerical behavior to ``semilag_v_kernel`` invoked with
    ``E_total = E + H``, but eliminates the host-side addition and the
    ``(nx,)`` upload of ``E_total`` performed by the legacy path on
    every time step.
    """
    i, j = wp.tid()
    E_total_i = E[i] + H[i]
    v_foot = vs[j] - E_total_i * dt
    i0, i1, t = _periodic_indices_fast(
        v_foot, vs[0], dv, nv, period_v_full
    )
    one = wp.float64(1.0)
    f_out[i, j] = (one - t) * f_in[i, i0] + t * f_in[i, i1]


@wp.kernel
def semilag_v_fused_adjoint_kernel(
    g_out: wp.array2d(dtype=wp.float64),
    g_in: wp.array2d(dtype=wp.float64),
    g_E_total: wp.array(dtype=wp.float64),
    f_in: wp.array2d(dtype=wp.float64),
    vs: wp.array(dtype=wp.float64),
    E: wp.array(dtype=wp.float64),
    H: wp.array(dtype=wp.float64),
    dt: wp.float64,
    dv: wp.float64,
    nv: wp.int32,
    period_v_full: wp.float64,
):
    """Adjoint of ``semilag_v_fused_kernel``.

    Produces the cotangent on ``f_in`` (``g_in``) and on ``E_total``
    (``g_E_total``).  Because ``E_total = E + H`` is a sum, the caller
    simply adds ``g_E_total`` to both ``g_E`` and ``g_H`` after the
    kernel returns.  Operating on ``E`` and ``H`` separately keeps the
    full forward / backward path away from any ``E_total`` upload.
    """
    i, j = wp.tid()
    E_total_i = E[i] + H[i]
    v_foot = vs[j] - E_total_i * dt
    i0, i1, t = _periodic_indices_fast(
        v_foot, vs[0], dv, nv, period_v_full
    )
    one = wp.float64(1.0)
    g = g_out[i, j]
    wp.atomic_add(g_in, i, i0, (one - t) * g)
    wp.atomic_add(g_in, i, i1, t * g)
    df_dvfoot = (f_in[i, i1] - f_in[i, i0]) / dv
    wp.atomic_add(g_E_total, i, -dt * df_dvfoot * g)


# --- F4 helpers: tiny vector ops used to keep the adjoint on-device -----


@wp.kernel
def axpy_1d_kernel(
    a: wp.float64,
    x: wp.array(dtype=wp.float64),
    y: wp.array(dtype=wp.float64),
):
    """y[i] += a * x[i]."""
    i = wp.tid()
    y[i] = y[i] + a * x[i]


@wp.kernel
def axpy_2d_kernel(
    a: wp.float64,
    x: wp.array2d(dtype=wp.float64),
    y: wp.array2d(dtype=wp.float64),
):
    """y[i, j] += a * x[i, j]."""
    i, j = wp.tid()
    y[i, j] = y[i, j] + a * x[i, j]


@wp.kernel
def copy_1d_kernel(
    src: wp.array(dtype=wp.float64),
    dst: wp.array(dtype=wp.float64),
):
    """dst[i] = src[i]."""
    i = wp.tid()
    dst[i] = src[i]

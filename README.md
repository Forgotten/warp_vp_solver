# vlasov-poisson-warp

A 1D-1V Vlasov-Poisson semi-Lagrangian solver built on
[NVIDIA Warp](https://github.com/NVIDIA/warp).

The forward time loop runs entirely in Warp kernels (CUDA when
available, CPU otherwise); a hand-written discrete adjoint exposed
through `jax.custom_vjp` lets `jax.grad` and Optax drive
gradient-based optimization through the solver.

## Install

```bash
git clone https://github.com/Forgotten/vlasov-poisson-warp.git
cd vlasov-poisson-warp
pip install -e ".[dev,notebooks]"
```

Requires Python >= 3.10.  `warp-lang` will use the GPU automatically on
CUDA hosts and fall back to a CPU device elsewhere.

## Layout

```
warp_vp_solver/
  mesh.py        # phase-space mesh (NumPy)
  kernels.py     # Warp @wp.kernel / @wp.func GPU kernels (legacy + optimized)
  poisson.py     # FFT-based Poisson solve (host FFT + GPU tile_fft path)
  solver.py      # WarpVlasovPoissonSolver + jax.custom_vjp adjoint
  utils.py       # external_electric_field, cost functions, plotting
tests/           # pytest suite (47 tests; bit-for-bit cross-checks)
notebooks/       # 01_Two_Stream_Warp.ipynb, 02_Bump_on_Tail_Warp.ipynb
docs/            # implementation plan + design notes
```

## Quick start

```python
import numpy as np
import warp as wp
from warp_vp_solver import make_mesh, WarpVlasovPoissonSolver

wp.init()

mesh = make_mesh(length_x=10*np.pi, length_v=6.0, nx=256, nv=256)
mu = 2.4
f_eq = (np.exp(-0.5*(mesh.V - mu)**2) + np.exp(-0.5*(mesh.V + mu)**2)) \
       / (2 * np.sqrt(2*np.pi))
f_iv = (1 + 1e-3 * np.cos(0.2 * mesh.X)) * f_eq

solver = WarpVlasovPoissonSolver(mesh=mesh, dt=0.05, f_eq=f_eq)
H = np.zeros(mesh.nx)
f_final, f_history, E_history, ee_history = solver.run_forward(
    f_iv, H, t_final=5.0
)
```

## Optimized vs. legacy paths

`WarpVlasovPoissonSolver` accepts an `optimized` flag (default `True`):

* `optimized=True` uses the fused `semilag_v_fused_kernel`, an
  on-device cotangent pipeline for the adjoint, and supports
  `record_history=False` for forward-only workloads.
* `optimized=False` runs a 1:1 reference implementation that mirrors
  the original JAX semi-Lagrangian scheme; useful as ground truth for
  cross-validation.

The two paths are bit-for-bit equivalent on every output, verified by
`tests/test_optimized.py`.

## FFT Poisson solve

The package ships both a fused `wp.tile_fft` kernel (Warp >= 1.6 with
cuFFT) and a host-side NumPy FFT fallback.  The fallback is selected
automatically when `wp.tile_fft` is unavailable (e.g. on macOS or any
CPU-only Warp build); both paths live in `warp_vp_solver/poisson.py`.

## Tests

```bash
pytest tests/ -v
```

47 tests covering mesh construction, every kernel and its adjoint
(verified by both inner-product equality and finite-difference
probes), the FFT Poisson solver, the full time loop, gradients
through the `custom_vjp` wrapper, and bit-for-bit equivalence between
the optimized and legacy paths.

## License

MIT.  See [LICENSE](LICENSE).

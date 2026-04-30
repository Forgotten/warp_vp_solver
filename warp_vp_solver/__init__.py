"""Vlasov-Poisson semi-Lagrangian solver on NVIDIA Warp."""

from .mesh import Mesh, make_mesh
from .solver import WarpVlasovPoissonSolver, run_forward_jax_compatible
from .utils import (
    electric_energy_in_time,
    external_electric_field,
    kl_divergence,
    make_cost_function_ee,
    make_cost_function_eet,
    make_cost_function_kl,
    plot_distribution,
    plot_feq_distribution,
    plot_inital_solve,
    plot_results_BoT,
    plot_results_TS,
)

__all__ = [
    "Mesh",
    "WarpVlasovPoissonSolver",
    "electric_energy_in_time",
    "external_electric_field",
    "kl_divergence",
    "make_cost_function_ee",
    "make_cost_function_eet",
    "make_cost_function_kl",
    "make_mesh",
    "plot_distribution",
    "plot_feq_distribution",
    "plot_inital_solve",
    "plot_results_BoT",
    "plot_results_TS",
    "run_forward_jax_compatible",
]
__version__ = "0.1.0"

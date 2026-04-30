"""Utility functions: external field, cost functions, plotting.

These mirror ``vp_solver.utils`` from the original JAX package.  Cost
functions still operate on JAX arrays so that ``jax.grad`` / ``optax``
can drive optimization, and they invoke
``WarpVlasovPoissonSolver.run_forward_jax`` (the ``custom_vjp``
wrapper) on the solver path.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import axes, figure

from .mesh import Mesh
from .solver import WarpVlasovPoissonSolver


matplotlib.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})


# ===== External electric field =====

def external_electric_field(
    ak: jax.Array,
    mesh: Mesh,
    k_0: float,
) -> jax.Array:
    """H(x) = sum_k a_k cos(k0 k x) + b_k sin(k0 k x)."""
    N = ak.shape[1]
    k = jnp.arange(1, N + 1)
    xs = jnp.asarray(mesh.xs)
    cos_term = ak[0, :] @ jnp.cos(k_0 * k[:, None] * xs)
    sin_term = ak[1, :] @ jnp.sin(k_0 * k[:, None] * xs)
    return cos_term + sin_term


# ===== Cost functions =====

def kl_divergence(
    f_T: jax.Array,
    solver: WarpVlasovPoissonSolver,
    eps: float = 1e-12,
) -> jax.Array:
    """KL(f_T || f_eq) integrated over phase space."""
    xs = jnp.asarray(solver.mesh.xs)
    vs = jnp.asarray(solver.mesh.vs)
    f_eq = jnp.asarray(solver.f_eq)

    norm_final = jnp.trapezoid(jnp.trapezoid(f_T, vs, axis=1), xs) + eps
    norm_eq = jnp.trapezoid(jnp.trapezoid(f_eq, vs, axis=1), xs) + eps

    f_final_normed = f_T / norm_final
    f_eq_normed = f_eq / norm_eq

    return jnp.trapezoid(
        jnp.trapezoid(
            jax.scipy.special.rel_entr(f_final_normed, f_eq_normed + eps),
            vs,
            axis=1,
        ),
        xs,
    )


def make_cost_function_kl(
    solver: WarpVlasovPoissonSolver,
    solver_jit: Callable,
    f_iv: jax.Array,
    k_0: float,
    t_final: float,
) -> Callable[[jax.Array], jax.Array]:
    """KL-divergence cost as a function of the external-field coefficients."""
    def cost_function_kl(a_k: jax.Array) -> jax.Array:
        H = external_electric_field(a_k, solver.mesh, k_0)
        f_array, _, _, _ = solver_jit(f_iv, H, t_final)
        return kl_divergence(f_array, solver)

    return cost_function_kl


def make_cost_function_ee(
    solver: WarpVlasovPoissonSolver,
    solver_jit: Callable,
    f_iv: jax.Array,
    k_0: float,
    t_final: float,
) -> Callable[[jax.Array], jax.Array]:
    """Final-time electric energy cost."""
    def cost_function_ee(a_k: jax.Array) -> jax.Array:
        H = external_electric_field(a_k, solver.mesh, k_0)
        _, _, _, ee_array = solver_jit(f_iv, H, t_final)
        return ee_array[-1]

    return cost_function_ee


def electric_energy_in_time(
    ee_array: jax.Array,
    solver: WarpVlasovPoissonSolver,
) -> jax.Array:
    """Time-integrated electric energy."""
    return jnp.trapezoid(ee_array, dx=solver.dt)


def make_cost_function_eet(
    solver: WarpVlasovPoissonSolver,
    solver_jit: Callable,
    f_iv: jax.Array,
    k_0: float,
    t_final: float,
) -> Callable[[jax.Array], jax.Array]:
    """Time-integrated electric energy cost."""
    def cost_function_eet(a_k: jax.Array) -> jax.Array:
        H = external_electric_field(a_k, solver.mesh, k_0)
        _, _, _, ee_array = solver_jit(f_iv, H, t_final)
        return electric_energy_in_time(ee_array, solver)

    return cost_function_eet


# ===== Plotting =====

def plot_feq_distribution(
    fig: figure.Figure,
    ax: axes.Axes,
    f_eq,
    title: str,
    mesh: Mesh,
    sci: bool = False,
) -> None:
    f_eq = np.asarray(f_eq)
    im = ax.imshow(
        f_eq.T,
        extent=(
            float(mesh.xs[0]), float(mesh.xs[-1]),
            float(mesh.vs[0]), float(mesh.vs[-1]),
        ),
        aspect="auto",
        cmap="plasma",
    )
    ax.set_title(title)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$v$")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if sci:
        cbar.ax.set_yscale("log")


def plot_distribution(
    fig: figure.Figure,
    ax: axes.Axes,
    data,
    title: str,
    time: float,
    mesh: Mesh,
    sci: bool = False,
) -> None:
    data = np.asarray(data)
    im = ax.imshow(
        data.T,
        extent=(
            float(mesh.xs[0]), float(mesh.xs[-1]),
            float(mesh.vs[0]), float(mesh.vs[-1]),
        ),
        aspect="auto",
        cmap="plasma",
    )
    ax.set_title(f"{title} (T={time:.0f})")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$v$")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if sci:
        cbar.ax.set_yscale("log")


def plot_inital_solve(
    fig: figure.Figure,
    axs: list[axes.Axes],
    f_eq,
    f_array_1,
    ee_array_1,
    f_array_2,
    ee_array_2,
    mesh: Mesh,
    t_values,
    sci: bool = False,
) -> None:
    plot_feq_distribution(
        fig, axs[0], f_eq, "Distribution of $f_{eq}$", mesh, sci
    )
    plot_distribution(
        fig, axs[1], f_array_1,
        "Distribution of $f[H\\equiv 0]$",
        float(np.asarray(t_values)[-1]), mesh, sci,
    )
    plot_distribution(
        fig, axs[2], f_array_2,
        "Distribution of $f[H]$",
        float(np.asarray(t_values)[-1]), mesh, sci,
    )
    axs[3].ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    axs[3].plot(np.asarray(t_values), np.asarray(ee_array_1), label="No $H$")
    axs[3].plot(
        np.asarray(t_values), np.asarray(ee_array_2), label="Good initial $H$"
    )
    axs[3].set_xlabel("$t$")
    axs[3].set_title("$\\mathcal{E}_{f}(t)$")
    axs[3].legend()


def plot_results_TS(
    fig: figure.Figure,
    axs: list[axes.Axes],
    f_final,
    E_array,
    H,
    ee_array,
    objective_values,
    t_values,
    mesh: Mesh,
) -> None:
    t_values = np.asarray(t_values)
    E_array = np.asarray(E_array)
    H = np.asarray(H)
    dt = t_values[1] - t_values[0]

    plot_distribution(
        fig, axs[0], f_final, "Distribution of $f[H]$",
        float(t_values[-1]), mesh,
    )

    axs[1].ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    axs[1].plot(mesh.xs, H, label="$H(x)$")
    for idx in (0, 99, 199, 299):
        if idx < E_array.shape[0]:
            axs[1].plot(
                mesh.xs, E_array[idx] - H,
                label=f"$E(t={idx*dt:.0f},x)$",
            )
    axs[1].set_xlabel("$x$")
    axs[1].set_title("Electric fields")
    axs[1].legend(loc="upper right")

    axs[2].ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    axs[2].plot(t_values, np.asarray(ee_array))
    axs[2].set_xlabel("$t$")
    axs[2].set_title("$\\mathcal{E}_{f}(t)$")

    axs[3].plot(np.asarray(objective_values))
    axs[3].set_yscale("log")
    axs[3].set_xlabel("Iteration")
    axs[3].set_title("Convergence of Objective")


def plot_results_BoT(
    fig: figure.Figure,
    axs: list[axes.Axes],
    f_final,
    E_array,
    H,
    ee_array,
    objective_values,
    t_values,
    mesh: Mesh,
) -> None:
    t_values = np.asarray(t_values)
    E_array = np.asarray(E_array)
    H = np.asarray(H)
    dt = t_values[1] - t_values[0]

    plot_distribution(
        fig, axs[0], f_final, "Distribution of $f[H]$",
        float(t_values[-1]), mesh,
    )

    axs[1].ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    axs[1].plot(mesh.xs, H, label="$H(x)$")
    for idx in (0, 199, 299, 399):
        if idx < E_array.shape[0]:
            axs[1].plot(
                mesh.xs, E_array[idx] - H,
                label=f"$E(t={idx*dt:.0f},x)$",
            )
    axs[1].set_xlabel("$x$")
    axs[1].set_title("Electric fields")
    axs[1].legend(loc="upper right")

    axs[2].ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    axs[2].plot(t_values, np.asarray(ee_array))
    axs[2].set_xlabel("$t$")
    axs[2].set_title("$\\mathcal{E}_{f}(t)$")

    axs[3].plot(np.asarray(objective_values))
    axs[3].set_yscale("log")
    axs[3].set_xlabel("Iteration")
    axs[3].set_title("Convergence of Objective")

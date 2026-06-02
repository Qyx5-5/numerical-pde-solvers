"""Canonical numerical PDE cases with optional visualizations."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from numerical_pde import (
    ExplicitPDESolver,
    Grid1D,
    central_second,
    l2_error,
    mass,
    periodic,
    upwind_first,
)

FIGSIZE = (9.0, 3.8)
COLORS = {
    "primary": "#1f77b4",
    "secondary": "#d62728",
    "accent": "#2ca02c",
    "muted": "#6b7280",
}


def set_publication_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
        }
    )


def save_figure(fig: plt.Figure, output: Path, formats: tuple[str, ...]) -> list[Path]:
    output.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        path = output.with_suffix(f".{fmt}")
        fig.savefig(path, bbox_inches="tight", transparent=False)
        written.append(path)
    plt.close(fig)
    return written


def solve_heat(points: int = 128, diffusivity: float = 0.05, final_time: float = 0.25):
    grid = Grid1D(0.0, 2.0 * np.pi, points)
    u0 = np.sin(grid.x)

    def rhs(u: np.ndarray, dx: float, _time: float) -> np.ndarray:
        return diffusivity * central_second(u, dx)

    solver = ExplicitPDESolver(grid=grid, rhs=rhs, boundary=periodic, viscosity=diffusivity, cfl=0.2)
    result = solver.solve(u0, final_time)
    exact = np.exp(-diffusivity * final_time) * np.sin(grid.x)
    return result, exact


def solve_advection_diffusion(
    points: int = 128,
    velocity: float = 1.0,
    diffusivity: float = 0.01,
    final_time: float = 0.25,
):
    grid = Grid1D(0.0, 1.0, points)
    u0 = np.exp(-80.0 * (grid.x - 0.25) ** 2)

    def rhs(u: np.ndarray, dx: float, _time: float) -> np.ndarray:
        return -velocity * upwind_first(u, dx, velocity=np.full_like(u, velocity)) + diffusivity * central_second(u, dx)

    solver = ExplicitPDESolver(grid=grid, rhs=rhs, boundary=periodic, viscosity=diffusivity, cfl=0.2)
    return solver.solve(u0, final_time)


def solve_poisson(points: int = 128):
    grid = Grid1D(0.0, 1.0, points, periodic=False)
    interior = points - 2
    x = grid.x
    rhs = -(np.pi**2) * np.sin(np.pi * x[1:-1])
    diagonals = [
        np.ones(interior - 1),
        -2.0 * np.ones(interior),
        np.ones(interior - 1),
    ]
    matrix = diags(diagonals, offsets=[-1, 0, 1], format="csr") / grid.dx**2
    u = np.zeros(points)
    u[1:-1] = spsolve(matrix, rhs)
    exact = np.sin(np.pi * x)
    return grid, u, exact


def plot_heat(output: Path, points: int = 128, formats: tuple[str, ...] = ("png",)) -> list[Path]:
    set_publication_style()
    result, exact = solve_heat(points=points)
    error = result.u[-1] - exact
    fig, (ax, err_ax) = plt.subplots(1, 2, figsize=FIGSIZE, constrained_layout=True)
    ax.plot(result.x, result.u[0], color=COLORS["muted"], label="initial")
    ax.plot(result.x, result.u[-1], color=COLORS["primary"], label="numerical")
    ax.plot(result.x, exact, "--", color=COLORS["secondary"], label="exact")
    ax.set_xlabel("x")
    ax.set_ylabel("u(x,t)")
    ax.legend(frameon=False)
    err_ax.plot(result.x, error, color=COLORS["accent"])
    err_ax.set_xlabel("x")
    err_ax.set_ylabel("numerical - exact")
    err_ax.set_title(f"L2 = {l2_error(result.u[-1], exact, result.x[1] - result.x[0]):.2e}")
    fig.suptitle("Heat Equation", y=1.03)
    return save_figure(fig, output, formats)


def plot_advection_diffusion(output: Path, points: int = 128, formats: tuple[str, ...] = ("png",)) -> list[Path]:
    set_publication_style()
    result = solve_advection_diffusion(points=points)
    dx = result.x[1] - result.x[0]
    masses = mass(result.u, dx)
    fig, (ax, mass_ax) = plt.subplots(1, 2, figsize=FIGSIZE, constrained_layout=True)
    ax.plot(result.x, result.u[0], color=COLORS["muted"], label="initial")
    ax.plot(result.x, result.u[-1], color=COLORS["primary"], label="final")
    ax.set_xlabel("x")
    ax.set_ylabel("u(x,t)")
    ax.legend(frameon=False)
    mass_ax.plot(result.t, masses, color=COLORS["accent"])
    mass_ax.set_xlabel("t")
    mass_ax.set_ylabel("mass")
    mass_ax.set_title(f"drift = {abs(masses[-1] - masses[0]):.2e}")
    fig.suptitle("Advection-Diffusion Equation", y=1.03)
    return save_figure(fig, output, formats)


def plot_poisson(output: Path, points: int = 128, formats: tuple[str, ...] = ("png",)) -> list[Path]:
    set_publication_style()
    grid, numerical, exact = solve_poisson(points)
    fig, (ax, err_ax) = plt.subplots(1, 2, figsize=FIGSIZE, constrained_layout=True)
    ax.plot(grid.x, numerical, color=COLORS["primary"], label="numerical")
    ax.plot(grid.x, exact, "--", color=COLORS["secondary"], label="exact")
    ax.set_xlabel("x")
    ax.set_ylabel("u(x)")
    ax.legend(frameon=False)
    err_ax.plot(grid.x, numerical - exact, color=COLORS["accent"])
    err_ax.set_xlabel("x")
    err_ax.set_ylabel("numerical - exact")
    err_ax.set_title(f"L2 = {l2_error(numerical, exact, grid.dx):.2e}")
    fig.suptitle("Poisson Equation", y=1.03)
    return save_figure(fig, output, formats)


def run_case(name: str, output_dir: Path = Path("outputs"), formats: tuple[str, ...] = ("png",)) -> list[Path]:
    if name == "heat":
        result, exact = solve_heat()
        paths = plot_heat(output_dir / "heat", formats=formats)
        print(f"L2 error: {l2_error(result.u[-1], exact, result.x[1] - result.x[0]):.3e}")
        return paths
    if name == "advection_diffusion":
        result = solve_advection_diffusion()
        paths = plot_advection_diffusion(output_dir / "advection_diffusion", formats=formats)
        print(f"mass drift: {abs(mass(result.u[-1], result.x[1] - result.x[0]) - mass(result.u[0], result.x[1] - result.x[0])):.3e}")
        return paths
    if name == "poisson":
        grid, numerical, exact = solve_poisson()
        paths = plot_poisson(output_dir / "poisson", formats=formats)
        print(f"L2 error: {l2_error(numerical, exact, grid.dx):.3e}")
        return paths
    raise ValueError(f"unknown case '{name}'")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=["heat", "poisson", "advection_diffusion"])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"], choices=["png", "pdf", "svg"])
    args = parser.parse_args()
    paths = run_case(args.case, args.output_dir, formats=tuple(args.formats))
    print("saved " + ", ".join(str(path) for path in paths))


if __name__ == "__main__":
    main()

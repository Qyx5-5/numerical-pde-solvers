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


def plot_time_series(x: np.ndarray, initial: np.ndarray, final: np.ndarray, output: Path, title: str) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 3.5))
    plt.plot(x, initial, label="initial")
    plt.plot(x, final, label="final")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()
    return output


def plot_poisson(output: Path, points: int = 128) -> Path:
    grid, numerical, exact = solve_poisson(points)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 3.5))
    plt.plot(grid.x, numerical, label="numerical")
    plt.plot(grid.x, exact, "--", label="exact")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.title("Poisson Equation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()
    return output


def run_case(name: str, output_dir: Path = Path("outputs")) -> Path:
    if name == "heat":
        result, exact = solve_heat()
        path = plot_time_series(result.x, result.u[0], result.u[-1], output_dir / "heat.png", "Heat Equation")
        print(f"L2 error: {l2_error(result.u[-1], exact, result.x[1] - result.x[0]):.3e}")
        return path
    if name == "advection_diffusion":
        result = solve_advection_diffusion()
        path = plot_time_series(
            result.x,
            result.u[0],
            result.u[-1],
            output_dir / "advection_diffusion.png",
            "Advection-Diffusion Equation",
        )
        print(f"mass drift: {abs(mass(result.u[-1], result.x[1] - result.x[0]) - mass(result.u[0], result.x[1] - result.x[0])):.3e}")
        return path
    if name == "poisson":
        grid, numerical, exact = solve_poisson()
        path = plot_poisson(output_dir / "poisson.png")
        print(f"L2 error: {l2_error(numerical, exact, grid.dx):.3e}")
        return path
    raise ValueError(f"unknown case '{name}'")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=["heat", "poisson", "advection_diffusion"])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    path = run_case(args.case, args.output_dir)
    print(f"saved {path}")


if __name__ == "__main__":
    main()

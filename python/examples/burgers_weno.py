"""Viscous Burgers equation with a WENO derivative."""

from __future__ import annotations

import numpy as np

from numerical_pde import ExplicitPDESolver, Grid1D, central_second, periodic, weno5_first


def initial_condition(x: np.ndarray) -> np.ndarray:
    return np.where(x < np.pi, 1.0, 0.0)


def make_solver(points: int = 128, viscosity: float = 0.005) -> tuple[ExplicitPDESolver, np.ndarray]:
    grid = Grid1D(0.0, 2.0 * np.pi, points)
    u0 = initial_condition(grid.x)

    def rhs(u: np.ndarray, dx: float, _time: float) -> np.ndarray:
        return -u * weno5_first(u, dx) + viscosity * central_second(u, dx)

    solver = ExplicitPDESolver(grid=grid, rhs=rhs, boundary=periodic, viscosity=viscosity, cfl=0.2)
    return solver, u0


if __name__ == "__main__":
    solver, u0 = make_solver()
    result = solver.solve(u0, final_time=0.2)
    print(f"steps: {len(result.t) - 1}")
    print(f"max/min: {result.u[-1].max():.3f}/{result.u[-1].min():.3f}")

"""Viscous Burgers equation with finite differences."""

from __future__ import annotations

import numpy as np

from numerical_pde import ExplicitPDESolver, Grid1D, central_second, mass, periodic, upwind_first


def initial_condition(x: np.ndarray) -> np.ndarray:
    return np.sin(x)


def make_solver(points: int = 128, viscosity: float = 0.01) -> tuple[ExplicitPDESolver, np.ndarray]:
    grid = Grid1D(0.0, 2.0 * np.pi, points)
    u0 = initial_condition(grid.x)

    def rhs(u: np.ndarray, dx: float, _time: float) -> np.ndarray:
        return -u * upwind_first(u, dx) + viscosity * central_second(u, dx)

    solver = ExplicitPDESolver(grid=grid, rhs=rhs, boundary=periodic, viscosity=viscosity)
    return solver, u0


if __name__ == "__main__":
    solver, u0 = make_solver()
    result = solver.solve(u0, final_time=0.5)
    print(f"steps: {len(result.t) - 1}")
    print(f"mass drift: {mass(result.u[-1], solver.grid.dx) - mass(result.u[0], solver.grid.dx):.3e}")

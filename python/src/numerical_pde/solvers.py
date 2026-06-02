from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .boundary import periodic
from .grids import Grid1D


RHS = Callable[[np.ndarray, float, float], np.ndarray]
Boundary = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class SolverResult:
    x: np.ndarray
    t: np.ndarray
    u: np.ndarray


def adaptive_dt(u: np.ndarray, dx: float, viscosity: float = 0.0, cfl: float = 0.4) -> float:
    max_speed = max(float(np.max(np.abs(u))), 1e-12)
    advective = cfl * dx / max_speed
    if viscosity <= 0.0:
        return advective
    diffusive = cfl * dx**2 / viscosity
    return min(advective, diffusive)


def rk3_step(u: np.ndarray, dt: float, rhs: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    u1 = u + dt * rhs(u)
    u2 = 0.75 * u + 0.25 * (u1 + dt * rhs(u1))
    return (u + 2.0 * (u2 + dt * rhs(u2))) / 3.0


@dataclass
class ExplicitPDESolver:
    grid: Grid1D
    rhs: RHS
    boundary: Boundary = periodic
    viscosity: float = 0.0
    cfl: float = 0.4
    max_steps: int = 10000
    method: str = "rk3"

    def solve(self, u0: np.ndarray, final_time: float) -> SolverResult:
        if u0.shape != self.grid.x.shape:
            raise ValueError("u0 shape must match grid")

        states = [self.boundary(u0.astype(float).copy())]
        times = [0.0]
        while times[-1] < final_time and len(times) <= self.max_steps:
            current = states[-1]
            dt = min(adaptive_dt(current, self.grid.dx, self.viscosity, self.cfl), final_time - times[-1])

            def step_rhs(v: np.ndarray) -> np.ndarray:
                return self.rhs(v, self.grid.dx, times[-1])

            if self.method == "euler":
                next_u = current + dt * step_rhs(current)
            elif self.method == "rk3":
                next_u = rk3_step(current, dt, step_rhs)
            else:
                raise ValueError(f"unknown method '{self.method}'")

            states.append(self.boundary(next_u))
            times.append(times[-1] + dt)

        if times[-1] < final_time:
            raise RuntimeError("maximum step count reached before final_time")
        return SolverResult(x=self.grid.x, t=np.asarray(times), u=np.vstack(states))

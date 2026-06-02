"""Compact numerical methods for PDE experiments."""

from .analysis import l2_error, mass
from .boundary import dirichlet, neumann, periodic
from .grids import Grid1D
from .operators import central_first, central_second, spectral_derivative, upwind_first, weno5_first
from .solvers import ExplicitPDESolver, SolverResult, adaptive_dt, rk3_step

__all__ = [
    "ExplicitPDESolver",
    "Grid1D",
    "SolverResult",
    "adaptive_dt",
    "central_first",
    "central_second",
    "dirichlet",
    "l2_error",
    "mass",
    "neumann",
    "periodic",
    "rk3_step",
    "spectral_derivative",
    "upwind_first",
    "weno5_first",
]

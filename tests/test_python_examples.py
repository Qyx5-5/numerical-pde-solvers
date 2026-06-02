import numpy as np

from examples.burgers_fdm import make_solver as make_fdm_solver
from examples.burgers_weno import make_solver as make_weno_solver


def test_burgers_fdm_smoke():
    solver, u0 = make_fdm_solver(points=32)
    result = solver.solve(u0, final_time=0.02)
    assert result.u.shape[1] == 32
    assert np.isclose(result.t[-1], 0.02)
    assert np.all(np.isfinite(result.u))


def test_burgers_weno_smoke():
    solver, u0 = make_weno_solver(points=32)
    result = solver.solve(u0, final_time=0.01)
    assert result.u.shape[1] == 32
    assert np.isclose(result.t[-1], 0.01)
    assert np.all(np.isfinite(result.u))


def test_burgers_fdm_short_run_keeps_mass_bounded():
    solver, u0 = make_fdm_solver(points=64)
    result = solver.solve(u0, final_time=0.05)
    dx = solver.grid.dx
    mass_drift = abs(np.sum(result.u[-1]) * dx - np.sum(result.u[0]) * dx)

    assert mass_drift < 1e-2


def test_burgers_weno_short_run_has_bounded_amplitude():
    solver, u0 = make_weno_solver(points=64)
    result = solver.solve(u0, final_time=0.02)

    assert result.u[-1].max() < 1.2
    assert result.u[-1].min() > -0.2

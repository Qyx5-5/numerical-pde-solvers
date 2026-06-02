import numpy as np

from examples.burgers_fdm import make_solver as make_fdm_solver
from examples.burgers_weno import make_solver as make_weno_solver


def test_burgers_fdm_smoke():
    solver, u0 = make_fdm_solver(points=32)
    result = solver.solve(u0, final_time=0.02)
    assert result.u.shape[1] == 32
    assert np.isclose(result.t[-1], 0.02)


def test_burgers_weno_smoke():
    solver, u0 = make_weno_solver(points=32)
    result = solver.solve(u0, final_time=0.01)
    assert result.u.shape[1] == 32
    assert np.isclose(result.t[-1], 0.01)

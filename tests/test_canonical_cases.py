from pathlib import Path

import numpy as np

from examples.canonical_cases import plot_poisson, run_case, solve_advection_diffusion, solve_heat, solve_poisson
from numerical_pde import l2_error, mass


def test_heat_case_matches_analytic_decay():
    result, exact = solve_heat(points=128, final_time=0.1)
    dx = result.x[1] - result.x[0]

    assert l2_error(result.u[-1], exact, dx) < 1e-4


def test_poisson_case_matches_analytic_solution():
    grid, numerical, exact = solve_poisson(points=128)

    assert l2_error(numerical, exact, grid.dx) < 1e-4


def test_advection_diffusion_case_stays_finite_and_mass_bounded():
    result = solve_advection_diffusion(points=96, final_time=0.1)
    dx = result.x[1] - result.x[0]
    drift = abs(mass(result.u[-1], dx) - mass(result.u[0], dx))

    assert np.all(np.isfinite(result.u))
    assert drift < 1e-2


def test_poisson_plot_writes_png(tmp_path: Path):
    outputs = plot_poisson(tmp_path / "poisson", points=64, formats=("png", "pdf"))

    assert {path.suffix for path in outputs} == {".png", ".pdf"}
    assert all(path.exists() and path.stat().st_size > 0 for path in outputs)


def test_run_case_writes_each_png(tmp_path: Path):
    for name in ["heat", "poisson", "advection_diffusion"]:
        outputs = run_case(name, tmp_path, formats=("png",))
        assert len(outputs) == 1
        assert outputs[0].exists()
        assert outputs[0].suffix == ".png"

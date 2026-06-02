import numpy as np

from numerical_pde import (
    ExplicitPDESolver,
    Grid1D,
    central_first,
    central_second,
    dirichlet,
    l2_error,
    mass,
    neumann,
    rk3_step,
    spectral_derivative,
    upwind_first,
    weno5_first,
)


def test_grid_shape_and_spacing():
    grid = Grid1D(0.0, 1.0, 10)
    assert grid.x.shape == (10,)
    assert np.isclose(grid.dx, 0.1)


def test_finite_difference_shapes():
    grid = Grid1D(0.0, 2.0 * np.pi, 32)
    u = np.sin(grid.x)
    assert central_first(u, grid.dx).shape == u.shape
    assert central_second(u, grid.dx).shape == u.shape


def test_central_first_is_second_order_convergent_for_sine():
    errors = []
    for n in (64, 128, 256):
        grid = Grid1D(0.0, 2.0 * np.pi, n)
        errors.append(l2_error(central_first(np.sin(grid.x), grid.dx), np.cos(grid.x), grid.dx))

    assert errors[0] / errors[1] > 3.8
    assert errors[1] / errors[2] > 3.8


def test_central_second_is_second_order_convergent_for_sine():
    errors = []
    for n in (64, 128, 256):
        grid = Grid1D(0.0, 2.0 * np.pi, n)
        errors.append(l2_error(central_second(np.sin(grid.x), grid.dx), -np.sin(grid.x), grid.dx))

    assert errors[0] / errors[1] > 3.8
    assert errors[1] / errors[2] > 3.8


def test_spectral_derivative_of_sine():
    grid = Grid1D(0.0, 2.0 * np.pi, 64)
    du = spectral_derivative(np.sin(grid.x), 2.0 * np.pi)
    assert np.max(np.abs(du - np.cos(grid.x))) < 1e-10


def test_spectral_second_derivative_of_sine():
    grid = Grid1D(0.0, 2.0 * np.pi, 64)
    d2u = spectral_derivative(np.sin(grid.x), 2.0 * np.pi, order=2)
    assert np.max(np.abs(d2u + np.sin(grid.x))) < 1e-10


def test_upwind_first_uses_velocity_direction():
    u = np.arange(8, dtype=float)
    dx = 0.5
    positive = upwind_first(u, dx, velocity=np.ones_like(u))
    negative = upwind_first(u, dx, velocity=-np.ones_like(u))

    assert np.allclose(positive[1:], 2.0)
    assert np.allclose(negative[:-1], 2.0)


def test_weno_derivative_is_reasonable_for_smooth_periodic_data():
    grid = Grid1D(0.0, 2.0 * np.pi, 128)
    error = l2_error(weno5_first(np.sin(grid.x), grid.dx), np.cos(grid.x), grid.dx)
    assert error < 0.2


def test_boundary_conditions_return_modified_copies():
    u = np.arange(5, dtype=float)

    fixed = dirichlet(u, left=-1.0, right=2.0)
    derivative = neumann(u, dx=0.5, left=4.0, right=-2.0)

    assert np.array_equal(u, np.arange(5, dtype=float))
    assert fixed[0] == -1.0 and fixed[-1] == 2.0
    assert derivative[0] == -1.0 and derivative[-1] == 2.0


def test_mass_vectorized():
    values = np.ones((3, 8))
    assert np.allclose(mass(values, 0.5), 4.0)


def test_rk3_step_integrates_linear_decay_accurately():
    u = np.array([1.0])
    dt = 0.01
    for _ in range(100):
        u = rk3_step(u, dt, lambda value: -value)

    assert np.allclose(u[0], np.exp(-1.0), atol=1e-7)


def test_solver_matches_heat_equation_sine_decay():
    nu = 0.02
    final_time = 0.1
    grid = Grid1D(0.0, 2.0 * np.pi, 128)
    u0 = np.sin(grid.x)

    def rhs(u, dx, _time):
        return nu * central_second(u, dx)

    result = ExplicitPDESolver(grid=grid, rhs=rhs, viscosity=nu, cfl=0.2).solve(u0, final_time)
    exact = np.exp(-nu * final_time) * np.sin(grid.x)

    assert l2_error(result.u[-1], exact, grid.dx) < 1e-4

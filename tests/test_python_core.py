import numpy as np

from numerical_pde import Grid1D, central_first, central_second, mass, spectral_derivative


def test_grid_shape_and_spacing():
    grid = Grid1D(0.0, 1.0, 10)
    assert grid.x.shape == (10,)
    assert np.isclose(grid.dx, 0.1)


def test_finite_difference_shapes():
    grid = Grid1D(0.0, 2.0 * np.pi, 32)
    u = np.sin(grid.x)
    assert central_first(u, grid.dx).shape == u.shape
    assert central_second(u, grid.dx).shape == u.shape


def test_spectral_derivative_of_sine():
    grid = Grid1D(0.0, 2.0 * np.pi, 64)
    du = spectral_derivative(np.sin(grid.x), 2.0 * np.pi)
    assert np.max(np.abs(du - np.cos(grid.x))) < 1e-10


def test_mass_vectorized():
    values = np.ones((3, 8))
    assert np.allclose(mass(values, 0.5), 4.0)

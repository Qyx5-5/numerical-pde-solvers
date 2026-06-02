# Numerical PDE Solvers

A compact collection of reusable numerical methods for PDE experiments in Python and MATLAB.

## Features

- Generic 1D grid, boundary, operator, time-stepping, and analysis utilities
- Finite-difference, WENO-style, and spectral derivative operators
- Burgers equation examples kept outside the core package
- MATLAB split-step spectral routines with a small Gross-Pitaevskii example

## Install

```bash
cd python
pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np
from numerical_pde import ExplicitPDESolver, Grid1D, central_second, upwind_first

grid = Grid1D(0.0, 2.0 * np.pi, 128)
nu = 0.01
u0 = np.sin(grid.x)

def rhs(u, dx, time):
    return -u * upwind_first(u, dx) + nu * central_second(u, dx)

solver = ExplicitPDESolver(grid=grid, rhs=rhs, viscosity=nu)
result = solver.solve(u0, final_time=0.5)
```

## Layout

```text
python/src/numerical_pde/   reusable Python methods
python/examples/            concrete PDE cases
matlab/tssp/                generic split-step spectral routines
matlab/examples/gpe/        compact Gross-Pitaevskii example
tests/                      Python smoke and correctness tests
```

## Examples

```bash
cd python
python examples/burgers_fdm.py
python examples/burgers_weno.py
```

For MATLAB:

```matlab
cd matlab/examples/gpe
run_gpe_example
```

## Tests

```bash
$env:PYTHONPATH = "python/src;python"
pytest
```

## Adding A PDE

Put reusable discretization mechanics in `python/src/numerical_pde` or `matlab/tssp`. Keep equation setup, parameters, initial conditions, and plotting in examples.

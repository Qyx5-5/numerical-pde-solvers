# Numerical PDE Solvers

A compact Python collection of reusable numerical methods for PDE experiments.

## Features

- Generic 1D grid, boundary, operator, time-stepping, and analysis utilities
- Finite-difference, WENO-style, and spectral derivative operators
- Burgers equation examples kept outside the core package

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
python/src/numerical_pde/   reusable methods
python/examples/            concrete PDE cases
tests/                      smoke and correctness tests
```

## Canonical Figures

These cases show what each numerical method should preserve or approximate.

### Heat: diffusion with exact decay

![Heat equation case](docs/figures/heat.png)

The case solves $u_t=\nu u_{xx}$ with $\nu=0.05$ on periodic $x\in[0,2\pi]$ and $u(x,0)=\sin x$. The exact solution is $u(x,t)=e^{-\nu t}\sin x$.

### Poisson: boundary-value problem

![Poisson equation case](docs/figures/poisson.png)

The case solves $u_{xx}=-\pi^2\sin(\pi x)$ on $x\in[0,1]$ with $u(0)=u(1)=0$. The exact solution is $u(x)=\sin(\pi x)$, and the right panel shows numerical error.

### Advection-Diffusion: transport plus smoothing

![Advection-diffusion case](docs/figures/advection_diffusion.png)

The case solves $u_t+u_x=\nu u_{xx}$ with $\nu=0.01$ on periodic $x\in[0,1]$. The pulse moves and broadens; the mass panel checks conservation of $\int u(x,t)\,dx$.

## Run Examples

```bash
cd python
python examples/burgers_fdm.py
python examples/burgers_weno.py
python examples/canonical_cases.py heat
python examples/canonical_cases.py poisson
python examples/canonical_cases.py advection_diffusion
```

Generated figures are saved under `outputs/`, which is ignored by Git. Use `--formats png pdf svg` to choose formats.

## Tests

```bash
pytest
```

## Adding A PDE

Put reusable discretization mechanics in `python/src/numerical_pde`. Keep equation setup, parameters, initial conditions, and plotting in examples.

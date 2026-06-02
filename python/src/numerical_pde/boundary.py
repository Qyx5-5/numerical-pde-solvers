from __future__ import annotations

import numpy as np


def periodic(u: np.ndarray) -> np.ndarray:
    return u


def dirichlet(u: np.ndarray, left: float = 0.0, right: float = 0.0) -> np.ndarray:
    v = u.copy()
    v[0] = left
    v[-1] = right
    return v


def neumann(u: np.ndarray, dx: float, left: float = 0.0, right: float = 0.0) -> np.ndarray:
    v = u.copy()
    v[0] = v[1] - left * dx
    v[-1] = v[-2] + right * dx
    return v

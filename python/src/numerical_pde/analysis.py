from __future__ import annotations

import numpy as np


def mass(u: np.ndarray, dx: float) -> np.ndarray:
    return np.sum(u, axis=-1) * dx


def l2_error(numerical: np.ndarray, reference: np.ndarray, dx: float) -> float:
    return float(np.sqrt(np.sum((numerical - reference) ** 2) * dx))

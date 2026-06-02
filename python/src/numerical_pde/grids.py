from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Grid1D:
    start: float
    stop: float
    points: int
    periodic: bool = True

    @property
    def x(self) -> np.ndarray:
        return np.linspace(self.start, self.stop, self.points, endpoint=not self.periodic)

    @property
    def dx(self) -> float:
        return (self.stop - self.start) / self.points if self.periodic else (self.stop - self.start) / (self.points - 1)

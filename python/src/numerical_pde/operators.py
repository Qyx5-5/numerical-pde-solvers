from __future__ import annotations

import numpy as np
from numpy.fft import fft, fftfreq, ifft


def central_first(u: np.ndarray, dx: float) -> np.ndarray:
    return (np.roll(u, -1) - np.roll(u, 1)) / (2.0 * dx)


def central_second(u: np.ndarray, dx: float) -> np.ndarray:
    return (np.roll(u, -1) - 2.0 * u + np.roll(u, 1)) / dx**2


def upwind_first(u: np.ndarray, dx: float, velocity: np.ndarray | None = None) -> np.ndarray:
    v = u if velocity is None else velocity
    backward = (u - np.roll(u, 1)) / dx
    forward = (np.roll(u, -1) - u) / dx
    return np.where(v >= 0.0, backward, forward)


def _weno5(stencil: np.ndarray) -> float:
    eps = 1e-6
    v1, v2, v3, v4, v5 = stencil
    beta1 = 13 / 12 * (v1 - 2 * v2 + v3) ** 2 + 0.25 * (v1 - 4 * v2 + 3 * v3) ** 2
    beta2 = 13 / 12 * (v2 - 2 * v3 + v4) ** 2 + 0.25 * (v2 - v4) ** 2
    beta3 = 13 / 12 * (v3 - 2 * v4 + v5) ** 2 + 0.25 * (3 * v3 - 4 * v4 + v5) ** 2
    gamma = np.array([0.1, 0.6, 0.3])
    alpha = gamma / (eps + np.array([beta1, beta2, beta3])) ** 2
    omega = alpha / np.sum(alpha)
    q1 = v1 / 3 - 7 * v2 / 6 + 11 * v3 / 6
    q2 = -v2 / 6 + 5 * v3 / 6 + v4 / 3
    q3 = v3 / 3 + 5 * v4 / 6 - v5 / 6
    return float(omega @ np.array([q1, q2, q3]))


def weno5_first(u: np.ndarray, dx: float) -> np.ndarray:
    padded = np.pad(u, 3, mode="wrap")
    derivative = np.zeros_like(u, dtype=float)
    for i in range(len(u)):
        left = _weno5(padded[i : i + 5])
        right = _weno5(padded[i + 1 : i + 6])
        derivative[i] = (right - left) / dx
    return derivative


def spectral_derivative(u: np.ndarray, length: float, order: int = 1) -> np.ndarray:
    if order < 1:
        raise ValueError("order must be positive")
    k = 2.0 * np.pi * fftfreq(len(u), d=length / len(u))
    return ifft((1j * k) ** order * fft(u)).real

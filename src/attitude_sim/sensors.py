"""Noisy body-rate and vector-observation sensor models."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from attitude_sim.quaternions import quat_to_rotation


def _as_rng(seed: int | np.random.Generator | None) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


@dataclass
class GyroModel:
    """Rate gyro: ``ω_m = ω + b + η_v``, ``ḃ = η_u``.

    ``sigma_v`` is white-noise density on the rate measurement
    (rad/s/√Hz) and ``sigma_u`` is bias random-walk density
    (rad/s²/√Hz).  Over a sample ``dt``, discrete noise std is
    ``sigma_v / √dt`` on the measurement and ``sigma_u √dt`` on the bias.
    """

    sigma_v: float = 5e-4
    sigma_u: float = 1e-6
    bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    seed: int | np.random.Generator | None = None

    def __post_init__(self) -> None:
        self.bias = np.asarray(self.bias, dtype=float).reshape(3).copy()
        self._rng = _as_rng(self.seed)

    def measure(self, omega: np.ndarray, dt: float) -> np.ndarray:
        omega = np.asarray(omega, dtype=float).reshape(3)
        self.bias = self.bias + self.sigma_u * np.sqrt(dt) * self._rng.standard_normal(3)
        noise = (self.sigma_v / np.sqrt(dt)) * self._rng.standard_normal(3)
        return omega + self.bias + noise


@dataclass
class VectorSensor:
    """Unit-vector observation in the body frame (mag, sun, etc.).

    ``v_b = R(q)ᵀ v_I + η``, then re-normalized.  ``sigma`` is the
    per-axis Cartesian noise before normalization.
    """

    v_inertial: np.ndarray
    sigma: float = 2e-3
    seed: int | np.random.Generator | None = None
    name: str = "vector"

    def __post_init__(self) -> None:
        v = np.asarray(self.v_inertial, dtype=float).reshape(3)
        n = np.linalg.norm(v)
        if n < 1e-15:
            raise ValueError(f"{self.name} inertial reference must be non-zero")
        self.v_inertial = v / n
        self._rng = _as_rng(self.seed)

    def measure(self, q: np.ndarray) -> np.ndarray:
        R = quat_to_rotation(q)
        v_b = R.T @ self.v_inertial
        v_b = v_b + self.sigma * self._rng.standard_normal(3)
        n = np.linalg.norm(v_b)
        if n < 1e-15:
            return np.array([1.0, 0.0, 0.0])
        return v_b / n

"""Attitude / rate estimators: Mahony complementary filter and MEKF.

Both consume a noisy gyro and optional unit-vector measurements
(magnetometer, sun sensor) and return ``(q̂, ω̂)`` for the controller.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from attitude_sim.quaternions import (
    quat_integrate_const_omega,
    quat_multiply,
    quat_normalize,
    quat_to_rotation,
    skew,
)
from attitude_sim.sensors import VectorSensor


@dataclass
class ComplementaryFilter:
    """SO(3) Mahony complementary filter with gyro-bias estimation.

    Nominal kinematics are integrated with ``ω̂ = ω_m − b̂ + k_p ω_mes``.
    Vector innovations ``ω_mes = Σ v̂_b × v_b`` pull the estimate toward
    the TRIAD-like observations; bias is updated with ``ḃ̂ = −k_i ω_mes``.
    """

    kp: float = 1.5
    ki: float = 0.08
    q: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    bias: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self) -> None:
        self.q = quat_normalize(self.q)
        self.bias = np.asarray(self.bias, dtype=float).reshape(3).copy()

    def reset(self, q: np.ndarray | None = None, bias: np.ndarray | None = None) -> None:
        if q is not None:
            self.q = quat_normalize(q)
        if bias is not None:
            self.bias = np.asarray(bias, dtype=float).reshape(3).copy()

    def step(
        self,
        omega_m: np.ndarray,
        dt: float,
        vector_meas: list[tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        omega_m = np.asarray(omega_m, dtype=float).reshape(3)
        omega_corr = np.zeros(3)
        if vector_meas:
            R = quat_to_rotation(self.q)
            for v_b, v_I in vector_meas:
                v_hat = R.T @ (v_I / np.linalg.norm(v_I))
                v_b = v_b / np.linalg.norm(v_b)
                # v_meas × v̂ ≈ δθ_perp for q = q̂ ⊗ δq(δθ); add to ω̂ so q̂ catches up.
                omega_corr = omega_corr + np.cross(v_b, v_hat)
            omega_corr /= len(vector_meas)
        omega_hat = omega_m - self.bias + self.kp * omega_corr
        self.bias = self.bias - self.ki * omega_corr * dt
        self.q = quat_integrate_const_omega(self.q, omega_hat, dt)
        return self.q.copy(), omega_hat.copy()


@dataclass
class MultiplicativeEKF:
    """6-state multiplicative EKF: attitude error ``δα`` and gyro bias.

    Error definition ``q = q̂ ⊗ δq(δα)`` with ``δq ≈ [1, δα/2]``.
    Gyro model ``ω_m = ω + b + η_v``, ``ḃ = η_u``.  Vector updates use
    ``H = [[v̂_b ×], 0]``.
    """

    sigma_v: float = 5e-4
    sigma_u: float = 1e-6
    q: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    P: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.q = quat_normalize(self.q)
        self.bias = np.asarray(self.bias, dtype=float).reshape(3).copy()
        if self.P is None:
            self.P = np.diag([5e-4, 5e-4, 5e-4, 1e-6, 1e-6, 1e-6])
        else:
            self.P = np.asarray(self.P, dtype=float).reshape(6, 6)

    def reset(self, q: np.ndarray | None = None, bias: np.ndarray | None = None) -> None:
        if q is not None:
            self.q = quat_normalize(q)
        if bias is not None:
            self.bias = np.asarray(bias, dtype=float).reshape(3).copy()

    def predict(self, omega_m: np.ndarray, dt: float) -> np.ndarray:
        omega_m = np.asarray(omega_m, dtype=float).reshape(3)
        omega_hat = omega_m - self.bias
        self.q = quat_integrate_const_omega(self.q, omega_hat, dt)

        F = np.zeros((6, 6))
        F[0:3, 0:3] = -skew(omega_hat)
        F[0:3, 3:6] = -np.eye(3)
        Phi = np.eye(6) + F * dt + 0.5 * (F @ F) * (dt**2)

        G = np.zeros((6, 6))
        G[0:3, 0:3] = -np.eye(3)
        G[3:6, 3:6] = np.eye(3)
        Qc = np.diag([self.sigma_v**2] * 3 + [self.sigma_u**2] * 3)
        Qd = G @ Qc @ G.T * dt
        self.P = Phi @ self.P @ Phi.T + Qd
        self.P = 0.5 * (self.P + self.P.T)
        return omega_hat

    def update_vector(self, v_b_meas: np.ndarray, v_inertial: np.ndarray, sigma: float) -> None:
        v_I = np.asarray(v_inertial, dtype=float).reshape(3)
        v_I = v_I / np.linalg.norm(v_I)
        v_b_meas = np.asarray(v_b_meas, dtype=float).reshape(3)
        v_b_meas = v_b_meas / np.linalg.norm(v_b_meas)
        v_hat = quat_to_rotation(self.q).T @ v_I
        v_hat = v_hat / np.linalg.norm(v_hat)

        H = np.zeros((3, 6))
        H[0:3, 0:3] = skew(v_hat)
        R = (sigma**2) * np.eye(3)
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.solve(S, np.eye(3))
        dx = K @ (v_b_meas - v_hat)
        I_KH = np.eye(6) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        self._inject(dx)

    def _inject(self, dx: np.ndarray) -> None:
        dtheta = dx[:3]
        half = 0.5 * dtheta
        n2 = float(half @ half)
        w = np.sqrt(max(0.0, 1.0 - n2)) if n2 <= 1.0 else 0.0
        dq = quat_normalize(np.array([w, half[0], half[1], half[2]]))
        self.q = quat_normalize(quat_multiply(self.q, dq))
        self.bias = self.bias + dx[3:6]

    def step(
        self,
        omega_m: np.ndarray,
        dt: float,
        vector_meas: list[tuple[np.ndarray, np.ndarray]] | None = None,
        vector_sigma: float = 2e-3,
    ) -> tuple[np.ndarray, np.ndarray]:
        omega_hat = self.predict(omega_m, dt)
        if vector_meas:
            for v_b, v_I in vector_meas:
                self.update_vector(v_b, v_I, vector_sigma)
            omega_hat = omega_m - self.bias
        return self.q.copy(), np.asarray(omega_hat, dtype=float).reshape(3).copy()


def make_estimator(
    mode: str,
    q0: np.ndarray | None = None,
    **kwargs,
) -> ComplementaryFilter | MultiplicativeEKF | None:
    mode = mode.lower()
    if mode == "truth":
        return None
    q0 = np.array([1.0, 0.0, 0.0, 0.0]) if q0 is None else q0
    if mode in {"mahony", "complementary", "cf"}:
        return ComplementaryFilter(q=q0, **kwargs)
    if mode in {"mekf", "kalman", "ekf"}:
        return MultiplicativeEKF(q=q0, **kwargs)
    raise ValueError(f"unknown estimator {mode!r}; use 'truth', 'mekf', or 'mahony'")


def vectors_from_sensors(
    q: np.ndarray,
    sensors: list[VectorSensor],
) -> list[tuple[np.ndarray, np.ndarray]]:
    return [(s.measure(q), s.v_inertial) for s in sensors]

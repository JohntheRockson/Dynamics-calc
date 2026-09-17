"""Attitude controllers: quaternion-error PID and linearized LQR.

Both laws use the shortest-path multiplicative error

    q_e = q_des* ⊗ q̂ ,   δθ ≈ 2 sign(q_e0) q_e[1:4]

and body-rate feedback.  The controller may be fed either true plant
state or filter estimates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import linalg

from attitude_sim.quaternions import attitude_error_vector, rotation_vector_error


def _as_pd_gain(value: np.ndarray | float, inertia: np.ndarray) -> np.ndarray:
    """Broadcast a scalar / 3-vector / 3x3 into a 3x3 body-frame gain."""
    J = np.asarray(inertia, dtype=float).reshape(3, 3)
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return float(arr) * np.eye(3)
    if arr.shape == (3,):
        return np.diag(arr)
    if arr.shape == (3, 3):
        return arr
    raise ValueError(f"gain must be scalar, 3-vector, or 3x3; got shape {arr.shape}")


@dataclass
class PIDAttitudeController:
    """PID on quaternion vector error + body rate, with anti-windup.

    ``τ = −Kp e_q − Kd (ω − ω_des) − Ki z`` plus optional gyroscopic
    cancellation ``ω × Jω``.  Gains default to inertia-scaled PD for a
    target natural frequency and damping.
    """

    inertia: np.ndarray
    kp: np.ndarray | float | None = None
    kd: np.ndarray | float | None = None
    ki: np.ndarray | float | None = None
    wn: float = 0.45
    zeta: float = 1.0
    integral_limit: float = 0.2
    torque_limit: float | None = 0.02
    gyroscopic_cancel: bool = True
    _z: np.ndarray = field(default_factory=lambda: np.zeros(3), init=False, repr=False)

    def __post_init__(self) -> None:
        J = np.asarray(self.inertia, dtype=float).reshape(3, 3)
        self.inertia = 0.5 * (J + J.T)
        # q_e ≈ θ/2 so Kp q_e ≈ (wn² J) θ  ⇒  Kp = 2 wn² J
        if self.kp is None:
            self.kp = 2.0 * (self.wn**2) * self.inertia
        if self.kd is None:
            self.kd = 2.0 * self.zeta * self.wn * self.inertia
        if self.ki is None:
            self.ki = 0.08 * (self.wn**3) * self.inertia
        self.kp = _as_pd_gain(self.kp, self.inertia)
        self.kd = _as_pd_gain(self.kd, self.inertia)
        self.ki = _as_pd_gain(self.ki, self.inertia)

    def reset(self) -> None:
        self._z[:] = 0.0

    def command(
        self,
        q: np.ndarray,
        omega: np.ndarray,
        q_des: np.ndarray,
        omega_des: np.ndarray | None = None,
        dt: float = 0.01,
    ) -> np.ndarray:
        omega = np.asarray(omega, dtype=float).reshape(3)
        omega_des = np.zeros(3) if omega_des is None else np.asarray(omega_des, dtype=float).reshape(3)
        e_q = attitude_error_vector(q, q_des)
        e_w = omega - omega_des
        z_next = self._z + e_q * dt
        max_z = self.integral_limit
        n = np.linalg.norm(z_next)
        if n > max_z > 0.0:
            z_next *= max_z / n
        tau = -self.kp @ e_q - self.kd @ e_w - self.ki @ z_next
        if self.gyroscopic_cancel:
            tau = tau + np.cross(omega, self.inertia @ omega)
        tau = _saturate(tau, self.torque_limit)
        # Anti-windup: only integrate when unsaturated (or error is helping)
        if self.torque_limit is None or np.linalg.norm(tau) < self.torque_limit * (1.0 - 1e-9):
            self._z = z_next
        return tau


@dataclass
class LQRAttitudeController:
    """Continuous LQR on the linearized multiplicative-error model.

    About rest (``ω = 0``, ``q = q_des``) the error state ``x = [δθ, ω]``
    obeys

        δθ̇ = ω ,   ω̇ = J⁻¹ τ

    so ``A = [[0, I], [0, 0]]``, ``B = [[0], [J⁻¹]]``.  The CARE is
    solved once; the online law is ``τ = −K [δθ; ω]`` with optional
    gyroscopic cancellation and torque saturation.
    """

    inertia: np.ndarray
    q_att: float = 6.0
    q_rate: float = 0.8
    r_torque: float = 8.0
    torque_limit: float | None = 0.02
    gyroscopic_cancel: bool = True
    K: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        J = np.asarray(self.inertia, dtype=float).reshape(3, 3)
        self.inertia = 0.5 * (J + J.T)
        A, B = linearize_attitude(self.inertia)
        Q = np.diag([self.q_att] * 3 + [self.q_rate] * 3)
        R = self.r_torque * np.eye(3)
        P = linalg.solve_continuous_are(A, B, Q, R)
        self.K = np.linalg.solve(R, B.T @ P)

    def reset(self) -> None:
        return None

    def command(
        self,
        q: np.ndarray,
        omega: np.ndarray,
        q_des: np.ndarray,
        omega_des: np.ndarray | None = None,
        dt: float = 0.01,
    ) -> np.ndarray:
        del dt
        omega = np.asarray(omega, dtype=float).reshape(3)
        omega_des = np.zeros(3) if omega_des is None else np.asarray(omega_des, dtype=float).reshape(3)
        dtheta = rotation_vector_error(q, q_des)
        x = np.concatenate([dtheta, omega - omega_des])
        tau = -self.K @ x
        if self.gyroscopic_cancel:
            tau = tau + np.cross(omega, self.inertia @ omega)
        return _saturate(tau, self.torque_limit)


def linearize_attitude(inertia: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(A, B)`` for ``x = [δθ, ω]``, ``u = τ`` about rest."""
    Jinv = np.linalg.inv(np.asarray(inertia, dtype=float).reshape(3, 3))
    A = np.zeros((6, 6))
    A[0:3, 3:6] = np.eye(3)
    B = np.zeros((6, 3))
    B[3:6, :] = Jinv
    return A, B


def _saturate(tau: np.ndarray, limit: float | None) -> np.ndarray:
    if limit is None or limit <= 0.0:
        return tau
    n = np.linalg.norm(tau)
    if n > limit:
        return tau * (limit / n)
    return tau


def make_controller(mode: str, inertia: np.ndarray, **kwargs) -> PIDAttitudeController | LQRAttitudeController:
    mode = mode.lower()
    if mode == "pid":
        return PIDAttitudeController(inertia=inertia, **kwargs)
    if mode == "lqr":
        return LQRAttitudeController(inertia=inertia, **kwargs)
    raise ValueError(f"unknown controller mode {mode!r}; use 'pid' or 'lqr'")

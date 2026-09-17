"""Rigid-body rotational plant: quaternion kinematics + Euler's equation.

State is ``x = [q(4), ω(3)]``.  With body inertia ``J`` and external
control torque ``τ`` (both in the body frame),

    q̇ = (1/2) q ⊗ ω
    J ω̇ = τ − ω × (J ω)

Time stepping uses classical RK4 with quaternion renormalization after
each step.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from attitude_sim.quaternions import quat_derivative, quat_normalize, quat_to_rotation


def rk4_step(
    fun: Callable[[float, np.ndarray], np.ndarray],
    t: float,
    y: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Classical fourth-order Runge–Kutta step."""
    y = np.asarray(y, dtype=float)
    k1 = np.asarray(fun(t, y), dtype=float)
    k2 = np.asarray(fun(t + 0.5 * dt, y + 0.5 * dt * k1), dtype=float)
    k3 = np.asarray(fun(t + 0.5 * dt, y + 0.5 * dt * k2), dtype=float)
    k4 = np.asarray(fun(t + dt, y + dt * k3), dtype=float)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


@dataclass
class RigidBody:
    """Torque-driven rigid body with a constant body-frame inertia matrix."""

    inertia: np.ndarray

    def __post_init__(self) -> None:
        J = np.asarray(self.inertia, dtype=float).reshape(3, 3)
        if J.shape != (3, 3):
            raise ValueError("inertia must be 3x3")
        if abs(J - J.T).max() > 1e-12:
            raise ValueError("inertia must be symmetric")
        evals = np.linalg.eigvalsh(J)
        if np.any(evals <= 0.0):
            raise ValueError("inertia must be positive definite")
        self.inertia = 0.5 * (J + J.T)

    def omega_dot(self, omega: np.ndarray, tau: np.ndarray) -> np.ndarray:
        """Euler's rotational equation resolved in the body frame."""
        omega = np.asarray(omega, dtype=float).reshape(3)
        tau = np.asarray(tau, dtype=float).reshape(3)
        h = self.inertia @ omega
        return np.linalg.solve(self.inertia, tau - np.cross(omega, h))

    def derivatives(self, q: np.ndarray, omega: np.ndarray, tau: np.ndarray) -> np.ndarray:
        qdot = quat_derivative(q, omega)
        wdot = self.omega_dot(omega, tau)
        return np.concatenate([qdot, wdot])

    def kinetic_energy(self, omega: np.ndarray) -> float:
        omega = np.asarray(omega, dtype=float).reshape(3)
        return 0.5 * float(omega @ (self.inertia @ omega))

    def angular_momentum_body(self, omega: np.ndarray) -> np.ndarray:
        return self.inertia @ np.asarray(omega, dtype=float).reshape(3)

    def angular_momentum_inertial(self, q: np.ndarray, omega: np.ndarray) -> np.ndarray:
        """Inertial-frame angular momentum ``h_I = R(q) J ω``."""
        return quat_to_rotation(q) @ self.angular_momentum_body(omega)


def step_rigid_body(
    body: RigidBody,
    q: np.ndarray,
    omega: np.ndarray,
    tau: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance the plant one RK4 step with zero-order-hold torque."""

    tau = np.asarray(tau, dtype=float).reshape(3)

    def fun(_t: float, y: np.ndarray) -> np.ndarray:
        return body.derivatives(y[:4], y[4:], tau)

    y = rk4_step(fun, 0.0, np.concatenate([q, omega]), dt)
    q_next = quat_normalize(y[:4])
    omega_next = y[4:].copy()
    return q_next, omega_next

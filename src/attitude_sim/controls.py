"""Attitude controllers: quaternion-error PID and linearized LQR.

Both laws use the shortest-path multiplicative error

    q_e = q_des* ⊗ q̂ ,   δθ ≈ 2 sign(q_e0) q_e[1:4]

and body-rate feedback.  The controller may be fed either true plant
state or filter estimates.

Default gains are sized to the M1 smallsat-class plant
``J ≈ diag(0.05, 0.06, 0.07) kg·m²`` and actuator ``|τ| ≤ 0.02 N·m``
(see ``docs/controls.md``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import linalg

from attitude_sim.quaternions import attitude_error_vector, rotation_vector_error

# --- plant-scale defaults (M1 smallsat + 20 mN·m wheels) -------------------
# Opening 75° PD torque is ~τ_max; 2% settling is a few tens of seconds.
DEFAULT_TORQUE_LIMIT = 0.02
PID_WN = 0.5
PID_ZETA = 1.0
# Ki = coeff · wn³ J  →  PI zero near wn/4; slow mode still inside a 30 s hold.
PID_KI_WN_COEFF = 0.5
# ∫e_q dt clamp: Ki · limit ≳ 0.3 τ_max so a 2–5 mN·m bias can be held.
PID_INTEGRAL_LIMIT = 3.0
# Only integrate when ||e_q|| is small (~11° geodesic) so the slew does not wind up.
PID_INTEGRAL_GATE = 0.10

# Bryson references for Q, R.  K_θ ≈ τ_ref / θ_ref puts the linear region
# around 14°, not a 1° bang-bang, while wn_LQR ≈ 1.1 rad/s stays near PID.
LQR_THETA_REF = 0.25  # rad
LQR_OMEGA_REF = 0.20  # rad/s
LQR_TAU_REF = DEFAULT_TORQUE_LIMIT  # N·m


def _as_pd_gain(value: np.ndarray | float, inertia: np.ndarray) -> np.ndarray:
    """Broadcast a scalar / 3-vector / 3x3 into a 3x3 body-frame gain."""
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return float(arr) * np.eye(3)
    if arr.shape == (3,):
        return np.diag(arr)
    if arr.shape == (3, 3):
        return arr
    raise ValueError(f"gain must be scalar, 3-vector, or 3x3; got shape {arr.shape}")


def pid_gains_from_wn(
    inertia: np.ndarray,
    wn: float = PID_WN,
    zeta: float = PID_ZETA,
    ki_wn_coeff: float = PID_KI_WN_COEFF,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inertia-scaled PID matrices for quaternion-vector error ``e_q ≈ θ/2``.

    Matching a rotation-vector PD ``τ = −wn² J θ − 2 ζ wn J ω`` requires

        Kp = 2 wn² J ,   Kd = 2 ζ wn J ,   Ki = ki_wn_coeff · wn³ J.
    """
    J = np.asarray(inertia, dtype=float).reshape(3, 3)
    kp = 2.0 * (wn**2) * J
    kd = 2.0 * zeta * wn * J
    ki = ki_wn_coeff * (wn**3) * J
    return kp, kd, ki


def bryson_lqr_weights(
    theta_ref: float = LQR_THETA_REF,
    omega_ref: float = LQR_OMEGA_REF,
    tau_ref: float = LQR_TAU_REF,
) -> tuple[float, float, float]:
    """Return ``(q_att, q_rate, r_torque)`` from Bryson reference magnitudes."""
    return 1.0 / (theta_ref**2), 1.0 / (omega_ref**2), 1.0 / (tau_ref**2)


@dataclass
class PIDAttitudeController:
    """PID on quaternion vector error + body rate, with anti-windup.

    ``τ = −Kp e_q − Kd (ω − ω_des) − Ki z`` plus optional gyroscopic
    cancellation ``ω × Jω`` and optional ``|τ| ≤ τ_max``.  Gains default
    to inertia-scaled PD for a target natural frequency and damping.
    """

    inertia: np.ndarray
    kp: np.ndarray | float | None = None
    kd: np.ndarray | float | None = None
    ki: np.ndarray | float | None = None
    wn: float = PID_WN
    zeta: float = PID_ZETA
    ki_wn_coeff: float = PID_KI_WN_COEFF
    integral_limit: float = PID_INTEGRAL_LIMIT
    integral_gate: float = PID_INTEGRAL_GATE
    torque_limit: float | None = DEFAULT_TORQUE_LIMIT
    gyroscopic_cancel: bool = True
    _z: np.ndarray = field(default_factory=lambda: np.zeros(3), init=False, repr=False)

    def __post_init__(self) -> None:
        J = np.asarray(self.inertia, dtype=float).reshape(3, 3)
        self.inertia = 0.5 * (J + J.T)
        kp0, kd0, ki0 = pid_gains_from_wn(self.inertia, self.wn, self.zeta, self.ki_wn_coeff)
        if self.kp is None:
            self.kp = kp0
        if self.kd is None:
            self.kd = kd0
        if self.ki is None:
            self.ki = ki0
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
        integrate = np.linalg.norm(e_q) <= self.integral_gate
        z_next = self._z + e_q * dt if integrate else self._z.copy()
        max_z = self.integral_limit
        n = np.linalg.norm(z_next)
        if n > max_z > 0.0:
            z_next *= max_z / n
        tau = -np.asarray(self.kp) @ e_q - np.asarray(self.kd) @ e_w - np.asarray(self.ki) @ z_next
        if self.gyroscopic_cancel:
            tau = tau + np.cross(omega, self.inertia @ omega)
        tau = _saturate(tau, self.torque_limit)
        # Anti-windup: freeze the integrator while the command is saturated.
        if self.torque_limit is None or np.linalg.norm(tau) < self.torque_limit * (1.0 - 1e-9):
            self._z = z_next
        return tau


@dataclass
class LQRAttitudeController:
    """Continuous LQR on the linearized multiplicative-error model.

    About rest (``ω = 0``, ``q = q_des``) the error state ``x = [δθ, ω]``
    obeys

        δθ̇ = ω ,   ω̇ = J⁻¹ τ

    so ``A = [[0, I], [0, 0]]``, ``B = [[0], [J⁻¹]]``.  Default ``Q``, ``R``
    are Bryson placeholders; the CARE is solved once unless ``K`` is
    supplied.  The online law is ``τ = −K [δθ; ω]`` with optional
    gyroscopic cancellation and torque saturation.
    """

    inertia: np.ndarray
    q_att: float = 1.0 / (LQR_THETA_REF**2)
    q_rate: float = 1.0 / (LQR_OMEGA_REF**2)
    r_torque: float = 1.0 / (LQR_TAU_REF**2)
    torque_limit: float | None = DEFAULT_TORQUE_LIMIT
    gyroscopic_cancel: bool = True
    K: np.ndarray | None = None
    Q: np.ndarray | None = field(default=None, repr=False)
    R: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        J = np.asarray(self.inertia, dtype=float).reshape(3, 3)
        self.inertia = 0.5 * (J + J.T)
        if self.Q is None:
            self.Q = np.diag([self.q_att] * 3 + [self.q_rate] * 3)
        else:
            self.Q = np.asarray(self.Q, dtype=float)
        if self.R is None:
            self.R = self.r_torque * np.eye(3)
        else:
            self.R = np.asarray(self.R, dtype=float)
        if self.K is None:
            A, B = linearize_attitude(self.inertia)
            P = linalg.solve_continuous_are(A, B, self.Q, self.R)
            self.K = np.linalg.solve(self.R, B.T @ P)
        else:
            self.K = np.asarray(self.K, dtype=float)
            if self.K.shape != (3, 6):
                raise ValueError(f"LQR gain K must be 3x6; got {self.K.shape}")

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
        tau = -np.asarray(self.K) @ x
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

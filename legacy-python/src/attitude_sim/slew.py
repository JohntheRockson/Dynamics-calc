"""Rest-to-rest eigenaxis slew profiles (bang-coast-bang / trapezoidal).

A principal rotation from ``q0`` to ``qf`` is an eigenaxis manoeuvre: the
relative quaternion ``q0* ⊗ qf`` is a single-axis rotation ``(ê, Φ)``.
The scalar angle ``θ(t)`` is a rest-to-rest trapezoid (coast if ``ω_max``
is reached, otherwise a triangular bang-bang), and the body-frame
references are

    q_des(t) = q0 ⊗ exp(θ(t) ê),   ω_des = θ̇ ê,   α_des = θ̈ ê.

``ω_des`` / ``α_des`` plug into the existing PID / LQR
``command(..., omega_des=, alpha_des=)`` path.  Inertial ``J α_des`` is
added *before* each law's ``apply_torque_limits`` helper (PID anti-windup
sees it; LQR has no integrator).
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from attitude_sim.quaternions import (
    axis_angle_to_quat,
    geodesic_angle,
    quat_error,
    quat_multiply,
    quat_normalize,
)

_EPS = 1e-15
_ANGLE_EPS = 1e-12


class SupportsAttitudeCommand(Protocol):
    """PID / LQR ``command`` surface used by ``command_slew``."""

    inertia: np.ndarray

    def command(
        self,
        q: np.ndarray,
        omega: np.ndarray,
        q_des: np.ndarray,
        omega_des: np.ndarray | None = None,
        dt: float = 0.01,
    ) -> np.ndarray: ...


def _unit(axis: np.ndarray) -> np.ndarray:
    a = np.asarray(axis, dtype=float).reshape(3)
    n = float(np.linalg.norm(a))
    if n < _EPS:
        raise ValueError("eigenaxis must be a non-zero 3-vector")
    return a / n


def _shortest_eigenaxis(q0: np.ndarray, qf: np.ndarray) -> tuple[np.ndarray, float]:
    """Body-frame eigenaxis ``ê`` and geodesic angle ``Φ ∈ [0, π]`` from ``q0`` to ``qf``.

    ``qf = q0 ⊗ axis_angle_to_quat(ê, Φ)`` (shortest path, scalar-first).
    """
    qe = quat_error(qf, q0)
    if qe[0] < 0.0:
        qe = -qe
    vec = qe[1:].copy()
    n = float(np.linalg.norm(vec))
    angle = 2.0 * float(np.arctan2(n, float(qe[0])))
    if n < _EPS or angle < _ANGLE_EPS:
        return np.array([0.0, 0.0, 1.0]), 0.0
    return vec / n, angle


def eigenaxis_inertia(inertia: np.ndarray, axis: np.ndarray) -> float:
    """Scalar moment ``êᵀ J ê`` about a unit eigenaxis."""
    J = np.asarray(inertia, dtype=float).reshape(3, 3)
    a = _unit(axis)
    return float(a @ J @ a)


def alpha_max_from_torque(
    inertia: np.ndarray,
    axis: np.ndarray,
    tau_max: float | np.ndarray,
) -> float:
    """Largest ``|α|`` along ``ê`` such that ``τ_ff = J α ê`` respects ``τ_max``.

    A scalar ``tau_max`` is a Euclidean ball ``|τ| ≤ τ_max`` (same geometry
    as the controller clamp).  A length-3 vector is a per-axis box
    ``|τ_i| ≤ τ_max,i``.
    """
    J = np.asarray(inertia, dtype=float).reshape(3, 3)
    a = _unit(axis)
    ja = J @ a
    lim = np.asarray(tau_max, dtype=float)
    if lim.ndim == 0:
        t = float(lim)
        if t <= 0.0:
            raise ValueError(f"tau_max must be positive, got {t}")
        n = float(np.linalg.norm(ja))
        if n < _EPS:
            raise ValueError("J ê is numerically zero; cannot size alpha_max")
        return t / n
    if lim.shape != (3,):
        raise ValueError(f"tau_max must be a scalar or length-3 vector; got shape {lim.shape}")
    if np.any(lim < 0.0):
        raise ValueError(f"per-axis tau_max must be non-negative, got {lim}")
    allowed = np.inf
    for i in range(3):
        coeff = abs(float(ja[i]))
        if coeff < _EPS:
            continue
        allowed = min(allowed, float(lim[i]) / coeff)
    if not np.isfinite(allowed) or allowed <= 0.0:
        raise ValueError("tau_max / J ê does not admit a positive alpha_max along ê")
    return float(allowed)


def feedforward_accel_torque(inertia: np.ndarray, alpha_des: np.ndarray) -> np.ndarray:
    """Inertial feedforward ``J α_des`` (gyroscopic ``ω × Jω`` stays in the law)."""
    J = np.asarray(inertia, dtype=float).reshape(3, 3)
    return J @ np.asarray(alpha_des, dtype=float).reshape(3)


@dataclass
class EigenaxisSlewSample:
    """Instantaneous eigenaxis reference at time ``t``."""

    t: float
    theta: float
    omega: float
    alpha: float
    axis: np.ndarray
    q_des: np.ndarray
    omega_des: np.ndarray
    alpha_des: np.ndarray
    done: bool


@dataclass
class EigenaxisSlewProfile:
    """Rest-to-rest bang-coast-bang (trapezoidal) eigenaxis angle profile.

    Times are seconds, angles radians.  ``omega_cruise`` is the signed
    coast rate (zero-length coast ⇒ triangular bang-bang).  Switching
    instants: accelerate on ``[0, t_acc)``, coast on
    ``[t_acc, t_acc + t_coast)``, decelerate on
    ``[t_acc + t_coast, t_final)``, then hold ``θ = θ_f``, ``ω = 0``.
    """

    q0: np.ndarray
    qf: np.ndarray
    axis: np.ndarray
    theta0: float
    theta_f: float
    alpha_max: float
    omega_max: float
    omega_cruise: float
    t_acc: float
    t_coast: float
    t_final: float
    triangular: bool
    sign: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.q0 = quat_normalize(self.q0)
        self.qf = quat_normalize(self.qf)
        self.axis = _unit(self.axis)
        delta = float(self.theta_f) - float(self.theta0)
        self.sign = 0.0 if abs(delta) < _ANGLE_EPS else float(np.sign(delta))

    @property
    def delta_theta(self) -> float:
        return float(self.theta_f) - float(self.theta0)

    def sample(self, t: float) -> EigenaxisSlewSample:
        """Scalar-time reference: ``(θ, ω, α)`` and body-frame ``(q, ω, α)_des``."""
        t = float(t)
        theta0 = float(self.theta0)
        theta_f = float(self.theta_f)
        axis = self.axis
        if self.t_final <= _EPS or abs(self.delta_theta) < _ANGLE_EPS:
            theta, omega, alpha = theta_f, 0.0, 0.0
            done = True
        elif t < 0.0:
            theta, omega, alpha = theta0, 0.0, 0.0
            done = False
        elif t >= self.t_final:
            theta, omega, alpha = theta_f, 0.0, 0.0
            done = True
        else:
            theta, omega, alpha = self._scalar_kinematics(t)
            done = False
        q_rel = axis_angle_to_quat(axis, theta - theta0)
        q_des = quat_normalize(quat_multiply(self.q0, q_rel))
        omega_des = omega * axis
        alpha_des = alpha * axis
        return EigenaxisSlewSample(
            t=t,
            theta=theta,
            omega=omega,
            alpha=alpha,
            axis=axis.copy(),
            q_des=q_des,
            omega_des=omega_des,
            alpha_des=alpha_des,
            done=done,
        )

    def __call__(self, t: float) -> EigenaxisSlewSample:
        return self.sample(t)

    def _scalar_kinematics(self, t: float) -> tuple[float, float, float]:
        """Piecewise ``(θ, ω, α)`` for ``0 < t < t_final``."""
        s = self.sign
        a = float(self.alpha_max)
        w = abs(float(self.omega_cruise))
        t_acc = float(self.t_acc)
        t_coast = float(self.t_coast)
        theta0 = float(self.theta0)
        t1 = t_acc
        t2 = t_acc + t_coast
        if t < t1:
            # Accelerate: ω(0)=0, α = s α_max.
            return theta0 + s * 0.5 * a * t * t, s * a * t, s * a
        theta_acc = theta0 + s * 0.5 * a * t_acc * t_acc
        if t < t2:
            dt = t - t1
            return theta_acc + s * w * dt, s * w, 0.0
        # Decelerate from coast (or from peak, if triangular).
        dt = t - t2
        theta_coast = theta_acc + s * w * t_coast
        return (
            theta_coast + s * (w * dt - 0.5 * a * dt * dt),
            s * (w - a * dt),
            -s * a,
        )

    def evaluate(self, t: np.ndarray) -> dict[str, np.ndarray]:
        """Vectorized sampling: stacked ``theta, omega, alpha, q_des, ...``."""
        ts = np.asarray(t, dtype=float).reshape(-1)
        n = ts.size
        theta = np.empty(n)
        omega = np.empty(n)
        alpha = np.empty(n)
        q_des = np.empty((n, 4))
        omega_des = np.empty((n, 3))
        alpha_des = np.empty((n, 3))
        done = np.empty(n, dtype=bool)
        for i, ti in enumerate(ts):
            s = self.sample(float(ti))
            theta[i] = s.theta
            omega[i] = s.omega
            alpha[i] = s.alpha
            q_des[i] = s.q_des
            omega_des[i] = s.omega_des
            alpha_des[i] = s.alpha_des
            done[i] = s.done
        return {
            "t": ts.copy(),
            "theta": theta,
            "omega": omega,
            "alpha": alpha,
            "q_des": q_des,
            "omega_des": omega_des,
            "alpha_des": alpha_des,
            "done": done,
        }

    def command(
        self,
        ctrl: SupportsAttitudeCommand,
        q: np.ndarray,
        omega: np.ndarray,
        t: float,
        dt: float,
        *,
        feedforward: bool = True,
    ) -> np.ndarray:
        """Sample at ``t`` and emit a torque from ``ctrl`` (see ``command_slew``)."""
        return command_slew(ctrl, q, omega, self.sample(t), dt, feedforward=feedforward)


def rest_to_rest_eigenaxis(
    q0: np.ndarray | None = None,
    qf: np.ndarray | None = None,
    *,
    axis: np.ndarray | None = None,
    theta0: float = 0.0,
    theta_f: float | None = None,
    omega_max: float | None = None,
    alpha_max: float | None = None,
    inertia: np.ndarray | None = None,
    tau_max: float | np.ndarray | None = None,
) -> EigenaxisSlewProfile:
    """Build a rest-to-rest eigenaxis profile from ``θ0 → θf``.

    Attitude may be given as a quaternion pair ``(q0, qf)`` (shortest-path
    eigenaxis, ``θ: 0 → Φ``) or as a body axis plus scalar angles.  In
    scalar mode ``q0`` is an optional reference (default identity):

        q_start = q0 ⊗ exp(θ0 ê),   q_end = q0 ⊗ exp(θf ê).

    Acceleration is either ``alpha_max`` or torque-limited from ``J`` and
    ``tau_max`` (``α_max = τ_max / |J ê|`` for a Euclidean ball).

    ``omega_max`` is the coast-rate cap.  Omit it for time-optimal
    bang-bang (triangular: peak rate ``√(α_max |Δθ|)``).  If the cap is
    above that peak, the profile collapses to triangular as well.
    """
    q_id = np.array([1.0, 0.0, 0.0, 0.0])
    q_ref = q_id.copy() if q0 is None else quat_normalize(q0)

    if qf is not None:
        q0_arr = q_ref
        qf_arr = quat_normalize(qf)
        axis_q, phi = _shortest_eigenaxis(q0_arr, qf_arr)
        if axis is None:
            axis_u = axis_q
        else:
            axis_u = _unit(axis)
            if phi > _ANGLE_EPS and abs(float(axis_u @ axis_q)) < 0.99:
                axis_u = axis_q
            elif float(axis_u @ axis_q) < 0.0:
                axis_u = -axis_u
        theta0_used = 0.0
        theta_f_used = phi
    else:
        if axis is None:
            raise ValueError("rest_to_rest_eigenaxis needs (q0, qf) or axis plus theta_f")
        if theta_f is None:
            raise ValueError("theta_f is required when qf is omitted")
        axis_u = _unit(axis)
        theta0_used = float(theta0)
        theta_f_used = float(theta_f)
        q0_arr = quat_multiply(q_ref, axis_angle_to_quat(axis_u, theta0_used))
        qf_arr = quat_multiply(q_ref, axis_angle_to_quat(axis_u, theta_f_used))

    delta = theta_f_used - theta0_used

    if alpha_max is None:
        if inertia is None or tau_max is None:
            raise ValueError("alpha_max, or both inertia and tau_max, are required")
        alpha_used = alpha_max_from_torque(inertia, axis_u, tau_max)
    else:
        alpha_used = float(alpha_max)
        if alpha_used <= 0.0:
            raise ValueError(f"alpha_max must be positive, got {alpha_used}")

    if abs(delta) < _ANGLE_EPS:
        return EigenaxisSlewProfile(
            q0=q0_arr,
            qf=q0_arr.copy(),
            axis=axis_u,
            theta0=theta0_used,
            theta_f=theta0_used,
            alpha_max=alpha_used,
            omega_max=0.0 if omega_max is None else float(omega_max),
            omega_cruise=0.0,
            t_acc=0.0,
            t_coast=0.0,
            t_final=0.0,
            triangular=True,
        )

    w_bang = float(np.sqrt(alpha_used * abs(delta)))
    if omega_max is None:
        omega_cap = w_bang
    else:
        omega_cap = float(omega_max)
        if omega_cap <= 0.0:
            raise ValueError(f"omega_max must be positive, got {omega_cap}")

    if omega_cap >= w_bang * (1.0 - 1e-12):
        t_acc = float(np.sqrt(abs(delta) / alpha_used))
        t_coast = 0.0
        omega_cruise = float(np.sign(delta)) * alpha_used * t_acc
        triangular = True
        omega_used = abs(omega_cruise)
    else:
        t_acc = omega_cap / alpha_used
        t_coast = (abs(delta) - omega_cap * omega_cap / alpha_used) / omega_cap
        if t_coast < 0.0:
            t_coast = 0.0
        omega_cruise = float(np.sign(delta)) * omega_cap
        triangular = False
        omega_used = omega_cap

    t_final = 2.0 * t_acc + t_coast
    return EigenaxisSlewProfile(
        q0=q0_arr,
        qf=qf_arr,
        axis=axis_u,
        theta0=theta0_used,
        theta_f=theta_f_used,
        alpha_max=alpha_used,
        omega_max=omega_used,
        omega_cruise=omega_cruise,
        t_acc=t_acc,
        t_coast=t_coast,
        t_final=t_final,
        triangular=triangular,
    )


def _accepts_alpha_des(command: Any) -> bool:
    try:
        return "alpha_des" in inspect.signature(command).parameters
    except (TypeError, ValueError):
        return False


def command_slew(
    ctrl: SupportsAttitudeCommand,
    q: np.ndarray,
    omega: np.ndarray,
    sample: EigenaxisSlewSample,
    dt: float,
    *,
    feedforward: bool = True,
) -> np.ndarray:
    """Track a slew sample with an existing PID / LQR ``command``.

    Always passes ``q_des`` and ``omega_des``.  If ``feedforward`` and the
    law accepts ``alpha_des``, inertial ``J α`` is included *inside* the
    unsaturated command so the existing ``apply_torque_limits`` helper
    sees it.  Otherwise ``J α`` is added after ``command`` and re-clamped
    with that same helper (or the controller's ``_limit_torque``).
    """
    command: Any = ctrl.command
    if feedforward and _accepts_alpha_des(command):
        return command(
            q,
            omega,
            sample.q_des,
            omega_des=sample.omega_des,
            dt=dt,
            alpha_des=sample.alpha_des,
        )
    tau = command(q, omega, sample.q_des, omega_des=sample.omega_des, dt=dt)
    if not feedforward:
        return tau
    tau = np.asarray(tau, dtype=float).reshape(3) + feedforward_accel_torque(
        ctrl.inertia, sample.alpha_des
    )
    limiter = getattr(ctrl, "_limit_torque", None)
    if callable(limiter):
        return limiter(tau)
    from attitude_sim.controls import apply_torque_limits

    return apply_torque_limits(
        tau,
        getattr(ctrl, "torque_limit", None),
        getattr(ctrl, "tau_max", None),
    )


# Quaternion shortest-path helper used in tests / docs.
def geodesic_along_profile(sample: EigenaxisSlewSample, q0: np.ndarray) -> float:
    """Geodesic angle from ``q0`` to ``sample.q_des`` (radians)."""
    return geodesic_angle(sample.q_des, q0)

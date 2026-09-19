"""Three-axis reaction-wheel assembly for the SimLab actuator path.

This is an **additive** actuator model.  It does not rewrite
``attitude_sim.gyrostat`` (one-rotor dual-spin plant), ``polhode``, or
``RigidBody``.  Closed-loop SimLab still integrates Euler's equation on
the rigid body; the wheels contribute the applied torque that already
enters after the PID/LQR command (the existing ``TorqueActuator`` sat /
lag path).

Orthogonal body-aligned wheels with diagonal inertia ``I_w``:

    h_w = I_w Ω
    I_w Ω̇ = τ_m + τ_f + τ_dump

The motor torque is the reaction of the body command after per-axis
``|τ|`` saturation and a momentum-saturation gate

    τ_m^des = −τ_cmd
    τ_m = clip(τ_m^des, ±τ_max)

and if ``|h_i| ≥ h_max,i`` and ``τ_m,i h_i > 0`` then ``τ_m,i ← 0``
(cannot spin further into the wall).  Viscous + smoothed Coulomb
friction on the wheel is

    τ_f = −b Ω − c tanh(Ω / ε)

The torque passed to ``RigidBody`` (which already includes ``−ω × Jω``) is

    τ = −ḣ_w − ω × h_w

so the closed-loop plant matches the gyrostat Euler equation without
calling ``Gyrostat`` / ``step_gyrostat``.  ``gyroscopic=False`` drops
the ``ω × h_w`` term.

``τ_dump`` is a reserved wheel-side dump torque for a future
magnetorquer coupler.  Magnetorquers are **not** modeled here — do not
duplicate a mag-torquer module in this file.

Default construction with no ``I_w`` / ``h_max`` / friction is not
used; ``make_actuator`` still returns identity ``TorqueActuator`` so
prior closed-loop runs match.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from attitude_sim.actuators import clip_torque

DEFAULT_WHEEL_INERTIA = 2.0e-4
"""Smallsat-class spin inertia per axis (kg·m²) when only ``h_max`` is set."""

_COULOMB_EPS = 0.05  # rad/s; smooth sign so Coulomb does not chatter at rest
_SAT_FRAC = 1.0 - 1e-9


def _as_positive_axes(
    value: float | np.ndarray,
    name: str,
) -> np.ndarray:
    """Broadcast a positive scalar or length-3 vector."""
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        out = np.full(3, float(arr))
    else:
        out = arr.reshape(3).copy()
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must be finite")
    if np.any(out <= 0.0):
        raise ValueError(f"{name} must be strictly positive, got {out}")
    return out


def _as_nonneg_axes(
    value: float | np.ndarray | None,
    name: str,
) -> np.ndarray | None:
    if value is None:
        return None
    # Reuse clip_torque validation (allows a zero limit that locks an axis).
    clip_torque(np.zeros(3), value)
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(3, float(arr))
    return arr.reshape(3).copy()


def gate_momentum_saturation(
    tau_m: np.ndarray,
    h: np.ndarray,
    h_max: float | np.ndarray | None,
) -> np.ndarray:
    """Zero motor torque that would drive |h_i| further past h_max,i."""
    out = np.asarray(tau_m, dtype=float).reshape(3).copy()
    if h_max is None:
        return out
    lim = _as_nonneg_axes(h_max, "h_max")
    if lim is None:  # pragma: no cover
        return out
    h = np.asarray(h, dtype=float).reshape(3)
    at_hi = h >= lim * _SAT_FRAC
    at_lo = h <= -lim * _SAT_FRAC
    out = np.where(at_hi & (out > 0.0), 0.0, out)
    out = np.where(at_lo & (out < 0.0), 0.0, out)
    return out


def friction_torque(
    omega_w: np.ndarray,
    visc: float,
    coulomb: float,
    *,
    eps: float = _COULOMB_EPS,
) -> np.ndarray:
    """Wheel-frame friction τ_f = −b Ω − c tanh(Ω / ε)."""
    w = np.asarray(omega_w, dtype=float).reshape(3)
    b = float(visc)
    c = float(coulomb)
    if b < 0.0 or c < 0.0:
        raise ValueError(f"friction coefficients must be >= 0, got visc={b}, coulomb={c}")
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError("coulomb smoothing eps must be positive")
    return -b * w - c * np.tanh(w / eps)


@dataclass
class ReactionWheelAssembly:
    """Three-axis RW: τ_cmd → ḣ_w with |τ|, |h| sat and friction.

    ``apply`` has the same contract as :class:`~attitude_sim.actuators.TorqueActuator`
    plus an optional body rate for the gyroscopic ω × h_w couple.
    """

    tau_max: float | np.ndarray | None = None
    h_max: float | np.ndarray | None = None
    wheel_inertia: float | np.ndarray = DEFAULT_WHEEL_INERTIA
    visc_friction: float = 0.0
    coulomb_friction: float = 0.0
    time_constant: float | None = None
    gyroscopic: bool = True
    coulomb_eps: float = _COULOMB_EPS
    # Reserved magnetorquer dump torque on the wheels (N·m).  External to
    # this module; SimLab does not synthesize an MTQ law here.
    tau_dump: np.ndarray | None = None
    _h: np.ndarray = field(default_factory=lambda: np.zeros(3), init=False, repr=False)
    _tau_m: np.ndarray = field(default_factory=lambda: np.zeros(3), init=False, repr=False)
    _tau: np.ndarray = field(default_factory=lambda: np.zeros(3), init=False, repr=False)
    _tau_sat: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=bool), init=False, repr=False
    )
    _h_sat: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=bool), init=False, repr=False
    )
    _I_w: np.ndarray = field(init=False, repr=False)
    _h_lim: np.ndarray | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.time_constant is not None and float(self.time_constant) < 0.0:
            raise ValueError(f"actuator time constant must be >= 0, got {self.time_constant}")
        self._I_w = _as_positive_axes(self.wheel_inertia, "wheel_inertia")
        self._h_lim = _as_nonneg_axes(self.h_max, "h_max")
        if self.tau_max is not None:
            clip_torque(np.zeros(3), self.tau_max)
        if float(self.visc_friction) < 0.0 or float(self.coulomb_friction) < 0.0:
            raise ValueError("friction coefficients must be >= 0")
        if self.tau_dump is not None:
            self.tau_dump = np.asarray(self.tau_dump, dtype=float).reshape(3).copy()

    def reset(
        self,
        tau0: np.ndarray | None = None,
        h0: np.ndarray | None = None,
    ) -> None:
        """Zero (or set) stored momentum and lagged motor torque."""
        del tau0  # body torque is not a wheel state; keep TorqueActuator signature
        if h0 is None:
            self._h[:] = 0.0
        else:
            h = np.asarray(h0, dtype=float).reshape(3)
            self._h[:] = clip_torque(h, self.h_max) if self.h_max is not None else h
        self._tau_m[:] = 0.0
        self._tau[:] = 0.0
        self._tau_sat[:] = False
        self._h_sat[:] = False

    @property
    def inertia(self) -> np.ndarray:
        """Wheel spin inertias I_w (kg·m²)."""
        return self._I_w.copy()

    @property
    def momentum(self) -> np.ndarray:
        """Wheel angular momentum h_w (N·m·s)."""
        return self._h.copy()

    @property
    def wheel_speed(self) -> np.ndarray:
        """Relative wheel rates Ω = I_w^{-1} h_w (rad/s)."""
        return self._h / self._I_w

    @property
    def torque(self) -> np.ndarray:
        """Last applied body torque (N·m)."""
        return self._tau.copy()

    @property
    def torque_saturated(self) -> np.ndarray:
        """Per-axis motor-torque saturation flags at the last sample."""
        return self._tau_sat.copy()

    @property
    def momentum_saturated(self) -> np.ndarray:
        """Per-axis momentum-saturation flags at the last sample."""
        return self._h_sat.copy()

    def apply(
        self,
        command: np.ndarray,
        dt: float,
        omega: np.ndarray | None = None,
    ) -> np.ndarray:
        """Advance wheels one sample.  Returns the body torque for ``RigidBody``."""
        dt = float(dt)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")
        tau_cmd = np.asarray(command, dtype=float).reshape(3)
        tau_m_des = clip_torque(-tau_cmd, self.tau_max)
        T = None if self.time_constant is None else float(self.time_constant)
        if T is None or T <= 0.0:
            tau_m = tau_m_des
        else:
            alpha = float(np.exp(-dt / T))
            tau_m = alpha * self._tau_m + (1.0 - alpha) * tau_m_des
            tau_m = clip_torque(tau_m, self.tau_max)
        tau_m = gate_momentum_saturation(tau_m, self._h, self.h_max)
        omega_w = self._h / self._I_w
        tau_f = friction_torque(
            omega_w,
            self.visc_friction,
            self.coulomb_friction,
            eps=self.coulomb_eps,
        )
        dump = (
            np.zeros(3)
            if self.tau_dump is None
            else np.asarray(self.tau_dump, dtype=float).reshape(3)
        )
        h_dot_unsat = tau_m + tau_f + dump
        h_next = self._h + h_dot_unsat * dt
        if self.h_max is not None:
            h_next = clip_torque(h_next, self.h_max)
        h_dot = (h_next - self._h) / dt
        gyro = np.zeros(3)
        if self.gyroscopic and omega is not None:
            w = np.asarray(omega, dtype=float).reshape(3)
            gyro = np.cross(w, self._h)
        tau_body = -h_dot - gyro
        self._h = h_next
        self._tau_m = tau_m
        self._tau = tau_body
        self._tau_sat = _at_limit(tau_cmd, self.tau_max)
        self._h_sat = _at_limit(self._h, self.h_max)
        return tau_body.copy()


def _at_limit(
    value: np.ndarray,
    limit: float | np.ndarray | None,
) -> np.ndarray:
    """True on axes sitting on a finite per-axis box."""
    if limit is None:
        return np.zeros(3, dtype=bool)
    lim = np.asarray(limit, dtype=float)
    box = np.full(3, float(lim)) if lim.ndim == 0 else lim.reshape(3)
    return np.abs(np.asarray(value, dtype=float).reshape(3)) >= box * _SAT_FRAC


def make_reaction_wheels(
    *,
    tau_max: float | np.ndarray | None = None,
    h_max: float | np.ndarray | None = None,
    wheel_inertia: float | np.ndarray | None = None,
    visc_friction: float = 0.0,
    coulomb_friction: float = 0.0,
    time_constant: float | None = None,
    gyroscopic: bool = True,
) -> ReactionWheelAssembly:
    """Factory matching ``make_actuator`` / ``make_controller`` style."""
    inertia: float | np.ndarray = (
        DEFAULT_WHEEL_INERTIA if wheel_inertia is None else wheel_inertia
    )
    return ReactionWheelAssembly(
        tau_max=tau_max,
        h_max=h_max,
        wheel_inertia=inertia,
        visc_friction=visc_friction,
        coulomb_friction=coulomb_friction,
        time_constant=time_constant,
        gyroscopic=gyroscopic,
    )

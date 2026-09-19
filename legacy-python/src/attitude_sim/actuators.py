"""Reaction-wheel / torque-limit actuators for the closed-loop SimLab path.

Controllers may still apply an optional Euclidean ``|τ| ≤ τ_max`` clamp
(``attitude_sim.controls``).  This module sits *after* the controller and
models three-axis wheels: independent per-axis limits ``|τ_i| ≤ τ_max,i``,
an optional first-order lag

    τ̇ = (u − τ) / T,

and an optional wheel-momentum dump.  Default construction is identity
(unlimited, no lag, no dump) so existing closed-loop behaviour is
unchanged.  Logged SimLab torque is the *applied* wheel torque that
enters the plant (disturbance ``τ_d`` is still added after this stage).

Momentum storage, friction, and ``|h|`` saturation live in the additive
``attitude_sim.reaction_wheels.ReactionWheelAssembly`` (same ``apply``
contract).  ``make_actuator`` returns that assembly when wheel inertia,
``h_max``, or friction is configured; otherwise this clip/lag/dump box is
used.  Magnetorquer dump on the RW path is *not* modeled here — see the
reserved ``tau_dump`` hook on the RW assembly.  The deadzone dump API
below applies only to ``TorqueActuator``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from attitude_sim.reaction_wheels import ReactionWheelAssembly

# Dump gain [1/s]: τ_dump = k · deadzone(h, h_dump).  1.0 unloads excess
# on a ~1 s time scale when the wheel box has room.
DEFAULT_DUMP_GAIN = 1.0


def clip_torque(
    tau: np.ndarray,
    tau_max: float | np.ndarray | None,
) -> np.ndarray:
    """Clip body torque to per-axis wheel limits ``±τ_max``.

    ``tau_max`` may be ``None`` (unlimited), a positive scalar applied to
    every axis, or a length-3 vector of non-negative limits.  A zero limit
    locks that axis.  Negative limits raise ``ValueError``.
    """
    out = np.asarray(tau, dtype=float).reshape(3).copy()
    if tau_max is None:
        return out
    lim = np.asarray(tau_max, dtype=float)
    if lim.ndim == 0:
        lo = float(lim)
        if lo < 0.0:
            raise ValueError(f"torque limit must be non-negative, got {lo}")
        return np.clip(out, -lo, lo)
    if lim.shape != (3,):
        raise ValueError(f"tau_max must be a scalar or length-3 vector; got shape {lim.shape}")
    if np.any(lim < 0.0):
        raise ValueError(f"per-axis torque limits must be non-negative, got {lim}")
    return np.clip(out, -lim, lim)


def parse_tau_max(text: str | None, name: str = "tau_max") -> float | np.ndarray | None:
    """Parse a CLI / config torque-limit string: empty → None, scalar, or ``x,y,z``."""
    if text is None:
        return None
    raw = str(text).strip()
    if raw == "":
        return None
    if "," in raw:
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) != 3:
            raise ValueError(f"{name} must be a scalar or three comma-separated numbers, got {text!r}")
        arr = np.array([float(p) for p in parts], dtype=float)
        if np.any(arr < 0.0):
            raise ValueError(f"{name} limits must be non-negative, got {arr}")
        return arr
    val = float(raw)
    if val < 0.0:
        raise ValueError(f"{name} must be non-negative, got {val}")
    return val


def momentum_dump_torque(
    momentum: np.ndarray,
    h_dump: float | np.ndarray | None,
    dump_gain: float = DEFAULT_DUMP_GAIN,
) -> np.ndarray:
    """Deadzone dump torque from a wheel-momentum proxy ``h`` (N·m·s).

    Wheel momentum obeys ``ḣ = −τ`` (action–reaction: ``τ`` is the torque
    applied *to the spacecraft*).  To unload, the wheels apply extra torque
    with the same sign as ``h``:

        τ_dump = k · sign(h) · max(|h| − h_dump, 0)

    so ``ḣ = −τ_dump`` drives ``|h|`` toward the threshold.  ``h_dump is
    None`` or ``dump_gain == 0`` is a no-op.  ``h_dump`` may be a scalar or
    a length-3 non-negative vector (same geometry as ``clip_torque``).
    """
    h = np.asarray(momentum, dtype=float).reshape(3)
    gain = float(dump_gain)
    if not np.isfinite(gain) or gain < 0.0:
        raise ValueError(f"dump_gain must be a non-negative finite number, got {dump_gain}")
    if h_dump is None or gain == 0.0:
        return np.zeros(3)
    # Reuse the clip_torque limit grammar (scalar / 3-vector, non-negative).
    clip_torque(np.zeros(3), h_dump)
    thresh = np.asarray(h_dump, dtype=float)
    if thresh.ndim == 0:
        t = np.full(3, float(thresh))
    else:
        t = thresh.reshape(3)
    excess = np.sign(h) * np.maximum(np.abs(h) - t, 0.0)
    return gain * excess


@dataclass
class TorqueActuator:
    """Per-axis saturation, optional first-order lag, optional momentum dump.

    ``tau_max is None`` and ``time_constant`` ``None``/``0`` is a pass-through
    (applied torque equals the command).  The lag is discretized exactly
    under a zero-order hold on the clipped command:

        τ⁺ = e^{−Δt/T} τ + (1 − e^{−Δt/T}) u.

    Wheel momentum ``h`` (N·m·s) is a stored proxy ``ḣ = −τ``.  When
    ``h_dump`` is set, a deadzone dump is added *after* the attitude
    command is boxed, using leftover wheel authority so the PID/LQR
    command — and therefore PID anti-windup — is unchanged.  An equal
    and opposite external torque (``external_torque``) is paired with
    the dump so the spacecraft net dump is zero (magnetorquer /
    thruster stand-in).
    """

    tau_max: float | np.ndarray | None = None
    time_constant: float | None = None
    h_dump: float | np.ndarray | None = None
    dump_gain: float = DEFAULT_DUMP_GAIN
    _tau: np.ndarray = field(default_factory=lambda: np.zeros(3), init=False, repr=False)
    _h: np.ndarray = field(default_factory=lambda: np.zeros(3), init=False, repr=False)
    _tau_dump: np.ndarray = field(default_factory=lambda: np.zeros(3), init=False, repr=False)

    def __post_init__(self) -> None:
        if self.time_constant is not None and float(self.time_constant) < 0.0:
            raise ValueError(f"actuator time constant must be >= 0, got {self.time_constant}")
        gain = float(self.dump_gain)
        if not np.isfinite(gain) or gain < 0.0:
            raise ValueError(f"dump_gain must be a non-negative finite number, got {self.dump_gain}")
        self.dump_gain = gain
        if self.tau_max is not None:
            # Validate limits up front (also normalizes scalar vs vector).
            clip_torque(np.zeros(3), self.tau_max)
        if self.h_dump is not None:
            clip_torque(np.zeros(3), self.h_dump)

    def reset(self, tau0: np.ndarray | None = None, h0: np.ndarray | None = None) -> None:
        if tau0 is None:
            self._tau[:] = 0.0
        else:
            self._tau[:] = clip_torque(tau0, self.tau_max)
        if h0 is None:
            self._h[:] = 0.0
        else:
            self._h[:] = np.asarray(h0, dtype=float).reshape(3)
        self._tau_dump[:] = 0.0

    @property
    def torque(self) -> np.ndarray:
        return self._tau.copy()

    @property
    def momentum(self) -> np.ndarray:
        """Wheel-momentum proxy ``h`` (N·m·s), ``ḣ = −τ``."""
        return self._h.copy()

    @property
    def dump_command(self) -> np.ndarray:
        """Dump torque boxed into leftover wheel authority this sample."""
        return self._tau_dump.copy()

    @property
    def external_torque(self) -> np.ndarray:
        """External cancel of the dump (spacecraft net dump is zero)."""
        return -self._tau_dump

    def apply(
        self,
        command: np.ndarray,
        dt: float,
        omega: np.ndarray | None = None,
    ) -> np.ndarray:
        """Advance one sample: clip command, leftover dump, lag, clip.

        Returns applied wheel ``τ``.  Pair ``external_torque`` on the plant
        so the dump does not disturb attitude.  ``omega`` is ignored
        (body-rate coupling belongs to the RW assembly).
        """
        del omega
        u_cmd = clip_torque(command, self.tau_max)
        raw_dump = momentum_dump_torque(self._h, self.h_dump, self.dump_gain)
        # Command has priority: dump uses only leftover box authority.
        # clip(u_cmd + dump) − u_cmd is that leftover projection and
        # reuses clip_torque (same box as make_actuator / apply_torque_limits).
        tau_dump = clip_torque(u_cmd + raw_dump, self.tau_max) - u_cmd
        self._tau_dump = tau_dump
        u = u_cmd + tau_dump
        T = None if self.time_constant is None else float(self.time_constant)
        if T is None or T <= 0.0:
            self._tau = u
        else:
            if dt <= 0.0:
                raise ValueError(f"dt must be positive, got {dt}")
            alpha = float(np.exp(-dt / T))
            self._tau = alpha * self._tau + (1.0 - alpha) * u
            self._tau = clip_torque(self._tau, self.tau_max)
        if dt > 0.0:
            self._h = self._h - self._tau * float(dt)
        return self._tau.copy()


def make_actuator(
    tau_max: float | np.ndarray | None = None,
    time_constant: float | None = None,
    *,
    wheel_inertia: float | np.ndarray | None = None,
    h_max: float | np.ndarray | None = None,
    visc_friction: float = 0.0,
    coulomb_friction: float = 0.0,
    gyroscopic: bool = True,
    h_dump: float | np.ndarray | None = None,
    dump_gain: float = DEFAULT_DUMP_GAIN,
) -> TorqueActuator | ReactionWheelAssembly:
    """Factory matching ``make_controller`` / ``make_estimator`` style.

    Extra RW kwargs (``wheel_inertia``, ``h_max``, friction) select
    :class:`~attitude_sim.reaction_wheels.ReactionWheelAssembly`.  Same
    ``apply`` / ``reset`` / ``torque`` / ``momentum`` surface as
    ``TorqueActuator``.  Dump kwargs are keyword-only and apply only to
    the clip/lag ``TorqueActuator`` path — the fuller RW model keeps its
    reserved ``tau_dump`` hook.
    """
    use_rw = (
        wheel_inertia is not None
        or h_max is not None
        or float(visc_friction) != 0.0
        or float(coulomb_friction) != 0.0
    )
    if not use_rw:
        return TorqueActuator(
            tau_max=tau_max,
            time_constant=time_constant,
            h_dump=h_dump,
            dump_gain=dump_gain,
        )
    from attitude_sim.reaction_wheels import make_reaction_wheels

    return make_reaction_wheels(
        tau_max=tau_max,
        h_max=h_max,
        wheel_inertia=wheel_inertia,
        visc_friction=visc_friction,
        coulomb_friction=coulomb_friction,
        time_constant=time_constant,
        gyroscopic=gyroscopic,
    )

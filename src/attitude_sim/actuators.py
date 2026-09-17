"""Reaction-wheel / torque-limit actuators for the closed-loop SimLab path.

Controllers may still apply an optional Euclidean ``|τ| ≤ τ_max`` clamp
(``attitude_sim.controls``).  This module sits *after* the controller and
models three-axis wheels: independent per-axis limits ``|τ_i| ≤ τ_max,i``
and an optional first-order lag

    τ̇ = (u − τ) / T.

Default construction is identity (unlimited, no lag) so existing closed-loop
behaviour is unchanged.  Logged SimLab torque is the *applied* wheel torque
that enters the plant (disturbance ``τ_d`` is still added after this stage).

Momentum storage, friction, and ``|h|`` saturation live in the additive
``attitude_sim.reaction_wheels.ReactionWheelAssembly`` (same ``apply``
contract).  ``make_actuator`` returns that assembly when wheel inertia,
``h_max``, or friction is configured; otherwise this clip/lag box is
unchanged.  Magnetorquer dump is *not* modeled here — see the reserved
``tau_dump`` hook on the RW assembly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from attitude_sim.reaction_wheels import ReactionWheelAssembly


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


@dataclass
class TorqueActuator:
    """Per-axis saturation plus optional first-order lag.

    ``tau_max is None`` and ``time_constant`` ``None``/``0`` is a pass-through
    (applied torque equals the command).  The lag is discretized exactly
    under a zero-order hold on the clipped command:

        τ⁺ = e^{−Δt/T} τ + (1 − e^{−Δt/T}) u.
    """

    tau_max: float | np.ndarray | None = None
    time_constant: float | None = None
    _tau: np.ndarray = field(default_factory=lambda: np.zeros(3), init=False, repr=False)

    def __post_init__(self) -> None:
        if self.time_constant is not None and float(self.time_constant) < 0.0:
            raise ValueError(f"actuator time constant must be >= 0, got {self.time_constant}")
        if self.tau_max is not None:
            # Validate limits up front (also normalizes scalar vs vector).
            clip_torque(np.zeros(3), self.tau_max)

    def reset(self, tau0: np.ndarray | None = None) -> None:
        if tau0 is None:
            self._tau[:] = 0.0
        else:
            self._tau[:] = clip_torque(tau0, self.tau_max)

    @property
    def torque(self) -> np.ndarray:
        return self._tau.copy()

    def apply(
        self,
        command: np.ndarray,
        dt: float,
        omega: np.ndarray | None = None,
    ) -> np.ndarray:
        """Advance one sample: clip command, lag, clip output. Returns applied τ.

        ``omega`` is ignored (body-rate coupling belongs to the RW assembly).
        """
        del omega
        u = clip_torque(command, self.tau_max)
        T = None if self.time_constant is None else float(self.time_constant)
        if T is None or T <= 0.0:
            self._tau = u
        else:
            if dt <= 0.0:
                raise ValueError(f"dt must be positive, got {dt}")
            alpha = float(np.exp(-dt / T))
            self._tau = alpha * self._tau + (1.0 - alpha) * u
            self._tau = clip_torque(self._tau, self.tau_max)
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
) -> TorqueActuator | ReactionWheelAssembly:
    """Factory matching ``make_controller`` / ``make_estimator`` style.

    Extra RW kwargs (``wheel_inertia``, ``h_max``, friction) select
    :class:`~attitude_sim.reaction_wheels.ReactionWheelAssembly`.  Same
    ``apply`` / ``reset`` / ``torque`` surface as ``TorqueActuator``.
    Magnetorquers are not constructed here.
    """
    use_rw = (
        wheel_inertia is not None
        or h_max is not None
        or float(visc_friction) != 0.0
        or float(coulomb_friction) != 0.0
    )
    if not use_rw:
        return TorqueActuator(tau_max=tau_max, time_constant=time_constant)
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

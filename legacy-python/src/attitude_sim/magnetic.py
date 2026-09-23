"""Magnetic-torquer actuator: dipole ``m``, ``τ = m × B``, per-axis ``|m|`` sat.

Library-only (not a SimLab CLI flag).  Controllers still emit a torque
command; this module maps that command onto a body dipole, saturates each
axis independently, and returns the physical torque

    τ = m × B

which is always orthogonal to ``B``.  The leftover / remanent dipole can
optionally be handed to :class:`~attitude_sim.disturbances.ResidualDipoleTorque`
so the same ``τ = m × B`` physics is used for the uncommanded residual
(no second dipole model).

The cross-product allocation (when a torque is commanded) is

    m = (B × τ_cmd) / ‖B‖² ,

which realizes the component of ``τ_cmd`` perpendicular to ``B``.  A
near-zero field returns ``m = 0``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from attitude_sim.actuators import clip_torque, parse_tau_max
from attitude_sim.disturbances import CircularOrbit, ResidualDipoleTorque, magnetic_dipole_torque

_B2_EPS = 1e-24


def clip_dipole(
    m: np.ndarray,
    m_max: float | np.ndarray | None,
) -> np.ndarray:
    """Clip body dipole to per-axis limits ``±m_max`` (A·m²).

    Same contract as :func:`~attitude_sim.actuators.clip_torque`: ``None``
    is unlimited, a scalar applies to every axis, a length-3 vector is
    per-axis.  A zero limit locks that coil.  Negative limits raise.
    """
    return clip_torque(m, m_max)


def parse_m_max(text: str | None, name: str = "m_max") -> float | np.ndarray | None:
    """Parse a dipole-limit string: empty → None, scalar, or ``x,y,z``."""
    return parse_tau_max(text, name=name)


def magnetic_torque(m_body: np.ndarray, B_body: np.ndarray) -> np.ndarray:
    """Body-frame magnetic torque ``τ = m × B`` (N·m)."""
    return magnetic_dipole_torque(m_body, B_body)


def dipole_from_torque(
    tau_cmd: np.ndarray,
    B_body: np.ndarray,
    *,
    min_b2: float = _B2_EPS,
) -> np.ndarray:
    """Cross-product dipole that realizes ``τ_⊥ = (I − B̂B̂ᵀ) τ_cmd``.

    ``m = (B × τ_cmd) / ‖B‖²``.  Returns zeros when ``‖B‖² < min_b2``.
    """
    tau = np.asarray(tau_cmd, dtype=float).reshape(3)
    b = np.asarray(B_body, dtype=float).reshape(3)
    b2 = float(b @ b)
    if not np.isfinite(b2) or b2 < float(min_b2):
        return np.zeros(3)
    return np.cross(b, tau) / b2


@dataclass
class MagneticTorquer:
    """Three-axis magnetic torquer with per-axis ``|m_i|`` saturation.

    ``m_max is None`` is unlimited.  ``residual_m`` is an uncommanded
    remanent dipole (A·m²) that does **not** enter :meth:`apply_dipole` /
    :meth:`apply_torque`; hand it to the residual-dipole disturbance via
    :meth:`residual_disturbance` so commanded and leftover dipoles are
    not double-counted.
    """

    m_max: float | np.ndarray | None = None
    residual_m: np.ndarray = field(default_factory=lambda: np.zeros(3))
    _m: np.ndarray = field(default_factory=lambda: np.zeros(3), init=False, repr=False)

    def __post_init__(self) -> None:
        self.residual_m = np.asarray(self.residual_m, dtype=float).reshape(3).copy()
        if self.m_max is not None:
            clip_dipole(np.zeros(3), self.m_max)

    def reset(self, m0: np.ndarray | None = None) -> None:
        if m0 is None:
            self._m[:] = 0.0
        else:
            self._m[:] = clip_dipole(m0, self.m_max)

    @property
    def dipole(self) -> np.ndarray:
        return self._m.copy()

    def clip(self, m_cmd: np.ndarray) -> np.ndarray:
        return clip_dipole(m_cmd, self.m_max)

    def leftover_after_sat(self, m_cmd: np.ndarray) -> np.ndarray:
        """Unsaturated remainder ``m_cmd − clip(m_cmd)`` (A·m²)."""
        cmd = np.asarray(m_cmd, dtype=float).reshape(3)
        return cmd - self.clip(cmd)

    def apply_dipole(self, m_cmd: np.ndarray, B_body: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Saturate a commanded dipole and return ``(τ, m_applied)``."""
        self._m = self.clip(m_cmd)
        tau = magnetic_torque(self._m, B_body)
        return tau.copy(), self._m.copy()

    def apply_torque(self, tau_cmd: np.ndarray, B_body: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Allocate ``m`` from ``τ_cmd``, saturate, return ``(τ, m_applied)``."""
        return self.apply_dipole(dipole_from_torque(tau_cmd, B_body), B_body)

    def residual_disturbance(
        self,
        *,
        orbit: CircularOrbit | None = None,
        model: str = "tilted",
    ) -> ResidualDipoleTorque | None:
        """Handoff remanent ``residual_m`` to :class:`ResidualDipoleTorque`.

        Returns ``None`` when the remanent dipole is identically zero so
        callers can skip the env-torque term.
        """
        if not np.any(self.residual_m):
            return None
        return residual_dipole_handoff(self.residual_m, orbit=orbit, model=model)


def residual_dipole_handoff(
    m_residual: np.ndarray,
    *,
    orbit: CircularOrbit | None = None,
    model: str = "tilted",
) -> ResidualDipoleTorque:
    """Wrap a leftover / remanent dipole as the existing disturbance model."""
    return ResidualDipoleTorque(
        m_body=np.asarray(m_residual, dtype=float).reshape(3),
        orbit=orbit,
        model=model,
    )


def make_magnetic_torquer(
    m_max: float | np.ndarray | None = None,
    residual_m: np.ndarray | None = None,
) -> MagneticTorquer:
    """Factory matching ``make_actuator`` / ``make_controller`` style."""
    return MagneticTorquer(
        m_max=m_max,
        residual_m=np.zeros(3) if residual_m is None else residual_m,
    )

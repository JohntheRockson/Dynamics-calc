"""Environmental disturbance torques for rigid-body attitude sims.

These models return a **body-frame** torque ``τ_b`` that can be added to
the control/actuator torque in Euler's equation (see
``attitude_sim.plant``).  They do not change the integrator.

Orbit and nadir frames
----------------------
Inertial vectors map as ``v_I = R(q) v_b`` (scalar-first ``q``).  A
circular-orbit **RSW / local-vertical** frame ``O`` is right-handed with

    x̂_o = r̂     (zenith: Earth centre → spacecraft),
    ẑ_o = ĥ     (orbit normal, ``r × v``),
    ŷ_o = ẑ_o × x̂_o   (along-track for a circular orbit).

**Nadir** is ``−r̂``.  Body-frame unit vectors are ``v_b = R(q)ᵀ v_I``.

Gravity-gradient torque
-----------------------
For a rigid body with body inertia ``J = Jᵀ ≻ 0`` in a central
inverse-square field the body-frame gravity-gradient torque is

    τ_gg = 3 (μ / r³) (r̂_b × J r̂_b)

(``r̂_b`` = zenith in the body frame).  Equivalently
``3 n² (r̂_b × J r̂_b)`` on a circular orbit, where ``n = √(μ / r³)``.
Replacing ``r̂`` by nadir ``−r̂`` does not change ``τ_gg``.  If ``J`` is
principal in the body frame and ``r̂_b`` lies along a principal axis,
``J r̂_b ∥ r̂_b`` and ``τ_gg = 0``.  The torque does not depend on ``ω``.

Residual magnetic dipole
------------------------
A residual spacecraft dipole ``m`` (A·m²) in a field ``B`` (T) produces

    τ_m = m × B

resolved in the same frame.  In the body frame
``B_b = R(q)ᵀ B_I``.  Parallel ``m`` and ``B`` give zero torque.

Earth field options (dipole frozen in ECI; no ECEF rotation / IGRF):

* **tilted dipole** — ``B_I = [3 (m_E · r̂) r̂ − m_E] / r³`` with
  ``m_E`` in T·m³ pointing toward magnetic north (geographic south when
  untilted: ``m_E = −μ_m ẑ``).  Default tilt is 11.5° toward RA 0.
* **orbit-normal** — ``B_I = (μ_m / r³) ĥ``, the equatorial untilted
  dipole (constant inertial direction along ``+ẑ`` for a prograde
  equatorial circular orbit).

API
---
Primitives ``gravity_gradient_torque`` / ``magnetic_dipole_torque`` plus
:class:`GravityGradientTorque`, :class:`ResidualDipoleTorque`, and
:class:`EnvironmentalTorques` all expose

    τ_body(q, ω, t=None, *, orbit=None)

``t`` is used with a bound :class:`CircularOrbit`; an explicit
:class:`OrbitState` wins when both are given.  ``ω`` is part of the
uniform signature (unused by these two conservative models).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from attitude_sim.plant import validate_inertia
from attitude_sim.quaternions import quat_to_rotation

# WGS-84 GM (m³/s²) and a dipole moment that gives ~30 μT at 6371 km.
MU_EARTH = 3.986004418e14
EARTH_MAG_MOMENT = 7.94e15  # T·m³  (|m_E| so B_eq ≈ |m_E| / r³)
EARTH_DIPOLE_TILT_RAD = np.deg2rad(11.5)

_RADIUS_EPS = 1e-15


def inertial_to_body(q: np.ndarray, v_eci: np.ndarray) -> np.ndarray:
    """Express an inertial vector in the body frame: ``v_b = R(q)ᵀ v_I``."""
    return quat_to_rotation(q).T @ np.asarray(v_eci, dtype=float).reshape(3)


def body_to_inertial(q: np.ndarray, v_body: np.ndarray) -> np.ndarray:
    """Express a body vector in inertial coordinates: ``v_I = R(q) v_b``."""
    return quat_to_rotation(q) @ np.asarray(v_body, dtype=float).reshape(3)


def magnetic_field_body(q: np.ndarray, B_eci: np.ndarray) -> np.ndarray:
    """Body-frame magnetic field ``B_b = R(q)ᵀ B_I``."""
    return inertial_to_body(q, B_eci)


def _as_vec3(name: str, v: np.ndarray) -> np.ndarray:
    x = np.asarray(v, dtype=float).reshape(3)
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} must be finite")
    return x


def _unit(name: str, v: np.ndarray) -> np.ndarray:
    x = _as_vec3(name, v)
    n = float(np.linalg.norm(x))
    if n < _RADIUS_EPS:
        raise ValueError(f"{name} must be nonzero")
    return x / n


def _mu_over_r3(mu: float, radius: float) -> float:
    mu = float(mu)
    radius = float(radius)
    if not np.isfinite(mu) or mu <= 0.0:
        raise ValueError("mu must be positive")
    if not np.isfinite(radius) or radius <= _RADIUS_EPS:
        raise ValueError("orbit radius must be positive")
    return mu / (radius * radius * radius)


@dataclass
class OrbitState:
    """Earth-centred inertial barycentre state for a disturbance evaluation.

    ``r_eci`` (m) and ``v_eci`` (m/s) are inertial.  ``mu`` is the
    gravitational parameter (m³/s²).  ``r̂`` is zenith; ``ĥ`` is the
    instantaneous orbit normal.
    """

    r_eci: np.ndarray
    v_eci: np.ndarray
    mu: float = MU_EARTH

    def __post_init__(self) -> None:
        self.r_eci = _as_vec3("r_eci", self.r_eci)
        self.v_eci = _as_vec3("v_eci", self.v_eci)
        self.mu = float(self.mu)
        if not np.isfinite(self.mu) or self.mu <= 0.0:
            raise ValueError("mu must be positive")

    @property
    def radius(self) -> float:
        r = float(np.linalg.norm(self.r_eci))
        if r <= _RADIUS_EPS:
            raise ValueError("orbit radius must be positive")
        return r

    @property
    def r_hat(self) -> np.ndarray:
        """Zenith unit vector in ECI (Earth centre → spacecraft)."""
        return self.r_eci / self.radius

    @property
    def nadir_hat(self) -> np.ndarray:
        """Nadir unit vector in ECI (spacecraft → Earth centre)."""
        return -self.r_hat

    @property
    def h_hat(self) -> np.ndarray:
        """Orbit-normal unit vector ``normalize(r × v)`` in ECI."""
        return _unit("orbit angular momentum r × v", np.cross(self.r_eci, self.v_eci))

    @property
    def mu_over_r3(self) -> float:
        return _mu_over_r3(self.mu, self.radius)

    @property
    def mean_motion(self) -> float:
        """Circular-orbit mean motion ``n = √(μ / r³)`` (rad/s)."""
        return float(np.sqrt(self.mu_over_r3))


@dataclass
class CircularOrbit:
    """Keplerian circular orbit in ECI, parametrized by argument of latitude.

    At time ``t`` the argument of latitude is ``u = u0 + n t`` with
    ``n = √(μ / a³)``.  Inclination and RAAN are the usual 3-1-3 angles
    (argument of periapsis is absorbed into ``u``).
    """

    radius: float
    inclination: float = 0.0
    raan: float = 0.0
    arg_latitude0: float = 0.0
    mu: float = MU_EARTH

    def __post_init__(self) -> None:
        self.radius = float(self.radius)
        self.inclination = float(self.inclination)
        self.raan = float(self.raan)
        self.arg_latitude0 = float(self.arg_latitude0)
        self.mu = float(self.mu)
        if not np.isfinite(self.radius) or self.radius <= _RADIUS_EPS:
            raise ValueError("orbit radius must be positive")
        if not np.isfinite(self.mu) or self.mu <= 0.0:
            raise ValueError("mu must be positive")
        for name, val in (
            ("inclination", self.inclination),
            ("raan", self.raan),
            ("arg_latitude0", self.arg_latitude0),
        ):
            if not np.isfinite(val):
                raise ValueError(f"{name} must be finite")

    @property
    def mean_motion(self) -> float:
        return float(np.sqrt(_mu_over_r3(self.mu, self.radius)))

    def state_at(self, t: float) -> OrbitState:
        """Inertial position/velocity at time ``t`` (s)."""
        t = float(t)
        if not np.isfinite(t):
            raise ValueError("t must be finite")
        n = self.mean_motion
        u = self.arg_latitude0 + n * t
        cu, su = np.cos(u), np.sin(u)
        ci, si = np.cos(self.inclination), np.sin(self.inclination)
        cO, sO = np.cos(self.raan), np.sin(self.raan)
        a = self.radius
        r_eci = a * np.array(
            [
                cO * cu - sO * su * ci,
                sO * cu + cO * su * ci,
                su * si,
            ]
        )
        v_eci = a * n * np.array(
            [
                -cO * su - sO * cu * ci,
                -sO * su + cO * cu * ci,
                cu * si,
            ]
        )
        return OrbitState(r_eci, v_eci, self.mu)


def gravity_gradient_torque(
    inertia: np.ndarray,
    r_hat_body: np.ndarray,
    mu_over_r3: float,
) -> np.ndarray:
    """Body-frame gravity-gradient torque ``3 (μ/r³) (r̂_b × J r̂_b)``.

    ``r_hat_body`` is zenith (or nadir — the sign cancels) expressed in
    the body frame.  It is re-normalized.
    """
    J = validate_inertia(inertia)
    r_hat = _unit("r_hat_body", r_hat_body)
    n2 = float(mu_over_r3)
    if not np.isfinite(n2):
        raise ValueError("mu/r³ must be finite")
    return 3.0 * n2 * np.cross(r_hat, J @ r_hat)


def magnetic_dipole_torque(m_body: np.ndarray, B_body: np.ndarray) -> np.ndarray:
    """Body-frame residual-dipole torque ``τ = m × B``."""
    return np.cross(_as_vec3("m_body", m_body), _as_vec3("B_body", B_body))


def earth_dipole_moment_eci(
    *,
    moment: float = EARTH_MAG_MOMENT,
    tilt_rad: float = EARTH_DIPOLE_TILT_RAD,
    ra_rad: float = 0.0,
) -> np.ndarray:
    """Earth dipole moment vector in ECI (T·m³).

    Untilted (``tilt_rad = 0``) the moment points at geographic south:
    ``m_E = −moment ẑ``.  A positive tilt rotates that axis toward
    equatorial right ascension ``ra_rad``.
    """
    moment = float(moment)
    tilt_rad = float(tilt_rad)
    ra_rad = float(ra_rad)
    if not np.isfinite(moment) or moment < 0.0:
        raise ValueError("magnetic moment magnitude must be nonnegative")
    if not np.isfinite(tilt_rad) or not np.isfinite(ra_rad):
        raise ValueError("dipole tilt and RA must be finite")
    st, ct = np.sin(tilt_rad), np.cos(tilt_rad)
    return moment * np.array(
        [st * np.cos(ra_rad), st * np.sin(ra_rad), -ct],
        dtype=float,
    )


def dipole_field_eci(r_eci: np.ndarray, moment_eci: np.ndarray) -> np.ndarray:
    """Dipole field ``B = [3 (m · r̂) r̂ − m] / r³`` (T) at inertial ``r``.

    ``moment_eci`` is the dipole moment vector in T·m³.
    """
    r = _as_vec3("r_eci", r_eci)
    m = _as_vec3("moment_eci", moment_eci)
    radius = float(np.linalg.norm(r))
    if radius <= _RADIUS_EPS:
        raise ValueError("position radius must be positive")
    r_hat = r / radius
    return (3.0 * float(np.dot(m, r_hat)) * r_hat - m) / (radius**3)


def earth_dipole_field_eci(
    r_eci: np.ndarray,
    *,
    moment: float = EARTH_MAG_MOMENT,
    tilt_rad: float = EARTH_DIPOLE_TILT_RAD,
    ra_rad: float = 0.0,
) -> np.ndarray:
    """Tilted (or untilted) Earth dipole field in ECI (T)."""
    return dipole_field_eci(
        r_eci,
        earth_dipole_moment_eci(moment=moment, tilt_rad=tilt_rad, ra_rad=ra_rad),
    )


def orbit_normal_dipole_field_eci(
    orbit: OrbitState,
    *,
    moment: float = EARTH_MAG_MOMENT,
    sign: float = 1.0,
) -> np.ndarray:
    """Equatorial-dipole option: ``B_I = sign (μ_m / r³) ĥ`` (T).

    Matches an untilted Earth dipole on a prograde equatorial circular
    orbit when ``sign = +1`` (``B`` along ``+ẑ``).
    """
    moment = float(moment)
    sign = float(sign)
    if not np.isfinite(moment) or moment < 0.0:
        raise ValueError("magnetic moment magnitude must be nonnegative")
    if not np.isfinite(sign) or sign not in (-1.0, 1.0):
        raise ValueError("sign must be +1 or -1")
    return (sign * moment / (orbit.radius**3)) * orbit.h_hat


def magnetic_field_eci(
    orbit: OrbitState,
    *,
    model: str = "tilted",
    moment: float = EARTH_MAG_MOMENT,
    tilt_rad: float = EARTH_DIPOLE_TILT_RAD,
    ra_rad: float = 0.0,
    sign: float = 1.0,
) -> np.ndarray:
    """Earth field in ECI for ``model='tilted'`` or ``'orbit_normal'``."""
    model = str(model)
    if model == "tilted":
        return earth_dipole_field_eci(
            orbit.r_eci, moment=moment, tilt_rad=tilt_rad, ra_rad=ra_rad
        )
    if model == "orbit_normal":
        return orbit_normal_dipole_field_eci(orbit, moment=moment, sign=sign)
    raise ValueError("model must be 'tilted' or 'orbit_normal'")


def _resolve_orbit(
    t: float | None,
    orbit: OrbitState | None,
    bound: CircularOrbit | None,
) -> OrbitState:
    if orbit is not None:
        if not isinstance(orbit, OrbitState):
            raise TypeError("orbit must be an OrbitState")
        return orbit
    if bound is None:
        raise ValueError("pass orbit=OrbitState or bind a CircularOrbit and t")
    if t is None:
        raise ValueError("t is required when orbit state is omitted")
    return bound.state_at(t)


@dataclass
class GravityGradientTorque:
    """Gravity-gradient disturbance with ``τ_body(q, ω, t, *, orbit)``.

    Bind a :class:`CircularOrbit` to evaluate from time ``t``; otherwise
    pass ``orbit=OrbitState(...)``.
    """

    inertia: np.ndarray
    orbit: CircularOrbit | None = None

    def __post_init__(self) -> None:
        self.inertia = validate_inertia(self.inertia)

    def tau_body(
        self,
        q: np.ndarray,
        omega: np.ndarray,
        t: float | None = None,
        *,
        orbit: OrbitState | None = None,
    ) -> np.ndarray:
        """Body-frame gravity-gradient torque at this attitude.

        ``omega`` is unused (GG depends on attitude and position only).
        """
        _as_vec3("omega", omega)
        state = _resolve_orbit(t, orbit, self.orbit)
        r_hat_b = inertial_to_body(q, state.r_hat)
        return gravity_gradient_torque(self.inertia, r_hat_b, state.mu_over_r3)


@dataclass
class ResidualDipoleTorque:
    """Residual-dipole disturbance ``τ = m_b × B_b(q, orbit)``.

    ``model`` is ``'tilted'`` (default) or ``'orbit_normal'``.  Bind a
    :class:`CircularOrbit` to evaluate from time ``t``.
    """

    m_body: np.ndarray
    orbit: CircularOrbit | None = None
    model: str = "tilted"
    moment: float = EARTH_MAG_MOMENT
    tilt_rad: float = EARTH_DIPOLE_TILT_RAD
    ra_rad: float = 0.0
    sign: float = 1.0

    def __post_init__(self) -> None:
        self.m_body = _as_vec3("m_body", self.m_body)
        self.model = str(self.model)
        if self.model not in ("tilted", "orbit_normal"):
            raise ValueError("model must be 'tilted' or 'orbit_normal'")
        self.moment = float(self.moment)
        self.tilt_rad = float(self.tilt_rad)
        self.ra_rad = float(self.ra_rad)
        self.sign = float(self.sign)
        if not np.isfinite(self.moment) or self.moment < 0.0:
            raise ValueError("magnetic moment magnitude must be nonnegative")
        if self.sign not in (-1.0, 1.0):
            raise ValueError("sign must be +1 or -1")

    def B_eci(self, orbit: OrbitState) -> np.ndarray:
        """Inertial magnetic field (T) at this orbit state."""
        return magnetic_field_eci(
            orbit,
            model=self.model,
            moment=self.moment,
            tilt_rad=self.tilt_rad,
            ra_rad=self.ra_rad,
            sign=self.sign,
        )

    def B_body(self, q: np.ndarray, orbit: OrbitState) -> np.ndarray:
        """Body-frame magnetic field given attitude ``q``."""
        return magnetic_field_body(q, self.B_eci(orbit))

    def tau_body(
        self,
        q: np.ndarray,
        omega: np.ndarray,
        t: float | None = None,
        *,
        orbit: OrbitState | None = None,
    ) -> np.ndarray:
        """Body-frame residual-dipole torque.

        ``omega`` is unused (this model has no eddy-current term).
        """
        _as_vec3("omega", omega)
        state = _resolve_orbit(t, orbit, self.orbit)
        return magnetic_dipole_torque(self.m_body, self.B_body(q, state))


@dataclass
class EnvironmentalTorques:
    """Sum of gravity-gradient and/or residual-dipole body torques."""

    gravity_gradient: GravityGradientTorque | None = None
    residual_dipole: ResidualDipoleTorque | None = None
    extra: Sequence[GravityGradientTorque | ResidualDipoleTorque] = field(default_factory=tuple)

    def tau_body(
        self,
        q: np.ndarray,
        omega: np.ndarray,
        t: float | None = None,
        *,
        orbit: OrbitState | None = None,
    ) -> np.ndarray:
        tau = np.zeros(3, dtype=float)
        for model in (self.gravity_gradient, self.residual_dipole, *self.extra):
            if model is None:
                continue
            piece = np.asarray(
                model.tau_body(q, omega, t, orbit=orbit), dtype=np.float64
            )
            tau = tau + piece.reshape(3)
        return tau

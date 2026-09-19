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

Aerodynamic torque (simple panel / box)
---------------------------------------
Drag on a single flat face (or a cannonball / equivalent-area box) is

    F_aero = −½ ρ |v_rel|² C_d A n̂

so the body torque about the centre of mass is

    τ_aero = r_cp × F_aero = r_cp × (−½ ρ |v_rel|² C_d A n̂).

``n̂`` is the force-direction unit vector in the same frame as ``r_cp``.
If omitted, ``n̂ = v̂_rel`` (ram / cannonball: drag anti-velocity).  Optional
``project_area`` multiplies ``A`` by ``max(n̂ · v̂_rel, 0)`` so a leeward
face contributes nothing.  ``r_cp ∥ F`` (including ``r_cp = 0``) gives
``τ = 0``.  ``ω × r_cp`` on the surface velocity is neglected (orbital
speed ≫ body rates × panel size).

Density is an **exponential atmosphere** (isothermal, spherical Earth)

    ρ(h) = ρ_ref exp(−(h − h_ref) / H),     h = |r| − R_E.

Defaults are a ~400 km LEO mean (``ρ_ref = 2.5×10⁻¹² kg/m³``,
``H = 60 km``), not NRLMSISE / Jacchia.  Relative velocity is inertial
``v`` unless ``co_rotating=True``, which subtracts ``ω_E × r``
(``ω_E`` along ECI ``ẑ``).  No winds, no shadowing, one face.

Solar radiation pressure torque
-------------------------------
A cannonball / absorbing-plate model with force along the sun unit
``û_sun`` (spacecraft → sun) and projected area ``A cosθ`` is

    F_srp = P_srp c_r A cosθ û_sun
    τ_srp = r_cp × F_srp.

``P_srp ≈ 4.56×10⁻⁶ N/m²`` at 1 AU.  ``c_r`` is a reflectivity
coefficient (1 = absorb, 2 = perfect specular reflection along ``û_sun``
in this one-term model; there is no separate specular/diffuse pair).
``cosθ = n̂ · û_sun`` when a panel normal is given; ``cosθ = 1`` for a
cannonball (``n̂`` omitted).  ``cosθ ≤ 0`` (back face) gives ``F = 0``.

**Eclipse** is an umbra on/off stub: ``eclipse=True`` zeroes SRP (no
penumbra, no solar ephemeris).  Optional cylindrical-Earth umbra
(``eclipse='cylindrical'``) is sun-at-infinity geometry: the spacecraft
is in shadow iff it is anti-sun of Earth and the perpendicular distance
to the sun line is ``< R_E``.  Constant ``û_sun`` in ECI by default
(same spirit as the sun-sensor stub).

API
---
Primitives ``gravity_gradient_torque`` / ``magnetic_dipole_torque`` /
``aerodynamic_torque`` / ``srp_torque`` plus
:class:`GravityGradientTorque`, :class:`ResidualDipoleTorque`,
:class:`AerodynamicTorque`, :class:`SolarRadiationPressureTorque`, and
:class:`EnvironmentalTorques` all expose

    τ_body(q, ω, t=None, *, orbit=None)

``t`` is used with a bound :class:`CircularOrbit`; an explicit
:class:`OrbitState` wins when both are given.  ``ω`` is part of the
uniform signature (unused by GG, dipole, aero, and SRP as implemented).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from attitude_sim.plant import validate_inertia
from attitude_sim.quaternions import quat_to_rotation

# WGS-84 GM (m³/s²) and a dipole moment that gives ~30 μT at 6371 km.
MU_EARTH = 3.986004418e14
EARTH_MAG_MOMENT = 7.94e15  # T·m³  (|m_E| so B_eq ≈ |m_E| / r³)
EARTH_DIPOLE_TILT_RAD = np.deg2rad(11.5)
R_EARTH = 6.371e6  # m, spherical mean radius
OMEGA_EARTH = 7.292115e-5  # rad/s, inertial rotation about ECI ẑ

# Exponential atmosphere: ~400 km LEO mean, not a full density model.
ATM_H_REF = 400e3  # m
ATM_RHO_REF = 2.5e-12  # kg/m³
ATM_SCALE_HEIGHT = 60e3  # m

# Solar constant / c at 1 AU (N/m²).  No 1/r_sun² scale (sun at infinity).
P_SRP_1AU = 4.56e-6

_RADIUS_EPS = 1e-15
_SPEED_EPS = 1e-15


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


def _nonneg(name: str, value: float) -> float:
    x = float(value)
    if not np.isfinite(x) or x < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return x


def _positive(name: str, value: float) -> float:
    x = float(value)
    if not np.isfinite(x) or x <= 0.0:
        raise ValueError(f"{name} must be positive")
    return x


def exponential_density(
    radius: float,
    *,
    rho_ref: float = ATM_RHO_REF,
    h_ref: float = ATM_H_REF,
    scale_height: float = ATM_SCALE_HEIGHT,
    earth_radius: float = R_EARTH,
) -> float:
    """Exponential atmosphere ``ρ = ρ_ref exp(−(h − h_ref)/H)``, ``h = |r| − R_E``.

    ``radius`` is geocentric radius (m), not altitude.  This is a single
    scale-height LEO stub, not NRLMSISE.
    """
    radius = _positive("orbit radius", radius)
    rho_ref = _nonneg("rho_ref", rho_ref)
    h_ref = float(h_ref)
    scale_height = _positive("scale_height", scale_height)
    earth_radius = _positive("earth_radius", earth_radius)
    if not np.isfinite(h_ref):
        raise ValueError("h_ref must be finite")
    altitude = radius - earth_radius
    return float(rho_ref * np.exp(-(altitude - h_ref) / scale_height))


@dataclass
class ExponentialAtmosphere:
    """Isothermal exponential density vs geocentric radius."""

    rho_ref: float = ATM_RHO_REF
    h_ref: float = ATM_H_REF
    scale_height: float = ATM_SCALE_HEIGHT
    earth_radius: float = R_EARTH

    def __post_init__(self) -> None:
        self.rho_ref = _nonneg("rho_ref", self.rho_ref)
        self.h_ref = float(self.h_ref)
        self.scale_height = _positive("scale_height", self.scale_height)
        self.earth_radius = _positive("earth_radius", self.earth_radius)
        if not np.isfinite(self.h_ref):
            raise ValueError("h_ref must be finite")

    def density(self, radius: float) -> float:
        """``ρ`` (kg/m³) at geocentric radius ``radius`` (m)."""
        return exponential_density(
            radius,
            rho_ref=self.rho_ref,
            h_ref=self.h_ref,
            scale_height=self.scale_height,
            earth_radius=self.earth_radius,
        )


def relative_velocity_eci(
    orbit: OrbitState,
    *,
    co_rotating: bool = False,
    omega_earth: float = OMEGA_EARTH,
) -> np.ndarray:
    """Inertial atmosphere-relative velocity (m/s).

    Default: atmosphere at rest in ECI (``v_rel = v``).  ``co_rotating``
    subtracts ``ω_E × r`` with ``ω_E = omega_earth ẑ``.
    """
    v = _as_vec3("v_eci", orbit.v_eci)
    if not co_rotating:
        return v
    omega_earth = float(omega_earth)
    if not np.isfinite(omega_earth):
        raise ValueError("omega_earth must be finite")
    omega = np.array([0.0, 0.0, omega_earth])
    return v - np.cross(omega, orbit.r_eci)


def aerodynamic_force(
    rho: float,
    v_rel: np.ndarray,
    cd: float,
    area: float,
    n_hat: np.ndarray | None = None,
    *,
    project_area: bool = False,
) -> np.ndarray:
    """Drag ``F = −½ ρ |v|² C_d A n̂`` (same frame as ``v_rel`` / ``n̂``).

    If ``n_hat`` is omitted, ``n̂ = v̂_rel`` (ram / cannonball).  Zero
    relative speed returns a zero force.  ``project_area`` scales ``A``
    by ``max(n̂ · v̂_rel, 0)``.
    """
    rho = _nonneg("rho", rho)
    cd = _nonneg("cd", cd)
    area = _nonneg("area", area)
    v = _as_vec3("v_rel", v_rel)
    speed = float(np.linalg.norm(v))
    if speed < _SPEED_EPS:
        return np.zeros(3)
    v_hat = v / speed
    if n_hat is None:
        n = v_hat
    else:
        n = _unit("n_hat", n_hat)
    area_eff = area
    if project_area:
        area_eff = area * max(float(np.dot(n, v_hat)), 0.0)
        if area_eff == 0.0:
            return np.zeros(3)
    return -0.5 * rho * speed * speed * cd * area_eff * n


def aerodynamic_torque(
    r_cp: np.ndarray,
    rho: float,
    v_rel: np.ndarray,
    cd: float,
    area: float,
    n_hat: np.ndarray | None = None,
    *,
    project_area: bool = False,
) -> np.ndarray:
    """``τ = r_cp × (−½ ρ |v_rel|² C_d A n̂)``.

    ``r_cp`` parallel to the force (including the origin) gives zero torque.
    """
    force = aerodynamic_force(
        rho, v_rel, cd, area, n_hat, project_area=project_area
    )
    return np.cross(_as_vec3("r_cp", r_cp), force)


def in_cylindrical_umbra(
    r_eci: np.ndarray,
    sun_eci: np.ndarray,
    *,
    earth_radius: float = R_EARTH,
) -> bool:
    """Earth cylindrical-shadow stub (sun at infinity; umbra on/off).

    True iff the spacecraft is anti-sun of Earth (``r · û_sun < 0``) and
    the perpendicular distance to the sun line is ``< earth_radius``.
    No penumbra, no finite-sun cone.
    """
    r = _as_vec3("r_eci", r_eci)
    sun = _unit("sun_eci", sun_eci)
    earth_radius = _positive("earth_radius", earth_radius)
    if float(np.dot(r, sun)) >= 0.0:
        return False
    perp = float(np.linalg.norm(np.cross(r, sun)))
    return perp < earth_radius


def srp_force(
    area: float,
    cr: float,
    sun_hat: np.ndarray,
    n_hat: np.ndarray | None = None,
    *,
    p_srp: float = P_SRP_1AU,
    eclipse: bool = False,
) -> np.ndarray:
    """SRP ``F = P_srp c_r A cosθ û_sun`` (same frame as the unit vectors).

    ``n_hat is None`` is the cannonball (``cosθ = 1``).  Otherwise
    ``cosθ = n̂ · û_sun``; ``cosθ ≤ 0`` or ``eclipse=True`` returns zero.
    """
    if eclipse:
        return np.zeros(3)
    area = _nonneg("area", area)
    cr = _nonneg("cr", cr)
    p_srp = _nonneg("p_srp", p_srp)
    u_sun = _unit("sun_hat", sun_hat)
    if n_hat is None:
        cos_theta = 1.0
    else:
        cos_theta = float(np.dot(_unit("n_hat", n_hat), u_sun))
        if cos_theta <= 0.0:
            return np.zeros(3)
    return p_srp * cr * area * cos_theta * u_sun


def srp_torque(
    r_cp: np.ndarray,
    area: float,
    cr: float,
    sun_hat: np.ndarray,
    n_hat: np.ndarray | None = None,
    *,
    p_srp: float = P_SRP_1AU,
    eclipse: bool = False,
) -> np.ndarray:
    """``τ = r_cp × (P_srp c_r A cosθ û_sun)``.

    ``eclipse=True`` (umbra) or ``r_cp ∥ F`` gives zero torque.
    """
    force = srp_force(
        area, cr, sun_hat, n_hat, p_srp=p_srp, eclipse=eclipse
    )
    return np.cross(_as_vec3("r_cp", r_cp), force)


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
class AerodynamicTorque:
    """Panel / box aerodynamic disturbance ``τ = r_cp × (−½ ρ v² C_d A n̂)``.

    Bind a :class:`CircularOrbit` to evaluate from time ``t``.  ``n_hat_body``
    omitted uses ram ``v̂_rel`` in the body frame.  Density comes from
    :class:`ExponentialAtmosphere`.
    """

    r_cp_body: np.ndarray
    area: float
    cd: float = 2.2
    n_hat_body: np.ndarray | None = None
    orbit: CircularOrbit | None = None
    atmosphere: ExponentialAtmosphere = field(default_factory=ExponentialAtmosphere)
    co_rotating: bool = False
    project_area: bool = False
    omega_earth: float = OMEGA_EARTH

    def __post_init__(self) -> None:
        self.r_cp_body = _as_vec3("r_cp_body", self.r_cp_body)
        self.area = _nonneg("area", self.area)
        self.cd = _nonneg("cd", self.cd)
        if self.n_hat_body is not None:
            self.n_hat_body = _unit("n_hat_body", self.n_hat_body)
        if not isinstance(self.atmosphere, ExponentialAtmosphere):
            raise TypeError("atmosphere must be an ExponentialAtmosphere")
        self.co_rotating = bool(self.co_rotating)
        self.project_area = bool(self.project_area)
        self.omega_earth = float(self.omega_earth)
        if not np.isfinite(self.omega_earth):
            raise ValueError("omega_earth must be finite")

    def tau_body(
        self,
        q: np.ndarray,
        omega: np.ndarray,
        t: float | None = None,
        *,
        orbit: OrbitState | None = None,
    ) -> np.ndarray:
        """Body-frame aerodynamic torque.

        ``omega`` is unused (surface velocity ``ω × r_cp`` is neglected).
        """
        _as_vec3("omega", omega)
        state = _resolve_orbit(t, orbit, self.orbit)
        rho = self.atmosphere.density(state.radius)
        v_rel_b = inertial_to_body(
            q,
            relative_velocity_eci(
                state, co_rotating=self.co_rotating, omega_earth=self.omega_earth
            ),
        )
        return aerodynamic_torque(
            self.r_cp_body,
            rho,
            v_rel_b,
            self.cd,
            self.area,
            self.n_hat_body,
            project_area=self.project_area,
        )


@dataclass
class SolarRadiationPressureTorque:
    """SRP disturbance ``τ = r_cp × (P_srp c_r A cosθ û_sun)``.

    ``eclipse`` is an umbra on/off stub: ``True`` / ``'on'`` zeroes the
    torque; ``False`` / ``'off'`` is sunlit.  ``'cylindrical'`` uses
    :func:`in_cylindrical_umbra` with the bound orbit and ``sun_eci``.
    ``n_hat_body is None`` is the cannonball (``cosθ = 1``).
    """

    r_cp_body: np.ndarray
    area: float
    cr: float = 1.0
    n_hat_body: np.ndarray | None = None
    sun_eci: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    orbit: CircularOrbit | None = None
    p_srp: float = P_SRP_1AU
    eclipse: bool | str = False

    def __post_init__(self) -> None:
        self.r_cp_body = _as_vec3("r_cp_body", self.r_cp_body)
        self.area = _nonneg("area", self.area)
        self.cr = _nonneg("cr", self.cr)
        self.p_srp = _nonneg("p_srp", self.p_srp)
        if self.n_hat_body is not None:
            self.n_hat_body = _unit("n_hat_body", self.n_hat_body)
        self.sun_eci = _unit("sun_eci", self.sun_eci)
        if isinstance(self.eclipse, str):
            mode = str(self.eclipse).lower()
            if mode in ("on", "umbra", "true"):
                self.eclipse = True
            elif mode in ("off", "sunlit", "false"):
                self.eclipse = False
            elif mode != "cylindrical":
                raise ValueError(
                    "eclipse must be bool, 'on'/'off', or 'cylindrical'"
                )
            else:
                self.eclipse = "cylindrical"
        elif not isinstance(self.eclipse, (bool, np.bool_)):
            raise TypeError("eclipse must be bool or str")
        else:
            self.eclipse = bool(self.eclipse)

    def in_umbra(self, orbit: OrbitState) -> bool:
        """Umbra flag: stored stub, or cylindrical Earth shadow at ``orbit``."""
        if self.eclipse == "cylindrical":
            return in_cylindrical_umbra(orbit.r_eci, self.sun_eci)
        return bool(self.eclipse)

    def tau_body(
        self,
        q: np.ndarray,
        omega: np.ndarray,
        t: float | None = None,
        *,
        orbit: OrbitState | None = None,
    ) -> np.ndarray:
        """Body-frame SRP torque.

        ``omega`` is unused.  Cylindrical eclipse needs an orbit state.
        """
        _as_vec3("omega", omega)
        if self.eclipse == "cylindrical":
            state = _resolve_orbit(t, orbit, self.orbit)
            eclipsed = in_cylindrical_umbra(state.r_eci, self.sun_eci)
        else:
            eclipsed = bool(self.eclipse)
        sun_b = inertial_to_body(q, self.sun_eci)
        return srp_torque(
            self.r_cp_body,
            self.area,
            self.cr,
            sun_b,
            self.n_hat_body,
            p_srp=self.p_srp,
            eclipse=eclipsed,
        )


class DisturbanceTorque(Protocol):
    """Duck-typed body-torque model: ``τ_body(q, ω, t, *, orbit)``."""

    def tau_body(
        self,
        q: np.ndarray,
        omega: np.ndarray,
        t: float | None = None,
        *,
        orbit: OrbitState | None = None,
    ) -> np.ndarray: ...


@dataclass
class EnvironmentalTorques:
    """Sum of gravity-gradient, residual-dipole, aero, and/or SRP body torques."""

    gravity_gradient: GravityGradientTorque | None = None
    residual_dipole: ResidualDipoleTorque | None = None
    aerodynamic: AerodynamicTorque | None = None
    srp: SolarRadiationPressureTorque | None = None
    extra: Sequence[DisturbanceTorque] = field(default_factory=tuple)

    def tau_body(
        self,
        q: np.ndarray,
        omega: np.ndarray,
        t: float | None = None,
        *,
        orbit: OrbitState | None = None,
    ) -> np.ndarray:
        tau = np.zeros(3, dtype=float)
        for model in (
            self.gravity_gradient,
            self.residual_dipole,
            self.aerodynamic,
            self.srp,
            *self.extra,
        ):
            if model is None:
                continue
            tau = tau + np.asarray(
                model.tau_body(q, omega, t, orbit=orbit), dtype=float
            ).reshape(3)
        return tau

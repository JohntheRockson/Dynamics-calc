"""Fuel-slosh / spherical-pendulum disturbance plant (one slosh mode).

This is a **library** extension of ``attitude_sim.plant``.  It does not
change ``RigidBody``, the SimLab integrator, controllers, estimators,
``attitude_sim.disturbances`` (GG / dipole / aero / SRP stay untouched),
``attitude_sim.polhode``, ``attitude_sim.gyrostat``, or any open flex /
hinged-appendage slice.

Spherical-pendulum equivalent-mechanical model
---------------------------------------------
A rigid hub with body-frame inertia ``J = Jᵀ ≻ 0`` (dry / non-sloshing
inertia about the attitude reference ``O``) carries **one** slosh mass
``m ≥ 0`` on a massless rod of length ``ℓ > 0``.  The hinge is at body
point ``r_h``.  This is the classical spherical-pendulum tank analogue
(Abramson / Dodge) reduced to **planar** motion in a body-fixed plane —
one slosh mode, one pendulum angle ``θ``.  The second spherical angle is
omitted (a second orthogonal pendulum would complete the 2-DOF spherical
tank model).

Geometry.  ``û_0`` is the unit rest direction (hinge → bob at ``θ = 0``).
``ê_s`` is the unit side axis in the swing plane (``û_θ`` at rest).
The plane normal is ``ê_n = û_0 × ê_s``.  The bob unit vector and
position are

    û(θ) = cosθ û_0 + sinθ ê_s
    r    = r_h + ℓ û
    ṙ    = ℓ θ̇ û_θ,     û_θ = −sinθ û_0 + cosθ ê_s.

Effective gravity ``g_eff ≥ 0`` is a **body-fixed** tank / thrust
acceleration along ``û_0`` (``g⃗ = g_eff û_0``).  The bob hangs along
``û_0`` at rest.  This is **not** the zero-g spring-mass slosh form:
small-angle restoring is ``θ̈ + (g_eff / ℓ) θ = 0`` on a fixed hub, so

    ω_n = √(g_eff / ℓ).

An optional hinge torsional spring ``k ≥ 0`` is the documented zero-g
equivalent (``g_eff = 0`` ⇒ ``ω_n = √(k / (m ℓ²))``).  Default ``k = 0``.
Viscous hinge damping ``c ≥ 0`` contributes the Rayleigh term
``R = (1/2) c θ̇²``.

Coupled Newton–Euler / Lagrangian form
--------------------------------------
Kinetic energy, effective potential, and dissipation:

    T = (1/2) ω · (J ω) + (1/2) m |ω × r + ℓ θ̇ û_θ|²
    V = −m g⃗ · r + (1/2) k θ²
    R = (1/2) c θ̇².

Body-frame composite angular momentum about ``O`` is

    H = J ω + m r × v,     v = ω × r + ℓ θ̇ û_θ
      = (J + J_p) ω + m ℓ θ̇ (r × û_θ)

with the point-mass inertia ``J_p = m (|r|² I − r rᵀ)``.  Euler on the
composite system (ideal rod forces do no torque about ``O``) is

    Ḣ + ω × H = τ + m r × g⃗

and the pendulum Lagrange / virtual-work equation (project bob
acceleration on ``û_θ``) is

    m ℓ (r × û_θ) · ω̇ + m ℓ² θ̈
        = m ℓ g⃗ · û_θ − m ℓ [ω × (ω × r)] · û_θ − k θ − c θ̇.

Together these are the symmetric mass-matrix system in
``(ω̇, θ̈)``.  The **slosh reaction torque on the hub** is the extra
term in Euler's equation on ``J`` alone

    J ω̇ + ω × (J ω) = τ + τ_slosh
    τ_slosh = J ω̇ + ω × (J ω) − τ.

Reduction to rigid-body Euler
-----------------------------
* ``m → 0``: ``τ_slosh = 0`` and ``ω̇`` is exactly
  :class:`~attitude_sim.plant.RigidBody` on ``J``.
* Frozen pendulum (``θ̈ = 0``, ``θ̇`` treated as 0) at rest ``θ = 0``:
  ``H = J_eq ω`` with locked inertia

      J_eq = J + m (|r_0|² I − r_0 r_0ᵀ),     r_0 = r_h + ℓ û_0

  and, when ``r_h × û_0 = 0`` (default hinge at ``O``), ``r_0 × g⃗ = 0``
  so ``ω̇`` matches ``RigidBody`` on ``J_eq``.

Energy
------
``E = T + V`` satisfies ``Ė = ω · τ − c θ̇²`` for this body-fixed
``g⃗``.  Torque-free undamped motion conserves ``E`` (up to RK4
truncation).  Damping dissipates energy.

RK4 step
--------
:func:`step_slosh` integrates ``y = [ω, θ, θ̇]``.
:func:`step_slosh_attitude` also integrates ``q̇ = (1/2) q ⊗ ω̂`` and
renormalizes ``q`` after the Euclidean step.  Both reuse
:func:`attitude_sim.plant.rk4_step` with zero-order-hold ``τ``.  Not a
SimLab CLI flag.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from attitude_sim.plant import RigidBody, rk4_step, validate_inertia
from attitude_sim.quaternions import quat_derivative, quat_normalize, quat_to_rotation

_AXIS_NORM_EPS = 1e-15
_MASS_ZERO = 1e-15


def _as_omega(omega: np.ndarray) -> np.ndarray:
    w = np.asarray(omega, dtype=float).reshape(3)
    if not np.all(np.isfinite(w)):
        raise ValueError("omega must be finite")
    return w


def _as_tau(tau: np.ndarray) -> np.ndarray:
    t = np.asarray(tau, dtype=float).reshape(3)
    if not np.all(np.isfinite(t)):
        raise ValueError("tau must be finite")
    return t


def _as_vector3(name: str, value: np.ndarray) -> np.ndarray:
    v = np.asarray(value, dtype=float).reshape(3)
    if not np.all(np.isfinite(v)):
        raise ValueError(f"{name} must be finite")
    return v


def _as_scalar(name: str, value: float, *, min_value: float | None = None) -> float:
    x = float(value)
    if not np.isfinite(x):
        raise ValueError(f"{name} must be finite")
    if min_value is not None and x < min_value:
        raise ValueError(f"{name} must be >= {min_value}")
    return x


def unit_axis(axis: np.ndarray | int | str, *, name: str = "axis") -> np.ndarray:
    """Return a unit body-frame axis.

    ``axis`` is a length-3 vector, a principal-index ``0/1/2``, or a
    name ``\"x\"`` / ``\"y\"`` / ``\"z\"``.
    """
    if isinstance(axis, str):
        key = axis.strip().lower()
        named = {"x": 0, "y": 1, "z": 2}
        if key not in named:
            raise ValueError(f"{name} name must be 'x', 'y', or 'z'")
        axis = named[key]
    if isinstance(axis, (int, np.integer)):
        k = int(axis)
        if k not in (0, 1, 2):
            raise ValueError(f"{name} index must be 0, 1, or 2")
        e = np.zeros(3)
        e[k] = 1.0
        return e
    a = np.asarray(axis, dtype=float).reshape(3)
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be finite")
    n = float(np.linalg.norm(a))
    if n <= _AXIS_NORM_EPS:
        raise ValueError(f"{name} must be non-zero")
    return a / n


def pendulum_frame(
    rest_axis: np.ndarray | int | str = "z",
    swing_axis: np.ndarray | int | str = "x",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Orthonormal pendulum frame ``(û_0, ê_s, ê_n)``.

    ``rest_axis`` is the hanging / rest direction ``û_0``.  ``swing_axis``
    is orthonormalized in the plane to give ``ê_s`` (``û_θ`` at ``θ = 0``).
    ``ê_n = û_0 × ê_s`` is the plane normal.  ``rest_axis=\"-z\"`` is
    accepted as the conventional downward hang.
    """
    if isinstance(rest_axis, str) and rest_axis.strip().lower() in {"-z", "-x", "-y"}:
        key = rest_axis.strip().lower()
        sign_name = {"-x": 0, "-y": 1, "-z": 2}[key]
        u0 = np.zeros(3)
        u0[sign_name] = -1.0
    else:
        u0 = unit_axis(rest_axis, name="rest_axis")
    s_raw = unit_axis(swing_axis, name="swing_axis")
    s_proj = s_raw - (s_raw @ u0) * u0
    n_s = float(np.linalg.norm(s_proj))
    if n_s <= _AXIS_NORM_EPS:
        raise ValueError("swing_axis must not be parallel to rest_axis")
    es = s_proj / n_s
    en = np.cross(u0, es)
    en = en / float(np.linalg.norm(en))
    return u0, es, en


def slosh_natural_frequency(
    g_eff: float,
    length: float,
    *,
    mass: float = 1.0,
    stiffness: float = 0.0,
) -> float:
    """Fixed-hub small-angle slosh frequency.

    Pendulum (default): ``ω_n = √(g_eff / ℓ)``.  Optional hinge spring
    (zero-g form when ``g_eff = 0``): ``ω_n = √(g_eff / ℓ + k / (m ℓ²))``.
    """
    g = _as_scalar("g_eff", g_eff, min_value=0.0)
    ell = _as_scalar("length", length, min_value=0.0)
    if ell <= _AXIS_NORM_EPS:
        raise ValueError("length must be positive for a natural frequency")
    k = _as_scalar("stiffness", stiffness, min_value=0.0)
    m = _as_scalar("mass", mass, min_value=0.0)
    extra = 0.0
    if k > 0.0:
        if m <= _MASS_ZERO:
            raise ValueError("mass must be positive when stiffness > 0")
        extra = k / (m * ell * ell)
    wn2 = g / ell + extra
    if wn2 < 0.0:
        raise ValueError("squared natural frequency must be non-negative")
    return float(np.sqrt(wn2))


@dataclass
class SloshPendulum:
    """Rigid hub plus one planar (spherical-slice) slosh pendulum.

    ``inertia`` is the dry-hub tensor ``J`` (validated, SPD, triangle
    inequalities).  ``mass`` / ``length`` are the slosh mass and rod
    length.  ``g_eff`` is the body-fixed effective gravity along
    ``rest_axis``.  ``damping`` / ``stiffness`` are optional hinge ``c``
    and ``k``.
    """

    inertia: np.ndarray
    mass: float = 0.05
    length: float = 0.25
    g_eff: float = 1.0
    damping: float = 0.0
    stiffness: float = 0.0
    hinge: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rest_axis: np.ndarray | int | str = "-z"
    swing_axis: np.ndarray | int | str = "x"
    _j_inv: np.ndarray = field(init=False, repr=False, compare=False)
    _u0: np.ndarray = field(init=False, repr=False, compare=False)
    _es: np.ndarray = field(init=False, repr=False, compare=False)
    _en: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        J = validate_inertia(self.inertia)
        self.inertia = J
        self._j_inv = np.linalg.inv(J)
        self.mass = _as_scalar("mass", self.mass, min_value=0.0)
        self.length = _as_scalar("length", self.length, min_value=0.0)
        if self.mass > _MASS_ZERO and self.length <= _AXIS_NORM_EPS:
            raise ValueError("length must be positive when mass > 0")
        self.g_eff = _as_scalar("g_eff", self.g_eff, min_value=0.0)
        self.damping = _as_scalar("damping", self.damping, min_value=0.0)
        self.stiffness = _as_scalar("stiffness", self.stiffness, min_value=0.0)
        self.hinge = _as_vector3("hinge", self.hinge)
        self._u0, self._es, self._en = pendulum_frame(self.rest_axis, self.swing_axis)
        self.rest_axis = self._u0
        self.swing_axis = self._es

    @property
    def inertia_inv(self) -> np.ndarray:
        """Cached ``J⁻¹``."""
        return self._j_inv

    @property
    def rest_direction(self) -> np.ndarray:
        """Unit rest / hanging direction ``û_0``."""
        return self._u0.copy()

    @property
    def plane_normal(self) -> np.ndarray:
        """Unit swing-plane normal ``ê_n = û_0 × ê_s``."""
        return self._en.copy()

    @property
    def has_slosh(self) -> bool:
        """True when the slosh mass is strictly positive."""
        return self.mass > _MASS_ZERO

    @property
    def g_vector(self) -> np.ndarray:
        """Body-fixed effective gravity ``g⃗ = g_eff û_0``."""
        return self.g_eff * self._u0

    def bob_unit(self, theta: float) -> np.ndarray:
        """Pendulum unit vector ``û(θ)`` (hinge → bob)."""
        th = _as_scalar("theta", theta)
        return np.cos(th) * self._u0 + np.sin(th) * self._es

    def bob_unit_theta(self, theta: float) -> np.ndarray:
        """``û_θ = ∂û/∂θ``."""
        th = _as_scalar("theta", theta)
        return -np.sin(th) * self._u0 + np.cos(th) * self._es

    def bob_position(self, theta: float) -> np.ndarray:
        """Body-frame bob position ``r = r_h + ℓ û``."""
        return self.hinge + self.length * self.bob_unit(theta)

    def bob_velocity(self, omega: np.ndarray, theta: float, theta_dot: float) -> np.ndarray:
        """Body components of the inertial bob velocity ``ω × r + ℓ θ̇ û_θ``."""
        w = _as_omega(omega)
        r = self.bob_position(theta)
        td = _as_scalar("theta_dot", theta_dot)
        return np.cross(w, r) + self.length * td * self.bob_unit_theta(theta)

    def point_mass_inertia(self, theta: float) -> np.ndarray:
        """``J_p = m (|r|² I − r rᵀ)`` at the current angle."""
        r = self.bob_position(theta)
        return self.mass * (float(r @ r) * np.eye(3) - np.outer(r, r))

    def locked_inertia(self, theta: float = 0.0) -> np.ndarray:
        """Frozen-pendulum inertia ``J + J_p(θ)``."""
        return self.inertia + self.point_mass_inertia(theta)

    def natural_frequency(self) -> float:
        """Fixed-hub small-angle ``ω_n = √(g_eff / ℓ + k / (m ℓ²))``."""
        if self.mass <= _MASS_ZERO and self.stiffness > 0.0:
            raise ValueError("mass must be positive when stiffness > 0")
        mass = 1.0 if self.mass <= _MASS_ZERO else self.mass
        return slosh_natural_frequency(
            self.g_eff, self.length, mass=mass, stiffness=self.stiffness
        )

    def angular_momentum_body(
        self,
        omega: np.ndarray,
        theta: float,
        theta_dot: float,
    ) -> np.ndarray:
        """Body-frame ``H = J ω + m r × v``."""
        w = _as_omega(omega)
        r = self.bob_position(theta)
        v = self.bob_velocity(w, theta, theta_dot)
        return self.inertia @ w + self.mass * np.cross(r, v)

    def angular_momentum_inertial(
        self,
        q: np.ndarray,
        omega: np.ndarray,
        theta: float,
        theta_dot: float,
    ) -> np.ndarray:
        """Inertial-frame ``H_I = R(q) H``."""
        return quat_to_rotation(q) @ self.angular_momentum_body(omega, theta, theta_dot)

    def kinetic_energy(self, omega: np.ndarray, theta: float, theta_dot: float) -> float:
        """``T = (1/2) ωᵀ J ω + (1/2) m |v|²``."""
        w = _as_omega(omega)
        v = self.bob_velocity(w, theta, theta_dot)
        t_hub = 0.5 * float(w @ (self.inertia @ w))
        return t_hub + 0.5 * self.mass * float(v @ v)

    def potential_energy(self, theta: float) -> float:
        """``V = −m g⃗ · r + (1/2) k θ²``."""
        th = _as_scalar("theta", theta)
        r = self.bob_position(th)
        return float(-self.mass * self.g_vector @ r + 0.5 * self.stiffness * th * th)

    def mechanical_energy(self, omega: np.ndarray, theta: float, theta_dot: float) -> float:
        """``E = T + V``."""
        return self.kinetic_energy(omega, theta, theta_dot) + self.potential_energy(theta)

    def _assemble(self, omega: np.ndarray, theta: float, theta_dot: float, tau: np.ndarray) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        float,
    ]:
        """Return mass matrix, Euler/θ right-hand sides, ``r``, ``û_θ``, ``θ̇``."""
        w = _as_omega(omega)
        tau_b = _as_tau(tau)
        th = _as_scalar("theta", theta)
        td = _as_scalar("theta_dot", theta_dot)
        r = self.bob_position(th)
        u_th = self.bob_unit_theta(th)
        u = self.bob_unit(th)
        ell = self.length
        m = self.mass
        g_vec = self.g_vector
        coupling = m * ell * np.cross(r, u_th)
        jp = self.point_mass_inertia(th)
        mass_w = self.inertia + jp
        r_dot = ell * td * u_th
        # J̇_p ω = m (2 (r·ṙ) ω − ṙ (r·ω) − r (ṙ·ω))
        jp_dot_w = m * (
            2.0 * float(r @ r_dot) * w - r_dot * float(r @ w) - r * float(r_dot @ w)
        )
        h = self.angular_momentum_body(w, th, td)
        euler_rhs = (
            tau_b
            + m * np.cross(r, g_vec)
            - np.cross(w, h)
            - jp_dot_w
            + m * ell * td * td * np.cross(r, u)
        )
        centripetal = float(u_th @ np.cross(w, np.cross(w, r)))
        theta_rhs = (
            m * ell * float(g_vec @ u_th)
            - m * ell * centripetal
            - self.stiffness * th
            - self.damping * td
        )
        m_mat = np.zeros((4, 4))
        m_mat[:3, :3] = mass_w
        m_mat[:3, 3] = coupling
        m_mat[3, :3] = coupling
        m_mat[3, 3] = m * ell * ell
        rhs = np.zeros(4)
        rhs[:3] = euler_rhs
        rhs[3] = theta_rhs
        return m_mat, rhs, r, u_th, td

    def derivatives(
        self,
        omega: np.ndarray,
        tau: np.ndarray,
        theta: float,
        theta_dot: float,
        *,
        frozen: bool = False,
        fixed_hub: bool = False,
    ) -> tuple[np.ndarray, float, float]:
        """Vector field ``(ω̇, θ̇, θ̈)``.

        ``frozen=True`` locks the pendulum at the current ``θ`` (relative
        rate treated as zero) and returns rigid ``ω̇`` on ``J_eq``.
        ``fixed_hub=True`` holds ``ω̇ = 0`` and integrates only ``θ``
        (modal / √(g_eff/ℓ) checkout).  ``m = 0`` is Euler on ``J``
        with ``θ̈ = 0``.
        """
        w = _as_omega(omega)
        tau_b = _as_tau(tau)
        if not self.has_slosh:
            wdot = self._j_inv @ (tau_b - np.cross(w, self.inertia @ w))
            return wdot, 0.0, 0.0
        if frozen:
            th = _as_scalar("theta", theta)
            body = RigidBody(self.locked_inertia(th))
            r0 = self.bob_position(th)
            tau_g = self.mass * np.cross(r0, self.g_vector)
            wdot = body.omega_dot(w, tau_b + tau_g)
            return wdot, 0.0, 0.0
        td = _as_scalar("theta_dot", theta_dot)
        m_mat, rhs, _r, _u_th, td = self._assemble(w, theta, td, tau_b)
        if fixed_hub:
            # Only the θ row: m ℓ² θ̈ = rhs_θ − coupling · ω̇, with ω̇ = 0.
            i_th = float(m_mat[3, 3])
            tddot = 0.0 if abs(i_th) <= _MASS_ZERO else float(rhs[3] / i_th)
            return np.zeros(3), td, tddot
        acc = np.linalg.solve(m_mat, rhs)
        return acc[:3].copy(), td, float(acc[3])

    def omega_dot(
        self,
        omega: np.ndarray,
        tau: np.ndarray,
        theta: float,
        theta_dot: float,
        *,
        frozen: bool = False,
    ) -> np.ndarray:
        """Hub ``ω̇`` at the current slosh state."""
        wdot, _, _ = self.derivatives(omega, tau, theta, theta_dot, frozen=frozen)
        return wdot

    def theta_ddot(
        self,
        omega: np.ndarray,
        tau: np.ndarray,
        theta: float,
        theta_dot: float,
        *,
        frozen: bool = False,
        fixed_hub: bool = False,
    ) -> float:
        """Pendulum acceleration ``θ̈``."""
        _, _, tddot = self.derivatives(
            omega, tau, theta, theta_dot, frozen=frozen, fixed_hub=fixed_hub
        )
        return tddot

    def slosh_reaction_torque(
        self,
        omega: np.ndarray,
        tau: np.ndarray,
        theta: float,
        theta_dot: float,
        *,
        frozen: bool = False,
    ) -> np.ndarray:
        """Torque on the hub from the slosh reaction.

            τ_slosh = J ω̇ + ω × (J ω) − τ

        so that Euler on the dry hub recovers ``ω̇``.  ``m = 0`` or a
        frozen rest state with ``r_h ∥ û_0`` gives ``τ_slosh → 0``
        relative to the matching rigid inertia (dry ``J`` or ``J_eq``).
        """
        w = _as_omega(omega)
        tau_b = _as_tau(tau)
        wdot = self.omega_dot(w, tau_b, theta, theta_dot, frozen=frozen)
        return self.inertia @ wdot + np.cross(w, self.inertia @ w) - tau_b

    def as_rigid_body(self, theta: float = 0.0, *, locked: bool = False) -> RigidBody:
        """Dry-hub ``RigidBody``, or locked ``J_eq`` when ``locked=True``."""
        if locked:
            return RigidBody(self.locked_inertia(theta))
        return RigidBody(self.inertia)


def pack_slosh_state(omega: np.ndarray, theta: float, theta_dot: float) -> np.ndarray:
    """Pack ``y = [ω, θ, θ̇]`` as a length-5 array."""
    w = _as_omega(omega)
    th = _as_scalar("theta", theta)
    td = _as_scalar("theta_dot", theta_dot)
    return np.array([w[0], w[1], w[2], th, td], dtype=float)


def unpack_slosh_state(y: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Unpack a length-5 slosh state into ``(ω, θ, θ̇)``."""
    y = np.asarray(y, dtype=float).reshape(-1)
    if y.shape != (5,):
        raise ValueError("slosh state must have length 5 (ω, θ, θ̇)")
    return y[:3].copy(), float(y[3]), float(y[4])


def slosh_omega_dot(
    plant: SloshPendulum,
    omega: np.ndarray,
    tau: np.ndarray,
    theta: float,
    theta_dot: float,
    *,
    frozen: bool = False,
) -> np.ndarray:
    """Module-level hub ``ω̇``."""
    return plant.omega_dot(omega, tau, theta, theta_dot, frozen=frozen)


def slosh_reaction_torque(
    plant: SloshPendulum,
    omega: np.ndarray,
    tau: np.ndarray,
    theta: float,
    theta_dot: float,
    *,
    frozen: bool = False,
) -> np.ndarray:
    """Module-level slosh reaction torque on the hub."""
    return plant.slosh_reaction_torque(omega, tau, theta, theta_dot, frozen=frozen)


def step_slosh(
    plant: SloshPendulum,
    omega: np.ndarray,
    theta: float,
    theta_dot: float,
    tau: np.ndarray,
    dt: float,
    *,
    frozen: bool = False,
    fixed_hub: bool = False,
) -> tuple[np.ndarray, float, float]:
    """Classical RK4 step of ``[ω, θ, θ̇]`` with ZOH hub torque.

    Returns ``(ω⁺, θ⁺, θ̇⁺)``.
    """
    tau_b = _as_tau(tau)

    def fun(_t: float, y: np.ndarray) -> np.ndarray:
        w, th, td = unpack_slosh_state(y)
        wdot, thdot, tddot = plant.derivatives(
            w, tau_b, th, td, frozen=frozen, fixed_hub=fixed_hub
        )
        return pack_slosh_state(wdot, thdot, tddot)

    y = rk4_step(fun, 0.0, pack_slosh_state(omega, theta, theta_dot), dt)
    return unpack_slosh_state(y)


def step_slosh_attitude(
    plant: SloshPendulum,
    q: np.ndarray,
    omega: np.ndarray,
    theta: float,
    theta_dot: float,
    tau: np.ndarray,
    dt: float,
    *,
    frozen: bool = False,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """RK4 step of ``[q, ω, θ, θ̇]`` with ZOH hub torque.

    Quaternion kinematics match :func:`attitude_sim.plant.step_rigid_body`
    (``q̇ = (1/2) q ⊗ ω̂``, then ``q ← q / ||q||``).  Returns
    ``(q⁺, ω⁺, θ⁺, θ̇⁺)``.
    """
    tau_b = _as_tau(tau)
    q0 = quat_normalize(q)
    y0 = np.concatenate([q0, pack_slosh_state(omega, theta, theta_dot)])

    def fun(_t: float, y: np.ndarray) -> np.ndarray:
        q_k = y[:4]
        w, th, td = unpack_slosh_state(y[4:])
        qdot = quat_derivative(q_k, w)
        wdot, thdot, tddot = plant.derivatives(w, tau_b, th, td, frozen=frozen)
        return np.concatenate([qdot, pack_slosh_state(wdot, thdot, tddot)])

    y = rk4_step(fun, 0.0, y0, dt)
    q_next = quat_normalize(y[:4])
    w, th, td = unpack_slosh_state(y[4:])
    return q_next, w, th, td

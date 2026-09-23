"""Hinged rigid appendage / 1-DOF flexural plant (hub + one panel).

This is a **library** extension of ``attitude_sim.plant``.  It does not
change ``RigidBody``, the SimLab integrator, controllers, estimators,
``attitude_sim.disturbances`` (aero / SRP stay untouched), or the open
polhode / gyrostat plant slices.

Model
-----
A rigid hub with body-frame inertia ``J_h = J_hᵀ ≻ 0`` carries **one**
hinged rigid panel.  The hinge axis ``â`` is fixed in the hub.  The
small-angle flexure is the scalar hinge angle ``θ`` (panel relative to
the undeformed / locked pose).  Hinge stiffness ``k ≥ 0`` and viscous
damping ``c ≥ 0`` act on ``θ``.  The panel's inertia about the hinge is
the scalar ``I_p ≥ 0``.  ``I_p = 0`` is the no-appendage degeneracy
(hub-only Euler).

The hinge is taken at the hub CM (minimal two-body model).  Finite-angle
panel DCMs, translation of the system CM, and additional flexural modes
are out of scope.

Lagrangian (planar / hinge-axis)
--------------------------------
Let ``ψ`` be the hub angle about ``â``, ``ω_a = ψ̇ = ω · â``, and
``I_h = 1 / (âᵀ J_h⁻¹ â)`` the hub inertia seen by a torque along
``â`` (equals ``âᵀ J_h â`` when ``â`` is a principal axis).  Kinetic
energy, hinge potential, and Rayleigh dissipation are

    T = (1/2) I_h ψ̇² + (1/2) I_p (ψ̇ + θ̇)²
    V = (1/2) k θ²
    R = (1/2) c θ̇²

Lagrange's equations with generalized force ``τ_a = τ · â`` on ``ψ``
give the coupled mass-matrix form

    [ I_h + I_p    I_p ] [ ψ̈ ]   [ τ_a       ]
    [ I_p          I_p ] [ θ̈ ] = [ −k θ − c θ̇ ].

The locked (θ ≡ 0) inertia about ``â`` is ``I_h + I_p``.  The free–free
flexural reduced inertia and undamped modal frequency are

    I_eff = I_h I_p / (I_h + I_p)     (I_p > 0)
    ω_n   = √(k / I_eff).

``I_eff⁻¹ = âᵀ J_h⁻¹ â + 1/I_p`` is the same number in 3-D (hub
compliance along ``â`` plus the panel).  ``ω_n = 0`` is the rigid
rotation about ``â``.

Newton–Euler (3-D hub)
----------------------
Total body-frame angular momentum (hinge at the hub CM) is

    H = J_h ω + I_p (ω · â + θ̇) â.

Euler on the system and the panel hinge moment balance are

    Ḣ + ω × H = τ
    I_p (â · ω̇ + θ̈) = −k θ − c θ̇.

Eliminating the internal hinge torque yields the hub form used here

    J_h ω̇ = τ + (k θ + c θ̇) â − ω × H
    θ̈     = −â · ω̇ − (k θ + c θ̇) / I_p    (I_p > 0).

At ``θ = θ̇ = 0`` and ``ω = 0``, ``ω̇ = J_h⁻¹ τ`` matches
:class:`~attitude_sim.plant.RigidBody` on the hub.  Torque-free principal
spin about an axis orthogonal to ``â`` with ``θ = θ̇ = 0`` stays there
and follows the rigid hub.  ``I_p = 0`` is identically Euler on ``J_h``.

Energy
------
Mechanical energy ``E = T + V`` with

    T = (1/2) ω · (J_h ω) + (1/2) I_p (ω · â + θ̇)²

satisfies ``Ė = ω · τ − c θ̇²``.  Torque-free undamped motion conserves
``E`` (up to RK4 truncation).  Damping dissipates energy.

Linearization about θ = 0
-------------------------
Rest equilibrium ``ω = 0``, ``θ = 0``, ``θ̇ = 0``.  State
``x = [ω(3), θ, θ̇] ∈ R⁵``, input ``u = τ``.  Gyroscopic terms drop.
The flexural pair has eigenvalues ``± j ω_n`` when ``c = 0`` (plus a
rigid zero from free rotation about ``â``).  See
:meth:`HingedAppendage.linearized_state_space`.

RK4 step
--------
:func:`step_flex` integrates ``y = [ω, θ, θ̇]``.
:func:`step_flex_attitude` also integrates ``q̇ = (1/2) q ⊗ ω̂`` and
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
_IP_ZERO = 1e-15


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


def _as_scalar(name: str, value: float, *, min_value: float | None = None) -> float:
    x = float(value)
    if not np.isfinite(x):
        raise ValueError(f"{name} must be finite")
    if min_value is not None and x < min_value:
        raise ValueError(f"{name} must be >= {min_value}")
    return x


def unit_hinge_axis(axis: np.ndarray | int | str) -> np.ndarray:
    """Return a unit body-frame hinge axis.

    ``axis`` is a length-3 vector, a principal-index ``0/1/2``, or a
    name ``\"x\"`` / ``\"y\"`` / ``\"z\"``.
    """
    if isinstance(axis, str):
        key = axis.strip().lower()
        named = {"x": 0, "y": 1, "z": 2}
        if key not in named:
            raise ValueError("hinge axis name must be 'x', 'y', or 'z'")
        axis = named[key]
    if isinstance(axis, (int, np.integer)):
        k = int(axis)
        if k not in (0, 1, 2):
            raise ValueError("hinge axis index must be 0, 1, or 2")
        e = np.zeros(3)
        e[k] = 1.0
        return e
    a = np.asarray(axis, dtype=float).reshape(3)
    if not np.all(np.isfinite(a)):
        raise ValueError("hinge axis must be finite")
    n = float(np.linalg.norm(a))
    if n <= _AXIS_NORM_EPS:
        raise ValueError("hinge axis must be non-zero")
    return a / n


def effective_hinge_inertia(
    inertia: np.ndarray,
    panel_inertia: float,
    axis: np.ndarray | int | str = "z",
) -> float:
    """Reduced flexural inertia ``I_eff`` about the hinge.

        I_eff⁻¹ = âᵀ J_h⁻¹ â + 1/I_p

    which is ``I_h I_p / (I_h + I_p)`` when ``â`` is principal
    (``I_h = âᵀ J_h â``).  ``panel_inertia`` must be strictly positive.
    """
    J = validate_inertia(inertia)
    i_p = _as_scalar("panel_inertia", panel_inertia, min_value=0.0)
    if i_p <= _IP_ZERO:
        raise ValueError("panel_inertia must be positive for I_eff")
    a = unit_hinge_axis(axis)
    hub_compl = float(a @ np.linalg.solve(J, a))
    if hub_compl <= 0.0:
        raise ValueError("hub compliance along the hinge must be positive")
    return 1.0 / (hub_compl + 1.0 / i_p)


def flex_modal_frequency(
    stiffness: float,
    inertia: np.ndarray,
    panel_inertia: float,
    axis: np.ndarray | int | str = "z",
) -> float:
    """Undamped free–free hinge frequency ``ω_n = √(k / I_eff)``."""
    k = _as_scalar("stiffness", stiffness, min_value=0.0)
    i_eff = effective_hinge_inertia(inertia, panel_inertia, axis)
    return float(np.sqrt(k / i_eff))


@dataclass
class HingedAppendage:
    """Rigid hub plus one hinged rigid panel (small-angle 1-DOF flexure).

    ``inertia`` is the hub tensor ``J_h`` (validated, SPD, triangle
    inequalities).  ``hinge_axis`` is stored as a unit vector.  Panel
    inertia about the hinge is ``panel_inertia``; ``stiffness`` / ``damping``
    are the hinge ``k`` and ``c``.
    """

    inertia: np.ndarray
    hinge_axis: np.ndarray | int | str = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    panel_inertia: float = 0.02
    stiffness: float = 1.0
    damping: float = 0.0
    _j_inv: np.ndarray = field(init=False, repr=False, compare=False)
    _axis: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        J = validate_inertia(self.inertia)
        self.inertia = J
        self._j_inv = np.linalg.inv(J)
        self._axis = unit_hinge_axis(self.hinge_axis)
        self.hinge_axis = self._axis
        self.panel_inertia = _as_scalar("panel_inertia", self.panel_inertia, min_value=0.0)
        self.stiffness = _as_scalar("stiffness", self.stiffness, min_value=0.0)
        self.damping = _as_scalar("damping", self.damping, min_value=0.0)

    @property
    def inertia_inv(self) -> np.ndarray:
        """Cached ``J_h⁻¹``."""
        return self._j_inv

    @property
    def axis(self) -> np.ndarray:
        """Unit hinge axis in the hub body frame."""
        return self._axis.copy()

    @property
    def has_panel(self) -> bool:
        """True when the panel inertia is strictly positive."""
        return self.panel_inertia > _IP_ZERO

    def hub_inertia_about_hinge(self) -> float:
        """Hub inertia seen by a hinge-axis torque, ``1 / (âᵀ J_h⁻¹ â)``."""
        return 1.0 / float(self._axis @ (self._j_inv @ self._axis))

    def effective_inertia(self) -> float:
        """Flexural reduced inertia ``I_eff`` (requires ``I_p > 0``)."""
        return effective_hinge_inertia(self.inertia, self.panel_inertia, self._axis)

    def modal_frequency(self) -> float:
        """Undamped ``ω_n = √(k / I_eff)``."""
        return flex_modal_frequency(
            self.stiffness, self.inertia, self.panel_inertia, self._axis
        )

    def hinge_rate(self, omega: np.ndarray, theta_dot: float) -> float:
        """Inertial panel rate about ``â``: ``ω · â + θ̇``."""
        w = _as_omega(omega)
        td = _as_scalar("theta_dot", theta_dot)
        return float(w @ self._axis) + td

    def hinge_torque(self, theta: float, theta_dot: float) -> float:
        """Internal restoring hinge torque ``k θ + c θ̇`` (on the hub)."""
        th = _as_scalar("theta", theta)
        td = _as_scalar("theta_dot", theta_dot)
        return self.stiffness * th + self.damping * td

    def angular_momentum_body(self, omega: np.ndarray, theta_dot: float) -> np.ndarray:
        """Body-frame ``H = J_h ω + I_p (ω · â + θ̇) â``."""
        w = _as_omega(omega)
        td = _as_scalar("theta_dot", theta_dot)
        return self.inertia @ w + self.panel_inertia * self.hinge_rate(w, td) * self._axis

    def angular_momentum_inertial(
        self,
        q: np.ndarray,
        omega: np.ndarray,
        theta_dot: float,
    ) -> np.ndarray:
        """Inertial-frame ``H_I = R(q) H``."""
        return quat_to_rotation(q) @ self.angular_momentum_body(omega, theta_dot)

    def kinetic_energy(self, omega: np.ndarray, theta_dot: float) -> float:
        """``T = (1/2) ωᵀ J_h ω + (1/2) I_p (ω · â + θ̇)²``."""
        w = _as_omega(omega)
        td = _as_scalar("theta_dot", theta_dot)
        t_hub = 0.5 * float(w @ (self.inertia @ w))
        rel = self.hinge_rate(w, td)
        return t_hub + 0.5 * self.panel_inertia * rel * rel

    def potential_energy(self, theta: float) -> float:
        """Hinge strain energy ``V = (1/2) k θ²``."""
        th = _as_scalar("theta", theta)
        return 0.5 * self.stiffness * th * th

    def mechanical_energy(self, omega: np.ndarray, theta: float, theta_dot: float) -> float:
        """``E = T + V``."""
        return self.kinetic_energy(omega, theta_dot) + self.potential_energy(theta)

    def omega_dot(
        self,
        omega: np.ndarray,
        tau: np.ndarray,
        theta: float,
        theta_dot: float,
    ) -> np.ndarray:
        """Hub Euler / Newton–Euler ``ω̇`` at the current flexure state.

            ω̇ = J_h⁻¹ (τ + (k θ + c θ̇) â − ω × H)
        """
        w = _as_omega(omega)
        tau_b = _as_tau(tau)
        h = self.angular_momentum_body(w, theta_dot)
        hinge = self.hinge_torque(theta, theta_dot) * self._axis
        return self._j_inv @ (tau_b + hinge - np.cross(w, h))

    def theta_ddot(
        self,
        omega: np.ndarray,
        tau: np.ndarray,
        theta: float,
        theta_dot: float,
        omega_dot: np.ndarray | None = None,
    ) -> float:
        """Hinge acceleration ``θ̈``.

        ``I_p = 0`` freezes the flexure (``θ̈ = 0``).  Otherwise

            θ̈ = −â · ω̇ − (k θ + c θ̇) / I_p.
        """
        if not self.has_panel:
            return 0.0
        if omega_dot is None:
            wdot = self.omega_dot(omega, tau, theta, theta_dot)
        else:
            wdot = np.asarray(omega_dot, dtype=float).reshape(3)
        return float(-self._axis @ wdot - self.hinge_torque(theta, theta_dot) / self.panel_inertia)

    def derivatives(
        self,
        omega: np.ndarray,
        tau: np.ndarray,
        theta: float,
        theta_dot: float,
    ) -> tuple[np.ndarray, float, float]:
        """Packed flexure vector field ``(ω̇, θ̇, θ̈)``."""
        wdot = self.omega_dot(omega, tau, theta, theta_dot)
        tddot = self.theta_ddot(omega, tau, theta, theta_dot, omega_dot=wdot)
        td = _as_scalar("theta_dot", theta_dot)
        return wdot, td, tddot

    def linearized_state_space(self) -> tuple[np.ndarray, np.ndarray]:
        """Rest linearization about ``θ = 0``, ``ω = 0``.

        State ``x = [ω_x, ω_y, ω_z, θ, θ̇]``, input ``u = τ``.
        Returns ``(A, B)`` with ``A ∈ R^{5×5}``, ``B ∈ R^{5×3}``.
        Gyroscopic ``ω × H`` terms vanish at the rest equilibrium.
        ``I_p = 0`` drops the flexure rows to a frozen ``θ`` (``θ̈ = 0``).
        """
        a = self._axis
        g = self._j_inv
        g_a = g @ a
        a_g_a = float(a @ g_a)
        k = self.stiffness
        c = self.damping
        a_mat = np.zeros((5, 5))
        b_mat = np.zeros((5, 3))
        b_mat[:3, :] = g
        a_mat[:3, 3] = k * g_a
        a_mat[:3, 4] = c * g_a
        a_mat[3, 4] = 1.0
        if self.has_panel:
            i_p = self.panel_inertia
            coeff = a_g_a + 1.0 / i_p
            a_mat[4, 3] = -k * coeff
            a_mat[4, 4] = -c * coeff
            b_mat[4, :] = -a @ g
        return a_mat, b_mat

    def as_rigid_body(self) -> RigidBody:
        """Return a ``RigidBody`` with the hub inertia ``J_h``."""
        return RigidBody(self.inertia)

    def locked_inertia(self) -> np.ndarray:
        """Locked-appendage inertia ``J_h + I_p â âᵀ`` (θ frozen)."""
        a = self._axis.reshape(3, 1)
        return self.inertia + self.panel_inertia * (a @ a.T)


def pack_flex_state(omega: np.ndarray, theta: float, theta_dot: float) -> np.ndarray:
    """Pack ``y = [ω, θ, θ̇]`` as a length-5 array."""
    w = _as_omega(omega)
    th = _as_scalar("theta", theta)
    td = _as_scalar("theta_dot", theta_dot)
    return np.array([w[0], w[1], w[2], th, td], dtype=float)


def unpack_flex_state(y: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Unpack a length-5 flexure state into ``(ω, θ, θ̇)``."""
    y = np.asarray(y, dtype=float).reshape(-1)
    if y.shape != (5,):
        raise ValueError("flex state must have length 5 (ω, θ, θ̇)")
    return y[:3].copy(), float(y[3]), float(y[4])


def linearized_flex_state_space(plant: HingedAppendage) -> tuple[np.ndarray, np.ndarray]:
    """Module-level rest linearization ``(A, B)`` about ``θ = 0``."""
    return plant.linearized_state_space()


def step_flex(
    plant: HingedAppendage,
    omega: np.ndarray,
    theta: float,
    theta_dot: float,
    tau: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, float, float]:
    """Classical RK4 step of ``[ω, θ, θ̇]`` with ZOH hub torque.

    Returns ``(ω⁺, θ⁺, θ̇⁺)``.
    """
    tau_b = _as_tau(tau)

    def fun(_t: float, y: np.ndarray) -> np.ndarray:
        w, th, td = unpack_flex_state(y)
        wdot, thdot, tddot = plant.derivatives(w, tau_b, th, td)
        return pack_flex_state(wdot, thdot, tddot)

    y = rk4_step(fun, 0.0, pack_flex_state(omega, theta, theta_dot), dt)
    return unpack_flex_state(y)


def step_flex_attitude(
    plant: HingedAppendage,
    q: np.ndarray,
    omega: np.ndarray,
    theta: float,
    theta_dot: float,
    tau: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """RK4 step of ``[q, ω, θ, θ̇]`` with ZOH hub torque.

    Quaternion kinematics match :func:`attitude_sim.plant.step_rigid_body`
    (``q̇ = (1/2) q ⊗ ω̂``, then ``q ← q / ||q||``).  Returns
    ``(q⁺, ω⁺, θ⁺, θ̇⁺)``.
    """
    tau_b = _as_tau(tau)
    q0 = quat_normalize(q)
    y0 = np.concatenate([q0, pack_flex_state(omega, theta, theta_dot)])

    def fun(_t: float, y: np.ndarray) -> np.ndarray:
        q_k = y[:4]
        w, th, td = unpack_flex_state(y[4:])
        qdot = quat_derivative(q_k, w)
        wdot, thdot, tddot = plant.derivatives(w, tau_b, th, td)
        return np.concatenate([qdot, pack_flex_state(wdot, thdot, tddot)])

    y = rk4_step(fun, 0.0, y0, dt)
    q_next = quat_normalize(y[:4])
    w, th, td = unpack_flex_state(y[4:])
    return q_next, w, th, td

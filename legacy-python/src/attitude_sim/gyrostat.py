"""Dual-spin / gyrostat rigid-body plant (one body-axis rotor).

This is a **library** extension of ``attitude_sim.plant``.  It does not
change ``RigidBody``, the SimLab integrator, controllers, estimators, or
``attitude_sim.disturbances`` (aero / SRP stay untouched).

Gyrostat model
--------------
A simple gyrostat is a rigid carrier plus **one** rotor whose spin axis
``â`` is fixed in the body.  ``J = Jᵀ ≻ 0`` is the locked-rotor
(system) inertia — the carrier plus the rotor treated as if it were not
spinning relative to the body.  The rotor's **relative** angular
momentum is the body-frame vector

    h_w = h â = I_w Ω â

with scalar wheel momentum ``h``, wheel spin inertia ``I_w > 0`` about
``â``, and relative spin rate ``Ω``.  Constant-speed operation holds
``h`` (equivalently ``Ω``) fixed; commanded / variable-speed operation
treats ``Ω`` as a state driven by a motor torque ``τ_w`` along ``â``.

Euler / gyrostat equations
--------------------------
Total body-frame angular momentum is

    H = J ω + h_w.

Euler's equation on SO(3) (Hughes / Wie) is

    Ḣ + ω × H = τ

with external body torque ``τ``.  Expanding ``Ḣ = J ω̇ + ḣ_w`` gives
the gyrostat equation documented here:

    J ω̇ + ω × (J ω + h_w) + ḣ_w = τ
    ω̇ = J⁻¹ (τ − ω × (J ω + h_w) − ḣ_w).

Constant-speed (or a commanded constant ``h_w``) has ``ḣ_w = 0``.  A
variable-speed wheel along ``â`` has ``ḣ_w = τ_w â`` and

    I_w Ω̇ = τ_w,
    ω̇   = J⁻¹ (τ − τ_w â − ω × (J ω + I_w Ω â)).

When ``h_w = 0`` and ``ḣ_w = 0`` this is exactly Euler's equation on
``RigidBody``.

Torque-free first integrals (``τ = 0``, constant ``h_w``)
---------------------------------------------------------
The carrier Hamiltonian / Jacobi integral

    T = (1/2) ω · (J ω)

is conserved: ``ω · J ω̇ = −ω · (ω × (J ω + h_w)) = 0``.  The motor
that holds ``Ω`` fixed may exchange energy with the rotor, so
``T + ω · h_w + (1/2) I_w Ω²`` is **not** a first integral of the
constant-speed model.

Angular momentum is a Casimir of the Lie–Poisson structure:

    |H| = |J ω + h_w|

is conserved in the body frame, and the inertial vector

    H_I = R(q) (J ω + h_w)

is constant.  ``v_I = R(q) v_b`` is the same DCM convention as
``attitude_sim.plant``.  With a variable-speed wheel and ``τ = 0`` the
same ``H_I`` is still conserved (action–reaction: ``τ_w`` is internal);
``Ω`` then changes and ``T`` is no longer conserved.

Spin about the rotor axis
-------------------------
``ω = Ω_b â`` with ``h_w ∥ â`` is an equilibrium of the torque-free
gyrostat equation.  Linearizing a principal-axis gyrostat
``J = diag(I₁, I₂, I₃)`` about ``ω = Ω e_k``, ``h_w = h e_k`` gives a
transverse mode

    λ² = −[(I_k − I_i) Ω + h] [(I_k − I_j) Ω + h] / (I_i I_j)

(``{i, j, k} = {1, 2, 3}``).  ``λ² < 0`` is oscillatory (linearly
stable), ``λ² > 0`` is hyperbolic.  ``h = 0`` recovers the
intermediate-axis (tennis-racket) theorem.  A large enough ``|h|`` can
stabilize spin about the intermediate axis (classic dual-spin).  For an
axisymmetric ``J = diag(I, I, I₃)`` with the rotor along ``e₃``,
``ω₃`` is constant and the transverse rate rotates at

    Ω_p = ((I₃ − I) ω₃ + h) / I.

RK4 step
--------
:func:`step_gyrostat` is optional classical RK4 of the rate state
``y = ω ∈ R³`` or, when the wheel is variable, ``y = [ω, Ω] ∈ R⁴``.
Torque ``τ`` and motor torque ``τ_w`` are zero-order-held.
:func:`step_gyrostat_attitude` also integrates quaternion kinematics
``q̇ = (1/2) q ⊗ ω̂`` (same as the rigid-body plant) and renormalizes
``q`` after the Euclidean step so inertial-momentum checks can use
``H_I``.  Both reuse :func:`attitude_sim.plant.rk4_step`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from attitude_sim.plant import RigidBody, rk4_step, validate_inertia
from attitude_sim.quaternions import quat_derivative, quat_normalize, quat_to_rotation

_AXIS_NORM_EPS = 1e-15


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


def unit_rotor_axis(axis: np.ndarray | int | str) -> np.ndarray:
    """Return a unit body-frame rotor axis.

    ``axis`` is a length-3 vector, a principal-index ``0/1/2``, or a
    name ``\"x\"`` / ``\"y\"`` / ``\"z\"``.
    """
    if isinstance(axis, str):
        key = axis.strip().lower()
        named = {"x": 0, "y": 1, "z": 2}
        if key not in named:
            raise ValueError("rotor axis name must be 'x', 'y', or 'z'")
        axis = named[key]
    if isinstance(axis, (int, np.integer)):
        k = int(axis)
        if k not in (0, 1, 2):
            raise ValueError("rotor axis index must be 0, 1, or 2")
        e = np.zeros(3)
        e[k] = 1.0
        return e
    a = np.asarray(axis, dtype=float).reshape(3)
    if not np.all(np.isfinite(a)):
        raise ValueError("rotor axis must be finite")
    n = float(np.linalg.norm(a))
    if n <= _AXIS_NORM_EPS:
        raise ValueError("rotor axis must be non-zero")
    return a / n


def gyrostat_omega_dot(
    inertia: np.ndarray,
    omega: np.ndarray,
    tau: np.ndarray,
    h_wheel: np.ndarray,
    h_wheel_dot: np.ndarray | None = None,
) -> np.ndarray:
    """Gyrostat Euler equation in the body frame.

        ω̇ = J⁻¹ (τ − ω × (J ω + h_w) − ḣ_w)

    ``h_wheel_dot`` defaults to zero (constant-speed / commanded constant
    ``h_w``).  ``inertia`` is validated like :class:`RigidBody`.
    """
    J = validate_inertia(inertia)
    w = _as_omega(omega)
    tau_b = _as_tau(tau)
    h_w = _as_vector3("h_wheel", h_wheel)
    h_dot = np.zeros(3) if h_wheel_dot is None else _as_vector3("h_wheel_dot", h_wheel_dot)
    return np.linalg.solve(J, tau_b - np.cross(w, J @ w + h_w) - h_dot)


def transverse_lambda_squared(
    inertia: np.ndarray,
    axis: int,
    spin_rate: float = 1.0,
    h_wheel: float = 0.0,
) -> float:
    """Transverse ``λ²`` for principal-axis gyrostat spin ``ω = Ω e_k``.

        λ² = −[(I_k − I_i) Ω + h] [(I_k − I_j) Ω + h] / (I_i I_j)

    ``axis`` is the principal-moment index from
    :func:`attitude_sim.plant.principal_moments_and_axes` (``0`` = min,
    ``1`` = intermediate, ``2`` = max).  ``h_wheel`` is the signed rotor
    momentum along that same principal axis.  ``λ² < 0`` oscillatory,
    ``λ² > 0`` hyperbolic.
    """
    k = int(axis)
    if k not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2 (principal-moment index, ascending)")
    omega = float(spin_rate)
    h = float(h_wheel)
    if not np.isfinite(omega) or not np.isfinite(h):
        raise ValueError("spin_rate and h_wheel must be finite")
    J = validate_inertia(inertia)
    moments = np.linalg.eigvalsh(0.5 * (J + J.T))
    i, j = (k + 1) % 3, (k + 2) % 3
    i_k = float(moments[k])
    i_i = float(moments[i])
    i_j = float(moments[j])
    return -((i_k - i_i) * omega + h) * ((i_k - i_j) * omega + h) / (i_i * i_j)


@dataclass
class Gyrostat:
    """Rigid body plus one body-axis rotor (constant-speed or commanded).

    ``inertia`` is the locked-rotor inertia (validated, SPD, triangle
    inequalities).  ``rotor_axis`` is stored as a unit vector.  Constant-
    speed mode keeps a scalar ``wheel_momentum`` ``h`` along that axis
    (``h_w = h â``, ``ḣ_w = 0``).  Variable-speed mode sets
    ``wheel_inertia = I_w > 0`` and treats relative spin ``Ω`` as state
    with ``h_w = I_w Ω â`` and motor torque ``τ_w``.
    """

    inertia: np.ndarray
    rotor_axis: np.ndarray | int | str = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    wheel_momentum: float = 0.0
    wheel_inertia: float | None = None
    _j_inv: np.ndarray = field(init=False, repr=False, compare=False)
    _axis: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        J = validate_inertia(self.inertia)
        self.inertia = J
        self._j_inv = np.linalg.inv(J)
        self._axis = unit_rotor_axis(self.rotor_axis)
        self.rotor_axis = self._axis
        h = float(self.wheel_momentum)
        if not np.isfinite(h):
            raise ValueError("wheel_momentum must be finite")
        self.wheel_momentum = h
        if self.wheel_inertia is not None:
            i_w = float(self.wheel_inertia)
            if not np.isfinite(i_w) or i_w <= 0.0:
                raise ValueError("wheel_inertia must be positive")
            self.wheel_inertia = i_w

    @property
    def inertia_inv(self) -> np.ndarray:
        """Cached ``J⁻¹``."""
        return self._j_inv

    @property
    def axis(self) -> np.ndarray:
        """Unit rotor axis in the body frame."""
        return self._axis.copy()

    def h_wheel_vector(self, omega_w: float | None = None) -> np.ndarray:
        """Relative rotor momentum ``h_w = h â`` or ``I_w Ω â``."""
        if omega_w is None:
            return self.wheel_momentum * self._axis
        if self.wheel_inertia is None:
            raise ValueError("omega_w requires wheel_inertia (variable-speed gyrostat)")
        rate = float(omega_w)
        if not np.isfinite(rate):
            raise ValueError("omega_w must be finite")
        return self.wheel_inertia * rate * self._axis

    def h_wheel_dot(self, tau_w: float = 0.0) -> np.ndarray:
        """Relative momentum rate ``ḣ_w = τ_w â`` (internal motor)."""
        tw = float(tau_w)
        if not np.isfinite(tw):
            raise ValueError("tau_w must be finite")
        if self.wheel_inertia is None and tw != 0.0:
            raise ValueError("tau_w requires wheel_inertia (variable-speed gyrostat)")
        return tw * self._axis

    def omega_dot(
        self,
        omega: np.ndarray,
        tau: np.ndarray,
        *,
        omega_w: float | None = None,
        tau_w: float = 0.0,
        h_wheel: np.ndarray | None = None,
    ) -> np.ndarray:
        """Gyrostat ``ω̇`` at the current body rate.

        ``h_wheel`` overrides the stored / ``Ω``-derived ``h_w`` (commanded
        rotor momentum).  ``tau_w`` is the motor torque along ``â``.
        """
        w = _as_omega(omega)
        tau_b = _as_tau(tau)
        h_w = self.h_wheel_vector(omega_w) if h_wheel is None else _as_vector3("h_wheel", h_wheel)
        h_dot = self.h_wheel_dot(tau_w)
        return self._j_inv @ (tau_b - np.cross(w, self.inertia @ w + h_w) - h_dot)

    def omega_w_dot(self, tau_w: float) -> float:
        """Wheel spin equation ``Ω̇ = τ_w / I_w``."""
        if self.wheel_inertia is None:
            raise ValueError("omega_w_dot requires wheel_inertia (variable-speed gyrostat)")
        tw = float(tau_w)
        if not np.isfinite(tw):
            raise ValueError("tau_w must be finite")
        return tw / self.wheel_inertia

    def kinetic_energy(self, omega: np.ndarray) -> float:
        """Carrier Hamiltonian ``T = (1/2) ωᵀ J ω``.

        Torque-free first integral when ``h_w`` is constant.  The holding
        motor that keeps ``Ω`` fixed may exchange energy with the rotor,
        so ``T + ω · h_w + (1/2) I_w Ω²`` is not conserved in
        constant-speed mode.
        """
        w = _as_omega(omega)
        return 0.5 * float(w @ (self.inertia @ w))

    def rotor_kinetic_energy(self, omega_w: float) -> float:
        """Rotor spin energy ``(1/2) I_w Ω²`` (variable-speed only)."""
        if self.wheel_inertia is None:
            raise ValueError("rotor_kinetic_energy requires wheel_inertia")
        rate = float(omega_w)
        if not np.isfinite(rate):
            raise ValueError("omega_w must be finite")
        return 0.5 * self.wheel_inertia * rate * rate

    def angular_momentum_body(
        self,
        omega: np.ndarray,
        omega_w: float | None = None,
        *,
        h_wheel: np.ndarray | None = None,
    ) -> np.ndarray:
        """Body-frame ``H = J ω + h_w``."""
        w = _as_omega(omega)
        h_w = self.h_wheel_vector(omega_w) if h_wheel is None else _as_vector3("h_wheel", h_wheel)
        return self.inertia @ w + h_w

    def angular_momentum_inertial(
        self,
        q: np.ndarray,
        omega: np.ndarray,
        omega_w: float | None = None,
        *,
        h_wheel: np.ndarray | None = None,
    ) -> np.ndarray:
        """Inertial-frame ``H_I = R(q) (J ω + h_w)``."""
        return quat_to_rotation(q) @ self.angular_momentum_body(
            omega, omega_w, h_wheel=h_wheel
        )

    def as_rigid_body(self) -> RigidBody:
        """Return a ``RigidBody`` with the same locked-rotor inertia."""
        return RigidBody(self.inertia)


def pack_rate_state(omega: np.ndarray, omega_w: float | None = None) -> np.ndarray:
    """Pack ``y = ω`` or ``y = [ω, Ω]``."""
    w = _as_omega(omega)
    if omega_w is None:
        return w.copy()
    rate = float(omega_w)
    if not np.isfinite(rate):
        raise ValueError("omega_w must be finite")
    return np.array([w[0], w[1], w[2], rate], dtype=float)


def unpack_rate_state(y: np.ndarray) -> tuple[np.ndarray, float | None]:
    """Unpack a length-3 or length-4 rate state into ``(ω, Ω or None)``."""
    y = np.asarray(y, dtype=float).reshape(-1)
    if y.shape == (3,):
        return y.copy(), None
    if y.shape == (4,):
        return y[:3].copy(), float(y[3])
    raise ValueError("rate state must have length 3 (ω) or 4 (ω, Ω)")


def step_gyrostat(
    gyro: Gyrostat,
    omega: np.ndarray,
    tau: np.ndarray,
    dt: float,
    *,
    omega_w: float | None = None,
    tau_w: float = 0.0,
    h_wheel: np.ndarray | None = None,
) -> tuple[np.ndarray, float | None]:
    """Classical RK4 step of gyrostat rates ``ω`` (and optional ``Ω``).

    Constant-speed: pass ``omega_w=None`` (uses stored ``wheel_momentum``;
    ``h_wheel`` may override ``h_w``).  Variable-speed: pass ``omega_w``
    and a ``Gyrostat`` with ``wheel_inertia``; ``τ`` and ``τ_w`` are
    held ZOH.  Returns ``(ω⁺, Ω⁺ or None)``.
    """
    tau_b = _as_tau(tau)
    tw = float(tau_w)
    if not np.isfinite(tw):
        raise ValueError("tau_w must be finite")
    variable = omega_w is not None
    if variable and gyro.wheel_inertia is None:
        raise ValueError("omega_w requires wheel_inertia (variable-speed gyrostat)")
    if (not variable) and tw != 0.0:
        raise ValueError("tau_w requires a variable-speed step (pass omega_w)")

    def fun(_t: float, y: np.ndarray) -> np.ndarray:
        w, rate = unpack_rate_state(y)
        if rate is None:
            return gyro.omega_dot(w, tau_b, h_wheel=h_wheel)
        return pack_rate_state(
            gyro.omega_dot(w, tau_b, omega_w=rate, tau_w=tw),
            gyro.omega_w_dot(tw),
        )

    y = rk4_step(fun, 0.0, pack_rate_state(omega, omega_w), dt)
    return unpack_rate_state(y)


def step_gyrostat_attitude(
    gyro: Gyrostat,
    q: np.ndarray,
    omega: np.ndarray,
    tau: np.ndarray,
    dt: float,
    *,
    omega_w: float | None = None,
    tau_w: float = 0.0,
    h_wheel: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float | None]:
    """RK4 step of ``[q, ω]`` or ``[q, ω, Ω]`` with ZOH torques.

    Quaternion kinematics match :func:`attitude_sim.plant.step_rigid_body`
    (``q̇ = (1/2) q ⊗ ω̂``, then ``q ← q / ||q||``).  Rate dynamics use
    the gyrostat equation.  Returns ``(q⁺, ω⁺, Ω⁺ or None)``.
    """
    tau_b = _as_tau(tau)
    tw = float(tau_w)
    if not np.isfinite(tw):
        raise ValueError("tau_w must be finite")
    variable = omega_w is not None
    if variable and gyro.wheel_inertia is None:
        raise ValueError("omega_w requires wheel_inertia (variable-speed gyrostat)")
    if (not variable) and tw != 0.0:
        raise ValueError("tau_w requires a variable-speed step (pass omega_w)")

    q0 = quat_normalize(q)
    if variable:
        y0 = np.concatenate([q0, pack_rate_state(omega, omega_w)])
    else:
        y0 = np.concatenate([q0, _as_omega(omega)])

    def fun(_t: float, y: np.ndarray) -> np.ndarray:
        q_k = y[:4]
        w = y[4:7]
        qdot = quat_derivative(q_k, w)
        if y.shape[0] == 7:
            wdot = gyro.omega_dot(w, tau_b, h_wheel=h_wheel)
            return np.concatenate([qdot, wdot])
        rate = float(y[7])
        wdot = gyro.omega_dot(w, tau_b, omega_w=rate, tau_w=tw)
        return np.concatenate([qdot, wdot, np.array([gyro.omega_w_dot(tw)])])

    y = rk4_step(fun, 0.0, y0, dt)
    q_next = quat_normalize(y[:4])
    if y.shape[0] == 7:
        return q_next, y[4:7].copy(), None
    return q_next, y[4:7].copy(), float(y[7])

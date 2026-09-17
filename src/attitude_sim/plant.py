"""Rigid-body rotational plant: quaternion kinematics + Euler's equation.

State
-----
The plant state is the concatenated vector

    x = [q(4), ω(3)]  ∈ R^7

with a scalar-first unit quaternion ``q = [q_w, q_x, q_y, q_z]`` (Hamilton
product) and body-frame angular velocity ``ω``.  Attitude maps body vectors
into inertial coordinates by

    v_I = R(q) v_b.

Kinematics (q-dot)
------------------
Pack the body rate as the pure quaternion ``ω̂ = [0, ω]``.  Right-trivialized
kinematics on S^3 are

    q̇ = (1/2) q ⊗ ω̂.

``q`` is the attitude of the body relative to inertial; ``ω`` is resolved in
the body frame.  The Euclidean chart does not preserve ``||q|| = 1``, so the
integrator projects ``q ← q / ||q||`` after every completed step.

Euler's equation
----------------
For a rigid body with constant body-frame inertia ``J = Jᵀ ≻ 0`` and
external torque ``τ`` (also body-frame),

    J ω̇ = τ − ω × (J ω)
    ω̇   = J⁻¹ (τ − ω × (J ω)).

Torque-free first integrals (``τ = 0``) are the rotational kinetic energy

    T = (1/2) ω · (J ω)

and the inertial-frame angular momentum

    h_I = R(q) J ω.

``|h_b| = |J ω|`` is the same conserved magnitude in the body frame.
Classical RK4 is not symplectic, so these invariants hold only up to
truncation error; ``tests/test_plant.py`` records explicit tolerances.

RK4 step
--------
Torque is held with a zero-order hold over the sample.  One classical
fourth-order Runge–Kutta step of size ``h`` is

    k₁ = f(t,       y)
    k₂ = f(t + h/2, y + (h/2) k₁)
    k₃ = f(t + h/2, y + (h/2) k₂)
    k₄ = f(t + h,   y + h k₃)
    y⁺ = y + (h/6) (k₁ + 2 k₂ + 2 k₃ + k₄)

applied to ``x``, after which ``q`` is renormalized.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from attitude_sim.quaternions import quat_derivative, quat_normalize, quat_to_rotation

_SYMM_TOL = 1e-12
_TRIANGLE_TOL = 1e-9


def pack_state(q: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Pack ``x = [q(4), ω(3)]`` as a length-7 float array."""
    return np.concatenate(
        [
            np.asarray(q, dtype=float).reshape(4),
            np.asarray(omega, dtype=float).reshape(3),
        ]
    )


def unpack_state(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unpack a length-7 state into ``(q, ω)`` copies."""
    y = np.asarray(y, dtype=float).reshape(7)
    return y[:4].copy(), y[4:].copy()


def principal_moments_and_axes(inertia: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return principal moments (ascending) and a right-handed principal-axis DCM.

    Columns of ``axes`` are body-frame principal axes, so

        J = axes @ diag(moments) @ axes.T

    and ``det(axes) = +1``.  ``inertia`` is symmetrized before the
    eigendecomposition; it is not otherwise validated (use
    :func:`validate_inertia` for SPD / triangle checks).
    """
    J = np.asarray(inertia, dtype=float).reshape(3, 3)
    J = 0.5 * (J + J.T)
    moments, axes = np.linalg.eigh(J)
    if np.linalg.det(axes) < 0.0:
        axes = axes.copy()
        axes[:, 0] *= -1.0
    return moments, axes


def inertia_from_principal(
    moments: np.ndarray,
    axes: np.ndarray | None = None,
) -> np.ndarray:
    """Assemble ``J = R diag(I1, I2, I3) Rᵀ``.

    Parameters
    ----------
    moments:
        Principal moments ``[I1, I2, I3]`` (any order).  Must be a physical
        inertia: positive and satisfying the triangle inequalities.
    axes:
        Optional 3×3 matrix whose columns are principal axes expressed in the
        body frame.  ``None`` yields a diagonal (already-principal) ``J``.
        Must be a proper rotation (``Rᵀ R = I``, ``det R = +1``).
    """
    I = np.asarray(moments, dtype=float).reshape(3)
    if axes is None:
        R = np.eye(3)
    else:
        R = np.asarray(axes, dtype=float).reshape(3, 3)
        if np.max(np.abs(R.T @ R - np.eye(3))) > 1e-8:
            raise ValueError("principal axes must be orthonormal")
        if abs(np.linalg.det(R) - 1.0) > 1e-8:
            raise ValueError("principal axes must be a proper rotation (det = +1)")
    J = R @ np.diag(I) @ R.T
    return validate_inertia(J)


def is_principal(inertia: np.ndarray, *, atol: float = 1e-12) -> bool:
    """Return True if ``inertia`` is diagonal in the current body frame."""
    J = np.asarray(inertia, dtype=float).reshape(3, 3)
    off = J - np.diag(np.diag(J))
    return float(np.max(np.abs(off))) <= atol


def validate_inertia(inertia: np.ndarray, *, require_physical: bool = True) -> np.ndarray:
    """Return a symmetrized 3×3 inertia, or raise ``ValueError``.

    Checks
    ------
    * shape ``(3, 3)``
    * symmetry (off-diagonal mismatch ≤ 1e-12, then average)
    * positive definite (all eigenvalues of the symmetric part > 0)
    * if ``require_physical``, principal moments satisfy the rigid-body
      triangle inequalities

          Iᵢ + Iⱼ ≥ Iₖ    ⇔    I₁ + I₂ + I₃ ≥ 2 Iₖ

      which is equivalent to the second moments of mass being non-negative.
      Equality is allowed (planar mass distribution).
    """
    J = np.asarray(inertia, dtype=float)
    if J.shape != (3, 3):
        raise ValueError("inertia must be a 3x3 matrix")
    if float(np.max(np.abs(J - J.T))) > _SYMM_TOL:
        raise ValueError("inertia must be symmetric")
    J = 0.5 * (J + J.T)
    moments = np.linalg.eigvalsh(J)
    if np.any(moments <= 0.0):
        raise ValueError("inertia must be positive definite")
    if require_physical:
        total = float(moments.sum())
        if np.any(total + _TRIANGLE_TOL < 2.0 * moments):
            raise ValueError(
                "principal moments must satisfy I_i + I_j >= I_k "
                "(physical rigid-body inertia)"
            )
    return J


def rk4_step(
    fun: Callable[[float, np.ndarray], np.ndarray],
    t: float,
    y: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Classical fourth-order Runge–Kutta step.

    Integrates ``ẏ = f(t, y)`` over ``[t, t+dt]``:

        k₁ = f(t,       y)
        k₂ = f(t + h/2, y + (h/2) k₁)
        k₃ = f(t + h/2, y + (h/2) k₂)
        k₄ = f(t + h,   y + h k₃)
        y⁺ = y + (h/6) (k₁ + 2 k₂ + 2 k₃ + k₄)

    Stage states are copied so ``fun`` cannot alias-mutate the input ``y``.
    ``dt`` must be positive.
    """
    dt = float(dt)
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    y = np.array(y, dtype=float, copy=True)
    half = 0.5 * dt
    k1 = np.asarray(fun(t, y.copy()), dtype=float)
    k2 = np.asarray(fun(t + half, y + half * k1), dtype=float)
    k3 = np.asarray(fun(t + half, y + half * k2), dtype=float)
    k4 = np.asarray(fun(t + dt, y + dt * k3), dtype=float)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


@dataclass
class RigidBody:
    """Torque-driven rigid body with a constant body-frame inertia matrix.

    ``inertia`` is validated, symmetrized, and stored as a physical 3×3
    tensor (SPD with triangle inequalities on the principal moments).
    The inverse ``J⁻¹`` is cached for Euler's equation.
    """

    inertia: np.ndarray
    _j_inv: np.ndarray = field(init=False, repr=False, compare=False)
    _moments: np.ndarray = field(init=False, repr=False, compare=False)
    _axes: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        J = validate_inertia(self.inertia)
        moments, axes = principal_moments_and_axes(J)
        self.inertia = J
        self._j_inv = np.linalg.inv(J)
        self._moments = moments
        self._axes = axes

    @property
    def principal_moments(self) -> np.ndarray:
        """Principal moments of inertia, ascending (kg·m²)."""
        return self._moments.copy()

    @property
    def principal_axes(self) -> np.ndarray:
        """Right-handed DCM whose columns are principal axes in the body frame."""
        return self._axes.copy()

    @property
    def inertia_inv(self) -> np.ndarray:
        """Cached ``J⁻¹``."""
        return self._j_inv

    def omega_dot(self, omega: np.ndarray, tau: np.ndarray) -> np.ndarray:
        """Euler's rotational equation resolved in the body frame.

            ω̇ = J⁻¹ (τ − ω × (J ω))
        """
        omega = np.asarray(omega, dtype=float).reshape(3)
        tau = np.asarray(tau, dtype=float).reshape(3)
        h = self.inertia @ omega
        return self._j_inv @ (tau - np.cross(omega, h))

    def derivatives(self, q: np.ndarray, omega: np.ndarray, tau: np.ndarray) -> np.ndarray:
        """Packed vector field ``[q̇, ω̇]`` at the current attitude and rate."""
        qdot = quat_derivative(q, omega)
        wdot = self.omega_dot(omega, tau)
        return np.concatenate([qdot, wdot])

    def kinetic_energy(self, omega: np.ndarray) -> float:
        """Rotational kinetic energy ``T = (1/2) ωᵀ J ω``."""
        omega = np.asarray(omega, dtype=float).reshape(3)
        return 0.5 * float(omega @ (self.inertia @ omega))

    def angular_momentum_body(self, omega: np.ndarray) -> np.ndarray:
        """Body-frame angular momentum ``h_b = J ω``."""
        return self.inertia @ np.asarray(omega, dtype=float).reshape(3)

    def angular_momentum_inertial(self, q: np.ndarray, omega: np.ndarray) -> np.ndarray:
        """Inertial-frame angular momentum ``h_I = R(q) J ω``."""
        return quat_to_rotation(q) @ self.angular_momentum_body(omega)


def step_rigid_body(
    body: RigidBody,
    q: np.ndarray,
    omega: np.ndarray,
    tau: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance the plant one RK4 step with zero-order-hold torque.

    After the Euclidean RK4 update the quaternion is renormalized so that
    subsequent kinematics stay on S^3.
    """
    tau = np.asarray(tau, dtype=float).reshape(3)

    def fun(_t: float, y: np.ndarray) -> np.ndarray:
        return body.derivatives(y[:4], y[4:], tau)

    y = rk4_step(fun, 0.0, pack_state(q, omega), dt)
    q_next, omega_next = unpack_state(y)
    return quat_normalize(q_next), omega_next

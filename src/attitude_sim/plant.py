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
classical RK4 integrator projects ``q ← q / ||q||`` after every completed
step.  The optional RKMK4 (Lie-group) step stays on S^3 by construction.

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
truncation error.  The RKMK4 attitude step is also a classical RK tableau
(on the Lie algebra), so energy / momentum residuals are comparable, not
machine-zero.  ``tests/test_plant.py`` records tight tolerances at a
small fixed ``dt``; ``tests/test_plant_perturbations.py`` records looser
bounds under mild principal-inertia mismatch and varied ``dt``;
``tests/test_plant_rkmk4.py`` records unit-norm-by-construction and
side-by-side RK4 vs RKMK4 drift.

Principal-axis perturbations
----------------------------
Monte Carlo-style mismatch scales the principal moments while keeping the
same principal-axis DCM:

    J' = R diag((1+εᵢ) Iᵢ) Rᵀ,     εᵢ ∈ (−frac, frac).

:func:`scale_principal_inertia` is the deterministic reconstruction (SPD
and triangle inequalities are re-validated).  :func:`perturb_principal_inertia`
samples independent ``εᵢ`` with a uniform-scale fallback.  SimLab's
``monte_carlo.perturb_inertia`` is the runtime sampler; these helpers are
the plant-owned reconstruction (same ``R diag(s ⊙ I) Rᵀ``) used by
torque-free I-mismatch tests.  Work–energy / spherical closed-form
checks live in ``tests/test_plant.py``.

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

RKMK4 step
----------
Optional Lie-group integrator selected by ``method="rkmk4"``.  Right-
trivialized kinematics are an ODE on the Lie group S^3 ≅ Spin(3):

    q̇ = q · λ(ω),     λ(ω) = ω ∈ so(3) ≅ R³.

Munthe–Kaas RK4 pulls the vector field back to the Lie algebra with the
inverse exponential differential, runs the classical RK4 tableau there,
and returns via the exponential map.  With Butcher coefficients
``c = (0, 1/2, 1/2, 1)``, ``b = (1/6, 1/3, 1/3, 1/6)``:

    uᵢ = h Σⱼ aᵢⱼ kⱼ
    ωᵢ = ω + h Σⱼ aᵢⱼ kⱼ^ω
    kᵢ = dexp⁻¹_{uᵢ}(ωᵢ)
    kᵢ^ω = J⁻¹ (τ − ωᵢ × J ωᵢ)
    φ  = h Σᵢ bᵢ kᵢ
    q⁺ = q ⊗ Exp(φ)
    ω⁺ = ω + h Σᵢ bᵢ kᵢ^ω

The exponential ``Exp: R³ → S³`` (rotation vector → unit quaternion) is

    Exp(φ) = [cos(θ/2), sin(θ/2) φ̂],     θ = ||φ||,   φ̂ = φ/θ

and is unit by construction (``cos²(θ/2) + sin²(θ/2) = 1``).  The inverse
exponential differential on so(3) is

    dexp⁻¹_φ(v) = v + ½ φ×v
                  + [1 − (θ/2) cot(θ/2)] / θ²  ·  φ×(φ×v)

with the small-θ series ``v + ½ φ×v + (1/12) φ×(φ×v)`` (right-trivialized,
matching ``q̇ = (1/2) q ⊗ ω̂``).  Euler's equation
lives in Euclidean R³, so the same RK4 tableau applies directly to ``ω``.
Unlike classical RK4, RKMK4 does **not** renormalize ``q``.

:func:`step_rigid_body` is the API switch (``method="rk4"`` default,
``method="rkmk4"`` for the geometric step).  Torque is a zero-order hold
in the body frame over the sample, as with RK4.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from attitude_sim.quaternions import (
    quat_derivative,
    quat_multiply,
    quat_normalize,
    quat_to_rotation,
)

_SYMM_TOL = 1e-12
_TRIANGLE_TOL = 1e-9
_SO3_EXP_EPS = 1e-14
_DEXP_SERIES_EPS = 1e-8

INTEGRATOR_RK4 = "rk4"
INTEGRATOR_RKMK4 = "rkmk4"
_INTEGRATOR_METHODS = (INTEGRATOR_RK4, INTEGRATOR_RKMK4)


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
    :func:`validate_inertia` for SPD / triangle checks).  Entries must
    be finite so Monte Carlo-style principal scaling cannot ``eigh`` a
    NaN tensor.
    """
    J = np.asarray(inertia, dtype=float).reshape(3, 3)
    if not np.all(np.isfinite(J)):
        raise ValueError("inertia must be finite")
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
    * finite entries (no NaN / Inf)
    * symmetry (off-diagonal mismatch ≤ 1e-12, then average)
    * positive definite (all principal moments / eigenvalues of the
      symmetric part strictly positive)
    * if ``require_physical``, principal moments satisfy the rigid-body
      triangle inequalities

          Iᵢ + Iⱼ ≥ Iₖ    ⇔    I₁ + I₂ + I₃ ≥ 2 Iₖ

      which is equivalent to the second moments of mass being non-negative.
      Equality is allowed (planar mass distribution).
    """
    J = np.asarray(inertia, dtype=float)
    if J.shape != (3, 3):
        raise ValueError("inertia must be a 3x3 matrix")
    if not np.all(np.isfinite(J)):
        raise ValueError("inertia must be finite")
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


def scale_principal_inertia(inertia: np.ndarray, scale: np.ndarray | float) -> np.ndarray:
    """Return ``J`` with principal moments multiplied by ``scale``.

    Principal axes are unchanged:

        J' = R diag(s ⊙ I) Rᵀ

    ``scale`` is a positive scalar or length-3 vector of strictly positive
    finite factors.  The result is re-validated (SPD, triangle inequalities).
    """
    J = validate_inertia(inertia)
    s = np.asarray(scale, dtype=float)
    if s.shape == ():
        s = np.full(3, float(s))
    else:
        s = s.reshape(3)
    if not np.all(np.isfinite(s)):
        raise ValueError("principal-moment scale must be finite")
    if np.any(s <= 0.0):
        raise ValueError("principal-moment scale must be positive")
    moments, axes = principal_moments_and_axes(J)
    return inertia_from_principal(moments * s, axes)


def random_principal_scale(rng: np.random.Generator, frac: float) -> np.ndarray:
    """Independent factors ``1 + U(-frac, frac)`` for principal-moment scaling.

    ``0 ≤ frac < 1`` so every draw is strictly positive.  ``frac == 0``
    returns ones.  This does not assemble an inertia; pair it with
    :func:`scale_principal_inertia`.  Closed-loop Monte Carlo sampling
    lives in SimLab; plant tests use this helper for mild mismatch.
    """
    frac = float(frac)
    if not np.isfinite(frac) or frac < 0.0 or frac >= 1.0:
        raise ValueError("frac must satisfy 0 <= frac < 1")
    if frac == 0.0:
        return np.ones(3)
    return 1.0 + rng.uniform(-frac, frac, size=3)


def perturb_principal_inertia(
    inertia: np.ndarray,
    frac: float,
    rng: np.random.Generator,
    *,
    max_tries: int = 16,
) -> np.ndarray:
    """Scale principal moments by independent ``1+U(-frac, frac)`` draws.

    Independent per-axis scales are retried if triangle inequalities fail;
    the last fallback is a uniform scale of all three moments (always a
    physical inertia when ``0 ≤ frac < 1`` and the original ``J`` was
    physical).  ``frac <= 0`` returns a copy of the validated inertia.

    Runtime closed-loop Monte Carlo is owned by SimLab; this reconstruction
    is plant-owned so unit tests share one SPD perturbation path.
    """
    J = validate_inertia(inertia)
    frac = float(frac)
    if not np.isfinite(frac) or frac < 0.0 or frac >= 1.0:
        raise ValueError("frac must satisfy 0 <= frac < 1")
    if frac == 0.0:
        return J.copy()
    tries = int(max_tries)
    if tries < 1:
        raise ValueError("max_tries must be >= 1")
    for _ in range(tries):
        try:
            return scale_principal_inertia(J, random_principal_scale(rng, frac))
        except ValueError as exc:
            if "principal moments" not in str(exc):
                raise
            continue
    s = 1.0 + float(rng.uniform(-frac, frac))
    return scale_principal_inertia(J, s)


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
    ``dt`` must be finite and strictly positive.
    """
    dt = float(dt)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be positive")
    y = np.array(y, dtype=float, copy=True)
    half = 0.5 * dt
    k1 = np.asarray(fun(t, y.copy()), dtype=float)
    k2 = np.asarray(fun(t + half, y + half * k1), dtype=float)
    k3 = np.asarray(fun(t + half, y + half * k2), dtype=float)
    k4 = np.asarray(fun(t + dt, y + dt * k3), dtype=float)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _require_positive_dt(dt: float) -> float:
    dt = float(dt)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be positive")
    return dt


def so3_quat_exp(phi: np.ndarray) -> np.ndarray:
    """Exponential map ``Exp: so(3) ≅ R³ → S³``.

    A rotation vector ``φ`` (axis × angle, radians) maps to the unit
    quaternion

        Exp(φ) = [cos(θ/2), sin(θ/2) φ̂],     θ = ||φ||

    which is unit by construction (``cos² + sin² = 1``).  Right-trivialized
    kinematics then compose as ``q⁺ = q ⊗ Exp(φ)``.  ``θ = 0`` returns the
    identity; the small-θ branch uses ``v = φ/2`` on the sphere
    ``w = √(1 − ||v||²)`` so the result stays in S^3 without a separate
    normalization step.
    """
    phi = np.asarray(phi, dtype=float).reshape(3)
    theta = float(np.linalg.norm(phi))
    if theta < _SO3_EXP_EPS:
        v = 0.5 * phi
        w = float(np.sqrt(max(0.0, 1.0 - float(v @ v))))
        return np.array([w, v[0], v[1], v[2]])
    half = 0.5 * theta
    scale = np.sin(half) / theta
    return np.array([np.cos(half), scale * phi[0], scale * phi[1], scale * phi[2]])


def dexpinv_so3(phi: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Right-trivialized ``dexp⁻¹_φ(v)`` on so(3).

        dexp⁻¹_φ(v) = v + ½ φ×v
                      + [1 − (θ/2) cot(θ/2)] / θ²  ·  φ×(φ×v)

    with ``θ = ||φ||``.  The sign of the linear commutator is the
    *right*-trivialized convention (``q̇ = (1/2) q ⊗ ω̂`` / ``Ṙ = R ω̂``).
    The small-θ series (Bernoulli truncation used by classical RKMK4) is

        v + ½ φ×v + (1/12) φ×(φ×v).

    When ``v`` is parallel to ``φ``, the cross terms vanish and the map is
    the identity, so a constant body rate is integrated exactly.
    """
    phi = np.asarray(phi, dtype=float).reshape(3)
    vec = np.asarray(vec, dtype=float).reshape(3)
    theta = float(np.linalg.norm(phi))
    w = np.cross(phi, vec)
    if theta < _DEXP_SERIES_EPS:
        return vec + 0.5 * w + (1.0 / 12.0) * np.cross(phi, w)
    half = 0.5 * theta
    s = np.sin(half)
    # |φ| from one RKMK stage is O(|ω| dt) ≪ 2π for practical sample times.
    cot_half = np.cos(half) / s
    beta = (1.0 - half * cot_half) / (theta * theta)
    return vec + 0.5 * w + beta * np.cross(phi, w)


def rkmk4_step(
    body: RigidBody,
    q: np.ndarray,
    omega: np.ndarray,
    tau: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """One Munthe–Kaas RK4 step on S^3 with Euclidean RK4 on ``ω``.

    Attitude is advanced by the group exponential (unit quaternion by
    construction).  Euler's equation is the same RK4 tableau in R³.
    ``dt`` must be finite and strictly positive.  ``tau`` is held ZOH in
    the body frame.  See the module docstring for the stage equations.
    """
    dt = _require_positive_dt(dt)
    q = np.asarray(q, dtype=float).reshape(4)
    omega = np.asarray(omega, dtype=float).reshape(3)
    tau = np.asarray(tau, dtype=float).reshape(3)
    half = 0.5 * dt

    k1_w = body.omega_dot(omega, tau)
    k1_q = omega

    w2 = omega + half * k1_w
    u2 = half * k1_q
    k2_w = body.omega_dot(w2, tau)
    k2_q = dexpinv_so3(u2, w2)

    w3 = omega + half * k2_w
    u3 = half * k2_q
    k3_w = body.omega_dot(w3, tau)
    k3_q = dexpinv_so3(u3, w3)

    w4 = omega + dt * k3_w
    u4 = dt * k3_q
    k4_w = body.omega_dot(w4, tau)
    k4_q = dexpinv_so3(u4, w4)

    omega_next = omega + (dt / 6.0) * (k1_w + 2.0 * k2_w + 2.0 * k3_w + k4_w)
    phi = (dt / 6.0) * (k1_q + 2.0 * k2_q + 2.0 * k3_q + k4_q)
    q_next = quat_multiply(q, so3_quat_exp(phi))
    return q_next, omega_next


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
    *,
    method: str = INTEGRATOR_RK4,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance the plant one step with zero-order-hold torque.

    Parameters
    ----------
    method:
        Integrator switch.  ``"rk4"`` (default) is classical Euclidean RK4
        on ``x = [q, ω]`` followed by quaternion renormalization.
        ``"rkmk4"`` is Munthe–Kaas RK4 on S^3 (exponential-map attitude
        update, unit-norm by construction) with the same RK4 tableau on
        ``ω``.  Unknown names raise ``ValueError``.

    After Euclidean RK4 the quaternion is renormalized so subsequent
    kinematics stay on S^3.  RKMK4 does not renormalize: ``||q|| = 1``
    comes from ``q ⊗ Exp(φ)``.
    """
    tau = np.asarray(tau, dtype=float).reshape(3)
    name = str(method).strip().lower()
    if name == INTEGRATOR_RKMK4:
        return rkmk4_step(body, q, omega, tau, dt)
    if name != INTEGRATOR_RK4:
        raise ValueError(
            f"unknown integrator {method!r}; expected one of {_INTEGRATOR_METHODS}"
        )

    def fun(_t: float, y: np.ndarray) -> np.ndarray:
        return body.derivatives(y[:4], y[4:], tau)

    y = rk4_step(fun, 0.0, pack_state(q, omega), dt)
    q_next, omega_next = unpack_state(y)
    return quat_normalize(q_next), omega_next

"""Torque-free rigid-body polhode / energy–Casimir analysis.

These helpers do **not** change the SimLab integrator, controllers, or
estimators.  They evaluate first integrals of Euler's equation, sample
the polhode (and optionally the herpolhode), and classify principal-axis
spin by the intermediate-axis (tennis-racket) theorem.

Energy and angular momentum
---------------------------
For a rigid body with constant body-frame inertia ``J = Jᵀ ≻ 0`` and
body rate ``ω``,

    T   = (1/2) ω · (J ω)          rotational kinetic energy
    h_b = J ω                      body-frame angular momentum
    h_I = R(q) J ω                 inertial-frame angular momentum

``v_I = R(q) v_b`` is the same DCM convention as ``attitude_sim.plant``.
Energy is a scalar, so ``T = (1/2) ω_I · h_I`` as well.

Casimirs / torque-free first integrals
--------------------------------------
Euler's equation (body frame)

    J ω̇ = τ − ω × (J ω)

is Lie–Poisson on so(3)* when ``τ = 0``.  The Hamiltonian is ``T`` and
the Casimir is the squared angular-momentum magnitude

    C = |h|² = |J ω|² = |h_I|².

Both ``T`` and ``C`` are conserved along torque-free solutions (exactly
for the ODE; up to RK4 truncation in ``step_rigid_body``).  The inertial
*vector* ``h_I`` is likewise constant, which is stronger than ``C``.

Polhode and herpolhode (Poinsot)
--------------------------------
Body-frame ``ω(t)`` therefore stays on the intersection of the inertia
ellipsoid and the angular-momentum ellipsoid

    ωᵀ J ω = 2 T,     (J ω) · (J ω) = |h|².

That space curve is the **polhode**.  Mapped to inertial coordinates,
``ω_I = R(q) ω`` traces the **herpolhode**.  Because ``ω · h = 2 T``
is conserved, ``ω_I`` lies in the invariable plane

    ω_I · h_I = 2 T

(offset ``2 T / |h|`` from the origin along ``ĥ_I``).  For a spherical
inertia both curves degenerate to a point (``ω`` is an equilibrium).

:func:`sample_polhode` integrates the existing plant at ``τ = 0``.
:func:`sample_polhode_intersection` samples the same algebraic curve
geometrically (no time) in the principal frame.

Intermediate-axis (tennis-racket) theorem
-----------------------------------------
Let ``I₁ < I₂ < I₃`` be the principal moments and ``(e₁, e₂, e₃)`` the
corresponding principal axes.  Torque-free spin about ``e₁`` (min) or
``e₃`` (max) is (linearly) stable; spin about the intermediate axis
``e₂`` is unstable.  Linearizing Euler's equation about
``ω = Ω e_k`` with ``{i, j, k} = {1, 2, 3}`` gives a transverse mode

    λ² = − (I_k − I_i)(I_k − I_j) / (I_i I_j)  ·  Ω².

``λ² < 0`` is oscillatory (stable), ``λ² > 0`` is hyperbolic (unstable),
and ``λ² = 0`` is the axisymmetric / spherical degeneracy (marginal).
:func:`principal_spin_stability` uses principal-moment indices
``0 = min``, ``1 = intermediate``, ``2 = max`` from
:func:`attitude_sim.plant.principal_moments_and_axes`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from attitude_sim.plant import (
    RigidBody,
    principal_moments_and_axes,
    step_rigid_body,
    validate_inertia,
)
from attitude_sim.quaternions import quat_normalize, quat_to_rotation

_IDENTITY_Q = np.array([1.0, 0.0, 0.0, 0.0])
_SQ_ATOL = 1e-12
_LAMBDA_ATOL = 1e-12
_REGIME_RTOL = 1e-10


class SpinStability(str, Enum):
    """Linear stability of torque-free principal-axis spin."""

    STABLE = "stable"
    UNSTABLE = "unstable"
    MARGINAL = "marginal"


class PolhodeRegime(str, Enum):
    """Which principal axis a triaxial polhode encircles."""

    AROUND_MIN = "around_min"
    AROUND_MAX = "around_max"
    SEPARATRIX = "separatrix"
    POINT = "point"


@dataclass
class PolhodeTrajectory:
    """Time-sampled torque-free trajectory in body (and optional inertial) frame.

    ``T`` and ``h2`` are the energy and Casimir time series.  When
    ``herpolhode`` was requested, ``q``, ``omega_inertial``, and
    ``h_inertial`` are length-``N`` arrays; otherwise they are ``None``.
    """

    t: np.ndarray
    omega: np.ndarray
    T: np.ndarray
    h2: np.ndarray
    h_body: np.ndarray
    q: np.ndarray | None = None
    omega_inertial: np.ndarray | None = None
    h_inertial: np.ndarray | None = None


def _as_inertia(inertia: np.ndarray | RigidBody) -> np.ndarray:
    if isinstance(inertia, RigidBody):
        return inertia.inertia
    return validate_inertia(inertia)


def _as_omega(omega: np.ndarray) -> np.ndarray:
    w = np.asarray(omega, dtype=float).reshape(3)
    if not np.all(np.isfinite(w)):
        raise ValueError("omega must be finite")
    return w


def _as_quat(q: np.ndarray) -> np.ndarray:
    return quat_normalize(np.asarray(q, dtype=float).reshape(4))


def kinetic_energy(inertia: np.ndarray | RigidBody, omega: np.ndarray) -> float:
    """Rotational kinetic energy ``T = (1/2) ω · J ω``."""
    J = _as_inertia(inertia)
    w = _as_omega(omega)
    return 0.5 * float(w @ (J @ w))


def angular_momentum_body(inertia: np.ndarray | RigidBody, omega: np.ndarray) -> np.ndarray:
    """Body-frame angular momentum ``h_b = J ω``."""
    J = _as_inertia(inertia)
    w = _as_omega(omega)
    return J @ w


def angular_momentum_inertial(
    inertia: np.ndarray | RigidBody,
    q: np.ndarray,
    omega: np.ndarray,
) -> np.ndarray:
    """Inertial-frame angular momentum ``h_I = R(q) J ω``."""
    return quat_to_rotation(_as_quat(q)) @ angular_momentum_body(inertia, omega)


def omega_inertial(q: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Inertial-frame angular velocity ``ω_I = R(q) ω``."""
    return quat_to_rotation(_as_quat(q)) @ _as_omega(omega)


def casimir_h2(inertia: np.ndarray | RigidBody, omega: np.ndarray) -> float:
    """Lie–Poisson Casimir ``C = |h|² = |J ω|²``."""
    h = angular_momentum_body(inertia, omega)
    return float(h @ h)


def energy_casimir(
    inertia: np.ndarray | RigidBody,
    omega: np.ndarray,
) -> tuple[float, float]:
    """Return ``(T, |h|²)``, the torque-free first integrals."""
    J = _as_inertia(inertia)
    w = _as_omega(omega)
    h = J @ w
    t = 0.5 * float(w @ h)
    return t, float(h @ h)


def polhode_residuals(
    inertia: np.ndarray | RigidBody,
    omega: np.ndarray,
    T: float,
    h2: float,
) -> tuple[float, float]:
    """Signed residuals of the energy ellipsoid and momentum sphere.

        r_T  = ωᵀ J ω − 2 T
        r_h  = |J ω|² − |h|²
    """
    t, c = energy_casimir(inertia, omega)
    return 2.0 * t - 2.0 * float(T), c - float(h2)


def herpolhode_plane_residual(
    omega_I: np.ndarray,
    h_I: np.ndarray,
    T: float,
) -> float:
    """Signed residual of the invariable-plane constraint ``ω_I · h_I − 2 T``."""
    w = np.asarray(omega_I, dtype=float).reshape(3)
    h = np.asarray(h_I, dtype=float).reshape(3)
    return float(w @ h) - 2.0 * float(T)


def polhode_regime(
    inertia: np.ndarray | RigidBody,
    omega: np.ndarray,
    *,
    atol: float = 0.0,
) -> PolhodeRegime:
    """Classify which principal axis the polhode through ``ω`` encircles.

    For triaxial ``I_min < I_mid < I_max`` the separatrix is
    ``|h|² = 2 T I_mid``.  Below it the curve loops around the min-inertia
    axis; above it, around the max-inertia axis.  Principal-axis spin
    (vanishing transverse rate) is :attr:`PolhodeRegime.POINT`.
    """
    J = _as_inertia(inertia)
    w = _as_omega(omega)
    moments, axes = principal_moments_and_axes(J)
    w_p = axes.T @ w
    t, h2 = energy_casimir(J, w)
    if t <= 0.0:
        return PolhodeRegime.POINT
    spread = float(moments[-1] - moments[0])
    if spread <= 1e-12 * max(float(moments[-1]), 1.0):
        # Spherical: both quadrics are the same constraint; ω is an equilibrium.
        return PolhodeRegime.POINT
    i_mid = float(moments[1])
    sep = 2.0 * t * i_mid
    scale = max(abs(sep), abs(h2), 1.0)
    tol = max(float(atol), _REGIME_RTOL * scale)
    transverse = w_p.copy()
    k = int(np.argmax(np.abs(w_p)))
    transverse[k] = 0.0
    if float(np.linalg.norm(transverse)) <= 1e-14 * max(float(np.linalg.norm(w_p)), 1.0):
        return PolhodeRegime.POINT
    if h2 > sep + tol:
        return PolhodeRegime.AROUND_MAX
    if h2 < sep - tol:
        return PolhodeRegime.AROUND_MIN
    return PolhodeRegime.SEPARATRIX


def _encircled_axis(moments: np.ndarray, t: float, h2: float) -> int:
    """Principal-moment index (ascending) that the polhode encircles."""
    sep = 2.0 * t * float(moments[1])
    scale = max(abs(sep), abs(h2), 1.0)
    if h2 >= sep - _REGIME_RTOL * scale:
        return 2
    return 0


def sample_polhode_intersection(
    inertia: np.ndarray | RigidBody,
    omega0: np.ndarray,
    *,
    n: int = 180,
) -> np.ndarray:
    """Geometrically sample the energy–Casimir intersection through ``ω0``.

    Works in the principal frame.  For a closed polhode around the min or
    max principal axis, each azimuth ``θ`` in the transverse plane gives a
    2×2 linear system for ``(ω_k², ρ²)``.  The branch keeps the sign of
    ``ω0`` along the encircled axis.  Returns shape ``(n, 3)`` in the
    **body** frame.  Principal-axis spin and spherical inertia collapse
    to copies of ``ω0`` (the algebraic intersection is a point, or the
    whole sphere — physically ``ω`` is constant).
    """
    n = int(n)
    if n < 1:
        raise ValueError("n must be >= 1")
    J = _as_inertia(inertia)
    w0 = _as_omega(omega0)
    moments, axes = principal_moments_and_axes(J)
    w_p0 = axes.T @ w0
    t, h2 = energy_casimir(J, w0)
    out = np.repeat(w0.reshape(1, 3), n, axis=0)
    if t <= 0.0:
        return out

    if polhode_regime(J, w0) is PolhodeRegime.POINT:
        return out

    k = _encircled_axis(moments, t, h2)
    i, j = (k + 1) % 3, (k + 2) % 3
    I_k = float(moments[k])
    I_i = float(moments[i])
    I_j = float(moments[j])
    rho0 = float(np.hypot(w_p0[i], w_p0[j]))
    theta0 = float(np.arctan2(w_p0[j], w_p0[i])) if rho0 > 0.0 else 0.0
    sign_k = 1.0 if w_p0[k] >= 0.0 else -1.0
    thetas = theta0 + np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    rhs = np.array([2.0 * t, h2], dtype=float)

    pts = np.empty((n, 3), dtype=float)
    for idx, theta in enumerate(thetas):
        c2 = float(np.cos(theta) ** 2)
        s2 = float(np.sin(theta) ** 2)
        a12 = I_i * c2 + I_j * s2
        a22 = I_i * I_i * c2 + I_j * I_j * s2
        A = np.array([[I_k, a12], [I_k * I_k, a22]], dtype=float)
        try:
            wk2, rho2 = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            pts[idx] = w_p0
            continue
        if wk2 < -_SQ_ATOL or rho2 < -_SQ_ATOL:
            pts[idx] = w_p0
            continue
        wk2 = max(0.0, float(wk2))
        rho2 = max(0.0, float(rho2))
        rho = float(np.sqrt(rho2))
        w_p = np.zeros(3)
        w_p[k] = sign_k * float(np.sqrt(wk2))
        w_p[i] = rho * float(np.cos(theta))
        w_p[j] = rho * float(np.sin(theta))
        pts[idx] = w_p

    return (axes @ pts.T).T


def sample_polhode(
    inertia: np.ndarray | RigidBody,
    omega0: np.ndarray,
    t_final: float,
    dt: float,
    *,
    q0: np.ndarray | None = None,
    herpolhode: bool = False,
    method: str = "rk4",
) -> PolhodeTrajectory:
    """Integrate torque-free Euler / quaternion kinematics and sample the polhode.

    ``ω(t)`` is body-frame.  With ``herpolhode=True`` the returned
    trajectory also stores ``q(t)``, ``ω_I(t) = R(q) ω``, and ``h_I(t)``.
    ``dt`` must be finite and strictly positive; ``t_final`` must be
    finite and ``>= 0``.
    """
    t_final = float(t_final)
    dt = float(dt)
    if not np.isfinite(t_final) or t_final < 0.0:
        raise ValueError("t_final must be finite and >= 0")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be positive")
    J = _as_inertia(inertia)
    w = _as_omega(omega0).copy()
    q = _as_quat(_IDENTITY_Q if q0 is None else q0)
    body = inertia if isinstance(inertia, RigidBody) else RigidBody(J)
    n_steps = int(np.round(t_final / dt)) if t_final > 0.0 else 0
    n = n_steps + 1
    t = np.arange(n, dtype=float) * dt
    omega = np.empty((n, 3), dtype=float)
    h_body = np.empty((n, 3), dtype=float)
    T = np.empty(n, dtype=float)
    h2 = np.empty(n, dtype=float)
    q_hist: np.ndarray | None = np.empty((n, 4), dtype=float) if herpolhode else None
    w_I: np.ndarray | None = np.empty((n, 3), dtype=float) if herpolhode else None
    h_I: np.ndarray | None = np.empty((n, 3), dtype=float) if herpolhode else None
    tau0 = np.zeros(3)

    for k in range(n):
        omega[k] = w
        h_body[k] = body.angular_momentum_body(w)
        T[k] = 0.5 * float(w @ h_body[k])
        h2[k] = float(h_body[k] @ h_body[k])
        if herpolhode:
            assert q_hist is not None and w_I is not None and h_I is not None
            q_hist[k] = q
            R = quat_to_rotation(q)
            w_I[k] = R @ w
            h_I[k] = R @ h_body[k]
        if k < n_steps:
            q, w = step_rigid_body(body, q, w, tau0, dt, method=method)

    return PolhodeTrajectory(
        t=t,
        omega=omega,
        T=T,
        h2=h2,
        h_body=h_body,
        q=q_hist,
        omega_inertial=w_I,
        h_inertial=h_I,
    )


def sample_herpolhode(
    inertia: np.ndarray | RigidBody,
    omega0: np.ndarray,
    t_final: float,
    dt: float,
    *,
    q0: np.ndarray | None = None,
    method: str = "rk4",
) -> PolhodeTrajectory:
    """:func:`sample_polhode` with inertial ``ω_I`` / ``h_I`` filled in."""
    return sample_polhode(
        inertia,
        omega0,
        t_final,
        dt,
        q0=q0,
        herpolhode=True,
        method=method,
    )


def transverse_lambda_squared(
    inertia: np.ndarray | RigidBody,
    axis: int,
    spin_rate: float = 1.0,
) -> float:
    """Transverse eigenvalue squared for principal-axis spin ``ω = Ω e_k``.

        λ² = − (I_k − I_i)(I_k − I_j) / (I_i I_j) · Ω²

    ``axis`` is the principal-moment index from
    :func:`principal_moments_and_axes` (``0`` = min, ``1`` = intermediate,
    ``2`` = max).  ``λ² < 0`` oscillatory, ``λ² > 0`` hyperbolic.
    """
    k = int(axis)
    if k not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2 (principal-moment index, ascending)")
    omega = float(spin_rate)
    if not np.isfinite(omega):
        raise ValueError("spin_rate must be finite")
    J = _as_inertia(inertia)
    moments, _ = principal_moments_and_axes(J)
    i, j = (k + 1) % 3, (k + 2) % 3
    i_k = float(moments[k])
    i_i = float(moments[i])
    i_j = float(moments[j])
    return -((i_k - i_i) * (i_k - i_j) / (i_i * i_j)) * omega * omega


def principal_spin_stability(
    inertia: np.ndarray | RigidBody,
    axis: int,
) -> SpinStability:
    """Tennis-racket / intermediate-axis classification for principal axis ``axis``.

    ``axis`` is ``0`` (min moment, stable), ``1`` (intermediate, unstable
    if triaxial), or ``2`` (max moment, stable).  Repeated principal
    moments yield :attr:`SpinStability.MARGINAL`.  The linear test is
    independent of the spin rate except the degenerate rest case
    ``Ω = 0``, which is not used here (``Ω = 1``).
    """
    lam = transverse_lambda_squared(inertia, axis, spin_rate=1.0)
    if abs(lam) <= _LAMBDA_ATOL:
        return SpinStability.MARGINAL
    if lam < 0.0:
        return SpinStability.STABLE
    return SpinStability.UNSTABLE


def intermediate_axis_theorem(
    inertia: np.ndarray | RigidBody,
) -> tuple[SpinStability, SpinStability, SpinStability]:
    """Stability of (min, intermediate, max) principal-axis spin."""
    return (
        principal_spin_stability(inertia, 0),
        principal_spin_stability(inertia, 1),
        principal_spin_stability(inertia, 2),
    )


def intermediate_principal_axis(inertia: np.ndarray | RigidBody) -> np.ndarray:
    """Body-frame unit vector along the intermediate principal axis.

    For a triaxial inertia this is the tennis-racket (unstable) axis.
    Axisymmetric bodies have a degenerate mid eigenspace; ``eigh`` then
    returns one orthonormal completion.
    """
    J = _as_inertia(inertia)
    _, axes = principal_moments_and_axes(J)
    return axes[:, 1].copy()

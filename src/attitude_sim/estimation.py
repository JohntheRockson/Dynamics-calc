"""Attitude / rate estimators: Mahony complementary filter and MEKF.

Both consume a noisy gyro and optional unit-vector measurements
(magnetometer, sun sensor) and return ``(q̂, ω̂)`` for the controller.

Quaternion convention (this repo)
---------------------------------
Scalar-first ``q = [w, x, y, z]``.  ``q`` maps body → inertial:

    v_N = q ⊗ v_B ⊗ q*    ⇔    v_I = R(q) v_B

Error-state MEKF and Mahony both use the *body-frame* multiplicative
error ``q = q̂ ⊗ δq(δα)`` with ``δq ≈ [1, δα/2]`` (right multiplication).

Gyro process / measurement noise
--------------------------------
The filter's process noise matches the Farrenkopf gyro used in
``sensors.GyroModel``:

    ω_m = ω + b + η_v ,   ḃ = η_u

``σ_v`` (ARW, rad/s/√Hz) and ``σ_u`` (RRW, rad/s²/√Hz) enter the
discrete 6-state covariance through ``farrenkopf_Qd`` (see
``docs/estimation.md``).  Vector sensors contribute measurement
covariance on the unit-sphere tangent plane.

Truth is sampled from the plant ``(q, ω)`` pair returned by
``step_rigid_body``; this module does not integrate Euler's equation.

TRIAD (``triad`` / ``triad_from_meas``) is a two-vector coarse attitude
for lost-in-space init; it does not replace MEKF or Mahony.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from attitude_sim.quaternions import (
    axis_angle_to_quat,
    quat_integrate_const_omega,
    quat_multiply,
    quat_normalize,
    quat_to_rotation,
    rotation_to_quat,
    skew,
)
from attitude_sim.sensors import VectorSensor

VectorMeas = tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, float]


def farrenkopf_Qd(sigma_v: float, sigma_u: float, dt: float) -> np.ndarray:
    """Discrete process noise for ``x = [δα, δb]`` (Farrenkopf gyro).

    Integrating ``ẋ = F x + G w`` with ``F ≈ [[0, -I], [0, 0]]``
    (the ``-ω̂×`` kinematics rotation is applied in ``Phi``, not in
    ``Q``) and ``G = diag(-I, I)`` gives, over a sample ``dt``,

        Q_αα = (σ_v² dt + σ_u² dt³ / 3) I
        Q_αb = − (σ_u² dt² / 2) I
        Q_bb = (σ_u² dt) I
    """
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    sv2 = float(sigma_v) ** 2
    su2 = float(sigma_u) ** 2
    qaa = (sv2 * dt + su2 * dt**3 / 3.0) * np.eye(3)
    qab = -0.5 * su2 * dt**2 * np.eye(3)
    qbb = su2 * dt * np.eye(3)
    Q = np.zeros((6, 6))
    Q[:3, :3] = qaa
    Q[:3, 3:] = qab
    Q[3:, :3] = qab
    Q[3:, 3:] = qbb
    return Q


def mekf_stm(omega_hat: np.ndarray, dt: float) -> np.ndarray:
    """State transition ``Φ = exp(F dt)`` for ``F = [[-ω̂×, -I], [0, 0]]``.

    Closed form: ``Φ_αα = exp(-[ω̂×] dt)``, ``Φ_αb = −∫_0^dt exp(-[ω̂×] τ) dτ``.
    """
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    w = np.asarray(omega_hat, dtype=float).reshape(3)
    wn = float(np.linalg.norm(w))
    theta = wn * dt
    if theta < 1e-10:
        Wx = skew(w)
        Phi_aa = np.eye(3) - Wx * dt + 0.5 * (Wx @ Wx) * (dt**2)
        Gamma = np.eye(3) * dt - 0.5 * Wx * (dt**2) + (Wx @ Wx) * (dt**3) / 6.0
    else:
        u = w / wn
        ux = skew(u)
        ux2 = ux @ ux
        # exp(-[ω×] dt) = I − sinθ [u×] + (1−cosθ) [u×]²
        Phi_aa = np.eye(3) - np.sin(theta) * ux + (1.0 - np.cos(theta)) * ux2
        # ∫_0^dt exp(-[ω×] τ) dτ
        Gamma = (
            np.eye(3) * dt
            - ((1.0 - np.cos(theta)) / wn) * ux
            + (dt - np.sin(theta) / wn) * ux2
        )
    Phi = np.eye(6)
    Phi[:3, :3] = Phi_aa
    Phi[:3, 3:] = -Gamma
    return Phi


def _unit3(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(v))
    if n < 1e-15:
        raise ValueError("vector measurement must be non-zero")
    return v / n


def iter_vector_meas(
    vector_meas: Sequence[VectorMeas] | None,
) -> list[tuple[np.ndarray, np.ndarray, float | None]]:
    """Normalize ``(v_b, v_I[, sigma])`` tuples from sensors or tests."""
    if not vector_meas:
        return []
    out: list[tuple[np.ndarray, np.ndarray, float | None]] = []
    for item in vector_meas:
        if len(item) == 2:
            v_b, v_I = item
            sigma: float | None = None
        else:
            v_b, v_I, sigma = item[0], item[1], float(item[2])
        out.append((_unit3(v_b), _unit3(v_I), sigma))
    return out


def _inject_body_error(q: np.ndarray, dtheta: np.ndarray) -> np.ndarray:
    """Right-multiply ``q ← q ⊗ δq(δα)`` with an exact axis-angle ``δq``."""
    dtheta = np.asarray(dtheta, dtype=float).reshape(3)
    angle = float(np.linalg.norm(dtheta))
    dq = axis_angle_to_quat(dtheta, angle)
    return quat_normalize(quat_multiply(q, dq))


@dataclass
class ComplementaryFilter:
    """SO(3) Mahony complementary filter with gyro-bias estimation.

    Nominal kinematics are integrated with the *corrected* rate

        ω_kin = ω_m − b̂ + k_p ω_mes

    while the rate returned to the controller is the unbiased gyro

        ω̂ = ω_m − b̂

    Vector innovations ``ω_mes = Σ w_i (v_b × v̂_b) / Σ w_i`` pull the
    estimate toward the observations (``v_b × v̂ ≈ δα_⊥`` for the
    body-frame error ``q = q̂ ⊗ δq``).  Bias: ``ḃ̂ = −k_i ω_mes``.
    Weights default to equal; pass a per-vector ``sigma`` to weight by
    ``1/σ²``.
    """

    kp: float = 1.5
    ki: float = 0.08
    q: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    bias: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self) -> None:
        self.q = quat_normalize(self.q)
        self.bias = np.asarray(self.bias, dtype=float).reshape(3).copy()

    def reset(self, q: np.ndarray | None = None, bias: np.ndarray | None = None) -> None:
        if q is not None:
            self.q = quat_normalize(q)
        if bias is not None:
            self.bias = np.asarray(bias, dtype=float).reshape(3).copy()

    def step(
        self,
        omega_m: np.ndarray,
        dt: float,
        vector_meas: Sequence[VectorMeas] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        omega_m = np.asarray(omega_m, dtype=float).reshape(3)
        omega_corr = np.zeros(3)
        meas = iter_vector_meas(vector_meas)
        if meas:
            R = quat_to_rotation(self.q)
            wsum = 0.0
            acc = np.zeros(3)
            for v_b, v_I, sigma in meas:
                v_hat = R.T @ v_I
                vn = float(np.linalg.norm(v_hat))
                if vn < 1e-15:
                    continue
                v_hat = v_hat / vn
                weight = 1.0 if sigma is None or sigma <= 0.0 else 1.0 / (sigma * sigma)
                # v_meas × v̂ ≈ δα_perp for q = q̂ ⊗ δq(δα); add to ω_kin so q̂ catches up.
                acc = acc + weight * np.cross(v_b, v_hat)
                wsum += weight
            if wsum > 0.0:
                omega_corr = acc / wsum
        omega_hat = omega_m - self.bias
        omega_kin = omega_hat + self.kp * omega_corr
        self.bias = self.bias - self.ki * omega_corr * dt
        self.q = quat_integrate_const_omega(self.q, omega_kin, dt)
        return self.q.copy(), omega_hat.copy()


@dataclass
class MultiplicativeEKF:
    """6-state multiplicative EKF: attitude error ``δα`` and gyro bias.

    Error definition ``q = q̂ ⊗ δq(δα)`` with ``δq ≈ [1, δα/2]``.
    Gyro model ``ω_m = ω + b + η_v``, ``ḃ = η_u``.  Vector updates use
    ``H = [[v̂_b ×], 0]`` so that ``v_b − v̂_b ≈ [v̂_b ×] δα``.

    Prediction uses the closed-form STM ``mekf_stm`` and Farrenkopf
    discrete process noise ``farrenkopf_Qd(σ_v, σ_u, dt)``.  Sequential
    Joseph updates apply each unit-vector observation with a rank-2
    tangent-plane ``R``.
    """

    sigma_v: float = 5e-4
    sigma_u: float = 1e-6
    q: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    P: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.q = quat_normalize(self.q)
        self.bias = np.asarray(self.bias, dtype=float).reshape(3).copy()
        if self.P is None:
            # ~3 deg attitude 1σ, ~3 mrad/s bias 1σ (honest vs a few-mrad gyro bias).
            self.P = np.diag([3e-3, 3e-3, 3e-3, 1e-5, 1e-5, 1e-5])
        else:
            self.P = np.asarray(self.P, dtype=float).reshape(6, 6).copy()
        self.P = 0.5 * (self.P + self.P.T)

    def reset(
        self,
        q: np.ndarray | None = None,
        bias: np.ndarray | None = None,
        P: np.ndarray | None = None,
    ) -> None:
        if q is not None:
            self.q = quat_normalize(q)
        if bias is not None:
            self.bias = np.asarray(bias, dtype=float).reshape(3).copy()
        if P is not None:
            self.P = 0.5 * (np.asarray(P, dtype=float).reshape(6, 6) + np.asarray(P, dtype=float).reshape(6, 6).T)

    def predict(self, omega_m: np.ndarray, dt: float) -> np.ndarray:
        omega_m = np.asarray(omega_m, dtype=float).reshape(3)
        omega_hat = omega_m - self.bias
        self.q = quat_integrate_const_omega(self.q, omega_hat, dt)

        Phi = mekf_stm(omega_hat, dt)
        Qd = farrenkopf_Qd(self.sigma_v, self.sigma_u, dt)
        assert self.P is not None
        self.P = Phi @ self.P @ Phi.T + Qd
        self.P = 0.5 * (self.P + self.P.T)
        return omega_hat

    def update_vector(self, v_b_meas: np.ndarray, v_inertial: np.ndarray, sigma: float) -> None:
        v_I = _unit3(v_inertial)
        v_b_meas = _unit3(v_b_meas)
        v_hat = quat_to_rotation(self.q).T @ v_I
        v_hat = _unit3(v_hat)

        H = np.zeros((3, 6))
        # v_b ≈ v̂ + [v̂×] δα  for q = q̂ ⊗ δq(δα)
        H[0:3, 0:3] = skew(v_hat)
        # Rank-2 unit-vector noise (tangent to the sphere) plus a nugget.
        sig2 = float(sigma) ** 2
        R = sig2 * (np.eye(3) - np.outer(v_hat, v_hat)) + (1e-12 * max(sig2, 1.0)) * np.eye(3)
        assert self.P is not None
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.solve(S, np.eye(3))
        dx = K @ (v_b_meas - v_hat)
        I_KH = np.eye(6) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        self._inject(dx)

    def _inject(self, dx: np.ndarray) -> None:
        self.q = _inject_body_error(self.q, dx[:3])
        self.bias = self.bias + np.asarray(dx[3:6], dtype=float).reshape(3)

    def step(
        self,
        omega_m: np.ndarray,
        dt: float,
        vector_meas: Sequence[VectorMeas] | None = None,
        vector_sigma: float = 2e-3,
    ) -> tuple[np.ndarray, np.ndarray]:
        omega_hat = self.predict(omega_m, dt)
        for v_b, v_I, sigma in iter_vector_meas(vector_meas):
            self.update_vector(v_b, v_I, vector_sigma if sigma is None else sigma)
        omega_hat = np.asarray(omega_m, dtype=float).reshape(3) - self.bias
        return self.q.copy(), omega_hat.copy()


def make_estimator(
    mode: str,
    q0: np.ndarray | None = None,
    **kwargs,
) -> ComplementaryFilter | MultiplicativeEKF | None:
    mode = mode.lower()
    if mode == "truth":
        return None
    q0 = np.array([1.0, 0.0, 0.0, 0.0]) if q0 is None else q0
    if mode in {"mahony", "complementary", "cf"}:
        return ComplementaryFilter(q=q0, **kwargs)
    if mode in {"mekf", "kalman", "ekf"}:
        return MultiplicativeEKF(q=q0, **kwargs)
    raise ValueError(f"unknown estimator {mode!r}; use 'truth', 'mekf', or 'mahony'")


def vectors_from_sensors(
    q: np.ndarray,
    sensors: list[VectorSensor],
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Sample each vector sensor and attach its Cartesian ``sigma`` for R."""
    return [(s.measure(q), s.v_inertial, s.sigma) for s in sensors]


_TRIAD_ALIGN_EPS = 1e-8


def triad(
    v1_body: np.ndarray,
    v1_inertial: np.ndarray,
    v2_body: np.ndarray,
    v2_inertial: np.ndarray,
) -> np.ndarray:
    """Coarse attitude from two vector observations (TRIAD).

    The first pair is the **primary** (more trusted) observation.  TRIAD
    builds right-handed frames

        t1 = v1 ,   t2 = (v1 × v2) / ‖v1 × v2‖ ,   t3 = t1 × t2

    in body and inertial coordinates and returns the scalar-first
    quaternion ``q`` with ``v_I = R(q) v_B`` (this repo's convention).
    Equivalent DCM: ``R = M_I M_Bᵀ`` with columns ``(t1, t2, t3)``.

    Raises ``ValueError`` if a pair is degenerate or the two directions
    are (anti)parallel, so a unique TRIAD frame does not exist.
    """
    b1 = _unit3(v1_body)
    r1 = _unit3(v1_inertial)
    b2 = _unit3(v2_body)
    r2 = _unit3(v2_inertial)
    tb = np.cross(b1, b2)
    tn = np.cross(r1, r2)
    nb = float(np.linalg.norm(tb))
    nn = float(np.linalg.norm(tn))
    if nb < _TRIAD_ALIGN_EPS or nn < _TRIAD_ALIGN_EPS:
        raise ValueError("TRIAD needs two non-parallel vector observations")
    t2b = tb / nb
    t2n = tn / nn
    t3b = np.cross(b1, t2b)
    t3n = np.cross(r1, t2n)
    M_b = np.column_stack((b1, t2b, t3b))
    M_n = np.column_stack((r1, t2n, t3n))
    # A maps inertial → body (b = A r); this repo's R is Aᵀ (v_I = R v_B).
    R = M_n @ M_b.T
    return rotation_to_quat(R)


def triad_from_meas(vector_meas: Sequence[VectorMeas]) -> np.ndarray:
    """TRIAD from ``(v_b, v_I[, sigma])`` pairs.  First two observations are used.

    When both pairs carry a ``sigma``, the lower-σ observation is the
    TRIAD primary (sun before mag with the default SimLab stubs).
    """
    meas = iter_vector_meas(vector_meas)
    if len(meas) < 2:
        raise ValueError("TRIAD needs two non-parallel vector observations")

    def _sigma_key(item: tuple[np.ndarray, np.ndarray, float | None]) -> float:
        sigma = item[2]
        return float("inf") if sigma is None else float(sigma)

    # Stable: unspecified σ keeps input order; specified σ prefers accuracy.
    ordered = sorted(meas[:2], key=_sigma_key)
    v1_b, v1_I, _ = ordered[0]
    v2_b, v2_I, _ = ordered[1]
    return triad(v1_b, v1_I, v2_b, v2_I)

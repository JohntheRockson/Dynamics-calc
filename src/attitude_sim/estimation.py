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

Coarse attitude: ``triad_attitude`` / ``triad_q0_from_sensors`` (Wahba
TRIAD).  Consistency: ``mekf_error_state``, ``nees``,
``chi2_mean_nees_bounds`` (see ``docs/estimation.md``).
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
    rotation_vector_error,
    skew,
)
from attitude_sim.sensors import VectorSensor

# 6-state MEKF error x = [δα, δb]; consistent NEES is χ² with this many dof.
MEKF_NEES_DOF = 6
_TRIAD_PARALLEL_EPS = 1e-8

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


def triad_attitude(
    v_b1: np.ndarray,
    v_I1: np.ndarray,
    v_b2: np.ndarray,
    v_I2: np.ndarray,
) -> np.ndarray:
    """Wahba TRIAD: two body/inertial unit-vector pairs → scalar-first ``q``.

    Builds orthonormal triads in each frame (first pair is the primary
    observation) and the unique rotation ``v_I = R(q) v_B`` that maps the
    body triad onto the inertial triad.  Vectors must be non-parallel.
    """
    b1 = _unit3(v_b1)
    r1 = _unit3(v_I1)
    b2 = _unit3(v_b2)
    r2 = _unit3(v_I2)
    tb = np.cross(b1, b2)
    tr = np.cross(r1, r2)
    nb = float(np.linalg.norm(tb))
    nr = float(np.linalg.norm(tr))
    if nb < _TRIAD_PARALLEL_EPS or nr < _TRIAD_PARALLEL_EPS:
        raise ValueError("TRIAD reference vectors must be non-parallel")
    tb = tb / nb
    tr = tr / nr
    body = np.column_stack((b1, tb, np.cross(b1, tb)))
    inertial = np.column_stack((r1, tr, np.cross(r1, tr)))
    return rotation_to_quat(inertial @ body.T)


def triad_q0_from_sensors(q: np.ndarray, sensors: list[VectorSensor]) -> np.ndarray:
    """Coarse ``q`` from the first two *available* mag/sun (or other) stubs.

    Samples each sensor at the true attitude ``q`` (the only truth the
    sensors ever see).  Occulted / out-of-FOV sensors are skipped.
    Raises ``ValueError`` if fewer than two measurements are available.
    """
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for sensor in sensors:
        v_b = sensor.measure(q)
        if v_b is None:
            continue
        pairs.append((v_b, sensor.v_inertial))
        if len(pairs) >= 2:
            break
    if len(pairs) < 2:
        raise ValueError("TRIAD coarse init needs two available vector sensors")
    (v_b1, v_I1), (v_b2, v_I2) = pairs
    return triad_attitude(v_b1, v_I1, v_b2, v_I2)


def mekf_error_state(
    q_hat: np.ndarray,
    bias_hat: np.ndarray,
    q_true: np.ndarray,
    bias_true: np.ndarray,
) -> np.ndarray:
    """MEKF error ``x = [δα, δb]`` for ``q = q̂ ⊗ δq(δα)``, ``δb = b − b̂``.

    ``δα`` is the body-frame rotation vector
    ``2 sign(δq_w) δq_{1:3}`` with ``δq = q̂* ⊗ q``.
    """
    dalpha = np.asarray(rotation_vector_error(q_true, q_hat), dtype=float).reshape(3)
    db = np.asarray(bias_true, dtype=float).reshape(3) - np.asarray(bias_hat, dtype=float).reshape(3)
    return np.concatenate([dalpha, db])


def nees(x: np.ndarray, P: np.ndarray) -> float:
    """Normalized estimation error squared ``xᵀ P⁻¹ x``."""
    x = np.asarray(x, dtype=float).reshape(-1)
    P = np.asarray(P, dtype=float)
    if P.ndim != 2 or P.shape != (x.size, x.size):
        raise ValueError(f"P shape {P.shape} does not match error length {x.size}")
    return float(x @ np.linalg.solve(P, x))


def chi2_mean_nees_bounds(
    dof: int,
    n_trials: int,
    alpha: float = 0.01,
) -> tuple[float, float]:
    """Two-sided bounds on the mean of ``n_trials`` i.i.d. ``χ²_dof`` samples.

    If ``ε_i ~ χ²_ν`` independently, then ``n ε̄ ~ χ²_{nν}``, so at
    confidence ``1 − α``

        ``ε̄ ∈ [ χ²_{nν}(α/2) / n ,  χ²_{nν}(1 − α/2) / n ]``.

    A consistent 6-state MEKF has ``E[NEES] = 6``.  For ``N = 32`` and
    ``α = 0.01`` the interval is about ``[4.54, 7.69]``.
    """
    from scipy.stats import chi2

    if dof <= 0 or n_trials <= 0:
        raise ValueError("dof and n_trials must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    df = int(dof) * int(n_trials)
    lo = float(chi2.ppf(0.5 * alpha, df)) / n_trials
    hi = float(chi2.ppf(1.0 - 0.5 * alpha, df)) / n_trials
    return lo, hi


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
    """Sample each *available* vector sensor and attach Cartesian ``sigma``.

    Occulted / out-of-FOV stubs (``measure`` returns ``None``) are skipped
    so the filter runs gyro-only on that sample.
    """
    out: list[tuple[np.ndarray, np.ndarray, float]] = []
    for sensor in sensors:
        v_b = sensor.measure(q)
        if v_b is None:
            continue
        out.append((v_b, sensor.v_inertial, sensor.sigma))
    return out

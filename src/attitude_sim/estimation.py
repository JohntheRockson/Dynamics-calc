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
``chi2_mean_nees_bounds`` (state / NEES) and ``nis``,
``mekf_vector_nis``, ``InnovationLog``, ``chi2_mean_nis_bounds``
(measurement / rank-2 NIS).  See ``docs/estimation.md``.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

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
# Unit-vector updates are rank-2 (tangent to the sphere); NIS is χ²_2.
VECTOR_NIS_DOF = 2
# Small diagonal added to the rank-2 R so S is numerically SPD in 3-D.
MEKF_VECTOR_R_NUGGET = 1e-12
_TRIAD_PARALLEL_EPS = 1e-8

VectorMeas = (
    tuple[np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, float]
    | tuple[np.ndarray, np.ndarray, float, str]
)


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
) -> list[tuple[np.ndarray, np.ndarray, float | None, str]]:
    """Normalize ``(v_b, v_I[, sigma[, name]])`` tuples from sensors or tests."""
    if not vector_meas:
        return []
    out: list[tuple[np.ndarray, np.ndarray, float | None, str]] = []
    for item in vector_meas:
        name = "vector"
        if len(item) == 2:
            v_b, v_I = item[0], item[1]
            sigma: float | None = None
        elif len(item) == 3:
            v_b, v_I, sigma = item[0], item[1], float(item[2])
        elif len(item) == 4:
            v_b, v_I, sigma, name = item[0], item[1], float(item[2]), str(item[3])
        else:
            raise ValueError("vector meas must be (v_b, v_I[, sigma[, name]])")
        out.append((_unit3(v_b), _unit3(v_I), sigma, name))
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


def chi2_two_sided_bounds(dof: int, alpha: float = 0.01) -> tuple[float, float]:
    """Single-sample two-sided ``χ²_dof`` interval at confidence ``1 − α``.

    Returns ``(χ²_ν(α/2), χ²_ν(1 − α/2))``.  A consistent 6-state NEES
    uses ``ν = 6``; a rank-2 unit-vector NIS uses ``ν = 2``.
    """
    from scipy.stats import chi2

    if dof <= 0:
        raise ValueError("dof must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    lo = float(chi2.ppf(0.5 * alpha, int(dof)))
    hi = float(chi2.ppf(1.0 - 0.5 * alpha, int(dof)))
    return lo, hi


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
    ``α = 0.01`` the interval is about ``[4.54, 7.69]``.  Rank-2 NIS
    uses the same formula with ``dof = VECTOR_NIS_DOF`` (``E[NIS] = 2``).
    """
    if dof <= 0 or n_trials <= 0:
        raise ValueError("dof and n_trials must be positive")
    lo, hi = chi2_two_sided_bounds(int(dof) * int(n_trials), alpha)
    return lo / n_trials, hi / n_trials


# Mean-NIS uses the same χ² averaging as mean-NEES (different dof).
chi2_mean_nis_bounds = chi2_mean_nees_bounds


def nis(nu: np.ndarray, S: np.ndarray) -> float:
    """Normalized innovation squared ``νᵀ S⁻¹ ν``.

    Same quadratic form as ``nees``; the name marks a *measurement*
    residual against its predicted covariance ``S = H P Hᵀ + R``.
    """
    return nees(nu, S)


def mekf_vector_HR(v_hat: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """``H`` (3×6) and rank-2 tangent-plane ``R`` matching ``update_vector``.

    ``H = [[v̂× | 0]]`` so ``v_b − v̂ ≈ [v̂×] δα``, and

        ``R = σ² (I − v̂ v̂ᵀ) + ε max(σ², 1) I``

    with ``ε = MEKF_VECTOR_R_NUGGET``.  Extracted so NIS uses the same
    matrices as the Joseph update (kinematics are unchanged).
    """
    v_hat = _unit3(v_hat)
    H = np.zeros((3, 6))
    H[0:3, 0:3] = skew(v_hat)
    sig2 = float(sigma) ** 2
    nugget = MEKF_VECTOR_R_NUGGET * max(sig2, 1.0)
    R = sig2 * (np.eye(3) - np.outer(v_hat, v_hat)) + nugget * np.eye(3)
    return H, R


def mekf_predicted_body_vector(q_hat: np.ndarray, v_inertial: np.ndarray) -> np.ndarray:
    """Predicted body-frame unit vector ``v̂_b = R(q̂)ᵀ v_I``."""
    return _unit3(quat_to_rotation(q_hat).T @ _unit3(v_inertial))


def tangent_plane_basis(v: np.ndarray) -> np.ndarray:
    """Orthonormal 3×2 basis spanning the plane orthogonal to unit ``v``.

    Columns ``(e1, e2)`` satisfy ``e_i · v = 0`` and ``e1 × e2`` is
    parallel to ``v``.  Used to reduce rank-2 unit-vector innovations
    to a well-conditioned 2-vector (NIS is then ``χ²_2``, not ``χ²_3``).
    """
    v = _unit3(v)
    k = int(np.argmin(np.abs(v)))
    axis = np.zeros(3)
    axis[k] = 1.0
    e1 = np.cross(v, axis)
    n1 = float(np.linalg.norm(e1))
    if n1 < 1e-15:
        raise ValueError("failed to build a tangent-plane basis")
    e1 = e1 / n1
    e2 = np.cross(v, e1)
    e2 = e2 / float(np.linalg.norm(e2))
    return np.column_stack((e1, e2))


def lag1_sample_correlation(series: np.ndarray) -> np.ndarray:
    """Lag-1 Pearson correlation of each column of an ``(N, m)`` series.

    A white sequence has ``E[r] = 0`` and large-``N`` variance
    ``≈ 1/(N−1)`` per component (see ``lag1_whiteness_bound``).
    """
    x = np.asarray(series, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.ndim != 2:
        raise ValueError("series must be 1-D or 2-D")
    n, m = x.shape
    if n < 3:
        raise ValueError("need at least 3 samples for lag-1 correlation")
    a = x[:-1] - x[:-1].mean(axis=0)
    b = x[1:] - x[1:].mean(axis=0)
    num = np.sum(a * b, axis=0)
    den = np.sqrt(np.sum(a * a, axis=0) * np.sum(b * b, axis=0))
    out = np.zeros(m)
    ok = den > 1e-15
    out[ok] = num[ok] / den[ok]
    return out


def lag1_whiteness_bound(n: int, alpha: float = 0.05) -> float:
    """Approximate two-sided ``|r|`` bound ``z_{1−α/2} / √(n−1)``.

    Large-``n`` Gaussian approximation for the lag-1 sample correlation
    of a white scalar sequence.  For ``n = 101`` and 95% this is
    ``≈ 1.96 / 10``.
    """
    from scipy.stats import norm

    if n < 3:
        raise ValueError("n must be >= 3")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    z = float(norm.ppf(1.0 - 0.5 * alpha))
    return z / float(np.sqrt(n - 1))


@dataclass
class InnovationSample:
    """One pre-update unit-vector innovation (rank-2 tangent plane)."""

    nis: float
    dof: int
    nu: np.ndarray
    nu_tangent: np.ndarray
    S_tangent: np.ndarray
    sensor: str = "vector"
    t: float | None = None
    index: int = 0
    v_hat: np.ndarray | None = None

    def whitened(self) -> np.ndarray:
        """``S₂^{−1/2} ν₂``; a consistent filter has these ~ i.i.d. ``N(0, I₂)``."""
        S = 0.5 * (np.asarray(self.S_tangent, dtype=float) + np.asarray(self.S_tangent, dtype=float).T)
        nu2 = np.asarray(self.nu_tangent, dtype=float).reshape(2)
        try:
            L = np.linalg.cholesky(S)
            return np.linalg.solve(L, nu2)
        except np.linalg.LinAlgError:
            evals, evecs = np.linalg.eigh(S)
            evals = np.clip(evals, 1e-18, None)
            return evecs @ ((evecs.T @ nu2) / np.sqrt(evals))


@dataclass
class InnovationLog:
    """Optional recorder for MEKF vector NIS / whitened innovations.

    Attach with ``MultiplicativeEKF.enable_innovation_log()`` (default
    off).  Each ``update_vector`` appends a pre-Joseph sample so NIS
    uses the same ``S`` as the Kalman gain.  ``write_csv`` dumps a
    table for offline plots; it does not change filter kinematics.
    """

    samples: list[InnovationSample] = field(default_factory=list)

    def clear(self) -> None:
        self.samples.clear()

    def record_from_update(
        self,
        *,
        nu: np.ndarray,
        v_hat: np.ndarray,
        S: np.ndarray,
        sensor: str = "vector",
        t: float | None = None,
    ) -> InnovationSample:
        """Project the 3-D residual onto the tangent plane and store NIS."""
        nu = np.asarray(nu, dtype=float).reshape(3)
        v_hat = _unit3(v_hat)
        B = tangent_plane_basis(v_hat)
        nu2 = B.T @ nu
        S3 = np.asarray(S, dtype=float)
        S2 = B.T @ S3 @ B
        S2 = 0.5 * (S2 + S2.T)
        sample = InnovationSample(
            nis=float(nu2 @ np.linalg.solve(S2, nu2)),
            dof=VECTOR_NIS_DOF,
            nu=nu.copy(),
            nu_tangent=nu2.copy(),
            S_tangent=S2,
            sensor=sensor,
            t=t,
            index=len(self.samples),
            v_hat=v_hat.copy(),
        )
        self.samples.append(sample)
        return sample

    def samples_for(self, sensor: str | None = None) -> list[InnovationSample]:
        if sensor is None:
            return list(self.samples)
        return [s for s in self.samples if s.sensor == sensor]

    def nis_values(self, sensor: str | None = None) -> np.ndarray:
        return np.array([s.nis for s in self.samples_for(sensor)], dtype=float)

    def mean_nis(self, sensor: str | None = None) -> float:
        values = self.nis_values(sensor)
        if values.size == 0:
            raise ValueError("innovation log is empty")
        return float(values.mean())

    def whitened_sequence(self, sensor: str | None = None) -> np.ndarray:
        rows = [s.whitened() for s in self.samples_for(sensor)]
        if not rows:
            raise ValueError("innovation log is empty")
        return np.vstack(rows)

    def lag1_correlation(self, sensor: str | None = None) -> np.ndarray:
        return lag1_sample_correlation(self.whitened_sequence(sensor))

    def summary(self) -> dict[str, float | int]:
        out: dict[str, float | int] = {
            "n": len(self.samples),
            "dof": VECTOR_NIS_DOF,
        }
        if self.samples:
            out["mean_nis"] = self.mean_nis()
            for name in sorted({s.sensor for s in self.samples}):
                out[f"n_{name}"] = len(self.samples_for(name))
                out[f"mean_nis_{name}"] = self.mean_nis(name)
        return out

    def write_csv(self, path: str | Path) -> Path:
        """Write one row per vector update.  Empty logs still get a header."""
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "index",
                    "t",
                    "sensor",
                    "nis",
                    "dof",
                    "nu_x",
                    "nu_y",
                    "nu_z",
                    "nu2_0",
                    "nu2_1",
                    "S2_00",
                    "S2_01",
                    "S2_10",
                    "S2_11",
                ]
            )
            for sample in self.samples:
                writer.writerow(
                    [
                        sample.index,
                        "" if sample.t is None else float(sample.t),
                        sample.sensor,
                        float(sample.nis),
                        sample.dof,
                        float(sample.nu[0]),
                        float(sample.nu[1]),
                        float(sample.nu[2]),
                        float(sample.nu_tangent[0]),
                        float(sample.nu_tangent[1]),
                        float(sample.S_tangent[0, 0]),
                        float(sample.S_tangent[0, 1]),
                        float(sample.S_tangent[1, 0]),
                        float(sample.S_tangent[1, 1]),
                    ]
                )
        return dest


def mekf_vector_innovation_stats(
    v_b_meas: np.ndarray,
    v_inertial: np.ndarray,
    q_hat: np.ndarray,
    P: np.ndarray,
    sigma: float,
    *,
    sensor: str = "vector",
    t: float | None = None,
    index: int = 0,
) -> InnovationSample:
    """Pre-update rank-2 NIS for a unit-vector observation.

    Uses predicted ``P`` (before the Joseph update).  ``S = H P Hᵀ + R``
    is projected onto the plane orthogonal to ``v̂_b`` so the statistic
    is ``χ²_2``, matching the rank-2 ``R``.
    """
    v_hat = mekf_predicted_body_vector(q_hat, v_inertial)
    nu = _unit3(v_b_meas) - v_hat
    H, R = mekf_vector_HR(v_hat, sigma)
    P6 = np.asarray(P, dtype=float).reshape(6, 6)
    S = H @ P6 @ H.T + R
    B = tangent_plane_basis(v_hat)
    nu2 = B.T @ nu
    S2 = B.T @ S @ B
    S2 = 0.5 * (S2 + S2.T)
    return InnovationSample(
        nis=float(nu2 @ np.linalg.solve(S2, nu2)),
        dof=VECTOR_NIS_DOF,
        nu=nu.copy(),
        nu_tangent=nu2.copy(),
        S_tangent=S2,
        sensor=sensor,
        t=t,
        index=index,
        v_hat=v_hat.copy(),
    )


def mekf_vector_nis(
    v_b_meas: np.ndarray,
    v_inertial: np.ndarray,
    q_hat: np.ndarray,
    P: np.ndarray,
    sigma: float,
) -> float:
    """Scalar rank-2 NIS ``ν₂ᵀ S₂⁻¹ ν₂`` for one unit-vector update."""
    return mekf_vector_innovation_stats(v_b_meas, v_inertial, q_hat, P, sigma).nis


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
        *,
        t: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        del t  # accepted so SimLab can pass a timestamp; Mahony has no NIS log
        omega_m = np.asarray(omega_m, dtype=float).reshape(3)
        omega_corr = np.zeros(3)
        meas = iter_vector_meas(vector_meas)
        if meas:
            R = quat_to_rotation(self.q)
            wsum = 0.0
            acc = np.zeros(3)
            for v_b, v_I, sigma, _name in meas:
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
    tangent-plane ``R``.  Optional ``innovation_log`` records pre-update
    rank-2 NIS; default ``None`` leaves the filter bit-identical.
    """

    sigma_v: float = 5e-4
    sigma_u: float = 1e-6
    q: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    P: np.ndarray | None = None
    innovation_log: InnovationLog | None = None

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

    def enable_innovation_log(self) -> InnovationLog:
        """Attach (or return) the optional vector-innovation recorder."""
        if self.innovation_log is None:
            self.innovation_log = InnovationLog()
        return self.innovation_log

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

    def update_vector(
        self,
        v_b_meas: np.ndarray,
        v_inertial: np.ndarray,
        sigma: float,
        *,
        sensor: str = "vector",
        t: float | None = None,
    ) -> None:
        v_b_meas = _unit3(v_b_meas)
        v_hat = mekf_predicted_body_vector(self.q, v_inertial)
        # H = [[v̂× | 0]], rank-2 R: same matrices as mekf_vector_HR / NIS.
        H, R = mekf_vector_HR(v_hat, sigma)
        assert self.P is not None
        S = H @ self.P @ H.T + R
        nu = v_b_meas - v_hat
        if self.innovation_log is not None:
            self.innovation_log.record_from_update(
                nu=nu, v_hat=v_hat, S=S, sensor=sensor, t=t
            )
        K = self.P @ H.T @ np.linalg.solve(S, np.eye(3))
        dx = K @ nu
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
        *,
        t: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        omega_hat = self.predict(omega_m, dt)
        for v_b, v_I, sigma, name in iter_vector_meas(vector_meas):
            self.update_vector(
                v_b,
                v_I,
                vector_sigma if sigma is None else sigma,
                sensor=name,
                t=t,
            )
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
) -> list[tuple[np.ndarray, np.ndarray, float, str]]:
    """Sample each *available* vector sensor and attach Cartesian ``sigma``.

    Occulted / out-of-FOV stubs (``measure`` returns ``None``) are skipped
    so the filter runs gyro-only on that sample.  The optional fourth
    field is the sensor name (mag/sun) for ``InnovationLog``.
    """
    out: list[tuple[np.ndarray, np.ndarray, float, str]] = []
    for sensor in sensors:
        v_b = sensor.measure(q)
        if v_b is None:
            continue
        out.append((v_b, sensor.v_inertial, sensor.sigma, sensor.name))
    return out

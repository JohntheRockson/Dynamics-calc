"""Normalized Estimation Error Squared (NEES) helpers for the 6-state MEKF.

The MEKF error is the body-frame multiplicative attitude error plus gyro
bias error, matching ``MultiplicativeEKF``:

    q = q̂ ⊗ δq(δα),   δq ≈ [1, δα/2],   δb = b − b̂,   x = [δα, δb].

For a consistent linear-Gaussian filter, a single sample of
ε = xᵀ P⁻¹ x is χ² with dim(x) degrees of freedom (mean = dim).
This module does **not** change the filter; it only scores (q̂, b̂, P)
against a known truth trajectory.

The CI smoke in ``tests/test_nees.py`` uses a short open-loop (torque-free)
plant run with matching Farrenkopf gyro and vector-sensor noise.  See
``docs/estimation.md`` for how to read the chi-square band.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from attitude_sim.estimation import MultiplicativeEKF, vectors_from_sensors
from attitude_sim.plant import RigidBody, step_rigid_body
from attitude_sim.quaternions import (
    axis_angle_to_quat,
    quat_conjugate,
    quat_error,
    quat_multiply,
    quat_normalize,
)
from attitude_sim.sensors import GyroModel, magnetometer, sun_sensor

# Smoke-test band: intentionally looser than a textbook Monte Carlo ANEES
# study.  Independent χ²_n has mean n; a short correlated trajectory can
# wander.  Values ≪ n ⇒ pessimistic P; values ≫ n ⇒ optimistic P.
NEES_FULL_DOF = 6
NEES_ATT_DOF = 3
NEES_BIAS_DOF = 3
# Roughly 0.25 n … 3 n (see docs/estimation.md).
NEES_FULL_BAND = (1.5, 18.0)
NEES_ATT_BAND = (0.5, 12.0)
NEES_BIAS_BAND = (0.5, 12.0)


def _spd(P: np.ndarray, n: int) -> np.ndarray:
    P = np.asarray(P, dtype=float)
    if P.shape != (n, n):
        raise ValueError(f"P must be shape {(n, n)}, got {P.shape}")
    return 0.5 * (P + P.T)


def nees(error: np.ndarray, P: np.ndarray) -> float:
    """Scalar NEES ``ε = xᵀ P⁻¹ x`` for a stacked error and covariance."""
    x = np.asarray(error, dtype=float).reshape(-1)
    P = _spd(P, x.size)
    return float(x @ np.linalg.solve(P, x))


def mekf_attitude_error(q_true: np.ndarray, q_hat: np.ndarray) -> np.ndarray:
    """Geodesic rotation vector ``δα`` for ``q_true = q̂ ⊗ δq(δα)``.

    Matches MEKF injection (``axis_angle_to_quat``): ``||δα||`` is the
    principal rotation angle, not the first-order ``2 vec(δq)``.
    """
    qe = quat_error(q_true, q_hat)
    if qe[0] < 0.0:
        qe = -qe
    vec = qe[1:]
    n = float(np.linalg.norm(vec))
    if n < 1e-15:
        return np.zeros(3)
    theta = 2.0 * float(np.arctan2(n, float(qe[0])))
    return (theta / n) * vec


def mekf_error_state(
    q_true: np.ndarray,
    q_hat: np.ndarray,
    b_true: np.ndarray,
    b_hat: np.ndarray,
) -> np.ndarray:
    """Stack ``x = [δα, δb]`` with ``q_true = q̂ ⊗ δq(δα)`` and ``δb = b − b̂``."""
    dalpha = mekf_attitude_error(q_true, q_hat)
    db = np.asarray(b_true, dtype=float).reshape(3) - np.asarray(b_hat, dtype=float).reshape(3)
    return np.concatenate([dalpha, db])


def split_mekf_nees(
    error: np.ndarray,
    P: np.ndarray,
) -> tuple[float, float, float]:
    """Return ``(full, attitude, bias)`` NEES using marginal ``P_αα`` / ``P_bb``."""
    x = np.asarray(error, dtype=float).reshape(6)
    P = _spd(P, 6)
    full = nees(x, P)
    att = nees(x[:3], P[:3, :3])
    bias = nees(x[3:], P[3:, 3:])
    return full, att, bias


def apply_mekf_error(
    q_true: np.ndarray,
    b_true: np.ndarray,
    error: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build ``(q̂, b̂)`` such that ``mekf_error_state`` recovers ``error``.

    ``q_true = q̂ ⊗ δq(δα)`` ⇒ ``q̂ = q_true ⊗ δq(δα)*``.
    """
    x = np.asarray(error, dtype=float).reshape(6)
    dalpha = x[:3]
    angle = float(np.linalg.norm(dalpha))
    dq = axis_angle_to_quat(dalpha, angle)
    q_hat = quat_normalize(quat_multiply(q_true, quat_conjugate(dq)))
    b_hat = np.asarray(b_true, dtype=float).reshape(3) - x[3:]
    return q_hat, b_hat


def default_nees_p0() -> np.ndarray:
    """Well-conditioned ``P0`` for the smoke: ~1° attitude, ~2 mrad/s bias 1σ."""
    return np.diag([3.0e-4, 3.0e-4, 3.0e-4, 4.0e-6, 4.0e-6, 4.0e-6])


@dataclass
class MekfNeesResult:
    """Time series of full / attitude / bias NEES for one truth run."""

    t: np.ndarray
    full: np.ndarray
    attitude: np.ndarray
    bias: np.ndarray

    def means(self, skip: int = 0) -> tuple[float, float, float]:
        """Mean ``(full, attitude, bias)`` NEES from index ``skip`` onward."""
        return (
            float(np.mean(self.full[skip:])),
            float(np.mean(self.attitude[skip:])),
            float(np.mean(self.bias[skip:])),
        )


def run_open_loop_mekf_nees(
    *,
    dt: float = 0.02,
    t_final: float = 1.6,
    sigma_v: float = 5e-4,
    sigma_u: float = 1e-6,
    mag_sigma: float = 3e-3,
    sun_sigma: float = 2e-3,
    seed: int = 0,
    P0: np.ndarray | None = None,
    sample_initial_error: bool = True,
) -> MekfNeesResult:
    """Score MEKF NEES on a short torque-free plant trajectory.

    Truth gyro / vector sensors use the same densities the filter is given.
    The initial error is drawn from ``P0`` (and the filter is initialized
    with that ``P0``) so t=0 NEES is χ²_6 when the draw is Gaussian.

    Discrete timing (this helper, not the SimLab log): gyro integrates
    over ``[t_k, t_{k+1})``, the plant is stepped to ``t_{k+1}``, vector
    observations are taken at ``t_{k+1}``, then ``MultiplicativeEKF.step``
    predicts and updates.  NEES at index ``k`` is scored against that
    aligned ``(q, b, q̂, b̂, P)``.
    """
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if t_final < 0.0:
        raise ValueError("t_final must be non-negative")
    rng = np.random.default_rng(seed)
    P0 = default_nees_p0() if P0 is None else _spd(P0, 6)

    body = RigidBody(np.diag([0.05, 0.06, 0.07]))
    q = quat_normalize(np.array([0.9, 0.1, -0.2, 0.3]))
    omega = np.array([0.05, -0.04, 0.08])
    b_true0 = np.array([0.002, -0.001, 0.0015])

    if sample_initial_error:
        x0 = rng.multivariate_normal(np.zeros(6), P0)
    else:
        x0 = np.zeros(6)
    q_hat, b_hat = apply_mekf_error(q, b_true0, x0)

    filt = MultiplicativeEKF(q=q_hat, bias=b_hat, sigma_v=sigma_v, sigma_u=sigma_u, P=P0)
    gyro = GyroModel(sigma_v=sigma_v, sigma_u=sigma_u, bias=b_true0, seed=rng)
    mag = magnetometer(sigma=mag_sigma, seed=rng)
    sun = sun_sensor(sigma=sun_sigma, seed=rng)

    n = int(np.round(t_final / dt)) + 1
    t = np.empty(n)
    full = np.empty(n)
    att = np.empty(n)
    bias = np.empty(n)

    for k in range(n):
        t[k] = k * dt
        x = mekf_error_state(q, filt.q, gyro.bias, filt.bias)
        full[k], att[k], bias[k] = split_mekf_nees(x, filt.P)
        if k == n - 1:
            break
        omega_m = gyro.measure(omega, dt)
        q, omega = step_rigid_body(body, q, omega, np.zeros(3), dt)
        vecs = vectors_from_sensors(q, [mag, sun])
        filt.step(omega_m, dt, vecs)

    return MekfNeesResult(t=t, full=full, attitude=att, bias=bias)


def mean_nees_monte_carlo(
    n_runs: int = 8,
    *,
    seed0: int = 11,
    skip: int = 1,
    **kwargs,
) -> tuple[float, float, float]:
    """Mean NEES over ``n_runs`` independent open-loop trajectories.

    ``skip`` drops the first samples of each run (t=0 is the drawn prior;
    the filter smoke uses the post-update series).
    """
    if n_runs < 1:
        raise ValueError("n_runs must be positive")
    fulls: list[float] = []
    atts: list[float] = []
    biases: list[float] = []
    for i in range(n_runs):
        result = run_open_loop_mekf_nees(seed=seed0 + i, **kwargs)
        f, a, b = result.means(skip=skip)
        fulls.append(f)
        atts.append(a)
        biases.append(b)
    return float(np.mean(fulls)), float(np.mean(atts)), float(np.mean(biases))

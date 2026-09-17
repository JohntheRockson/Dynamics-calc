"""Estimator-only checks: noise mapping, quaternion convention, filter vs plant truth."""

import numpy as np
import pytest

from attitude_sim.estimation import (
    MEKF_NEES_DOF,
    ComplementaryFilter,
    MultiplicativeEKF,
    _inject_body_error,
    chi2_mean_nees_bounds,
    farrenkopf_Qd,
    make_estimator,
    mekf_error_state,
    mekf_stm,
    nees,
    triad_attitude,
    triad_q0_from_sensors,
    vectors_from_sensors,
)
from attitude_sim.plant import RigidBody, step_rigid_body
from attitude_sim.quaternions import (
    axis_angle_to_quat,
    geodesic_angle,
    quat_integrate_const_omega,
    quat_multiply,
    quat_normalize,
    quat_to_rotation,
)
from attitude_sim.sensors import GyroModel, magnetometer, sun_sensor


def _vector_meas(q, v_I, sigma=0.0):
    v_b = quat_to_rotation(q).T @ (v_I / np.linalg.norm(v_I))
    if sigma > 0.0:
        rng = np.random.default_rng(0)
        v_b = v_b + sigma * rng.standard_normal(3)
    return v_b / np.linalg.norm(v_b), v_I / np.linalg.norm(v_I)


def test_mekf_vector_update_reduces_error():
    q_true = axis_angle_to_quat(np.array([0.3, 0.2, 0.9]), 0.35)
    q_hat0 = np.array([1.0, 0.0, 0.0, 0.0])
    filt = MultiplicativeEKF(q=q_hat0, sigma_v=1e-4, sigma_u=1e-8)
    v_I = np.array([0.2, 0.1, 1.0])
    v2 = np.array([1.0, 0.0, 0.0])
    e0 = geodesic_angle(filt.q, q_true)
    for _ in range(80):
        meas = [_vector_meas(q_true, v_I), _vector_meas(q_true, v2)]
        filt.step(np.zeros(3), 0.02, meas, vector_sigma=1e-3)
    assert geodesic_angle(filt.q, q_true) < 0.25 * e0


def test_mahony_vector_update_reduces_error():
    q_true = axis_angle_to_quat(np.array([0.3, 0.2, 0.9]), 0.35)
    filt = ComplementaryFilter(q=np.array([1.0, 0.0, 0.0, 0.0]), kp=2.0, ki=0.05)
    v_I = np.array([0.2, 0.1, 1.0])
    v2 = np.array([1.0, 0.0, 0.0])
    e0 = geodesic_angle(filt.q, q_true)
    for _ in range(80):
        meas = [_vector_meas(q_true, v_I), _vector_meas(q_true, v2)]
        filt.step(np.zeros(3), 0.02, meas)
    assert geodesic_angle(filt.q, q_true) < 0.25 * e0


def test_mekf_tracks_constant_rate():
    wz = 0.15
    q = np.array([1.0, 0.0, 0.0, 0.0])
    filt = MultiplicativeEKF(q=q.copy())
    dt = 0.01
    v_I = np.array([0.0, 0.0, 1.0])
    v2 = np.array([1.0, 0.0, 0.0])
    omega = np.array([0.0, 0.0, wz])
    for _ in range(200):
        q = quat_multiply(q, axis_angle_to_quat(np.array([0.0, 0.0, 1.0]), wz * dt))
        meas = [_vector_meas(q, v_I), _vector_meas(q, v2)]
        omega_m = omega + np.array([0.0, 0.0, 0.0])
        filt.step(omega_m, dt, meas, vector_sigma=1e-3)
    assert geodesic_angle(filt.q, q) < np.deg2rad(3.0)


def test_quaternion_scalar_first_and_body_to_inertial():
    """Estimates stay scalar-first; R(q) maps body → inertial (v_I = R v_B)."""
    axis = np.array([0.0, 0.0, 1.0])
    q = axis_angle_to_quat(axis, 0.4)
    filt = MultiplicativeEKF(q=q)
    assert filt.q.shape == (4,)
    assert abs(np.linalg.norm(filt.q) - 1.0) < 1e-12
    # Small rotation about +z: |w| > |vector|, w ≈ cos(θ/2) > 0.
    assert filt.q[0] > np.linalg.norm(filt.q[1:])
    v_B = np.array([1.0, 0.0, 0.0])
    v_I = quat_to_rotation(filt.q) @ v_B
    # +z rotation takes body x toward +y in inertial coordinates.
    assert v_I[1] > 0.0
    np.testing.assert_allclose(np.linalg.norm(v_I), 1.0, atol=1e-12)


def test_mekf_injects_body_frame_right_multiply():
    """q = q̂ ⊗ δq(δα): a +x error injection must right-multiply, not left."""
    q0 = axis_angle_to_quat(np.array([0.0, 0.0, 1.0]), 0.5)
    filt = MultiplicativeEKF(q=q0.copy())
    dtheta = np.array([0.12, 0.0, 0.0])
    filt._inject(np.concatenate([dtheta, np.zeros(3)]))
    expected = quat_multiply(q0, axis_angle_to_quat(dtheta, 0.12))
    # Same attitude (q and -q); geodesic angle is the invariant.
    assert geodesic_angle(filt.q, expected) < 1e-12
    left = quat_multiply(axis_angle_to_quat(dtheta, 0.12), q0)
    assert geodesic_angle(filt.q, left) > 1e-3


def test_farrenkopf_Qd_structure():
    dt = 0.05
    sv, su = 2e-3, 4e-5
    Q = farrenkopf_Qd(sv, su, dt)
    np.testing.assert_allclose(Q[:3, :3], (sv**2 * dt + su**2 * dt**3 / 3.0) * np.eye(3))
    np.testing.assert_allclose(Q[3:, 3:], (su**2 * dt) * np.eye(3))
    np.testing.assert_allclose(Q[:3, 3:], -0.5 * su**2 * dt**2 * np.eye(3))
    np.testing.assert_allclose(Q, Q.T, atol=1e-16)
    evals = np.linalg.eigvalsh(Q)
    assert np.all(evals >= -1e-18)


def test_mekf_stm_rest_and_spin():
    dt = 0.02
    Phi0 = mekf_stm(np.zeros(3), dt)
    np.testing.assert_allclose(Phi0[:3, :3], np.eye(3), atol=1e-12)
    np.testing.assert_allclose(Phi0[:3, 3:], -dt * np.eye(3), atol=1e-12)
    np.testing.assert_allclose(Phi0[3:, :3], 0.0, atol=1e-15)
    np.testing.assert_allclose(Phi0[3:, 3:], np.eye(3), atol=1e-15)

    wz = 0.7
    Phi = mekf_stm(np.array([0.0, 0.0, wz]), dt)
    # Φ_αα is a rotation about −z by wz dt (exp(-[ω×]dt)).
    c, s = np.cos(wz * dt), np.sin(wz * dt)
    expected_aa = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(Phi[:3, :3], expected_aa, atol=1e-12)
    # det Φ = 1 for the 6x6 (block triangular with det Φ_αα = 1).
    assert abs(np.linalg.det(Phi) - 1.0) < 1e-10


def test_mekf_predict_grows_covariance():
    filt = MultiplicativeEKF(sigma_v=1e-3, sigma_u=1e-5, P=1e-12 * np.eye(6))
    p0 = np.trace(filt.P)
    filt.predict(np.array([0.1, -0.05, 0.02]), 0.01)
    assert np.trace(filt.P) > p0
    evals = np.linalg.eigvalsh(filt.P)
    assert np.all(evals > 0.0)


def test_mekf_estimates_constant_gyro_bias():
    q_true = np.array([1.0, 0.0, 0.0, 0.0])
    bias_true = np.array([0.012, -0.007, 0.004])
    filt = MultiplicativeEKF(q=q_true.copy(), sigma_v=1e-4, sigma_u=1e-7)
    v_I = np.array([0.0, 0.0, 1.0])
    v2 = np.array([1.0, 0.0, 0.0])
    dt = 0.02
    for _ in range(250):
        meas = [_vector_meas(q_true, v_I), _vector_meas(q_true, v2)]
        filt.step(bias_true, dt, meas, vector_sigma=5e-4)
    np.testing.assert_allclose(filt.bias, bias_true, atol=2e-3)
    assert geodesic_angle(filt.q, q_true) < np.deg2rad(0.5)


def test_mahony_returns_unbiased_rate_not_innovation():
    """Controller rate is ω_m − b̂; k_p ω_mes stays in the kinematics only."""
    q_true = axis_angle_to_quat(np.array([0.0, 1.0, 0.0]), 0.25)
    filt = ComplementaryFilter(q=np.array([1.0, 0.0, 0.0, 0.0]), kp=3.0, ki=0.0)
    v_I = np.array([0.0, 0.0, 1.0])
    v2 = np.array([1.0, 0.0, 0.0])
    omega_m = np.array([0.05, -0.02, 0.01])
    _, omega_hat = filt.step(omega_m, 0.01, [_vector_meas(q_true, v_I), _vector_meas(q_true, v2)])
    np.testing.assert_allclose(omega_hat, omega_m - filt.bias, atol=1e-15)
    # With a large initial attitude error, kinematics correction is nonzero,
    # so ω_kin would not equal ω_m.  The returned rate still matches ω_m.
    assert geodesic_angle(filt.q, q_true) > 1e-3


def test_mekf_tracks_plant_truth_with_noisy_sensors():
    """Sample (q, ω) from the RK4 plant; estimator should track geodesic error down."""
    body = RigidBody(np.diag([0.05, 0.06, 0.07]))
    q = quat_normalize([0.9, 0.1, -0.2, 0.3])
    omega = np.array([0.05, -0.04, 0.08])
    dt = 0.01
    gyro = GyroModel(sigma_v=3e-4, sigma_u=1e-6, bias=np.array([0.002, -0.001, 0.001]), seed=4)
    mag = magnetometer(sigma=2e-3, seed=5)
    sun = sun_sensor(sigma=1.5e-3, seed=6)
    filt = MultiplicativeEKF(q=q.copy(), sigma_v=3e-4, sigma_u=1e-6)
    errors = []
    for _ in range(400):
        omega_m = gyro.measure(omega, dt)
        vecs = vectors_from_sensors(q, [mag, sun])
        q_hat, _ = filt.step(omega_m, dt, vecs)
        errors.append(geodesic_angle(q_hat, q))
        q, omega = step_rigid_body(body, q, omega, np.zeros(3), dt)
    late = np.array(errors[200:])
    assert np.median(late) < np.deg2rad(2.0)
    assert np.max(late) < np.deg2rad(6.0)


def test_mahony_tracks_plant_truth_with_noisy_sensors():
    body = RigidBody(np.diag([0.05, 0.06, 0.07]))
    q = quat_normalize([0.9, 0.1, -0.2, 0.3])
    omega = np.array([0.05, -0.04, 0.08])
    dt = 0.01
    gyro = GyroModel(sigma_v=3e-4, sigma_u=1e-6, bias=np.array([0.002, -0.001, 0.001]), seed=7)
    mag = magnetometer(sigma=2e-3, seed=8)
    sun = sun_sensor(sigma=1.5e-3, seed=9)
    filt = ComplementaryFilter(q=q.copy(), kp=1.8, ki=0.1)
    errors = []
    for _ in range(400):
        omega_m = gyro.measure(omega, dt)
        vecs = vectors_from_sensors(q, [mag, sun])
        q_hat, _ = filt.step(omega_m, dt, vecs)
        errors.append(geodesic_angle(q_hat, q))
        q, omega = step_rigid_body(body, q, omega, np.zeros(3), dt)
    late = np.array(errors[200:])
    assert np.median(late) < np.deg2rad(3.0)


def test_make_estimator_modes_and_unknown():
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    assert make_estimator("truth", q0=q0) is None
    assert isinstance(make_estimator("mekf", q0=q0), MultiplicativeEKF)
    assert isinstance(make_estimator("mahony", q0=q0), ComplementaryFilter)
    with pytest.raises(ValueError, match="unknown estimator"):
        make_estimator("triad", q0=q0)


def test_mekf_unit_norm_and_symmetric_p_after_steps():
    filt = MultiplicativeEKF()
    v_I = np.array([0.2, 0.4, 0.9])
    v2 = np.array([1.0, -0.1, 0.0])
    q = np.array([1.0, 0.0, 0.0, 0.0])
    for _ in range(50):
        q = quat_multiply(q, axis_angle_to_quat(np.array([0.1, 0.2, 0.9]), 0.01))
        filt.step(np.array([0.1, 0.2, 0.9]), 0.01, [_vector_meas(q, v_I), _vector_meas(q, v2)])
    assert abs(np.linalg.norm(filt.q) - 1.0) < 1e-12
    np.testing.assert_allclose(filt.P, filt.P.T, atol=1e-12)
    assert filt.q[0] >= 0.0 or abs(filt.q[0]) < 1.0  # finite; sign is free


def test_farrenkopf_and_stm_reject_non_positive_dt():
    with pytest.raises(ValueError, match="dt must be positive"):
        farrenkopf_Qd(1e-3, 1e-6, 0.0)
    with pytest.raises(ValueError, match="dt must be positive"):
        mekf_stm(np.zeros(3), -0.01)


def test_triad_recovers_true_attitude_noise_free():
    q_true = axis_angle_to_quat(np.array([0.3, -0.2, 0.8]), 0.9)
    mag = magnetometer(sigma=0.0, seed=0)
    sun = sun_sensor(sigma=0.0, seed=1)
    v_b_m = mag.measure(q_true)
    v_b_s = sun.measure(q_true)
    q_hat = triad_attitude(v_b_m, mag.v_inertial, v_b_s, sun.v_inertial)
    assert geodesic_angle(q_hat, q_true) < 1e-10
    assert abs(np.linalg.norm(q_hat) - 1.0) < 1e-12
    # Scalar-first, body → inertial: TRIAD R maps body mag to inertial mag.
    R = quat_to_rotation(q_hat)
    np.testing.assert_allclose(R @ v_b_m, mag.v_inertial, atol=1e-12)


def test_triad_rejects_parallel_references():
    v = np.array([1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="non-parallel"):
        triad_attitude(v, v, v, v)


def test_triad_q0_from_noisy_sensors_is_coarse():
    q_true = axis_angle_to_quat(np.array([0.1, 0.7, 0.2]), 1.1)
    mag = magnetometer(sigma=3e-3, seed=21)
    sun = sun_sensor(sigma=2e-3, seed=22)
    q_hat = triad_q0_from_sensors(q_true, [mag, sun])
    err = geodesic_angle(q_hat, q_true)
    assert err > 0.0
    assert err < np.deg2rad(5.0)


def test_triad_q0_needs_two_available_sensors():
    q = np.array([1.0, 0.0, 0.0, 0.0])
    mag = magnetometer(sigma=0.0, seed=0)
    sun = sun_sensor(sigma=0.0, eclipse=True, seed=1)
    with pytest.raises(ValueError, match="two available"):
        triad_q0_from_sensors(q, [mag, sun])
    with pytest.raises(ValueError, match="two available"):
        triad_q0_from_sensors(q, [mag])


def test_mekf_error_state_matches_right_multiply_inject():
    q_true = axis_angle_to_quat(np.array([0.0, 0.0, 1.0]), 0.4)
    dalpha = np.array([0.04, -0.02, 0.03])
    dbias = np.array([0.001, -0.002, 0.0005])
    q_hat = _inject_body_error(q_true, -dalpha)
    bias_true = np.array([0.002, 0.0, -0.001])
    x = mekf_error_state(q_hat, bias_true - dbias, q_true, bias_true)
    np.testing.assert_allclose(x[:3], dalpha, atol=2e-5)
    np.testing.assert_allclose(x[3:], dbias, atol=1e-15)


def test_chi2_mean_nees_bounds_six_state():
    lo, hi = chi2_mean_nees_bounds(MEKF_NEES_DOF, 32, alpha=0.01)
    # Documented 99% interval for mean of 32 i.i.d. χ²_6 samples.
    assert lo == pytest.approx(4.540, abs=0.01)
    assert hi == pytest.approx(7.694, abs=0.01)
    with pytest.raises(ValueError):
        chi2_mean_nees_bounds(6, 0)
    with pytest.raises(ValueError):
        chi2_mean_nees_bounds(6, 10, alpha=0.0)


def test_nees_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="P shape"):
        nees(np.zeros(6), np.eye(3))


def test_mekf_nees_honest_prior_matches_chi2():
    """x ~ N(0, P0) ⇒ NEES = xᵀ P0⁻¹ x ~ χ²_6.  Mean of N draws in 99% bounds."""
    n = 32
    rng = np.random.default_rng(0)
    P0 = np.diag([3e-3, 3e-3, 3e-3, 1e-5, 1e-5, 1e-5])
    values = [
        nees(rng.multivariate_normal(np.zeros(6), P0), P0) for _ in range(n)
    ]
    mean = float(np.mean(values))
    lo, hi = chi2_mean_nees_bounds(MEKF_NEES_DOF, n, alpha=0.01)
    assert lo < mean < hi, f"prior ANEES={mean:.3f} not in 99% χ² bounds [{lo:.3f}, {hi:.3f}]"


def _mekf_matched_nees_trial(seed: int, n_steps: int = 200, dt: float = 0.02) -> float:
    """One synthetic MEKF trial with matched Farrenkopf gyro + mag/sun stubs.

    Truth is constant-rate quaternion kinematics (no plant torque).  Gyro
    ARW/RRW densities match ``sigma_v`` / ``sigma_u`` in ``farrenkopf_Qd``.
    Vector stubs use the same Cartesian ``sigma`` the filter puts on the
    rank-2 tangent-plane ``R``.  Initial error is drawn from ``P0``
    (honest prior).  Measurement is taken at the end of each ``dt`` after
    truth and the filter both propagate.
    """
    rng = np.random.default_rng(seed)
    q_true = axis_angle_to_quat(rng.normal(size=3), 0.4)
    omega = np.array([0.03, -0.02, 0.025])
    bias0 = np.array([0.002, -0.001, 0.0015])
    sigma_v, sigma_u = 5e-4, 1e-6
    P0 = np.diag([3e-3, 3e-3, 3e-3, 1e-5, 1e-5, 1e-5])
    x0 = rng.multivariate_normal(np.zeros(6), P0)
    q_hat = _inject_body_error(q_true, -x0[:3])
    filt = MultiplicativeEKF(
        q=q_hat,
        bias=bias0 - x0[3:],
        P=P0.copy(),
        sigma_v=sigma_v,
        sigma_u=sigma_u,
    )
    gyro = GyroModel(sigma_v=sigma_v, sigma_u=sigma_u, bias=bias0.copy(), seed=rng)
    mag = magnetometer(sigma=3e-3, seed=rng)
    sun = sun_sensor(sigma=2e-3, seed=rng)

    for _ in range(n_steps):
        omega_m = gyro.measure(omega, dt)
        q_true = quat_integrate_const_omega(q_true, omega, dt)
        vecs = vectors_from_sensors(q_true, [mag, sun])
        filt.step(omega_m, dt, vecs)
        omega_m = gyro.measure(omega, dt)
        q_true = quat_integrate_const_omega(q_true, omega, dt)
        vecs = vectors_from_sensors(q_true, [mag, sun])
        filt.step(omega_m, dt, vecs)
    x = mekf_error_state(filt.q, filt.bias, q_true, gyro.bias)
    return nees(x, filt.P)


def test_mekf_nees_matched_synthetic_is_consistent():
    """Ensemble ANEES after filtering stays inside 99% χ²_6 mean bounds.

    Noise assumptions (see docs/estimation.md):
    - Farrenkopf gyro ``ω_m = ω + b + η_v``, ``ḃ = η_u`` with
      ``σ_v = 5e-4`` rad/s/√Hz, ``σ_u = 1e-6`` rad/s²/√Hz, ``σ_n = 0``.
    - Mag ``σ = 3e-3``, sun ``σ = 2e-3`` Cartesian; filter ``R`` is the
      matching rank-2 tangent-plane plus nugget.  No IGRF / FOV / eclipse.
    - Honest ``P0 = diag(3e-3 I, 1e-5 I)``.  ``N = 32`` i.i.d. trials.
    Expected: ``E[NEES] = 6``; 99% bounds on the mean ≈ ``[4.54, 7.69]``.
    Single-trial 99% χ²_6 interval is about ``[0.68, 18.55]``.
    """
    n = 32
    values = np.array([_mekf_matched_nees_trial(seed=1000 + i) for i in range(n)])
    assert np.all(np.isfinite(values))
    mean = float(values.mean())
    lo, hi = chi2_mean_nees_bounds(MEKF_NEES_DOF, n, alpha=0.01)
    single_lo, single_hi = 0.6757, 18.5476  # χ²_6 99%
    # A few trials may sit in the tails; the *mean* is the consistency gate.
    assert values.min() > 0.0
    assert values.max() < 5.0 * single_hi
    assert lo < mean < hi, (
        f"matched ANEES={mean:.3f} not in 99% χ²_{MEKF_NEES_DOF} mean bounds "
        f"[{lo:.3f}, {hi:.3f}] (N={n}; single-trial 99% [{single_lo}, {single_hi}])"
    )

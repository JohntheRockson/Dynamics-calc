"""Estimator-only checks: noise mapping, quaternion convention, filter vs plant truth."""

import numpy as np
import pytest

from attitude_sim.estimation import (
    ComplementaryFilter,
    MultiplicativeEKF,
    farrenkopf_Qd,
    make_estimator,
    mekf_stm,
    vectors_from_sensors,
)
from attitude_sim.plant import RigidBody, step_rigid_body
from attitude_sim.quaternions import (
    axis_angle_to_quat,
    geodesic_angle,
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
    for k in range(400):
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
    for k in range(50):
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

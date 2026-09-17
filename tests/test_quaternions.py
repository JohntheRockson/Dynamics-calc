"""Quaternion algebra and kinematics checks."""

import numpy as np
import pytest

from attitude_sim.quaternions import (
    axis_angle_to_quat,
    geodesic_angle,
    quat_conjugate,
    quat_derivative,
    quat_error,
    quat_integrate_const_omega,
    quat_multiply,
    quat_normalize,
    quat_to_euler321,
    quat_to_rotation,
    rotation_to_quat,
    rotation_vector_error,
)


def test_normalize_and_identity_multiply():
    q = quat_normalize([0.2, 0.4, -0.1, 0.8])
    assert q.shape == (4,)
    assert abs(np.linalg.norm(q) - 1.0) < 1e-15
    ident = np.array([1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(quat_multiply(q, ident), q, atol=1e-15)
    np.testing.assert_allclose(quat_multiply(ident, q), q, atol=1e-15)


def test_conjugate_is_inverse():
    q = quat_normalize([0.3, -0.2, 0.5, 0.7])
    prod = quat_multiply(q, quat_conjugate(q))
    np.testing.assert_allclose(prod, [1.0, 0.0, 0.0, 0.0], atol=1e-15)


def test_rotation_roundtrip():
    q = axis_angle_to_quat(np.array([1.0, -2.0, 0.5]), 1.3)
    R = quat_to_rotation(q)
    assert abs(np.linalg.det(R) - 1.0) < 1e-12
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
    q2 = rotation_to_quat(R)
    # q and -q are the same attitude
    align = q if np.dot(q, q2) >= 0 else -q
    np.testing.assert_allclose(align, q2, atol=1e-10)


def test_const_omega_z_closed_form():
    wz = 0.4
    dt = 1.25
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    q = quat_integrate_const_omega(q0, np.array([0.0, 0.0, wz]), dt)
    expected = axis_angle_to_quat(np.array([0.0, 0.0, 1.0]), wz * dt)
    np.testing.assert_allclose(q, expected, atol=1e-12)
    assert abs(np.linalg.norm(q) - 1.0) < 1e-12


def test_kinematics_matches_finite_difference():
    q = quat_normalize([0.8, 0.2, -0.1, 0.4])
    omega = np.array([0.3, -0.2, 0.5])
    dt = 1e-6
    q_next = quat_integrate_const_omega(q, omega, dt)
    qdot_fd = (q_next - q) / dt
    qdot = quat_derivative(q, omega)
    np.testing.assert_allclose(qdot_fd, qdot, atol=2e-6)


def test_error_zero_when_attitudes_match():
    q = axis_angle_to_quat(np.array([0.1, 0.7, 0.2]), 0.9)
    qe = quat_error(q, q)
    np.testing.assert_allclose(np.abs(qe), [1.0, 0.0, 0.0, 0.0], atol=1e-15)
    assert geodesic_angle(q, q) < 1e-15
    np.testing.assert_allclose(rotation_vector_error(q, q), 0.0, atol=1e-15)


def test_shortest_path_sign():
    q = np.array([1.0, 0.0, 0.0, 0.0])
    q_des = axis_angle_to_quat(np.array([0.0, 0.0, 1.0]), np.pi * 0.5)
    # Error vector should have a -z component so -Kp * e_q produces +z torque.
    e = quat_error(q, q_des)
    if e[0] < 0:
        e = -e
    assert e[3] < 0.0


@pytest.mark.parametrize("n_steps", [10, 250])
def test_repeated_integration_preserves_unit_norm(n_steps):
    q = np.array([1.0, 0.0, 0.0, 0.0])
    rng = np.random.default_rng(0)
    for _ in range(n_steps):
        omega = rng.normal(scale=0.4, size=3)
        q = quat_integrate_const_omega(q, omega, 0.05)
    assert abs(np.linalg.norm(q) - 1.0) < 1e-12


def test_normalize_zero_quaternion_is_identity():
    np.testing.assert_allclose(quat_normalize([0.0, 0.0, 0.0, 0.0]), [1.0, 0.0, 0.0, 0.0])


def test_double_cover_is_same_attitude():
    q = axis_angle_to_quat(np.array([0.2, 0.5, 0.8]), 0.7)
    assert geodesic_angle(q, -q) < 1e-15
    np.testing.assert_allclose(quat_to_rotation(q), quat_to_rotation(-q), atol=1e-15)


def test_quat_to_euler321_principal_rotations():
    ypr_yaw = quat_to_euler321(axis_angle_to_quat(np.array([0.0, 0.0, 1.0]), 0.4))
    np.testing.assert_allclose(ypr_yaw, [0.4, 0.0, 0.0], atol=1e-12)
    ypr_pitch = quat_to_euler321(axis_angle_to_quat(np.array([0.0, 1.0, 0.0]), 0.3))
    np.testing.assert_allclose(ypr_pitch, [0.0, 0.3, 0.0], atol=1e-12)
    ypr_roll = quat_to_euler321(axis_angle_to_quat(np.array([1.0, 0.0, 0.0]), 0.25))
    np.testing.assert_allclose(ypr_roll, [0.0, 0.0, 0.25], atol=1e-12)

"""Estimator-only checks: vector updates should reduce attitude error."""

import numpy as np

from attitude_sim.estimation import ComplementaryFilter, MultiplicativeEKF
from attitude_sim.quaternions import (
    axis_angle_to_quat,
    geodesic_angle,
    quat_multiply,
    quat_to_rotation,
)


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

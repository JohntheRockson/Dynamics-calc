"""Controller construction, inertia-scaled defaults, and disturbance rejection."""

import numpy as np
import pytest

from attitude_sim.controls import (
    DEFAULT_TORQUE_LIMIT,
    LQR_THETA_REF,
    AttitudeLQR,
    LQRAttitudeController,
    PIDAttitudeController,
    bryson_lqr_weights,
    care_residual,
    design_attitude_lqr,
    linearize_attitude,
    make_controller,
    pid_gains_from_wn,
    solve_care,
)
from attitude_sim.plant import RigidBody, step_rigid_body
from attitude_sim.quaternions import axis_angle_to_quat, geodesic_angle


def _inertia() -> np.ndarray:
    return np.diag([0.05, 0.06, 0.07])


def _hold_against_disturbance(ctrl, inertia, tau_dist, t_final=30.0, dt=0.01):
    """True-state hold at identity under a constant body-frame torque."""
    body = RigidBody(inertia)
    q = np.array([1.0, 0.0, 0.0, 0.0])
    q_des = q.copy()
    omega = np.zeros(3)
    ctrl.reset()
    tau_dist = np.asarray(tau_dist, dtype=float).reshape(3)
    errors = []
    for _ in range(int(np.round(t_final / dt))):
        tau = ctrl.command(q, omega, q_des, dt=dt)
        q, omega = step_rigid_body(body, q, omega, tau + tau_dist, dt)
        errors.append(geodesic_angle(q, q_des))
    return q, omega, np.asarray(errors)


def test_make_controller_modes():
    J = _inertia()
    pid = make_controller("pid", J, torque_limit=0.02)
    lqr = make_controller("lqr", J, torque_limit=0.02)
    q = np.array([1.0, 0.0, 0.0, 0.0])
    q_des = axis_angle_to_quat(np.array([0.0, 0.0, 1.0]), 0.4)
    omega = np.zeros(3)
    tau_pid = pid.command(q, omega, q_des, dt=0.01)
    tau_lqr = lqr.command(q, omega, q_des, dt=0.01)
    assert tau_pid.shape == (3,)
    assert tau_lqr.shape == (3,)
    # Rest-to-rest about +z should command a +z torque from identity.
    assert tau_pid[2] > 0.0
    assert tau_lqr[2] > 0.0


def test_pid_default_gains_scale_with_inertia():
    J1 = _inertia()
    J2 = 2.0 * J1
    p1 = PIDAttitudeController(J1)
    p2 = PIDAttitudeController(J2)
    kp, kd, ki = pid_gains_from_wn(J1, p1.wn, p1.zeta, p1.ki_wn_coeff)
    np.testing.assert_allclose(p1.kp, kp)
    np.testing.assert_allclose(p1.kd, kd)
    np.testing.assert_allclose(p1.ki, ki)
    np.testing.assert_allclose(p2.kp, 2.0 * p1.kp)
    np.testing.assert_allclose(p2.kd, 2.0 * p1.kd)
    np.testing.assert_allclose(p2.ki, 2.0 * p1.ki)
    # Opening 75° PD torque stays on the order of the 20 mN·m actuator.
    q = np.array([1.0, 0.0, 0.0, 0.0])
    q_des = axis_angle_to_quat(np.array([0.2, 0.5, 0.84]), np.deg2rad(75.0))
    tau0 = p1.command(q, np.zeros(3), q_des, dt=0.01)
    assert np.linalg.norm(tau0) <= DEFAULT_TORQUE_LIMIT * (1.0 + 1e-9)
    assert np.linalg.norm(tau0) > 0.5 * DEFAULT_TORQUE_LIMIT


def test_lqr_bryson_placeholders_and_optional_k():
    J = _inertia()
    q_att, q_rate, r_torque = bryson_lqr_weights()
    ctrl = LQRAttitudeController(J)
    np.testing.assert_allclose(ctrl.q_att, q_att)
    np.testing.assert_allclose(ctrl.q_rate, q_rate)
    np.testing.assert_allclose(ctrl.r_torque, r_torque)
    k_theta = np.sqrt(ctrl.q_att / ctrl.r_torque)
    np.testing.assert_allclose(np.diag(ctrl.K[:, :3]), k_theta, rtol=1e-6)
    # Linear region ~ θ_ref, not a ~1° saturating bang-bang.
    assert DEFAULT_TORQUE_LIMIT / k_theta >= LQR_THETA_REF * 0.99
    A, B = linearize_attitude(J)
    eigs = np.linalg.eigvals(A - B @ ctrl.K)
    assert np.all(np.real(eigs) < -1e-6)
    # Optional gain path: skip CARE when K is supplied.
    K_forced = np.zeros((3, 6))
    K_forced[:, :3] = 0.05 * np.eye(3)
    K_forced[:, 3:] = 0.1 * np.eye(3)
    injected = LQRAttitudeController(J, K=K_forced)
    np.testing.assert_allclose(injected.K, K_forced)


def test_lqr_linear_closed_loop_hurwitz():
    J = _inertia()
    ctrl = LQRAttitudeController(J)
    A, B = linearize_attitude(J)
    eigs = np.linalg.eigvals(A - B @ ctrl.K)
    assert np.all(np.real(eigs) < -1e-6)


def test_pid_rejects_constant_body_disturbance():
    J = _inertia()
    tau_dist = np.array([0.002, -0.001, 0.0008])
    pid = PIDAttitudeController(J)
    pd = PIDAttitudeController(J, ki=0.0)
    _, omega_i, err_i = _hold_against_disturbance(pid, J, tau_dist, t_final=30.0)
    _, _omega_pd, err_pd = _hold_against_disturbance(pd, J, tau_dist, t_final=30.0)
    assert np.rad2deg(err_i[-1]) < 1.5
    assert err_i[-1] < 0.25 * err_pd[-1]
    assert np.linalg.norm(omega_i) < 0.02


def test_lqr_holds_against_constant_body_disturbance():
    J = _inertia()
    tau_dist = np.array([0.002, -0.001, 0.0008])
    lqr = LQRAttitudeController(J)
    _, omega, err = _hold_against_disturbance(lqr, J, tau_dist, t_final=20.0)
    # Proportional LQR cannot null a bias; stiffness keeps the residual small.
    assert np.rad2deg(err[-1]) < 3.0
    assert np.rad2deg(np.max(err)) < 5.0
    assert np.linalg.norm(omega) < 0.02
    # Predicted DC residual δθ ≈ K_θ⁻¹ τ_d (no integrator).
    k_theta = float(np.mean(np.diag(lqr.K[:, :3])))
    predicted = np.linalg.norm(tau_dist) / k_theta
    assert abs(err[-1] - predicted) / predicted < 0.25


def test_make_controller_unknown_mode():
    with pytest.raises(ValueError, match="unknown controller"):
        make_controller("sliding", _inertia())


def test_attitude_lqr_alias_and_mode():
    J = _inertia()
    assert AttitudeLQR is LQRAttitudeController
    a = make_controller("lqr", J)
    b = make_controller("attitude-lqr", J)
    np.testing.assert_allclose(a.K, b.K)


def test_solve_care_residual_and_numpy_path():
    J = _inertia()
    A, B = linearize_attitude(J)
    ctrl = LQRAttitudeController(J, care_method="scipy")
    K, P, A2, B2 = design_attitude_lqr(J, Q=ctrl.Q, R=ctrl.R, method="scipy")
    np.testing.assert_allclose(A2, A)
    np.testing.assert_allclose(B2, B)
    np.testing.assert_allclose(K, ctrl.K, rtol=1e-8)
    resid = care_residual(A, B, ctrl.Q, ctrl.R, P)
    assert np.linalg.norm(resid, ord="fro") < 1e-8
    assert np.all(np.linalg.eigvalsh(P) > 0.0)
    np.testing.assert_allclose(K, np.linalg.solve(ctrl.R, B.T @ P), rtol=1e-8)
    P_np = solve_care(A, B, ctrl.Q, ctrl.R, method="numpy")
    P_sp = solve_care(A, B, ctrl.Q, ctrl.R, method="scipy")
    np.testing.assert_allclose(P_np, P_sp, rtol=1e-6, atol=1e-8)
    with pytest.raises(ValueError, match="positive definite"):
        solve_care(A, B, np.eye(6), np.zeros((3, 3)))
    with pytest.raises(ValueError, match="unknown CARE method"):
        solve_care(A, B, np.eye(6), np.eye(3), method="kleinman")


def test_lqr_gain_scale_keeps_hurwitz():
    J = _inertia()
    A, B = linearize_attitude(J)
    nominal = LQRAttitudeController(J)
    scaled = LQRAttitudeController(J, gain_scale=0.7)
    np.testing.assert_allclose(scaled.K, 0.7 * nominal.K)
    eigs = np.linalg.eigvals(A - B @ scaled.K)
    assert np.all(np.real(eigs) < -1e-6)
    pid0 = PIDAttitudeController(J)
    pid = PIDAttitudeController(J, gain_scale=1.3)
    np.testing.assert_allclose(pid.kp, 1.3 * pid0.kp)
    with pytest.raises(ValueError, match="gain_scale"):
        LQRAttitudeController(J, gain_scale=0.0)


def _eigenaxis_slew(ctrl, inertia, angle_rad, axis, t_final=12.0, dt=0.01):
    """True-state rest-to-rest eigenaxis slew (identity → axis-angle)."""
    body = RigidBody(inertia)
    q = np.array([1.0, 0.0, 0.0, 0.0])
    q_des = axis_angle_to_quat(np.asarray(axis, dtype=float), float(angle_rad))
    omega = np.zeros(3)
    ctrl.reset()
    errors = []
    for _ in range(int(np.round(t_final / dt))):
        tau = ctrl.command(q, omega, q_des, dt=dt)
        q, omega = step_rigid_body(body, q, omega, tau, dt)
        errors.append(geodesic_angle(q, q_des))
    return q, omega, np.asarray(errors)


def test_lqr_settles_small_eigenaxis_vs_pid():
    """Linear-region eigenaxis (~8°) must settle; LQR competitive with PID."""
    J = _inertia()
    axis = np.array([0.2, 0.5, 0.84])
    angle = np.deg2rad(8.0)
    lqr = LQRAttitudeController(J)
    pid = PIDAttitudeController(J)
    pd = PIDAttitudeController(J, ki=0.0)
    _, w_lqr, err_lqr = _eigenaxis_slew(lqr, J, angle, axis, t_final=12.0)
    _, w_pd, err_pd = _eigenaxis_slew(pd, J, axis=axis, angle_rad=angle, t_final=12.0)
    _, _, err_pid = _eigenaxis_slew(pid, J, angle, axis, t_final=12.0)
    assert np.rad2deg(err_lqr[0]) == pytest.approx(8.0, abs=0.05)
    assert np.rad2deg(err_lqr[-1]) < 0.25
    assert np.rad2deg(err_pd[-1]) < 0.5
    assert np.linalg.norm(w_lqr) < 0.01
    assert np.linalg.norm(w_pd) < 0.02
    assert err_lqr[-1] < 0.05 * err_lqr[0]
    # Same plant / IC: LQR matches PD and is at least as tight as full PID
    # (PID's slow integral mode is still moving at 12 s).
    assert err_lqr[-1] < 5.0 * err_pd[-1] + 1e-6
    assert err_lqr[-1] < err_pid[-1]
    assert err_pid[-1] < 0.5 * err_pid[0]


def test_controllers_are_double_cover_invariant():
    """q and −q are the same attitude; torque must not flip with the cover."""
    J = _inertia()
    q = axis_angle_to_quat(np.array([0.2, 0.5, 0.8]), 0.7)
    q_des = axis_angle_to_quat(np.array([0.1, -0.2, 0.9]), -0.4)
    omega = np.array([0.05, -0.02, 0.01])
    pid = PIDAttitudeController(J)
    lqr = LQRAttitudeController(J)
    pid.reset()
    t_pid = pid.command(q, omega, q_des, dt=0.01)
    pid.reset()
    t_pid_neg_q = pid.command(-q, omega, q_des, dt=0.01)
    pid.reset()
    t_pid_neg_des = pid.command(q, omega, -q_des, dt=0.01)
    np.testing.assert_allclose(t_pid, t_pid_neg_q, atol=1e-15)
    np.testing.assert_allclose(t_pid, t_pid_neg_des, atol=1e-15)
    t_lqr = lqr.command(q, omega, q_des, dt=0.01)
    np.testing.assert_allclose(t_lqr, lqr.command(-q, omega, q_des, dt=0.01), atol=1e-15)
    np.testing.assert_allclose(t_lqr, lqr.command(q, omega, -q_des, dt=0.01), atol=1e-15)

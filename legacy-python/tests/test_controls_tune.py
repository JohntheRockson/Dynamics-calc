"""Gain / cost auto-tune helpers: SPD Q/R, positive PID, settle smoke."""

import numpy as np
import pytest

from attitude_sim.controls import (
    DEFAULT_TORQUE_LIMIT,
    LQR_OMEGA_REF,
    LQR_TAU_REF,
    LQR_THETA_REF,
    PID_KI_WN_COEFF,
    PID_WN,
    PID_ZETA,
    LQRAttitudeController,
    PIDAttitudeController,
    as_inertia_ref,
    bryson_lqr_costs,
    bryson_lqr_weights,
    cubesat_controller_kwargs,
    cubesat_gain_report,
    default_cubesat_inertia,
    design_attitude_lqr,
    is_spd,
    lqr_second_order_equiv,
    make_cubesat_controller,
    pid_gains_from_wn,
    recommended_pid_wn,
    tune_pid_second_order,
)
from attitude_sim.plant import RigidBody, step_rigid_body
from attitude_sim.quaternions import axis_angle_to_quat, geodesic_angle


def _inertia() -> np.ndarray:
    return default_cubesat_inertia()


def test_as_inertia_ref_broadcasts_and_rejects():
    J = as_inertia_ref(0.06)
    np.testing.assert_allclose(J, 0.06 * np.eye(3))
    np.testing.assert_allclose(as_inertia_ref([0.05, 0.06, 0.07]), _inertia())
    np.testing.assert_allclose(as_inertia_ref(_inertia()), _inertia())
    with pytest.raises(ValueError, match="positive"):
        as_inertia_ref(0.0)
    with pytest.raises(ValueError, match="positive"):
        as_inertia_ref([-0.01, 0.06, 0.07])
    with pytest.raises(ValueError, match="positive definite"):
        as_inertia_ref(np.diag([0.05, -0.06, 0.07]))
    with pytest.raises(ValueError, match="shape"):
        as_inertia_ref(np.ones(4))


def test_tune_pid_second_order_positive_spd_gains():
    J = _inertia()
    kp, kd, ki = tune_pid_second_order(J, PID_WN, PID_ZETA)
    for gain, name in ((kp, "Kp"), (kd, "Kd"), (ki, "Ki")):
        assert is_spd(gain), name
        assert np.all(np.diag(gain) > 0.0)
    np.testing.assert_allclose(kp, 2.0 * (PID_WN**2) * J)
    np.testing.assert_allclose(kd, 2.0 * PID_ZETA * PID_WN * J)
    np.testing.assert_allclose(ki, PID_KI_WN_COEFF * (PID_WN**3) * J)
    # Documented stock diagonals (docs/controls.md tuning report).
    np.testing.assert_allclose(np.diag(kp), [0.025, 0.030, 0.035])
    np.testing.assert_allclose(np.diag(kd), [0.05, 0.06, 0.07])
    np.testing.assert_allclose(np.diag(ki), [0.003125, 0.00375, 0.004375])
    # Scalar I_ref and the historical wrapper agree on a 3×3.
    kp_s, kd_s, ki_s = tune_pid_second_order(0.06, 0.4, 0.8)
    assert is_spd(kp_s) and is_spd(kd_s) and is_spd(ki_s)
    np.testing.assert_allclose((kp, kd, ki), pid_gains_from_wn(J, PID_WN, PID_ZETA, PID_KI_WN_COEFF))


def test_tune_pid_second_order_pd_only_and_rejects():
    kp, kd, ki = tune_pid_second_order(_inertia(), 0.5, 1.0, ki_wn_coeff=0.0)
    assert is_spd(kp) and is_spd(kd)
    np.testing.assert_allclose(ki, 0.0)
    with pytest.raises(ValueError, match="wn"):
        tune_pid_second_order(_inertia(), 0.0, 1.0)
    with pytest.raises(ValueError, match="zeta"):
        tune_pid_second_order(_inertia(), 0.5, -1.0)
    with pytest.raises(ValueError, match="ki_wn_coeff"):
        tune_pid_second_order(_inertia(), 0.5, 1.0, ki_wn_coeff=-0.1)


def test_recommended_pid_wn_matches_stock_rounding():
    J = _inertia()
    wn = recommended_pid_wn(J)
    # λ_max(J)=0.07, 75°, 20 mN·m → ≈ 0.467; shipped PID_WN is the round 0.5.
    expected = np.sqrt(DEFAULT_TORQUE_LIMIT / (0.07 * np.deg2rad(75.0)))
    assert wn == pytest.approx(expected)
    assert 0.45 < wn < PID_WN
    assert PID_WN == pytest.approx(0.5)
    twice = recommended_pid_wn(2.0 * J, tau_max=2.0 * DEFAULT_TORQUE_LIMIT)
    assert twice == pytest.approx(wn)
    with pytest.raises(ValueError, match="tau_max"):
        recommended_pid_wn(J, tau_max=0.0)


def test_bryson_lqr_costs_spd_and_design_defaults():
    Q, R = bryson_lqr_costs()
    assert Q.shape == (6, 6)
    assert R.shape == (3, 3)
    assert is_spd(Q)
    assert is_spd(R)
    q_att, q_rate, r_torque = bryson_lqr_weights()
    np.testing.assert_allclose(np.diag(Q), [q_att] * 3 + [q_rate] * 3)
    np.testing.assert_allclose(R, r_torque * np.eye(3))
    np.testing.assert_allclose(np.diag(Q)[:3], 16.0)
    np.testing.assert_allclose(np.diag(Q)[3:], 25.0)
    np.testing.assert_allclose(R, 2500.0 * np.eye(3))
    J = _inertia()
    K, _P, _A, _B = design_attitude_lqr(J)
    K2, _P2, _A2, _B2 = design_attitude_lqr(J, Q=Q, R=R)
    np.testing.assert_allclose(K, K2)
    ctrl = LQRAttitudeController(J)
    np.testing.assert_allclose(ctrl.Q, Q)
    np.testing.assert_allclose(ctrl.R, R)
    with pytest.raises(ValueError, match="theta_ref"):
        bryson_lqr_costs(theta_ref=0.0)
    with pytest.raises(ValueError, match="q_att"):
        bryson_lqr_costs(q_att=-1.0)


def test_cubesat_gain_report_matches_helpers_and_docs():
    rep = cubesat_gain_report()
    np.testing.assert_allclose(rep.inertia, _inertia())
    assert rep.tau_max == DEFAULT_TORQUE_LIMIT
    assert rep.wn == PID_WN
    assert rep.zeta == PID_ZETA
    assert rep.theta_ref == LQR_THETA_REF
    assert rep.omega_ref == LQR_OMEGA_REF
    assert rep.tau_ref == LQR_TAU_REF
    kp, kd, ki = tune_pid_second_order(rep.inertia, rep.wn, rep.zeta)
    np.testing.assert_allclose(rep.kp, kp)
    np.testing.assert_allclose(rep.kd, kd)
    np.testing.assert_allclose(rep.ki, ki)
    Q, R = bryson_lqr_costs()
    np.testing.assert_allclose(rep.Q, Q)
    np.testing.assert_allclose(rep.R, R)
    assert is_spd(rep.Q) and is_spd(rep.R)
    assert np.all(rep.lqr_wn_equiv > 0.0)
    assert np.all(rep.lqr_zeta_equiv > 0.5)
    # Mid-axis ~ 1.15 rad/s, ζ ≈ 1 (docs table).
    assert rep.lqr_wn_equiv[1] == pytest.approx(np.sqrt(0.08 / 0.06), rel=1e-3)
    assert np.mean(rep.lqr_zeta_equiv) == pytest.approx(1.0, abs=0.15)
    # Wheel resize retunes R and recommended wn together.
    tight = cubesat_gain_report(tau_max=0.01)
    np.testing.assert_allclose(tight.R, 4.0 * R)
    assert tight.recommended_wn == pytest.approx(recommended_pid_wn(rep.inertia, tau_max=0.01))


def test_lqr_second_order_equiv_on_care_gain():
    J = _inertia()
    K, _P, _A, _B = design_attitude_lqr(J)
    wn, zeta = lqr_second_order_equiv(K, J)
    np.testing.assert_allclose(wn**2 * np.diag(J), np.diag(K[:, :3]), rtol=1e-10)
    assert np.all(wn > 0.0)
    assert np.all(zeta > 0.0)
    with pytest.raises(ValueError, match="3x6"):
        lqr_second_order_equiv(np.eye(3), J)


def _eigenaxis_slew(ctrl, inertia, angle_rad, axis, t_final=12.0, dt=0.01):
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


def test_tuned_gains_settle_small_eigenaxis():
    """Closed-loop smoke: helpers' gains settle an 8° rest-to-rest slew."""
    J = _inertia()
    axis = np.array([0.2, 0.5, 0.84])
    angle = np.deg2rad(8.0)
    kp, kd, _ki0 = tune_pid_second_order(J, PID_WN, PID_ZETA, ki_wn_coeff=0.0)
    kp_i, kd_i, ki_i = tune_pid_second_order(J, PID_WN, PID_ZETA)
    Q, R = bryson_lqr_costs()
    pd = PIDAttitudeController(J, kp=kp, kd=kd, ki=0.0)
    pid = PIDAttitudeController(J, kp=kp_i, kd=kd_i, ki=ki_i)
    lqr = LQRAttitudeController(J, Q=Q, R=R)
    _, w_pd, err_pd = _eigenaxis_slew(pd, J, angle, axis, t_final=12.0)
    _, _w_pid, err_pid = _eigenaxis_slew(pid, J, angle, axis, t_final=12.0)
    _, w_lqr, err_lqr = _eigenaxis_slew(lqr, J, angle, axis, t_final=12.0)
    assert np.rad2deg(err_pd[-1]) < 0.5
    assert np.rad2deg(err_lqr[-1]) < 0.25
    assert np.linalg.norm(w_pd) < 0.02
    assert np.linalg.norm(w_lqr) < 0.01
    assert err_pd[-1] < 0.1 * err_pd[0]
    assert err_lqr[-1] < 0.05 * err_lqr[0]
    # Full PID still shrinks the error; the slow Ki mode is moving at 12 s.
    assert err_pid[-1] < 0.5 * err_pid[0]


def test_make_cubesat_controller_matches_report():
    J = _inertia()
    pid_kw = cubesat_controller_kwargs("pid", J)
    lqr_kw = cubesat_controller_kwargs("lqr", J)
    np.testing.assert_allclose(pid_kw["kp"], 2.0 * (PID_WN**2) * J)
    Q, R = bryson_lqr_costs()
    np.testing.assert_allclose(lqr_kw["Q"], Q)
    np.testing.assert_allclose(lqr_kw["R"], R)
    pid = make_cubesat_controller("pid", J)
    np.testing.assert_allclose(pid.kp, pid_kw["kp"])
    lqr = make_cubesat_controller("lqr", J)
    np.testing.assert_allclose(lqr.Q, Q)
    with pytest.raises(ValueError, match="unknown controller"):
        cubesat_controller_kwargs("nope", J)


def test_is_spd_rejects_indefinite():
    assert is_spd(np.eye(3))
    assert is_spd(np.zeros((3, 3)), semi=True)
    assert not is_spd(np.zeros((3, 3)))
    assert not is_spd(np.diag([1.0, -1.0, 1.0]))
    assert not is_spd(np.array([[1.0, 2.0], [0.0, 1.0]]))
    assert not is_spd(np.ones(3))

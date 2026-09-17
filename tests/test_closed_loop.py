"""Closed-loop SimLab smoke tests, including control on filter estimates."""

import warnings

import numpy as np
import pytest

from attitude_sim.estimation import ComplementaryFilter, MultiplicativeEKF
from attitude_sim.quaternions import axis_angle_to_quat, geodesic_angle
from attitude_sim.sim import SimConfig, make_scenario_config, make_sim_estimator, run_slew


def _cfg(**kwargs) -> SimConfig:
    base = dict(
        dt=0.01,
        t_final=22.0,
        plot=False,
        gif=False,
        seed=3,
        torque_limit=0.02,
    )
    base.update(kwargs)
    return SimConfig(**base)


def test_pid_truth_slew_settles():
    log = run_slew(_cfg(controller="pid", estimator="truth"))
    np.testing.assert_allclose(np.linalg.norm(log.q, axis=1), 1.0, atol=1e-12)
    assert log.final_att_error_deg < 2.0
    assert np.linalg.norm(log.omega[-1]) < 0.02
    # Error should drop substantially from the opening transient.
    assert log.att_error[-1] < 0.25 * log.att_error[0]


def test_lqr_truth_slew_settles():
    log = run_slew(_cfg(controller="lqr", estimator="truth"))
    assert log.final_att_error_deg < 2.5
    assert np.linalg.norm(log.omega[-1]) < 0.03


def test_lqr_small_eigenaxis_simlab_settles():
    """SimLab wiring: small eigenaxis (linear region) settles under LQR."""
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    q_des = axis_angle_to_quat(np.array([0.0, 0.0, 1.0]), np.deg2rad(8.0))
    log = run_slew(
        _cfg(
            controller="lqr",
            estimator="truth",
            q0=q0,
            q_des=q_des,
            t_final=10.0,
            seed=11,
        )
    )
    assert log.final_att_error_deg < 0.25
    assert np.linalg.norm(log.omega[-1]) < 0.01
    assert log.att_error[-1] < 0.05 * log.att_error[0]


def test_pid_on_mekf_estimates_smoke():
    log = run_slew(_cfg(controller="pid", estimator="mekf", t_final=28.0))
    assert log.final_att_error_deg < 6.0
    assert np.linalg.norm(log.omega[-1]) < 0.05
    assert log.est_att_error is not None
    np.testing.assert_allclose(np.linalg.norm(log.q, axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(log.q_hat, axis=1), 1.0, atol=1e-12)
    # Filter should be tracking the plant by the second half of the run.
    late = log.est_att_error[len(log.t) // 2 :]
    assert np.rad2deg(np.median(late)) < 5.0


def test_lqr_on_mahony_estimates_smoke():
    log = run_slew(_cfg(controller="lqr", estimator="mahony", t_final=28.0))
    assert log.final_att_error_deg < 8.0
    assert np.linalg.norm(log.omega[-1]) < 0.06


def test_pid_truth_detumble_dumps_rate():
    log = run_slew(
        make_scenario_config(
            "detumble",
            controller="pid",
            estimator="truth",
            t_final=22.0,
            plot=False,
            gif=False,
            seed=3,
        )
    )
    w0 = float(np.linalg.norm(log.omega[0]))
    w1 = float(np.linalg.norm(log.omega[-1]))
    assert w0 > 0.5
    assert w1 < 0.03
    assert w1 < 0.15 * w0
    assert log.final_att_error_deg < 3.0
    assert log.scenario == "detumble"


def test_scenario_plot_names(tmp_path):
    log = run_slew(
        _cfg(
            scenario="detumble",
            estimator="truth",
            t_final=0.05,
            plot=True,
            gif=False,
            out_dir=tmp_path,
            omega0=np.array([0.2, 0.0, 0.0]),
        )
    )
    assert log.plot_path == tmp_path / "detumble_summary.png"
    assert log.plot_path.is_file()


def test_pid_hold_rejects_body_disturbance():
    q_des = np.array([1.0, 0.0, 0.0, 0.0])
    tau_dist = np.array([0.002, -0.001, 0.0008])
    log = run_slew(
        _cfg(
            controller="pid",
            estimator="truth",
            q0=q_des,
            q_des=q_des,
            tau_dist=tau_dist,
            t_final=30.0,
        )
    )
    assert log.final_att_error_deg < 1.5
    assert np.linalg.norm(log.omega[-1]) < 0.02
    # Logged τ is the control command (not plant torque); at rest it cancels τ_d.
    np.testing.assert_allclose(log.tau[-1], -tau_dist, atol=2e-4)


def test_lqr_hold_rejects_body_disturbance():
    q_des = np.array([1.0, 0.0, 0.0, 0.0])
    tau_dist = np.array([0.002, -0.001, 0.0008])
    log = run_slew(
        _cfg(
            controller="lqr",
            estimator="truth",
            q0=q_des,
            q_des=q_des,
            tau_dist=tau_dist,
            t_final=20.0,
        )
    )
    assert log.final_att_error_deg < 3.0
    assert np.linalg.norm(log.omega[-1]) < 0.02


def test_pid_mekf_detumble_dumps_rate():
    log = run_slew(
        make_scenario_config(
            "detumble",
            controller="pid",
            estimator="mekf",
            t_final=22.0,
            plot=False,
            gif=False,
            seed=3,
        )
    )
    w0 = float(np.linalg.norm(log.omega[0]))
    w1 = float(np.linalg.norm(log.omega[-1]))
    assert w0 > 0.5
    assert w1 < 0.05
    assert w1 < 0.15 * w0
    assert log.final_att_error_deg < 6.0
    np.testing.assert_allclose(np.linalg.norm(log.q, axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(log.q_hat, axis=1), 1.0, atol=1e-12)


def test_make_sim_estimator_forwards_gyro_noise():
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    cfg = SimConfig(estimator="mekf", gyro_sigma_v=1.23e-3, gyro_sigma_u=4.56e-6)
    est = make_sim_estimator(cfg, q0)
    assert isinstance(est, MultiplicativeEKF)
    assert est.sigma_v == 1.23e-3
    assert est.sigma_u == 4.56e-6
    assert make_sim_estimator(SimConfig(estimator="truth"), q0) is None
    mahony = make_sim_estimator(SimConfig(estimator="mahony"), q0)
    assert isinstance(mahony, ComplementaryFilter)


def test_run_rejects_non_positive_dt():
    with pytest.raises(ValueError, match="dt must be positive"):
        run_slew(_cfg(dt=0.0, t_final=0.1, estimator="truth"))


def test_gyro_only_estimator_warns():
    with pytest.warns(UserWarning, match="vector sensors"):
        run_slew(_cfg(estimator="mekf", use_mag=False, use_sun=False, t_final=0.05))
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        run_slew(_cfg(estimator="truth", use_mag=False, use_sun=False, t_final=0.05))
    assert not any("vector sensors" in str(w.message) for w in rec)


def test_truth_path_skips_sensor_sampling(monkeypatch):
    from attitude_sim.sensors import GyroModel, VectorSensor

    gyro_calls = {"n": 0}
    vec_calls = {"n": 0}
    orig_gyro = GyroModel.measure
    orig_vec = VectorSensor.measure

    def gyro_measure(self, omega, dt):
        gyro_calls["n"] += 1
        return orig_gyro(self, omega, dt)

    def vec_measure(self, q):
        vec_calls["n"] += 1
        return orig_vec(self, q)

    monkeypatch.setattr(GyroModel, "measure", gyro_measure)
    monkeypatch.setattr(VectorSensor, "measure", vec_measure)

    log = run_slew(_cfg(estimator="truth", t_final=0.05))
    assert gyro_calls["n"] == 0
    assert vec_calls["n"] == 0
    np.testing.assert_allclose(log.q_hat, log.q)
    np.testing.assert_allclose(log.omega_hat, log.omega)

    run_slew(_cfg(estimator="mekf", t_final=0.05))
    assert gyro_calls["n"] > 0
    assert vec_calls["n"] > 0


def test_default_estimator_starts_at_true_q0():
    q0 = axis_angle_to_quat(np.array([0.2, 0.5, 0.8]), 0.7)
    log = run_slew(_cfg(q0=q0, estimator="mekf", t_final=0.03, coarse_init=False))
    assert log.est_att_error is not None
    assert np.rad2deg(log.est_att_error[0]) < 2.0


def test_coarse_init_does_not_start_at_true_q0_and_stays_unit():
    q0 = axis_angle_to_quat(np.array([0.2, 0.5, 0.8]), 0.7)
    log_true = run_slew(_cfg(q0=q0, estimator="mekf", t_final=0.03, coarse_init=False, seed=4))
    log_triad = run_slew(_cfg(q0=q0, estimator="mekf", t_final=0.03, coarse_init=True, seed=4))
    np.testing.assert_allclose(log_true.q[0], log_triad.q[0])
    assert geodesic_angle(log_true.q_hat[0], log_triad.q_hat[0]) > np.deg2rad(0.05)
    assert log_triad.est_att_error is not None
    assert np.rad2deg(log_triad.est_att_error[0]) < 15.0
    np.testing.assert_allclose(np.linalg.norm(log_triad.q_hat, axis=1), 1.0, atol=1e-12)


def test_coarse_init_without_two_sensors_warns_and_keeps_true_q0():
    q0 = axis_angle_to_quat(np.array([0.0, 0.0, 1.0]), 0.5)
    with pytest.warns(UserWarning, match="coarse TRIAD init skipped"):
        log = run_slew(
            _cfg(
                q0=q0,
                estimator="mekf",
                t_final=0.03,
                coarse_init=True,
                use_sun=False,
            )
        )
    assert log.est_att_error is not None
    assert np.rad2deg(log.est_att_error[0]) < 2.0

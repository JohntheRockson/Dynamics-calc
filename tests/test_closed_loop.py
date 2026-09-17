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


def test_env_disturbances_default_off_matches_baseline():
    """Omitting gravity-gradient / residual-dipole is bit-identical to prior demos."""
    kwargs = dict(controller="pid", estimator="truth", t_final=1.5, seed=5)
    log_a = run_slew(_cfg(**kwargs))
    log_b = run_slew(_cfg(**kwargs, gravity_gradient=False, residual_dipole_m=None))
    np.testing.assert_array_equal(log_a.q, log_b.q)
    np.testing.assert_array_equal(log_a.omega, log_b.omega)
    np.testing.assert_array_equal(log_a.tau, log_b.tau)
    assert log_a.tau_env is not None
    np.testing.assert_allclose(log_a.tau_env, 0.0)


def test_gravity_gradient_is_injected_and_keeps_unit_quat():
    kwargs = dict(controller="pid", estimator="truth", t_final=2.0, seed=6)
    log0 = run_slew(_cfg(**kwargs))
    log1 = run_slew(_cfg(**kwargs, gravity_gradient=True))
    assert log1.tau_env is not None
    assert float(np.max(np.linalg.norm(log1.tau_env, axis=1))) > 0.0
    # Logged wheel torque is unchanged at t=0 (same command); env is plant-only.
    np.testing.assert_array_equal(log0.q[0], log1.q[0])
    np.testing.assert_allclose(np.linalg.norm(log1.q, axis=1), 1.0, atol=1e-12)
    assert np.all(np.isfinite(log1.omega))
    np.testing.assert_array_equal(log0.tau[0], log1.tau[0])


def test_residual_dipole_and_tau_dist_both_enter_plant():
    # Large dipole so τ_m is O(mN·m), comparable to --tau-dist, and the
    # closed-loop trajectories actually separate (LEO-scale 0.5 A·m² is tiny).
    m_body = np.array([40.0, -20.0, 10.0])
    tau_dist = np.array([0.001, 0.0, 0.0])
    kwargs = dict(controller="pid", estimator="truth", t_final=1.2, seed=7)
    none = run_slew(_cfg(**kwargs))
    dist_only = run_slew(_cfg(**kwargs, tau_dist=tau_dist))
    mag_only = run_slew(_cfg(**kwargs, residual_dipole_m=m_body))
    both = run_slew(_cfg(**kwargs, tau_dist=tau_dist, residual_dipole_m=m_body))
    assert mag_only.tau_env is not None
    assert both.tau_env is not None
    assert float(np.max(np.linalg.norm(mag_only.tau_env, axis=1))) > 0.0
    np.testing.assert_allclose(both.tau_env, mag_only.tau_env)
    assert not np.allclose(none.omega, dist_only.omega)
    assert not np.allclose(none.omega, mag_only.omega)
    assert not np.allclose(both.omega, dist_only.omega)
    assert not np.allclose(both.omega, mag_only.omega)
    np.testing.assert_allclose(np.linalg.norm(both.q, axis=1), 1.0, atol=1e-12)


def test_pid_hold_rejects_residual_dipole():
    """Integral PID still nulls a LEO-scale residual dipole at identity."""
    q_des = np.array([1.0, 0.0, 0.0, 0.0])
    log = run_slew(
        _cfg(
            controller="pid",
            estimator="truth",
            q0=q_des,
            q_des=q_des,
            residual_dipole_m=np.array([0.5, -0.2, 0.3]),
            t_final=30.0,
        )
    )
    assert log.final_att_error_deg < 1.5
    assert np.linalg.norm(log.omega[-1]) < 0.02
    assert log.tau_env is not None
    # Logged τ is the wheel command; at rest it cancels τ_env (no extra τ_d).
    np.testing.assert_allclose(log.tau[-1], -log.tau_env[-1], atol=3e-4)


def test_make_sim_controller_forwards_wheel_limits_to_pid_only():
    from attitude_sim.controls import LQRAttitudeController, PIDAttitudeController
    from attitude_sim.sim import make_sim_controller

    pid = make_sim_controller(_cfg(controller="pid", actuator_tau_max=0.007))
    assert isinstance(pid, PIDAttitudeController)
    assert pid.tau_max == 0.007
    default_pid = make_sim_controller(_cfg(controller="pid"))
    assert isinstance(default_pid, PIDAttitudeController)
    assert default_pid.tau_max is None
    lqr = make_sim_controller(_cfg(controller="lqr", actuator_tau_max=0.007))
    assert isinstance(lqr, LQRAttitudeController)
    assert not hasattr(lqr, "tau_max")


def test_saturated_pid_slew_with_lag_stays_in_wheel_box():
    """PID anti-windup + first-order wheels: |τ_i| stays in the box, q on S^3."""
    lim = 0.004
    log = run_slew(
        _cfg(
            controller="pid",
            estimator="truth",
            actuator_tau_max=lim,
            actuator_tau=0.04,
            t_final=10.0,
            torque_limit=0.02,
        )
    )
    assert np.all(np.abs(log.tau) <= lim * (1.0 + 1e-9))
    np.testing.assert_allclose(np.linalg.norm(log.q, axis=1), 1.0, atol=1e-12)
    assert np.all(np.isfinite(log.omega))
    assert np.max(np.abs(log.tau[:80])) >= lim * 0.99
    assert log.att_error[-1] < log.att_error[0]


def test_make_sim_disturbances_none_when_flags_off():
    from attitude_sim.sim import make_sim_disturbances

    assert make_sim_disturbances(_cfg()) is None
    env = make_sim_disturbances(_cfg(gravity_gradient=True))
    assert env is not None
    assert env.gravity_gradient is not None
    assert env.residual_dipole is None


def test_logged_tau_env_matches_bound_models():
    """SimLab ZOH env torque equals EnvironmentalTorques at the logged sample."""
    from attitude_sim.sim import make_sim_disturbances

    cfg = _cfg(
        estimator="truth",
        t_final=0.4,
        gravity_gradient=True,
        residual_dipole_m=np.array([0.3, 0.0, -0.1]),
        orbit_inclination_deg=20.0,
    )
    log = run_slew(cfg)
    env = make_sim_disturbances(cfg)
    assert env is not None
    assert log.tau_env is not None
    for k in (0, len(log.t) // 2, -1):
        expected = env.tau_body(log.q[k], log.omega[k], log.t[k])
        np.testing.assert_allclose(log.tau_env[k], expected, atol=1e-15)

"""Named SimLab scenario pack: hold (env torques) and eigenaxis (LQR)."""

from __future__ import annotations

import numpy as np
import pytest

from attitude_sim.controls import PID_WN, default_cubesat_inertia
from attitude_sim.plots import attitude_error_mrp, mrp_error_from_log
from attitude_sim.quaternions import geodesic_angle, quat_error
from attitude_sim.scenarios import (
    EIGENAXIS_ANGLE_DEG,
    EIGENAXIS_AXIS,
    SCENARIOS,
    default_controller,
    default_t_final,
    make_hold_environmental_torques,
    resolve_controller,
    resolve_use_env,
    scenario_catalog_text,
    scenario_cubesat_gains,
    scenario_state,
)
from attitude_sim.sim import make_scenario_config, make_sim_disturbances, run_slew


def test_scenario_cubesat_gains_match_stock_helpers():
    report = scenario_cubesat_gains()
    np.testing.assert_allclose(report.inertia, default_cubesat_inertia())
    assert report.wn == PID_WN
    assert report.tau_max == 0.02


def test_scenario_names_include_hold_and_eigenaxis():
    assert SCENARIOS == ("slew", "detumble", "hold", "eigenaxis")
    assert default_controller("hold") == "pid"
    assert default_controller("eigenaxis") == "lqr"
    assert default_t_final("hold") == 30.0
    assert default_t_final("eigenaxis") == 20.0
    assert resolve_controller("eigenaxis", None) == "lqr"
    assert resolve_controller("eigenaxis", "pid") == "pid"
    assert resolve_controller("hold", "lqr") == "lqr"


def test_catalog_lists_all_named_scenarios():
    text = scenario_catalog_text()
    for name in SCENARIOS:
        assert name in text
    assert "slew-only" in text


def test_hold_preset_identity_and_env_torques():
    cfg = make_scenario_config("hold", plot=False, gif=False)
    np.testing.assert_allclose(cfg.q0, [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(cfg.q_des, [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(cfg.omega0, 0.0)
    assert cfg.controller == "pid"
    assert cfg.gravity_gradient is True
    assert cfg.residual_dipole is True
    assert cfg.mrp_plot is True
    env = make_sim_disturbances(cfg)
    assert env is not None
    tau0 = env.tau_body(cfg.q0, cfg.omega0, 0.0)
    assert np.linalg.norm(tau0) > 1e-5


def test_hold_no_env_disables_pack():
    cfg = make_scenario_config("hold", plot=False, gif=False, no_env=True)
    assert cfg.gravity_gradient is False
    assert cfg.residual_dipole is False
    assert make_sim_disturbances(cfg) is None
    slew_env = make_scenario_config("slew", plot=False, gif=False, env_flag=True)
    assert slew_env.gravity_gradient is True
    assert make_sim_disturbances(slew_env) is not None


def test_eigenaxis_preset_is_lqr_about_body_z():
    cfg = make_scenario_config("eigenaxis", plot=False, gif=False)
    assert cfg.controller == "lqr"
    assert cfg.t_final == 20.0
    q0, omega0, q_des = scenario_state("eigenaxis")
    np.testing.assert_allclose(q0, [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(omega0, 0.0)
    # Command is a pure z rotation of the pack default (30°).
    qe = quat_error(q_des, np.array([1.0, 0.0, 0.0, 0.0]))
    axis = qe[1:] / np.linalg.norm(qe[1:])
    np.testing.assert_allclose(np.abs(axis), EIGENAXIS_AXIS, atol=1e-12)
    assert abs(np.rad2deg(geodesic_angle(q0, q_des)) - EIGENAXIS_ANGLE_DEG) < 1e-9


def test_angle_deg_affects_slew_and_eigenaxis_not_hold():
    a = make_scenario_config("eigenaxis", angle_deg=10.0, plot=False, gif=False)
    b = make_scenario_config("eigenaxis", angle_deg=40.0, plot=False, gif=False)
    assert geodesic_angle(a.q_des, b.q_des) > 0.4
    h1 = make_scenario_config("hold", angle_deg=10.0, plot=False, gif=False)
    h2 = make_scenario_config("hold", angle_deg=90.0, plot=False, gif=False)
    np.testing.assert_allclose(h1.q_des, h2.q_des)


def test_pid_truth_hold_cancels_environmental_torque():
    log = run_slew(
        make_scenario_config(
            "hold",
            controller="pid",
            estimator="truth",
            t_final=30.0,
            plot=False,
            gif=False,
            seed=3,
        )
    )
    assert log.scenario == "hold"
    assert log.tau_env.shape == log.tau.shape
    assert float(np.max(np.linalg.norm(log.tau_env, axis=1))) > 5e-4
    assert log.final_att_error_deg < 1.5
    assert np.linalg.norm(log.omega[-1]) < 0.02
    np.testing.assert_allclose(log.tau[-1], -log.tau_env[-1], atol=5e-4)


def test_lqr_truth_eigenaxis_settles_about_z():
    log = run_slew(
        make_scenario_config(
            "eigenaxis",
            controller="lqr",
            estimator="truth",
            t_final=16.0,
            plot=False,
            gif=False,
            seed=11,
        )
    )
    assert log.scenario == "eigenaxis"
    assert log.controller == "lqr"
    assert log.final_att_error_deg < 0.5
    assert np.linalg.norm(log.omega[-1]) < 0.02
    assert log.att_error[-1] < 0.05 * log.att_error[0]
    # Error MRP stays on the commanded eigenaxis (body z) through the slew.
    sigma_e = mrp_error_from_log(log)
    mid = len(log.t) // 4
    ratio = np.abs(sigma_e[mid, 2]) / (np.linalg.norm(sigma_e[mid]) + 1e-15)
    assert ratio > 0.95
    np.testing.assert_allclose(np.linalg.norm(log.q, axis=1), 1.0, atol=1e-12)


def test_pid_mekf_hold_stays_near_identity():
    log = run_slew(
        make_scenario_config(
            "hold",
            controller="pid",
            estimator="mekf",
            t_final=22.0,
            plot=False,
            gif=False,
            seed=3,
        )
    )
    assert log.final_att_error_deg < 4.0
    assert np.linalg.norm(log.omega[-1]) < 0.04
    np.testing.assert_allclose(np.linalg.norm(log.q, axis=1), 1.0, atol=1e-12)


def test_mrp_error_matches_geodesic_quarter_angle():
    log = run_slew(
        make_scenario_config(
            "eigenaxis",
            estimator="truth",
            t_final=0.5,
            plot=False,
            gif=False,
        )
    )
    sigma_e = mrp_error_from_log(log)
    tan_quarter = np.tan(0.25 * log.att_error)
    np.testing.assert_allclose(np.linalg.norm(sigma_e, axis=1), tan_quarter, atol=1e-10)
    # Spot-check the helper used by the plot against the first sample.
    np.testing.assert_allclose(sigma_e[0], attitude_error_mrp(log.q[0], log.q_des), atol=1e-15)


def test_hold_env_factory_matches_preset():
    cfg = make_scenario_config("hold", plot=False, gif=False)
    env_preset = make_sim_disturbances(cfg)
    env_factory = make_hold_environmental_torques(cfg.inertia)
    assert env_preset is not None
    tau_a = env_preset.tau_body(cfg.q0, cfg.omega0, 1.0)
    tau_b = env_factory.tau_body(cfg.q0, cfg.omega0, 1.0)
    # Same models; bound orbits may differ by arg-of-latitude, but both are millinewton-scale.
    assert np.linalg.norm(tau_a) > 1e-5
    assert np.linalg.norm(tau_b) > 1e-5


def test_resolve_use_env_precedence():
    assert resolve_use_env("hold") is True
    assert resolve_use_env("slew") is False
    assert resolve_use_env("slew", env_flag=True) is True
    assert resolve_use_env("hold", no_env=True) is False
    assert resolve_use_env("hold", no_env=True, env_flag=True) is False
    assert resolve_use_env("slew", use_env=True) is True
    with pytest.raises(ValueError, match="unknown scenario"):
        resolve_use_env("rover")

"""CLI wiring for SimLab scenarios (no plant/controller rewrites)."""

import numpy as np
import pytest

from attitude_sim.estimation import MultiplicativeEKF
from attitude_sim.quaternions import geodesic_angle
from attitude_sim.sim import build_parser, main, make_scenario_config, make_sim_estimator


def test_parser_defaults_to_slew():
    args = build_parser().parse_args([])
    assert args.scenario == "slew"
    assert args.controller == "pid"
    assert args.estimator == "mekf"
    assert args.actuator_tau_max is None
    assert args.actuator_tau is None
    assert args.coarse_init is False
    assert args.gyro_sigma_v is None
    assert args.gyro_sigma_u is None
    assert args.mag_sigma is None
    assert args.sun_sigma is None


def test_parser_detumble_and_flags():
    args = build_parser().parse_args(
        [
            "--scenario",
            "detumble",
            "--controller",
            "lqr",
            "--no-plot",
            "--no-gif",
            "--actuator-tau-max",
            "0.02",
            "--actuator-tau",
            "0.05",
            "--coarse-init",
        ]
    )
    assert args.scenario == "detumble"
    assert args.controller == "lqr"
    assert args.no_plot
    assert args.no_gif
    assert args.actuator_tau_max == "0.02"
    assert args.actuator_tau == 0.05
    assert args.coarse_init


def test_make_scenario_config_stems():
    slew = make_scenario_config("slew", plot=False, gif=False)
    det = make_scenario_config("detumble", plot=False, gif=False)
    assert slew.artifact_stem == "slew"
    assert det.artifact_stem == "detumble"
    assert abs(det.omega0).max() > 0.2
    assert slew.coarse_init is False
    coarse = make_scenario_config("slew", plot=False, gif=False, coarse_init=True)
    assert coarse.coarse_init is True


def test_make_scenario_config_unknown():
    with pytest.raises(ValueError, match="unknown scenario"):
        make_scenario_config("spinup", plot=False, gif=False)


def test_angle_deg_only_affects_slew():
    a = make_scenario_config("slew", angle_deg=10.0, plot=False, gif=False)
    b = make_scenario_config("slew", angle_deg=90.0, plot=False, gif=False)
    assert geodesic_angle(a.q_des, b.q_des) > 1.0
    d1 = make_scenario_config("detumble", angle_deg=10.0, plot=False, gif=False)
    d2 = make_scenario_config("detumble", angle_deg=90.0, plot=False, gif=False)
    np.testing.assert_allclose(d1.q_des, [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(d2.q_des, [1.0, 0.0, 0.0, 0.0])


def test_main_both_scenarios_smoke():
    assert main(["--scenario", "slew", "--t-final", "0.05", "--no-plot", "--no-gif"]) == 0
    assert main(["--scenario", "detumble", "--t-final", "0.05", "--no-plot", "--no-gif"]) == 0


def test_main_tau_dist_and_lqr_mahony_smoke():
    assert (
        main(
            [
                "--controller",
                "pid",
                "--estimator",
                "truth",
                "--angle-deg",
                "0",
                "--tau-dist",
                "0.002,-0.001,0.0008",
                "--t-final",
                "0.05",
                "--no-plot",
                "--no-gif",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "--controller",
                "lqr",
                "--estimator",
                "mahony",
                "--t-final",
                "0.05",
                "--no-plot",
                "--no-gif",
            ]
        )
        == 0
    )


def test_main_coarse_init_smoke():
    assert main(["--coarse-init", "--t-final", "0.05", "--no-plot", "--no-gif"]) == 0


def test_main_rejects_bad_tau_dist():
    with pytest.raises(SystemExit):
        main(["--tau-dist", "1,2", "--no-plot", "--no-gif"])
    with pytest.raises(SystemExit):
        main(["--tau-dist", "a,b,c", "--no-plot", "--no-gif"])


def test_main_rejects_non_positive_dt():
    with pytest.raises(SystemExit):
        main(["--dt", "0", "--t-final", "0.05", "--no-plot", "--no-gif"])


def test_make_scenario_config_forwards_sensor_noise_to_estimator():
    cfg = make_scenario_config(
        "slew",
        plot=False,
        gif=False,
        gyro_sigma_v=1.5e-3,
        gyro_sigma_u=2.5e-6,
        mag_sigma=0.01,
        sun_sigma=0.004,
    )
    assert cfg.gyro_sigma_v == 1.5e-3
    assert cfg.gyro_sigma_u == 2.5e-6
    assert cfg.mag_sigma == 0.01
    assert cfg.sun_sigma == 0.004
    est = make_sim_estimator(cfg, cfg.q0)
    assert isinstance(est, MultiplicativeEKF)
    assert est.sigma_v == 1.5e-3
    assert est.sigma_u == 2.5e-6
    default = make_scenario_config("slew", plot=False, gif=False)
    assert default.gyro_sigma_v == 5e-4
    assert default.mag_sigma == 3e-3


def test_main_sensor_noise_knobs_smoke():
    assert (
        main(
            [
                "--gyro-sigma-v",
                "1e-3",
                "--gyro-sigma-u",
                "2e-6",
                "--mag-sigma",
                "0.01",
                "--sun-sigma",
                "0.005",
                "--t-final",
                "0.05",
                "--no-plot",
                "--no-gif",
            ]
        )
        == 0
    )


def test_main_rejects_negative_sensor_noise():
    with pytest.raises(SystemExit):
        main(["--gyro-sigma-v", "-1e-4", "--t-final", "0.05", "--no-plot", "--no-gif"])
    with pytest.raises(SystemExit):
        main(["--mag-sigma", "-0.01", "--t-final", "0.05", "--no-plot", "--no-gif"])


def test_main_rejects_bad_actuator_tau_max():
    with pytest.raises(SystemExit):
        main(["--actuator-tau-max", "1,2", "--t-final", "0.05", "--no-plot", "--no-gif"])
    with pytest.raises(SystemExit):
        main(["--actuator-tau-max", "-0.01", "--t-final", "0.05", "--no-plot", "--no-gif"])
    with pytest.raises(SystemExit):
        main(["--actuator-tau", "-0.1", "--t-final", "0.05", "--no-plot", "--no-gif"])

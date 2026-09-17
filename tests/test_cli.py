"""CLI wiring for SimLab scenarios (no plant/controller rewrites)."""

import numpy as np
import pytest

from attitude_sim.quaternions import geodesic_angle
from attitude_sim.sim import build_parser, main, make_scenario_config


def test_parser_defaults_to_slew():
    args = build_parser().parse_args([])
    assert args.scenario == "slew"
    assert args.controller == "pid"
    assert args.estimator == "mekf"
    assert args.actuator_tau_max is None
    assert args.actuator_tau is None
    assert args.coarse_init is False
    assert args.env_gg is False
    assert args.env_dipole is None
    assert args.dipole_model == "tilted"
    assert args.orbit_radius == 7.0e6
    assert args.orbit_inc_deg == 0.0
    assert args.orbit_raan_deg == 0.0
    assert args.orbit_u0_deg == 0.0


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


def test_parser_env_disturbance_flags():
    args = build_parser().parse_args(
        [
            "--env-gg",
            "--env-dipole",
            "0.08,-0.01,0.02",
            "--dipole-model",
            "orbit_normal",
            "--orbit-radius",
            "6.8e6",
            "--orbit-inc-deg",
            "51.6",
            "--orbit-raan-deg",
            "10",
            "--orbit-u0-deg",
            "30",
        ]
    )
    assert args.env_gg is True
    assert args.env_dipole == "0.08,-0.01,0.02"
    assert args.dipole_model == "orbit_normal"
    assert args.orbit_radius == 6.8e6
    assert args.orbit_inc_deg == 51.6
    assert args.orbit_raan_deg == 10.0
    assert args.orbit_u0_deg == 30.0


def test_make_scenario_config_stems():
    slew = make_scenario_config("slew", plot=False, gif=False)
    det = make_scenario_config("detumble", plot=False, gif=False)
    assert slew.artifact_stem == "slew"
    assert det.artifact_stem == "detumble"
    assert abs(det.omega0).max() > 0.2
    assert slew.coarse_init is False
    assert slew.env_gg is False
    assert slew.env_dipole is None
    coarse = make_scenario_config("slew", plot=False, gif=False, coarse_init=True)
    assert coarse.coarse_init is True
    env = make_scenario_config(
        "slew",
        plot=False,
        gif=False,
        env_gg=True,
        env_dipole=np.array([0.1, 0.0, -0.02]),
        dipole_model="orbit_normal",
        orbit_inclination_deg=51.6,
    )
    assert env.env_gg is True
    np.testing.assert_allclose(env.env_dipole, [0.1, 0.0, -0.02])
    assert env.dipole_model == "orbit_normal"
    assert env.orbit_inclination_deg == 51.6


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


def test_main_env_gg_and_dipole_smoke():
    assert (
        main(
            [
                "--controller",
                "pid",
                "--estimator",
                "truth",
                "--angle-deg",
                "0",
                "--env-gg",
                "--env-dipole",
                "0.08,-0.01,0.02",
                "--dipole-model",
                "orbit_normal",
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
                "--env-gg",
                "--tau-dist",
                "0.001,0,0",
                "--t-final",
                "0.05",
                "--no-plot",
                "--no-gif",
            ]
        )
        == 0
    )


def test_main_rejects_bad_tau_dist():
    with pytest.raises(SystemExit):
        main(["--tau-dist", "1,2", "--no-plot", "--no-gif"])
    with pytest.raises(SystemExit):
        main(["--tau-dist", "a,b,c", "--no-plot", "--no-gif"])


def test_main_rejects_bad_env_dipole_and_orbit():
    with pytest.raises(SystemExit):
        main(["--env-dipole", "1,2", "--no-plot", "--no-gif"])
    with pytest.raises(SystemExit):
        main(["--env-dipole", "a,b,c", "--no-plot", "--no-gif"])
    with pytest.raises(SystemExit):
        main(["--orbit-radius", "0", "--env-gg", "--no-plot", "--no-gif"])
    with pytest.raises(SystemExit):
        main(["--dipole-model", "igrf", "--no-plot", "--no-gif"])


def test_main_rejects_non_positive_dt():
    with pytest.raises(SystemExit):
        main(["--dt", "0", "--t-final", "0.05", "--no-plot", "--no-gif"])


def test_main_rejects_bad_actuator_tau_max():
    with pytest.raises(SystemExit):
        main(["--actuator-tau-max", "1,2", "--t-final", "0.05", "--no-plot", "--no-gif"])
    with pytest.raises(SystemExit):
        main(["--actuator-tau-max", "-0.01", "--t-final", "0.05", "--no-plot", "--no-gif"])
    with pytest.raises(SystemExit):
        main(["--actuator-tau", "-0.1", "--t-final", "0.05", "--no-plot", "--no-gif"])

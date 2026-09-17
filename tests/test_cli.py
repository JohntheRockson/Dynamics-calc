"""CLI wiring for SimLab scenarios (no plant/controller rewrites)."""

import numpy as np
import pytest

from attitude_sim.estimation import MultiplicativeEKF
from attitude_sim.quaternions import geodesic_angle
from attitude_sim.sim import (
    build_parser,
    main,
    make_scenario_config,
    make_sim_disturbances,
    make_sim_estimator,
)


def test_parser_defaults_to_slew():
    args = build_parser().parse_args([])
    assert args.scenario == "slew"
    assert args.controller is None  # resolved to pid by the slew preset
    assert args.estimator == "mekf"
    assert args.actuator_tau_max is None
    assert args.actuator_tau is None
    assert args.coarse_init is False
    assert args.angle_deg is None
    assert args.env is False
    assert args.no_env is False
    assert args.mrp_plot is False
    assert args.list_scenarios is False
    assert args.gyro_sigma_v is None
    assert args.gyro_sigma_u is None
    assert args.mag_sigma is None
    assert args.sun_sigma is None
    assert args.gravity_gradient is False
    assert args.residual_dipole is False
    assert args.aerodynamic is False
    assert args.srp is False
    assert args.dipole_model == "tilted"
    assert args.orbit_radius == 7.0e6
    assert args.orbit_inc_deg is None
    assert args.dipole_m is None
    assert args.panel_area == 0.4
    assert args.aero_cd == 2.2
    assert args.srp_cr == 1.0
    assert args.srp_eclipse == "off"


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


def test_parser_hold_eigenaxis_and_env_flags():
    hold = build_parser().parse_args(["--scenario", "hold", "--no-env", "--mrp-plot"])
    assert hold.scenario == "hold"
    assert hold.no_env
    assert hold.mrp_plot
    eigen = build_parser().parse_args(["--scenario", "eigenaxis", "--env", "--angle-deg", "12"])
    assert eigen.scenario == "eigenaxis"
    assert eigen.env
    assert eigen.angle_deg == 12.0


def test_make_scenario_config_stems():
    slew = make_scenario_config("slew", plot=False, gif=False)
    det = make_scenario_config("detumble", plot=False, gif=False)
    hold = make_scenario_config("hold", plot=False, gif=False)
    eigen = make_scenario_config("eigenaxis", plot=False, gif=False)
    assert slew.artifact_stem == "slew"
    assert det.artifact_stem == "detumble"
    assert hold.artifact_stem == "hold"
    assert eigen.artifact_stem == "eigenaxis"
    assert abs(det.omega0).max() > 0.2
    assert slew.coarse_init is False
    assert slew.controller == "pid"
    assert eigen.controller == "lqr"
    assert hold.gravity_gradient is True
    assert hold.residual_dipole is True
    assert hold.aerodynamic is False
    assert hold.srp is False
    assert make_sim_disturbances(hold) is not None
    assert eigen.gravity_gradient is False
    assert make_sim_disturbances(eigen) is None
    coarse = make_scenario_config("slew", plot=False, gif=False, coarse_init=True)
    assert coarse.coarse_init is True
    env = make_scenario_config(
        "slew",
        plot=False,
        gif=False,
        gravity_gradient=True,
        residual_dipole=True,
        aerodynamic=True,
        srp=True,
        orbit_radius=6.8e6,
        orbit_inclination_deg=51.6,
        dipole_m=np.array([0.2, 0.0, 0.0]),
        dipole_model="orbit_normal",
        panel_area=1.2,
        panel_r_cp=np.array([0.08, 0.0, 0.03]),
        aero_cd=2.0,
        srp_cr=1.5,
        srp_eclipse="cylindrical",
    )
    assert env.gravity_gradient is True
    assert env.residual_dipole is True
    assert env.aerodynamic is True
    assert env.srp is True
    assert env.orbit_radius == 6.8e6
    assert env.orbit_inclination_deg == 51.6
    assert env.dipole_model == "orbit_normal"
    np.testing.assert_allclose(env.dipole_m, [0.2, 0.0, 0.0])
    assert env.panel_area == 1.2
    np.testing.assert_allclose(env.panel_r_cp, [0.08, 0.0, 0.03])
    assert env.aero_cd == 2.0
    assert env.srp_cr == 1.5
    assert env.srp_eclipse == "cylindrical"
    models = make_sim_disturbances(env)
    assert models is not None
    assert models.gravity_gradient is not None
    assert models.residual_dipole is not None
    assert models.aerodynamic is not None
    assert models.srp is not None


def test_make_scenario_config_unknown():
    with pytest.raises(ValueError, match="unknown scenario"):
        make_scenario_config("spinup", plot=False, gif=False)


def test_angle_deg_only_affects_slew_and_eigenaxis():
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
    assert main(["--scenario", "hold", "--t-final", "0.05", "--no-plot", "--no-gif"]) == 0
    assert main(["--scenario", "eigenaxis", "--t-final", "0.05", "--no-plot", "--no-gif"]) == 0


def test_main_list_scenarios():
    assert main(["--list-scenarios"]) == 0


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


def test_main_env_and_mrp_cli_smoke(tmp_path):
    assert (
        main(
            [
                "--scenario",
                "slew",
                "--env",
                "--mrp-plot",
                "--t-final",
                "0.08",
                "--no-gif",
                "--out-dir",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert (tmp_path / "slew_summary.png").is_file()
    assert (tmp_path / "slew_mrp.png").is_file()
    assert (tmp_path / "slew_env_torque.png").is_file()
    assert (
        main(
            [
                "--scenario",
                "hold",
                "--no-env",
                "--estimator",
                "truth",
                "--t-final",
                "0.08",
                "--no-gif",
                "--out-dir",
                str(tmp_path / "hold_off"),
            ]
        )
        == 0
    )
    assert (tmp_path / "hold_off" / "hold_summary.png").is_file()
    assert not (tmp_path / "hold_off" / "hold_env_torque.png").exists()


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


def test_main_env_disturbance_flags_smoke():
    assert (
        main(
            [
                "--controller",
                "pid",
                "--estimator",
                "truth",
                "--gravity-gradient",
                "--residual-dipole",
                "--orbit-radius",
                "6.8e6",
                "--orbit-inc-deg",
                "51.6",
                "--dipole-m",
                "0.2,0.05,-0.01",
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


def test_main_aero_srp_flags_smoke():
    assert (
        main(
            [
                "--controller",
                "pid",
                "--estimator",
                "truth",
                "--aerodynamic",
                "--srp",
                "--panel-area",
                "0.8",
                "--panel-rcp",
                "0.04,0,0.02",
                "--aero-cd",
                "2.2",
                "--srp-cr",
                "1.3",
                "--srp-eclipse",
                "cylindrical",
                "--orbit-radius",
                "6.778e6",
                "--t-final",
                "0.05",
                "--no-plot",
                "--no-gif",
            ]
        )
        == 0
    )


def test_main_env_plus_actuator_sat_smoke():
    assert (
        main(
            [
                "--controller",
                "pid",
                "--estimator",
                "truth",
                "--gravity-gradient",
                "--residual-dipole",
                "--actuator-tau-max",
                "0.008",
                "--t-final",
                "0.05",
                "--no-plot",
                "--no-gif",
            ]
        )
        == 0
    )


def test_main_rejects_bad_env_flags():
    with pytest.raises(SystemExit):
        main(["--dipole-m", "1,2", "--residual-dipole", "--t-final", "0.05", "--no-plot", "--no-gif"])
    with pytest.raises(SystemExit):
        main(["--orbit-radius", "0", "--gravity-gradient", "--t-final", "0.05", "--no-plot", "--no-gif"])
    with pytest.raises(SystemExit):
        main(["--dipole-model", "igrf", "--t-final", "0.05", "--no-plot", "--no-gif"])
    with pytest.raises(SystemExit):
        main(["--panel-rcp", "0.1,0.2", "--aerodynamic", "--t-final", "0.05", "--no-plot", "--no-gif"])
    with pytest.raises(SystemExit):
        main(["--panel-area", "-1", "--aerodynamic", "--t-final", "0.05", "--no-plot", "--no-gif"])
    with pytest.raises(SystemExit):
        main(["--aero-cd", "-0.1", "--aerodynamic", "--t-final", "0.05", "--no-plot", "--no-gif"])
    with pytest.raises(SystemExit):
        main(["--srp-cr", "-1", "--srp", "--t-final", "0.05", "--no-plot", "--no-gif"])
    with pytest.raises(SystemExit):
        main(["--srp-eclipse", "penumbra", "--srp", "--t-final", "0.05", "--no-plot", "--no-gif"])


def test_main_rejects_bad_actuator_tau_max():
    with pytest.raises(SystemExit):
        main(["--actuator-tau-max", "1,2", "--t-final", "0.05", "--no-plot", "--no-gif"])
    with pytest.raises(SystemExit):
        main(["--actuator-tau-max", "-0.01", "--t-final", "0.05", "--no-plot", "--no-gif"])
    with pytest.raises(SystemExit):
        main(["--actuator-tau", "-0.1", "--t-final", "0.05", "--no-plot", "--no-gif"])

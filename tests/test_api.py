"""Public package API and packaging smoke."""

import attitude_sim


def test_version_and_public_exports():
    assert attitude_sim.__version__ == "0.1.0"
    for name in (
        "MahonyFilter",
        "ComplementaryFilter",
        "run_sim",
        "run_slew",
        "SimConfig",
        "SimLog",
        "make_controller",
        "AttitudeLQR",
        "design_attitude_lqr",
        "solve_care",
        "tune_pid_second_order",
        "bryson_lqr_costs",
        "cubesat_gain_report",
        "make_estimator",
        "geodesic_angle",
        "axis_angle_to_quat",
        "quat_to_rotation",
        "is_principal",
        "TorqueActuator",
        "clip_torque",
        "make_actuator",
        "shape_pid_command",
        "triad_attitude",
        "InnovationLog",
        "GravityGradientTorque",
        "ResidualDipoleTorque",
        "AerodynamicTorque",
        "SolarRadiationPressureTorque",
        "gravity_gradient_torque",
        "inertial_torque",
        "magnetic_dipole_torque",
        "trapezoid_inertial_impulse",
        "aerodynamic_torque",
        "srp_torque",
        "quat_to_mrp",
        "mrp_to_quat",
        "mrp_switch",
        "mrp_B",
        "mrp_derivative",
        "SCENARIOS",
        "make_scenario_config",
        "make_hold_environmental_torques",
        "plot_mrp_error",
        "plot_env_torque",
        "attitude_error_mrp",
        "scenario_catalog_text",
        "kinetic_energy",
        "energy_casimir",
        "sample_polhode",
        "sample_herpolhode",
        "principal_spin_stability",
        "SpinStability",
        "Gyrostat",
        "gyrostat_omega_dot",
        "step_gyrostat",
        "step_gyrostat_attitude",
        "HingedAppendage",
        "linearized_flex_state_space",
        "step_flex",
        "step_flex_attitude",
    ):
        assert hasattr(attitude_sim, name)
    assert attitude_sim.MahonyFilter is attitude_sim.ComplementaryFilter
    assert attitude_sim.AttitudeLQR is attitude_sim.LQRAttitudeController
    assert attitude_sim.run_sim is attitude_sim.run_slew
    assert "MahonyFilter" in attitude_sim.__all__
    assert "AttitudeLQR" in attitude_sim.__all__
    assert "run_sim" in attitude_sim.__all__
    assert "TorqueActuator" in attitude_sim.__all__
    assert "triad_attitude" in attitude_sim.__all__
    assert "InnovationLog" in attitude_sim.__all__
    assert "shape_pid_command" in attitude_sim.__all__
    assert "tune_pid_second_order" in attitude_sim.__all__
    assert "bryson_lqr_costs" in attitude_sim.__all__
    assert "cubesat_gain_report" in attitude_sim.__all__
    assert "inertial_torque" in attitude_sim.__all__
    assert "trapezoid_inertial_impulse" in attitude_sim.__all__
    assert "SCENARIOS" in attitude_sim.__all__
    assert "make_scenario_config" in attitude_sim.__all__
    assert "hold" in attitude_sim.SCENARIOS
    assert "eigenaxis" in attitude_sim.SCENARIOS
    assert "sample_polhode" in attitude_sim.__all__
    assert "SpinStability" in attitude_sim.__all__
    assert "Gyrostat" in attitude_sim.__all__
    assert "step_gyrostat" in attitude_sim.__all__
    assert "HingedAppendage" in attitude_sim.__all__
    assert "step_flex" in attitude_sim.__all__

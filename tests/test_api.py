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
        "make_estimator",
        "geodesic_angle",
        "axis_angle_to_quat",
        "quat_to_rotation",
        "is_principal",
        "TorqueActuator",
        "clip_torque",
        "make_actuator",
        "triad_attitude",
        "GravityGradientTorque",
        "ResidualDipoleTorque",
        "gravity_gradient_torque",
        "magnetic_dipole_torque",
        "quat_to_mrp",
        "mrp_to_quat",
        "mrp_switch",
        "mrp_B",
        "mrp_derivative",
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

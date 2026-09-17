"""Rigid-body attitude dynamics, control, and estimation (Milestone 1)."""

from attitude_sim.actuators import TorqueActuator, clip_torque, make_actuator
from attitude_sim.controls import (
    AttitudeLQR,
    LQRAttitudeController,
    PIDAttitudeController,
    design_attitude_lqr,
    make_controller,
    shape_pid_command,
    solve_care,
)
from attitude_sim.disturbances import (
    CircularOrbit,
    EnvironmentalTorques,
    GravityGradientTorque,
    OrbitState,
    ResidualDipoleTorque,
    gravity_gradient_torque,
    magnetic_dipole_torque,
    magnetic_field_body,
)
from attitude_sim.estimation import (
    ComplementaryFilter,
    MultiplicativeEKF,
    make_estimator,
    triad_attitude,
)
from attitude_sim.plant import (
    RigidBody,
    inertia_from_principal,
    is_principal,
    principal_moments_and_axes,
    rk4_step,
    rkmk4_step,
    step_rigid_body,
    validate_inertia,
)
from attitude_sim.quaternions import (
    axis_angle_to_quat,
    geodesic_angle,
    quat_conjugate,
    quat_error,
    quat_multiply,
    quat_normalize,
    quat_to_euler321,
    quat_to_rotation,
)
from attitude_sim.sim import SimConfig, SimLog, run_sim, run_slew

# Mahony is the complementary-filter implementation; keep both names public.
MahonyFilter = ComplementaryFilter

__all__ = [
    "AttitudeLQR",
    "CircularOrbit",
    "ComplementaryFilter",
    "EnvironmentalTorques",
    "GravityGradientTorque",
    "LQRAttitudeController",
    "MahonyFilter",
    "MultiplicativeEKF",
    "OrbitState",
    "PIDAttitudeController",
    "ResidualDipoleTorque",
    "RigidBody",
    "SimConfig",
    "SimLog",
    "TorqueActuator",
    "axis_angle_to_quat",
    "clip_torque",
    "design_attitude_lqr",
    "geodesic_angle",
    "gravity_gradient_torque",
    "inertia_from_principal",
    "is_principal",
    "magnetic_dipole_torque",
    "magnetic_field_body",
    "make_actuator",
    "make_controller",
    "make_estimator",
    "principal_moments_and_axes",
    "quat_conjugate",
    "quat_error",
    "quat_multiply",
    "quat_normalize",
    "quat_to_euler321",
    "quat_to_rotation",
    "rk4_step",
    "rkmk4_step",
    "run_sim",
    "run_slew",
    "shape_pid_command",
    "solve_care",
    "step_rigid_body",
    "triad_attitude",
    "validate_inertia",
]

__version__ = "0.1.0"

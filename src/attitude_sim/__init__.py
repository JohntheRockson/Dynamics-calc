"""Rigid-body attitude dynamics, control, and estimation (Milestone 1)."""

from attitude_sim.actuators import TorqueActuator, clip_torque, make_actuator
from attitude_sim.controls import LQRAttitudeController, PIDAttitudeController, make_controller
from attitude_sim.estimation import ComplementaryFilter, MultiplicativeEKF, make_estimator
from attitude_sim.plant import (
    RigidBody,
    inertia_from_principal,
    is_principal,
    principal_moments_and_axes,
    rk4_step,
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
    "ComplementaryFilter",
    "LQRAttitudeController",
    "MahonyFilter",
    "MultiplicativeEKF",
    "PIDAttitudeController",
    "RigidBody",
    "SimConfig",
    "SimLog",
    "TorqueActuator",
    "axis_angle_to_quat",
    "clip_torque",
    "geodesic_angle",
    "inertia_from_principal",
    "is_principal",
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
    "run_sim",
    "run_slew",
    "step_rigid_body",
    "validate_inertia",
]

__version__ = "0.1.0"

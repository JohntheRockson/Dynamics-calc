"""Rigid-body attitude dynamics, control, and estimation (Milestone 1)."""

from attitude_sim.controls import LQRAttitudeController, PIDAttitudeController
from attitude_sim.estimation import ComplementaryFilter, MultiplicativeEKF
from attitude_sim.plant import (
    RigidBody,
    inertia_from_principal,
    principal_moments_and_axes,
    rk4_step,
    step_rigid_body,
    validate_inertia,
)
from attitude_sim.quaternions import (
    quat_conjugate,
    quat_error,
    quat_multiply,
    quat_normalize,
    quat_to_euler321,
)

__all__ = [
    "ComplementaryFilter",
    "LQRAttitudeController",
    "MultiplicativeEKF",
    "PIDAttitudeController",
    "RigidBody",
    "inertia_from_principal",
    "principal_moments_and_axes",
    "quat_conjugate",
    "quat_error",
    "quat_multiply",
    "quat_normalize",
    "quat_to_euler321",
    "rk4_step",
    "step_rigid_body",
    "validate_inertia",
]

__version__ = "0.1.0"

"""Controller construction and LQR closed-loop stability of the linear model."""

import numpy as np

from attitude_sim.controls import LQRAttitudeController, linearize_attitude, make_controller
from attitude_sim.quaternions import axis_angle_to_quat


def test_make_controller_modes():
    J = np.diag([0.05, 0.06, 0.07])
    pid = make_controller("pid", J, torque_limit=0.02)
    lqr = make_controller("lqr", J, torque_limit=0.02)
    q = np.array([1.0, 0.0, 0.0, 0.0])
    q_des = axis_angle_to_quat(np.array([0.0, 0.0, 1.0]), 0.4)
    omega = np.zeros(3)
    tau_pid = pid.command(q, omega, q_des, dt=0.01)
    tau_lqr = lqr.command(q, omega, q_des, dt=0.01)
    assert tau_pid.shape == (3,)
    assert tau_lqr.shape == (3,)
    # Rest-to-rest about +z should command a +z torque from identity.
    assert tau_pid[2] > 0.0
    assert tau_lqr[2] > 0.0


def test_lqr_linear_closed_loop_hurwitz():
    J = np.diag([0.05, 0.06, 0.07])
    ctrl = LQRAttitudeController(J)
    A, B = linearize_attitude(J)
    eigs = np.linalg.eigvals(A - B @ ctrl.K)
    assert np.all(np.real(eigs) < -1e-6)

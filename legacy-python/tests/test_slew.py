"""Rest-to-rest eigenaxis slew profile: continuity, limits, and closed-loop smoke."""

import numpy as np
import pytest

from attitude_sim.controls import (
    DEFAULT_TORQUE_LIMIT,
    LQRAttitudeController,
    PIDAttitudeController,
    apply_torque_limits,
)
from attitude_sim.plant import RigidBody, step_rigid_body
from attitude_sim.quaternions import axis_angle_to_quat, geodesic_angle
from attitude_sim.slew import (
    EigenaxisSlewProfile,
    alpha_max_from_torque,
    command_slew,
    feedforward_accel_torque,
    geodesic_along_profile,
    rest_to_rest_eigenaxis,
)


def _inertia() -> np.ndarray:
    return np.diag([0.05, 0.06, 0.07])


def _axis() -> np.ndarray:
    return np.array([0.2, 0.5, 0.84])


def _profile_trap(**kwargs) -> EigenaxisSlewProfile:
    base = dict(
        axis=_axis(),
        theta0=0.0,
        theta_f=np.deg2rad(45.0),
        omega_max=0.12,
        alpha_max=0.25,
    )
    base.update(kwargs)
    return rest_to_rest_eigenaxis(**base)


def test_trapezoid_rest_to_rest_boundary_and_coast():
    prof = _profile_trap()
    assert not prof.triangular
    assert prof.t_coast > 0.0
    assert prof.t_final == pytest.approx(2.0 * prof.t_acc + prof.t_coast)
    s0 = prof.sample(0.0)
    sf = prof.sample(prof.t_final)
    assert s0.theta == pytest.approx(prof.theta0)
    assert s0.omega == pytest.approx(0.0)
    assert abs(s0.alpha) == pytest.approx(prof.alpha_max)
    assert sf.theta == pytest.approx(prof.theta_f)
    assert sf.omega == pytest.approx(0.0, abs=1e-15)
    assert sf.alpha == pytest.approx(0.0)
    assert sf.done
    mid = prof.sample(prof.t_acc + 0.5 * prof.t_coast)
    assert mid.alpha == pytest.approx(0.0)
    assert abs(mid.omega) == pytest.approx(prof.omega_max)
    np.testing.assert_allclose(np.linalg.norm(s0.q_des), 1.0)
    np.testing.assert_allclose(np.linalg.norm(sf.q_des), 1.0)
    np.testing.assert_allclose(s0.q_des, prof.q0, atol=1e-15)
    assert geodesic_angle(sf.q_des, prof.qf) < 1e-12


def test_bang_bang_triangle_when_omega_max_omitted_or_high():
    trap = _profile_trap()
    bang = rest_to_rest_eigenaxis(
        axis=_axis(),
        theta_f=trap.theta_f,
        alpha_max=trap.alpha_max,
    )
    assert bang.triangular
    assert bang.t_coast == pytest.approx(0.0)
    assert bang.t_final < trap.t_final
    peak = abs(bang.sample(bang.t_acc).omega)
    assert peak == pytest.approx(np.sqrt(trap.alpha_max * trap.theta_f), rel=1e-12)
    too_fast = rest_to_rest_eigenaxis(
        axis=_axis(),
        theta_f=trap.theta_f,
        alpha_max=trap.alpha_max,
        omega_max=10.0,
    )
    assert too_fast.triangular
    assert too_fast.t_final == pytest.approx(bang.t_final)


def test_profile_theta_omega_continuous_alpha_may_jump():
    prof = _profile_trap()
    dt = 1e-4
    t = np.arange(0.0, prof.t_final + dt, dt)
    traj = prof.evaluate(t)
    dtheta = np.diff(traj["theta"])
    domega = np.diff(traj["omega"])
    # θ is C¹: |Δθ| ≤ (|ω| + ½|α|dt) dt
    assert np.max(np.abs(dtheta)) < (prof.omega_max + prof.alpha_max * dt) * dt * 1.01
    # ω is C⁰: |Δω| ≤ α_max dt (bang edges included)
    assert np.max(np.abs(domega)) <= prof.alpha_max * dt * (1.0 + 1e-9)
    # Interior kinematics: θ̇ ≈ ω, ω̇ ≈ α away from the three switches.
    switches = np.array([prof.t_acc, prof.t_acc + prof.t_coast, prof.t_final])
    interior = np.ones(t.size - 1, dtype=bool)
    for ts in switches:
        interior &= np.abs(t[:-1] - ts) > 5 * dt
    theta_dot = dtheta[interior] / dt
    omega_mid = 0.5 * (traj["omega"][:-1] + traj["omega"][1:])[interior]
    np.testing.assert_allclose(theta_dot, omega_mid, rtol=0.0, atol=5e-4)
    omega_dot = domega[interior] / dt
    alpha_mid = 0.5 * (traj["alpha"][:-1] + traj["alpha"][1:])[interior]
    np.testing.assert_allclose(omega_dot, alpha_mid, rtol=0.0, atol=5e-3)


def test_geodesic_matches_eigenangle_and_call_alias():
    prof = _profile_trap()
    for t in (0.0, 0.3 * prof.t_final, prof.t_acc, prof.t_final):
        s = prof(t)
        geo = geodesic_along_profile(s, prof.q0)
        np.testing.assert_allclose(geo, abs(s.theta - prof.theta0), atol=1e-12)
        np.testing.assert_allclose(s.omega_des, s.omega * s.axis)
        np.testing.assert_allclose(s.alpha_des, s.alpha * s.axis)


def test_quaternion_pair_matches_scalar_axis_angle():
    axis = _axis()
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    qf = axis_angle_to_quat(axis, np.deg2rad(30.0))
    from_q = rest_to_rest_eigenaxis(q0, qf, alpha_max=0.4, omega_max=0.1)
    from_th = rest_to_rest_eigenaxis(axis=axis, theta_f=np.deg2rad(30.0), alpha_max=0.4, omega_max=0.1)
    assert from_q.t_final == pytest.approx(from_th.t_final)
    np.testing.assert_allclose(from_q.axis, from_th.axis, atol=1e-12)
    assert geodesic_angle(from_q.qf, qf) < 1e-12
    assert geodesic_angle(from_q.sample(from_q.t_final).q_des, qf) < 1e-12
    # Double cover of the target is the same attitude.
    from_neg = rest_to_rest_eigenaxis(q0, -qf, alpha_max=0.4, omega_max=0.1)
    assert from_neg.theta_f == pytest.approx(from_q.theta_f)
    assert geodesic_angle(from_neg.qf, qf) < 1e-12


def test_negative_stroke_and_zero_slew():
    prof = rest_to_rest_eigenaxis(axis=_axis(), theta0=0.4, theta_f=-0.2, alpha_max=0.5, omega_max=0.2)
    assert prof.sign < 0.0
    assert prof.sample(0.0).theta == pytest.approx(0.4)
    assert prof.sample(prof.t_final).theta == pytest.approx(-0.2)
    assert prof.sample(prof.t_acc + 0.01).omega < 0.0
    hold = rest_to_rest_eigenaxis(axis=_axis(), theta_f=0.0, alpha_max=0.5)
    assert hold.t_final == pytest.approx(0.0)
    s = hold.sample(1.0)
    assert s.done
    assert s.omega == pytest.approx(0.0)
    q = np.array([1.0, 0.0, 0.0, 0.0])
    same = rest_to_rest_eigenaxis(q, q, alpha_max=0.3)
    assert same.t_final == pytest.approx(0.0)


def test_alpha_from_inertia_and_tau_max():
    J = _inertia()
    axis = _axis()
    a = axis / np.linalg.norm(axis)
    tau = 0.02
    alpha = alpha_max_from_torque(J, axis, tau)
    np.testing.assert_allclose(np.linalg.norm(J @ (alpha * a)), tau)
    boxed = alpha_max_from_torque(J, axis, np.array([0.02, 0.015, 0.01]))
    tau_ff = J @ (boxed * a)
    assert np.all(np.abs(tau_ff) <= np.array([0.02, 0.015, 0.01]) * (1.0 + 1e-12))
    prof = rest_to_rest_eigenaxis(
        axis=axis,
        theta_f=np.deg2rad(40.0),
        omega_max=0.1,
        inertia=J,
        tau_max=DEFAULT_TORQUE_LIMIT,
    )
    assert prof.alpha_max == pytest.approx(alpha)
    ff = feedforward_accel_torque(J, prof.sample(0.01).alpha_des)
    assert np.linalg.norm(ff) <= DEFAULT_TORQUE_LIMIT * (1.0 + 1e-9)


def test_rest_to_rest_rejects_bad_limits():
    with pytest.raises(ValueError, match="alpha_max"):
        rest_to_rest_eigenaxis(axis=_axis(), theta_f=0.4)
    with pytest.raises(ValueError, match="positive"):
        rest_to_rest_eigenaxis(axis=_axis(), theta_f=0.4, alpha_max=-0.1)
    with pytest.raises(ValueError, match="omega_max"):
        rest_to_rest_eigenaxis(axis=_axis(), theta_f=0.4, alpha_max=0.2, omega_max=0.0)
    with pytest.raises(ValueError, match="theta_f"):
        rest_to_rest_eigenaxis(axis=_axis(), alpha_max=0.2)
    with pytest.raises(ValueError, match="non-zero"):
        rest_to_rest_eigenaxis(axis=np.zeros(3), theta_f=0.2, alpha_max=0.2)
    with pytest.raises(ValueError, match="tau_max"):
        alpha_max_from_torque(_inertia(), _axis(), 0.0)


def test_pid_alpha_des_is_euler_feedforward_on_the_profile():
    J = _inertia()
    prof = _profile_trap()
    pid = PIDAttitudeController(J, torque_limit=None, ki=0.0)
    pid.reset()
    s = prof.sample(0.05)
    tau = pid.command(
        s.q_des,
        s.omega_des,
        s.q_des,
        omega_des=s.omega_des,
        dt=0.01,
        alpha_des=s.alpha_des,
    )
    expected = J @ s.alpha_des + np.cross(s.omega_des, J @ s.omega_des)
    np.testing.assert_allclose(tau, expected, atol=1e-12)
    pid.reset()
    tau_no_ff = pid.command(
        s.q_des, s.omega_des, s.q_des, omega_des=s.omega_des, dt=0.01
    )
    np.testing.assert_allclose(tau_no_ff, np.cross(s.omega_des, J @ s.omega_des), atol=1e-12)


def test_command_slew_pid_and_lqr_alpha_des_before_existing_limiter():
    J = _inertia()
    prof = _profile_trap()
    s = prof.sample(0.04)
    pid = PIDAttitudeController(J, torque_limit=None, ki=0.0)
    pid.reset()
    tau_pid = command_slew(pid, s.q_des, s.omega_des, s, dt=0.01, feedforward=True)
    expected = J @ s.alpha_des + np.cross(s.omega_des, J @ s.omega_des)
    np.testing.assert_allclose(tau_pid, expected, atol=1e-12)
    lqr = LQRAttitudeController(J, torque_limit=None)
    tau_lqr = command_slew(lqr, s.q_des, s.omega_des, s, dt=0.01, feedforward=True)
    np.testing.assert_allclose(tau_lqr, expected, atol=1e-10)
    tau_lqr_rate = command_slew(lqr, s.q_des, s.omega_des, s, dt=0.01, feedforward=False)
    np.testing.assert_allclose(tau_lqr_rate, np.cross(s.omega_des, J @ s.omega_des), atol=1e-10)
    # Tight Euclidean ball: FF is clamped by apply_torque_limits, not a new saturator.
    tight = LQRAttitudeController(J, torque_limit=1e-4, tau_max=None, gyroscopic_cancel=False)
    tau_tight = tight.command(
        s.q_des, s.omega_des, s.q_des, omega_des=s.omega_des, dt=0.01, alpha_des=s.alpha_des
    )
    np.testing.assert_allclose(tau_tight, apply_torque_limits(J @ s.alpha_des, 1e-4, None))


def _profiled_slew(ctrl, prof, t_final, dt=0.01, feedforward=True):
    body = RigidBody(_inertia())
    q = prof.q0.copy()
    omega = np.zeros(3)
    ctrl.reset()
    errors = []
    track = []
    t = 0.0
    n = int(np.round(t_final / dt))
    for _ in range(n):
        s = prof.sample(t)
        tau = command_slew(ctrl, q, omega, s, dt, feedforward=feedforward)
        q, omega = step_rigid_body(body, q, omega, tau, dt)
        t += dt
        errors.append(geodesic_angle(q, prof.qf))
        track.append(geodesic_angle(q, s.q_des))
    return q, omega, np.asarray(errors), np.asarray(track)


def test_profiled_pid_slew_settles():
    J = _inertia()
    prof = rest_to_rest_eigenaxis(
        axis=_axis(),
        theta_f=np.deg2rad(40.0),
        omega_max=0.12,
        inertia=J,
        tau_max=DEFAULT_TORQUE_LIMIT,
    )
    pid = PIDAttitudeController(J)
    hold = 6.0
    _q, omega, err, track = _profiled_slew(pid, prof, t_final=prof.t_final + hold)
    assert np.rad2deg(err[0]) == pytest.approx(40.0, abs=0.05)
    assert np.rad2deg(err[-1]) < 0.5
    assert np.linalg.norm(omega) < 0.02
    assert err[-1] < 0.05 * err[0]
    # Tracking during the manoeuvre stays well inside the opening geodesic.
    during = track[: int(np.round(prof.t_final / 0.01))]
    assert np.rad2deg(np.max(during)) < 8.0


def test_profiled_lqr_slew_settles_on_omega_des():
    """LQR tracks the profile via existing ω_des / α_des and apply_torque_limits."""
    J = _inertia()
    prof = rest_to_rest_eigenaxis(
        axis=_axis(),
        theta_f=np.deg2rad(20.0),
        omega_max=0.10,
        inertia=J,
        tau_max=DEFAULT_TORQUE_LIMIT,
    )
    lqr = LQRAttitudeController(J)
    _q, omega, err, track = _profiled_slew(
        lqr, prof, t_final=prof.t_final + 8.0, feedforward=True
    )
    assert np.rad2deg(err[-1]) < 0.5
    assert np.linalg.norm(omega) < 0.02
    assert err[-1] < 0.1 * err[0]
    during = track[: int(np.round(prof.t_final / 0.01))]
    assert np.rad2deg(np.median(during)) < 5.0

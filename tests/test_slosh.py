"""Fuel-slosh pendulum: Euler reduction, √(g_eff/ℓ) frequency, damping."""

import numpy as np
import pytest

from attitude_sim.plant import RigidBody, step_rigid_body
from attitude_sim.quaternions import quat_normalize
from attitude_sim.slosh import (
    SloshPendulum,
    pack_slosh_state,
    pendulum_frame,
    slosh_natural_frequency,
    slosh_omega_dot,
    slosh_reaction_torque,
    step_slosh,
    step_slosh_attitude,
    unit_axis,
    unpack_slosh_state,
)

_ASYM_J = 0.5 * (
    np.array(
        [
            [2.0, 0.1, 0.0],
            [0.1, 3.0, 0.15],
            [0.0, 0.15, 4.0],
        ]
    )
    + np.array(
        [
            [2.0, 0.1, 0.0],
            [0.1, 3.0, 0.15],
            [0.0, 0.15, 4.0],
        ]
    ).T
)

_TRIAXIAL = np.diag([1.0, 2.0, 3.0])

EULER_MATCH_ATOL = 1e-14
FROZEN_MATCH_ATOL = 1e-12
ENERGY_REL_TOL = 2e-10
QUAT_NORM_ABS_TOL = 1e-12
FREQ_REL_TOL = 0.03


def _plant(**kwargs) -> SloshPendulum:
    defaults = dict(
        inertia=_TRIAXIAL,
        mass=0.08,
        length=0.4,
        g_eff=2.5,
        damping=0.0,
        stiffness=0.0,
    )
    defaults.update(kwargs)
    return SloshPendulum(**defaults)


def test_unit_axis_and_pendulum_frame():
    np.testing.assert_allclose(unit_axis("z"), [0.0, 0.0, 1.0])
    np.testing.assert_allclose(unit_axis(0), [1.0, 0.0, 0.0])
    np.testing.assert_allclose(unit_axis([0.0, 2.0, 0.0]), [0.0, 1.0, 0.0])
    u0, es, en = pendulum_frame("-z", "x")
    np.testing.assert_allclose(u0, [0.0, 0.0, -1.0])
    np.testing.assert_allclose(es, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(en, [0.0, -1.0, 0.0])
    with pytest.raises(ValueError, match="non-zero"):
        unit_axis([0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="name"):
        unit_axis("w")
    with pytest.raises(ValueError, match="index"):
        unit_axis(3)
    with pytest.raises(ValueError, match="parallel"):
        pendulum_frame("z", "z")


def test_validation_rejects_bad_mass_length_and_damping():
    with pytest.raises(ValueError, match="mass"):
        SloshPendulum(_TRIAXIAL, mass=-0.1)
    with pytest.raises(ValueError, match="length"):
        SloshPendulum(_TRIAXIAL, mass=0.1, length=0.0)
    with pytest.raises(ValueError, match="damping"):
        SloshPendulum(_TRIAXIAL, damping=-1.0)
    with pytest.raises(ValueError, match="g_eff"):
        SloshPendulum(_TRIAXIAL, g_eff=-1.0)


def test_pack_unpack_roundtrip_and_bad_state():
    w = np.array([0.1, -0.2, 0.3])
    y = pack_slosh_state(w, 0.4, -0.5)
    w2, th, td = unpack_slosh_state(y)
    np.testing.assert_allclose(w2, w)
    assert th == pytest.approx(0.4)
    assert td == pytest.approx(-0.5)
    with pytest.raises(ValueError, match="length 5"):
        unpack_slosh_state(np.zeros(4))


def test_m_zero_omega_dot_matches_euler():
    body = RigidBody(_ASYM_J)
    slosh = _plant(inertia=_ASYM_J, mass=0.0)
    omega = np.array([0.3, -0.8, 0.2])
    tau = np.array([0.04, -0.01, 0.02])
    np.testing.assert_allclose(
        slosh.omega_dot(omega, tau, 0.2, -0.3),
        body.omega_dot(omega, tau),
        atol=EULER_MATCH_ATOL,
    )
    np.testing.assert_allclose(
        slosh_omega_dot(slosh, omega, tau, 0.2, -0.3),
        body.omega_dot(omega, tau),
        atol=EULER_MATCH_ATOL,
    )
    np.testing.assert_allclose(
        slosh.slosh_reaction_torque(omega, tau, 0.2, -0.3),
        np.zeros(3),
        atol=EULER_MATCH_ATOL,
    )


def test_m_zero_rk4_matches_rigid_body():
    body = RigidBody(_ASYM_J)
    slosh = _plant(inertia=_ASYM_J, mass=0.0)
    q = quat_normalize([0.4, 0.3, 0.2, 0.8])
    omega = np.array([-0.5, 0.4, 0.7])
    tau = np.array([0.01, -0.02, 0.015])
    dt = 0.002
    q_r, w_r = q.copy(), omega.copy()
    q_s, w_s = q.copy(), omega.copy()
    theta, theta_dot = 0.15, -0.1
    for _ in range(250):
        q_r, w_r = step_rigid_body(body, q_r, w_r, tau, dt)
        q_s, w_s, theta, theta_dot = step_slosh_attitude(
            slosh, q_s, w_s, theta, theta_dot, tau, dt
        )
    np.testing.assert_allclose(w_s, w_r, atol=1e-13)
    np.testing.assert_allclose(q_s, q_r, atol=1e-13)
    assert theta == pytest.approx(0.15)
    assert theta_dot == pytest.approx(-0.1)


def test_frozen_rest_matches_locked_rigid_body():
    slosh = _plant(inertia=_ASYM_J, hinge=np.zeros(3))
    body = slosh.as_rigid_body(0.0, locked=True)
    omega = np.array([0.25, -0.4, 0.15])
    tau = np.array([0.02, 0.01, -0.03])
    np.testing.assert_allclose(
        slosh.omega_dot(omega, tau, 0.0, 0.0, frozen=True),
        body.omega_dot(omega, tau),
        atol=FROZEN_MATCH_ATOL,
    )
    np.testing.assert_allclose(
        slosh.locked_inertia(0.0),
        body.inertia,
        atol=1e-14,
    )


def test_frozen_rk4_matches_locked_rigid_body():
    slosh = _plant(inertia=_ASYM_J)
    body = slosh.as_rigid_body(0.0, locked=True)
    q = quat_normalize([0.7, 0.1, -0.2, 0.4])
    omega = np.array([0.2, -0.1, 0.3])
    tau = np.zeros(3)
    dt = 0.002
    q_r, w_r = q.copy(), omega.copy()
    q_s, w_s = q.copy(), omega.copy()
    theta, theta_dot = 0.0, 0.0
    for _ in range(400):
        q_r, w_r = step_rigid_body(body, q_r, w_r, tau, dt)
        q_s, w_s, theta, theta_dot = step_slosh_attitude(
            slosh, q_s, w_s, theta, theta_dot, tau, dt, frozen=True
        )
    np.testing.assert_allclose(w_s, w_r, atol=1e-12)
    np.testing.assert_allclose(q_s, q_r, atol=1e-12)
    assert theta == pytest.approx(0.0)
    assert theta_dot == pytest.approx(0.0)


def test_rest_equilibrium_stays_put():
    slosh = _plant()
    omega = np.zeros(3)
    tau = np.zeros(3)
    wdot, td, tddot = slosh.derivatives(omega, tau, 0.0, 0.0)
    np.testing.assert_allclose(wdot, 0.0, atol=1e-14)
    assert td == pytest.approx(0.0)
    assert tddot == pytest.approx(0.0)
    w, th, tdot = omega.copy(), 0.0, 0.0
    for _ in range(200):
        w, th, tdot = step_slosh(slosh, w, th, tdot, tau, 0.01)
    np.testing.assert_allclose(w, 0.0, atol=1e-12)
    assert abs(th) < 1e-12
    assert abs(tdot) < 1e-12


def test_fixed_hub_frequency_matches_sqrt_g_over_ell():
    g_eff = 3.6
    length = 0.9
    slosh = _plant(mass=0.05, length=length, g_eff=g_eff, damping=0.0)
    wn = slosh.natural_frequency()
    assert wn == pytest.approx(np.sqrt(g_eff / length))
    assert slosh_natural_frequency(g_eff, length) == pytest.approx(wn)

    theta, theta_dot = 0.08, 0.0
    omega = np.zeros(3)
    tau = np.zeros(3)
    dt = 0.002
    t_final = 2.0 * (2.0 * np.pi / wn)
    n = int(t_final / dt)
    times = [0.0]
    thetas = [theta]
    t = 0.0
    for _ in range(n):
        omega, theta, theta_dot = step_slosh(
            slosh, omega, theta, theta_dot, tau, dt, fixed_hub=True
        )
        t += dt
        times.append(t)
        thetas.append(theta)
        np.testing.assert_allclose(omega, 0.0, atol=1e-14)

    th = np.asarray(thetas)
    # Zero-crossing period of a lightly displaced undamped pendulum.
    sign = np.sign(th)
    crossings = np.where(np.diff(sign) < 0.0)[0]
    assert len(crossings) >= 2
    period = times[crossings[1]] - times[crossings[0]]
    freq = 2.0 * np.pi / period
    assert abs(freq - wn) / wn < FREQ_REL_TOL


def test_zero_g_spring_frequency_is_documented_equivalent():
    mass = 0.06
    length = 0.3
    k = 0.48
    slosh = _plant(mass=mass, length=length, g_eff=0.0, stiffness=k)
    wn = slosh.natural_frequency()
    assert wn == pytest.approx(np.sqrt(k / (mass * length * length)))
    theta, theta_dot = 0.05, 0.0
    omega = np.zeros(3)
    dt = 0.002
    t_final = 2.0 * (2.0 * np.pi / wn)
    n = int(t_final / dt)
    times = [0.0]
    thetas = [theta]
    t = 0.0
    for _ in range(n):
        omega, theta, theta_dot = step_slosh(
            slosh, omega, theta, theta_dot, np.zeros(3), dt, fixed_hub=True
        )
        t += dt
        times.append(t)
        thetas.append(theta)
    th = np.asarray(thetas)
    crossings = np.where(np.diff(np.sign(th)) < 0.0)[0]
    period = times[crossings[1]] - times[crossings[0]]
    freq = 2.0 * np.pi / period
    assert abs(freq - wn) / wn < FREQ_REL_TOL


def test_undamped_energy_conserved():
    slosh = _plant(inertia=_ASYM_J, damping=0.0)
    omega = np.array([0.15, -0.05, 0.08])
    theta, theta_dot = 0.25, 0.1
    e0 = slosh.mechanical_energy(omega, theta, theta_dot)
    tau = np.zeros(3)
    for _ in range(2500):
        omega, theta, theta_dot = step_slosh(slosh, omega, theta, theta_dot, tau, 0.002)
    e1 = slosh.mechanical_energy(omega, theta, theta_dot)
    assert abs(e1 - e0) / max(abs(e0), 1e-12) < ENERGY_REL_TOL


def test_damping_decays_energy():
    slosh = _plant(damping=0.04)
    omega = np.array([0.05, 0.0, 0.0])
    theta, theta_dot = 0.35, 0.0
    e0 = slosh.mechanical_energy(omega, theta, theta_dot)
    for _ in range(3000):
        omega, theta, theta_dot = step_slosh(
            slosh, omega, theta, theta_dot, np.zeros(3), 0.002
        )
    e1 = slosh.mechanical_energy(omega, theta, theta_dot)
    assert e1 < 0.6 * e0
    assert e1 < e0 - 1e-4


def test_applied_torque_breaks_energy_conservation():
    slosh = _plant(damping=0.0)
    omega = np.array([0.1, -0.05, 0.02])
    theta, theta_dot = 0.1, 0.0
    e0 = slosh.mechanical_energy(omega, theta, theta_dot)
    tau = np.array([0.05, -0.02, 0.03])
    for _ in range(400):
        omega, theta, theta_dot = step_slosh(slosh, omega, theta, theta_dot, tau, 0.002)
    e1 = slosh.mechanical_energy(omega, theta, theta_dot)
    assert abs(e1 - e0) / max(abs(e0), 1e-12) > 1e-3


def test_slosh_reaction_torque_is_consistent_with_euler_residual():
    slosh = _plant(inertia=_ASYM_J)
    omega = np.array([0.2, -0.15, 0.05])
    tau = np.array([0.01, 0.0, -0.02])
    theta, theta_dot = 0.2, -0.3
    wdot = slosh.omega_dot(omega, tau, theta, theta_dot)
    tau_s = slosh_reaction_torque(slosh, omega, tau, theta, theta_dot)
    residual = slosh.inertia @ wdot + np.cross(omega, slosh.inertia @ omega) - tau
    np.testing.assert_allclose(tau_s, residual, atol=1e-14)
    assert np.linalg.norm(tau_s) > 1e-6


def test_attitude_step_keeps_unit_quaternion():
    slosh = _plant()
    q = quat_normalize([0.2, 0.5, -0.3, 0.8])
    omega = np.array([0.3, -0.2, 0.1])
    theta, theta_dot = 0.12, 0.05
    for _ in range(500):
        q, omega, theta, theta_dot = step_slosh_attitude(
            slosh, q, omega, theta, theta_dot, np.zeros(3), 0.002
        )
    assert abs(np.linalg.norm(q) - 1.0) < QUAT_NORM_ABS_TOL
    h = slosh.angular_momentum_inertial(q, omega, theta, theta_dot)
    assert np.all(np.isfinite(h))


def test_as_rigid_body_dry_hub_matches_inertia():
    slosh = _plant(inertia=_ASYM_J)
    body = slosh.as_rigid_body()
    np.testing.assert_allclose(body.inertia, slosh.inertia)
    assert slosh.has_slosh
    np.testing.assert_allclose(slosh.g_vector, slosh.g_eff * slosh.rest_direction)
    np.testing.assert_allclose(slosh.plane_normal, [0.0, -1.0, 0.0], atol=1e-12)


def test_natural_frequency_helpers_reject_degenerate():
    with pytest.raises(ValueError, match="length"):
        slosh_natural_frequency(1.0, 0.0)
    with pytest.raises(ValueError, match="mass"):
        slosh_natural_frequency(0.0, 0.2, mass=0.0, stiffness=1.0)
    empty = _plant(mass=0.0, stiffness=0.0, length=0.0)
    # m=0, k=0: frequency uses g_eff/ℓ only when ℓ>0; length 0 is rejected.
    with pytest.raises(ValueError, match="length"):
        empty.natural_frequency()

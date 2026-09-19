"""Gyrostat plant: Euler reduction, torque-free integrals, rotor-axis spin."""

import numpy as np
import pytest

from attitude_sim.gyrostat import (
    Gyrostat,
    gyrostat_omega_dot,
    pack_rate_state,
    step_gyrostat,
    step_gyrostat_attitude,
    transverse_lambda_squared,
    unit_rotor_axis,
    unpack_rate_state,
)
from attitude_sim.plant import RigidBody, step_rigid_body
from attitude_sim.quaternions import quat_normalize

# Asymmetric J with products of inertia (same numbers as tests/test_plant.py).
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

# Classical RK4, not symplectic.  Bounds sit well above observed residuals.
ENERGY_REL_TOL = 1e-12
H_BODY_NORM_REL_TOL = 1e-12
H_INERTIAL_ATOL = 1e-12
QUAT_NORM_ABS_TOL = 1e-12
EULER_MATCH_ATOL = 1e-14
AXISYM_OMEGA_ATOL = 1e-12
SPIN_EQ_ATOL = 1e-12


def test_unit_rotor_axis_names_indices_and_vectors():
    np.testing.assert_allclose(unit_rotor_axis("z"), [0.0, 0.0, 1.0])
    np.testing.assert_allclose(unit_rotor_axis(0), [1.0, 0.0, 0.0])
    np.testing.assert_allclose(unit_rotor_axis([0.0, 2.0, 0.0]), [0.0, 1.0, 0.0])
    with pytest.raises(ValueError, match="non-zero"):
        unit_rotor_axis([0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="name"):
        unit_rotor_axis("w")
    with pytest.raises(ValueError, match="index"):
        unit_rotor_axis(3)


def test_hw_zero_omega_dot_matches_euler():
    body = RigidBody(_ASYM_J)
    gyro = Gyrostat(_ASYM_J, rotor_axis="z", wheel_momentum=0.0)
    omega = np.array([0.3, -0.8, 0.2])
    tau = np.array([0.04, -0.01, 0.02])
    np.testing.assert_allclose(
        gyro.omega_dot(omega, tau),
        body.omega_dot(omega, tau),
        atol=EULER_MATCH_ATOL,
    )
    np.testing.assert_allclose(
        gyrostat_omega_dot(_ASYM_J, omega, tau, np.zeros(3)),
        body.omega_dot(omega, tau),
        atol=EULER_MATCH_ATOL,
    )


def test_hw_zero_rk4_matches_rigid_body():
    body = RigidBody(_ASYM_J)
    gyro = Gyrostat(_ASYM_J, wheel_momentum=0.0)
    q = quat_normalize([0.4, 0.3, 0.2, 0.8])
    omega = np.array([-0.5, 0.4, 0.7])
    tau = np.array([0.01, -0.02, 0.015])
    dt = 0.002
    q_r, w_r = q.copy(), omega.copy()
    q_g, w_g = q.copy(), omega.copy()
    for _ in range(250):
        q_r, w_r = step_rigid_body(body, q_r, w_r, tau, dt)
        q_g, w_g, rate = step_gyrostat_attitude(gyro, q_g, w_g, tau, dt)
        assert rate is None
    np.testing.assert_allclose(w_g, w_r, atol=1e-13)
    np.testing.assert_allclose(q_g, q_r, atol=1e-13)


def test_torque_free_energy_and_momentum_constant_hw():
    gyro = Gyrostat(_ASYM_J, rotor_axis="z", wheel_momentum=0.35)
    q = np.array([1.0, 0.0, 0.0, 0.0])
    omega = np.array([0.3, -0.8, 0.2])
    dt = 0.002
    t_final = 8.0
    n = int(t_final / dt)
    e0 = gyro.kinetic_energy(omega)
    h0 = gyro.angular_momentum_inertial(q, omega)
    hb0 = np.linalg.norm(gyro.angular_momentum_body(omega))
    tau0 = np.zeros(3)

    for _ in range(n):
        q, omega, _ = step_gyrostat_attitude(gyro, q, omega, tau0, dt)

    e1 = gyro.kinetic_energy(omega)
    h1 = gyro.angular_momentum_inertial(q, omega)
    hb1 = np.linalg.norm(gyro.angular_momentum_body(omega))
    assert abs(e1 - e0) / e0 < ENERGY_REL_TOL
    np.testing.assert_allclose(h1, h0, atol=H_INERTIAL_ATOL)
    assert abs(hb1 - hb0) / hb0 < H_BODY_NORM_REL_TOL
    assert abs(np.linalg.norm(q) - 1.0) < QUAT_NORM_ABS_TOL
    # Holding-motor pairing ω·h_w is not a first integral.
    pairing0 = float(np.array([0.3, -0.8, 0.2]) @ gyro.h_wheel_vector())
    pairing1 = float(omega @ gyro.h_wheel_vector())
    assert abs(pairing1 - pairing0) > 1e-6


def test_rate_only_rk4_conserves_hamiltonian_and_casimir():
    gyro = Gyrostat(_TRIAXIAL, rotor_axis=2, wheel_momentum=0.4)
    omega = np.array([0.25, -0.15, 0.6])
    e0 = gyro.kinetic_energy(omega)
    hb0 = np.linalg.norm(gyro.angular_momentum_body(omega))
    for _ in range(3000):
        omega, rate = step_gyrostat(gyro, omega, np.zeros(3), 0.002)
        assert rate is None
    assert abs(gyro.kinetic_energy(omega) - e0) / e0 < ENERGY_REL_TOL
    hb1 = np.linalg.norm(gyro.angular_momentum_body(omega))
    assert abs(hb1 - hb0) / hb0 < H_BODY_NORM_REL_TOL


def test_applied_torque_breaks_energy_conservation():
    gyro = Gyrostat(_ASYM_J, wheel_momentum=0.2)
    omega = np.array([0.3, -0.8, 0.2])
    e0 = gyro.kinetic_energy(omega)
    tau = np.array([0.05, -0.02, 0.03])
    for _ in range(400):
        omega, _ = step_gyrostat(gyro, omega, tau, 0.002)
    assert abs(gyro.kinetic_energy(omega) - e0) / e0 > 1e-3


def test_rotor_axis_spin_is_equilibrium():
    gyro = Gyrostat(_TRIAXIAL, rotor_axis="z", wheel_momentum=0.5)
    omega0 = np.array([0.0, 0.0, 1.1])
    omega = omega0.copy()
    for _ in range(400):
        omega, _ = step_gyrostat(gyro, omega, np.zeros(3), 0.01)
    np.testing.assert_allclose(omega, omega0, atol=SPIN_EQ_ATOL)
    np.testing.assert_allclose(gyro.omega_dot(omega0, np.zeros(3)), 0.0, atol=1e-15)


def test_axisymmetric_precession_matches_closed_form():
    """J = diag(I, I, I3), h_w = h e3: ω3 constant, transverse rate at

        Ω_p = ((I3 − I) ω3 + h) / I.
    """
    inertia_t, inertia_3, h = 2.0, 3.0, 0.4
    gyro = Gyrostat(np.diag([inertia_t, inertia_t, inertia_3]), "z", wheel_momentum=h)
    omega0 = np.array([0.3, 0.1, 0.8])
    omega_p = ((inertia_3 - inertia_t) * omega0[2] + h) / inertia_t
    dt = 0.001
    t_final = 5.0
    n = int(t_final / dt)
    omega = omega0.copy()
    for _ in range(n):
        omega, _ = step_gyrostat(gyro, omega, np.zeros(3), dt)
    t = n * dt
    c, s = np.cos(omega_p * t), np.sin(omega_p * t)
    expected = np.array(
        [
            omega0[0] * c - omega0[1] * s,
            omega0[0] * s + omega0[1] * c,
            omega0[2],
        ]
    )
    np.testing.assert_allclose(omega, expected, atol=AXISYM_OMEGA_ATOL)


def test_axisymmetric_precession_reduces_to_euler_when_hw_zero():
    inertia_t, inertia_3 = 2.0, 3.0
    body = RigidBody(np.diag([inertia_t, inertia_t, inertia_3]))
    gyro = Gyrostat(np.diag([inertia_t, inertia_t, inertia_3]), wheel_momentum=0.0)
    omega = np.array([0.3, 0.1, 0.8])
    q = np.array([1.0, 0.0, 0.0, 0.0])
    dt = 0.002
    for _ in range(800):
        q_b, w_b = step_rigid_body(body, q, omega, np.zeros(3), dt)
        q_g, w_g, _ = step_gyrostat_attitude(gyro, q, omega, np.zeros(3), dt)
        np.testing.assert_allclose(w_g, w_b, atol=1e-14)
        q, omega = q_g, w_g
    np.testing.assert_allclose(q, q_b, atol=1e-13)


def test_max_axis_spin_stays_near_rotor_axis():
    """Spin about max principal axis + rotor along that axis stays bounded."""
    gyro = Gyrostat(_TRIAXIAL, rotor_axis="z", wheel_momentum=0.3)
    assert transverse_lambda_squared(_TRIAXIAL, 2, spin_rate=1.0, h_wheel=0.3) < 0.0
    pert = 0.03
    omega = np.array([pert, pert, 1.0])
    for _ in range(int(12.0 / 0.002)):
        omega, _ = step_gyrostat(gyro, omega, np.zeros(3), 0.002)
    assert omega[2] > 0.5
    assert float(np.hypot(omega[0], omega[1])) < 8.0 * pert


def test_dual_spin_stabilizes_intermediate_axis():
    """Large |h| on the intermediate axis makes λ² < 0; a tilt stays O(pert)."""
    # I = (1, 2, 3), Ω = 1 about e2, h along e2: λ² = −(h² − 1)/3.
    assert transverse_lambda_squared(_TRIAXIAL, 1, 1.0, h_wheel=0.0) > 0.0
    assert transverse_lambda_squared(_TRIAXIAL, 1, 1.0, h_wheel=2.0) < 0.0
    gyro = Gyrostat(_TRIAXIAL, rotor_axis="y", wheel_momentum=2.0)
    pert = 0.03
    omega = np.array([pert, 1.0, pert])
    growth = []
    for k in range(int(12.0 / 0.002)):
        omega, _ = step_gyrostat(gyro, omega, np.zeros(3), 0.002)
        if k % 50 == 0:
            growth.append(float(np.hypot(omega[0], omega[2])))
    assert omega[1] > 0.5
    assert max(growth) < 8.0 * pert


def test_intermediate_axis_without_hw_is_unstable():
    """Sanity: h = 0 recovers tennis-racket growth about the mid axis."""
    gyro = Gyrostat(_TRIAXIAL, rotor_axis="y", wheel_momentum=0.0)
    pert = 0.03
    omega = np.array([pert, 1.0, pert])
    growth = []
    for k in range(int(18.0 / 0.002)):
        omega, _ = step_gyrostat(gyro, omega, np.zeros(3), 0.002)
        if k % 50 == 0:
            growth.append(float(np.hypot(omega[0], omega[2])))
    assert max(growth) > 10.0 * pert


def test_transverse_lambda_squared_analytic_and_rejects_bad_axis():
    # h = 0, I = (1, 2, 3): tennis-racket values from the plant / polhode notes.
    np.testing.assert_allclose(transverse_lambda_squared(_TRIAXIAL, 0, 1.0, 0.0), -1.0 / 3.0)
    np.testing.assert_allclose(transverse_lambda_squared(_TRIAXIAL, 1, 1.0, 0.0), 1.0 / 3.0)
    np.testing.assert_allclose(transverse_lambda_squared(_TRIAXIAL, 2, 1.0, 0.0), -1.0)
    np.testing.assert_allclose(transverse_lambda_squared(_TRIAXIAL, 1, 1.0, 2.0), -1.0)
    with pytest.raises(ValueError, match="axis must be"):
        transverse_lambda_squared(_TRIAXIAL, 3)
    with pytest.raises(ValueError, match="finite"):
        transverse_lambda_squared(_TRIAXIAL, 0, spin_rate=np.inf)


def test_variable_wheel_rk4_conserves_total_momentum_tau_ext_zero():
    gyro = Gyrostat(_ASYM_J, rotor_axis="x", wheel_inertia=0.08, wheel_momentum=0.0)
    q = quat_normalize([0.5, -0.2, 0.1, 0.8])
    omega = np.array([0.2, -0.15, 0.3])
    omega_w = 4.0
    h0 = gyro.angular_momentum_inertial(q, omega, omega_w)
    tau_w = 0.02
    e0 = gyro.kinetic_energy(omega)
    for _ in range(800):
        q, omega, omega_w = step_gyrostat_attitude(
            gyro, q, omega, np.zeros(3), 0.002, omega_w=omega_w, tau_w=tau_w
        )
    assert omega_w is not None
    h1 = gyro.angular_momentum_inertial(q, omega, omega_w)
    np.testing.assert_allclose(h1, h0, atol=H_INERTIAL_ATOL)
    # Internal motor changes carrier energy and wheel speed.
    assert abs(gyro.kinetic_energy(omega) - e0) / max(e0, 1e-12) > 1e-4
    assert abs(omega_w - 4.0) > 1e-3
    assert gyro.rotor_kinetic_energy(omega_w) > 0.0
    assert abs(np.linalg.norm(q) - 1.0) < QUAT_NORM_ABS_TOL


def test_variable_wheel_zero_motor_matches_constant_speed():
    i_w, omega_w = 0.05, 3.0
    h = i_w * omega_w
    constant = Gyrostat(_TRIAXIAL, "z", wheel_momentum=h)
    variable = Gyrostat(_TRIAXIAL, "z", wheel_inertia=i_w, wheel_momentum=h)
    omega = np.array([0.2, -0.1, 0.5])
    w_c = omega.copy()
    w_v, rate = omega.copy(), omega_w
    for _ in range(500):
        w_c, _ = step_gyrostat(constant, w_c, np.zeros(3), 0.002)
        w_v, rate = step_gyrostat(variable, w_v, np.zeros(3), 0.002, omega_w=rate, tau_w=0.0)
    np.testing.assert_allclose(w_v, w_c, atol=1e-12)
    assert rate is not None
    np.testing.assert_allclose(rate, omega_w, atol=1e-14)


def test_gyrostat_omega_dot_includes_h_wheel_dot():
    omega = np.array([0.1, 0.0, 0.2])
    tau = np.zeros(3)
    h_w = np.array([0.0, 0.0, 0.3])
    h_dot = np.array([0.05, 0.0, 0.0])
    wdot = gyrostat_omega_dot(_TRIAXIAL, omega, tau, h_w, h_dot)
    expected = gyrostat_omega_dot(_TRIAXIAL, omega, tau - h_dot, h_w)
    np.testing.assert_allclose(wdot, expected, atol=1e-15)


def test_commanded_h_wheel_override():
    gyro = Gyrostat(_TRIAXIAL, wheel_momentum=0.0)
    omega = np.array([0.1, 0.2, 0.3])
    h_cmd = np.array([0.0, 0.0, 0.4])
    wdot = gyro.omega_dot(omega, np.zeros(3), h_wheel=h_cmd)
    expected = gyrostat_omega_dot(_TRIAXIAL, omega, np.zeros(3), h_cmd)
    np.testing.assert_allclose(wdot, expected, atol=1e-15)


def test_pack_unpack_rate_state():
    omega = np.array([0.1, -0.2, 0.3])
    y = pack_rate_state(omega)
    w, rate = unpack_rate_state(y)
    np.testing.assert_allclose(w, omega)
    assert rate is None
    y4 = pack_rate_state(omega, 2.5)
    w4, rate4 = unpack_rate_state(y4)
    np.testing.assert_allclose(w4, omega)
    assert rate4 == 2.5
    with pytest.raises(ValueError, match="length 3"):
        unpack_rate_state(np.ones(2))


def test_gyrostat_rejects_bad_construction_and_steps():
    with pytest.raises(ValueError, match="principal moments"):
        Gyrostat(np.diag([1.0, 1.0, 3.0]))
    with pytest.raises(ValueError, match="wheel_inertia must be positive"):
        Gyrostat(_TRIAXIAL, wheel_inertia=0.0)
    with pytest.raises(ValueError, match="wheel_momentum"):
        Gyrostat(_TRIAXIAL, wheel_momentum=np.nan)
    gyro = Gyrostat(_TRIAXIAL, wheel_momentum=0.1)
    with pytest.raises(ValueError, match="dt must be positive"):
        step_gyrostat(gyro, np.zeros(3), np.zeros(3), 0.0)
    with pytest.raises(ValueError, match="tau_w requires"):
        step_gyrostat(gyro, np.zeros(3), np.zeros(3), 0.01, tau_w=0.1)
    with pytest.raises(ValueError, match="omega_w requires"):
        step_gyrostat(gyro, np.zeros(3), np.zeros(3), 0.01, omega_w=1.0)
    with pytest.raises(ValueError, match="finite"):
        gyro.omega_dot([np.nan, 0.0, 0.0], np.zeros(3))
    with pytest.raises(ValueError, match="omega_w requires"):
        gyro.h_wheel_vector(omega_w=1.0)


def test_as_rigid_body_shares_inertia():
    gyro = Gyrostat(_ASYM_J, wheel_momentum=0.2)
    body = gyro.as_rigid_body()
    np.testing.assert_allclose(body.inertia, gyro.inertia)
    omega = np.array([0.2, 0.1, -0.3])
    # Locked-rotor energy matches RigidBody; gyro momentum adds h_w.
    assert abs(gyro.kinetic_energy(omega) - body.kinetic_energy(omega)) < 1e-15
    np.testing.assert_allclose(
        gyro.angular_momentum_body(omega),
        body.angular_momentum_body(omega) + gyro.h_wheel_vector(),
    )

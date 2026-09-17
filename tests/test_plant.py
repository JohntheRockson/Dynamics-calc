"""RK4 plant: quaternion norm, inertia helpers, torque-free conservation.

Torque-free tolerances below are for classical RK4 (not symplectic).  They
are documented bounds for the scenarios in this file, not machine epsilon.
"""

import numpy as np
import pytest

from attitude_sim.plant import (
    RigidBody,
    inertia_from_principal,
    is_principal,
    pack_state,
    principal_moments_and_axes,
    rk4_step,
    step_rigid_body,
    unpack_state,
    validate_inertia,
)
from attitude_sim.quaternions import quat_normalize


# Asymmetric J, dt = 0.002 s, T = 10 s.  Bounds sit ~100× above the observed
# RK4 residuals (~1e-14) for these cases so they stay honest without being flaky.
ENERGY_REL_TOL = 1e-12
H_INERTIAL_REL_TOL = 1e-12
H_INERTIAL_ABS_TOL = 1e-12
H_BODY_NORM_REL_TOL = 1e-12
QUAT_NORM_ABS_TOL = 1e-12

# Principal-axis spin (diagonal J, ω along a principal axis).
PRINCIPAL_SPIN_ATOL = 1e-12

# Spherical body: ω is an exact equilibrium of Euler's equation.
SPHERICAL_OMEGA_ATOL = 1e-14

# Axisymmetric closed-form precession at dt = 0.001 s, T = 5 s.
AXISYM_OMEGA_ATOL = 1e-12


def _asymmetric_inertia() -> np.ndarray:
    # Products of inertia so the implementation is not secretly diagonal-only.
    J = np.array(
        [
            [2.0, 0.1, 0.0],
            [0.1, 3.0, 0.15],
            [0.0, 0.15, 4.0],
        ]
    )
    return 0.5 * (J + J.T)


def test_rk4_matches_analytic_scalar_ode():
    def fun(_t, y):
        return -0.5 * y

    y = np.array([2.0])
    dt = 0.05
    for k in range(40):
        y = rk4_step(fun, k * dt, y, dt)
    expected = 2.0 * np.exp(-0.5 * 40 * dt)
    np.testing.assert_allclose(y[0], expected, rtol=1e-8)


def test_rk4_global_error_is_fourth_order():
    def fun(_t, y):
        return -y

    t_final = 1.0
    dts = np.array([0.1, 0.05, 0.025])
    errors = []
    for dt in dts:
        n = int(round(t_final / dt))
        y = np.array([1.0])
        t = 0.0
        for _ in range(n):
            y = rk4_step(fun, t, y, dt)
            t += dt
        errors.append(abs(y[0] - np.exp(-t_final)))
    errors = np.asarray(errors)
    ratios = errors[:-1] / errors[1:]
    # Halving dt should cut global error by ~ 16 for a 4th-order method.
    np.testing.assert_allclose(ratios, 16.0, rtol=0.15)
    assert np.all(errors[1:] < errors[:-1])


def test_rk4_rejects_non_positive_dt():
    def fun(_t, y):
        return y

    with pytest.raises(ValueError, match="dt must be positive"):
        rk4_step(fun, 0.0, np.array([1.0]), 0.0)
    with pytest.raises(ValueError, match="dt must be positive"):
        rk4_step(fun, 0.0, np.array([1.0]), -0.01)


def test_quaternion_norm_under_random_torque():
    body = RigidBody(_asymmetric_inertia())
    q = quat_normalize([0.6, 0.3, -0.2, 0.7])
    omega = np.array([0.4, -0.2, 0.15])
    rng = np.random.default_rng(2)
    dt = 0.01
    norms = []
    for _ in range(400):
        tau = rng.normal(scale=0.05, size=3)
        q, omega = step_rigid_body(body, q, omega, tau, dt)
        norms.append(np.linalg.norm(q))
    np.testing.assert_allclose(norms, 1.0, atol=QUAT_NORM_ABS_TOL)


def test_torque_free_energy_and_inertial_momentum():
    body = RigidBody(_asymmetric_inertia())
    q = np.array([1.0, 0.0, 0.0, 0.0])
    omega = np.array([0.3, -0.8, 0.2])
    dt = 0.002
    t_final = 10.0
    n = int(t_final / dt)
    E0 = body.kinetic_energy(omega)
    h0 = body.angular_momentum_inertial(q, omega)
    hb0 = np.linalg.norm(body.angular_momentum_body(omega))

    for _ in range(n):
        q, omega = step_rigid_body(body, q, omega, np.zeros(3), dt)

    E1 = body.kinetic_energy(omega)
    h1 = body.angular_momentum_inertial(q, omega)
    hb1 = np.linalg.norm(body.angular_momentum_body(omega))
    assert abs(E1 - E0) / E0 < ENERGY_REL_TOL
    np.testing.assert_allclose(h1, h0, rtol=H_INERTIAL_REL_TOL, atol=H_INERTIAL_ABS_TOL)
    assert abs(hb1 - hb0) / hb0 < H_BODY_NORM_REL_TOL
    assert abs(np.linalg.norm(q) - 1.0) < QUAT_NORM_ABS_TOL


def test_torque_free_invariants_hold_on_principal_tensor():
    """Same integrals for a diagonal inertia, independent of products of inertia."""
    body = RigidBody(np.diag([1.5, 2.5, 3.5]))
    q = quat_normalize([0.4, 0.3, 0.2, 0.8])
    omega = np.array([-0.5, 0.4, 0.7])
    dt = 0.002
    E0 = body.kinetic_energy(omega)
    h0 = body.angular_momentum_inertial(q, omega)
    hb0 = body.angular_momentum_body(omega)

    for _ in range(2500):
        q, omega = step_rigid_body(body, q, omega, np.zeros(3), dt)

    E1 = body.kinetic_energy(omega)
    h1 = body.angular_momentum_inertial(q, omega)
    hb1 = body.angular_momentum_body(omega)
    assert abs(E1 - E0) / E0 < ENERGY_REL_TOL
    np.testing.assert_allclose(h1, h0, rtol=H_INERTIAL_REL_TOL, atol=H_INERTIAL_ABS_TOL)
    # Body-frame h changes direction; only its magnitude is a first integral.
    assert abs(np.linalg.norm(hb1) - np.linalg.norm(hb0)) / np.linalg.norm(hb0) < H_BODY_NORM_REL_TOL
    assert abs(np.linalg.norm(q) - 1.0) < QUAT_NORM_ABS_TOL


def test_applied_torque_breaks_energy_conservation():
    body = RigidBody(_asymmetric_inertia())
    q = np.array([1.0, 0.0, 0.0, 0.0])
    omega = np.array([0.3, -0.8, 0.2])
    E0 = body.kinetic_energy(omega)
    tau = np.array([0.05, -0.02, 0.03])
    for _ in range(400):
        q, omega = step_rigid_body(body, q, omega, tau, 0.002)
    assert abs(body.kinetic_energy(omega) - E0) / E0 > 1e-3


def test_principal_axis_spin_is_equilibrium():
    body = RigidBody(np.diag([1.0, 2.0, 3.0]))
    q = np.array([1.0, 0.0, 0.0, 0.0])
    omega = np.array([0.0, 0.0, 1.2])
    for _ in range(200):
        q, omega = step_rigid_body(body, q, omega, np.zeros(3), 0.01)
    np.testing.assert_allclose(omega, [0.0, 0.0, 1.2], atol=PRINCIPAL_SPIN_ATOL)
    assert abs(np.linalg.norm(q) - 1.0) < QUAT_NORM_ABS_TOL


def test_spherical_inertia_keeps_omega_constant():
    body = RigidBody(2.0 * np.eye(3))
    q = quat_normalize([0.2, 0.5, -0.1, 0.8])
    omega0 = np.array([0.4, -0.3, 0.25])
    omega = omega0.copy()
    for _ in range(500):
        q, omega = step_rigid_body(body, q, omega, np.zeros(3), 0.01)
    np.testing.assert_allclose(omega, omega0, atol=SPHERICAL_OMEGA_ATOL)


def test_axisymmetric_precession_matches_closed_form():
    """For J = diag(I, I, I3), ω3 is constant and the transverse rate rotates at

        Ω = ((I3 − I) / I) ω3

    with

        ω1(t) =  ω10 cos(Ω t) − ω20 sin(Ω t)
        ω2(t) =  ω10 sin(Ω t) + ω20 cos(Ω t).
    """
    I, I3 = 2.0, 3.0
    body = RigidBody(np.diag([I, I, I3]))
    q = np.array([1.0, 0.0, 0.0, 0.0])
    omega0 = np.array([0.3, 0.1, 0.8])
    Omega = ((I3 - I) / I) * omega0[2]
    dt = 0.001
    t_final = 5.0
    n = int(t_final / dt)
    q, omega = q.copy(), omega0.copy()
    for _ in range(n):
        q, omega = step_rigid_body(body, q, omega, np.zeros(3), dt)
    t = n * dt
    c, s = np.cos(Omega * t), np.sin(Omega * t)
    expected = np.array(
        [
            omega0[0] * c - omega0[1] * s,
            omega0[0] * s + omega0[1] * c,
            omega0[2],
        ]
    )
    np.testing.assert_allclose(omega, expected, atol=AXISYM_OMEGA_ATOL)
    assert abs(np.linalg.norm(q) - 1.0) < QUAT_NORM_ABS_TOL


def test_validate_inertia_rejects_bad_tensors():
    with pytest.raises(ValueError, match="3x3"):
        validate_inertia(np.eye(2))
    with pytest.raises(ValueError, match="symmetric"):
        validate_inertia(np.array([[1.0, 0.2, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]))
    with pytest.raises(ValueError, match="positive definite"):
        validate_inertia(np.diag([1.0, -0.1, 2.0]))
    # SPD but not a physical rigid-body inertia (triangle inequality).
    with pytest.raises(ValueError, match="principal moments"):
        validate_inertia(np.diag([1.0, 1.0, 3.0]))
    planar = validate_inertia(np.diag([1.0, 1.0, 2.0]))
    np.testing.assert_allclose(planar, np.diag([1.0, 1.0, 2.0]))
    # Optional escape hatch: keep a non-physical SPD tensor if requested.
    J = validate_inertia(np.diag([1.0, 1.0, 3.0]), require_physical=False)
    np.testing.assert_allclose(J, np.diag([1.0, 1.0, 3.0]))


def test_principal_axes_reconstruct_inertia():
    J = _asymmetric_inertia()
    moments, axes = principal_moments_and_axes(J)
    assert abs(np.linalg.det(axes) - 1.0) < 1e-12
    np.testing.assert_allclose(axes.T @ axes, np.eye(3), atol=1e-12)
    reconstructed = axes @ np.diag(moments) @ axes.T
    np.testing.assert_allclose(reconstructed, J, atol=1e-12)
    body = RigidBody(J)
    np.testing.assert_allclose(np.sort(body.principal_moments), np.sort(moments))
    J2 = inertia_from_principal(moments, axes)
    np.testing.assert_allclose(J2, J, atol=1e-12)
    assert is_principal(np.diag([1.0, 2.0, 3.0]))
    assert not is_principal(J)


def test_inertia_from_principal_diagonal_and_rotated():
    J_diag = inertia_from_principal([1.0, 2.0, 2.5])
    np.testing.assert_allclose(J_diag, np.diag([1.0, 2.0, 2.5]))
    # 90° about z: body x ← y, body y ← −x.
    Rz = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    J_rot = inertia_from_principal([1.0, 2.0, 2.5], Rz)
    expected = Rz @ np.diag([1.0, 2.0, 2.5]) @ Rz.T
    np.testing.assert_allclose(J_rot, expected, atol=1e-12)
    with pytest.raises(ValueError, match="orthonormal"):
        inertia_from_principal([1.0, 2.0, 2.5], np.diag([1.0, 2.0, 1.0]))
    with pytest.raises(ValueError, match="proper rotation"):
        inertia_from_principal([1.0, 2.0, 2.5], np.diag([1.0, 1.0, -1.0]))


def test_rigid_body_rejects_nonphysical_inertia():
    with pytest.raises(ValueError, match="principal moments"):
        RigidBody(np.diag([1.0, 1.0, 3.0]))
    with pytest.raises(ValueError, match="positive definite"):
        RigidBody(np.diag([1.0, 0.0, 2.0]))


def test_pack_unpack_state_roundtrip():
    q = quat_normalize([0.4, 0.2, -0.1, 0.8])
    omega = np.array([0.1, -0.2, 0.3])
    y = pack_state(q, omega)
    assert y.shape == (7,)
    q2, w2 = unpack_state(y)
    np.testing.assert_allclose(q2, q)
    np.testing.assert_allclose(w2, omega)


def test_spherical_constant_torque_matches_closed_form():
    """For J = I I_3, ω̇ = τ/I is exact; RK4 of a constant vector field is exact."""
    inertia_scale = 2.0
    body = RigidBody(inertia_scale * np.eye(3))
    q = np.array([1.0, 0.0, 0.0, 0.0])
    omega = np.zeros(3)
    tau = np.array([0.01, -0.02, 0.03])
    dt = 0.01
    n = 200
    for _ in range(n):
        q, omega = step_rigid_body(body, q, omega, tau, dt)
    expected = tau / inertia_scale * n * dt
    np.testing.assert_allclose(omega, expected, atol=1e-12)
    expected_T = 0.5 * inertia_scale * float(expected @ expected)
    assert abs(body.kinetic_energy(omega) - expected_T) < 1e-14
    assert abs(np.linalg.norm(q) - 1.0) < QUAT_NORM_ABS_TOL


def test_work_energy_theorem_with_constant_torque():
    """ΔT ≈ ∫ ω·τ dt (trapezoid) for a ZOH torque on an asymmetric body."""
    body = RigidBody(np.diag([1.5, 2.5, 3.5]))
    q = quat_normalize([0.4, 0.3, 0.2, 0.8])
    omega = np.array([-0.5, 0.4, 0.7])
    tau = np.array([0.04, -0.02, 0.01])
    dt = 0.001
    t0 = body.kinetic_energy(omega)
    work = 0.0
    for _ in range(2000):
        w0 = omega.copy()
        q, omega = step_rigid_body(body, q, omega, tau, dt)
        work += 0.5 * float((w0 + omega) @ tau) * dt
    t1 = body.kinetic_energy(omega)
    assert abs(t1 - t0 - work) / abs(work) < 1e-6
    assert abs(np.linalg.norm(q) - 1.0) < QUAT_NORM_ABS_TOL

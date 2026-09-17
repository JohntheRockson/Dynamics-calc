"""RKMK4 / exponential-map attitude step: unit quaternion and RK4-comparable drift.

Classical RK4 on the Euclidean chart leaves S^3 and is projected afterwards.
RKMK4 composes ``q ⊗ Exp(φ)``, so ``||q|| = 1`` holds by construction
(no renormalization).  Torque-free energy / momentum residuals are
documented against the same RK4 bounds in ``tests/test_plant.py``.
"""

import numpy as np
import pytest

from attitude_sim.plant import (
    INTEGRATOR_RK4,
    INTEGRATOR_RKMK4,
    RigidBody,
    dexpinv_so3,
    pack_state,
    rk4_step,
    rkmk4_step,
    so3_quat_exp,
    step_rigid_body,
    unpack_state,
)
from attitude_sim.quaternions import (
    axis_angle_to_quat,
    quat_integrate_const_omega,
    quat_multiply,
    quat_normalize,
    quat_to_rotation,
)


# Same asymmetric J and RK4 bounds as tests/test_plant.py (dt = 0.002 s, T = 10 s).
ENERGY_REL_TOL = 1e-12
H_INERTIAL_REL_TOL = 1e-12
H_INERTIAL_ABS_TOL = 1e-12
H_BODY_NORM_REL_TOL = 1e-12
QUAT_NORM_ABS_TOL = 1e-12
# Exp / multiply roundoff, no explicit normalize.
QUAT_NORM_BY_CONSTRUCTION_ATOL = 1e-14
# Drift contrast: Euclidean RK4 without projection vs RKMK4.
EUCLIDEAN_NORM_DRIFT_MIN = 1e-8


def _asymmetric_inertia() -> np.ndarray:
    J = np.array(
        [
            [2.0, 0.1, 0.0],
            [0.1, 3.0, 0.15],
            [0.0, 0.15, 4.0],
        ]
    )
    return 0.5 * (J + J.T)


def _invariants(body: RigidBody, q, omega):
    e = body.kinetic_energy(omega)
    h = body.angular_momentum_inertial(q, omega)
    hb = np.linalg.norm(body.angular_momentum_body(omega))
    return e, h, hb


def test_so3_quat_exp_is_unit_by_construction():
    """||Exp(φ)|| = 1 from cos²+sin² (no quat_normalize)."""
    rng = np.random.default_rng(5)
    qs = [so3_quat_exp(np.zeros(3))]
    qs.append(so3_quat_exp(np.array([1e-16, 0.0, 0.0])))
    for _ in range(40):
        qs.append(so3_quat_exp(rng.normal(scale=1.3, size=3)))
    for q in qs:
        assert abs(float(np.linalg.norm(q)) - 1.0) < QUAT_NORM_BY_CONSTRUCTION_ATOL
    ident = so3_quat_exp(np.zeros(3))
    np.testing.assert_allclose(ident, [1.0, 0.0, 0.0, 0.0], atol=1e-15)


def test_so3_quat_exp_matches_axis_angle():
    phi = np.array([0.3, -0.4, 0.5])
    theta = float(np.linalg.norm(phi))
    q = so3_quat_exp(phi)
    expected = axis_angle_to_quat(phi, theta)
    np.testing.assert_allclose(q, expected, atol=1e-15)
    # Rodrigues: Exp(φ) rotates by θ about φ.
    R = quat_to_rotation(q)
    axis = phi / theta
    c, s = np.cos(theta), np.sin(theta)
    v = np.array([0.2, 0.7, -0.3])
    rod = v * c + np.cross(axis, v) * s + axis * (axis @ v) * (1.0 - c)
    np.testing.assert_allclose(R @ v, rod, atol=1e-12)


def test_dexpinv_so3_identity_on_parallel_and_zero():
    v = np.array([0.4, -0.1, 0.2])
    np.testing.assert_allclose(dexpinv_so3(np.zeros(3), v), v, atol=1e-15)
    phi = np.array([0.8, -0.2, 0.4])
    np.testing.assert_allclose(dexpinv_so3(phi, phi), phi, atol=1e-14)
    # Closed form vs Bernoulli series in the overlap region.
    phi_small = np.array([3e-6, -1e-6, 2e-6])
    vec = np.array([0.3, 0.1, -0.2])
    w = np.cross(phi_small, vec)
    series = vec - 0.5 * w + (1.0 / 12.0) * np.cross(phi_small, w)
    np.testing.assert_allclose(dexpinv_so3(phi_small, vec), series, atol=1e-16)


def test_step_rigid_body_method_switch_and_rejects_unknown():
    body = RigidBody(_asymmetric_inertia())
    q = quat_normalize([0.6, 0.3, -0.2, 0.7])
    omega = np.array([0.4, -0.2, 0.15])
    tau = np.array([0.01, -0.02, 0.03])
    dt = 0.01
    q_rk4, w_rk4 = step_rigid_body(body, q, omega, tau, dt)
    q_default, w_default = step_rigid_body(body, q, omega, tau, dt, method=INTEGRATOR_RK4)
    q_rkmk, w_rkmk = step_rigid_body(body, q, omega, tau, dt, method=INTEGRATOR_RKMK4)
    q_direct, w_direct = rkmk4_step(body, q, omega, tau, dt)
    np.testing.assert_allclose(q_rk4, q_default)
    np.testing.assert_allclose(w_rk4, w_default)
    np.testing.assert_allclose(q_rkmk, q_direct)
    np.testing.assert_allclose(w_rkmk, w_direct)
    # Distinct charts: Euclidean RK4 + project vs group exponential.
    assert np.linalg.norm(q_rk4 - q_rkmk) > 1e-14 or np.linalg.norm(w_rk4 - w_rkmk) > 1e-14
    with pytest.raises(ValueError, match="unknown integrator"):
        step_rigid_body(body, q, omega, tau, dt, method="euler")


def test_rkmk4_rejects_non_positive_dt():
    body = RigidBody(np.diag([1.5, 2.5, 3.5]))
    q = np.array([1.0, 0.0, 0.0, 0.0])
    omega = np.zeros(3)
    tau = np.zeros(3)
    with pytest.raises(ValueError, match="dt must be positive"):
        rkmk4_step(body, q, omega, tau, 0.0)
    with pytest.raises(ValueError, match="dt must be positive"):
        step_rigid_body(body, q, omega, tau, -0.01, method="rkmk4")
    with pytest.raises(ValueError, match="dt must be positive"):
        rkmk4_step(body, q, omega, tau, float("nan"))


def test_rkmk4_quaternion_unit_by_construction_random_torque():
    """Many forced steps, never calling quat_normalize on the RKMK4 output."""
    body = RigidBody(_asymmetric_inertia())
    q = quat_normalize([0.6, 0.3, -0.2, 0.7])
    omega = np.array([0.4, -0.2, 0.15])
    rng = np.random.default_rng(2)
    dt = 0.01
    norms = []
    for _ in range(400):
        tau = rng.normal(scale=0.05, size=3)
        q, omega = rkmk4_step(body, q, omega, tau, dt)
        norms.append(np.linalg.norm(q))
    np.testing.assert_allclose(norms, 1.0, atol=QUAT_NORM_BY_CONSTRUCTION_ATOL)
    # Composition of unit factors, not a post-hoc projection of a chart update.
    assert abs(float(np.linalg.norm(so3_quat_exp(np.array([1.2, -0.4, 0.3])))) - 1.0) < 1e-15


def test_rk4_euclidean_leaves_sphere_rkmk4_does_not():
    """Without the RK4 projection, ||q|| drifts; RKMK4 stays on S^3."""
    body = RigidBody(_asymmetric_inertia())
    q0 = quat_normalize([0.4, 0.3, 0.2, 0.8])
    omega0 = np.array([-0.5, 0.4, 0.7])
    tau = np.zeros(3)
    dt = 0.05
    n = 200

    def fun(_t: float, y: np.ndarray) -> np.ndarray:
        return body.derivatives(y[:4], y[4:], tau)

    y = pack_state(q0, omega0)
    for _ in range(n):
        y = rk4_step(fun, 0.0, y, dt)
    q_eucl, _w_eucl = unpack_state(y)
    assert abs(float(np.linalg.norm(q_eucl)) - 1.0) > EUCLIDEAN_NORM_DRIFT_MIN

    q, omega = q0.copy(), omega0.copy()
    for _ in range(n):
        q, omega = rkmk4_step(body, q, omega, tau, dt)
    assert abs(float(np.linalg.norm(q)) - 1.0) < QUAT_NORM_BY_CONSTRUCTION_ATOL


def test_rkmk4_const_omega_matches_exact_exponential():
    """Spherical torque-free: ω is constant, so RKMK4 is the exact Exp map."""
    body = RigidBody(2.0 * np.eye(3))
    q = quat_normalize([0.2, 0.5, -0.1, 0.8])
    omega = np.array([0.4, -0.3, 0.25])
    dt = 0.07
    n = 50
    q_num, w_num = q.copy(), omega.copy()
    for _ in range(n):
        q_num, w_num = rkmk4_step(body, q_num, w_num, np.zeros(3), dt)
    q_exact = quat_integrate_const_omega(q, omega, n * dt)
    np.testing.assert_allclose(w_num, omega, atol=1e-14)
    # Double cover: q and −q are the same attitude.
    if np.dot(q_num, q_exact) < 0.0:
        q_exact = -q_exact
    np.testing.assert_allclose(q_num, q_exact, atol=1e-12)
    assert abs(float(np.linalg.norm(q_num)) - 1.0) < QUAT_NORM_BY_CONSTRUCTION_ATOL


def test_rkmk4_torque_free_energy_and_momentum_comparable_to_rk4():
    """Same IC / dt / horizon as test_plant.test_torque_free_energy_and_inertial_momentum."""
    body = RigidBody(_asymmetric_inertia())
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    omega0 = np.array([0.3, -0.8, 0.2])
    dt = 0.002
    n = int(10.0 / dt)
    e0, h0, hb0 = _invariants(body, q0, omega0)

    q_rk4, w_rk4 = q0.copy(), omega0.copy()
    q_rkmk, w_rkmk = q0.copy(), omega0.copy()
    for _ in range(n):
        q_rk4, w_rk4 = step_rigid_body(body, q_rk4, w_rk4, np.zeros(3), dt, method="rk4")
        q_rkmk, w_rkmk = step_rigid_body(
            body, q_rkmk, w_rkmk, np.zeros(3), dt, method="rkmk4"
        )

    e_rk4, h_rk4, hb_rk4 = _invariants(body, q_rk4, w_rk4)
    e_rkmk, h_rkmk, hb_rkmk = _invariants(body, q_rkmk, w_rkmk)

    for e1, h1, hb1, q in (
        (e_rk4, h_rk4, hb_rk4, q_rk4),
        (e_rkmk, h_rkmk, hb_rkmk, q_rkmk),
    ):
        assert abs(e1 - e0) / e0 < ENERGY_REL_TOL
        np.testing.assert_allclose(h1, h0, rtol=H_INERTIAL_REL_TOL, atol=H_INERTIAL_ABS_TOL)
        assert abs(hb1 - hb0) / hb0 < H_BODY_NORM_REL_TOL
        assert abs(float(np.linalg.norm(q)) - 1.0) < QUAT_NORM_ABS_TOL

    # RKMK4 residual is not wildly worse than RK4 on this problem (factor 10).
    assert abs(e_rkmk - e0) <= 10.0 * max(abs(e_rk4 - e0), 1e-18)
    assert np.linalg.norm(h_rkmk - h0) <= 10.0 * max(np.linalg.norm(h_rk4 - h0), 1e-18)


def test_rkmk4_torque_free_on_principal_tensor():
    body = RigidBody(np.diag([1.5, 2.5, 3.5]))
    q = quat_normalize([0.4, 0.3, 0.2, 0.8])
    omega = np.array([-0.5, 0.4, 0.7])
    dt = 0.002
    e0, h0, hb0 = _invariants(body, q, omega)
    for _ in range(2500):
        q, omega = step_rigid_body(body, q, omega, np.zeros(3), dt, method="rkmk4")
    e1, h1, hb1 = _invariants(body, q, omega)
    assert abs(e1 - e0) / e0 < ENERGY_REL_TOL
    np.testing.assert_allclose(h1, h0, rtol=H_INERTIAL_REL_TOL, atol=H_INERTIAL_ABS_TOL)
    assert abs(hb1 - hb0) / hb0 < H_BODY_NORM_REL_TOL
    assert abs(float(np.linalg.norm(q)) - 1.0) < QUAT_NORM_BY_CONSTRUCTION_ATOL


def test_rkmk4_principal_axis_spin_is_equilibrium():
    body = RigidBody(np.diag([1.0, 2.0, 3.0]))
    q = np.array([1.0, 0.0, 0.0, 0.0])
    omega = np.array([0.0, 0.0, 1.2])
    for _ in range(200):
        q, omega = rkmk4_step(body, q, omega, np.zeros(3), 0.01)
    np.testing.assert_allclose(omega, [0.0, 0.0, 1.2], atol=1e-12)
    assert abs(float(np.linalg.norm(q)) - 1.0) < QUAT_NORM_BY_CONSTRUCTION_ATOL
    # Pure z-rate: Exp is a z-rotation, vector part only on z.
    np.testing.assert_allclose(q[1:3], 0.0, atol=1e-14)


def test_rkmk4_compose_two_unit_factors_without_normalize():
    """Sanity: q ⊗ Exp(φ) stays unit when q is unit, without quat_normalize."""
    q = quat_normalize([0.3, -0.5, 0.2, 0.7])
    phi = np.array([0.9, 0.1, -0.4])
    qn = quat_multiply(q, so3_quat_exp(phi))
    assert abs(float(np.linalg.norm(qn)) - 1.0) < QUAT_NORM_BY_CONSTRUCTION_ATOL

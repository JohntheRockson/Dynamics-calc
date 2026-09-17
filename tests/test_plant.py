"""RK4 plant: quaternion norm and torque-free conservation."""

import numpy as np

from attitude_sim.plant import RigidBody, rk4_step, step_rigid_body
from attitude_sim.quaternions import quat_normalize


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
    np.testing.assert_allclose(norms, 1.0, atol=1e-12)


def test_torque_free_energy_and_inertial_momentum():
    body = RigidBody(_asymmetric_inertia())
    q = np.array([1.0, 0.0, 0.0, 0.0])
    omega = np.array([0.3, -0.8, 0.2])
    dt = 0.002
    t_final = 8.0
    n = int(t_final / dt)
    E0 = body.kinetic_energy(omega)
    h0 = body.angular_momentum_inertial(q, omega)
    hb0 = np.linalg.norm(body.angular_momentum_body(omega))

    for _ in range(n):
        q, omega = step_rigid_body(body, q, omega, np.zeros(3), dt)

    E1 = body.kinetic_energy(omega)
    h1 = body.angular_momentum_inertial(q, omega)
    hb1 = np.linalg.norm(body.angular_momentum_body(omega))
    assert abs(E1 - E0) / E0 < 5e-5
    np.testing.assert_allclose(h1, h0, rtol=5e-5, atol=1e-7)
    assert abs(hb1 - hb0) / hb0 < 5e-5
    assert abs(np.linalg.norm(q) - 1.0) < 1e-12


def test_principal_axis_spin_is_equilibrium():
    body = RigidBody(np.diag([1.0, 2.0, 3.0]))
    q = np.array([1.0, 0.0, 0.0, 0.0])
    omega = np.array([0.0, 0.0, 1.2])
    for _ in range(200):
        q, omega = step_rigid_body(body, q, omega, np.zeros(3), 0.01)
    np.testing.assert_allclose(omega, [0.0, 0.0, 1.2], atol=1e-10)

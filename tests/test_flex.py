"""Hinged appendage plant: rigid reduction, modal frequency, energy decay."""

from __future__ import annotations

import numpy as np
import pytest

from attitude_sim.flex import (
    HingedAppendage,
    effective_hinge_inertia,
    flex_modal_frequency,
    linearized_flex_state_space,
    pack_flex_state,
    step_flex,
    step_flex_attitude,
    unit_hinge_axis,
    unpack_flex_state,
)
from attitude_sim.plant import RigidBody, step_rigid_body
from attitude_sim.quaternions import quat_normalize

_HUB_J = np.diag([0.05, 0.06, 0.07])
_TRIAXIAL = np.diag([1.0, 2.0, 3.0])
_IP = 0.02
_K = 4.0

# Classical RK4, not symplectic.  Flexural ω_n ~ 16 rad/s, so energy
# residuals sit above the rigid-body 1e-12 cases.  Observed ΔE/E ~ 1e-9
# at dt = 0.001 s, T = 4 s; bound sits ~10× above that.
ENERGY_REL_TOL = 1e-8
QUAT_NORM_ABS_TOL = 1e-12
RIGID_ATOL = 1e-13
FREQ_REL_TOL = 0.02
EIGEN_ATOL = 1e-9


def _plant(
    inertia: np.ndarray = _HUB_J,
    hinge_axis: np.ndarray | int | str = "z",
    panel_inertia: float = _IP,
    stiffness: float = _K,
    damping: float = 0.0,
) -> HingedAppendage:
    return HingedAppendage(
        inertia,
        hinge_axis=hinge_axis,
        panel_inertia=panel_inertia,
        stiffness=stiffness,
        damping=damping,
    )


def test_unit_hinge_axis_names_indices_and_vectors():
    np.testing.assert_allclose(unit_hinge_axis("z"), [0.0, 0.0, 1.0])
    np.testing.assert_allclose(unit_hinge_axis(0), [1.0, 0.0, 0.0])
    np.testing.assert_allclose(unit_hinge_axis([0.0, 2.0, 0.0]), [0.0, 1.0, 0.0])
    with pytest.raises(ValueError, match="non-zero"):
        unit_hinge_axis([0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="name"):
        unit_hinge_axis("w")
    with pytest.raises(ValueError, match="index"):
        unit_hinge_axis(3)


def test_effective_inertia_is_parallel_combination():
    i_h = 0.07
    i_eff = effective_hinge_inertia(_HUB_J, _IP, "z")
    expected = i_h * _IP / (i_h + _IP)
    np.testing.assert_allclose(i_eff, expected)
    plant = _plant()
    np.testing.assert_allclose(plant.effective_inertia(), expected)
    np.testing.assert_allclose(plant.hub_inertia_about_hinge(), i_h)
    with pytest.raises(ValueError, match="positive"):
        effective_hinge_inertia(_HUB_J, 0.0, "z")


def test_modal_frequency_matches_sqrt_k_over_i_eff():
    i_eff = effective_hinge_inertia(_HUB_J, _IP, "z")
    wn = flex_modal_frequency(_K, _HUB_J, _IP, "z")
    np.testing.assert_allclose(wn, np.sqrt(_K / i_eff))
    np.testing.assert_allclose(_plant().modal_frequency(), wn)


def test_theta_zero_at_rest_omega_dot_matches_rigid_hub():
    """At θ = θ̇ = 0, ω = 0 the hub equation reduces to Euler on J_h."""
    plant = _plant()
    body = plant.as_rigid_body()
    omega = np.zeros(3)
    tau = np.array([0.01, -0.02, 0.03])
    np.testing.assert_allclose(
        plant.omega_dot(omega, tau, 0.0, 0.0),
        body.omega_dot(omega, tau),
        atol=RIGID_ATOL,
    )
    # Instantaneous θ̈ is not zero: the panel lags the hub.
    assert plant.theta_ddot(omega, tau, 0.0, 0.0) != 0.0


def test_zero_panel_inertia_trajectory_matches_rigid():
    """I_p = 0 is identically the rigid hub; θ stays frozen."""
    body = RigidBody(_HUB_J)
    plant = _plant(panel_inertia=0.0, stiffness=1.0)
    q = quat_normalize([0.4, 0.3, 0.2, 0.8])
    omega = np.array([-0.5, 0.4, 0.7])
    tau = np.array([0.01, -0.02, 0.015])
    dt = 0.002
    q_r, w_r = q.copy(), omega.copy()
    q_f, w_f, th, td = q.copy(), omega.copy(), 0.0, 0.0
    for _ in range(250):
        q_r, w_r = step_rigid_body(body, q_r, w_r, tau, dt)
        q_f, w_f, th, td = step_flex_attitude(plant, q_f, w_f, th, td, tau, dt)
    np.testing.assert_allclose(w_f, w_r, atol=1e-13)
    np.testing.assert_allclose(q_f, q_r, atol=1e-13)
    assert th == 0.0
    assert td == 0.0


def test_theta_zero_principal_spin_orthogonal_to_hinge_stays_rigid():
    """Torque-free spin about a principal axis ⟂ â with θ = θ̇ = 0 follows RigidBody."""
    plant = _plant(inertia=_TRIAXIAL, hinge_axis="z", panel_inertia=0.5, stiffness=_K)
    body = plant.as_rigid_body()
    omega0 = np.array([1.2, 0.0, 0.0])
    q = np.array([1.0, 0.0, 0.0, 0.0])
    w_f, th, td = omega0.copy(), 0.0, 0.0
    q_r, w_r = q.copy(), omega0.copy()
    tau0 = np.zeros(3)
    np.testing.assert_allclose(plant.omega_dot(omega0, tau0, 0.0, 0.0), 0.0, atol=1e-15)
    assert abs(plant.theta_ddot(omega0, tau0, 0.0, 0.0)) < 1e-15
    for _ in range(400):
        q_r, w_r = step_rigid_body(body, q_r, w_r, tau0, 0.01)
        q, w_f, th, td = step_flex_attitude(plant, q, w_f, th, td, tau0, 0.01)
    np.testing.assert_allclose(w_f, omega0, atol=1e-12)
    np.testing.assert_allclose(w_r, omega0, atol=1e-12)
    assert abs(th) < 1e-12
    assert abs(td) < 1e-12


def test_linearized_eig_contains_modal_frequency():
    plant = _plant(damping=0.0)
    a_mat, b_mat = linearized_flex_state_space(plant)
    assert a_mat.shape == (5, 5)
    assert b_mat.shape == (5, 3)
    wn = plant.modal_frequency()
    eigs = np.linalg.eigvals(a_mat)
    imag = np.sort(np.abs(np.imag(eigs)))
    # Three rigid zeros (transverse ω and free–free rotation) plus ±j ω_n.
    assert imag[2] < EIGEN_ATOL
    np.testing.assert_allclose(imag[-1], wn, atol=EIGEN_ATOL)
    # Input matrix: hub inverse on the ω rows.
    np.testing.assert_allclose(b_mat[:3, :], plant.inertia_inv, atol=1e-15)


def test_free_oscillation_frequency_matches_sqrt_k_over_i_eff():
    """Small-θ free oscillation of the nonlinear plant ≈ √(k / I_eff)."""
    plant = _plant(damping=0.0)
    wn = plant.modal_frequency()
    period = 2.0 * np.pi / wn
    dt = 0.0005
    n = round(6.0 * period / dt)
    omega = np.zeros(3)
    theta, theta_dot = 0.04, 0.0
    t = 0.0
    zeros = []
    prev = theta
    for _ in range(n):
        omega, theta, theta_dot = step_flex(plant, omega, theta, theta_dot, np.zeros(3), dt)
        t += dt
        if prev > 0.0 >= theta or prev < 0.0 <= theta:
            zeros.append(t)
        prev = theta
    assert len(zeros) >= 8
    half_periods = np.diff(zeros)
    measured = 2.0 * float(np.mean(half_periods))
    assert abs(measured - period) / period < FREQ_REL_TOL
    # Amplitude stays O(θ0); not a rigid spin-up.
    assert abs(theta) < 0.06


def test_undamped_energy_conserved_torque_free():
    plant = _plant(damping=0.0)
    omega = np.array([0.05, -0.04, 0.03])
    theta, theta_dot = 0.08, -0.1
    e0 = plant.mechanical_energy(omega, theta, theta_dot)
    for _ in range(4000):
        omega, theta, theta_dot = step_flex(
            plant, omega, theta, theta_dot, np.zeros(3), 0.001
        )
    e1 = plant.mechanical_energy(omega, theta, theta_dot)
    assert abs(e1 - e0) / e0 < ENERGY_REL_TOL


def test_damping_decays_mechanical_energy():
    plant = _plant(damping=0.05)
    omega = np.array([0.02, 0.0, 0.01])
    theta, theta_dot = 0.1, 0.0
    e0 = plant.mechanical_energy(omega, theta, theta_dot)
    energies = [e0]
    for _ in range(2500):
        omega, theta, theta_dot = step_flex(
            plant, omega, theta, theta_dot, np.zeros(3), 0.002
        )
        energies.append(plant.mechanical_energy(omega, theta, theta_dot))
    e1 = energies[-1]
    assert e1 < 0.5 * e0
    # Dissipation: energy is essentially nonincreasing (RK4 wiggle << decay).
    diffs = np.diff(energies)
    assert float(np.sum(diffs[diffs > 0.0])) < 1e-6 * e0
    assert e1 > 0.0


def test_applied_torque_adds_work_undamped():
    plant = _plant(damping=0.0)
    omega = np.zeros(3)
    theta, theta_dot = 0.0, 0.0
    tau = np.array([0.0, 0.0, 0.02])
    e0 = plant.mechanical_energy(omega, theta, theta_dot)
    for _ in range(400):
        omega, theta, theta_dot = step_flex(plant, omega, theta, theta_dot, tau, 0.002)
    assert plant.mechanical_energy(omega, theta, theta_dot) > e0 + 1e-4


def test_attitude_step_renormalizes_quaternion():
    plant = _plant()
    q = quat_normalize([0.5, -0.2, 0.1, 0.8])
    omega = np.array([0.1, -0.05, 0.08])
    theta, theta_dot = 0.03, -0.02
    tau = np.array([0.001, 0.0, -0.001])
    for _ in range(200):
        q, omega, theta, theta_dot = step_flex_attitude(
            plant, q, omega, theta, theta_dot, tau, 0.01
        )
    assert abs(np.linalg.norm(q) - 1.0) < QUAT_NORM_ABS_TOL


def test_locked_inertia_and_lagrangian_mass_matrix_identity():
    """(I_h + I_p) ψ̈ + I_p θ̈ = τ_a and I_p (ψ̈ + θ̈) = −kθ at θ̇ = 0."""
    plant = _plant()
    i_h = plant.hub_inertia_about_hinge()
    i_p = plant.panel_inertia
    theta, theta_dot = 0.05, 0.0
    tau = np.array([0.0, 0.0, 0.01])
    omega = np.zeros(3)
    wdot, _td, tddot = plant.derivatives(omega, tau, theta, theta_dot)
    psi_ddot = float(wdot[2])
    lhs_psi = (i_h + i_p) * psi_ddot + i_p * tddot
    lhs_th = i_p * (psi_ddot + tddot)
    np.testing.assert_allclose(lhs_psi, tau[2], atol=1e-14)
    np.testing.assert_allclose(lhs_th, -plant.stiffness * theta, atol=1e-14)
    j_locked = plant.locked_inertia()
    np.testing.assert_allclose(j_locked[2, 2], i_h + i_p)
    np.testing.assert_allclose(j_locked[:2, :2], _HUB_J[:2, :2])


def test_pack_unpack_flex_state():
    omega = np.array([0.1, -0.2, 0.3])
    y = pack_flex_state(omega, 0.4, -0.5)
    w, th, td = unpack_flex_state(y)
    np.testing.assert_allclose(w, omega)
    assert th == 0.4
    assert td == -0.5
    with pytest.raises(ValueError, match="length 5"):
        unpack_flex_state(np.ones(4))


def test_rejects_bad_construction_and_nonfinite_state():
    with pytest.raises(ValueError, match="principal moments"):
        HingedAppendage(np.diag([1.0, 1.0, 3.0]))
    with pytest.raises(ValueError, match="panel_inertia"):
        HingedAppendage(_HUB_J, panel_inertia=-0.1)
    with pytest.raises(ValueError, match="stiffness"):
        HingedAppendage(_HUB_J, stiffness=-1.0)
    with pytest.raises(ValueError, match="damping"):
        HingedAppendage(_HUB_J, damping=-0.01)
    plant = _plant()
    np.testing.assert_allclose(plant.axis, [0.0, 0.0, 1.0])
    np.testing.assert_allclose(plant.inertia_inv, np.linalg.inv(_HUB_J))
    q = np.array([1.0, 0.0, 0.0, 0.0])
    omega = np.array([0.1, -0.2, 0.3])
    h_b = plant.angular_momentum_body(omega, 0.4)
    h_i = plant.angular_momentum_inertial(q, omega, 0.4)
    np.testing.assert_allclose(h_i, h_b)
    with pytest.raises(ValueError, match="dt must be positive"):
        step_flex(plant, np.zeros(3), 0.0, 0.0, np.zeros(3), 0.0)
    with pytest.raises(ValueError, match="finite"):
        unit_hinge_axis([np.nan, 0.0, 0.0])
    with pytest.raises(ValueError, match="finite"):
        plant.omega_dot(np.zeros(3), [np.nan, 0.0, 0.0], 0.0, 0.0)
    with pytest.raises(ValueError, match="finite"):
        plant.omega_dot([np.nan, 0.0, 0.0], np.zeros(3), 0.0, 0.0)
    with pytest.raises(ValueError, match="finite"):
        plant.mechanical_energy(np.zeros(3), np.inf, 0.0)


def test_linearized_a_matches_numeric_jacobian():
    plant = _plant(damping=0.03)
    a_mat, _b = plant.linearized_state_space()
    eps = 1e-7
    x0 = np.zeros(5)
    tau0 = np.zeros(3)
    numeric = np.zeros((5, 5))
    for i in range(5):
        xp = x0.copy()
        xm = x0.copy()
        xp[i] += eps
        xm[i] -= eps
        fp = pack_flex_state(*plant.derivatives(xp[:3], tau0, xp[3], xp[4]))
        fm = pack_flex_state(*plant.derivatives(xm[:3], tau0, xm[3], xm[4]))
        numeric[:, i] = (fp - fm) / (2.0 * eps)
    np.testing.assert_allclose(a_mat, numeric, atol=1e-6)

"""Magnetic torquer: τ = m × B, per-axis |m| sat, residual-dipole handoff."""

import numpy as np
import pytest

from attitude_sim.disturbances import ResidualDipoleTorque, magnetic_dipole_torque
from attitude_sim.magnetic import (
    MagneticTorquer,
    clip_dipole,
    dipole_from_torque,
    magnetic_torque,
    make_magnetic_torquer,
    parse_m_max,
    residual_dipole_handoff,
)
from attitude_sim.quaternions import axis_angle_to_quat


def test_magnetic_torque_direction_and_magnitude():
    m = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 2.0, 0.0])
    tau = magnetic_torque(m, b)
    np.testing.assert_allclose(tau, [0.0, 0.0, 2.0])
    # Right-hand: x̂ × ŷ = ẑ; |τ| = |m| |B| sinθ.
    assert np.linalg.norm(tau) == pytest.approx(2.0)
    np.testing.assert_allclose(np.dot(tau, m), 0.0, atol=1e-15)
    np.testing.assert_allclose(np.dot(tau, b), 0.0, atol=1e-15)
    # Parallel dipole produces no torque.
    np.testing.assert_allclose(magnetic_torque(m, 3.0 * m), 0.0)
    np.testing.assert_allclose(tau, magnetic_dipole_torque(m, b))


def test_clip_dipole_scalar_and_per_axis():
    m = np.array([3.0, -2.5, 0.2])
    np.testing.assert_allclose(clip_dipole(m, None), m)
    np.testing.assert_allclose(clip_dipole(m, 1.0), [1.0, -1.0, 0.2])
    np.testing.assert_allclose(clip_dipole(m, np.array([0.5, 4.0, 0.0])), [0.5, -2.5, 0.0])
    with pytest.raises(ValueError, match="non-negative"):
        clip_dipole(m, -0.1)


def test_dipole_from_torque_is_cross_product_law():
    b = np.array([0.0, 0.0, 4e-5])
    tau_cmd = np.array([1e-3, 2e-3, 5e-3])
    m = dipole_from_torque(tau_cmd, b)
    tau = magnetic_torque(m, b)
    # Realizes only the component of τ_cmd perpendicular to B (here, drop z).
    np.testing.assert_allclose(tau, [1e-3, 2e-3, 0.0], rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(dipole_from_torque(tau_cmd, np.zeros(3)), 0.0)


def test_magnetic_torquer_saturates_per_axis():
    mtq = make_magnetic_torquer(m_max=np.array([0.4, 0.2, 0.1]))
    b = np.array([0.0, 0.0, 3e-5])
    tau, m_app = mtq.apply_dipole(np.array([1.0, -0.8, 0.05]), b)
    np.testing.assert_allclose(m_app, [0.4, -0.2, 0.05])
    assert np.all(np.abs(m_app) <= np.array([0.4, 0.2, 0.1]) + 1e-15)
    np.testing.assert_allclose(tau, magnetic_torque(m_app, b))
    leftover = mtq.leftover_after_sat(np.array([1.0, -0.8, 0.05]))
    np.testing.assert_allclose(leftover, [0.6, -0.6, 0.0])


def test_apply_torque_never_exceeds_dipole_box():
    mtq = MagneticTorquer(m_max=0.25)
    rng = np.random.default_rng(4)
    for _ in range(40):
        tau_cmd = rng.normal(scale=0.01, size=3)
        b = rng.normal(scale=3e-5, size=3)
        tau, m = mtq.apply_torque(tau_cmd, b)
        assert np.all(np.abs(m) <= 0.25 + 1e-12)
        np.testing.assert_allclose(np.dot(tau, b), 0.0, atol=1e-18)


def test_residual_dipole_handoff_matches_disturbance_model():
    residual = np.array([0.08, -0.02, 0.01])
    mtq = MagneticTorquer(m_max=1.0, residual_m=residual)
    dist = mtq.residual_disturbance()
    assert isinstance(dist, ResidualDipoleTorque)
    np.testing.assert_allclose(dist.m_body, residual)
    handed = residual_dipole_handoff(residual, model="orbit_normal")
    assert handed.model == "orbit_normal"
    q = axis_angle_to_quat(np.array([0.0, 0.0, 1.0]), 0.4)
    from attitude_sim.disturbances import OrbitState

    orbit = OrbitState(
        r_eci=np.array([7.0e6, 0.0, 0.0]),
        v_eci=np.array([0.0, 7.5e3, 0.0]),
    )
    assert dist is not None
    tau_d = dist.tau_body(q, np.zeros(3), orbit=orbit)
    np.testing.assert_allclose(tau_d, magnetic_torque(residual, dist.B_body(q, orbit)))
    tau_h = handed.tau_body(q, np.zeros(3), orbit=orbit)
    np.testing.assert_allclose(tau_h, magnetic_torque(residual, handed.B_body(q, orbit)))
    assert MagneticTorquer().residual_disturbance() is None


def test_parse_m_max():
    assert parse_m_max(None) is None
    assert parse_m_max("0.5") == 0.5
    np.testing.assert_allclose(parse_m_max("0.5,0.2,0.1"), [0.5, 0.2, 0.1])

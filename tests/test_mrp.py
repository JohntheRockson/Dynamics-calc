"""MRP conversions, shadow-set switching, and kinematics."""

import numpy as np
import pytest

from attitude_sim.mrp import (
    DEFAULT_SHADOW_THRESHOLD,
    axis_angle_to_mrp,
    mrp_B,
    mrp_dcm,
    mrp_derivative,
    mrp_norm_sq,
    mrp_shadow,
    mrp_switch,
    mrp_to_quat,
    mrp_to_rotation,
    quat_to_mrp,
    rotation_to_mrp,
)
from attitude_sim.quaternions import (
    axis_angle_to_quat,
    geodesic_angle,
    quat_integrate_const_omega,
    quat_normalize,
    quat_to_rotation,
)


def _random_unit_quat(rng: np.random.Generator) -> np.ndarray:
    q = rng.normal(size=4)
    return quat_normalize(q)


def test_identity_is_zero_mrp():
    np.testing.assert_allclose(quat_to_mrp([1.0, 0.0, 0.0, 0.0]), 0.0, atol=1e-15)
    np.testing.assert_allclose(mrp_to_quat(np.zeros(3)), [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(mrp_to_rotation(np.zeros(3)), np.eye(3), atol=1e-15)


def test_negative_identity_quaternion_maps_to_zero_mrp():
    """q = [−1, 0] is the same attitude as identity; the chart stays finite."""
    np.testing.assert_allclose(quat_to_mrp([-1.0, 0.0, 0.0, 0.0]), 0.0, atol=1e-15)


def test_axis_angle_matches_tan_quarter_angle():
    axis = np.array([0.0, 0.0, 1.0])
    angle = 0.8
    sigma = axis_angle_to_mrp(axis, angle)
    np.testing.assert_allclose(sigma, [0.0, 0.0, np.tan(angle / 4.0)], atol=1e-15)
    q = axis_angle_to_quat(axis, angle)
    np.testing.assert_allclose(quat_to_mrp(q), sigma, atol=1e-12)


@pytest.mark.parametrize("angle", [0.0, 0.3, np.pi / 2, np.pi - 1e-3, -1.2])
def test_quat_mrp_roundtrip(angle):
    q = axis_angle_to_quat(np.array([1.0, -0.4, 0.7]), angle)
    sigma = quat_to_mrp(q)
    q2 = mrp_to_quat(sigma)
    # Same attitude: q and −q are identified.
    assert geodesic_angle(q, q2) < 1e-12
    np.testing.assert_allclose(quat_to_mrp(q2), sigma, atol=1e-12)


def test_quat_mrp_roundtrip_random():
    rng = np.random.default_rng(7)
    for _ in range(24):
        q = _random_unit_quat(rng)
        sigma = quat_to_mrp(q)
        q2 = mrp_to_quat(sigma)
        # arccos(1−ε) on the geodesic is ~√ε; compare DCMs for the attitude.
        np.testing.assert_allclose(mrp_to_rotation(sigma), quat_to_rotation(q), atol=1e-12)
        np.testing.assert_allclose(quat_to_rotation(q2), quat_to_rotation(q), atol=1e-12)
        assert geodesic_angle(q, q2) < 1e-7
        sigma2 = quat_to_mrp(q2)
        np.testing.assert_allclose(sigma2, sigma, atol=1e-12)


def test_mrp_dcm_matches_quaternion_path_and_is_proper():
    rng = np.random.default_rng(11)
    for _ in range(16):
        q = _random_unit_quat(rng)
        sigma = quat_to_mrp(q)
        R_q = quat_to_rotation(q)
        R_m = mrp_to_rotation(sigma)
        R_d = mrp_dcm(sigma)
        np.testing.assert_allclose(R_m, R_q, atol=1e-12)
        np.testing.assert_allclose(R_d, R_q, atol=1e-12)
        assert abs(np.linalg.det(R_d) - 1.0) < 1e-12
        np.testing.assert_allclose(R_d @ R_d.T, np.eye(3), atol=1e-12)


def test_rotation_mrp_roundtrip():
    q = axis_angle_to_quat(np.array([0.2, 0.5, -0.8]), 1.1)
    R = quat_to_rotation(q)
    sigma = rotation_to_mrp(R)
    np.testing.assert_allclose(mrp_to_rotation(sigma), R, atol=1e-12)
    # rotation_to_quat uses q_w ≥ 0, so the MRP is the principal set.
    assert float(np.linalg.norm(sigma)) <= 1.0 + 1e-12


def test_double_cover_is_shadow_set():
    q = quat_normalize([0.4, 0.3, -0.2, 0.8])
    sigma = quat_to_mrp(q)
    sigma_neg = quat_to_mrp(-q)
    np.testing.assert_allclose(sigma_neg, mrp_shadow(sigma), atol=1e-12)
    np.testing.assert_allclose(mrp_to_rotation(sigma), mrp_to_rotation(sigma_neg), atol=1e-12)
    # Shadow of the shadow is the original (away from the origin).
    np.testing.assert_allclose(mrp_shadow(sigma_neg), sigma, atol=1e-12)


def test_shadow_undefined_at_origin():
    with pytest.raises(ValueError, match="origin"):
        mrp_shadow(np.zeros(3))


def test_switch_when_norm_exceeds_one():
    # Principal angle 240° ⇒ ||σ|| = tan(60°) = √3 > 1.
    sigma = axis_angle_to_mrp(np.array([0.0, 1.0, 0.0]), 4.0 * np.pi / 3.0)
    assert float(np.linalg.norm(sigma)) > 1.0
    switched = mrp_switch(sigma)
    assert float(np.linalg.norm(switched)) < 1.0
    np.testing.assert_allclose(switched, mrp_shadow(sigma), atol=1e-12)
    np.testing.assert_allclose(mrp_to_rotation(switched), mrp_to_rotation(sigma), atol=1e-12)
    # Equality at the threshold does not switch.
    on_sphere = sigma / np.linalg.norm(sigma)  # ||σ|| = 1
    np.testing.assert_allclose(mrp_switch(on_sphere), on_sphere)
    # Below threshold is a no-op.
    small = 0.4 * on_sphere
    np.testing.assert_allclose(mrp_switch(small), small)


def test_switch_configurable_threshold():
    # ||σ|| · ||σ^s|| = 1, so a threshold of 1 (or any t ≥ 1) is the
    # geometrically useful "keep the small set" switch.
    sigma = np.array([0.0, 0.0, 1.5])
    np.testing.assert_allclose(mrp_switch(sigma, threshold=2.0), sigma)
    switched = mrp_switch(sigma, threshold=1.2)
    np.testing.assert_allclose(switched, mrp_shadow(sigma))
    assert float(np.linalg.norm(switched)) < 1.0
    np.testing.assert_allclose(np.linalg.norm(switched), 1.0 / 1.5)
    with pytest.raises(ValueError, match="threshold"):
        mrp_switch(sigma, threshold=0.0)
    with pytest.raises(ValueError, match="threshold"):
        mrp_switch(sigma, threshold=-1.0)


def test_quat_to_mrp_switch_flag_avoids_large_set():
    q = axis_angle_to_quat(np.array([1.0, 0.0, 0.0]), 4.0 * np.pi / 3.0)
    # Long-way cover: flip so q_w < 0 ⇒ ||σ|| > 1.
    if q[0] > 0.0:
        q = -q
    sigma = quat_to_mrp(q, switch=False)
    assert float(np.linalg.norm(sigma)) > DEFAULT_SHADOW_THRESHOLD
    sigma_s = quat_to_mrp(q, switch=True)
    assert float(np.linalg.norm(sigma_s)) <= DEFAULT_SHADOW_THRESHOLD
    np.testing.assert_allclose(mrp_to_rotation(sigma_s), quat_to_rotation(q), atol=1e-12)


def test_singularity_avoided_near_full_rotation():
    """A 359° rotation has huge tan(Φ/4); the shadow is a 1° opposite turn."""
    axis = np.array([0.3, -0.1, 0.9])
    axis = axis / np.linalg.norm(axis)
    angle = 2.0 * np.pi - np.deg2rad(1.0)
    sigma = axis_angle_to_mrp(axis, angle)
    assert float(np.linalg.norm(sigma)) > 10.0
    switched = mrp_switch(sigma)
    expected = axis_angle_to_mrp(axis, angle - 2.0 * np.pi)  # −1°
    np.testing.assert_allclose(switched, expected, atol=1e-10)
    q = axis_angle_to_quat(axis, angle)
    assert geodesic_angle(mrp_to_quat(switched), q) < 1e-10
    with pytest.raises(ValueError, match="singular"):
        axis_angle_to_mrp(axis, 2.0 * np.pi)


def test_180_degree_mrp_has_unit_norm():
    sigma = axis_angle_to_mrp(np.array([0.0, 0.0, 1.0]), np.pi)
    np.testing.assert_allclose(np.linalg.norm(sigma), 1.0, atol=1e-12)
    q = mrp_to_quat(sigma)
    np.testing.assert_allclose(abs(q[0]), 0.0, atol=1e-12)


def test_B_identity_and_small_angle_kinematics():
    B0 = mrp_B(np.zeros(3))
    np.testing.assert_allclose(B0, np.eye(3), atol=1e-15)
    omega = np.array([0.2, -0.4, 0.1])
    # σ ≈ Φ/4 near the origin, so σ̇ ≈ ω/4.
    np.testing.assert_allclose(mrp_derivative(np.zeros(3), omega), 0.25 * omega)


def test_B_times_B_T_identity():
    rng = np.random.default_rng(3)
    for _ in range(16):
        sigma = rng.normal(scale=0.4, size=3)
        B = mrp_B(sigma)
        n2 = mrp_norm_sq(sigma)
        np.testing.assert_allclose(B @ B.T, ((1.0 + n2) ** 2) * np.eye(3), atol=1e-12)


def test_kinematics_match_quaternion_finite_difference():
    q = quat_normalize([0.8, 0.2, -0.1, 0.4])
    omega = np.array([0.3, -0.2, 0.5])
    sigma = quat_to_mrp(q)
    dt = 1e-7
    q_next = quat_integrate_const_omega(q, omega, dt)
    sigma_next = quat_to_mrp(q_next)
    sdot_fd = (sigma_next - sigma) / dt
    sdot = mrp_derivative(sigma, omega)
    np.testing.assert_allclose(sdot_fd, sdot, atol=2e-6)


def test_kinematics_match_finite_difference_on_mrp_chart():
    """Forward-Euler on σ̇ stays close to converting an exact quaternion step."""
    q = axis_angle_to_quat(np.array([0.1, 0.7, 0.2]), 0.9)
    omega = np.array([-0.15, 0.4, 0.05])
    sigma = quat_to_mrp(q)
    dt = 1e-4
    sigma_euler = sigma + dt * mrp_derivative(sigma, omega)
    sigma_exact = quat_to_mrp(quat_integrate_const_omega(q, omega, dt))
    np.testing.assert_allclose(sigma_euler, sigma_exact, atol=5e-8)


def test_shadow_set_obeys_the_same_kinematics():
    q = quat_normalize([0.2, 0.5, -0.3, 0.7])
    omega = np.array([0.4, 0.1, -0.2])
    sigma = quat_to_mrp(q)
    sigma_s = mrp_shadow(sigma)
    dt = 1e-7
    q_next = quat_integrate_const_omega(q, omega, dt)
    # Stay on the same cover as −q so the shadow chart is continuous.
    sigma_s_next = quat_to_mrp(-q_next)
    sdot_fd = (sigma_s_next - sigma_s) / dt
    np.testing.assert_allclose(sdot_fd, mrp_derivative(sigma_s, omega), atol=5e-6)


def test_switch_during_large_principal_rotation_stays_bounded():
    """Constant-rate 270° slew: without switch ||σ||>1; with switch ||σ||≤1."""
    axis = np.array([0.0, 0.0, 1.0])
    omega = np.array([0.0, 0.0, 1.5])  # rad/s
    q = np.array([1.0, 0.0, 0.0, 0.0])
    t_final = 0.75 * np.pi / omega[2]  # 135° … then keep going to 270°
    dt = 0.01
    sigma = quat_to_mrp(q)
    n_steps = round((2.0 * t_final) / dt)
    max_raw = 0.0
    max_switched = 0.0
    for _ in range(n_steps):
        q = quat_integrate_const_omega(q, omega, dt)
        sigma = quat_to_mrp(q)
        max_raw = max(max_raw, float(np.linalg.norm(sigma)))
        switched = mrp_switch(sigma)
        max_switched = max(max_switched, float(np.linalg.norm(switched)))
        np.testing.assert_allclose(
            mrp_to_rotation(switched),
            quat_to_rotation(q),
            atol=1e-12,
        )
    assert max_raw > 1.0
    assert max_switched <= 1.0 + 1e-12
    # Closed form at 270°: tan(67.5°) > 1, shadow is tan(−22.5°).
    sigma_270 = axis_angle_to_mrp(axis, 1.5 * np.pi)
    assert float(np.linalg.norm(sigma_270)) > 1.0
    np.testing.assert_allclose(
        mrp_switch(sigma_270),
        axis_angle_to_mrp(axis, 1.5 * np.pi - 2.0 * np.pi),
        atol=1e-12,
    )


def test_mrp_rejects_non_finite():
    with pytest.raises(ValueError, match="finite"):
        mrp_to_quat([np.nan, 0.0, 0.0])
    with pytest.raises(ValueError, match="finite"):
        mrp_derivative(np.zeros(3), [np.inf, 0.0, 0.0])

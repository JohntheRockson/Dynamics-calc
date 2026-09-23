"""Gyro ARW/RRW discrete mapping and unit-vector sensor stubs."""

import numpy as np
import pytest

from attitude_sim.quaternions import axis_angle_to_quat, quat_to_rotation
from attitude_sim.sensors import (
    STAR_FOV_HALF_ANGLE,
    GyroModel,
    VectorSensor,
    arw_si,
    available_sensors,
    gyro_arw_std,
    gyro_rrw_std,
    in_fov,
    magnetometer,
    rrw_si,
    star_tracker,
    sun_sensor,
)


def test_arw_rrw_unit_conversion():
    # 0.3 deg/√hr is a typical MEMS ARW; SI is rad/s/√Hz.
    sv = arw_si(0.3)
    np.testing.assert_allclose(sv, np.deg2rad(0.3) / 60.0)
    su = rrw_si(0.02)
    np.testing.assert_allclose(su, np.deg2rad(0.02) / 60.0)
    gyro = GyroModel.from_allan(0.3, 0.02, seed=0)
    np.testing.assert_allclose(gyro.sigma_v, sv)
    np.testing.assert_allclose(gyro.sigma_u, su)


def test_discrete_arw_scales_as_inv_sqrt_dt():
    sv = 5e-4
    assert gyro_arw_std(sv, 0.01) == pytest.approx(sv / np.sqrt(0.01))
    assert gyro_arw_std(sv, 0.04) == pytest.approx(0.5 * gyro_arw_std(sv, 0.01))
    with pytest.raises(ValueError):
        gyro_arw_std(sv, 0.0)


def test_discrete_rrw_scales_as_sqrt_dt():
    su = 1e-5
    assert gyro_rrw_std(su, 0.01) == pytest.approx(su * np.sqrt(0.01))
    assert gyro_rrw_std(su, 0.04) == pytest.approx(2.0 * gyro_rrw_std(su, 0.01))


def test_gyro_arw_sample_std_matches_density():
    dt = 0.01
    sv = 2e-3
    gyro = GyroModel(sigma_v=sv, sigma_u=0.0, sigma_n=0.0, bias=np.zeros(3), seed=11)
    omega = np.array([0.1, -0.2, 0.05])
    samples = np.array([gyro.measure(omega, dt) - omega for _ in range(4000)])
    # Mean ≈ 0 (no bias walk); per-axis std ≈ σ_v / √dt.
    np.testing.assert_allclose(samples.mean(axis=0), 0.0, atol=4e-3)
    expected = gyro_arw_std(sv, dt)
    np.testing.assert_allclose(samples.std(axis=0), expected, rtol=0.08)


def test_gyro_rrw_bias_walk_std():
    dt = 0.05
    su = 3e-4
    n = 800
    increments = []
    for seed in range(n):
        g = GyroModel(sigma_v=0.0, sigma_u=su, sigma_n=0.0, bias=np.zeros(3), seed=1000 + seed)
        before = g.bias.copy()
        g.measure(np.zeros(3), dt)
        increments.append(g.bias - before)
    inc = np.array(increments)
    expected = gyro_rrw_std(su, dt)
    np.testing.assert_allclose(inc.std(axis=0), expected, rtol=0.12)


def test_gyro_readout_noise_is_dt_independent():
    sn = 1e-3
    g1 = GyroModel(sigma_v=0.0, sigma_u=0.0, sigma_n=sn, seed=2)
    g2 = GyroModel(sigma_v=0.0, sigma_u=0.0, sigma_n=sn, seed=2)
    # Same seed, different dt: readout-only draws are identical sequences.
    y1 = g1.measure(np.zeros(3), 0.01)
    y2 = g2.measure(np.zeros(3), 0.25)
    np.testing.assert_allclose(y1, y2)


def test_vector_sensor_body_from_inertial_convention():
    q = axis_angle_to_quat(np.array([0.0, 0.0, 1.0]), np.pi / 2)
    v_I = np.array([1.0, 0.0, 0.0])
    s = VectorSensor(v_inertial=v_I, sigma=0.0, seed=0)
    v_b = s.measure(q)
    expected = quat_to_rotation(q).T @ v_I
    np.testing.assert_allclose(v_b, expected, atol=1e-12)
    # +90° about z: inertial x is −body y.
    np.testing.assert_allclose(v_b, [0.0, -1.0, 0.0], atol=1e-12)


def test_magnetometer_and_sun_stubs_are_unit_vectors():
    q = axis_angle_to_quat(np.array([0.2, -0.4, 0.7]), 0.8)
    mag = magnetometer(sigma=0.0, seed=1)
    sun = sun_sensor(sigma=0.0, seed=2)
    assert mag.name == "mag"
    assert sun.name == "sun"
    vm = mag.measure(q)
    vs = sun.measure(q)
    np.testing.assert_allclose(np.linalg.norm(vm), 1.0)
    np.testing.assert_allclose(np.linalg.norm(vs), 1.0)
    # Distinct inertial references so TRIAD-like pairs are well conditioned.
    assert abs(np.dot(mag.v_inertial, sun.v_inertial)) < 0.95


def test_vector_sensor_hard_iron_bias_shifts_measurement():
    q = np.array([1.0, 0.0, 0.0, 0.0])
    v_I = np.array([0.0, 0.0, 1.0])
    clean = VectorSensor(v_inertial=v_I, sigma=0.0, seed=0)
    biased = VectorSensor(
        v_inertial=v_I,
        sigma=0.0,
        bias_body=np.array([0.2, 0.0, 0.0]),
        seed=0,
    )
    v0 = clean.measure(q)
    vb = biased.measure(q)
    assert abs(vb[0]) > abs(v0[0])
    np.testing.assert_allclose(np.linalg.norm(vb), 1.0)


def test_zero_inertial_reference_rejected():
    with pytest.raises(ValueError, match="inertial reference"):
        VectorSensor(v_inertial=np.zeros(3))


def test_sun_eclipse_flag_drops_measurement():
    q = np.array([1.0, 0.0, 0.0, 0.0])
    sun = sun_sensor(sigma=0.0, eclipse=True, seed=0)
    assert sun.occulted
    assert sun.measure(q) is None
    assert not sun.available(q)


def test_vector_sensor_fov_gate_uses_true_body_direction():
    sun = sun_sensor(
        v_inertial=np.array([1.0, 0.0, 0.0]),
        sigma=0.0,
        fov_half_angle=np.deg2rad(20.0),
        boresight_body=np.array([0.0, 0.0, 1.0]),
        seed=0,
    )
    q_id = np.array([1.0, 0.0, 0.0, 0.0])
    # Identity: sun is +x_B, 90° from +z boresight → out of 20° FOV.
    assert sun.measure(q_id) is None
    # Rotate so body +z points at inertial +x (the sun).
    q_in = axis_angle_to_quat(np.array([0.0, 1.0, 0.0]), np.pi / 2)
    v_b = sun.measure(q_in)
    assert v_b is not None
    np.testing.assert_allclose(v_b, [0.0, 0.0, 1.0], atol=1e-12)


def test_vectors_from_sensors_skips_occulted_and_out_of_fov():
    from attitude_sim.estimation import vectors_from_sensors

    q = np.array([1.0, 0.0, 0.0, 0.0])
    mag = magnetometer(sigma=0.0, seed=0)
    sun = sun_sensor(sigma=0.0, eclipse=True, seed=1)
    vecs = vectors_from_sensors(q, [mag, sun])
    assert len(vecs) == 1
    assert vecs[0][1] is mag.v_inertial or np.allclose(vecs[0][1], mag.v_inertial)


def test_fov_half_angle_and_zero_boresight_rejected():
    with pytest.raises(ValueError, match="fov_half_angle"):
        VectorSensor(v_inertial=np.array([1.0, 0.0, 0.0]), fov_half_angle=-0.1)
    with pytest.raises(ValueError, match="boresight"):
        VectorSensor(v_inertial=np.array([1.0, 0.0, 0.0]), boresight_body=np.zeros(3))


def test_in_fov_helper_and_available_sensors():
    z = np.array([0.0, 0.0, 1.0])
    assert in_fov(z, z, STAR_FOV_HALF_ANGLE)
    assert not in_fov(np.array([1.0, 0.0, 0.0]), z, STAR_FOV_HALF_ANGLE)
    assert in_fov(np.array([1.0, 0.0, 0.0]), z, None)
    q = np.array([1.0, 0.0, 0.0, 0.0])
    mag = magnetometer(sigma=0.0, seed=0)
    sun = sun_sensor(sigma=0.0, eclipse=True, seed=1)
    assert [s.name for s in available_sensors([mag, sun], q)] == ["mag"]


def test_star_tracker_fov_drops_out_of_cone():
    star = star_tracker(
        v_inertial=np.array([1.0, 0.0, 0.0]),
        sigma=0.0,
        boresight_body=np.array([0.0, 0.0, 1.0]),
        seed=0,
    )
    assert star.name == "star"
    assert star.fov_half_angle == pytest.approx(STAR_FOV_HALF_ANGLE)
    q_id = np.array([1.0, 0.0, 0.0, 0.0])
    # Identity: star is +x_B, 90° from +z boresight → gated.
    assert star.measure(q_id) is None
    assert not star.available(q_id)
    q_in = axis_angle_to_quat(np.array([0.0, 1.0, 0.0]), np.pi / 2)
    v_b = star.measure(q_in)
    assert v_b is not None
    np.testing.assert_allclose(v_b, [0.0, 0.0, 1.0], atol=1e-12)

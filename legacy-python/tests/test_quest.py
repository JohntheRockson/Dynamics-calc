"""QUEST / Davenport q-method: Wahba coarse attitude beside TRIAD."""

import numpy as np
import pytest

from attitude_sim.estimation import (
    attitude_profile_matrix,
    coarse_q0_from_sensors,
    davenport_K,
    davenport_q0_from_sensors,
    davenport_q_method,
    normalize_coarse_init_method,
    quest_attitude,
    quest_q0_from_sensors,
    triad_attitude,
)
from attitude_sim.quaternions import (
    axis_angle_to_quat,
    geodesic_angle,
    quat_to_rotation,
)
from attitude_sim.sensors import VectorSensor, magnetometer, sun_sensor


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3)
    return v / np.linalg.norm(v)


def _pairs_from_q(q: np.ndarray, inertial: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    R = quat_to_rotation(q)
    r = [_unit(v) for v in inertial]
    b = [_unit(R.T @ ri) for ri in r]
    return b, r


_V_MAG = np.array([0.3, 0.1, 0.95])
_V_SUN = np.array([1.0, 0.05, 0.02])
_V_STAR = np.array([0.1, 0.9, 0.25])


def test_normalize_coarse_init_method_aliases():
    assert normalize_coarse_init_method("TRIAD") == "triad"
    assert normalize_coarse_init_method("quest") == "quest"
    assert normalize_coarse_init_method("q-method") == "davenport"
    with pytest.raises(ValueError, match="unknown coarse-init method"):
        normalize_coarse_init_method("wahba-svd")


def test_quest_and_davenport_match_triad_two_noise_free():
    q_true = axis_angle_to_quat(np.array([0.3, -0.2, 0.8]), 0.9)
    body, inertial = _pairs_from_q(q_true, [_V_MAG, _V_SUN])
    q_triad = triad_attitude(body[0], inertial[0], body[1], inertial[1])
    q_quest = quest_attitude(body, inertial)
    q_dav = davenport_q_method(body, inertial)
    for q_hat in (q_triad, q_quest, q_dav):
        assert geodesic_angle(q_hat, q_true) < 1e-10
        assert abs(np.linalg.norm(q_hat) - 1.0) < 1e-12
        assert q_hat[0] >= -1e-15
        R = quat_to_rotation(q_hat)
        np.testing.assert_allclose(R @ body[0], inertial[0], atol=1e-12)
        np.testing.assert_allclose(R @ body[1], inertial[1], atol=1e-12)
    np.testing.assert_allclose(q_quest, q_dav, atol=1e-12)
    np.testing.assert_allclose(q_quest, q_triad, atol=1e-12)


def test_quest_three_vectors_recovers_true_attitude():
    q_true = axis_angle_to_quat(np.array([0.4, 0.2, -0.7]), 1.3)
    body, inertial = _pairs_from_q(q_true, [_V_MAG, _V_SUN, _V_STAR])
    q_quest = quest_attitude(body, inertial)
    q_dav = davenport_q_method(body, inertial)
    assert geodesic_angle(q_quest, q_true) < 1e-10
    assert geodesic_angle(q_dav, q_true) < 1e-10
    np.testing.assert_allclose(q_quest, q_dav, atol=1e-12)
    R = quat_to_rotation(q_quest)
    for b_i, r_i in zip(body, inertial, strict=True):
        np.testing.assert_allclose(R @ b_i, r_i, atol=1e-12)


def test_quest_double_cover_canonical_scalar():
    """q and −q are the same attitude; solvers return q_w ≥ 0."""
    q_pos = axis_angle_to_quat(np.array([0.1, 0.7, 0.2]), 1.1)
    q_neg = -q_pos
    assert q_pos[0] > 0.0
    assert q_neg[0] < 0.0
    body_pos, inertial_pos = _pairs_from_q(q_pos, [_V_MAG, _V_SUN, _V_STAR])
    body_neg, inertial_neg = _pairs_from_q(q_neg, [_V_MAG, _V_SUN, _V_STAR])
    # Same rotation matrix ⇒ same vector pairs.
    for b_p, b_n in zip(body_pos, body_neg, strict=True):
        np.testing.assert_allclose(b_p, b_n, atol=1e-15)
    for solver in (quest_attitude, davenport_q_method):
        q_hat = solver(body_pos, inertial_pos)
        q_from_neg = solver(body_neg, inertial_neg)
        assert q_hat[0] >= -1e-15
        assert q_from_neg[0] >= -1e-15
        assert geodesic_angle(q_hat, q_pos) < 1e-10
        assert geodesic_angle(q_hat, q_neg) < 1e-10
        np.testing.assert_allclose(q_hat, q_from_neg, atol=1e-12)


def test_quest_rejects_parallel_single_and_mismatched():
    v = np.array([1.0, 0.0, 0.0])
    w = np.array([0.0, 1.0, 0.0])
    with pytest.raises(ValueError, match="non-parallel"):
        quest_attitude([v, v], [v, v])
    with pytest.raises(ValueError, match="non-parallel"):
        davenport_q_method([v, -v], [w, -w])
    with pytest.raises(ValueError, match="at least two"):
        quest_attitude([v], [w])
    with pytest.raises(ValueError, match="same length"):
        quest_attitude([v, w], [v])
    with pytest.raises(ValueError, match="strictly positive"):
        quest_attitude([v, w], [v, w], weights=[1.0, 0.0])
    with pytest.raises(ValueError, match="weights length"):
        davenport_q_method([v, w], [v, w], weights=[1.0])


def test_quest_180_deg_falls_back_to_davenport():
    """Exact 180° makes the Rodrigues vector vanish; Davenport still works."""
    q_axis = np.array([0.0, 1.0, 0.0, 0.0])
    body, inertial = _pairs_from_q(q_axis, [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])])
    q_dav = davenport_q_method(body, inertial)
    assert geodesic_angle(q_dav, q_axis) < 1e-10
    with pytest.raises(ValueError, match="Rodrigues formula is singular"):
        quest_attitude(body, inertial, fallback=False)
    q_quest = quest_attitude(body, inertial, fallback=True)
    assert geodesic_angle(q_quest, q_axis) < 1e-10
    np.testing.assert_allclose(np.abs(q_quest), np.abs(q_dav), atol=1e-12)

    q_skew = axis_angle_to_quat(np.array([0.4, 0.5, 0.7]), np.pi)
    body_s, inertial_s = _pairs_from_q(q_skew, [_V_MAG, _V_SUN, _V_STAR])
    with pytest.raises(ValueError, match="Rodrigues formula is singular"):
        quest_attitude(body_s, inertial_s, fallback=False)
    assert geodesic_angle(quest_attitude(body_s, inertial_s), q_skew) < 1e-10


def test_quest_near_180_about_skew_axis_without_fallback():
    """Near-180° (not the exact q_w=0 singularity) stays on the Rodrigues path."""
    q_true = axis_angle_to_quat(np.array([0.4, 0.5, 0.7]), np.deg2rad(179.0))
    body, inertial = _pairs_from_q(q_true, [_V_MAG, _V_SUN, _V_STAR])
    q_quest = quest_attitude(body, inertial, fallback=False)
    q_dav = davenport_q_method(body, inertial)
    assert geodesic_angle(q_quest, q_true) < 1e-9
    assert geodesic_angle(q_dav, q_true) < 1e-9
    assert geodesic_angle(q_quest, q_dav) < 1e-9


def test_davenport_K_identity_largest_eigenpair():
    body, inertial = _pairs_from_q(np.array([1.0, 0.0, 0.0, 0.0]), [_V_MAG, _V_SUN])
    B = attitude_profile_matrix(body, inertial)
    K = davenport_K(B)
    np.testing.assert_allclose(K, K.T, atol=1e-15)
    evals, evecs = np.linalg.eigh(K)
    q = evecs[:, int(np.argmax(evals))]
    if q[0] < 0.0:
        q = -q
    np.testing.assert_allclose(np.abs(q), [1.0, 0.0, 0.0, 0.0], atol=1e-10)
    assert evals[int(np.argmax(evals))] == pytest.approx(float(np.sum([1.0, 1.0])), abs=1e-10)


def test_quest_weights_downweight_outlier_vs_triad():
    q_true = axis_angle_to_quat(np.array([0.2, 0.5, 0.8]), 0.7)
    body, inertial = _pairs_from_q(q_true, [_V_MAG, _V_SUN, _V_STAR])
    twist = quat_to_rotation(axis_angle_to_quat(np.array([0.0, 0.0, 1.0]), 0.45))
    body_bad = [body[0], body[1], _unit(twist.T @ body[2])]
    q_light = quest_attitude(body_bad, inertial, weights=[1.0, 1.0, 1e-6])
    q_heavy = quest_attitude(body_bad, inertial, weights=[1.0, 1.0, 1e4])
    q_triad = triad_attitude(body_bad[0], inertial[0], body_bad[1], inertial[1])
    assert geodesic_angle(q_light, q_true) < 1e-4
    assert geodesic_angle(q_light, q_true) < geodesic_angle(q_heavy, q_true)
    # TRIAD ignores the third pair, so it still recovers the two-vector solution.
    assert geodesic_angle(q_triad, q_true) < 1e-10
    # A precise third vector plus two noisy primaries should beat TRIAD.
    rng = np.random.default_rng(11)
    body_noisy = [
        _unit(body[0] + 0.25 * rng.standard_normal(3)),
        _unit(body[1] + 0.25 * rng.standard_normal(3)),
        body[2],
    ]
    q_triad_n = triad_attitude(body_noisy[0], inertial[0], body_noisy[1], inertial[1])
    q_quest_n = quest_attitude(body_noisy, inertial, weights=[1.0, 1.0, 1e4])
    assert geodesic_angle(q_quest_n, q_true) < geodesic_angle(q_triad_n, q_true)
    assert geodesic_angle(q_quest_n, q_true) < np.deg2rad(2.0)


def test_quest_q0_from_noisy_sensors_is_coarse():
    q_true = axis_angle_to_quat(np.array([0.1, 0.7, 0.2]), 1.1)
    mag = magnetometer(sigma=3e-3, seed=21)
    sun = sun_sensor(sigma=2e-3, seed=22)
    q_hat = quest_q0_from_sensors(q_true, [mag, sun])
    err = geodesic_angle(q_hat, q_true)
    assert err > 0.0
    assert err < np.deg2rad(5.0)
    q_dav = davenport_q0_from_sensors(q_true, [mag, sun])
    assert geodesic_angle(q_dav, q_true) < np.deg2rad(5.0)


def test_quest_q0_uses_third_available_sensor():
    q_true = axis_angle_to_quat(np.array([0.5, -0.1, 0.3]), 0.8)
    mag = magnetometer(sigma=0.0, seed=0)
    sun = sun_sensor(sigma=0.0, seed=1)
    star = VectorSensor(v_inertial=_V_STAR, sigma=0.0, seed=2, name="star")
    q_hat = quest_q0_from_sensors(q_true, [mag, sun, star])
    assert geodesic_angle(q_hat, q_true) < 1e-10
    dispatched = coarse_q0_from_sensors(q_true, [mag, sun, star], method="davenport")
    assert geodesic_angle(dispatched, q_true) < 1e-10


def test_quest_q0_needs_two_available_sensors():
    q = np.array([1.0, 0.0, 0.0, 0.0])
    mag = magnetometer(sigma=0.0, seed=0)
    sun = sun_sensor(sigma=0.0, eclipse=True, seed=1)
    with pytest.raises(ValueError, match="two available"):
        quest_q0_from_sensors(q, [mag, sun])
    with pytest.raises(ValueError, match="two available"):
        davenport_q0_from_sensors(q, [mag])
    with pytest.raises(ValueError, match="two available"):
        coarse_q0_from_sensors(q, [mag], method="quest")

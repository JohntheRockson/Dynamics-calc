"""TRIAD coarse attitude: known geometry plus SimLab lost-in-space wiring."""

import numpy as np
import pytest

from attitude_sim.estimation import triad, triad_from_meas
from attitude_sim.quaternions import (
    axis_angle_to_quat,
    geodesic_angle,
    quat_to_rotation,
)
from attitude_sim.sim import SimConfig, make_sim_estimator, run_slew


def _body_of(q, v_I):
    v_I = np.asarray(v_I, dtype=float)
    v_I = v_I / np.linalg.norm(v_I)
    return quat_to_rotation(q).T @ v_I, v_I


def test_triad_known_90deg_about_z():
    """90° about +z: e1_I maps to −e2_B; e3 is shared."""
    q_true = axis_angle_to_quat(np.array([0.0, 0.0, 1.0]), 0.5 * np.pi)
    v1_b, v1_I = _body_of(q_true, np.array([1.0, 0.0, 0.0]))
    v2_b, v2_I = _body_of(q_true, np.array([0.0, 0.0, 1.0]))
    np.testing.assert_allclose(v1_b, [0.0, -1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(v2_b, [0.0, 0.0, 1.0], atol=1e-12)
    q_hat = triad(v1_b, v1_I, v2_b, v2_I)
    assert geodesic_angle(q_hat, q_true) < 1e-12
    R = quat_to_rotation(q_hat)
    np.testing.assert_allclose(R @ v1_b, v1_I, atol=1e-12)
    np.testing.assert_allclose(R @ v2_b, v2_I, atol=1e-12)


def test_triad_recovers_random_attitude_noiseless():
    rng = np.random.default_rng(4)
    axis = rng.normal(size=3)
    q_true = axis_angle_to_quat(axis, 1.1)
    v1_I = np.array([0.3, 0.1, 0.95])
    v2_I = np.array([1.0, 0.05, 0.02])
    v1_b, v1_I = _body_of(q_true, v1_I)
    v2_b, v2_I = _body_of(q_true, v2_I)
    q_hat = triad(v1_b, v1_I, v2_b, v2_I)
    assert geodesic_angle(q_hat, q_true) < 1e-12
    # Swapping primary/secondary is still an exact DCM with noiseless pairs.
    q_swap = triad(v2_b, v2_I, v1_b, v1_I)
    assert geodesic_angle(q_swap, q_true) < 1e-12


def test_triad_from_meas_prefers_lower_sigma_primary():
    q_true = axis_angle_to_quat(np.array([0.2, 0.5, 0.8]), 0.7)
    v_mag_b, v_mag_I = _body_of(q_true, np.array([0.3, 0.1, 0.95]))
    v_sun_b, v_sun_I = _body_of(q_true, np.array([1.0, 0.05, 0.02]))
    q_hat = triad_from_meas(
        [
            (v_mag_b, v_mag_I, 3e-3),
            (v_sun_b, v_sun_I, 2e-3),
        ]
    )
    assert geodesic_angle(q_hat, q_true) < 1e-12
    with pytest.raises(ValueError, match="two non-parallel"):
        triad_from_meas([(v_mag_b, v_mag_I, 3e-3)])


def test_triad_rejects_parallel_and_zero_vectors():
    v = np.array([0.0, 0.0, 1.0])
    w = np.array([1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="two non-parallel"):
        triad(v, v, v, v)
    with pytest.raises(ValueError, match="two non-parallel"):
        triad(v, w, -v, -w)
    with pytest.raises(ValueError, match="non-zero"):
        triad(np.zeros(3), w, v, np.array([0.0, 1.0, 0.0]))


def test_noisy_triad_stays_within_a_few_degrees():
    q_true = axis_angle_to_quat(np.array([0.4, -0.2, 0.9]), 0.9)
    rng = np.random.default_rng(0)
    v1_I = np.array([0.3, 0.1, 0.95])
    v2_I = np.array([1.0, 0.05, 0.02])
    v1_b, v1_I = _body_of(q_true, v1_I)
    v2_b, v2_I = _body_of(q_true, v2_I)
    v1_b = v1_b + 3e-3 * rng.standard_normal(3)
    v2_b = v2_b + 2e-3 * rng.standard_normal(3)
    q_hat = triad(v1_b, v1_I, v2_b, v2_I)
    assert geodesic_angle(q_hat, q_true) < np.deg2rad(2.0)


def test_simlab_triad_init_does_not_pass_true_q0(monkeypatch):
    captured: dict[str, np.ndarray] = {}
    orig = make_sim_estimator

    def wrap(cfg, q0):
        captured["q0"] = np.asarray(q0, dtype=float).copy()
        return orig(cfg, q0)

    monkeypatch.setattr("attitude_sim.sim.make_sim_estimator", wrap)
    q_true = axis_angle_to_quat(np.array([0.3, -0.5, 0.8]), np.deg2rad(55.0))
    run_slew(
        SimConfig(
            dt=0.01,
            t_final=0.02,
            estimator="mekf",
            init_from_triad=True,
            q0=q_true,
            plot=False,
            gif=False,
            seed=3,
        )
    )
    q_est0 = captured["q0"]
    assert geodesic_angle(q_est0, q_true) < np.deg2rad(5.0)
    assert geodesic_angle(q_est0, np.array([1.0, 0.0, 0.0, 0.0])) > np.deg2rad(20.0)
    # Noisy TRIAD is not bit-identical to the plant quaternion.
    assert not np.allclose(q_est0, q_true, atol=1e-12)


def test_init_from_triad_rejects_truth_and_single_sensor():
    q0 = axis_angle_to_quat(np.array([0.0, 0.0, 1.0]), 0.4)
    cfg = SimConfig(
        estimator="truth",
        init_from_triad=True,
        q0=q0,
        t_final=0.02,
        plot=False,
        gif=False,
    )
    with pytest.raises(ValueError, match="mekf"):
        run_slew(cfg)
    with pytest.raises(ValueError, match="two vector sensors"):
        run_slew(
            SimConfig(
                estimator="mekf",
                init_from_triad=True,
                use_sun=False,
                t_final=0.02,
                plot=False,
                gif=False,
            )
        )

"""MEKF measurement NIS / innovation-whiteness (rank-2 unit-vector updates)."""

from pathlib import Path

import numpy as np
import pytest

from attitude_sim.controls import make_controller
from attitude_sim.estimation import (
    MEKF_NEES_DOF,
    MEKF_VECTOR_R_NUGGET,
    VECTOR_NIS_DOF,
    InnovationLog,
    InnovationSample,
    MultiplicativeEKF,
    _inject_body_error,
    chi2_mean_nis_bounds,
    chi2_two_sided_bounds,
    iter_vector_meas,
    lag1_sample_correlation,
    lag1_whiteness_bound,
    mekf_predicted_body_vector,
    mekf_vector_HR,
    mekf_vector_innovation_stats,
    mekf_vector_nis,
    nees,
    nis,
    tangent_plane_basis,
    vectors_from_sensors,
)
from attitude_sim.plant import RigidBody, step_rigid_body
from attitude_sim.quaternions import axis_angle_to_quat, quat_integrate_const_omega, skew
from attitude_sim.sensors import GyroModel, magnetometer, sun_sensor
from attitude_sim.sim import default_inertia


def test_nis_is_the_same_quadratic_form_as_nees():
    x = np.array([0.5, -1.0, 2.0])
    P = np.diag([2.0, 0.5, 4.0])
    assert nis(x, P) == pytest.approx(nees(x, P))
    assert nis(x, P) == pytest.approx(0.5**2 / 2.0 + 1.0**2 / 0.5 + 2.0**2 / 4.0)


def test_chi2_two_sided_bounds_nees_and_nis_docs():
    lo6, hi6 = chi2_two_sided_bounds(MEKF_NEES_DOF, alpha=0.01)
    assert lo6 == pytest.approx(0.6757, abs=1e-3)
    assert hi6 == pytest.approx(18.5476, abs=1e-3)
    lo2, hi2 = chi2_two_sided_bounds(VECTOR_NIS_DOF, alpha=0.05)
    assert lo2 == pytest.approx(0.0506356, abs=1e-4)
    assert hi2 == pytest.approx(7.37776, abs=1e-4)
    lo99, hi99 = chi2_two_sided_bounds(VECTOR_NIS_DOF, alpha=0.01)
    assert lo99 == pytest.approx(0.010025, abs=1e-4)
    assert hi99 == pytest.approx(10.5966, abs=1e-3)
    with pytest.raises(ValueError):
        chi2_two_sided_bounds(0)
    with pytest.raises(ValueError):
        chi2_two_sided_bounds(2, alpha=1.0)


def test_chi2_mean_nis_bounds_two_dof():
    lo, hi = chi2_mean_nis_bounds(VECTOR_NIS_DOF, 32, alpha=0.01)
    # Mean of 32 i.i.d. χ²_2 samples: nν = 64, 99% ≈ [1.21, 3.03].
    assert lo == pytest.approx(1.207, abs=0.01)
    assert hi == pytest.approx(3.027, abs=0.01)
    from attitude_sim.estimation import chi2_mean_nees_bounds

    assert chi2_mean_nis_bounds is chi2_mean_nees_bounds


def test_tangent_plane_basis_orthonormal_and_orthogonal_to_v():
    rng = np.random.default_rng(4)
    vectors = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([-0.2, 0.5, 0.8]),
        rng.normal(size=3),
    ]
    for v in vectors:
        B = tangent_plane_basis(v)
        assert B.shape == (3, 2)
        np.testing.assert_allclose(B.T @ B, np.eye(2), atol=1e-12)
        u = v / np.linalg.norm(v)
        np.testing.assert_allclose(B.T @ u, 0.0, atol=1e-12)
        framed = np.column_stack((u, B))
        assert abs(abs(np.linalg.det(framed)) - 1.0) < 1e-12


def test_mekf_vector_HR_matches_rank2_model():
    v_hat = np.array([0.0, 0.0, 1.0])
    sigma = 2e-3
    H, R = mekf_vector_HR(v_hat, sigma)
    assert H.shape == (3, 6)
    np.testing.assert_allclose(H[:, 3:], 0.0)
    np.testing.assert_allclose(H[:3, :3], skew(v_hat))
    sig2 = sigma**2
    np.testing.assert_allclose(v_hat @ R @ v_hat, MEKF_VECTOR_R_NUGGET * max(sig2, 1.0), atol=1e-18)
    tangent = np.array([1.0, 0.0, 0.0])
    assert tangent @ R @ tangent == pytest.approx(sig2 + MEKF_VECTOR_R_NUGGET * max(sig2, 1.0))


def test_mekf_nis_honest_tangent_innovation_matches_chi2():
    """ν₂ ~ N(0, S₂) ⇒ NIS = ν₂ᵀ S₂⁻¹ ν₂ ~ χ²_2."""
    n = 32
    rng = np.random.default_rng(1)
    S2 = np.diag([4e-6, 9e-6])
    values = [nis(rng.multivariate_normal(np.zeros(2), S2), S2) for _ in range(n)]
    mean = float(np.mean(values))
    lo, hi = chi2_mean_nis_bounds(VECTOR_NIS_DOF, n, alpha=0.01)
    assert lo < mean < hi, f"honest NIS ANEES={mean:.3f} not in [{lo:.3f}, {hi:.3f}]"


def test_mekf_vector_nis_matches_logged_preupdate_s():
    filt = MultiplicativeEKF()
    log = filt.enable_innovation_log()
    assert filt.enable_innovation_log() is log
    v_I = np.array([0.0, 0.0, 1.0])
    v_b = np.array([0.02, -0.01, 0.999])
    v_b = v_b / np.linalg.norm(v_b)
    P_before = filt.P.copy()
    expected = mekf_vector_nis(v_b, v_I, filt.q, P_before, 2e-3)
    stats = mekf_vector_innovation_stats(v_b, v_I, filt.q, P_before, 2e-3, sensor="mag")
    assert stats.nis == pytest.approx(expected)
    assert stats.dof == VECTOR_NIS_DOF
    np.testing.assert_allclose(stats.v_hat, mekf_predicted_body_vector(filt.q, v_I))
    filt.update_vector(v_b, v_I, 2e-3, sensor="mag", t=0.01)
    assert len(log.samples) == 1
    assert log.samples[0].sensor == "mag"
    assert log.samples[0].t == pytest.approx(0.01)
    assert log.samples[0].nis == pytest.approx(expected, rel=1e-12)


def test_iter_vector_meas_optional_name_and_rejects_bad_arity():
    v_b = np.array([1.0, 0.0, 0.0])
    v_I = np.array([0.0, 1.0, 0.0])
    two = iter_vector_meas([(v_b, v_I)])
    assert two[0][2] is None
    assert two[0][3] == "vector"
    named = iter_vector_meas([(v_b, v_I, 0.01, "mag")])
    assert named[0][2] == pytest.approx(0.01)
    assert named[0][3] == "mag"
    assert iter_vector_meas(None) == []
    with pytest.raises(ValueError, match="vector meas must be"):
        iter_vector_meas([(v_b,)])  # type: ignore[list-item]


def test_vectors_from_sensors_forwards_name():
    q = np.array([1.0, 0.0, 0.0, 0.0])
    mag = magnetometer(sigma=0.0, seed=0)
    sun = sun_sensor(sigma=0.0, seed=1)
    vecs = vectors_from_sensors(q, [mag, sun])
    assert [item[3] for item in vecs] == ["mag", "sun"]


def test_lag1_whiteness_bound_and_white_sequence():
    bound = lag1_whiteness_bound(101, alpha=0.05)
    assert bound == pytest.approx(1.9599639845 / 10.0, rel=1e-9)
    rng = np.random.default_rng(0)
    e = rng.standard_normal((400, 2))
    r = lag1_sample_correlation(e)
    assert r.shape == (2,)
    # Ensemble-of-one: 99.9% scalar bound is ~3.3/√399 ≈ 0.17; stay well under 0.25.
    assert np.all(np.abs(r) < 0.25)
    r1 = lag1_sample_correlation(e[:, 0])
    assert r1.shape == (1,)
    with pytest.raises(ValueError):
        lag1_sample_correlation(np.zeros((2, 2)))
    with pytest.raises(ValueError):
        lag1_whiteness_bound(2)
    with pytest.raises(ValueError):
        lag1_sample_correlation(np.zeros((4, 2, 2)))


def test_innovation_log_csv_summary_and_empty_guards(tmp_path: Path):
    log = InnovationLog()
    with pytest.raises(ValueError, match="empty"):
        log.mean_nis()
    with pytest.raises(ValueError, match="empty"):
        log.whitened_sequence()
    dest = log.write_csv(tmp_path / "empty.csv")
    text = dest.read_text()
    assert text.splitlines()[0].startswith("index,t,sensor,nis")
    assert log.summary()["n"] == 0

    filt = MultiplicativeEKF()
    filt.innovation_log = log
    v_I = np.array([0.0, 0.0, 1.0])
    v_b = np.array([0.0, 0.1, np.sqrt(1.0 - 0.01)])
    filt.update_vector(v_b, v_I, 2e-3, sensor="mag", t=0.02)
    filt.update_vector(v_b, np.array([1.0, 0.0, 0.0]), 2e-3, sensor="sun", t=0.02)
    assert log.summary()["n"] == 2
    assert log.mean_nis() == pytest.approx(np.mean(log.nis_values()))
    path = log.write_csv(tmp_path / "nis.csv")
    rows = path.read_text().strip().splitlines()
    assert len(rows) == 3
    assert "mag" in rows[1] and "sun" in rows[2]
    w = log.whitened_sequence("mag")
    assert w.shape == (1, 2)
    log.clear()
    assert log.samples == []


def test_innovation_sample_whitened_eigh_fallback():
    sample = InnovationSample(
        nis=0.0,
        dof=2,
        nu=np.zeros(3),
        nu_tangent=np.array([1.0, 0.0]),
        S_tangent=np.zeros((2, 2)),
    )
    w = sample.whitened()
    assert w.shape == (2,)
    assert np.all(np.isfinite(w))


def _mekf_matched_innovation_trial(
    seed: int,
    n_steps: int = 150,
    dt: float = 0.02,
    filter_vector_scale: float = 1.0,
) -> InnovationLog:
    """Matched Farrenkopf + mag/sun trial; optional filter-R scale for power tests."""
    rng = np.random.default_rng(seed)
    q_true = axis_angle_to_quat(rng.normal(size=3), 0.4)
    omega = np.array([0.03, -0.02, 0.025])
    bias0 = np.array([0.002, -0.001, 0.0015])
    sigma_v, sigma_u = 5e-4, 1e-6
    P0 = np.diag([3e-3, 3e-3, 3e-3, 1e-5, 1e-5, 1e-5])
    x0 = rng.multivariate_normal(np.zeros(6), P0)
    filt = MultiplicativeEKF(
        q=_inject_body_error(q_true, -x0[:3]),
        bias=bias0 - x0[3:],
        P=P0.copy(),
        sigma_v=sigma_v,
        sigma_u=sigma_u,
    )
    filt.enable_innovation_log()
    gyro = GyroModel(sigma_v=sigma_v, sigma_u=sigma_u, bias=bias0.copy(), seed=rng)
    mag = magnetometer(sigma=3e-3, seed=rng)
    sun = sun_sensor(sigma=2e-3, seed=rng)
    for k in range(n_steps):
        omega_m = gyro.measure(omega, dt)
        q_true = quat_integrate_const_omega(q_true, omega, dt)
        vecs = vectors_from_sensors(q_true, [mag, sun])
        if filter_vector_scale != 1.0:
            vecs = [
                (vb, vI, float(sig) * filter_vector_scale, name) for vb, vI, sig, name in vecs
            ]
        filt.step(omega_m, dt, vecs, t=(k + 1) * dt)
    assert filt.innovation_log is not None
    return filt.innovation_log


def test_mekf_nis_matched_synthetic_is_consistent():
    """Last-step mag+sun NIS ensemble sits in 99% χ²_2 mean bounds.

    Same matched open-loop assumptions as the NEES suite (constant-rate
    kinematics, Farrenkopf gyro matching Qd, mag/sun stubs matching
    tangent-plane R, honest P0).  Each trial contributes the *last* mag
    and sun NIS (pre-Joseph, rank-2).  E[NIS] = 2.
    """
    n = 32
    last = []
    means = []
    lag1 = []
    whitened_means = []
    for i in range(n):
        log = _mekf_matched_innovation_trial(seed=4000 + i)
        mag = log.samples_for("mag")
        sun = log.samples_for("sun")
        assert mag and sun
        last.append(mag[-1].nis)
        last.append(sun[-1].nis)
        means.append(log.mean_nis())
        lag1.append(log.lag1_correlation("mag"))
        whitened_means.append(log.whitened_sequence("mag").mean(axis=0))
    last_arr = np.asarray(last, dtype=float)
    assert np.all(np.isfinite(last_arr))
    mean = float(last_arr.mean())
    lo, hi = chi2_mean_nis_bounds(VECTOR_NIS_DOF, last_arr.size, alpha=0.01)
    single_lo, single_hi = chi2_two_sided_bounds(VECTOR_NIS_DOF, alpha=0.01)
    assert last_arr.min() > 0.0
    assert last_arr.max() < 8.0 * single_hi
    assert lo < mean < hi, (
        f"matched mean NIS={mean:.3f} not in 99% χ²_{VECTOR_NIS_DOF} mean bounds "
        f"[{lo:.3f}, {hi:.3f}] (N={last_arr.size}; single-trial 99% [{single_lo:.3f}, {single_hi:.3f}])"
    )
    # Time-averaged NIS (correlated in a trial) should still sit near 2.
    ta = float(np.mean(means))
    assert 1.2 < ta < 3.2, f"time-average NIS={ta:.3f} drifted off 2"
    r = np.vstack(lag1)
    assert np.all(np.abs(r.mean(axis=0)) < 0.15), f"ensemble lag-1={r.mean(axis=0)}"
    wm = np.vstack(whitened_means)
    assert np.all(np.abs(wm.mean(axis=0)) < 0.2)


def test_mekf_nis_detects_mismatched_r():
    """Too-small filter R inflates NIS; too-large R deflates it (test power)."""
    n = 16
    tight = np.array(
        [
            _mekf_matched_innovation_trial(seed=5000 + i, n_steps=80, filter_vector_scale=0.1)
            .samples_for("mag")[-1]
            .nis
            for i in range(n)
        ]
    )
    loose = np.array(
        [
            _mekf_matched_innovation_trial(seed=5100 + i, n_steps=80, filter_vector_scale=10.0)
            .samples_for("mag")[-1]
            .nis
            for i in range(n)
        ]
    )
    lo, hi = chi2_mean_nis_bounds(VECTOR_NIS_DOF, n, alpha=0.01)
    assert float(tight.mean()) > hi, f"tight-R mean NIS={tight.mean():.3f} failed to exceed {hi:.3f}"
    assert float(loose.mean()) < lo, f"loose-R mean NIS={loose.mean():.3f} failed to fall below {lo:.3f}"


def _mekf_closed_loop_nis_trial(seed: int, n_steps: int = 120, dt: float = 0.02) -> float:
    rng = np.random.default_rng(seed)
    body = RigidBody(default_inertia())
    ctrl = make_controller("pid", default_inertia(), torque_limit=0.02)
    ctrl.reset()
    q_true = axis_angle_to_quat(rng.normal(size=3), 0.25)
    omega = np.array([0.02, -0.015, 0.01])
    q_des = axis_angle_to_quat(np.array([0.2, 0.5, 0.84]), np.deg2rad(20.0))
    bias0 = np.array([0.002, -0.001, 0.0015])
    P0 = np.diag([3e-3, 3e-3, 3e-3, 1e-5, 1e-5, 1e-5])
    x0 = rng.multivariate_normal(np.zeros(6), P0)
    filt = MultiplicativeEKF(
        q=_inject_body_error(q_true, -x0[:3]),
        bias=bias0 - x0[3:],
        P=P0.copy(),
        sigma_v=5e-4,
        sigma_u=1e-6,
    )
    log = filt.enable_innovation_log()
    gyro = GyroModel(sigma_v=5e-4, sigma_u=1e-6, bias=bias0.copy(), seed=rng)
    mag = magnetometer(sigma=3e-3, seed=rng)
    sun = sun_sensor(sigma=2e-3, seed=rng)
    for _ in range(n_steps):
        omega_m = gyro.measure(omega, dt)
        vecs = vectors_from_sensors(q_true, [mag, sun])
        q_hat, omega_hat = filt.step(omega_m, dt, vecs)
        tau = ctrl.command(q_hat, omega_hat, q_des, omega_des=None, dt=dt)
        q_true, omega = step_rigid_body(body, q_true, omega, tau, dt)
    return log.mean_nis()


def test_mekf_nis_closed_loop_slew_is_finite_and_bounded():
    """Closed-loop mean NIS is a smoke gate, not matched χ²_2 consistency."""
    n = 8
    values = np.array([_mekf_closed_loop_nis_trial(seed=6000 + i) for i in range(n)])
    assert np.all(np.isfinite(values))
    assert values.min() > 0.0
    mean = float(values.mean())
    lo, hi = chi2_mean_nis_bounds(VECTOR_NIS_DOF, n, alpha=0.01)
    assert mean < 40.0, (
        f"closed-loop mean NIS={mean:.3f} exploded (matched 99% mean would be "
        f"[{lo:.3f}, {hi:.3f}]; this smoke allows up to 40)"
    )
    assert values.max() < 80.0

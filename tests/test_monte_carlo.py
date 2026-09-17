"""Smoke tests for the closed-loop Monte Carlo harness (tiny N)."""

import numpy as np

from attitude_sim.monte_carlo import (
    FAIL_DIVERGED,
    FAIL_NAN,
    FAIL_NON_UNIT,
    MonteCarloConfig,
    build_parser,
    classify_failure,
    log_uniform,
    main,
    perturb_inertia,
    run_monte_carlo,
    sample_bounded_attitude,
    settle_time_s,
)
from attitude_sim.plant import validate_inertia
from attitude_sim.quaternions import geodesic_angle
from attitude_sim.sim import SimLog, default_inertia


def _log(**kwargs) -> SimLog:
    n = 5
    t = np.linspace(0.0, 1.0, n)
    q = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
    w = np.zeros((n, 3))
    tau = np.zeros((n, 3))
    att = np.linspace(0.2, 0.01, n)
    defaults = dict(
        t=t,
        q=q,
        omega=w,
        tau=tau,
        q_hat=q.copy(),
        omega_hat=w.copy(),
        q_des=np.array([1.0, 0.0, 0.0, 0.0]),
        euler=np.zeros((n, 3)),
        euler_des=np.zeros(3),
        att_error=att,
        att_error_hat=None,
        est_att_error=None,
        controller="pid",
        estimator="truth",
    )
    defaults.update(kwargs)
    return SimLog(**defaults)


def test_parser_defaults():
    args = build_parser().parse_args([])
    assert args.n == 50
    assert args.controller == "pid"
    assert args.estimator == "mekf"
    assert args.inertia_frac == 0.05


def test_settle_time_proxy_finds_last_entry():
    t = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    err = np.array([0.5, 0.2, 0.04, 0.03, 0.02])
    ts = settle_time_s(t, err, 0.05)
    assert ts == 2.0
    assert np.isnan(settle_time_s(t, err, 0.01))


def test_sample_attitude_respects_bound():
    rng = np.random.default_rng(4)
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    for _ in range(20):
        q = sample_bounded_attitude(rng, 25.0)
        assert abs(float(np.linalg.norm(q)) - 1.0) < 1e-12
        assert np.rad2deg(geodesic_angle(q, identity)) <= 25.0 + 1e-9


def test_perturb_inertia_stays_physical():
    rng = np.random.default_rng(9)
    J0 = default_inertia()
    for _ in range(12):
        J = perturb_inertia(J0, 0.08, rng)
        validate_inertia(J)
        rel = np.linalg.norm(J - J0) / np.linalg.norm(J0)
        assert rel < 0.3
    J_same = perturb_inertia(J0, 0.0, rng)
    np.testing.assert_allclose(J_same, J0)


def test_log_uniform_stays_in_bounds():
    rng = np.random.default_rng(1)
    draws = np.array([log_uniform(rng, 0.5, 2.0) for _ in range(40)])
    assert draws.min() >= 0.5 - 1e-12
    assert draws.max() <= 2.0 + 1e-12


def test_classify_failure_reasons():
    ok, reason, _ = classify_failure(_log(), diverge_deg=25.0, diverge_omega=5.0)
    assert not ok and reason == ""

    nan_log = _log(omega=np.full((5, 3), np.nan))
    failed, reason, _ = classify_failure(nan_log, diverge_deg=25.0, diverge_omega=5.0)
    assert failed and reason == FAIL_NAN

    q_bad = np.tile(np.array([2.0, 0.0, 0.0, 0.0]), (5, 1))
    bad_q = _log(q=q_bad)
    failed, reason, _ = classify_failure(bad_q, diverge_deg=25.0, diverge_omega=5.0)
    assert failed and reason == FAIL_NON_UNIT

    diverged = _log(att_error=np.linspace(1.0, 1.0, 5))  # ~57 deg final
    failed, reason, _ = classify_failure(diverged, diverge_deg=25.0, diverge_omega=5.0)
    assert failed and reason == FAIL_DIVERGED


def test_monte_carlo_n5_smoke():
    """Tiny N closed-loop sweep so CI actually executes the harness."""
    summary = run_monte_carlo(
        MonteCarloConfig(
            n=5,
            seed=11,
            controller="pid",
            estimator="truth",
            dt=0.02,
            t_final=2.0,
            q0_max_deg=15.0,
            omega0_max=0.03,
            noise_scale_min=0.8,
            noise_scale_max=1.25,
            inertia_frac=0.03,
            settle_deg=2.0,
            diverge_deg=90.0,
        )
    )
    assert summary.n == 5
    assert len(summary.trials) == 5
    assert summary.n_nan == 0
    assert summary.n_non_unit == 0
    seeds = {t.seed for t in summary.trials}
    assert len(seeds) == 5
    for trial in summary.trials:
        assert np.isfinite(trial.peak_torque)
        assert trial.peak_torque >= 0.0
        assert trial.q_norm_err < 1e-9
        assert trial.noise_scale >= 0.8
        assert trial.noise_scale <= 1.25


def test_cli_n5_smoke(tmp_path):
    csv_path = tmp_path / "mc.csv"
    json_path = tmp_path / "mc.json"
    rc = main(
        [
            "--n",
            "5",
            "--seed",
            "2",
            "--estimator",
            "truth",
            "--t-final",
            "1.0",
            "--dt",
            "0.05",
            "--inertia-frac",
            "0.0",
            "--csv",
            str(csv_path),
            "--json",
            str(json_path),
        ]
    )
    assert rc == 0
    assert csv_path.is_file()
    assert json_path.is_file()
    text = csv_path.read_text()
    assert "final_att_error_deg" in text
    assert text.count("\n") >= 6

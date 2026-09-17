"""Smoke tests for the closed-loop Monte Carlo harness (tiny N)."""

import numpy as np

from attitude_sim.monte_carlo import (
    FAIL_DIVERGED,
    FAIL_NAN,
    FAIL_NON_UNIT,
    MonteCarloConfig,
    MonteCarloSummary,
    TrialResult,
    build_parser,
    classify_failure,
    log_uniform,
    main,
    perturb_inertia,
    run_monte_carlo,
    sample_bounded_attitude,
    settle_time_s,
    summary_from_json,
    write_json,
)
from attitude_sim.plant import validate_inertia
from attitude_sim.plots import (
    MC_ERROR_HIST_NAME,
    MC_SETTLE_SCATTER_NAME,
    plot_monte_carlo,
)
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
    assert args.plot is False
    assert args.out_dir.name == "outputs"
    assert args.from_json is None
    assert args.tau_dist_max == 0.0
    assert args.gain_scale_min == 1.0
    assert args.gain_scale_max == 1.0


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
            "--plot",
            "--out-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    assert csv_path.is_file()
    assert json_path.is_file()
    text = csv_path.read_text()
    assert "final_att_error_deg" in text
    assert text.count("\n") >= 6
    hist = tmp_path / MC_ERROR_HIST_NAME
    scatter = tmp_path / MC_SETTLE_SCATTER_NAME
    assert hist.is_file() and hist.stat().st_size > 100
    assert scatter.is_file() and scatter.stat().st_size > 100


def _synthetic_summary() -> MonteCarloSummary:
    trials = [
        TrialResult(
            trial=0,
            seed=1,
            final_att_error_deg=0.35,
            settle_time_s=12.0,
            peak_torque=0.02,
            q_norm_err=1e-12,
            omega0_norm=0.01,
            q0_from_identity_deg=8.0,
            noise_scale=0.7,
            failed=False,
        ),
        TrialResult(
            trial=1,
            seed=2,
            final_att_error_deg=1.1,
            settle_time_s=18.5,
            peak_torque=0.019,
            q_norm_err=1e-12,
            omega0_norm=0.02,
            q0_from_identity_deg=15.0,
            noise_scale=1.4,
            failed=False,
        ),
        TrialResult(
            trial=2,
            seed=3,
            final_att_error_deg=0.8,
            settle_time_s=float("nan"),
            peak_torque=0.02,
            q_norm_err=1e-12,
            omega0_norm=0.03,
            q0_from_identity_deg=20.0,
            noise_scale=1.9,
            failed=False,
        ),
        TrialResult(
            trial=3,
            seed=4,
            final_att_error_deg=40.0,
            settle_time_s=float("nan"),
            peak_torque=0.02,
            q_norm_err=1e-12,
            omega0_norm=0.04,
            q0_from_identity_deg=25.0,
            noise_scale=1.1,
            failed=True,
            fail_reason=FAIL_DIVERGED,
        ),
    ]
    return MonteCarloSummary(
        n=4,
        n_fail=1,
        n_nan=0,
        n_non_unit=0,
        n_diverged=1,
        n_settled=2,
        final_err_mean_deg=0.75,
        final_err_median_deg=0.8,
        final_err_p95_deg=1.08,
        settle_mean_s=15.25,
        peak_torque_max=0.02,
        peak_torque_mean=0.01975,
        controller="pid",
        estimator="mekf",
        t_final=40.0,
        settle_deg=2.0,
        trials=trials,
    )


def test_plot_helpers_write_png(tmp_path):
    paths = plot_monte_carlo(_synthetic_summary(), tmp_path)
    assert [p.name for p in paths] == [MC_ERROR_HIST_NAME, MC_SETTLE_SCATTER_NAME]
    for path in paths:
        assert path.is_file()
        assert path.stat().st_size > 100


def test_plot_helpers_empty_summary(tmp_path):
    empty = MonteCarloSummary(
        n=0,
        n_fail=0,
        n_nan=0,
        n_non_unit=0,
        n_diverged=0,
        n_settled=0,
        final_err_mean_deg=float("nan"),
        final_err_median_deg=float("nan"),
        final_err_p95_deg=float("nan"),
        settle_mean_s=float("nan"),
        peak_torque_max=float("nan"),
        peak_torque_mean=float("nan"),
        controller="pid",
        estimator="truth",
        t_final=10.0,
        settle_deg=2.0,
        trials=[],
    )
    paths = plot_monte_carlo(empty, tmp_path)
    assert all(p.is_file() and p.stat().st_size > 100 for p in paths)


def test_cli_from_json_plot(tmp_path):
    json_path = tmp_path / "mc.json"
    write_json(json_path, _synthetic_summary())
    loaded = summary_from_json(json_path)
    assert loaded.n == 4
    assert loaded.n_fail == 1
    assert np.isnan(loaded.trials[2].settle_time_s)
    out = tmp_path / "figs"
    rc = main(["--from-json", str(json_path), "--plot", "--out-dir", str(out)])
    assert rc == 0
    assert (out / MC_ERROR_HIST_NAME).is_file()
    assert (out / MC_SETTLE_SCATTER_NAME).is_file()


def test_monte_carlo_lqr_gain_and_disturbance_hook():
    """Same harness, LQR + mild gain scale / body-disturbance sampling."""
    summary = run_monte_carlo(
        MonteCarloConfig(
            n=3,
            seed=4,
            controller="lqr",
            estimator="truth",
            dt=0.05,
            t_final=1.0,
            q0_max_deg=8.0,
            omega0_max=0.02,
            noise_scale_min=1.0,
            noise_scale_max=1.0,
            inertia_frac=0.0,
            tau_dist_max=0.001,
            gain_scale_min=0.8,
            gain_scale_max=1.25,
            settle_deg=2.0,
            diverge_deg=180.0,
        )
    )
    assert summary.n == 3
    assert summary.controller == "lqr"
    assert summary.n_nan == 0
    assert summary.n_non_unit == 0
    scales = [t.gain_scale for t in summary.trials]
    dists = [t.tau_dist_norm for t in summary.trials]
    assert min(scales) >= 0.8 - 1e-12
    assert max(scales) <= 1.25 + 1e-12
    assert max(dists) <= 0.001 + 1e-12
    assert min(dists) >= 0.0
    assert any(d > 0.0 for d in dists) or summary.n == 3

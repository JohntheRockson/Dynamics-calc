"""Closed-loop SimLab smoke tests, including control on filter estimates."""

import numpy as np

from attitude_sim.sim import SimConfig, make_scenario_config, run_slew


def _cfg(**kwargs) -> SimConfig:
    base = dict(
        dt=0.01,
        t_final=22.0,
        plot=False,
        gif=False,
        seed=3,
        torque_limit=0.02,
    )
    base.update(kwargs)
    return SimConfig(**base)


def test_pid_truth_slew_settles():
    log = run_slew(_cfg(controller="pid", estimator="truth"))
    np.testing.assert_allclose(np.linalg.norm(log.q, axis=1), 1.0, atol=1e-12)
    assert log.final_att_error_deg < 2.0
    assert np.linalg.norm(log.omega[-1]) < 0.02
    # Error should drop substantially from the opening transient.
    assert log.att_error[-1] < 0.25 * log.att_error[0]


def test_lqr_truth_slew_settles():
    log = run_slew(_cfg(controller="lqr", estimator="truth"))
    assert log.final_att_error_deg < 2.5
    assert np.linalg.norm(log.omega[-1]) < 0.03


def test_pid_on_mekf_estimates_smoke():
    log = run_slew(_cfg(controller="pid", estimator="mekf", t_final=28.0))
    assert log.final_att_error_deg < 6.0
    assert np.linalg.norm(log.omega[-1]) < 0.05
    assert log.est_att_error is not None
    # Filter should be tracking the plant by the second half of the run.
    late = log.est_att_error[len(log.t) // 2 :]
    assert np.rad2deg(np.median(late)) < 5.0


def test_lqr_on_mahony_estimates_smoke():
    log = run_slew(_cfg(controller="lqr", estimator="mahony", t_final=28.0))
    assert log.final_att_error_deg < 8.0
    assert np.linalg.norm(log.omega[-1]) < 0.06


def test_pid_truth_detumble_dumps_rate():
    log = run_slew(
        make_scenario_config(
            "detumble",
            controller="pid",
            estimator="truth",
            t_final=22.0,
            plot=False,
            gif=False,
            seed=3,
        )
    )
    w0 = float(np.linalg.norm(log.omega[0]))
    w1 = float(np.linalg.norm(log.omega[-1]))
    assert w0 > 0.5
    assert w1 < 0.03
    assert w1 < 0.15 * w0
    assert log.final_att_error_deg < 3.0
    assert log.scenario == "detumble"


def test_scenario_plot_names(tmp_path):
    log = run_slew(
        _cfg(
            scenario="detumble",
            estimator="truth",
            t_final=0.05,
            plot=True,
            gif=False,
            out_dir=tmp_path,
            omega0=np.array([0.2, 0.0, 0.0]),
        )
    )
    assert log.plot_path == tmp_path / "detumble_summary.png"
    assert log.plot_path.is_file()

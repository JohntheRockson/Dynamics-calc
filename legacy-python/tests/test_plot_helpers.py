"""Agg PNG helpers for MRP error and environmental-torque post-process plots."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from attitude_sim.plots import mrp_error_from_log, plot_env_torque, plot_mrp_error, plot_slew
from attitude_sim.sim import make_scenario_config, run_slew


def _ok_png(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 100


def test_hold_and_eigenaxis_write_summary_mrp_and_env_pngs(tmp_path: Path):
    hold = run_slew(
        make_scenario_config(
            "hold",
            estimator="truth",
            t_final=0.2,
            plot=True,
            gif=False,
            out_dir=tmp_path,
            seed=1,
        )
    )
    assert hold.plot_path == tmp_path / "hold_summary.png"
    assert hold.mrp_plot_path == tmp_path / "hold_mrp.png"
    assert hold.env_plot_path == tmp_path / "hold_env_torque.png"
    assert _ok_png(hold.plot_path)
    assert _ok_png(hold.mrp_plot_path)
    assert _ok_png(hold.env_plot_path)

    eigen = run_slew(
        make_scenario_config(
            "eigenaxis",
            estimator="truth",
            t_final=0.2,
            plot=True,
            gif=False,
            out_dir=tmp_path,
            seed=1,
        )
    )
    assert eigen.plot_path == tmp_path / "eigenaxis_summary.png"
    assert eigen.mrp_plot_path == tmp_path / "eigenaxis_mrp.png"
    assert eigen.env_plot_path is None
    assert _ok_png(eigen.plot_path)
    assert _ok_png(eigen.mrp_plot_path)
    assert not (tmp_path / "eigenaxis_env_torque.png").exists()
    assert not (tmp_path / "hold_attitude.gif").exists()


def test_slew_mrp_plot_opt_in(tmp_path: Path):
    log = run_slew(
        make_scenario_config(
            "slew",
            estimator="truth",
            t_final=0.15,
            plot=True,
            gif=False,
            mrp_plot=True,
            out_dir=tmp_path,
        )
    )
    assert _ok_png(tmp_path / "slew_summary.png")
    assert log.mrp_plot_path == tmp_path / "slew_mrp.png"
    assert _ok_png(log.mrp_plot_path)
    # Default slew still skips the extra MRP file unless opted in.
    bare = run_slew(
        make_scenario_config(
            "slew",
            estimator="truth",
            t_final=0.05,
            plot=True,
            gif=False,
            out_dir=tmp_path / "bare",
        )
    )
    assert bare.mrp_plot_path is None
    assert not (tmp_path / "bare" / "slew_mrp.png").exists()


def test_plot_helpers_accept_log_without_rerun(tmp_path: Path):
    log = run_slew(
        make_scenario_config(
            "hold",
            estimator="truth",
            t_final=0.12,
            plot=False,
            gif=False,
        )
    )
    mrp_path = plot_mrp_error(log, tmp_path / "from_log_mrp.png")
    env_path = plot_env_torque(log, tmp_path / "from_log_env.png")
    summary = plot_slew(log, tmp_path / "from_log_summary.png")
    assert _ok_png(mrp_path)
    assert _ok_png(env_path)
    assert _ok_png(summary)
    sigma = mrp_error_from_log(log)
    assert sigma.shape == (len(log.t), 3)
    assert np.all(np.isfinite(sigma))

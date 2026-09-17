"""Committed SimLab / Monte Carlo figures, with a cheap Agg PNG fallback.

CI does not smoke attitude GIFs (Pillow is slow). Agg-backend PNGs are the
coverage path: Monte Carlo helpers already in ``tests/test_monte_carlo.py``,
plus a short SimLab ``--no-gif`` PNG here if committed copies are missing.
"""

from __future__ import annotations

from pathlib import Path

from attitude_sim.monte_carlo import main as mc_main
from attitude_sim.plots import MC_ERROR_HIST_NAME, MC_SETTLE_SCATTER_NAME
from attitude_sim.sim import main as sim_main

REPO = Path(__file__).resolve().parents[1]
FIGURES = REPO / "docs" / "figures"
NOTEBOOK = REPO / "notebooks" / "simlab_demo.ipynb"

SLEW_PNG = "slew_summary.png"
SLEW_GIF = "slew_attitude.gif"
MC_PNGS = (MC_ERROR_HIST_NAME, MC_SETTLE_SCATTER_NAME)


def _ok_png(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 100


def test_simlab_demo_notebook_links_figures():
    """Walkthrough notebook exists and points at the committed recruiter figures."""
    assert NOTEBOOK.is_file(), f"missing demo notebook: {NOTEBOOK}"
    text = NOTEBOOK.read_text(encoding="utf-8")
    for name in (SLEW_PNG, SLEW_GIF, *MC_PNGS):
        assert name in text, f"{NOTEBOOK.name} should reference {name}"
    assert "--scenario" in text and "detumble" in text
    assert "attitude_sim.monte_carlo" in text


def test_docs_figures_exist_or_regenerate_in_tmp(tmp_path: Path):
    """``docs/figures`` slew + ``mc_*.png`` copies, or Agg PNG regen in tmp.

    GIF regeneration is intentionally skipped (same policy as CI ``--no-gif``).
    When the committed copies are present, still write a short slew PNG in
    ``tmp`` so the Agg CLI path stays exercised.
    """
    slew_png = FIGURES / SLEW_PNG
    mc_paths = [FIGURES / name for name in MC_PNGS]
    have_slew = _ok_png(slew_png)
    have_mc = all(_ok_png(p) for p in mc_paths)

    if have_slew and have_mc:
        gif = FIGURES / SLEW_GIF
        # Recruiter GIF is checked in, never regenerated here (Pillow / CI skip).
        assert gif.is_file() and gif.stat().st_size > 100
        for scenario in ("slew", "detumble"):
            rc = sim_main(
                [
                    "--scenario",
                    scenario,
                    "--t-final",
                    "0.2",
                    "--no-gif",
                    "--out-dir",
                    str(tmp_path),
                ]
            )
            assert rc == 0
            assert _ok_png(tmp_path / f"{scenario}_summary.png")
            assert not (tmp_path / f"{scenario}_attitude.gif").exists()
        return

    if not have_slew:
        rc = sim_main(
            [
                "--scenario",
                "slew",
                "--t-final",
                "0.2",
                "--no-gif",
                "--out-dir",
                str(tmp_path),
            ]
        )
        assert rc == 0
        assert _ok_png(tmp_path / SLEW_PNG)

    if not have_mc:
        rc = mc_main(
            [
                "--n",
                "5",
                "--estimator",
                "truth",
                "--t-final",
                "0.2",
                "--dt",
                "0.05",
                "--inertia-frac",
                "0.0",
                "--diverge-deg",
                "180",
                "--plot",
                "--out-dir",
                str(tmp_path),
            ]
        )
        assert rc == 0
        assert all(_ok_png(tmp_path / name) for name in MC_PNGS)

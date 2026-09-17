"""MEKF NEES consistency smoke (open-loop truth, matching Farrenkopf / vector noise).

The chi-square band is documented in ``docs/estimation.md`` and mirrored by
``attitude_sim.nees.NEES_*_BAND``.  This is a loose smoke, not a full ANEES
campaign: keep ``n_runs`` / ``t_final`` small so CI stays fast.
"""

import numpy as np
import pytest

from attitude_sim.nees import (
    NEES_ATT_BAND,
    NEES_BIAS_BAND,
    NEES_FULL_BAND,
    apply_mekf_error,
    default_nees_p0,
    mean_nees_monte_carlo,
    mekf_error_state,
    nees,
    run_open_loop_mekf_nees,
    split_mekf_nees,
)


def test_nees_quadratic_form_and_split():
    x = np.array([1.0, -2.0, 0.5, 0.1, 0.0, -0.2])
    assert nees(x, np.eye(6)) == pytest.approx(float(x @ x))
    P = np.diag([4.0, 1.0, 1.0, 0.25, 1.0, 4.0])
    expected = (
        x[0] ** 2 / 4.0
        + x[1] ** 2
        + x[2] ** 2
        + x[3] ** 2 / 0.25
        + x[4] ** 2
        + x[5] ** 2 / 4.0
    )
    full, att, bias = split_mekf_nees(x, P)
    assert full == pytest.approx(expected)
    assert att == pytest.approx(x[0] ** 2 / 4.0 + x[1] ** 2 + x[2] ** 2)
    assert bias == pytest.approx(x[3] ** 2 / 0.25 + x[4] ** 2 + x[5] ** 2 / 4.0)
    assert nees(np.zeros(6), P) == pytest.approx(0.0)


def test_mekf_error_roundtrip_matches_right_multiply():
    q_true = np.array([0.8, 0.2, -0.1, 0.4])
    q_true = q_true / np.linalg.norm(q_true)
    b_true = np.array([0.01, -0.02, 0.005])
    x = np.array([0.03, -0.02, 0.01, 0.004, -0.001, 0.002])
    q_hat, b_hat = apply_mekf_error(q_true, b_true, x)
    recovered = mekf_error_state(q_true, q_hat, b_true, b_hat)
    np.testing.assert_allclose(recovered, x, atol=1e-12)


def test_initial_draw_mean_nees_is_chi_square():
    """Draws from P0 should have E[ε] = 6 (χ²_6); 200 draws stay in a loose band."""
    rng = np.random.default_rng(0)
    P0 = default_nees_p0()
    samples = [nees(rng.multivariate_normal(np.zeros(6), P0), P0) for _ in range(200)]
    mean = float(np.mean(samples))
    # Independent χ²_6, N=200: std of the mean is sqrt(12/200) ≈ 0.24.
    # Use a still-loose 4σ-ish window so this is not a flaky CI gate.
    assert 5.0 < mean < 7.0, mean


def test_mekf_open_loop_mean_nees_in_chi_square_band():
    """Short matching-noise open-loop run: mean NEES in the documented loose band.

    Band (see docs/estimation.md):
      full 6-state  χ²_6  E[ε]=6  smoke [1.5, 18]
      attitude δα   χ²_3  E[ε]=3  smoke [0.5, 12]
      bias δb       χ²_3  E[ε]=3  smoke [0.5, 12]
    """
    # 8 runs × 1.6 s × dt=0.02 ≈ 80 filter steps each; skip t=0 (prior draw).
    full, att, bias = mean_nees_monte_carlo(n_runs=8, seed0=11, skip=1)
    assert NEES_FULL_BAND[0] <= full <= NEES_FULL_BAND[1], f"full NEES={full}"
    assert NEES_ATT_BAND[0] <= att <= NEES_ATT_BAND[1], f"attitude NEES={att}"
    assert NEES_BIAS_BAND[0] <= bias <= NEES_BIAS_BAND[1], f"bias NEES={bias}"


def test_open_loop_nees_series_shape_and_unit_runtime_guard():
    result = run_open_loop_mekf_nees(dt=0.02, t_final=0.4, seed=3)
    n = int(np.round(0.4 / 0.02)) + 1
    assert result.t.shape == (n,)
    assert result.full.shape == (n,)
    assert np.all(np.isfinite(result.full))
    assert np.all(result.full >= 0.0)
    assert result.t[0] == pytest.approx(0.0)
    assert result.t[-1] == pytest.approx(0.4)

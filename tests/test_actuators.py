"""Per-axis torque limits and first-order actuator lag."""

import numpy as np
import pytest

from attitude_sim.actuators import (
    TorqueActuator,
    clip_torque,
    make_actuator,
    parse_tau_max,
)
from attitude_sim.sim import SimConfig, main, run_slew


def _cfg(**kwargs) -> SimConfig:
    base = dict(
        dt=0.01,
        t_final=4.0,
        plot=False,
        gif=False,
        seed=3,
        torque_limit=None,
        estimator="truth",
        controller="pid",
    )
    base.update(kwargs)
    return SimConfig(**base)


def test_clip_torque_unlimited_is_identity():
    tau = np.array([0.04, -0.03, 0.01])
    np.testing.assert_array_equal(clip_torque(tau, None), tau)
    # Original is not mutated.
    tau2 = tau.copy()
    clip_torque(tau2, 0.01)
    np.testing.assert_array_equal(tau2, tau)


def test_clip_torque_scalar_and_per_axis():
    tau = np.array([0.03, -0.03, 0.005])
    np.testing.assert_allclose(clip_torque(tau, 0.02), [0.02, -0.02, 0.005])
    lim = np.array([0.01, 0.05, 0.0])
    np.testing.assert_allclose(clip_torque(tau, lim), [0.01, -0.03, 0.0])


def test_clip_torque_is_not_euclidean():
    # Euclidean |τ| ≤ 0.02 would scale this vector; per-axis clips independently.
    tau = np.array([0.03, 0.03, 0.0])
    clipped = clip_torque(tau, 0.02)
    np.testing.assert_allclose(clipped, [0.02, 0.02, 0.0])
    assert np.linalg.norm(clipped) > 0.02


def test_clip_torque_rejects_negative_limits():
    with pytest.raises(ValueError, match="non-negative"):
        clip_torque(np.zeros(3), -0.01)
    with pytest.raises(ValueError, match="non-negative"):
        clip_torque(np.zeros(3), np.array([0.02, -0.01, 0.02]))


def test_parse_tau_max():
    assert parse_tau_max(None) is None
    assert parse_tau_max("") is None
    assert parse_tau_max("0.02") == 0.02
    np.testing.assert_allclose(parse_tau_max("0.02,0.01,0.015"), [0.02, 0.01, 0.015])
    with pytest.raises(ValueError):
        parse_tau_max("1,2")
    with pytest.raises(ValueError, match="non-negative"):
        parse_tau_max("-0.01")


def test_first_order_lag_step_response():
    T = 0.10
    dt = 0.001
    act = make_actuator(tau_max=1.0, time_constant=T)
    act.reset()
    n = int(round(T / dt))
    tau = np.zeros(3)
    for _ in range(n):
        tau = act.apply(np.array([1.0, 0.0, 0.0]), dt)
    # Exact ZOH: 1 - exp(-t/T) at t = T is 1 - e^{-1}.
    np.testing.assert_allclose(tau[0], 1.0 - np.exp(-1.0), atol=1e-8)
    np.testing.assert_allclose(tau[1:], 0.0, atol=1e-12)


def test_lag_output_never_exceeds_limits():
    act = TorqueActuator(tau_max=0.02, time_constant=0.05)
    rng = np.random.default_rng(0)
    for _ in range(200):
        cmd = rng.normal(scale=0.1, size=3)
        tau = act.apply(cmd, 0.01)
        assert np.all(np.abs(tau) <= 0.02 + 1e-12)


def test_instantaneous_actuator_matches_clip():
    act = make_actuator(tau_max=0.01, time_constant=None)
    cmd = np.array([0.04, -0.02, 0.005])
    np.testing.assert_allclose(act.apply(cmd, 0.01), clip_torque(cmd, 0.01))


def test_closed_loop_torque_never_exceeds_per_axis_limit():
    lim = 0.005
    for controller in ("pid", "lqr"):
        log = run_slew(_cfg(controller=controller, actuator_tau_max=lim, t_final=6.0))
        peak = np.max(np.abs(log.tau), axis=0)
        assert np.all(peak <= lim * (1.0 + 1e-9)), (controller, peak)
        # The 75° opening command would exceed 5 mN·m without the wheels.
        assert np.max(np.abs(log.tau)) > 0.5 * lim


def test_closed_loop_lag_and_asymmetric_limits():
    lim = np.array([0.008, 0.004, 0.006])
    log = run_slew(
        _cfg(
            controller="pid",
            actuator_tau_max=lim,
            actuator_tau=0.05,
            t_final=5.0,
        )
    )
    assert np.all(np.abs(log.tau) <= lim * (1.0 + 1e-9))


def test_unsaturated_actuator_matches_prior_closed_loop():
    """Default (unlimited, no lag) matches a huge-limit actuator within tolerance."""
    kwargs = dict(controller="pid", estimator="truth", t_final=3.0, torque_limit=0.02)
    log_prior = run_slew(_cfg(**kwargs))  # actuator identity
    log_hi = run_slew(_cfg(**kwargs, actuator_tau_max=10.0))
    np.testing.assert_allclose(log_prior.tau, log_hi.tau, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(log_prior.q, log_hi.q, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(log_prior.omega, log_hi.omega, rtol=0.0, atol=1e-12)


def test_cli_actuator_flags_smoke():
    assert (
        main(
            [
                "--controller",
                "pid",
                "--estimator",
                "truth",
                "--t-final",
                "0.05",
                "--no-plot",
                "--no-gif",
                "--actuator-tau-max",
                "0.02",
                "--actuator-tau",
                "0.01",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "--controller",
                "lqr",
                "--estimator",
                "truth",
                "--t-final",
                "0.05",
                "--no-plot",
                "--no-gif",
                "--actuator-tau-max",
                "0.02,0.015,0.01",
            ]
        )
        == 0
    )

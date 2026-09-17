"""Per-axis torque limits, first-order lag, and optional momentum dump."""

import numpy as np
import pytest

from attitude_sim.actuators import (
    TorqueActuator,
    clip_torque,
    make_actuator,
    momentum_dump_torque,
    parse_tau_max,
)
from attitude_sim.controls import PIDAttitudeController, apply_torque_limits
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
    n = round(T / dt)
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


def test_momentum_dump_torque_deadzone_and_off():
    h = np.array([0.04, -0.005, 0.02])
    np.testing.assert_allclose(momentum_dump_torque(h, None), 0.0)
    np.testing.assert_allclose(momentum_dump_torque(h, 0.01, dump_gain=0.0), 0.0)
    # Excess is 0.03, 0, 0.01; gain 2 → (0.06, 0, 0.02).
    np.testing.assert_allclose(momentum_dump_torque(h, 0.01, dump_gain=2.0), [0.06, 0.0, 0.02])
    per_axis = momentum_dump_torque(h, np.array([0.03, 0.0, 0.05]), dump_gain=1.0)
    np.testing.assert_allclose(per_axis, [0.01, -0.005, 0.0])
    with pytest.raises(ValueError, match="non-negative"):
        momentum_dump_torque(h, -0.01)
    with pytest.raises(ValueError, match="dump_gain"):
        momentum_dump_torque(h, 0.01, dump_gain=-1.0)


def test_wheel_momentum_integrates_minus_torque():
    act = make_actuator(tau_max=1.0)
    act.reset()
    dt = 0.01
    cmd = np.array([0.02, -0.01, 0.0])
    for _ in range(50):
        act.apply(cmd, dt)
    np.testing.assert_allclose(act.momentum, -cmd * 50 * dt, atol=1e-15)
    np.testing.assert_allclose(act.dump_command, 0.0)
    np.testing.assert_allclose(act.external_torque, 0.0)


def test_dump_unloads_excess_and_pairs_external():
    act = make_actuator(tau_max=0.05, h_dump=0.01, dump_gain=2.0)
    act.reset(h0=np.array([0.04, 0.0, -0.03]))
    dt = 0.01
    # Zero attitude command: leftover authority is the full box.
    tau = act.apply(np.zeros(3), dt)
    expected_dump = momentum_dump_torque(np.array([0.04, 0.0, -0.03]), 0.01, 2.0)
    np.testing.assert_allclose(tau, expected_dump)
    np.testing.assert_allclose(act.dump_command, expected_dump)
    np.testing.assert_allclose(act.external_torque, -expected_dump)
    # ḣ = −τ unloads |h| toward the threshold.
    assert abs(act.momentum[0]) < 0.04
    assert abs(act.momentum[2]) < 0.03
    # Below threshold: no dump.
    quiet = make_actuator(h_dump=0.05, dump_gain=2.0)
    quiet.reset(h0=np.array([0.01, -0.02, 0.0]))
    np.testing.assert_allclose(quiet.apply(np.zeros(3), dt), 0.0)


def test_dump_uses_leftover_authority_not_stealing_command():
    """Attitude command keeps the box; dump cannot deepen saturation."""
    act = make_actuator(tau_max=0.02, h_dump=0.0, dump_gain=10.0)
    act.reset(h0=np.array([1.0, 0.0, 0.0]))
    cmd = np.array([0.02, 0.0, 0.0])
    tau = act.apply(cmd, 0.01)
    # Command already at +τ_max; leftover for a +dump is zero.
    np.testing.assert_allclose(tau, cmd)
    np.testing.assert_allclose(act.dump_command, 0.0)
    # Opposite-sign dump can use leftover down to −τ_max.
    act.reset(h0=np.array([-1.0, 0.0, 0.0]))
    tau_rev = act.apply(cmd, 0.01)
    assert tau_rev[0] < cmd[0]
    assert tau_rev[0] >= -0.02 - 1e-15
    np.testing.assert_allclose(tau_rev, clip_torque(cmd + act.dump_command, 0.02))


def test_dump_disabled_matches_prior_apply():
    cmd = np.array([0.03, -0.01, 0.005])
    prior = make_actuator(tau_max=0.02, time_constant=0.05)
    dump_off = make_actuator(tau_max=0.02, time_constant=0.05, h_dump=None)
    prior.reset()
    dump_off.reset()
    for _ in range(20):
        a = prior.apply(cmd, 0.01)
        b = dump_off.apply(cmd, 0.01)
        np.testing.assert_allclose(a, b)


def test_make_actuator_rejects_bad_dump_kwargs():
    with pytest.raises(ValueError, match="dump_gain"):
        make_actuator(h_dump=0.01, dump_gain=-0.5)
    with pytest.raises(ValueError, match="non-negative"):
        TorqueActuator(h_dump=-0.01)


def test_closed_loop_pid_hold_dump_rejects_bias_without_windup():
    """PID Ki still nulls τ_d; paired dump unloads wheels and does not steal τ_cmd."""
    q_des = np.array([1.0, 0.0, 0.0, 0.0])
    tau_dist = np.array([0.002, -0.001, 0.0008])
    kwargs = dict(
        controller="pid",
        estimator="truth",
        q0=q_des,
        q_des=q_des,
        tau_dist=tau_dist,
        t_final=30.0,
        torque_limit=0.02,
    )
    log_off = run_slew(_cfg(**kwargs))
    log_on = run_slew(_cfg(**kwargs, actuator_h_dump=0.01, actuator_dump_gain=1.0))
    assert log_on.final_att_error_deg < 1.5
    assert np.linalg.norm(log_on.omega[-1]) < 0.02
    # Net spacecraft control (wheel + paired dump) still cancels τ_d.
    np.testing.assert_allclose(log_on.tau[-1] + log_on.tau_ext[-1], -tau_dist, atol=2e-4)
    # Dump actually ran and kept |h| near the threshold, not the open-loop ∫τ_d.
    assert np.max(np.linalg.norm(log_on.h_wheel, axis=1)) > 0.01
    assert np.linalg.norm(log_on.h_wheel[-1]) < 0.025
    assert np.linalg.norm(log_on.h_wheel[-1]) < 0.6 * np.linalg.norm(log_off.h_wheel[-1])
    # PID anti-windup path is unchanged: same apply_torque_limits box as without dump.
    pid = PIDAttitudeController(np.diag([0.05, 0.06, 0.07]))
    tau_cmd = pid.command(q_des, np.zeros(3), q_des, dt=0.01)
    np.testing.assert_allclose(tau_cmd, apply_torque_limits(tau_cmd, 0.02, None))
    # Off vs on: attitude hold is the same order (pairing keeps the plant dump-free).
    assert abs(log_on.final_att_error_deg - log_off.final_att_error_deg) < 0.5


def test_closed_loop_lqr_dump_smoke():
    q_des = np.array([1.0, 0.0, 0.0, 0.0])
    tau_dist = np.array([0.002, -0.001, 0.0008])
    log = run_slew(
        _cfg(
            controller="lqr",
            estimator="truth",
            q0=q_des,
            q_des=q_des,
            tau_dist=tau_dist,
            actuator_h_dump=0.008,
            actuator_dump_gain=1.0,
            t_final=20.0,
        )
    )
    # LQR still has no integrator; dump is post-controller and paired.
    assert log.final_att_error_deg < 3.0
    assert np.linalg.norm(log.omega[-1]) < 0.02
    assert np.linalg.norm(log.h_wheel[-1]) < 0.02
    assert np.max(np.abs(log.tau_ext)) > 0.0


def test_cli_dump_flags_smoke():
    assert (
        main(
            [
                "--controller",
                "pid",
                "--estimator",
                "truth",
                "--angle-deg",
                "0",
                "--tau-dist",
                "0.002,-0.001,0.0008",
                "--actuator-h-dump",
                "0.01",
                "--actuator-dump-gain",
                "1.5",
                "--t-final",
                "0.05",
                "--no-plot",
                "--no-gif",
            ]
        )
        == 0
    )
    with pytest.raises(SystemExit):
        main(["--actuator-h-dump", "-0.01", "--t-final", "0.05", "--no-plot", "--no-gif"])
    with pytest.raises(SystemExit):
        main(["--actuator-dump-gain", "-1", "--t-final", "0.05", "--no-plot", "--no-gif"])

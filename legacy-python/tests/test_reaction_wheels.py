"""Reaction-wheel assembly: momentum / torque sat, friction dump, closed-loop."""

import numpy as np
import pytest

from attitude_sim.actuators import TorqueActuator, clip_torque, make_actuator
from attitude_sim.monte_carlo import (
    MonteCarloConfig,
    run_monte_carlo,
    rw_saturation_config,
)
from attitude_sim.reaction_wheels import (
    DEFAULT_WHEEL_INERTIA,
    ReactionWheelAssembly,
    friction_torque,
    gate_momentum_saturation,
    make_reaction_wheels,
)
from attitude_sim.sim import SimConfig, main, run_slew


def test_make_actuator_identity_without_rw_kwargs():
    act = make_actuator(tau_max=0.02)
    assert isinstance(act, TorqueActuator)
    assert not isinstance(act, ReactionWheelAssembly)


def test_make_actuator_selects_rw_when_h_max_set():
    act = make_actuator(tau_max=0.02, h_max=0.01)
    assert isinstance(act, ReactionWheelAssembly)
    np.testing.assert_allclose(act.inertia, DEFAULT_WHEEL_INERTIA)


def test_unsaturated_no_friction_matches_clip():
    """With ω=0, no friction, unlimited h: τ_body = clip(τ_cmd)."""
    tau_max = 0.05
    rw = make_reaction_wheels(tau_max=tau_max, gyroscopic=False)
    rw.reset()
    act = make_actuator(tau_max=tau_max)
    cmd = np.array([0.04, -0.02, 0.06])
    tau_rw = rw.apply(cmd, 0.01, omega=np.zeros(3))
    tau_box = act.apply(cmd, 0.01)
    np.testing.assert_allclose(tau_rw, tau_box, atol=1e-12)
    np.testing.assert_allclose(tau_box, clip_torque(cmd, tau_max))


def test_torque_sat_never_exceeds_box():
    rw = make_reaction_wheels(tau_max=0.01, h_max=10.0, gyroscopic=False)
    rng = np.random.default_rng(1)
    for _ in range(80):
        cmd = rng.normal(scale=0.1, size=3)
        tau = rw.apply(cmd, 0.01, omega=np.zeros(3))
        assert np.all(np.abs(tau) <= 0.01 + 1e-9)


def test_momentum_sat_blocks_spin_up_and_clips_h():
    h_max = 0.002
    tau_max = 0.05
    rw = make_reaction_wheels(tau_max=tau_max, h_max=h_max, gyroscopic=False)
    rw.reset()
    dt = 0.01
    cmd = np.array([0.04, 0.0, 0.0])
    for _ in range(40):
        tau = rw.apply(cmd, dt, omega=np.zeros(3))
        assert np.all(np.abs(rw.momentum) <= h_max + 1e-12)
    # At the wall, further command along +x cannot increase |h|.
    h_wall = rw.momentum.copy()
    assert abs(h_wall[0]) >= h_max * (1.0 - 1e-9)
    tau = rw.apply(cmd, dt, omega=np.zeros(3))
    np.testing.assert_allclose(abs(rw.momentum[0]), h_max, atol=1e-12)
    np.testing.assert_allclose(tau[0], 0.0, atol=1e-12)
    np.testing.assert_allclose(rw.momentum, h_wall)
    assert bool(rw.momentum_saturated[0])


def test_momentum_gate_allows_unload():
    h = np.array([0.01, 0.0, 0.0])
    # Motor torque that would increase |h| is zeroed.
    np.testing.assert_allclose(
        gate_momentum_saturation(np.array([0.02, 0.0, 0.0]), h, 0.01),
        [0.0, 0.0, 0.0],
    )
    # Opposite motor torque (unload) is kept.
    np.testing.assert_allclose(
        gate_momentum_saturation(np.array([-0.02, 0.0, 0.0]), h, 0.01),
        [-0.02, 0.0, 0.0],
    )


def test_friction_momentum_dump():
    """After a spin-up, τ_cmd=0 and viscous friction dumps |h| onto the body."""
    rw = make_reaction_wheels(
        tau_max=0.05,
        h_max=0.05,
        wheel_inertia=2e-4,
        visc_friction=2e-4,
        gyroscopic=False,
    )
    rw.reset()
    dt = 0.01
    for _ in range(40):
        rw.apply(np.array([0.02, 0.0, 0.0]), dt, omega=np.zeros(3))
    h_peak = float(np.abs(rw.momentum[0]))
    assert h_peak > 1e-4
    for _ in range(120):
        tau = rw.apply(np.zeros(3), dt, omega=np.zeros(3))
        # Friction dumps wheel momentum onto the body: τ_body has the same
        # sign as h_w while |h| decays.
        if abs(rw.momentum[0]) > 1e-8:
            assert tau[0] * rw.momentum[0] >= -1e-12
    assert abs(rw.momentum[0]) < 0.6 * h_peak


def test_friction_torque_formula():
    w = np.array([1.0, -0.5, 0.0])
    tau_f = friction_torque(w, visc=0.02, coulomb=0.0)
    np.testing.assert_allclose(tau_f, -0.02 * w)
    with pytest.raises(ValueError, match="friction"):
        friction_torque(w, visc=-0.01, coulomb=0.0)


def test_gyroscopic_couple_matches_omega_cross_h():
    rw = make_reaction_wheels(tau_max=1.0, h_max=1.0, gyroscopic=True)
    rw.reset(h0=np.array([0.0, 0.0, 0.04]))
    omega = np.array([0.3, 0.0, 0.0])
    tau = rw.apply(np.zeros(3), 0.01, omega=omega)
    # No motor, no friction: ḣ=0, τ = −ω × h.
    np.testing.assert_allclose(tau, -np.cross(omega, np.array([0.0, 0.0, 0.04])), atol=1e-12)


def test_rejects_bad_inertia_and_friction():
    with pytest.raises(ValueError, match="positive"):
        ReactionWheelAssembly(wheel_inertia=0.0)
    with pytest.raises(ValueError, match="friction"):
        ReactionWheelAssembly(visc_friction=-1e-6)


def _cfg(**kwargs) -> SimConfig:
    base = dict(
        dt=0.01,
        t_final=3.0,
        plot=False,
        gif=False,
        seed=5,
        torque_limit=0.02,
        estimator="truth",
        controller="pid",
    )
    base.update(kwargs)
    return SimConfig(**base)


def test_closed_loop_rw_logs_and_respects_h_max():
    h_max = 0.0012
    log = run_slew(
        _cfg(
            actuator_tau_max=0.02,
            rw_h_max=h_max,
            rw_inertia=2e-4,
            rw_visc=0.0,
            t_final=4.0,
        )
    )
    assert log.h_wheel.shape[0] == log.t.size
    assert log.omega_wheel.shape == log.h_wheel.shape
    assert np.all(np.abs(log.h_wheel) <= h_max * (1.0 + 1e-8))
    assert log.sat_fraction > 0.0
    assert np.isfinite(log.peak_rate)
    np.testing.assert_allclose(
        log.omega_wheel,
        log.h_wheel / 2e-4,
        atol=1e-12,
    )


def test_closed_loop_unlimited_rw_matches_clip_actuator():
    """No h_max, no friction, gyroscopic off: same τ as the clip box."""
    kwargs = dict(
        controller="pid",
        estimator="truth",
        t_final=2.0,
        torque_limit=0.02,
        actuator_tau_max=0.02,
        rw_gyroscopic=False,
    )
    log_box = run_slew(_cfg(**kwargs))
    log_rw = run_slew(_cfg(**kwargs, rw_inertia=2e-4))
    np.testing.assert_allclose(log_box.tau, log_rw.tau, atol=1e-10)
    np.testing.assert_allclose(log_box.omega, log_rw.omega, atol=1e-9)


def test_cli_rw_flags_smoke():
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
                "--rw-h-max",
                "0.004",
                "--rw-inertia",
                "2e-4",
                "--rw-visc",
                "1e-6",
            ]
        )
        == 0
    )


def test_monte_carlo_rw_sat_stress_smoke():
    """Tiny N: RW saturation + noisy sensors on the existing MC harness."""
    summary = run_monte_carlo(
        rw_saturation_config(
            n=3,
            seed=7,
            estimator="mekf",
            dt=0.05,
            t_final=1.2,
            q0_max_deg=12.0,
            omega0_max=0.04,
            inertia_frac=0.0,
            diverge_deg=180.0,
            settle_deg=20.0,
        )
    )
    assert summary.n == 3
    assert summary.n_nan == 0
    assert summary.n_non_unit == 0
    assert summary.sat_fraction_mean > 0.0
    for trial in summary.trials:
        assert np.isfinite(trial.peak_rate)
        assert trial.peak_rate >= 0.0
        assert 0.0 <= trial.sat_fraction <= 1.0
        assert np.isfinite(trial.peak_torque)


def test_rw_saturation_config_rejects_unknown():
    with pytest.raises(TypeError, match="unknown"):
        rw_saturation_config(not_a_field=1)


def test_stock_mc_reports_zero_sat_fraction():
    summary = run_monte_carlo(
        MonteCarloConfig(
            n=2,
            seed=1,
            estimator="truth",
            dt=0.05,
            t_final=0.4,
            inertia_frac=0.0,
            diverge_deg=180.0,
        )
    )
    assert summary.sat_fraction_mean == 0.0
    assert np.isfinite(summary.peak_rate_max)

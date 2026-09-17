"""SimLab: closed-loop scenarios, CLI, and ``python -m attitude_sim``."""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from attitude_sim.actuators import make_actuator, parse_tau_max
from attitude_sim.controls import make_controller
from attitude_sim.estimation import (
    ComplementaryFilter,
    MultiplicativeEKF,
    make_estimator,
    vectors_from_sensors,
)
from attitude_sim.plant import RigidBody, step_rigid_body
from attitude_sim.quaternions import (
    axis_angle_to_quat,
    geodesic_angle,
    quat_normalize,
    quat_to_euler321,
)
from attitude_sim.sensors import GyroModel, VectorSensor

SCENARIOS = ("slew", "detumble")
SLEW_AXIS = np.array([0.2, 0.5, 0.84])
DETUMBLE_Q0_AXIS = np.array([0.4, 0.2, 0.9])
DETUMBLE_OMEGA0 = np.array([0.55, -0.40, 0.30])


def default_inertia() -> np.ndarray:
    """Principal inertia of a smallsat-class rigid body (kg·m²)."""
    return np.diag([0.05, 0.06, 0.07])


@dataclass
class SimConfig:
    inertia: np.ndarray = field(default_factory=default_inertia)
    dt: float = 0.01
    t_final: float = 40.0
    controller: str = "pid"
    estimator: str = "mekf"
    scenario: str = "slew"
    q0: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    omega0: np.ndarray = field(default_factory=lambda: np.zeros(3))
    q_des: np.ndarray = field(
        default_factory=lambda: axis_angle_to_quat(SLEW_AXIS, np.deg2rad(75.0))
    )
    torque_limit: float | None = 0.02
    gain_scale: float = 1.0
    actuator_tau_max: float | np.ndarray | None = None
    actuator_tau: float | None = None
    tau_dist: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gyro_sigma_v: float = 5e-4
    gyro_sigma_u: float = 1e-6
    gyro_bias: np.ndarray = field(default_factory=lambda: np.array([0.002, -0.001, 0.0015]))
    mag_sigma: float = 3e-3
    sun_sigma: float = 2e-3
    use_mag: bool = True
    use_sun: bool = True
    seed: int = 1
    plot: bool = True
    gif: bool = True
    out_dir: Path = field(default_factory=lambda: Path("outputs"))

    @property
    def artifact_stem(self) -> str:
        return self.scenario


@dataclass
class SimLog:
    t: np.ndarray
    q: np.ndarray
    omega: np.ndarray
    tau: np.ndarray
    q_hat: np.ndarray
    omega_hat: np.ndarray
    q_des: np.ndarray
    euler: np.ndarray
    euler_des: np.ndarray
    att_error: np.ndarray
    att_error_hat: np.ndarray | None
    est_att_error: np.ndarray | None
    controller: str
    estimator: str
    scenario: str = "slew"
    plot_path: Path | None = None
    gif_path: Path | None = None

    @property
    def final_att_error_deg(self) -> float:
        return float(np.rad2deg(self.att_error[-1]))


def make_scenario_config(
    scenario: str = "slew",
    *,
    dt: float = 0.01,
    t_final: float | None = None,
    controller: str = "pid",
    estimator: str = "mekf",
    angle_deg: float = 75.0,
    tau_dist: np.ndarray | None = None,
    actuator_tau_max: float | np.ndarray | None = None,
    actuator_tau: float | None = None,
    use_mag: bool = True,
    use_sun: bool = True,
    seed: int = 1,
    plot: bool = True,
    gif: bool = True,
    out_dir: Path = Path("outputs"),
) -> SimConfig:
    """Named SimLab presets. Plant / controller / estimator cores are unchanged."""
    name = scenario.lower()
    if name not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; expected one of {SCENARIOS}")
    dist = np.zeros(3) if tau_dist is None else np.asarray(tau_dist, dtype=float).reshape(3)
    if name == "detumble":
        return SimConfig(
            dt=dt,
            t_final=30.0 if t_final is None else t_final,
            controller=controller,
            estimator=estimator,
            scenario="detumble",
            q0=axis_angle_to_quat(DETUMBLE_Q0_AXIS, np.deg2rad(40.0)),
            omega0=DETUMBLE_OMEGA0.copy(),
            q_des=np.array([1.0, 0.0, 0.0, 0.0]),
            tau_dist=dist,
            actuator_tau_max=actuator_tau_max,
            actuator_tau=actuator_tau,
            use_mag=use_mag,
            use_sun=use_sun,
            seed=seed,
            plot=plot,
            gif=gif,
            out_dir=out_dir,
        )
    return SimConfig(
        dt=dt,
        t_final=40.0 if t_final is None else t_final,
        controller=controller,
        estimator=estimator,
        scenario="slew",
        q_des=axis_angle_to_quat(SLEW_AXIS, np.deg2rad(angle_deg)),
        tau_dist=dist,
        actuator_tau_max=actuator_tau_max,
        actuator_tau=actuator_tau,
        use_mag=use_mag,
        use_sun=use_sun,
        seed=seed,
        plot=plot,
        gif=gif,
        out_dir=out_dir,
    )


def make_sim_estimator(
    cfg: SimConfig,
    q0: np.ndarray,
) -> ComplementaryFilter | MultiplicativeEKF | None:
    """Build the SimLab estimator, forwarding gyro densities to the MEKF.

    ``SimConfig.gyro_sigma_v`` / ``gyro_sigma_u`` are the truth-gyro ARW/RRW
    densities *and* the MEKF Farrenkopf process-noise densities.  Mahony has
    no process-noise matrix; ``truth`` returns ``None``.
    """
    mode = cfg.estimator.lower()
    if mode in {"mekf", "kalman", "ekf"}:
        return make_estimator(
            mode,
            q0=q0,
            sigma_v=cfg.gyro_sigma_v,
            sigma_u=cfg.gyro_sigma_u,
        )
    return make_estimator(mode, q0=q0)


def run_slew(cfg: SimConfig | None = None) -> SimLog:
    """Closed-loop SimLab run (slew or detumble); optionally writes plot/GIF.

    ``run_sim`` is a public alias — this is not slew-only.
    """
    cfg = cfg if cfg is not None else SimConfig()
    if cfg.dt <= 0.0:
        raise ValueError("dt must be positive")
    if cfg.t_final < 0.0:
        raise ValueError("t_final must be non-negative")
    rng = np.random.default_rng(cfg.seed)
    body = RigidBody(cfg.inertia)
    ctrl = make_controller(
        cfg.controller,
        cfg.inertia,
        torque_limit=cfg.torque_limit,
        gain_scale=cfg.gain_scale,
    )
    ctrl.reset()
    actuator = make_actuator(tau_max=cfg.actuator_tau_max, time_constant=cfg.actuator_tau)
    actuator.reset()

    q = quat_normalize(cfg.q0)
    omega = np.asarray(cfg.omega0, dtype=float).reshape(3).copy()
    q_des = quat_normalize(cfg.q_des)
    tau_dist = np.asarray(cfg.tau_dist, dtype=float).reshape(3)

    estimator = make_sim_estimator(cfg, q)
    # Full-state feedback does not consume measurements. Skip gyro / vector
    # construction and sampling so the truth path stays cheap.
    gyro: GyroModel | None = None
    sensors: list[VectorSensor] = []
    if estimator is not None:
        gyro = GyroModel(
            sigma_v=cfg.gyro_sigma_v,
            sigma_u=cfg.gyro_sigma_u,
            bias=np.asarray(cfg.gyro_bias, dtype=float).copy(),
            seed=rng,
        )
        if cfg.use_mag:
            sensors.append(
                VectorSensor(
                    v_inertial=np.array([0.3, 0.1, 0.95]),
                    sigma=cfg.mag_sigma,
                    seed=rng,
                    name="mag",
                )
            )
        if cfg.use_sun:
            sensors.append(
                VectorSensor(
                    v_inertial=np.array([1.0, 0.05, 0.02]),
                    sigma=cfg.sun_sigma,
                    seed=rng,
                    name="sun",
                )
            )
        if not sensors:
            warnings.warn(
                f"estimator {cfg.estimator!r} is running with no vector sensors "
                "(gyro-only); full attitude is not observable from rate alone",
                UserWarning,
                stacklevel=2,
            )

    n = int(np.round(cfg.t_final / cfg.dt)) + 1
    t = np.arange(n, dtype=float) * cfg.dt
    q_hist = np.zeros((n, 4))
    w_hist = np.zeros((n, 3))
    tau_hist = np.zeros((n, 3))
    qh_hist = np.zeros((n, 4))
    wh_hist = np.zeros((n, 3))

    for k in range(n):
        q_hist[k] = q
        w_hist[k] = omega

        if estimator is None or gyro is None:
            q_hat, omega_hat = q.copy(), omega.copy()
        else:
            omega_m = gyro.measure(omega, cfg.dt)
            vecs = vectors_from_sensors(q, sensors) if sensors else None
            q_hat, omega_hat = estimator.step(omega_m, cfg.dt, vecs)

        qh_hist[k] = q_hat
        wh_hist[k] = omega_hat
        tau_cmd = ctrl.command(q_hat, omega_hat, q_des, omega_des=None, dt=cfg.dt)
        tau = actuator.apply(tau_cmd, cfg.dt)
        tau_hist[k] = tau
        # τ[k] is held over [t[k], t[k+1]).  Do not take an extra unused
        # plant step after the last logged sample.
        if k + 1 < n:
            q, omega = step_rigid_body(body, q, omega, tau + tau_dist, cfg.dt)

    euler = np.vstack([quat_to_euler321(qi) for qi in q_hist])
    att_error = np.array([geodesic_angle(qi, q_des) for qi in q_hist])
    att_error_hat = np.array([geodesic_angle(qi, q_des) for qi in qh_hist])
    if estimator is None:
        est_att_error = None
        att_error_hat_out = None
    else:
        est_att_error = np.array([geodesic_angle(qh_hist[i], q_hist[i]) for i in range(n)])
        att_error_hat_out = att_error_hat

    log = SimLog(
        t=t,
        q=q_hist,
        omega=w_hist,
        tau=tau_hist,
        q_hat=qh_hist,
        omega_hat=wh_hist,
        q_des=q_des,
        euler=euler,
        euler_des=quat_to_euler321(q_des),
        att_error=att_error,
        att_error_hat=att_error_hat_out,
        est_att_error=est_att_error,
        controller=cfg.controller,
        estimator=cfg.estimator,
        scenario=cfg.scenario,
    )

    if cfg.plot or cfg.gif:
        from attitude_sim.plots import plot_slew, write_attitude_gif

        out = Path(cfg.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = cfg.artifact_stem
        if cfg.plot:
            log.plot_path = plot_slew(log, out / f"{stem}_summary.png")
        if cfg.gif:
            log.gif_path = write_attitude_gif(log, out / f"{stem}_attitude.gif")
    return log


run_sim = run_slew


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m attitude_sim",
        description=(
            "Milestone 1 SimLab: closed-loop rigid-body attitude scenarios "
            "(dynamics + control + estimation)."
        ),
    )
    p.add_argument(
        "--scenario",
        choices=SCENARIOS,
        default="slew",
        help="slew = 75° rest-to-rest; detumble = dump body rate then recover identity",
    )
    p.add_argument("--controller", choices=("pid", "lqr"), default="pid", help="feedback law")
    p.add_argument(
        "--estimator",
        choices=("truth", "mekf", "mahony"),
        default="mekf",
        help="controller measurement source (truth = full-state feedback)",
    )
    p.add_argument(
        "--t-final",
        type=float,
        default=None,
        help="run duration (s); default 40 slew / 30 detumble",
    )
    p.add_argument("--dt", type=float, default=0.01, help="sample / RK4 step (s)")
    p.add_argument("--out-dir", type=Path, default=Path("outputs"), help="plot/GIF directory")
    p.add_argument("--no-plot", action="store_true", help="skip PNG summary")
    p.add_argument("--no-gif", action="store_true", help="skip attitude GIF")
    p.add_argument("--no-mag", action="store_true", help="disable magnetometer")
    p.add_argument("--no-sun", action="store_true", help="disable sun sensor")
    p.add_argument("--seed", type=int, default=1, help="RNG seed for sensors")
    p.add_argument(
        "--angle-deg",
        type=float,
        default=75.0,
        help="commanded principal rotation for --scenario slew (deg); ignored for detumble",
    )
    p.add_argument(
        "--tau-dist",
        default="0,0,0",
        help="constant body-frame disturbance torque [N·m], comma-separated (e.g. 0.002,0,0)",
    )
    p.add_argument(
        "--actuator-tau-max",
        default=None,
        help=(
            "per-axis reaction-wheel torque limit [N·m]: scalar or x,y,z "
            "(default: unlimited; controller Euclidean |τ| clamp is unchanged)"
        ),
    )
    p.add_argument(
        "--actuator-tau",
        type=float,
        default=None,
        help="first-order actuator lag time constant [s] (default: none / instantaneous)",
    )
    return p


def _parse_vec3(text: str, name: str) -> np.ndarray:
    parts = [p.strip() for p in str(text).split(",")]
    if len(parts) != 3:
        raise ValueError(f"{name} must be three comma-separated numbers, got {text!r}")
    try:
        return np.array([float(p) for p in parts], dtype=float)
    except ValueError as exc:
        raise ValueError(f"{name} must be three comma-separated numbers, got {text!r}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        tau_dist = _parse_vec3(args.tau_dist, "--tau-dist")
        actuator_tau_max = parse_tau_max(args.actuator_tau_max, "--actuator-tau-max")
    except ValueError as exc:
        parser.error(str(exc))
    if args.dt <= 0.0:
        parser.error("--dt must be positive")
    if args.actuator_tau is not None and args.actuator_tau < 0.0:
        parser.error("--actuator-tau must be >= 0")
    cfg = make_scenario_config(
        args.scenario,
        dt=args.dt,
        t_final=args.t_final,
        controller=args.controller,
        estimator=args.estimator,
        angle_deg=args.angle_deg,
        tau_dist=tau_dist,
        actuator_tau_max=actuator_tau_max,
        actuator_tau=args.actuator_tau,
        use_mag=not args.no_mag,
        use_sun=not args.no_sun,
        seed=args.seed,
        plot=not args.no_plot,
        gif=not args.no_gif,
        out_dir=args.out_dir,
    )
    log = run_slew(cfg)
    print(
        f"{log.scenario} complete: controller={log.controller} estimator={log.estimator} "
        f"final_att_error={log.final_att_error_deg:.3f} deg  "
        f"final_||omega||={np.linalg.norm(log.omega[-1]):.4f} rad/s"
    )
    if log.plot_path is not None:
        print(f"plot: {log.plot_path}")
    if log.gif_path is not None:
        print(f"gif:  {log.gif_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

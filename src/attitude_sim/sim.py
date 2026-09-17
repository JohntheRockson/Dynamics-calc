"""SimLab: closed-loop scenarios, CLI, and ``python -m attitude_sim``."""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from attitude_sim.actuators import make_actuator, parse_tau_max
from attitude_sim.controls import make_controller
from attitude_sim.disturbances import (
    CircularOrbit,
    EnvironmentalTorques,
    GravityGradientTorque,
    ResidualDipoleTorque,
)
from attitude_sim.estimation import (
    ComplementaryFilter,
    InnovationLog,
    MultiplicativeEKF,
    make_estimator,
    triad_q0_from_sensors,
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
DIPOLE_MODELS = ("tilted", "orbit_normal")
# LEO-scale circular orbit used only when gravity-gradient / residual-dipole
# flags are on. Defaults preserve the prior constant-τ_d-only plant.
DEFAULT_ORBIT_RADIUS = 7.0e6
DEFAULT_DIPOLE_M = np.array([0.10, 0.0, 0.0])


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
    gravity_gradient: bool = False
    residual_dipole: bool = False
    orbit_radius: float = DEFAULT_ORBIT_RADIUS
    orbit_inclination_deg: float = 0.0
    orbit_raan_deg: float = 0.0
    dipole_m: np.ndarray = field(default_factory=lambda: DEFAULT_DIPOLE_M.copy())
    dipole_model: str = "tilted"
    gyro_sigma_v: float = 5e-4
    gyro_sigma_u: float = 1e-6
    gyro_bias: np.ndarray = field(default_factory=lambda: np.array([0.002, -0.001, 0.0015]))
    mag_sigma: float = 3e-3
    sun_sigma: float = 2e-3
    use_mag: bool = True
    use_sun: bool = True
    coarse_init: bool = False
    seed: int = 1
    plot: bool = True
    gif: bool = True
    out_dir: Path = field(default_factory=lambda: Path("outputs"))
    log_innovations: Path | None = None

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
    innovation_csv: Path | None = None
    mean_nis: float | None = None

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
    gravity_gradient: bool = False,
    residual_dipole: bool = False,
    orbit_radius: float | None = None,
    orbit_inclination_deg: float | None = None,
    orbit_raan_deg: float | None = None,
    dipole_m: np.ndarray | None = None,
    dipole_model: str | None = None,
    use_mag: bool = True,
    use_sun: bool = True,
    coarse_init: bool = False,
    gyro_sigma_v: float | None = None,
    gyro_sigma_u: float | None = None,
    mag_sigma: float | None = None,
    sun_sigma: float | None = None,
    seed: int = 1,
    plot: bool = True,
    gif: bool = True,
    out_dir: Path = Path("outputs"),
    log_innovations: Path | None = None,
) -> SimConfig:
    """Named SimLab presets. Plant / controller / estimator cores are unchanged.

    ``gyro_sigma_v`` / ``gyro_sigma_u`` (and mag/sun ``sigma``) retune the
    truth sensors *and* — via ``make_sim_estimator`` — the MEKF Farrenkopf
    ``Q_d``.  ``None`` keeps the ``SimConfig`` defaults.

    ``gravity_gradient`` / ``residual_dipole`` default off so the stock
    demos stay on the constant-``τ_d`` plant.  Orbit / dipole knobs are
    stored even when the models are off.
    """
    name = scenario.lower()
    if name not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; expected one of {SCENARIOS}")
    dist = np.zeros(3) if tau_dist is None else np.asarray(tau_dist, dtype=float).reshape(3)
    if name == "detumble":
        cfg = SimConfig(
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
            gravity_gradient=gravity_gradient,
            residual_dipole=residual_dipole,
            use_mag=use_mag,
            use_sun=use_sun,
            coarse_init=coarse_init,
            seed=seed,
            plot=plot,
            gif=gif,
            out_dir=out_dir,
        )
    else:
        cfg = SimConfig(
            dt=dt,
            t_final=40.0 if t_final is None else t_final,
            controller=controller,
            estimator=estimator,
            scenario="slew",
            q_des=axis_angle_to_quat(SLEW_AXIS, np.deg2rad(angle_deg)),
            tau_dist=dist,
            actuator_tau_max=actuator_tau_max,
            actuator_tau=actuator_tau,
            gravity_gradient=gravity_gradient,
            residual_dipole=residual_dipole,
            use_mag=use_mag,
            use_sun=use_sun,
            coarse_init=coarse_init,
            seed=seed,
            plot=plot,
            gif=gif,
            out_dir=out_dir,
        )
    if orbit_radius is not None:
        cfg.orbit_radius = float(orbit_radius)
    if orbit_inclination_deg is not None:
        cfg.orbit_inclination_deg = float(orbit_inclination_deg)
    if orbit_raan_deg is not None:
        cfg.orbit_raan_deg = float(orbit_raan_deg)
    if dipole_m is not None:
        cfg.dipole_m = np.asarray(dipole_m, dtype=float).reshape(3)
    if dipole_model is not None:
        cfg.dipole_model = str(dipole_model)
    if gyro_sigma_v is not None:
        cfg.gyro_sigma_v = float(gyro_sigma_v)
    if gyro_sigma_u is not None:
        cfg.gyro_sigma_u = float(gyro_sigma_u)
    if mag_sigma is not None:
        cfg.mag_sigma = float(mag_sigma)
    if sun_sigma is not None:
        cfg.sun_sigma = float(sun_sigma)
    if log_innovations is not None:
        cfg.log_innovations = Path(log_innovations)
    return cfg


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


def make_sim_disturbances(cfg: SimConfig) -> EnvironmentalTorques | None:
    """Optional gravity-gradient / residual-dipole models (default off).

    When both flags are false this returns ``None`` so the closed-loop
    plant matches the prior constant-``τ_d``-only contract.  Enabled
    models share one bound :class:`CircularOrbit` and are sampled ZOH at
    the left endpoint of each SimLab step (same hold as the actuator).
    """
    if not cfg.gravity_gradient and not cfg.residual_dipole:
        return None
    radius = float(cfg.orbit_radius)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("orbit_radius must be positive")
    inc = float(cfg.orbit_inclination_deg)
    raan = float(cfg.orbit_raan_deg)
    if not np.isfinite(inc) or not np.isfinite(raan):
        raise ValueError("orbit inclination and RAAN must be finite")
    model = str(cfg.dipole_model)
    if model not in DIPOLE_MODELS:
        raise ValueError(f"dipole_model must be one of {DIPOLE_MODELS}")
    orbit = CircularOrbit(
        radius=radius,
        inclination=np.deg2rad(inc),
        raan=np.deg2rad(raan),
    )
    gg = GravityGradientTorque(cfg.inertia, orbit=orbit) if cfg.gravity_gradient else None
    mag = None
    if cfg.residual_dipole:
        mag = ResidualDipoleTorque(
            np.asarray(cfg.dipole_m, dtype=float).reshape(3),
            orbit=orbit,
            model=model,
        )
    return EnvironmentalTorques(gravity_gradient=gg, residual_dipole=mag)


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
    env = make_sim_disturbances(cfg)

    # Full-state feedback does not consume measurements. Skip gyro / vector
    # construction and sampling so the truth path stays cheap. Coarse TRIAD
    # init only applies when an estimator will run.
    gyro: GyroModel | None = None
    sensors: list[VectorSensor] = []
    q_est0 = q
    if cfg.estimator.lower() != "truth":
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
        if cfg.coarse_init:
            try:
                q_est0 = triad_q0_from_sensors(q, sensors)
            except ValueError as exc:
                warnings.warn(
                    f"coarse TRIAD init skipped ({exc}); estimator starts at true q0",
                    UserWarning,
                    stacklevel=2,
                )
                q_est0 = q
    estimator = make_sim_estimator(cfg, q_est0)
    if estimator is not None and not sensors:
        warnings.warn(
            f"estimator {cfg.estimator!r} is running with no vector sensors "
            "(gyro-only); full attitude is not observable from rate alone",
            UserWarning,
            stacklevel=2,
        )
    if cfg.log_innovations is not None:
        if isinstance(estimator, MultiplicativeEKF):
            if estimator.innovation_log is None:
                estimator.innovation_log = InnovationLog()
        else:
            warnings.warn(
                "SimConfig.log_innovations is MEKF-only "
                f"(estimator={cfg.estimator!r}); skipping",
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
            q_hat, omega_hat = estimator.step(omega_m, cfg.dt, vecs, t=float(t[k]))

        qh_hist[k] = q_hat
        wh_hist[k] = omega_hat
        tau_cmd = ctrl.command(q_hat, omega_hat, q_des, omega_des=None, dt=cfg.dt)
        tau = actuator.apply(tau_cmd, cfg.dt)
        tau_hist[k] = tau
        # τ[k] is held over [t[k], t[k+1]).  Do not take an extra unused
        # plant step after the last logged sample.
        if k + 1 < n:
            tau_env = (
                env.tau_body(q, omega, float(t[k])) if env is not None else np.zeros(3)
            )
            q, omega = step_rigid_body(body, q, omega, tau + tau_dist + tau_env, cfg.dt)

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

    if (
        isinstance(estimator, MultiplicativeEKF)
        and estimator.innovation_log is not None
        and cfg.log_innovations is not None
    ):
        log.innovation_csv = estimator.innovation_log.write_csv(cfg.log_innovations)
        if estimator.innovation_log.samples:
            log.mean_nis = estimator.innovation_log.mean_nis()

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
    p.add_argument(
        "--coarse-init",
        action="store_true",
        help=(
            "TRIAD coarse attitude from mag+sun at t=0 so MEKF/Mahony need not "
            "start at true q0 (default: start at true q0, current demo)"
        ),
    )
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
        "--gravity-gradient",
        action="store_true",
        help=(
            "add gravity-gradient τ_gg = 3(μ/r³)(r̂_b × J r̂_b) on a circular orbit "
            "(default: off; prior demos unchanged)"
        ),
    )
    p.add_argument(
        "--residual-dipole",
        action="store_true",
        help=(
            "add residual-dipole τ_m = m_b × B_b from a frozen Earth dipole "
            "(default: off; prior demos unchanged)"
        ),
    )
    p.add_argument(
        "--orbit-radius",
        type=float,
        default=DEFAULT_ORBIT_RADIUS,
        metavar="M",
        help="circular-orbit radius [m] for env models (default: 7e6 LEO-scale)",
    )
    p.add_argument(
        "--orbit-inc-deg",
        type=float,
        default=0.0,
        metavar="DEG",
        help="orbit inclination [deg] for env models (default: 0, equatorial)",
    )
    p.add_argument(
        "--orbit-raan-deg",
        type=float,
        default=0.0,
        metavar="DEG",
        help="orbit RAAN [deg] for env models (default: 0)",
    )
    p.add_argument(
        "--dipole-m",
        default="0.1,0,0",
        help="residual body dipole [A·m²], comma-separated (used with --residual-dipole)",
    )
    p.add_argument(
        "--dipole-model",
        choices=DIPOLE_MODELS,
        default="tilted",
        help="Earth field for --residual-dipole: tilted (default 11.5°) or orbit_normal",
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
    p.add_argument(
        "--gyro-sigma-v",
        type=float,
        default=None,
        metavar="SIGMA",
        help=(
            "gyro ARW density σ_v [rad/s/√Hz]; also MEKF Farrenkopf Qd (default: SimConfig 5e-4)"
        ),
    )
    p.add_argument(
        "--gyro-sigma-u",
        type=float,
        default=None,
        metavar="SIGMA",
        help=(
            "gyro RRW density σ_u [rad/s²/√Hz]; also MEKF Farrenkopf Qd (default: SimConfig 1e-6)"
        ),
    )
    p.add_argument(
        "--mag-sigma",
        type=float,
        default=None,
        metavar="SIGMA",
        help="magnetometer Cartesian σ (default: SimConfig 3e-3)",
    )
    p.add_argument(
        "--sun-sigma",
        type=float,
        default=None,
        metavar="SIGMA",
        help="sun-sensor Cartesian σ (default: SimConfig 2e-3)",
    )
    p.add_argument(
        "--log-innovations",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "write MEKF vector NIS / innovation CSV (default: off). "
            "Mahony/truth skip with a warning"
        ),
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
        dipole_m = _parse_vec3(args.dipole_m, "--dipole-m")
        actuator_tau_max = parse_tau_max(args.actuator_tau_max, "--actuator-tau-max")
    except ValueError as exc:
        parser.error(str(exc))
    if args.dt <= 0.0:
        parser.error("--dt must be positive")
    if args.actuator_tau is not None and args.actuator_tau < 0.0:
        parser.error("--actuator-tau must be >= 0")
    if args.orbit_radius <= 0.0:
        parser.error("--orbit-radius must be positive")
    for flag, value in (
        ("--orbit-inc-deg", args.orbit_inc_deg),
        ("--orbit-raan-deg", args.orbit_raan_deg),
    ):
        if not np.isfinite(value):
            parser.error(f"{flag} must be finite")
    for flag, value in (
        ("--gyro-sigma-v", args.gyro_sigma_v),
        ("--gyro-sigma-u", args.gyro_sigma_u),
        ("--mag-sigma", args.mag_sigma),
        ("--sun-sigma", args.sun_sigma),
    ):
        if value is not None and value < 0.0:
            parser.error(f"{flag} must be >= 0")
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
        gravity_gradient=args.gravity_gradient,
        residual_dipole=args.residual_dipole,
        orbit_radius=args.orbit_radius,
        orbit_inclination_deg=args.orbit_inc_deg,
        orbit_raan_deg=args.orbit_raan_deg,
        dipole_m=dipole_m,
        dipole_model=args.dipole_model,
        use_mag=not args.no_mag,
        use_sun=not args.no_sun,
        coarse_init=args.coarse_init,
        gyro_sigma_v=args.gyro_sigma_v,
        gyro_sigma_u=args.gyro_sigma_u,
        mag_sigma=args.mag_sigma,
        sun_sigma=args.sun_sigma,
        seed=args.seed,
        plot=not args.no_plot,
        gif=not args.no_gif,
        out_dir=args.out_dir,
        log_innovations=args.log_innovations,
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
    if log.innovation_csv is not None:
        nis_txt = ""
        if log.mean_nis is not None:
            nis_txt = f"  mean_NIS={log.mean_nis:.3f} (χ²_2, E=2)"
        print(f"innovations: {log.innovation_csv}{nis_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

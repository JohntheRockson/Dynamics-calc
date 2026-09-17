"""SimLab: rest-to-rest slew scenario, CLI, and ``python -m attitude_sim``."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from attitude_sim.controls import make_controller
from attitude_sim.estimation import make_estimator, vectors_from_sensors
from attitude_sim.plant import RigidBody, step_rigid_body
from attitude_sim.quaternions import (
    axis_angle_to_quat,
    geodesic_angle,
    quat_normalize,
    quat_to_euler321,
)
from attitude_sim.sensors import GyroModel, VectorSensor


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
    q0: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    omega0: np.ndarray = field(default_factory=lambda: np.zeros(3))
    q_des: np.ndarray = field(
        default_factory=lambda: axis_angle_to_quat(np.array([0.2, 0.5, 0.84]), np.deg2rad(75.0))
    )
    torque_limit: float = 0.02
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
    plot_path: Path | None = None
    gif_path: Path | None = None

    @property
    def final_att_error_deg(self) -> float:
        return float(np.rad2deg(self.att_error[-1]))


def run_slew(cfg: SimConfig | None = None) -> SimLog:
    """Closed-loop rest-to-rest slew; optionally writes plot/GIF under ``out_dir``."""
    cfg = cfg if cfg is not None else SimConfig()
    rng = np.random.default_rng(cfg.seed)
    body = RigidBody(cfg.inertia)
    ctrl = make_controller(cfg.controller, cfg.inertia, torque_limit=cfg.torque_limit)
    ctrl.reset()

    q = quat_normalize(cfg.q0)
    omega = np.asarray(cfg.omega0, dtype=float).reshape(3).copy()
    q_des = quat_normalize(cfg.q_des)

    gyro = GyroModel(
        sigma_v=cfg.gyro_sigma_v,
        sigma_u=cfg.gyro_sigma_u,
        bias=np.asarray(cfg.gyro_bias, dtype=float).copy(),
        seed=rng,
    )
    sensors: list[VectorSensor] = []
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

    estimator = make_estimator(cfg.estimator, q0=q)

    n = int(np.round(cfg.t_final / cfg.dt)) + 1
    t = np.arange(n, dtype=float) * cfg.dt
    q_hist = np.zeros((n, 4))
    w_hist = np.zeros((n, 3))
    tau_hist = np.zeros((n, 3))
    qh_hist = np.zeros((n, 4))
    wh_hist = np.zeros((n, 3))

    for k, tk in enumerate(t):
        q_hist[k] = q
        w_hist[k] = omega

        omega_m = gyro.measure(omega, cfg.dt)
        vecs = vectors_from_sensors(q, sensors) if sensors else None
        if estimator is None:
            q_hat, omega_hat = q.copy(), omega.copy()
        else:
            q_hat, omega_hat = estimator.step(omega_m, cfg.dt, vecs)

        qh_hist[k] = q_hat
        wh_hist[k] = omega_hat
        tau = ctrl.command(q_hat, omega_hat, q_des, omega_des=None, dt=cfg.dt)
        tau_hist[k] = tau
        q, omega = step_rigid_body(body, q, omega, tau, cfg.dt)

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
    )

    if cfg.plot or cfg.gif:
        from attitude_sim.plots import plot_slew, write_attitude_gif

        out = Path(cfg.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        if cfg.plot:
            log.plot_path = plot_slew(log, out / "slew_summary.png")
        if cfg.gif:
            log.gif_path = write_attitude_gif(log, out / "slew_attitude.gif")
    return log


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m attitude_sim",
        description="Milestone 1 rest-to-rest rigid-body attitude slew (dynamics + control + estimation).",
    )
    p.add_argument("--controller", choices=("pid", "lqr"), default="pid", help="feedback law")
    p.add_argument(
        "--estimator",
        choices=("truth", "mekf", "mahony"),
        default="mekf",
        help="controller measurement source (truth = full-state feedback)",
    )
    p.add_argument("--t-final", type=float, default=40.0, help="slew duration (s)")
    p.add_argument("--dt", type=float, default=0.01, help="sample / RK4 step (s)")
    p.add_argument("--out-dir", type=Path, default=Path("outputs"), help="plot/GIF directory")
    p.add_argument("--no-plot", action="store_true", help="skip PNG summary")
    p.add_argument("--no-gif", action="store_true", help="skip attitude GIF")
    p.add_argument("--no-mag", action="store_true", help="disable magnetometer")
    p.add_argument("--no-sun", action="store_true", help="disable sun sensor")
    p.add_argument("--seed", type=int, default=1, help="RNG seed for sensors")
    p.add_argument("--angle-deg", type=float, default=75.0, help="commanded principal rotation (deg)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    q_des = axis_angle_to_quat(np.array([0.2, 0.5, 0.84]), np.deg2rad(args.angle_deg))
    cfg = SimConfig(
        dt=args.dt,
        t_final=args.t_final,
        controller=args.controller,
        estimator=args.estimator,
        q_des=q_des,
        use_mag=not args.no_mag,
        use_sun=not args.no_sun,
        seed=args.seed,
        plot=not args.no_plot,
        gif=not args.no_gif,
        out_dir=args.out_dir,
    )
    log = run_slew(cfg)
    print(
        f"slew complete: controller={log.controller} estimator={log.estimator} "
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

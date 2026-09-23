"""Matplotlib figures and an optional 3-D attitude GIF for a SimLab run."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from attitude_sim.mrp import quat_to_mrp
from attitude_sim.quaternions import quat_error, quat_to_rotation
from attitude_sim.scenarios import SCENARIO_TITLES

_SCENARIO_TITLES = dict(SCENARIO_TITLES)


def _scenario_title(log) -> str:
    name = getattr(log, "scenario", "slew")
    return _SCENARIO_TITLES.get(name, name)


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.35,
            "font.size": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 8,
        }
    )


def plot_slew(log, path: Path) -> Path:
    """Write a 3x2 summary figure of quaternion, Euler, rates, torque, errors."""
    _style()
    t = log.t
    euler_deg = np.rad2deg(log.euler)
    euler_des_deg = np.rad2deg(log.euler_des)
    att_err_deg = np.rad2deg(log.att_error)
    fig, axes = plt.subplots(3, 2, figsize=(11.5, 9.5), constrained_layout=True)

    labels_q = [r"$q_w$", r"$q_x$", r"$q_y$", r"$q_z$"]
    ax = axes[0, 0]
    for i, lab in enumerate(labels_q):
        ax.plot(t, log.q[:, i], label=lab)
        ax.plot(t, np.full_like(t, log.q_des[i]), ls="--", lw=1, alpha=0.7, color=f"C{i}")
    ax.set_ylabel("quaternion")
    ax.set_title("Attitude quaternion (dashed = command)")
    ax.legend(ncol=4, loc="best")

    ax = axes[0, 1]
    eul_labs = ["yaw", "pitch", "roll"]
    for i, lab in enumerate(eul_labs):
        ax.plot(t, euler_deg[:, i], label=lab)
        ax.plot(t, np.full_like(t, euler_des_deg[i]), ls="--", lw=1, alpha=0.7, color=f"C{i}")
    ax.set_ylabel("deg")
    ax.set_title("Euler 3-2-1 (yaw-pitch-roll)")
    ax.legend()

    ax = axes[1, 0]
    for i, lab in enumerate([r"$\omega_x$", r"$\omega_y$", r"$\omega_z$"]):
        ax.plot(t, log.omega[:, i], label=lab)
    ax.set_ylabel("rad/s")
    ax.set_title("Body rates")
    ax.legend()

    ax = axes[1, 1]
    for i, lab in enumerate([r"$\tau_x$", r"$\tau_y$", r"$\tau_z$"]):
        ax.plot(t, log.tau[:, i], label=lab)
    ax.set_ylabel("N·m")
    ax.set_title("Control torque")
    ax.legend()

    ax = axes[2, 0]
    ax.plot(t, att_err_deg, color="C3", label="true")
    if log.att_error_hat is not None:
        ax.plot(t, np.rad2deg(log.att_error_hat), color="C2", ls="--", label="from estimate")
    ax.set_ylabel("deg")
    ax.set_xlabel("t (s)")
    ax.set_title("Geodesic attitude error to command")
    ax.legend()

    ax = axes[2, 1]
    if log.est_att_error is not None:
        rate_err_deg = np.rad2deg(np.linalg.norm(log.omega_hat - log.omega, axis=1))
        l1 = ax.plot(t, np.rad2deg(log.est_att_error), color="C0", label="att. err (deg)")
        ax.set_ylabel("deg", color="C0")
        ax2 = ax.twinx()
        l2 = ax2.plot(t, rate_err_deg, color="C1", alpha=0.55, lw=0.8, label="rate err (deg/s)")
        ax2.set_ylabel("deg/s", color="C1")
        ax2.grid(False)
        lines = l1 + l2
        ax.legend(lines, [ln.get_label() for ln in lines], loc="upper right")
        ax.set_title("Estimator error")
    else:
        ax.plot(t, np.rad2deg(np.linalg.norm(log.omega, axis=1)), color="C1")
        ax.set_title(r"Rate magnitude $||\omega||$")
        ax.set_ylabel("deg/s")
    ax.set_xlabel("t (s)")

    fig.suptitle(
        f"{_scenario_title(log)}  |  controller={log.controller}  estimator={log.estimator}",
        fontsize=13,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def attitude_error_mrp(q: np.ndarray, q_des: np.ndarray) -> np.ndarray:
    """Shadow-switched MRP of the attitude error ``q_e = q_des* ⊗ q``.

    Post-process only: does not change the plant state.  ``||σ_e|| = tan(Φ/4)``
    for geodesic angle ``Φ``.
    """
    qe = quat_error(q, q_des)
    return quat_to_mrp(qe, switch=True)


def mrp_error_from_log(log) -> np.ndarray:
    """``(n, 3)`` MRP attitude error from logged quaternions (not the integrator)."""
    q_des = np.asarray(log.q_des, dtype=float).reshape(4)
    return np.vstack([attitude_error_mrp(qi, q_des) for qi in log.q])


def plot_mrp_error(log, path: Path) -> Path:
    """Write a post-process MRP chart of logged ``q`` vs ``q_des``.

    The plant still integrates on S^3; this figure is ``σ(q_e)`` reconstructed
    from the quaternion log (optional extra PNG, not a seventh summary panel).
    """
    _style()
    t = log.t
    sigma_e = mrp_error_from_log(log)
    sigma_q = np.vstack([quat_to_mrp(qi, switch=True) for qi in log.q])
    sigma_des = quat_to_mrp(log.q_des, switch=True)
    phi = np.asarray(log.att_error, dtype=float)
    tan_quarter = np.tan(0.25 * phi)

    fig, axes = plt.subplots(3, 1, figsize=(8.6, 8.4), constrained_layout=True, sharex=True)

    ax = axes[0]
    for i, lab in enumerate([r"$\sigma_{e,x}$", r"$\sigma_{e,y}$", r"$\sigma_{e,z}$"]):
        ax.plot(t, sigma_e[:, i], label=lab)
    ax.set_ylabel("MRP")
    ax.set_title(r"Attitude-error MRP $\sigma(q_{\mathrm{des}}^{\ast}\otimes q)$ (shadow-switched)")
    ax.legend(ncol=3, loc="best")

    ax = axes[1]
    ax.plot(t, np.linalg.norm(sigma_e, axis=1), color="C3", label=r"$||\sigma_e||$")
    ax.plot(t, tan_quarter, color="C0", ls="--", lw=1.2, label=r"$\tan(\Phi/4)$ from geodesic")
    ax.set_ylabel("MRP mag")
    ax.set_title("Chart magnitude vs geodesic quarter-angle")
    ax.legend(loc="best")

    ax = axes[2]
    for i, lab in enumerate([r"$\sigma_x$", r"$\sigma_y$", r"$\sigma_z$"]):
        ax.plot(t, sigma_q[:, i], label=lab)
        ax.plot(t, np.full_like(t, sigma_des[i]), ls="--", lw=1, alpha=0.7, color=f"C{i}")
    ax.set_ylabel("MRP")
    ax.set_xlabel("t (s)")
    ax.set_title(r"Body MRP $\sigma(q)$ (dashed = command)")
    ax.legend(ncol=3, loc="best")

    fig.suptitle(
        f"MRP chart (post-process)  |  {_scenario_title(log)}  |  "
        f"{log.controller}+{log.estimator}",
        fontsize=12,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_env_torque(log, path: Path) -> Path:
    """Write environmental-torque history plus the opposing control command."""
    _style()
    t = log.t
    tau_env = np.asarray(getattr(log, "tau_env", np.zeros((len(t), 3))), dtype=float)
    if tau_env.size == 0:
        tau_env = np.zeros((len(t), 3))
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 6.4), constrained_layout=True, sharex=True)

    ax = axes[0]
    for i, lab in enumerate([r"$\tau_{\mathrm{env},x}$", r"$\tau_{\mathrm{env},y}$", r"$\tau_{\mathrm{env},z}$"]):
        ax.plot(t, tau_env[:, i], label=lab)
    ax.set_ylabel("N·m")
    ax.set_title("Environmental body torque (GG + residual dipole)")
    ax.legend(ncol=3, loc="best")

    ax = axes[1]
    ax.plot(t, np.linalg.norm(tau_env, axis=1), color="C3", label=r"$||\tau_{\mathrm{env}}||$")
    ax.plot(t, np.linalg.norm(log.tau, axis=1), color="C0", ls="--", label=r"$||\tau_{\mathrm{ctrl}}||$")
    ax.set_ylabel("N·m")
    ax.set_xlabel("t (s)")
    ax.set_title("Torque magnitudes (control should cancel a slow environmental bias)")
    ax.legend(loc="best")

    fig.suptitle(
        f"Environmental torques  |  {_scenario_title(log)}  |  "
        f"{log.controller}+{log.estimator}",
        fontsize=12,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


MC_ERROR_HIST_NAME = "mc_final_att_error_hist.png"
MC_SETTLE_SCATTER_NAME = "mc_settle_vs_noise.png"


def _trial_attr(trial, name: str, default=None):
    if isinstance(trial, dict):
        return trial.get(name, default)
    return getattr(trial, name, default)


def _trial_float(trial, name: str) -> float:
    value = _trial_attr(trial, name, float("nan"))
    if value is None or value == "":
        return float("nan")
    return float(value)


def _trial_failed(trial) -> bool:
    return bool(_trial_attr(trial, "failed", False))


def _mc_title(summary) -> str:
    controller = getattr(summary, "controller", "?")
    estimator = getattr(summary, "estimator", "?")
    n = getattr(summary, "n", len(getattr(summary, "trials", []) or []))
    n_fail = getattr(summary, "n_fail", 0)
    t_final = getattr(summary, "t_final", float("nan"))
    t_txt = f"{t_final:g}" if np.isfinite(float(t_final)) else "?"
    return f"Monte Carlo slew  N={n}  fail={n_fail}  {controller}+{estimator}  t_final={t_txt} s"


def plot_mc_error_histogram(summary, path: Path) -> Path:
    """Histogram of final geodesic attitude error (deg) for healthy trials."""
    _style()
    trials = list(getattr(summary, "trials", []) or [])
    ok = [t for t in trials if not _trial_failed(t)]
    errs = np.array([_trial_float(t, "final_att_error_deg") for t in ok], dtype=float)
    errs = errs[np.isfinite(errs)]

    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    if errs.size:
        bins = min(24, max(8, int(np.sqrt(errs.size)) + 4))
        ax.hist(errs, bins=bins, color="#4c78a8", edgecolor="white", linewidth=0.6)
        mean = getattr(summary, "final_err_mean_deg", float(np.mean(errs)))
        median = getattr(summary, "final_err_median_deg", float(np.median(errs)))
        p95 = getattr(summary, "final_err_p95_deg", float(np.percentile(errs, 95)))
        if np.isfinite(float(mean)):
            ax.axvline(float(mean), color="#e45756", ls="--", lw=1.4, label=f"mean {mean:.3f}°")
        if np.isfinite(float(median)):
            ax.axvline(float(median), color="#f58518", ls="-.", lw=1.4, label=f"median {median:.3f}°")
        if np.isfinite(float(p95)):
            ax.axvline(float(p95), color="#54a24b", ls=":", lw=1.6, label=f"p95 {p95:.3f}°")
        ax.legend(loc="upper right")
    else:
        ax.text(0.5, 0.5, "no finite healthy-trial errors", ha="center", va="center", transform=ax.transAxes)

    ax.set_xlabel("final geodesic attitude error (deg)")
    ax.set_ylabel("trial count")
    ax.set_title("Final attitude-error histogram")
    fig.suptitle(_mc_title(summary), fontsize=12)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_mc_settle_vs_noise(summary, path: Path) -> Path:
    """Scatter of settle-time proxy vs log-uniform sensor-noise scale."""
    _style()
    trials = list(getattr(summary, "trials", []) or [])
    settle_deg = getattr(summary, "settle_deg", float("nan"))
    t_final = float(getattr(summary, "t_final", float("nan")))

    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    n_settled = n_unsettled = n_failed = 0
    for trial in trials:
        scale = _trial_float(trial, "noise_scale")
        settle = _trial_float(trial, "settle_time_s")
        if not np.isfinite(scale):
            continue
        if _trial_failed(trial):
            ax.scatter(
                scale,
                t_final if np.isfinite(t_final) else 0.0,
                marker="x",
                c="#e45756",
                s=36,
                zorder=3,
            )
            n_failed += 1
            continue
        if np.isfinite(settle):
            ax.scatter(
                scale,
                settle,
                marker="o",
                c="#4c78a8",
                s=32,
                edgecolors="k",
                linewidths=0.3,
                zorder=2,
            )
            n_settled += 1
        else:
            y = t_final if np.isfinite(t_final) else 0.0
            ax.scatter(
                scale, y, marker="^", c="#f58518", s=40, edgecolors="k", linewidths=0.3, zorder=2
            )
            n_unsettled += 1

    handles = []
    if n_settled:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="#4c78a8",
                markeredgecolor="k",
                markersize=7,
                label="settled",
            )
        )
    if n_unsettled:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="^",
                color="w",
                markerfacecolor="#f58518",
                markeredgecolor="k",
                markersize=8,
                label="unsettled (at t_final)",
            )
        )
    if n_failed:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="x",
                color="#e45756",
                markersize=7,
                linestyle="None",
                label="failed",
            )
        )
    if handles:
        ax.legend(handles=handles, loc="best")
    if n_settled + n_unsettled + n_failed == 0:
        ax.text(0.5, 0.5, "no trials to plot", ha="center", va="center", transform=ax.transAxes)

    ax.set_xscale("log")
    ax.set_xlabel("sensor-noise scale (log-uniform draw)")
    band = f" (< {settle_deg:g} deg)" if np.isfinite(float(settle_deg)) else ""
    ax.set_ylabel(f"settle-time proxy{band} (s)")
    ax.set_title("Settle time vs noise scale")
    fig.suptitle(_mc_title(summary), fontsize=12)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_monte_carlo(summary, out_dir: Path) -> list[Path]:
    """Write the standard Monte Carlo summary PNGs into ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    return [
        plot_mc_error_histogram(summary, out / MC_ERROR_HIST_NAME),
        plot_mc_settle_vs_noise(summary, out / MC_SETTLE_SCATTER_NAME),
    ]


def write_attitude_gif(log, path: Path, fps: int = 20, max_frames: int = 80) -> Path:
    """Animate a body-fixed box + body axes (inertial frame: X red, Y green, Z blue)."""
    _style()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n = len(log.t)
    stride = max(1, n // max_frames)
    idx = np.arange(0, n, stride)
    q = log.q[idx]
    t = log.t[idx]

    # Rectangular prism scaled like a smallsat bus.
    hx, hy, hz = 0.6, 0.4, 0.25
    corners = np.array(
        [
            [hx, hy, hz],
            [hx, hy, -hz],
            [hx, -hy, hz],
            [hx, -hy, -hz],
            [-hx, hy, hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [-hx, -hy, -hz],
        ],
        dtype=float,
    )
    faces = [
        [0, 1, 3, 2],
        [4, 5, 7, 6],
        [0, 1, 5, 4],
        [2, 3, 7, 6],
        [0, 2, 6, 4],
        [1, 3, 7, 5],
    ]
    face_colors = ["#4c78a8", "#f58518", "#54a24b", "#eeca3b", "#b279a2", "#ff9da6"]

    fig = plt.figure(figsize=(5.2, 5.0))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_proj_type("ortho")
    lim = 1.15
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("X_I")
    ax.set_ylabel("Y_I")
    ax.set_zlabel("Z_I")

    poly = Poly3DCollection(
        [corners[f] for f in faces],
        facecolors=face_colors,
        edgecolors="k",
        linewidths=0.6,
        alpha=0.85,
    )
    ax.add_collection3d(poly)
    quiv = [
        ax.quiver(0, 0, 0, 1, 0, 0, color="r", lw=2, arrow_length_ratio=0.15),
        ax.quiver(0, 0, 0, 0, 1, 0, color="g", lw=2, arrow_length_ratio=0.15),
        ax.quiver(0, 0, 0, 0, 0, 1, color="b", lw=2, arrow_length_ratio=0.15),
    ]
    title = ax.set_title("")

    def _draw(k: int) -> None:
        R = quat_to_rotation(q[k])
        verts = (R @ corners.T).T
        poly.set_verts([verts[f] for f in faces])
        for qv in quiv:
            qv.remove()
        axes_I = R @ np.eye(3)
        quiv[0] = ax.quiver(0, 0, 0, *axes_I[:, 0], color="r", lw=2, arrow_length_ratio=0.15)
        quiv[1] = ax.quiver(0, 0, 0, *axes_I[:, 1], color="g", lw=2, arrow_length_ratio=0.15)
        quiv[2] = ax.quiver(0, 0, 0, *axes_I[:, 2], color="b", lw=2, arrow_length_ratio=0.15)
        title.set_text(f"t = {t[k]:.2f} s")

    def update(k: int):
        _draw(k)
        return (poly,)

    _draw(0)
    anim = animation.FuncAnimation(fig, update, frames=len(idx), interval=1000 / fps, blit=False)
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return path

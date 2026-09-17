"""Matplotlib figures and an optional 3-D attitude GIF for a slew run."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from attitude_sim.quaternions import quat_to_rotation


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
        f"Rest-to-rest slew  |  controller={log.controller}  estimator={log.estimator}",
        fontsize=13,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def write_attitude_gif(log, path: Path, fps: int = 20, max_frames: int = 80) -> Path:
    """Animate a body-fixed box + body axes through the slew."""
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

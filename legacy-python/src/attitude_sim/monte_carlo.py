"""Monte Carlo / noise-sweep harness for closed-loop slews.

Each trial is a stock SimLab rest-to-rest slew (``run_slew``) with bounded
randomization of initial attitude/rate, sensor noise seed and scale, and
optional mild principal-inertia perturbation.  Plant, controllers, and
estimators are not rewritten — this module only samples ``SimConfig`` and
aggregates metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from attitude_sim.actuators import parse_tau_max
from attitude_sim.controls import cubesat_gain_report
from attitude_sim.plant import inertia_from_principal, principal_moments_and_axes
from attitude_sim.quaternions import axis_angle_to_quat, geodesic_angle
from attitude_sim.sim import (
    SimConfig,
    SimLog,
    default_inertia,
    make_scenario_config,
    run_slew,
)

FAIL_NAN = "nan"
FAIL_NON_UNIT = "non_unit_quat"
FAIL_DIVERGED = "diverged"

_QUAT_UNIT_TOL = 1e-6


@dataclass
class MonteCarloConfig:
    """Harness knobs.  Defaults are a local robustness sweep, not a CI load."""

    n: int = 50
    seed: int = 0
    controller: str = "pid"
    estimator: str = "mekf"
    dt: float = 0.01
    t_final: float = 40.0
    angle_deg: float = 75.0
    q0_max_deg: float = 30.0
    omega0_max: float = 0.05
    noise_scale_min: float = 0.5
    noise_scale_max: float = 2.0
    inertia_frac: float = 0.05
    settle_deg: float = 2.0
    diverge_deg: float = 25.0
    diverge_omega: float = 5.0
    torque_limit: float = 0.02
    tau_dist_max: float = 0.0
    gain_scale_min: float = 1.0
    gain_scale_max: float = 1.0
    use_mag: bool = True
    use_sun: bool = True
    # Base sensor densities *before* the log-uniform noise_scale.  None keeps
    # SimConfig defaults (5e-4 / 1e-6 / 3e-3 / 2e-3); make_sim_estimator
    # still copies gyro σ_v / σ_u into the MEKF Qd.
    gyro_sigma_v: float | None = None
    gyro_sigma_u: float | None = None
    mag_sigma: float | None = None
    sun_sigma: float | None = None
    actuator_tau_max: float | np.ndarray | None = None
    rw_inertia: float | np.ndarray | None = None
    rw_h_max: float | np.ndarray | None = None
    rw_visc: float = 0.0
    rw_coulomb: float = 0.0
    rw_gyroscopic: bool = True


@dataclass
class TrialResult:
    trial: int
    seed: int
    final_att_error_deg: float
    settle_time_s: float
    peak_torque: float
    q_norm_err: float
    omega0_norm: float
    q0_from_identity_deg: float
    noise_scale: float
    failed: bool
    fail_reason: str = ""
    gain_scale: float = 1.0
    tau_dist_norm: float = 0.0
    peak_rate: float = float("nan")
    sat_fraction: float = 0.0

    def as_row(self) -> dict[str, object]:
        row = asdict(self)
        for key, value in list(row.items()):
            if isinstance(value, float) and not math.isfinite(value):
                row[key] = ""
        return row


@dataclass
class MonteCarloSummary:
    n: int
    n_fail: int
    n_nan: int
    n_non_unit: int
    n_diverged: int
    n_settled: int
    final_err_mean_deg: float
    final_err_median_deg: float
    final_err_p95_deg: float
    settle_mean_s: float
    peak_torque_max: float
    peak_torque_mean: float
    controller: str
    estimator: str
    t_final: float
    settle_deg: float
    peak_rate_max: float = float("nan")
    peak_rate_mean: float = float("nan")
    sat_fraction_mean: float = 0.0
    trials: list[TrialResult] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        for key, value in list(payload.items()):
            if key == "trials":
                continue
            if isinstance(value, float) and not math.isfinite(value):
                payload[key] = None
        payload["trials"] = [t.as_row() for t in self.trials]
        return payload


def rw_saturation_config(**overrides: object) -> MonteCarloConfig:
    """Closed-loop MC preset: tight wheels + noisy sensors.

    Stresses ``|τ|`` / ``|h|`` saturation on the RW assembly together with
    a log-uniform sensor-noise scale.  Same ``run_slew`` harness as the
    stock slew sweep — plant / controllers / estimators are not rewritten.
    """
    cfg = MonteCarloConfig(
        n=8,
        seed=0,
        controller="pid",
        estimator="mekf",
        dt=0.02,
        t_final=8.0,
        angle_deg=75.0,
        q0_max_deg=20.0,
        omega0_max=0.08,
        noise_scale_min=1.5,
        noise_scale_max=5.0,
        inertia_frac=0.05,
        settle_deg=5.0,
        diverge_deg=90.0,
        torque_limit=0.02,
        tau_dist_max=0.002,
        actuator_tau_max=0.008,
        rw_h_max=0.0015,
        rw_inertia=2.0e-4,
        rw_visc=2.0e-6,
        rw_coulomb=0.0,
        rw_gyroscopic=True,
    )
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise TypeError(f"unknown MonteCarloConfig field {key!r}")
        setattr(cfg, key, value)
    return cfg


def random_unit(rng: np.random.Generator) -> np.ndarray:
    """Isotropic unit vector (3)."""
    v = rng.normal(size=3)
    n = float(np.linalg.norm(v))
    if n < 1e-15:
        return np.array([0.0, 0.0, 1.0])
    return v / n


def sample_bounded_attitude(rng: np.random.Generator, max_deg: float) -> np.ndarray:
    """Random attitude whose geodesic angle from identity is in ``[0, max_deg]``."""
    max_deg = float(max_deg)
    if max_deg <= 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0])
    angle = rng.uniform(0.0, np.deg2rad(max_deg))
    return axis_angle_to_quat(random_unit(rng), angle)


def log_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    """Draw a positive scale uniformly in log-space on ``[lo, hi]``."""
    lo = float(lo)
    hi = float(hi)
    if lo <= 0.0 or hi <= 0.0:
        raise ValueError("noise scale bounds must be positive")
    if hi < lo:
        lo, hi = hi, lo
    if math.isclose(lo, hi):
        return lo
    return float(np.exp(rng.uniform(math.log(lo), math.log(hi))))


def perturb_inertia(
    inertia: np.ndarray,
    frac: float,
    rng: np.random.Generator,
    *,
    max_tries: int = 16,
) -> np.ndarray:
    """Scale principal moments by ``1+U(-frac, frac)``, keeping a physical ``J``.

    Independent per-axis scales are retried if triangle inequalities fail;
    the last fallback is a uniform scale of all three moments.
    """
    J = np.asarray(inertia, dtype=float).reshape(3, 3)
    frac = float(frac)
    if frac <= 0.0:
        return J.copy()
    moments, axes = principal_moments_and_axes(J)
    for _ in range(max_tries):
        scale = 1.0 + rng.uniform(-frac, frac, size=3)
        try:
            return inertia_from_principal(moments * scale, axes)
        except ValueError:
            continue
    s = 1.0 + float(rng.uniform(-frac, frac))
    return inertia_from_principal(moments * s, axes)


def settle_time_s(
    t: np.ndarray,
    att_error_rad: np.ndarray,
    threshold_rad: float,
) -> float:
    """First time after which geodesic error stays below ``threshold_rad``.

    Returns NaN if the trajectory never remains inside the band.
    """
    t = np.asarray(t, dtype=float)
    err = np.asarray(att_error_rad, dtype=float)
    if t.size == 0 or err.size != t.size or not np.all(np.isfinite(err)):
        return float("nan")
    inside = err <= float(threshold_rad)
    if not bool(inside[-1]):
        return float("nan")
    # Last index that is *outside* the band; settle is the next sample.
    outside = np.where(~inside)[0]
    k = 0 if outside.size == 0 else int(outside[-1]) + 1
    if k >= t.size:
        return float("nan")
    return float(t[k])


def classify_failure(
    log: SimLog,
    *,
    diverge_deg: float,
    diverge_omega: float,
    quat_unit_tol: float = _QUAT_UNIT_TOL,
) -> tuple[bool, str, float]:
    """Return ``(failed, reason, max_|‖q‖-1|)``.

    Reasons: ``nan``, ``non_unit_quat``, ``diverged``, or empty if healthy.
    """
    q_norm = np.linalg.norm(np.asarray(log.q, dtype=float), axis=1)
    q_norm_err = float(np.nanmax(np.abs(q_norm - 1.0))) if q_norm.size else float("nan")
    arrays = (log.q, log.omega, log.tau, log.att_error, log.t)
    if any(a is None or not np.all(np.isfinite(a)) for a in arrays):
        return True, FAIL_NAN, q_norm_err
    if not math.isfinite(q_norm_err) or q_norm_err > quat_unit_tol:
        return True, FAIL_NON_UNIT, q_norm_err
    final_deg = float(np.rad2deg(log.att_error[-1]))
    w_max = float(np.max(np.linalg.norm(log.omega, axis=1)))
    if (
        (not math.isfinite(final_deg))
        or final_deg > float(diverge_deg)
        or w_max > float(diverge_omega)
    ):
        return True, FAIL_DIVERGED, q_norm_err
    return False, "", q_norm_err


def _trial_seed(master: int, trial: int) -> int:
    """Stable per-trial seed (independent of hash randomization)."""
    return int((int(master) + 10007 * (int(trial) + 1)) % (2**31 - 1))


def sample_trial_config(
    mc: MonteCarloConfig,
    trial: int,
    rng: np.random.Generator,
) -> tuple[SimConfig, dict[str, float]]:
    """Draw one randomized slew ``SimConfig`` plus the sampled extras."""
    q0 = sample_bounded_attitude(rng, mc.q0_max_deg)
    omega0 = rng.uniform(-mc.omega0_max, mc.omega0_max, size=3)
    noise_scale = log_uniform(rng, mc.noise_scale_min, mc.noise_scale_max)
    inertia = perturb_inertia(default_inertia(), mc.inertia_frac, rng)
    trial_seed = _trial_seed(mc.seed, trial)

    cfg = make_scenario_config(
        "slew",
        dt=mc.dt,
        t_final=mc.t_final,
        controller=mc.controller,
        estimator=mc.estimator,
        angle_deg=mc.angle_deg,
        use_mag=mc.use_mag,
        use_sun=mc.use_sun,
        gyro_sigma_v=mc.gyro_sigma_v,
        gyro_sigma_u=mc.gyro_sigma_u,
        mag_sigma=mc.mag_sigma,
        sun_sigma=mc.sun_sigma,
        seed=trial_seed,
        plot=False,
        gif=False,
    )
    cfg.q0 = q0
    cfg.omega0 = np.asarray(omega0, dtype=float)
    cfg.inertia = inertia
    cfg.torque_limit = mc.torque_limit
    cfg.gyro_sigma_v = cfg.gyro_sigma_v * noise_scale
    cfg.gyro_sigma_u = cfg.gyro_sigma_u * noise_scale
    cfg.mag_sigma = cfg.mag_sigma * noise_scale
    cfg.sun_sigma = cfg.sun_sigma * noise_scale

    tau_dist_norm = 0.0
    if mc.tau_dist_max > 0.0:
        mag = float(rng.uniform(0.0, mc.tau_dist_max))
        tau_dist = mag * random_unit(rng)
        cfg.tau_dist = tau_dist
        tau_dist_norm = mag

    lo = float(mc.gain_scale_min)
    hi = float(mc.gain_scale_max)
    if lo <= 0.0 or hi <= 0.0:
        raise ValueError("gain_scale bounds must be positive")
    if math.isclose(lo, hi):
        gain_scale = lo
    else:
        gain_scale = log_uniform(rng, lo, hi)
    cfg.gain_scale = gain_scale
    cfg.actuator_tau_max = mc.actuator_tau_max
    cfg.rw_inertia = mc.rw_inertia
    cfg.rw_h_max = mc.rw_h_max
    cfg.rw_visc = mc.rw_visc
    cfg.rw_coulomb = mc.rw_coulomb
    cfg.rw_gyroscopic = mc.rw_gyroscopic

    # #38 cubesat helpers: SimLab run_slew applies tune_pid_second_order /
    # bryson_lqr_costs via cubesat_controller_kwargs on this trial inertia.
    report = cubesat_gain_report(inertia, tau_max=float(mc.torque_limit))
    extras = {
        "noise_scale": float(noise_scale),
        "omega0_norm": float(np.linalg.norm(omega0)),
        "q0_from_identity_deg": float(
            np.rad2deg(geodesic_angle(q0, np.array([1.0, 0.0, 0.0, 0.0])))
        ),
        "gain_scale": float(gain_scale),
        "tau_dist_norm": float(tau_dist_norm),
        "cubesat_wn": float(report.wn),
        "cubesat_recommended_wn": float(report.recommended_wn),
        "cubesat_kp_00": float(report.kp[0, 0]),
    }
    return cfg, extras


def score_trial(
    log: SimLog, mc: MonteCarloConfig, extras: dict[str, float], trial: int, seed: int
) -> TrialResult:
    failed, reason, q_norm_err = classify_failure(
        log, diverge_deg=mc.diverge_deg, diverge_omega=mc.diverge_omega
    )
    peak = (
        float("nan")
        if log.tau.size == 0 or not np.all(np.isfinite(log.tau))
        else float(np.max(np.linalg.norm(log.tau, axis=1)))
    )
    final_deg = (
        float("nan")
        if not np.all(np.isfinite(log.att_error))
        else float(np.rad2deg(log.att_error[-1]))
    )
    settle = settle_time_s(log.t, log.att_error, np.deg2rad(mc.settle_deg))
    return TrialResult(
        trial=trial,
        seed=seed,
        final_att_error_deg=final_deg,
        settle_time_s=settle,
        peak_torque=peak,
        q_norm_err=q_norm_err,
        omega0_norm=extras["omega0_norm"],
        q0_from_identity_deg=extras["q0_from_identity_deg"],
        noise_scale=extras["noise_scale"],
        gain_scale=float(extras.get("gain_scale", 1.0)),
        tau_dist_norm=float(extras.get("tau_dist_norm", 0.0)),
        peak_rate=float(log.peak_rate),
        sat_fraction=float(log.sat_fraction),
        failed=failed,
        fail_reason=reason,
    )


def _finite(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def summarize(mc: MonteCarloConfig, trials: list[TrialResult]) -> MonteCarloSummary:
    ok = [t for t in trials if not t.failed]
    errs = _finite([t.final_att_error_deg for t in ok])
    settles = _finite([t.settle_time_s for t in ok])
    peaks = _finite([t.peak_torque for t in trials])
    rates = _finite([t.peak_rate for t in trials])
    sats = _finite([t.sat_fraction for t in trials])

    def _stat(arr: np.ndarray, fn) -> float:
        return float(fn(arr)) if arr.size else float("nan")

    return MonteCarloSummary(
        n=len(trials),
        n_fail=sum(1 for t in trials if t.failed),
        n_nan=sum(1 for t in trials if t.fail_reason == FAIL_NAN),
        n_non_unit=sum(1 for t in trials if t.fail_reason == FAIL_NON_UNIT),
        n_diverged=sum(1 for t in trials if t.fail_reason == FAIL_DIVERGED),
        n_settled=int(settles.size),
        final_err_mean_deg=_stat(errs, np.mean),
        final_err_median_deg=_stat(errs, np.median),
        final_err_p95_deg=_stat(errs, lambda a: np.percentile(a, 95)),
        settle_mean_s=_stat(settles, np.mean),
        peak_torque_max=_stat(peaks, np.max),
        peak_torque_mean=_stat(peaks, np.mean),
        peak_rate_max=_stat(rates, np.max),
        peak_rate_mean=_stat(rates, np.mean),
        sat_fraction_mean=_stat(sats, np.mean) if sats.size else 0.0,
        controller=mc.controller,
        estimator=mc.estimator,
        t_final=mc.t_final,
        settle_deg=mc.settle_deg,
        trials=trials,
    )


def run_monte_carlo(mc: MonteCarloConfig | None = None) -> MonteCarloSummary:
    """Run ``mc.n`` closed-loop slews and return aggregated metrics."""
    mc = mc if mc is not None else MonteCarloConfig()
    if mc.n < 1:
        raise ValueError("n must be >= 1")
    rng = np.random.default_rng(mc.seed)
    trials: list[TrialResult] = []
    for i in range(mc.n):
        cfg, extras = sample_trial_config(mc, i, rng)
        try:
            log = run_slew(cfg)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            log = _failed_placeholder(cfg)
        trials.append(score_trial(log, mc, extras, trial=i, seed=cfg.seed))
    return summarize(mc, trials)


def _failed_placeholder(cfg: SimConfig) -> SimLog:
    """Minimal log so a raised plant/filter error still counts as a NaN fail."""
    q = np.full((1, 4), np.nan)
    w = np.full((1, 3), np.nan)
    return SimLog(
        t=np.array([0.0]),
        q=q,
        omega=w,
        tau=w.copy(),
        q_hat=q.copy(),
        omega_hat=w.copy(),
        q_des=np.asarray(cfg.q_des, dtype=float),
        euler=w.copy(),
        euler_des=np.zeros(3),
        att_error=np.array([np.nan]),
        att_error_hat=None,
        est_att_error=None,
        controller=cfg.controller,
        estimator=cfg.estimator,
        scenario=cfg.scenario,
    )


def format_summary(summary: MonteCarloSummary) -> str:
    def _fmt(x: float, spec: str = ".3f") -> str:
        return "nan" if not math.isfinite(x) else format(x, spec)

    return "\n".join(
        [
            (
                f"Monte Carlo slew  N={summary.n}  controller={summary.controller}  "
                f"estimator={summary.estimator}  t_final={summary.t_final:.3g} s"
            ),
            (
                f"failures: {summary.n_fail}/{summary.n}  "
                f"(nan={summary.n_nan}  non_unit={summary.n_non_unit}  "
                f"diverged={summary.n_diverged})"
            ),
            (
                f"final att error [deg]: mean={_fmt(summary.final_err_mean_deg)}  "
                f"median={_fmt(summary.final_err_median_deg)}  "
                f"p95={_fmt(summary.final_err_p95_deg)}"
            ),
            (
                f"settle time [s] (< {summary.settle_deg:g} deg): "
                f"mean={_fmt(summary.settle_mean_s, '.2f')}  "
                f"(n_settled={summary.n_settled})"
            ),
            (
                f"peak |tau| [N·m]: max={_fmt(summary.peak_torque_max, '.4f')}  "
                f"mean={_fmt(summary.peak_torque_mean, '.4f')}"
            ),
            (
                f"peak |omega| [rad/s]: max={_fmt(summary.peak_rate_max, '.4f')}  "
                f"mean={_fmt(summary.peak_rate_mean, '.4f')}"
            ),
            (
                f"RW sat fraction: mean={_fmt(summary.sat_fraction_mean, '.3f')}  "
                f"(any-axis |τ| or |h| at limit)"
            ),
        ]
    )


def write_csv(path: Path, summary: MonteCarloSummary) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [t.as_row() for t in summary.trials]
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, summary: MonteCarloSummary) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary.as_dict(), indent=2) + "\n")


def _as_float(value: object) -> float:
    if value is None or value == "":
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"cannot convert {type(value).__name__} to float")


def _as_int(value: object, default: int = 0) -> int:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        return int(value)
    raise TypeError(f"cannot convert {type(value).__name__} to int")


def trial_from_row(row: dict) -> TrialResult:
    """Rebuild a ``TrialResult`` from CSV/JSON (empty strings are NaN)."""
    return TrialResult(
        trial=int(row["trial"]),
        seed=int(row["seed"]),
        final_att_error_deg=_as_float(row.get("final_att_error_deg")),
        settle_time_s=_as_float(row.get("settle_time_s")),
        peak_torque=_as_float(row.get("peak_torque")),
        q_norm_err=_as_float(row.get("q_norm_err")),
        omega0_norm=_as_float(row.get("omega0_norm")),
        q0_from_identity_deg=_as_float(row.get("q0_from_identity_deg")),
        noise_scale=_as_float(row.get("noise_scale")),
        failed=bool(row.get("failed", False)),
        fail_reason=str(row.get("fail_reason") or ""),
        gain_scale=_as_float(row.get("gain_scale"))
        if row.get("gain_scale") not in (None, "")
        else 1.0,
        tau_dist_norm=_as_float(row.get("tau_dist_norm"))
        if row.get("tau_dist_norm") not in (None, "")
        else 0.0,
        peak_rate=_as_float(row.get("peak_rate"))
        if row.get("peak_rate") not in (None, "")
        else float("nan"),
        sat_fraction=_as_float(row.get("sat_fraction"))
        if row.get("sat_fraction") not in (None, "")
        else 0.0,
    )


def summary_from_dict(payload: dict) -> MonteCarloSummary:
    """Rebuild a summary from ``MonteCarloSummary.as_dict`` / JSON."""
    trials = [trial_from_row(row) for row in payload.get("trials") or []]
    return MonteCarloSummary(
        n=_as_int(payload.get("n"), len(trials)),
        n_fail=_as_int(payload.get("n_fail")),
        n_nan=_as_int(payload.get("n_nan")),
        n_non_unit=_as_int(payload.get("n_non_unit")),
        n_diverged=_as_int(payload.get("n_diverged")),
        n_settled=_as_int(payload.get("n_settled")),
        final_err_mean_deg=_as_float(payload.get("final_err_mean_deg")),
        final_err_median_deg=_as_float(payload.get("final_err_median_deg")),
        final_err_p95_deg=_as_float(payload.get("final_err_p95_deg")),
        settle_mean_s=_as_float(payload.get("settle_mean_s")),
        peak_torque_max=_as_float(payload.get("peak_torque_max")),
        peak_torque_mean=_as_float(payload.get("peak_torque_mean")),
        peak_rate_max=_as_float(payload.get("peak_rate_max"))
        if payload.get("peak_rate_max") not in (None, "")
        else float("nan"),
        peak_rate_mean=_as_float(payload.get("peak_rate_mean"))
        if payload.get("peak_rate_mean") not in (None, "")
        else float("nan"),
        sat_fraction_mean=_as_float(payload.get("sat_fraction_mean"))
        if payload.get("sat_fraction_mean") not in (None, "")
        else 0.0,
        controller=str(payload.get("controller") or ""),
        estimator=str(payload.get("estimator") or ""),
        t_final=_as_float(payload.get("t_final")),
        settle_deg=_as_float(payload.get("settle_deg")),
        trials=trials,
    )


def summary_from_json(path: Path) -> MonteCarloSummary:
    return summary_from_dict(json.loads(Path(path).read_text()))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m attitude_sim.monte_carlo",
        description=(
            "Monte Carlo robustness sweep of closed-loop slews "
            "(randomized IC, sensor noise, optional inertia / gain / disturbance). "
            "Slew-only: detumble is a SimLab scenario, not part of this harness."
        ),
    )
    p.add_argument("--n", type=int, default=50, help="number of trials")
    p.add_argument("--seed", type=int, default=0, help="master RNG seed")
    p.add_argument("--controller", choices=("pid", "lqr"), default="pid")
    p.add_argument("--estimator", choices=("truth", "mekf", "mahony"), default="mekf")
    p.add_argument("--t-final", type=float, default=40.0, help="per-trial duration (s)")
    p.add_argument("--dt", type=float, default=0.01, help="sample / RK4 step (s)")
    p.add_argument("--angle-deg", type=float, default=75.0, help="commanded slew (deg)")
    p.add_argument(
        "--q0-max-deg",
        type=float,
        default=30.0,
        help="max initial geodesic angle from identity (deg)",
    )
    p.add_argument(
        "--omega0-max",
        type=float,
        default=0.05,
        help="half-width of uniform initial body-rate cube (rad/s)",
    )
    p.add_argument("--noise-scale-min", type=float, default=0.5, help="min sensor-noise scale")
    p.add_argument("--noise-scale-max", type=float, default=2.0, help="max sensor-noise scale")
    p.add_argument(
        "--gyro-sigma-v",
        type=float,
        default=None,
        metavar="SIGMA",
        help="base gyro ARW σ_v [rad/s/√Hz] before --noise-scale (default: SimConfig 5e-4)",
    )
    p.add_argument(
        "--gyro-sigma-u",
        type=float,
        default=None,
        metavar="SIGMA",
        help="base gyro RRW σ_u [rad/s²/√Hz] before --noise-scale (default: SimConfig 1e-6)",
    )
    p.add_argument(
        "--mag-sigma",
        type=float,
        default=None,
        metavar="SIGMA",
        help="base magnetometer Cartesian σ before --noise-scale (default: SimConfig 3e-3)",
    )
    p.add_argument(
        "--sun-sigma",
        type=float,
        default=None,
        metavar="SIGMA",
        help="base sun-sensor Cartesian σ before --noise-scale (default: SimConfig 2e-3)",
    )
    p.add_argument(
        "--inertia-frac",
        type=float,
        default=0.05,
        help="principal-moment perturbation |ε| (0 disables)",
    )
    p.add_argument(
        "--tau-dist-max",
        type=float,
        default=0.0,
        help="if >0, sample a constant body disturbance with ||τ_d|| ≤ this (N·m)",
    )
    p.add_argument(
        "--gain-scale-min",
        type=float,
        default=1.0,
        help="min implemented-gain scale (PID Kp/Kd/Ki or LQR K)",
    )
    p.add_argument(
        "--gain-scale-max",
        type=float,
        default=1.0,
        help="max implemented-gain scale (log-uniform with --gain-scale-min)",
    )
    p.add_argument("--settle-deg", type=float, default=2.0, help="settle-time error band (deg)")
    p.add_argument(
        "--diverge-deg",
        type=float,
        default=25.0,
        help="final geodesic error above this counts as diverged (deg)",
    )
    p.add_argument(
        "--actuator-tau-max",
        default=None,
        help="per-axis wheel torque limit [N·m]: scalar or x,y,z (enables RW with --rw-h-max)",
    )
    p.add_argument(
        "--rw-h-max",
        default=None,
        help="per-axis wheel momentum limit [N·m·s]: scalar or x,y,z (enables RW assembly)",
    )
    p.add_argument(
        "--rw-inertia",
        default=None,
        help="per-axis wheel inertia I_w [kg·m²] (default 2e-4 when --rw-h-max is set)",
    )
    p.add_argument(
        "--rw-visc",
        type=float,
        default=None,
        help="viscous wheel friction b [N·m·s]",
    )
    p.add_argument(
        "--rw-coulomb",
        type=float,
        default=None,
        help="smoothed Coulomb wheel friction c [N·m]",
    )
    p.add_argument(
        "--rw-sat-stress",
        action="store_true",
        help=(
            "preset: tight RW |τ|/|h| limits, body-disturbance fill, and noisy sensors "
            "(same harness; overlays --n/--seed/--estimator/...)"
        ),
    )
    p.add_argument("--csv", type=Path, default=None, help="optional per-trial CSV path")
    p.add_argument("--json", type=Path, default=None, help="optional summary JSON path")
    p.add_argument(
        "--from-json",
        type=Path,
        default=None,
        help="load a previous harness JSON (skip running trials; for plots/re-export)",
    )
    p.add_argument(
        "--plot",
        action="store_true",
        help="write summary PNGs (histogram + settle-vs-noise scatter) under --out-dir",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs"),
        help="directory for --plot figures (default: outputs/)",
    )
    return p


def config_from_args(args: argparse.Namespace) -> MonteCarloConfig:
    if args.rw_sat_stress:
        mc = rw_saturation_config(
            n=args.n,
            seed=args.seed,
            controller=args.controller,
            estimator=args.estimator,
            dt=args.dt,
            t_final=args.t_final,
            angle_deg=args.angle_deg,
            q0_max_deg=args.q0_max_deg,
            omega0_max=args.omega0_max,
            inertia_frac=args.inertia_frac,
            settle_deg=args.settle_deg,
            diverge_deg=args.diverge_deg,
            gain_scale_min=args.gain_scale_min,
            gain_scale_max=args.gain_scale_max,
            gyro_sigma_v=args.gyro_sigma_v,
            gyro_sigma_u=args.gyro_sigma_u,
            mag_sigma=args.mag_sigma,
            sun_sigma=args.sun_sigma,
        )
        # Keep the noisy-sensor / disturbance defaults unless the caller
        # moved the noise-scale bounds away from the parser defaults.
        if args.noise_scale_min != 0.5 or args.noise_scale_max != 2.0:
            mc.noise_scale_min = args.noise_scale_min
            mc.noise_scale_max = args.noise_scale_max
        if args.tau_dist_max != 0.0:
            mc.tau_dist_max = args.tau_dist_max
    else:
        mc = MonteCarloConfig(
            n=args.n,
            seed=args.seed,
            controller=args.controller,
            estimator=args.estimator,
            dt=args.dt,
            t_final=args.t_final,
            angle_deg=args.angle_deg,
            q0_max_deg=args.q0_max_deg,
            omega0_max=args.omega0_max,
            noise_scale_min=args.noise_scale_min,
            noise_scale_max=args.noise_scale_max,
            inertia_frac=args.inertia_frac,
            settle_deg=args.settle_deg,
            diverge_deg=args.diverge_deg,
            tau_dist_max=args.tau_dist_max,
            gain_scale_min=args.gain_scale_min,
            gain_scale_max=args.gain_scale_max,
            gyro_sigma_v=args.gyro_sigma_v,
            gyro_sigma_u=args.gyro_sigma_u,
            mag_sigma=args.mag_sigma,
            sun_sigma=args.sun_sigma,
        )
    if args.actuator_tau_max is not None:
        mc.actuator_tau_max = parse_tau_max(args.actuator_tau_max, "--actuator-tau-max")
    if args.rw_h_max is not None:
        mc.rw_h_max = parse_tau_max(args.rw_h_max, "--rw-h-max")
    if args.rw_inertia is not None:
        mc.rw_inertia = parse_tau_max(args.rw_inertia, "--rw-inertia")
    if args.rw_visc is not None:
        mc.rw_visc = float(args.rw_visc)
    if args.rw_coulomb is not None:
        mc.rw_coulomb = float(args.rw_coulomb)
    return mc


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.from_json is None:
        for flag, value in (
            ("--gyro-sigma-v", args.gyro_sigma_v),
            ("--gyro-sigma-u", args.gyro_sigma_u),
            ("--mag-sigma", args.mag_sigma),
            ("--sun-sigma", args.sun_sigma),
        ):
            if value is not None and value < 0.0:
                parser.error(f"{flag} must be >= 0")
        if args.rw_visc is not None and args.rw_visc < 0.0:
            parser.error("--rw-visc must be >= 0")
        if args.rw_coulomb is not None and args.rw_coulomb < 0.0:
            parser.error("--rw-coulomb must be >= 0")
    if args.from_json is not None:
        summary = summary_from_json(args.from_json)
    else:
        try:
            mc = config_from_args(args)
        except ValueError as exc:
            parser.error(str(exc))
        summary = run_monte_carlo(mc)
    print(format_summary(summary))
    if args.csv is not None:
        write_csv(args.csv, summary)
        print(f"csv:  {args.csv}")
    if args.json is not None:
        write_json(args.json, summary)
        print(f"json: {args.json}")
    if args.plot:
        from attitude_sim.plots import plot_monte_carlo

        for path in plot_monte_carlo(summary, args.out_dir):
            print(f"plot: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

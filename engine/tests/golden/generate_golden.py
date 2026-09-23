#!/usr/bin/env python3
"""Generate golden SimLog fixtures from the Python reference engine.

Run from the repo root with the legacy Python package installed:

    pip install -e legacy-python[dev]
    python3 engine/tests/golden/generate_golden.py

Every case here uses `--estimator truth` (no sensor sampling), so the
Python (NumPy PCG64) and Rust (ChaCha8) RNGs never enter the comparison --
these fixtures pin the *deterministic* core (plant, PID/LQR, actuators,
reaction wheels, disturbances) that both engines must agree on bit-for-bit
(to float tolerance). See `engine/tests/python_parity.rs` for the Rust side
and `README.md` for why noisy (MEKF/Mahony) paths are validated statistically
instead of cross-language bit-parity.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "legacy-python" / "src"))

from attitude_sim.sim import make_scenario_config, run_slew  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent


def log_to_dict(log, request: dict) -> dict:
    return {
        "request": request,
        "t": log.t.tolist(),
        "q": log.q.tolist(),
        "omega": log.omega.tolist(),
        "tau": log.tau.tolist(),
        "tau_env": log.tau_env.tolist(),
        "h_wheel": log.h_wheel.tolist(),
        "tau_ext": log.tau_ext.tolist(),
        "omega_wheel": log.omega_wheel.tolist(),
        "euler_deg": np.rad2deg(log.euler).tolist(),
        "att_error_deg": np.rad2deg(log.att_error).tolist(),
        "q_des": log.q_des.tolist(),
        "final_att_error_deg": log.final_att_error_deg,
        "peak_rate": log.peak_rate,
    }


CASES: list[dict] = [
    {"name": "slew_pid_truth", "scenario": "slew", "estimator": "truth", "t_final": 6.0},
    {"name": "slew_lqr_truth", "scenario": "slew", "estimator": "truth", "controller": "lqr", "t_final": 6.0},
    {"name": "detumble_pid_truth", "scenario": "detumble", "estimator": "truth", "t_final": 6.0},
    {"name": "hold_pid_truth", "scenario": "hold", "estimator": "truth", "t_final": 6.0},
    {"name": "eigenaxis_lqr_truth", "scenario": "eigenaxis", "estimator": "truth", "t_final": 5.0},
    {"name": "eigenaxis_custom_angle", "scenario": "eigenaxis", "estimator": "truth", "t_final": 4.0, "angle_deg": 55.0},
    {
        "name": "slew_tau_dist",
        "scenario": "slew",
        "estimator": "truth",
        "t_final": 6.0,
        "angle_deg": 0.0,
        "tau_dist": np.array([0.002, -0.001, 0.0008]),
    },
    {
        "name": "slew_actuator_lag",
        "scenario": "slew",
        "estimator": "truth",
        "t_final": 6.0,
        "actuator_tau_max": 0.02,
        "actuator_tau": 0.05,
    },
    {
        "name": "slew_reaction_wheels",
        "scenario": "slew",
        "estimator": "truth",
        "t_final": 6.0,
        "actuator_tau_max": 0.02,
        "rw_h_max": 0.004,
        "rw_visc": 1e-6,
    },
    {
        "name": "slew_actuator_dump",
        "scenario": "slew",
        "estimator": "truth",
        "t_final": 6.0,
        "angle_deg": 0.0,
        "tau_dist": np.array([0.002, -0.001, 0.0008]),
        "actuator_h_dump": 0.01,
    },
    {
        "name": "hold_aero_srp",
        "scenario": "hold",
        "estimator": "truth",
        "t_final": 6.0,
        "aerodynamic": True,
        "srp": True,
    },
    {
        "name": "slew_env_gg_dipole",
        "scenario": "slew",
        "estimator": "truth",
        "t_final": 6.0,
        "gravity_gradient": True,
        "residual_dipole": True,
        "actuator_tau_max": 0.008,
    },
    {
        "name": "slew_no_env_override",
        "scenario": "hold",
        "estimator": "truth",
        "t_final": 5.0,
        "no_env": True,
    },
    {
        "name": "eigenaxis_srp_cylindrical",
        "scenario": "eigenaxis",
        "estimator": "truth",
        "t_final": 5.0,
        "srp": True,
        "srp_eclipse": "cylindrical",
        "dipole_model": "orbit_normal",
        "residual_dipole": True,
    },
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for case in CASES:
        name = case["name"]
        kwargs = {k: v for k, v in case.items() if k != "name"}
        cfg = make_scenario_config(plot=False, gif=False, **kwargs)
        log = run_slew(cfg)
        request = dict(kwargs)
        for k, v in list(request.items()):
            if isinstance(v, np.ndarray):
                request[k] = v.tolist()
        payload = log_to_dict(log, request)
        dest = OUT_DIR / f"{name}.json"
        dest.write_text(json.dumps(payload))
        print(f"wrote {dest} ({len(log.t)} samples)")


if __name__ == "__main__":
    main()

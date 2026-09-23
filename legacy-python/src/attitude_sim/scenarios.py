"""Named SimLab closed-loop scenario pack.

Plant / controller / estimator cores are unchanged.  This module only
names initial conditions, attitude commands, default durations, default
controllers, and optional :class:`~attitude_sim.disturbances.EnvironmentalTorques`
bindings.  ``attitude_sim.sim`` consumes the presets and still integrates
on ``x = [q, omega]`` with ``step_rigid_body``.

Scenarios
---------
``slew``
    75° rest-to-rest about a skewed body axis (default PID).
``detumble``
    Tumbling rate, dump omega and recover identity (default PID).
``hold``
    Identity hold under demo-scale gravity-gradient + residual-dipole
    torques from :class:`EnvironmentalTorques` (default PID, so the
    integrator can reject the slowly varying body bias).
``eigenaxis``
    Rest-to-rest principal-axis (body z) slew, default **LQR**.
    ``--angle-deg`` selects the commanded principal rotation (default 30°).

The residual dipole on ``hold`` is **demo-scale** (tens of A·m²) so
``|tau_env|`` sits in the millinewton-metre band used by the existing
``--tau-dist`` checkout, visible on the 0.02 N·m actuator.  A CubeSat-real
residual would be ~0.01–0.1 A·m² and vanish on that plot;
gravity-gradient on this smallsat J is ~1e-8 N·m and is included for
wiring completeness.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from attitude_sim.controls import DEFAULT_TORQUE_LIMIT, CubesatGainReport, cubesat_gain_report
from attitude_sim.disturbances import (
    CircularOrbit,
    EnvironmentalTorques,
    GravityGradientTorque,
    ResidualDipoleTorque,
)
from attitude_sim.quaternions import axis_angle_to_quat

SCENARIOS = ("slew", "detumble", "hold", "eigenaxis")

SLEW_AXIS = np.array([0.2, 0.5, 0.84])
SLEW_ANGLE_DEG = 75.0
DETUMBLE_Q0_AXIS = np.array([0.4, 0.2, 0.9])
DETUMBLE_OMEGA0 = np.array([0.55, -0.40, 0.30])
EIGENAXIS_AXIS = np.array([0.0, 0.0, 1.0])
EIGENAXIS_ANGLE_DEG = 30.0
IDENTITY_Q = np.array([1.0, 0.0, 0.0, 0.0])

# LEO-class circular orbit (same radius family as tests/test_disturbances.py).
HOLD_ORBIT_RADIUS_M = 7.0e6
HOLD_ORBIT_INCLINATION_RAD = np.deg2rad(51.6)
HOLD_ARG_LATITUDE0_RAD = np.deg2rad(40.0)
# Demo-scale residual dipole so |m × B_LEO| ~ 1 mN·m.
HOLD_RESIDUAL_DIPOLE_A_M2 = np.array([40.0, -15.0, 8.0])

SCENARIO_T_FINAL: Mapping[str, float] = {
    "slew": 40.0,
    "detumble": 30.0,
    "hold": 30.0,
    "eigenaxis": 20.0,
}

SCENARIO_CONTROLLER: Mapping[str, str] = {
    "slew": "pid",
    "detumble": "pid",
    "hold": "pid",
    "eigenaxis": "lqr",
}

SCENARIO_TITLES: Mapping[str, str] = {
    "slew": "Rest-to-rest slew",
    "detumble": "Detumble to rest",
    "hold": "Hold under environmental torques",
    "eigenaxis": "Eigenaxis slew (LQR)",
}

SCENARIO_BLURBS: Mapping[str, str] = {
    "slew": "75° rest-to-rest slew about a skewed body axis (default PID)",
    "detumble": "tumbling initial rate, dump ω and recover identity (default PID)",
    "hold": "identity hold under EnvironmentalTorques (GG + demo-scale residual dipole, default PID)",
    "eigenaxis": "principal-axis (body z) rest-to-rest slew (default LQR, --angle-deg default 30)",
}


def scenario_cubesat_gains(
    inertia: np.ndarray | None = None,
    *,
    tau_max: float = DEFAULT_TORQUE_LIMIT,
) -> CubesatGainReport:
    """PID / LQR numbers used by SimLab and Monte Carlo defaults (#38 helpers).

    Does not change scenario ICs; ``run_slew`` / MC apply the same report
    to the (possibly perturbed) plant inertia.
    """
    return cubesat_gain_report(inertia, tau_max=tau_max)


def default_t_final(scenario: str) -> float:
    """Return the pack duration (s) for a named scenario."""
    name = _require_scenario(scenario)
    return float(SCENARIO_T_FINAL[name])


def default_controller(scenario: str) -> str:
    """Return the pack controller name (``pid`` or ``lqr``)."""
    name = _require_scenario(scenario)
    return str(SCENARIO_CONTROLLER[name])


def resolve_controller(scenario: str, controller: str | None) -> str:
    """Honor an explicit controller; otherwise use the scenario default."""
    if controller is not None and str(controller).strip():
        return str(controller).lower()
    return default_controller(scenario)


def default_angle_deg(scenario: str, angle_deg: float | None) -> float:
    """Commanded principal rotation (deg) for slew / eigenaxis.

    ``hold`` and ``detumble`` ignore the angle (identity command).
    """
    name = _require_scenario(scenario)
    if angle_deg is not None:
        return float(angle_deg)
    if name == "eigenaxis":
        return float(EIGENAXIS_ANGLE_DEG)
    return float(SLEW_ANGLE_DEG)


def scenario_uses_env_by_default(scenario: str) -> bool:
    """True when the named preset binds :class:`EnvironmentalTorques`."""
    return _require_scenario(scenario) == "hold"


def resolve_use_env(
    scenario: str,
    *,
    use_env: bool | None = None,
    env_flag: bool = False,
    no_env: bool = False,
) -> bool:
    """Resolve whether demo-scale environmental torques are enabled.

    Precedence: ``no_env`` (off) > explicit ``use_env`` > ``env_flag`` (on)
    > scenario default (on for ``hold`` only).
    """
    if no_env:
        return False
    if use_env is not None:
        return bool(use_env)
    if env_flag:
        return True
    return scenario_uses_env_by_default(scenario)


def make_hold_environmental_torques(
    inertia: np.ndarray,
    *,
    orbit: CircularOrbit | None = None,
    m_body: np.ndarray | None = None,
) -> EnvironmentalTorques:
    """Gravity-gradient + residual-dipole pack used by ``--scenario hold``.

    The same :class:`CircularOrbit` is bound to both models so
    ``τ_body(q, ω, t)`` evaluates from SimLab time without a plant rewrite.
    """
    circ = orbit if orbit is not None else default_hold_orbit()
    dipole = (
        HOLD_RESIDUAL_DIPOLE_A_M2.copy()
        if m_body is None
        else np.asarray(m_body, dtype=float).reshape(3)
    )
    return EnvironmentalTorques(
        gravity_gradient=GravityGradientTorque(inertia, orbit=circ),
        residual_dipole=ResidualDipoleTorque(dipole, orbit=circ, model="tilted"),
    )


def default_hold_orbit() -> CircularOrbit:
    """Inclined LEO circular orbit used by the hold preset."""
    return CircularOrbit(
        radius=HOLD_ORBIT_RADIUS_M,
        inclination=HOLD_ORBIT_INCLINATION_RAD,
        raan=0.0,
        arg_latitude0=HOLD_ARG_LATITUDE0_RAD,
    )


def scenario_state(
    scenario: str,
    *,
    angle_deg: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(q0, omega0, q_des)`` for a named scenario."""
    name = _require_scenario(scenario)
    ang = default_angle_deg(name, angle_deg)
    if name == "detumble":
        return (
            axis_angle_to_quat(DETUMBLE_Q0_AXIS, np.deg2rad(40.0)),
            DETUMBLE_OMEGA0.copy(),
            IDENTITY_Q.copy(),
        )
    if name == "hold":
        return (IDENTITY_Q.copy(), np.zeros(3), IDENTITY_Q.copy())
    if name == "eigenaxis":
        return (
            IDENTITY_Q.copy(),
            np.zeros(3),
            axis_angle_to_quat(EIGENAXIS_AXIS, np.deg2rad(ang)),
        )
    return (
        IDENTITY_Q.copy(),
        np.zeros(3),
        axis_angle_to_quat(SLEW_AXIS, np.deg2rad(ang)),
    )


def scenario_catalog_text() -> str:
    """Human-readable catalog for ``--list-scenarios``."""
    lines = ["Named SimLab closed-loop scenarios:", ""]
    for name in SCENARIOS:
        ctrl = SCENARIO_CONTROLLER[name]
        t_f = SCENARIO_T_FINAL[name]
        env = "env-torques" if scenario_uses_env_by_default(name) else "no-env"
        lines.append(
            f"  {name:10s}  t_final={t_f:g}s  default-controller={ctrl:3s}  {env}"
        )
        lines.append(f"             {SCENARIO_BLURBS[name]}")
    lines.append("")
    lines.append("Monte Carlo (`python -m attitude_sim.monte_carlo`) remains slew-only.")
    return "\n".join(lines)


def _require_scenario(scenario: str) -> str:
    name = str(scenario).lower()
    if name not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; expected one of {SCENARIOS}")
    return name

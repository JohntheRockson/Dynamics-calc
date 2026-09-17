"""Plant-only MC-ish sweeps: principal-inertia mismatch and varied dt.

Runtime closed-loop Monte Carlo lives in SimLab.  These tests exercise the
plant helpers and RK4 invariants only — no controllers or estimators.

Torque-free tolerances are for classical RK4 (not symplectic).  They are
documented bounds for the scenarios in this file, not machine epsilon.
The tight fixed-dt cases stay in ``tests/test_plant.py``.
"""

import numpy as np
import pytest

from attitude_sim.plant import (
    RigidBody,
    perturb_principal_inertia,
    random_principal_scale,
    rk4_step,
    scale_principal_inertia,
    step_rigid_body,
    validate_inertia,
)
from attitude_sim.quaternions import quat_normalize


# Smallsat-class principal tensor (same numbers as SimLab default).
_SMALLSAT_J = np.diag([0.05, 0.06, 0.07])

# Off-diagonal products so the sweep is not secretly diagonal-only.
_ASYM_J = np.array(
    [
        [2.0, 0.1, 0.0],
        [0.1, 3.0, 0.15],
        [0.0, 0.15, 4.0],
    ]
)

# Mild MC-style mismatch used by the SimLab harness default (±5%).
INERTIA_FRAC = 0.05

# Sweep horizon is short so CI stays cheap.  Classical RK4 is not symplectic;
# quaternion renormalization also leaks a little inertial-frame h.  Observed
# peaks on seed 11, N=8, T=2 s, ±5% principal scale:
#   dt=0.005: max ΔE/E ~5e-15,  max |Δh_I|/|h| ~1e-12,  max Δ|h_b|/|h| ~3e-15
#   dt=0.01:  max ΔE/E ~2e-14,  max |Δh_I|/|h| ~7e-12,  max Δ|h_b|/|h| ~1e-14
#   dt=0.02:  max ΔE/E ~4e-12,  max |Δh_I|/|h| ~6e-10,  max Δ|h_b|/|h| ~2e-12
# Bounds sit ~100× above those peaks so they stay honest without being flaky.
QUAT_NORM_ABS_TOL = 1e-12
ENERGY_REL_TOL = {
    0.005: 1e-12,
    0.01: 5e-12,
    0.02: 5e-10,
}
H_INERTIAL_REL_TOL = {
    0.005: 2e-10,
    0.01: 1e-9,
    0.02: 1e-7,
}
H_BODY_NORM_REL_TOL = {
    0.005: 1e-12,
    0.01: 5e-12,
    0.02: 5e-10,
}
T_FINAL = 2.0
N_TRIALS = 8
DTS = (0.005, 0.01, 0.02)


def _random_unit_quat(rng: np.random.Generator) -> np.ndarray:
    return quat_normalize(rng.normal(size=4))


def test_validate_inertia_rejects_nonfinite():
    J = _SMALLSAT_J.copy()
    J[0, 1] = np.nan
    J[1, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        validate_inertia(J)
    with pytest.raises(ValueError, match="finite"):
        validate_inertia(np.diag([1.0, np.inf, 2.0]))


def test_rk4_rejects_nonfinite_dt():
    def fun(_t, y):
        return y

    with pytest.raises(ValueError, match="dt must be positive"):
        rk4_step(fun, 0.0, np.array([1.0]), float("nan"))
    with pytest.raises(ValueError, match="dt must be positive"):
        rk4_step(fun, 0.0, np.array([1.0]), float("inf"))


def test_scale_principal_inertia_reconstructs_and_validates():
    J = validate_inertia(_ASYM_J)
    J_same = scale_principal_inertia(J, 1.0)
    np.testing.assert_allclose(J_same, J, atol=1e-12)
    J_uniform = scale_principal_inertia(J, 1.1)
    np.testing.assert_allclose(J_uniform, 1.1 * J, atol=1e-12)
    body = RigidBody(J_uniform)
    np.testing.assert_allclose(body.principal_moments, RigidBody(J).principal_moments * 1.1)
    assert abs(np.linalg.det(body.principal_axes) - 1.0) < 1e-12
    np.testing.assert_allclose(body.principal_axes.T @ body.principal_axes, np.eye(3), atol=1e-12)
    s = np.array([0.97, 1.02, 1.04])
    J_scaled = scale_principal_inertia(_SMALLSAT_J, s)
    validate_inertia(J_scaled)
    np.testing.assert_allclose(np.diag(J_scaled), np.array([0.05, 0.06, 0.07]) * s)
    with pytest.raises(ValueError, match="positive"):
        scale_principal_inertia(J, [1.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="finite"):
        scale_principal_inertia(J, [1.0, np.nan, 1.0])
    with pytest.raises(ValueError, match="principal moments"):
        # Independent scales that break I_i + I_j >= I_k.
        scale_principal_inertia(np.diag([1.0, 1.0, 1.5]), [0.5, 0.5, 1.5])


def test_random_principal_scale_bounds_and_rejects_bad_frac():
    rng = np.random.default_rng(3)
    np.testing.assert_allclose(random_principal_scale(rng, 0.0), np.ones(3))
    draws = np.vstack([random_principal_scale(rng, 0.08) for _ in range(40)])
    assert np.all(draws >= 1.0 - 0.08 - 1e-15)
    assert np.all(draws <= 1.0 + 0.08 + 1e-15)
    assert np.all(draws > 0.0)
    with pytest.raises(ValueError, match="frac"):
        random_principal_scale(rng, 1.0)
    with pytest.raises(ValueError, match="frac"):
        random_principal_scale(rng, -0.01)


def test_perturb_principal_inertia_stays_physical():
    rng = np.random.default_rng(9)
    for J0 in (_SMALLSAT_J, _ASYM_J):
        J0 = validate_inertia(J0)
        for _ in range(24):
            J = perturb_principal_inertia(J0, INERTIA_FRAC, rng)
            validate_inertia(J)
            moments = np.linalg.eigvalsh(J)
            assert np.all(moments > 0.0)
            rel = np.linalg.norm(J - J0) / np.linalg.norm(J0)
            assert rel < 3.0 * INERTIA_FRAC + 1e-12
        J_same = perturb_principal_inertia(J0, 0.0, rng)
        np.testing.assert_allclose(J_same, J0)
    with pytest.raises(ValueError, match="frac"):
        perturb_principal_inertia(_SMALLSAT_J, 1.0, rng)


def test_perturb_fallback_uniform_scale_when_independent_fails():
    """Near-planar J: independent ±40% scales often break the triangle."""
    rng = np.random.default_rng(1)
    J0 = validate_inertia(np.diag([1.0, 1.0, 1.999]))
    for _ in range(12):
        J = perturb_principal_inertia(J0, 0.4, rng, max_tries=1)
        validate_inertia(J)
        moments = np.sort(np.linalg.eigvalsh(J))
        assert np.all(moments > 0.0)


def _torque_free_drift(body: RigidBody, q0, omega0, dt: float, t_final: float):
    q = quat_normalize(q0)
    omega = np.asarray(omega0, dtype=float).reshape(3).copy()
    e0 = body.kinetic_energy(omega)
    h0 = body.angular_momentum_inertial(q, omega)
    hb0 = np.linalg.norm(body.angular_momentum_body(omega))
    n = int(round(t_final / dt))
    for _ in range(n):
        q, omega = step_rigid_body(body, q, omega, np.zeros(3), dt)
    assert np.all(np.isfinite(q))
    assert np.all(np.isfinite(omega))
    e1 = body.kinetic_energy(omega)
    h1 = body.angular_momentum_inertial(q, omega)
    hb1 = np.linalg.norm(body.angular_momentum_body(omega))
    energy_rel = abs(e1 - e0) / max(abs(e0), 1e-16)
    h_rel = float(np.linalg.norm(h1 - h0) / max(np.linalg.norm(h0), 1e-16))
    hb_rel = abs(hb1 - hb0) / max(hb0, 1e-16)
    q_err = abs(float(np.linalg.norm(q)) - 1.0)
    return energy_rel, h_rel, hb_rel, q_err


def test_torque_free_invariants_under_principal_perturbation_and_varied_dt():
    """MC-ish plant sweep: small I mismatch × varied dt, torque-free RK4.

    Asserts no NaN, unit quaternion, and bounded energy / inertial-momentum
    drift.  Tolerances are documented per ``dt`` in this module.
    """
    rng = np.random.default_rng(11)
    for dt in DTS:
        e_tol = ENERGY_REL_TOL[dt]
        h_tol = H_INERTIAL_REL_TOL[dt]
        hb_tol = H_BODY_NORM_REL_TOL[dt]
        for _ in range(N_TRIALS):
            J = perturb_principal_inertia(_SMALLSAT_J, INERTIA_FRAC, rng)
            body = RigidBody(J)
            q0 = _random_unit_quat(rng)
            omega0 = rng.uniform(-0.8, 0.8, size=3)
            energy_rel, h_rel, hb_rel, q_err = _torque_free_drift(
                body, q0, omega0, dt, T_FINAL
            )
            assert q_err < QUAT_NORM_ABS_TOL
            assert energy_rel < e_tol
            assert h_rel < h_tol
            assert hb_rel < hb_tol


def test_torque_free_invariants_on_rotated_asymmetric_perturbation():
    """Same integrals when J is not principal in the body frame."""
    rng = np.random.default_rng(4)
    dt = 0.01
    J = perturb_principal_inertia(_ASYM_J, INERTIA_FRAC, rng)
    body = RigidBody(J)
    q0 = _random_unit_quat(rng)
    omega0 = np.array([0.3, -0.5, 0.4])
    energy_rel, h_rel, hb_rel, q_err = _torque_free_drift(body, q0, omega0, dt, T_FINAL)
    assert q_err < QUAT_NORM_ABS_TOL
    assert energy_rel < ENERGY_REL_TOL[dt]
    assert h_rel < H_INERTIAL_REL_TOL[dt]
    assert hb_rel < H_BODY_NORM_REL_TOL[dt]


def test_unit_quat_under_torque_and_perturbed_inertia():
    rng = np.random.default_rng(7)
    body = RigidBody(perturb_principal_inertia(_ASYM_J, INERTIA_FRAC, rng))
    q = _random_unit_quat(rng)
    omega = rng.normal(scale=0.3, size=3)
    for dt in DTS:
        qi, wi = q.copy(), omega.copy()
        for _ in range(int(round(0.5 / dt))):
            tau = rng.normal(scale=0.04, size=3)
            qi, wi = step_rigid_body(body, qi, wi, tau, dt)
            assert np.all(np.isfinite(qi))
            assert np.all(np.isfinite(wi))
        assert abs(float(np.linalg.norm(qi)) - 1.0) < QUAT_NORM_ABS_TOL

"""Torque-free polhode / energy–Casimir helpers and tennis-racket theorem."""

import numpy as np
import pytest

from attitude_sim.plant import RigidBody, inertia_from_principal, step_rigid_body
from attitude_sim.polhode import (
    PolhodeRegime,
    SpinStability,
    angular_momentum_body,
    angular_momentum_inertial,
    casimir_h2,
    energy_casimir,
    herpolhode_plane_residual,
    intermediate_axis_theorem,
    intermediate_principal_axis,
    kinetic_energy,
    omega_inertial,
    polhode_regime,
    polhode_residuals,
    principal_spin_stability,
    sample_herpolhode,
    sample_polhode,
    sample_polhode_intersection,
    transverse_lambda_squared,
)
from attitude_sim.quaternions import quat_normalize

# Asymmetric J with products of inertia (same numbers as tests/test_plant.py).
_ASYM_J = 0.5 * (
    np.array(
        [
            [2.0, 0.1, 0.0],
            [0.1, 3.0, 0.15],
            [0.0, 0.15, 4.0],
        ]
    )
    + np.array(
        [
            [2.0, 0.1, 0.0],
            [0.1, 3.0, 0.15],
            [0.0, 0.15, 4.0],
        ]
    ).T
)

_TRIAXIAL = np.diag([1.0, 2.0, 3.0])
ENERGY_REL_TOL = 1e-12
H2_REL_TOL = 1e-12
H_INERTIAL_ATOL = 1e-12
PLANE_ATOL = 1e-11
INTERSECTION_ATOL = 1e-10


def test_kinetic_energy_and_momentum_match_rigid_body():
    body = RigidBody(_ASYM_J)
    q = quat_normalize([0.4, 0.3, 0.2, 0.8])
    omega = np.array([-0.5, 0.4, 0.7])
    t = kinetic_energy(_ASYM_J, omega)
    hb = angular_momentum_body(_ASYM_J, omega)
    hi = angular_momentum_inertial(_ASYM_J, q, omega)
    assert abs(t - body.kinetic_energy(omega)) < 1e-15
    np.testing.assert_allclose(hb, body.angular_momentum_body(omega), atol=1e-15)
    np.testing.assert_allclose(hi, body.angular_momentum_inertial(q, omega), atol=1e-15)
    # Scalar energy is the same pairing in the inertial frame.
    w_I = omega_inertial(q, omega)
    assert abs(t - 0.5 * float(w_I @ hi)) < 1e-14
    t2, h2 = energy_casimir(body, omega)
    assert abs(t2 - t) < 1e-15
    assert abs(h2 - casimir_h2(_ASYM_J, omega)) < 1e-15
    assert abs(h2 - float(hi @ hi)) < 1e-14


def test_energy_casimir_accepts_rigid_body_and_rejects_nonfinite_omega():
    body = RigidBody(_TRIAXIAL)
    omega = np.array([0.1, -0.2, 0.3])
    np.testing.assert_allclose(energy_casimir(body, omega), energy_casimir(_TRIAXIAL, omega))
    with pytest.raises(ValueError, match="finite"):
        kinetic_energy(_TRIAXIAL, [np.nan, 0.0, 0.0])


def test_polhode_residuals_zero_at_defining_omega():
    omega = np.array([0.3, -0.8, 0.2])
    t, h2 = energy_casimir(_ASYM_J, omega)
    r_t, r_h = polhode_residuals(_ASYM_J, omega, t, h2)
    assert abs(r_t) < 1e-15
    assert abs(r_h) < 1e-15


def test_sample_polhode_conserves_energy_and_casimir():
    omega0 = np.array([0.3, -0.8, 0.2])
    traj = sample_polhode(_ASYM_J, omega0, t_final=4.0, dt=0.002)
    t0, h20 = traj.T[0], traj.h2[0]
    assert abs(t0) > 0.0
    assert np.max(np.abs(traj.T - t0)) / abs(t0) < ENERGY_REL_TOL
    assert np.max(np.abs(traj.h2 - h20)) / abs(h20) < H2_REL_TOL
    np.testing.assert_allclose(traj.omega[0], omega0)
    assert traj.q is None
    assert traj.omega_inertial is None
    r_t, r_h = polhode_residuals(_ASYM_J, traj.omega[-1], t0, h20)
    assert abs(r_t) / max(2.0 * abs(t0), 1.0) < 1e-11
    assert abs(r_h) / max(abs(h20), 1.0) < 1e-11


def test_sample_herpolhode_invariable_plane_and_inertial_h():
    q0 = quat_normalize([0.5, -0.2, 0.1, 0.8])
    omega0 = np.array([0.25, 0.4, -0.15])
    traj = sample_herpolhode(_ASYM_J, omega0, t_final=3.0, dt=0.002, q0=q0)
    assert traj.q is not None and traj.omega_inertial is not None
    assert traj.h_inertial is not None
    h0 = traj.h_inertial[0]
    t0 = traj.T[0]
    np.testing.assert_allclose(traj.h_inertial - h0, 0.0, atol=H_INERTIAL_ATOL)
    plane = [
        herpolhode_plane_residual(w_I, h0, t0) for w_I in traj.omega_inertial
    ]
    assert np.max(np.abs(plane)) < PLANE_ATOL
    np.testing.assert_allclose(traj.q[0], q0, atol=1e-15)
    np.testing.assert_allclose(
        traj.omega_inertial[0],
        omega_inertial(q0, omega0),
        atol=1e-15,
    )


def test_sample_polhode_t_final_zero_and_rejects_bad_dt():
    omega0 = np.array([0.1, 0.2, 0.3])
    traj = sample_polhode(_TRIAXIAL, omega0, t_final=0.0, dt=0.01)
    assert traj.t.shape == (1,)
    np.testing.assert_allclose(traj.omega[0], omega0)
    with pytest.raises(ValueError, match="dt must be positive"):
        sample_polhode(_TRIAXIAL, omega0, t_final=1.0, dt=0.0)
    with pytest.raises(ValueError, match="t_final"):
        sample_polhode(_TRIAXIAL, omega0, t_final=-1.0, dt=0.01)


def test_geometric_intersection_matches_omega0_and_stays_on_ellipsoids():
    omega0 = np.array([0.4, -0.3, 0.5])
    pts = sample_polhode_intersection(_ASYM_J, omega0, n=120)
    assert pts.shape == (120, 3)
    np.testing.assert_allclose(pts[0], omega0, atol=1e-8)
    t0, h20 = energy_casimir(_ASYM_J, omega0)
    for w in pts[::7]:
        r_t, r_h = polhode_residuals(_ASYM_J, w, t0, h20)
        assert abs(r_t) < INTERSECTION_ATOL
        assert abs(r_h) < INTERSECTION_ATOL
    with pytest.raises(ValueError, match="n must be"):
        sample_polhode_intersection(_TRIAXIAL, omega0, n=0)


def test_geometric_intersection_principal_spin_is_a_point():
    omega0 = np.array([0.0, 0.0, 1.1])
    pts = sample_polhode_intersection(_TRIAXIAL, omega0, n=16)
    np.testing.assert_allclose(pts - omega0, 0.0, atol=1e-15)
    assert polhode_regime(_TRIAXIAL, omega0) is PolhodeRegime.POINT


def test_polhode_regime_min_max_separatrix():
    # I = (1, 2, 3).  |h|² ≦ 2 T I_mid encircles min / max.
    w_min = np.array([1.0, 0.2, 0.1])
    w_max = np.array([0.1, 0.2, 1.0])
    assert polhode_regime(_TRIAXIAL, w_min) is PolhodeRegime.AROUND_MIN
    assert polhode_regime(_TRIAXIAL, w_max) is PolhodeRegime.AROUND_MAX
    # Separatrix: |h|² = 2 T I_mid.  For I=(1,2,3) this is ω1² = ω3²
    # with ω2 = 0 on one branch of the figure-eight.
    i_mid = 2.0
    c = 0.2
    # a² I1 (I2 − I1) = c² I3 (I3 − I2) ⇒ a² = 3 c² for I=(1,2,3).
    omega_sep = np.array([c * np.sqrt(3.0), 0.0, c])
    t0, h20 = energy_casimir(_TRIAXIAL, omega_sep)
    assert abs(h20 - 2.0 * t0 * i_mid) < 1e-12
    assert polhode_regime(_TRIAXIAL, omega_sep) is PolhodeRegime.SEPARATRIX


def test_spherical_polhode_is_a_point_and_spin_is_marginal():
    J = 2.0 * np.eye(3)
    omega0 = np.array([0.4, -0.3, 0.25])
    traj = sample_polhode(J, omega0, t_final=2.0, dt=0.01)
    np.testing.assert_allclose(traj.omega - omega0, 0.0, atol=1e-14)
    assert polhode_regime(J, omega0) is PolhodeRegime.POINT
    pts = sample_polhode_intersection(J, omega0, n=8)
    np.testing.assert_allclose(pts - omega0, 0.0, atol=1e-14)
    assert intermediate_axis_theorem(J) == (
        SpinStability.MARGINAL,
        SpinStability.MARGINAL,
        SpinStability.MARGINAL,
    )


def test_axisymmetric_unique_axis_stable_repeated_marginal():
    J = np.diag([2.0, 2.0, 3.0])
    assert principal_spin_stability(J, 2) is SpinStability.STABLE
    assert principal_spin_stability(J, 0) is SpinStability.MARGINAL
    assert principal_spin_stability(J, 1) is SpinStability.MARGINAL
    # Closed-form precession: transverse rate rotates, energy/Casimir hold.
    omega0 = np.array([0.3, 0.1, 0.8])
    traj = sample_polhode(J, omega0, t_final=5.0, dt=0.001)
    t0, h20 = traj.T[0], traj.h2[0]
    assert np.max(np.abs(traj.T - t0)) / t0 < ENERGY_REL_TOL
    assert np.max(np.abs(traj.h2 - h20)) / h20 < H2_REL_TOL


def test_intermediate_axis_theorem_triaxial_analytic():
    J = _TRIAXIAL
    assert principal_spin_stability(J, 0) is SpinStability.STABLE
    assert principal_spin_stability(J, 1) is SpinStability.UNSTABLE
    assert principal_spin_stability(J, 2) is SpinStability.STABLE
    assert intermediate_axis_theorem(J) == (
        SpinStability.STABLE,
        SpinStability.UNSTABLE,
        SpinStability.STABLE,
    )
    # λ² formula: spin about min axis k=0, I=(1,2,3), Ω=1
    # λ² = −(I0−I1)(I0−I2)/(I1 I2) = −(1−2)(1−3)/(2·3) = −(−1)(−2)/6 = −1/3
    np.testing.assert_allclose(transverse_lambda_squared(J, 0, 1.0), -1.0 / 3.0)
    # Intermediate: λ² = −(2−3)(2−1)/(3·1) = −(−1)(1)/3 = 1/3 > 0
    np.testing.assert_allclose(transverse_lambda_squared(J, 1, 1.0), 1.0 / 3.0)
    # Max: λ² = −(3−1)(3−2)/(1·2) = −(2)(1)/2 = −1
    np.testing.assert_allclose(transverse_lambda_squared(J, 2, 1.0), -1.0)
    with pytest.raises(ValueError, match="axis must be"):
        transverse_lambda_squared(J, 3)
    with pytest.raises(ValueError, match="spin_rate"):
        transverse_lambda_squared(J, 0, spin_rate=np.inf)


def test_intermediate_axis_theorem_rotated_principal_frame():
    # 90° about z: body x ← y, body y ← −x.  Moments (1, 2, 2.5) stay triaxial.
    rz = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    J = inertia_from_principal([1.0, 2.0, 2.5], rz)
    assert intermediate_axis_theorem(J) == (
        SpinStability.STABLE,
        SpinStability.UNSTABLE,
        SpinStability.STABLE,
    )
    e_mid = intermediate_principal_axis(J)
    # Intermediate moment is 2, which is original body y, mapped to −x (2nd col of Rz).
    np.testing.assert_allclose(np.abs(e_mid @ rz[:, 1]), 1.0, atol=1e-12)


def test_min_and_max_axis_spin_bounded_under_perturbation():
    """Tennis-racket: min/max principal spin stays near the axis (polhode loops)."""
    body = RigidBody(_TRIAXIAL)
    dt = 0.002
    t_final = 12.0
    n = int(t_final / dt)
    pert = 0.03
    for axis, rate in ((0, 1.0), (2, 1.0)):
        omega = np.full(3, pert)
        omega[axis] = rate
        q = np.array([1.0, 0.0, 0.0, 0.0])
        for _ in range(n):
            q, omega = step_rigid_body(body, q, omega, np.zeros(3), dt)
        # Spin component keeps sign and stays O(1); transverse stays O(pert).
        assert omega[axis] > 0.5 * rate
        transverse = [omega[i] for i in range(3) if i != axis]
        assert np.linalg.norm(transverse) < 8.0 * pert
        assert principal_spin_stability(_TRIAXIAL, axis) is SpinStability.STABLE


def test_intermediate_axis_spin_is_unstable():
    """Tennis-racket: a small tilt off the intermediate axis grows (Dzhanibekov)."""
    body = RigidBody(_TRIAXIAL)
    dt = 0.002
    t_final = 18.0
    n = int(t_final / dt)
    pert = 0.03
    omega = np.array([pert, 1.0, pert])
    q = np.array([1.0, 0.0, 0.0, 0.0])
    growth = []
    for k in range(n):
        q, omega = step_rigid_body(body, q, omega, np.zeros(3), dt)
        if k % 50 == 0:
            growth.append(float(np.hypot(omega[0], omega[2])))
    # Transverse rate leaves the O(pert) neighbourhood; the mid-axis component
    # does not remain a near-equilibrium (flip / large polhode).
    assert max(growth) > 10.0 * pert
    assert principal_spin_stability(_TRIAXIAL, 1) is SpinStability.UNSTABLE


def test_applied_torque_breaks_casimirs():
    omega = np.array([0.3, -0.8, 0.2])
    t0, h20 = energy_casimir(_ASYM_J, omega)
    q = np.array([1.0, 0.0, 0.0, 0.0])
    body = RigidBody(_ASYM_J)
    tau = np.array([0.05, -0.02, 0.03])
    for _ in range(400):
        q, omega = step_rigid_body(body, q, omega, tau, 0.002)
    t1, h21 = energy_casimir(_ASYM_J, omega)
    assert abs(t1 - t0) / abs(t0) > 1e-3
    assert abs(h21 - h20) / abs(h20) > 1e-3

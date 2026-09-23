"""Analytic checks for gravity-gradient, residual-dipole, aero, and SRP torques."""

import numpy as np
import pytest

from attitude_sim.disturbances import (
    ATM_H_REF,
    ATM_RHO_REF,
    ATM_SCALE_HEIGHT,
    EARTH_MAG_MOMENT,
    MU_EARTH,
    P_SRP_1AU,
    R_EARTH,
    AerodynamicTorque,
    CircularOrbit,
    EnvironmentalTorques,
    ExponentialAtmosphere,
    GravityGradientTorque,
    OrbitState,
    ResidualDipoleTorque,
    SolarRadiationPressureTorque,
    aerodynamic_force,
    aerodynamic_torque,
    body_to_inertial,
    dipole_field_eci,
    earth_dipole_field_eci,
    earth_dipole_moment_eci,
    exponential_density,
    gravity_gradient_torque,
    in_cylindrical_umbra,
    inertial_to_body,
    magnetic_dipole_torque,
    magnetic_field_body,
    magnetic_field_eci,
    orbit_normal_dipole_field_eci,
    relative_velocity_eci,
    srp_force,
    srp_torque,
)
from attitude_sim.quaternions import axis_angle_to_quat, quat_normalize

# LEO-scale circular orbit used by the closed-form cases.
_R = 7.0e6
_N = np.sqrt(MU_EARTH / _R**3)
_MU_R3 = MU_EARTH / _R**3
_Q_ID = np.array([1.0, 0.0, 0.0, 0.0])
_W0 = np.zeros(3)


def _equatorial_state(u: float = 0.0) -> OrbitState:
    cu, su = np.cos(u), np.sin(u)
    r = _R * np.array([cu, su, 0.0])
    v = _R * _N * np.array([-su, cu, 0.0])
    return OrbitState(r, v, MU_EARTH)


def test_circular_orbit_equatorial_state_at_quarter_period():
    circ = CircularOrbit(radius=_R, inclination=0.0, raan=0.0, arg_latitude0=0.0)
    s0 = circ.state_at(0.0)
    np.testing.assert_allclose(s0.r_eci, [_R, 0.0, 0.0], atol=1e-8)
    np.testing.assert_allclose(s0.v_eci, [0.0, _R * _N, 0.0], rtol=1e-12)
    np.testing.assert_allclose(s0.h_hat, [0.0, 0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(s0.r_hat, [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(s0.nadir_hat, [-1.0, 0.0, 0.0], atol=1e-12)
    t_quarter = 0.5 * np.pi / circ.mean_motion
    s1 = circ.state_at(t_quarter)
    np.testing.assert_allclose(s1.r_eci, [0.0, _R, 0.0], atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(s1.v_eci), _R * _N, rtol=1e-12)
    assert abs(np.dot(s1.r_eci, s1.v_eci)) / (_R * _R * _N) < 1e-12


def test_circular_orbit_inclined_has_out_of_plane_position():
    circ = CircularOrbit(radius=_R, inclination=np.deg2rad(60.0), raan=0.0)
    t_quarter = 0.5 * np.pi / circ.mean_motion
    s = circ.state_at(t_quarter)
    np.testing.assert_allclose(s.r_eci[0], 0.0, atol=1e-6)
    np.testing.assert_allclose(s.r_eci[1], _R * np.cos(np.deg2rad(60.0)), atol=1e-6)
    np.testing.assert_allclose(s.r_eci[2], _R * np.sin(np.deg2rad(60.0)), atol=1e-6)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_gg_zero_when_principal_axis_along_zenith(axis):
    J = np.diag([1.5, 2.5, 3.5])
    r_hat = np.zeros(3)
    r_hat[axis] = 1.0
    tau = gravity_gradient_torque(J, r_hat, _MU_R3)
    np.testing.assert_allclose(tau, 0.0, atol=1e-15)


def test_gg_zero_for_principal_nadir_alignment_via_quaternion():
    """Body z along nadir (−x_I at this epoch): principal J ⇒ τ_gg = 0."""
    J = np.diag([0.05, 0.06, 0.07])
    # R_y(−90°): body z maps to inertial −x (nadir if r_I = +x).
    q = axis_angle_to_quat([0.0, 1.0, 0.0], -0.5 * np.pi)
    state = _equatorial_state(0.0)
    gg = GravityGradientTorque(J)
    tau = gg.tau_body(q, _W0, orbit=state)
    np.testing.assert_allclose(tau, 0.0, atol=1e-18)
    # Same with zenith along body +x at identity (r̂_b = e_x).
    tau_id = gg.tau_body(_Q_ID, np.array([0.4, -0.2, 0.1]), orbit=state)
    np.testing.assert_allclose(tau_id, 0.0, atol=1e-18)


def test_gg_nadir_equals_zenith():
    J = np.diag([1.5, 2.5, 3.5])
    r_hat = np.array([0.0, 1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)])
    tau_r = gravity_gradient_torque(J, r_hat, _MU_R3)
    tau_n = gravity_gradient_torque(J, -r_hat, _MU_R3)
    np.testing.assert_allclose(tau_r, tau_n, atol=1e-15)


def test_gg_closed_form_transverse_principal():
    """J = diag(I1,I2,I3), r̂_b = (0, 1/√2, 1/√2) ⇒ τ = 3μ/r³ ((I3−I2)/2, 0, 0)."""
    I1, I2, I3 = 1.5, 2.5, 3.5
    J = np.diag([I1, I2, I3])
    r_hat = np.array([0.0, 1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)])
    tau = gravity_gradient_torque(J, r_hat, _MU_R3)
    expected = 3.0 * _MU_R3 * np.array([(I3 - I2) / 2.0, 0.0, 0.0])
    np.testing.assert_allclose(tau, expected, atol=1e-18)
    assert abs(np.dot(tau, r_hat)) < 1e-15


def test_gg_spherical_inertia_is_identically_zero():
    J = 2.0 * np.eye(3)
    rng = np.random.default_rng(4)
    for _ in range(8):
        r_hat = rng.normal(size=3)
        tau = gravity_gradient_torque(J, r_hat, _MU_R3)
        np.testing.assert_allclose(tau, 0.0, atol=1e-15)


def test_gg_independent_of_omega_and_orthogonal_to_zenith():
    J = np.diag([0.05, 0.06, 0.07])
    q = quat_normalize([0.4, 0.3, 0.2, 0.8])
    gg = GravityGradientTorque(J, orbit=CircularOrbit(radius=_R))
    t = 12.0
    tau_a = gg.tau_body(q, np.zeros(3), t)
    tau_b = gg.tau_body(q, np.array([1.0, -2.0, 0.5]), t)
    np.testing.assert_allclose(tau_a, tau_b)
    r_hat_b = inertial_to_body(q, gg.orbit.state_at(t).r_hat)
    assert abs(float(np.dot(tau_a, r_hat_b))) < 1e-18


def test_gg_tau_body_prefers_explicit_orbit_state():
    J = np.diag([1.5, 2.5, 3.5])
    bound = CircularOrbit(radius=_R, arg_latitude0=0.0)
    other = _equatorial_state(0.25 * np.pi)  # r̂ = (1,1,0)/√2, not a principal axis
    gg = GravityGradientTorque(J, orbit=bound)
    tau_t = gg.tau_body(_Q_ID, _W0, 0.0)
    tau_orbit = gg.tau_body(_Q_ID, _W0, 0.0, orbit=other)
    # Identity + r along +x (t=0) is principal alignment → 0.
    np.testing.assert_allclose(tau_t, 0.0, atol=1e-18)
    r_hat_b = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
    expected = gravity_gradient_torque(J, r_hat_b, other.mu_over_r3)
    np.testing.assert_allclose(tau_orbit, expected, atol=1e-18)
    assert np.linalg.norm(tau_orbit) > 0.0


def test_magnetic_torque_zero_when_m_parallel_B():
    B = np.array([1.2e-5, -3.4e-6, 8.0e-6])
    for scale in (0.0, 1.0, -2.5, 7.0):
        tau = magnetic_dipole_torque(scale * B, B)
        np.testing.assert_allclose(tau, 0.0, atol=1e-18)


def test_magnetic_torque_right_hand_cross_product():
    m = np.array([0.2, 0.0, 0.0])
    B = np.array([0.0, 3.0e-5, 0.0])
    tau = magnetic_dipole_torque(m, B)
    np.testing.assert_allclose(tau, [0.0, 0.0, 0.2 * 3.0e-5], atol=1e-18)
    assert abs(float(np.dot(tau, m))) < 1e-18
    assert abs(float(np.dot(tau, B))) < 1e-18


def test_B_body_identity_and_z_rotation():
    B_I = np.array([1.0e-5, -2.0e-5, 3.0e-6])
    np.testing.assert_allclose(magnetic_field_body(_Q_ID, B_I), B_I)
    qz = axis_angle_to_quat([0.0, 0.0, 1.0], 0.5 * np.pi)
    B_b = magnetic_field_body(qz, B_I)
    # v_b = R_z(90°)ᵀ v_I = [By, −Bx, Bz]
    np.testing.assert_allclose(B_b, [B_I[1], -B_I[0], B_I[2]], atol=1e-18)
    np.testing.assert_allclose(body_to_inertial(qz, B_b), B_I, atol=1e-18)


def test_untilted_equatorial_dipole_matches_orbit_normal():
    """Untilted m_E = −μ_m ẑ on a prograde equatorial orbit ⇒ B = (μ_m/r³) ẑ."""
    state = _equatorial_state(1.1)
    B_tilt0 = earth_dipole_field_eci(state.r_eci, moment=EARTH_MAG_MOMENT, tilt_rad=0.0)
    B_on = orbit_normal_dipole_field_eci(state, moment=EARTH_MAG_MOMENT, sign=1.0)
    expected = (EARTH_MAG_MOMENT / _R**3) * np.array([0.0, 0.0, 1.0])
    np.testing.assert_allclose(B_tilt0, expected, rtol=1e-12, atol=1e-18)
    np.testing.assert_allclose(B_on, expected, rtol=1e-12, atol=1e-18)
    # Dipole formula at several true anomalies stays on +z (equator).
    for u in (0.0, 0.7, 2.1, 4.0):
        s = _equatorial_state(u)
        B = earth_dipole_field_eci(s.r_eci, tilt_rad=0.0)
        np.testing.assert_allclose(B, expected, rtol=1e-12, atol=1e-16)


def test_tilted_dipole_off_equator_has_radial_component():
    m = earth_dipole_moment_eci(tilt_rad=0.0)
    np.testing.assert_allclose(m, [0.0, 0.0, -EARTH_MAG_MOMENT])
    r = np.array([0.0, 0.0, _R])  # geographic pole
    B = dipole_field_eci(r, m)
    # At the north pole, r̂ = ẑ, m = −μ_m ẑ ⇒ B = [3(m·r̂)r̂ − m]/r³ = −2 μ_m/r³ ẑ
    np.testing.assert_allclose(B, [0.0, 0.0, -2.0 * EARTH_MAG_MOMENT / _R**3])
    tilted = earth_dipole_field_eci(r, tilt_rad=np.deg2rad(11.5), ra_rad=0.4)
    assert abs(tilted[0]) + abs(tilted[1]) > 1e-8
    assert not np.allclose(tilted, B)


def test_residual_dipole_tau_body_zero_when_m_along_B_body():
    state = _equatorial_state(0.0)
    q = quat_normalize([0.2, 0.5, -0.1, 0.8])
    mag = ResidualDipoleTorque(np.ones(3), model="tilted", tilt_rad=0.0)
    B_b = mag.B_body(q, state)
    mag_parallel = ResidualDipoleTorque(B_b, model="tilted", tilt_rad=0.0)
    tau = mag_parallel.tau_body(q, np.array([0.3, 0.0, -0.1]), orbit=state)
    np.testing.assert_allclose(tau, 0.0, atol=1e-18)


def test_residual_dipole_orbit_normal_cross_product_in_body():
    state = _equatorial_state(0.0)
    m_b = np.array([0.15, 0.0, 0.0])
    mag = ResidualDipoleTorque(m_b, model="orbit_normal")
    # Identity: B_b = B_I = (μ_m/r³) ẑ
    B_b = mag.B_body(_Q_ID, state)
    np.testing.assert_allclose(B_b, [0.0, 0.0, EARTH_MAG_MOMENT / _R**3], rtol=1e-12)
    tau = mag.tau_body(_Q_ID, _W0, orbit=state)
    expected = np.cross(m_b, B_b)
    np.testing.assert_allclose(tau, expected)
    np.testing.assert_allclose(tau, [0.0, -0.15 * B_b[2], 0.0], rtol=1e-12)


def test_magnetic_field_eci_rejects_unknown_model():
    with pytest.raises(ValueError, match="model"):
        magnetic_field_eci(_equatorial_state(), model="igrf")


def test_tau_body_requires_orbit_or_time():
    gg = GravityGradientTorque(np.diag([1.5, 2.5, 3.5]))
    mag = ResidualDipoleTorque([0.1, 0.0, 0.0])
    with pytest.raises(ValueError, match="CircularOrbit"):
        gg.tau_body(_Q_ID, _W0)
    with pytest.raises(ValueError, match="t is required"):
        GravityGradientTorque(np.diag([1.5, 2.5, 3.5]), orbit=CircularOrbit(_R)).tau_body(
            _Q_ID, _W0
        )
    with pytest.raises(ValueError, match="CircularOrbit"):
        mag.tau_body(_Q_ID, _W0)


def test_environmental_sum_matches_parts():
    J = np.diag([0.05, 0.06, 0.07])
    circ = CircularOrbit(radius=_R, inclination=np.deg2rad(30.0))
    q = quat_normalize([0.6, 0.1, -0.2, 0.7])
    omega = np.array([0.01, -0.02, 0.03])
    t = 25.0
    gg = GravityGradientTorque(J, orbit=circ)
    mag = ResidualDipoleTorque([0.08, -0.01, 0.02], orbit=circ, model="tilted")
    env = EnvironmentalTorques(gravity_gradient=gg, residual_dipole=mag)
    tau = env.tau_body(q, omega, t)
    np.testing.assert_allclose(
        tau,
        gg.tau_body(q, omega, t) + mag.tau_body(q, omega, t),
    )
    # Explicit orbit state is equivalent to t on the bound circular orbit.
    state = circ.state_at(t)
    np.testing.assert_allclose(env.tau_body(q, omega, orbit=state), tau)


def test_inertial_body_roundtrip():
    q = axis_angle_to_quat([1.0, 2.0, -0.5], 0.7)
    v_I = np.array([3.0, -1.0, 4.0])
    v_b = inertial_to_body(q, v_I)
    np.testing.assert_allclose(body_to_inertial(q, v_b), v_I, atol=1e-12)


def test_exponential_density_at_reference_and_decreases_with_altitude():
    rho0 = exponential_density(R_EARTH + ATM_H_REF)
    np.testing.assert_allclose(rho0, ATM_RHO_REF)
    atm = ExponentialAtmosphere()
    np.testing.assert_allclose(atm.density(R_EARTH + ATM_H_REF), ATM_RHO_REF)
    rho_high = atm.density(R_EARTH + ATM_H_REF + ATM_SCALE_HEIGHT)
    np.testing.assert_allclose(rho_high, ATM_RHO_REF / np.e, rtol=1e-12)
    assert rho_high < rho0
    with pytest.raises(ValueError, match="scale_height"):
        ExponentialAtmosphere(scale_height=0.0)
    with pytest.raises(ValueError, match="rho_ref"):
        exponential_density(R_EARTH + ATM_H_REF, rho_ref=-1.0)


def test_aero_force_opposes_ram_velocity():
    v = np.array([0.0, 7500.0, 0.0])
    rho, cd, area = 3e-12, 2.2, 0.4
    F = aerodynamic_force(rho, v, cd, area)
    expected = -0.5 * rho * 7500.0**2 * cd * area * np.array([0.0, 1.0, 0.0])
    np.testing.assert_allclose(F, expected)
    assert float(np.dot(F, v)) < 0.0
    np.testing.assert_allclose(aerodynamic_force(rho, np.zeros(3), cd, area), 0.0)
    np.testing.assert_allclose(aerodynamic_force(0.0, v, cd, area), 0.0)


def test_aero_torque_zero_when_cp_along_force():
    v = np.array([7000.0, 0.0, 0.0])
    rho, cd, area = 2.5e-12, 2.2, 1.0
    F = aerodynamic_force(rho, v, cd, area)
    # r_cp parallel to F (anti-ram) and the origin.
    for r_cp in (np.zeros(3), F, -3.0 * F, np.array([-0.2, 0.0, 0.0])):
        tau = aerodynamic_torque(r_cp, rho, v, cd, area)
        np.testing.assert_allclose(tau, 0.0, atol=1e-18)
    # Offset cp perpendicular to ram → torque about z.
    r_cp = np.array([0.0, 0.0, 0.15])
    tau = aerodynamic_torque(r_cp, rho, v, cd, area)
    np.testing.assert_allclose(tau, np.cross(r_cp, F))
    np.testing.assert_allclose(tau, [0.0, 0.15 * F[0], 0.0])
    assert abs(float(np.dot(tau, F))) < 1e-18
    assert abs(float(np.dot(tau, r_cp))) < 1e-18


def test_aero_project_area_zero_on_leeward_face():
    v = np.array([0.0, 0.0, 8000.0])
    n_lee = np.array([0.0, 0.0, -1.0])
    F = aerodynamic_force(1e-11, v, 2.2, 0.5, n_lee, project_area=True)
    np.testing.assert_allclose(F, 0.0)
    n_ram = np.array([0.0, 0.0, 1.0])
    F_ram = aerodynamic_force(1e-11, v, 2.2, 0.5, n_ram, project_area=True)
    assert float(np.linalg.norm(F_ram)) > 0.0


def test_aero_tau_body_ram_matches_primitive():
    state = _equatorial_state(0.0)  # v = (0, n R, 0)
    r_cp = np.array([0.05, 0.0, 0.02])
    area, cd = 0.3, 2.2
    aero = AerodynamicTorque(r_cp, area, cd=cd)
    tau = aero.tau_body(_Q_ID, _W0, orbit=state)
    rho = ExponentialAtmosphere().density(state.radius)
    expected = aerodynamic_torque(r_cp, rho, state.v_eci, cd, area)
    np.testing.assert_allclose(tau, expected, rtol=1e-12)
    # Independent of ω (surface velocity neglected).
    tau_w = aero.tau_body(_Q_ID, np.array([1.0, -0.5, 0.2]), orbit=state)
    np.testing.assert_allclose(tau, tau_w)
    # Co-rotating atmosphere changes v_rel = v − ω_E × r.
    aero_rot = AerodynamicTorque(r_cp, area, cd=cd, co_rotating=True)
    tau_rot = aero_rot.tau_body(_Q_ID, _W0, orbit=state)
    v_rel = relative_velocity_eci(state, co_rotating=True)
    expected_rot = aerodynamic_torque(r_cp, rho, v_rel, cd, area)
    np.testing.assert_allclose(tau_rot, expected_rot, rtol=1e-12)
    rel = float(np.linalg.norm(tau_rot - tau) / np.linalg.norm(tau))
    assert rel > 0.05


def test_srp_force_cannonball_and_backface_zero():
    u_sun = np.array([1.0, 0.0, 0.0])
    area, cr = 0.8, 1.5
    F = srp_force(area, cr, u_sun)
    np.testing.assert_allclose(F, P_SRP_1AU * cr * area * u_sun)
    n_back = np.array([-1.0, 0.0, 0.0])
    np.testing.assert_allclose(srp_force(area, cr, u_sun, n_back), 0.0)
    n_face = np.array([1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0), 0.0])
    cos_theta = 1.0 / np.sqrt(2.0)
    F_panel = srp_force(area, cr, u_sun, n_face)
    np.testing.assert_allclose(F_panel, P_SRP_1AU * cr * area * cos_theta * u_sun)


def test_srp_torque_zero_when_cp_along_force_or_eclipse():
    u_sun = np.array([0.0, 1.0, 0.0])
    area, cr = 1.0, 1.0
    F = srp_force(area, cr, u_sun)
    for r_cp in (np.zeros(3), F, np.array([0.0, 0.4, 0.0])):
        np.testing.assert_allclose(srp_torque(r_cp, area, cr, u_sun), 0.0, atol=1e-18)
    r_cp = np.array([0.1, 0.0, 0.0])
    tau = srp_torque(r_cp, area, cr, u_sun)
    np.testing.assert_allclose(tau, np.cross(r_cp, F))
    np.testing.assert_allclose(srp_torque(r_cp, area, cr, u_sun, eclipse=True), 0.0)
    np.testing.assert_allclose(srp_force(area, cr, u_sun, eclipse=True), 0.0)


def test_cylindrical_umbra_on_off():
    r_sunward = np.array([_R, 0.0, 0.0])
    sun = np.array([1.0, 0.0, 0.0])
    assert not in_cylindrical_umbra(r_sunward, sun)
    # Anti-sun of Earth, on the sun line → umbra.
    assert in_cylindrical_umbra(r_sunward, -sun)
    # Anti-sun but outside the Earth cylinder.
    r_miss = np.array([_R, 2.0 * R_EARTH, 0.0])
    assert not in_cylindrical_umbra(r_miss, -sun)


def test_srp_tau_body_eclipse_flag_and_cylindrical():
    r_cp = np.array([0.0, 0.0, 0.12])
    area, cr = 0.5, 1.8
    sun = np.array([1.0, 0.0, 0.0])
    srp = SolarRadiationPressureTorque(r_cp, area, cr=cr, sun_eci=sun)
    tau = srp.tau_body(_Q_ID, _W0)
    expected = srp_torque(r_cp, area, cr, sun, p_srp=P_SRP_1AU)
    np.testing.assert_allclose(tau, expected)
    srp_on = SolarRadiationPressureTorque(
        r_cp, area, cr=cr, sun_eci=sun, eclipse=True
    )
    np.testing.assert_allclose(srp_on.tau_body(_Q_ID, np.array([0.2, 0.0, 0.0])), 0.0)
    srp_umbra = SolarRadiationPressureTorque(
        r_cp, area, cr=cr, sun_eci=sun, eclipse="on"
    )
    np.testing.assert_allclose(srp_umbra.tau_body(_Q_ID, _W0), 0.0)
    # Equatorial +x: sun along +x is sunlit; sun along −x is cylindrical umbra.
    state = _equatorial_state(0.0)
    srp_cyl_sunlit = SolarRadiationPressureTorque(
        r_cp, area, cr=cr, sun_eci=sun, eclipse="cylindrical"
    )
    assert not srp_cyl_sunlit.in_umbra(state)
    tau_sunlit = srp_cyl_sunlit.tau_body(_Q_ID, _W0, orbit=state)
    np.testing.assert_allclose(tau_sunlit, expected)
    srp_cyl_shadow = SolarRadiationPressureTorque(
        r_cp, area, cr=cr, sun_eci=-sun, eclipse="cylindrical"
    )
    assert srp_cyl_shadow.in_umbra(state)
    np.testing.assert_allclose(srp_cyl_shadow.tau_body(_Q_ID, _W0, orbit=state), 0.0)
    with pytest.raises(ValueError, match="eclipse"):
        SolarRadiationPressureTorque(r_cp, area, eclipse="penumbra")


def test_environmental_sum_includes_aero_and_srp_without_breaking_gg_dipole():
    J = np.diag([0.05, 0.06, 0.07])
    circ = CircularOrbit(radius=_R, inclination=np.deg2rad(30.0))
    q = quat_normalize([0.6, 0.1, -0.2, 0.7])
    omega = np.array([0.01, -0.02, 0.03])
    t = 25.0
    gg = GravityGradientTorque(J, orbit=circ)
    mag = ResidualDipoleTorque([0.08, -0.01, 0.02], orbit=circ, model="tilted")
    aero = AerodynamicTorque([0.04, -0.01, 0.02], 0.25, cd=2.2, orbit=circ)
    srp = SolarRadiationPressureTorque(
        [0.0, 0.05, 0.0], 0.4, cr=1.2, orbit=circ, eclipse=False
    )
    env = EnvironmentalTorques(
        gravity_gradient=gg, residual_dipole=mag, aerodynamic=aero, srp=srp
    )
    tau = env.tau_body(q, omega, t)
    parts = (
        gg.tau_body(q, omega, t)
        + mag.tau_body(q, omega, t)
        + aero.tau_body(q, omega, t)
        + srp.tau_body(q, omega, t)
    )
    np.testing.assert_allclose(tau, parts)
    # GG + dipole only still matches (aero/SRP left None).
    env_old = EnvironmentalTorques(gravity_gradient=gg, residual_dipole=mag)
    np.testing.assert_allclose(
        env_old.tau_body(q, omega, t),
        gg.tau_body(q, omega, t) + mag.tau_body(q, omega, t),
    )
    # Eclipse stub zeroes only SRP in the sum.
    srp_off = SolarRadiationPressureTorque(
        [0.0, 0.05, 0.0], 0.4, cr=1.2, orbit=circ, eclipse=True
    )
    env_ecl = EnvironmentalTorques(
        gravity_gradient=gg, residual_dipole=mag, aerodynamic=aero, srp=srp_off
    )
    np.testing.assert_allclose(
        env_ecl.tau_body(q, omega, t),
        gg.tau_body(q, omega, t)
        + mag.tau_body(q, omega, t)
        + aero.tau_body(q, omega, t),
    )

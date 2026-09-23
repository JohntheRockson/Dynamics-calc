import type { SimRequest } from './types'

// Mirrors `SimRequest::default()` in `engine/src/sim.rs` field for field.
export function defaultSimRequest(): SimRequest {
  return {
    scenario: 'slew',
    controller: null,
    estimator: 'mekf',
    t_final: null,
    dt: 0.01,
    angle_deg: null,
    tau_dist: [0, 0, 0],

    inertia: [0.05, 0.06, 0.07],

    gain_scale: 1.0,
    torque_limit: 0.02,
    wn: 0.5,
    zeta: 1.0,

    gravity_gradient: false,
    residual_dipole: false,
    aerodynamic: false,
    srp: false,
    env: false,
    no_env: false,
    orbit_radius: 7.0e6,
    orbit_inc_deg: null,
    orbit_raan_deg: 0,
    dipole_m: null,
    dipole_model: 'tilted',
    panel_area: 0.4,
    panel_rcp: [0.05, 0, 0.02],
    aero_cd: 2.2,
    srp_cr: 1.0,
    srp_eclipse: 'off',

    actuator_tau_max: null,
    actuator_tau: null,
    rw_h_max: null,
    rw_inertia: null,
    rw_visc: 0,
    rw_coulomb: 0,
    rw_no_gyro: false,
    actuator_h_dump: null,
    actuator_dump_gain: 1.0,

    use_mag: true,
    use_sun: true,
    sun_eclipse: false,
    gyro_sigma_v: 5e-4,
    gyro_sigma_u: 1e-6,
    mag_sigma: 3e-3,
    sun_sigma: 2e-3,
    coarse_init: false,
    seed: 1,
  }
}

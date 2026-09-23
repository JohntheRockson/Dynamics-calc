// Mirrors the JSON wire format produced by `engine::sim` (SimRequest /
// SimLog) and served by `attitude-server`. Field names intentionally match
// the Rust struct fields (snake_case) rather than being camelCased, so the
// request/response shapes are self-documenting against the backend code.

export type Scenario = 'slew' | 'detumble' | 'hold' | 'eigenaxis'
export type ControllerKind = 'pid' | 'lqr'
export type EstimatorKind = 'truth' | 'mekf' | 'mahony'
export type DipoleModel = 'tilted' | 'orbit_normal'
export type SrpEclipseMode = 'off' | 'on' | 'cylindrical'

/** Rust `AxisLimit`: a scalar applied to all three axes, or a per-axis vector. */
export type AxisLimit = number | [number, number, number]

export interface Vec3Tuple extends Array<number> {
  0: number
  1: number
  2: number
}

export interface SimRequest {
  scenario: Scenario
  controller: ControllerKind | null
  estimator: EstimatorKind
  t_final: number | null
  dt: number
  angle_deg: number | null
  tau_dist: [number, number, number]

  inertia: [number, number, number]

  gain_scale: number
  torque_limit: number | null
  wn: number
  zeta: number

  gravity_gradient: boolean
  residual_dipole: boolean
  aerodynamic: boolean
  srp: boolean
  env: boolean
  no_env: boolean
  orbit_radius: number
  orbit_inc_deg: number | null
  orbit_raan_deg: number
  dipole_m: [number, number, number] | null
  dipole_model: DipoleModel
  panel_area: number
  panel_rcp: [number, number, number]
  aero_cd: number
  srp_cr: number
  srp_eclipse: SrpEclipseMode

  actuator_tau_max: AxisLimit | null
  actuator_tau: number | null
  rw_h_max: AxisLimit | null
  rw_inertia: AxisLimit | null
  rw_visc: number
  rw_coulomb: number
  rw_no_gyro: boolean
  actuator_h_dump: AxisLimit | null
  actuator_dump_gain: number

  use_mag: boolean
  use_sun: boolean
  sun_eclipse: boolean
  gyro_sigma_v: number
  gyro_sigma_u: number
  mag_sigma: number
  sun_sigma: number
  coarse_init: boolean
  seed: number
}

export interface ScenarioInfo {
  key: Scenario
  title: string
  blurb: string
  default_t_final: number
  default_controller: ControllerKind
  env_by_default: boolean
  default_angle_deg: number | null
}

export interface NisSample {
  t: number
  sensor: string
  nis: number
}

export interface SimSummary {
  final_att_error_deg: number
  peak_rate: number
  peak_tau: number
  sat_fraction: number | null
  mean_nis: number | null
}

export interface ResolvedEcho {
  scenario: Scenario
  controller: ControllerKind
  estimator: EstimatorKind
  t_final: number
  dt: number
  angle_deg: number
  n_samples: number
  env_active: boolean
  reaction_wheels_active: boolean
}

export interface SimLog {
  t: number[]
  q: [number, number, number, number][]
  omega: [number, number, number][]
  tau: [number, number, number][]
  q_hat: [number, number, number, number][]
  omega_hat: [number, number, number][]
  q_des: [number, number, number, number]
  euler_deg: [number, number, number][]
  euler_des_deg: [number, number, number]
  att_error_deg: number[]
  att_error_hat_deg: number[] | null
  est_att_error_deg: number[] | null
  tau_env: [number, number, number][]
  h_wheel: [number, number, number][]
  tau_ext: [number, number, number][]
  omega_wheel: [number, number, number][] | null
  rw_tau_sat: [boolean, boolean, boolean][] | null
  rw_h_sat: [boolean, boolean, boolean][] | null
  nis: NisSample[]
  summary: SimSummary
  resolved: ResolvedEcho
  warnings: string[]
}

export function axisLimitToScalarOrVec(limit: AxisLimit | null): { scalar: number; vec: [number, number, number] } {
  if (limit === null) return { scalar: 0, vec: [0, 0, 0] }
  if (typeof limit === 'number') return { scalar: limit, vec: [limit, limit, limit] }
  return { scalar: limit[0], vec: limit }
}

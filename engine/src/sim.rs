//! SimLab orchestration: resolve a wire-format [`SimRequest`] into a fully
//! specified configuration, then run the closed-loop
//! sensors -> estimator -> controller -> actuator -> plant
//! loop exactly as `attitude_sim.sim.run_slew` does, and return a
//! JSON-serializable [`SimLog`]. This is the only module that knows about
//! the HTTP/JSON wire format; everything else in the crate is a plain
//! numerical library.

use crate::actuators::{AxisLimit, TorqueActuator, DEFAULT_DUMP_GAIN};
use crate::controls::{
    bryson_lqr_costs, default_cubesat_inertia, Controller, LqrAttitudeController,
    PidAttitudeController, DEFAULT_TORQUE_LIMIT, LQR_OMEGA_REF, LQR_THETA_REF, PID_KI_WN_COEFF,
    PID_WN, PID_ZETA,
};
use crate::disturbances::{
    AeroParams, CircularOrbit, DipoleModel, EnvironmentalTorques, SrpEclipse, SrpParams,
};
use crate::estimation::{
    try_triad_q0_from_sensors, vectors_from_sensors, ComplementaryFilter, Estimator,
    MultiplicativeEkf,
};
use crate::plant::{step_rigid_body, PlantError, RigidBody};
use crate::quaternion::{Mat3, Quat, Vec3};
use crate::reaction_wheels::{make_reaction_wheels, ReactionWheelAssembly};
use crate::scenarios::{scenario_state, Scenario};
use crate::sensors::{magnetometer, make_rng, sun_sensor, GyroModel, VectorSensor};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Hard cap on samples per run. The Rust computation itself stays sub-100ms
/// even at 10x this (see `engine`'s benches-by-hand in the README), so this
/// is really a payload-size budget: at ~30 logged f64 channels/sample,
/// 8,000 samples is a few MB of JSON (well under a second to fetch, parse,
/// and hand to the charts/3D view -- the actual bottleneck for "feels
/// instant"), while every named scenario's default duration (20-40s at the
/// default 10ms step, i.e. <=4,001 samples) stays comfortably inside it.
pub const MAX_SAMPLES: usize = 8_000;

const DEFAULT_INERTIA: [f64; 3] = [0.05, 0.06, 0.07];
const DEFAULT_ORBIT_RADIUS: f64 = 7.0e6;
const DEFAULT_DIPOLE_M: [f64; 3] = [0.10, 0.0, 0.0];
const DEFAULT_PANEL_AREA: f64 = 0.4;
const DEFAULT_PANEL_RCP: [f64; 3] = [0.05, 0.0, 0.02];
const DEFAULT_AERO_CD: f64 = 2.2;
const DEFAULT_SRP_CR: f64 = 1.0;
const GYRO_BIAS_DEFAULT: [f64; 3] = [0.002, -0.001, 0.0015];
const HOLD_ORBIT_INC_DEG: f64 = 51.6;

#[derive(Debug, Error)]
pub enum SimError {
    #[error("dt must be positive")]
    InvalidDt,
    #[error("t_final must be non-negative")]
    InvalidTFinal,
    #[error("requested run needs {0} samples, which exceeds the server limit of {MAX_SAMPLES} (increase dt or shorten t_final)")]
    TooManySamples(usize),
    #[error("invalid inertia: {0}")]
    InvalidInertia(#[from] PlantError),
    #[error("unknown controller {0:?}; use 'pid' or 'lqr'")]
    InvalidController(String),
    #[error("unknown estimator {0:?}; use 'truth', 'mekf', or 'mahony'")]
    InvalidEstimator(String),
    #[error("unknown dipole model {0:?}; use 'tilted' or 'orbit_normal'")]
    InvalidDipoleModel(String),
    #[error("unknown srp eclipse mode {0:?}; use 'off', 'on', or 'cylindrical'")]
    InvalidSrpEclipse(String),
}

/// Wire-format simulation request. Every field has a default matching the
/// Python CLI's default so `{"scenario": "slew"}` reproduces
/// `python -m attitude_sim --scenario slew --no-gif`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SimRequest {
    pub scenario: Scenario,
    pub controller: Option<String>,
    pub estimator: String,
    pub t_final: Option<f64>,
    pub dt: f64,
    pub angle_deg: Option<f64>,
    pub tau_dist: [f64; 3],

    pub inertia: [f64; 3],

    pub gain_scale: f64,
    pub torque_limit: Option<f64>,
    pub wn: f64,
    pub zeta: f64,

    pub gravity_gradient: bool,
    pub residual_dipole: bool,
    pub aerodynamic: bool,
    pub srp: bool,
    pub env: bool,
    pub no_env: bool,
    pub orbit_radius: f64,
    pub orbit_inc_deg: Option<f64>,
    pub orbit_raan_deg: f64,
    pub dipole_m: Option<[f64; 3]>,
    pub dipole_model: String,
    pub panel_area: f64,
    pub panel_rcp: [f64; 3],
    pub aero_cd: f64,
    pub srp_cr: f64,
    pub srp_eclipse: String,

    pub actuator_tau_max: Option<AxisLimit>,
    pub actuator_tau: Option<f64>,
    pub rw_h_max: Option<AxisLimit>,
    pub rw_inertia: Option<AxisLimit>,
    pub rw_visc: f64,
    pub rw_coulomb: f64,
    pub rw_no_gyro: bool,
    pub actuator_h_dump: Option<AxisLimit>,
    pub actuator_dump_gain: f64,

    pub use_mag: bool,
    pub use_sun: bool,
    pub sun_eclipse: bool,
    pub gyro_sigma_v: f64,
    pub gyro_sigma_u: f64,
    pub mag_sigma: f64,
    pub sun_sigma: f64,
    pub coarse_init: bool,
    pub seed: u64,
}

impl Default for SimRequest {
    fn default() -> Self {
        SimRequest {
            scenario: Scenario::Slew,
            controller: None,
            estimator: "mekf".into(),
            t_final: None,
            dt: 0.01,
            angle_deg: None,
            tau_dist: [0.0, 0.0, 0.0],
            inertia: DEFAULT_INERTIA,
            gain_scale: 1.0,
            torque_limit: Some(DEFAULT_TORQUE_LIMIT),
            wn: PID_WN,
            zeta: PID_ZETA,
            gravity_gradient: false,
            residual_dipole: false,
            aerodynamic: false,
            srp: false,
            env: false,
            no_env: false,
            orbit_radius: DEFAULT_ORBIT_RADIUS,
            orbit_inc_deg: None,
            orbit_raan_deg: 0.0,
            dipole_m: None,
            dipole_model: "tilted".into(),
            panel_area: DEFAULT_PANEL_AREA,
            panel_rcp: DEFAULT_PANEL_RCP,
            aero_cd: DEFAULT_AERO_CD,
            srp_cr: DEFAULT_SRP_CR,
            srp_eclipse: "off".into(),
            actuator_tau_max: None,
            actuator_tau: None,
            rw_h_max: None,
            rw_inertia: None,
            rw_visc: 0.0,
            rw_coulomb: 0.0,
            rw_no_gyro: false,
            actuator_h_dump: None,
            actuator_dump_gain: DEFAULT_DUMP_GAIN,
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
}

/// Fully-resolved configuration (Rust analogue of Python's `SimConfig`,
/// after `make_scenario_config`'s default-resolution logic has run).
struct ResolvedConfig {
    scenario: Scenario,
    controller: String,
    estimator: String,
    dt: f64,
    t_final: f64,
    angle_deg: f64,
    q0: Quat,
    omega0: Vec3,
    q_des: Quat,
    inertia: Mat3,
    tau_dist: Vec3,
    gain_scale: f64,
    torque_limit: Option<f64>,
    wn: f64,
    zeta: f64,
    gravity_gradient: bool,
    residual_dipole: bool,
    aerodynamic: bool,
    srp: bool,
    orbit_radius: f64,
    orbit_inc_deg: f64,
    orbit_raan_deg: f64,
    dipole_m: Vec3,
    dipole_model: DipoleModel,
    panel_area: f64,
    panel_rcp: Vec3,
    aero_cd: f64,
    srp_cr: f64,
    srp_eclipse: SrpEclipse,
    actuator_tau_max: Option<AxisLimit>,
    actuator_tau: Option<f64>,
    rw_h_max: Option<AxisLimit>,
    rw_inertia: Option<AxisLimit>,
    rw_visc: f64,
    rw_coulomb: f64,
    rw_gyroscopic: bool,
    actuator_h_dump: Option<AxisLimit>,
    actuator_dump_gain: f64,
    use_mag: bool,
    use_sun: bool,
    sun_eclipse: bool,
    gyro_sigma_v: f64,
    gyro_sigma_u: f64,
    mag_sigma: f64,
    sun_sigma: f64,
    coarse_init: bool,
    seed: u64,
}

fn resolve(req: &SimRequest) -> Result<ResolvedConfig, SimError> {
    let scenario = req.scenario;
    let angle_deg = scenario.default_angle_deg(req.angle_deg);
    let (q0, omega0, q_des) = scenario_state(scenario, req.angle_deg);

    let controller = req
        .controller
        .as_deref()
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.to_lowercase())
        .unwrap_or_else(|| scenario.default_controller().to_string());
    if controller != "pid" && controller != "lqr" {
        return Err(SimError::InvalidController(controller));
    }

    let estimator = req.estimator.to_lowercase();
    if !["truth", "mekf", "mahony"].contains(&estimator.as_str()) {
        return Err(SimError::InvalidEstimator(estimator));
    }

    let t_final = req.t_final.unwrap_or_else(|| scenario.default_t_final());
    if req.dt <= 0.0 || !req.dt.is_finite() {
        return Err(SimError::InvalidDt);
    }
    if t_final < 0.0 || !t_final.is_finite() {
        return Err(SimError::InvalidTFinal);
    }

    let want_env = if req.no_env {
        false
    } else if req.env {
        true
    } else {
        scenario.uses_env_by_default()
    };
    let mut gravity_gradient = req.gravity_gradient || want_env;
    let mut residual_dipole = req.residual_dipole || want_env;
    if req.no_env {
        gravity_gradient = false;
        residual_dipole = false;
    }

    // Precedence (matches `make_scenario_config`): explicit request field >
    // hold/env preset default > global default.
    let orbit_inc_deg =
        req.orbit_inc_deg
            .unwrap_or(if want_env { HOLD_ORBIT_INC_DEG } else { 0.0 });
    let dipole_m = req.dipole_m.unwrap_or(if want_env {
        [40.0, -15.0, 8.0]
    } else {
        DEFAULT_DIPOLE_M
    });

    let dipole_model = match req.dipole_model.as_str() {
        "tilted" => DipoleModel::Tilted,
        "orbit_normal" => DipoleModel::OrbitNormal,
        other => return Err(SimError::InvalidDipoleModel(other.to_string())),
    };
    let srp_eclipse = match req.srp_eclipse.as_str() {
        "off" => SrpEclipse::Off,
        "on" => SrpEclipse::On,
        "cylindrical" => SrpEclipse::Cylindrical,
        other => return Err(SimError::InvalidSrpEclipse(other.to_string())),
    };

    let inertia = Mat3::from_diagonal(&Vec3::new(req.inertia[0], req.inertia[1], req.inertia[2]));

    Ok(ResolvedConfig {
        scenario,
        controller,
        estimator,
        dt: req.dt,
        t_final,
        angle_deg,
        q0,
        omega0,
        q_des,
        inertia,
        tau_dist: Vec3::new(req.tau_dist[0], req.tau_dist[1], req.tau_dist[2]),
        gain_scale: req.gain_scale,
        torque_limit: req.torque_limit,
        wn: req.wn,
        zeta: req.zeta,
        gravity_gradient,
        residual_dipole,
        aerodynamic: req.aerodynamic,
        srp: req.srp,
        orbit_radius: req.orbit_radius,
        orbit_inc_deg,
        orbit_raan_deg: req.orbit_raan_deg,
        dipole_m: Vec3::new(dipole_m[0], dipole_m[1], dipole_m[2]),
        dipole_model,
        panel_area: req.panel_area,
        panel_rcp: Vec3::new(req.panel_rcp[0], req.panel_rcp[1], req.panel_rcp[2]),
        aero_cd: req.aero_cd,
        srp_cr: req.srp_cr,
        srp_eclipse,
        actuator_tau_max: req.actuator_tau_max,
        actuator_tau: req.actuator_tau,
        rw_h_max: req.rw_h_max,
        rw_inertia: req.rw_inertia,
        rw_visc: req.rw_visc,
        rw_coulomb: req.rw_coulomb,
        rw_gyroscopic: !req.rw_no_gyro,
        actuator_h_dump: req.actuator_h_dump,
        actuator_dump_gain: req.actuator_dump_gain,
        use_mag: req.use_mag,
        use_sun: req.use_sun,
        sun_eclipse: req.sun_eclipse,
        gyro_sigma_v: req.gyro_sigma_v,
        gyro_sigma_u: req.gyro_sigma_u,
        mag_sigma: req.mag_sigma,
        sun_sigma: req.sun_sigma,
        coarse_init: req.coarse_init,
        seed: req.seed,
    })
}

/// Either actuator model, unified so the sim loop doesn't care which one it holds.
enum ActuatorStage {
    Simple(TorqueActuator),
    ReactionWheel(ReactionWheelAssembly),
}

impl ActuatorStage {
    fn apply(&mut self, cmd: &Vec3, dt: f64, omega: &Vec3) -> Vec3 {
        match self {
            ActuatorStage::Simple(a) => a.apply(cmd, dt),
            ActuatorStage::ReactionWheel(rw) => rw.apply(cmd, dt, omega),
        }
    }
    fn momentum(&self) -> Vec3 {
        match self {
            ActuatorStage::Simple(a) => a.momentum(),
            ActuatorStage::ReactionWheel(rw) => rw.momentum(),
        }
    }
    fn external_torque(&self) -> Vec3 {
        match self {
            ActuatorStage::Simple(a) => a.external_torque(),
            ActuatorStage::ReactionWheel(_) => Vec3::zeros(),
        }
    }
    fn as_reaction_wheel(&self) -> Option<&ReactionWheelAssembly> {
        match self {
            ActuatorStage::ReactionWheel(rw) => Some(rw),
            _ => None,
        }
    }
}

fn build_actuator(cfg: &ResolvedConfig) -> ActuatorStage {
    let use_rw = cfg.rw_inertia.is_some()
        || cfg.rw_h_max.is_some()
        || cfg.rw_visc != 0.0
        || cfg.rw_coulomb != 0.0;
    if use_rw {
        ActuatorStage::ReactionWheel(make_reaction_wheels(
            cfg.actuator_tau_max,
            cfg.rw_h_max,
            cfg.rw_inertia,
            cfg.rw_visc,
            cfg.rw_coulomb,
            cfg.actuator_tau,
            cfg.rw_gyroscopic,
        ))
    } else {
        ActuatorStage::Simple(TorqueActuator::new(
            cfg.actuator_tau_max,
            cfg.actuator_tau,
            cfg.actuator_h_dump,
            cfg.actuator_dump_gain,
        ))
    }
}

fn build_controller(cfg: &ResolvedConfig) -> Controller {
    let tau_for_tune = cfg.torque_limit.unwrap_or(DEFAULT_TORQUE_LIMIT);
    if cfg.controller == "lqr" {
        let (q, r) = bryson_lqr_costs(LQR_THETA_REF, LQR_OMEGA_REF, tau_for_tune);
        Controller::Lqr(LqrAttitudeController::new(
            cfg.inertia,
            &q,
            &r,
            cfg.gain_scale,
            cfg.torque_limit,
        ))
    } else {
        Controller::Pid(PidAttitudeController::new(
            cfg.inertia,
            cfg.wn,
            cfg.zeta,
            PID_KI_WN_COEFF,
            cfg.gain_scale,
            cfg.torque_limit,
        ))
    }
}

/// `None` when no environment model is requested (mirrors
/// `make_sim_disturbances` returning `None`). Note: the orbit here always
/// has `arg_latitude0 = 0`, matching the Python `run_slew` path exactly --
/// `scenarios::default_hold_orbit` (`arg_latitude0 = 40 deg`) is a separate
/// library convenience that the closed-loop CLI/app path does not call.
fn build_disturbances(cfg: &ResolvedConfig) -> Option<(EnvironmentalTorques, CircularOrbit)> {
    if !(cfg.gravity_gradient || cfg.residual_dipole || cfg.aerodynamic || cfg.srp) {
        return None;
    }
    let orbit = CircularOrbit::new(
        cfg.orbit_radius,
        cfg.orbit_inc_deg.to_radians(),
        cfg.orbit_raan_deg.to_radians(),
    );
    let mut env = EnvironmentalTorques::none();
    if cfg.gravity_gradient {
        env.gravity_gradient = Some(cfg.inertia);
    }
    if cfg.residual_dipole {
        env.residual_dipole = Some((cfg.dipole_m, cfg.dipole_model));
    }
    if cfg.aerodynamic {
        env.aerodynamic = Some(AeroParams {
            r_cp_body: cfg.panel_rcp,
            area: cfg.panel_area,
            cd: cfg.aero_cd,
        });
    }
    if cfg.srp {
        env.srp = Some(SrpParams {
            r_cp_body: cfg.panel_rcp,
            area: cfg.panel_area,
            cr: cfg.srp_cr,
            sun_eci: Vec3::new(1.0, 0.0, 0.0),
            eclipse: cfg.srp_eclipse,
        });
    }
    Some((env, orbit))
}

fn build_estimator(cfg: &ResolvedConfig, q0: Quat) -> Option<Estimator> {
    match cfg.estimator.as_str() {
        "truth" => None,
        "mekf" => Some(Estimator::Mekf(MultiplicativeEkf::new(
            q0,
            cfg.gyro_sigma_v,
            cfg.gyro_sigma_u,
        ))),
        "mahony" => Some(Estimator::Mahony(ComplementaryFilter::new(q0))),
        _ => unreachable!("validated in resolve()"),
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct NisSample {
    pub t: f64,
    pub sensor: String,
    pub nis: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct SimSummary {
    pub final_att_error_deg: f64,
    pub peak_rate: f64,
    pub peak_tau: f64,
    pub sat_fraction: Option<f64>,
    pub mean_nis: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResolvedEcho {
    pub scenario: String,
    pub controller: String,
    pub estimator: String,
    pub t_final: f64,
    pub dt: f64,
    pub angle_deg: f64,
    pub n_samples: usize,
    pub env_active: bool,
    pub reaction_wheels_active: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct SimLog {
    pub t: Vec<f64>,
    pub q: Vec<[f64; 4]>,
    pub omega: Vec<[f64; 3]>,
    pub tau: Vec<[f64; 3]>,
    pub q_hat: Vec<[f64; 4]>,
    pub omega_hat: Vec<[f64; 3]>,
    pub q_des: [f64; 4],
    pub euler_deg: Vec<[f64; 3]>,
    pub euler_des_deg: [f64; 3],
    pub att_error_deg: Vec<f64>,
    pub att_error_hat_deg: Option<Vec<f64>>,
    pub est_att_error_deg: Option<Vec<f64>>,
    pub tau_env: Vec<[f64; 3]>,
    pub h_wheel: Vec<[f64; 3]>,
    pub tau_ext: Vec<[f64; 3]>,
    pub omega_wheel: Option<Vec<[f64; 3]>>,
    pub rw_tau_sat: Option<Vec<[bool; 3]>>,
    pub rw_h_sat: Option<Vec<[bool; 3]>>,
    pub nis: Vec<NisSample>,
    pub summary: SimSummary,
    pub resolved: ResolvedEcho,
    pub warnings: Vec<String>,
}

fn quat_arr(q: &Quat) -> [f64; 4] {
    [q.w, q.x, q.y, q.z]
}
fn vec_arr(v: &Vec3) -> [f64; 3] {
    [v.x, v.y, v.z]
}

pub fn run_sim(req: &SimRequest) -> Result<SimLog, SimError> {
    let cfg = resolve(req)?;
    let body = RigidBody::new(cfg.inertia)?;
    let mut rng = make_rng(cfg.seed);

    let mut controller = build_controller(&cfg);
    controller.reset();
    let mut actuator = build_actuator(&cfg);

    let mut q = cfg.q0.normalize();
    let mut omega = cfg.omega0;
    let q_des = cfg.q_des.normalize();

    let disturbances = build_disturbances(&cfg);

    let mut warnings: Vec<String> = Vec::new();
    let want_estimator = cfg.estimator != "truth";

    let mut gyro: Option<GyroModel> = None;
    let mut sensors: Vec<VectorSensor> = Vec::new();
    let mut q_est0 = q;

    if want_estimator {
        gyro = Some(GyroModel::new(
            cfg.gyro_sigma_v,
            cfg.gyro_sigma_u,
            0.0,
            Vec3::new(
                GYRO_BIAS_DEFAULT[0],
                GYRO_BIAS_DEFAULT[1],
                GYRO_BIAS_DEFAULT[2],
            ),
        ));
        if cfg.use_mag {
            sensors.push(magnetometer(cfg.mag_sigma));
        }
        if cfg.use_sun {
            sensors.push(sun_sensor(cfg.sun_sigma, cfg.sun_eclipse));
        }
        if cfg.coarse_init {
            match try_triad_q0_from_sensors(&q, &sensors, &mut rng) {
                Some(qc) => q_est0 = qc,
                None => {
                    warnings.push(
                        "coarse TRIAD init skipped (fewer than two available vector sensors after FOV/eclipse gating); estimator starts at true q0".to_string(),
                    );
                }
            }
        }
    }

    let mut estimator = build_estimator(&cfg, q_est0);
    if want_estimator && sensors.is_empty() {
        warnings.push(format!(
            "estimator '{}' is running with no vector sensors (gyro-only); full attitude is not observable from rate alone",
            cfg.estimator
        ));
    }

    let n = (cfg.t_final / cfg.dt).round() as usize + 1;
    if n > MAX_SAMPLES {
        return Err(SimError::TooManySamples(n));
    }

    let mut t = vec![0.0; n];
    let mut q_hist = vec![Quat::IDENTITY; n];
    let mut w_hist = vec![Vec3::zeros(); n];
    let mut tau_hist = vec![Vec3::zeros(); n];
    let mut tau_env_hist = vec![Vec3::zeros(); n];
    let mut h_wheel_hist = vec![Vec3::zeros(); n];
    let mut tau_ext_hist = vec![Vec3::zeros(); n];
    let mut qh_hist = vec![Quat::IDENTITY; n];
    let mut wh_hist = vec![Vec3::zeros(); n];

    let is_rw = matches!(actuator, ActuatorStage::ReactionWheel(_));
    let mut ww_hist = if is_rw {
        vec![Vec3::zeros(); n]
    } else {
        Vec::new()
    };
    let mut tau_sat_hist = if is_rw {
        vec![[false; 3]; n]
    } else {
        Vec::new()
    };
    let mut h_sat_hist = if is_rw {
        vec![[false; 3]; n]
    } else {
        Vec::new()
    };

    for k in 0..n {
        t[k] = k as f64 * cfg.dt;
        q_hist[k] = q;
        w_hist[k] = omega;

        let (q_hat, omega_hat) = if !want_estimator {
            (q, omega)
        } else {
            let omega_m = gyro.as_mut().unwrap().measure(&omega, cfg.dt, &mut rng);
            let meas = if sensors.is_empty() {
                Vec::new()
            } else {
                vectors_from_sensors(&q, &sensors, &mut rng)
            };
            estimator
                .as_mut()
                .unwrap()
                .step(&omega_m, cfg.dt, &meas, t[k])
        };
        qh_hist[k] = q_hat;
        wh_hist[k] = omega_hat;

        let tau_cmd = controller.command(&q_hat, &omega_hat, &q_des, cfg.dt);
        let tau = actuator.apply(&tau_cmd, cfg.dt, &omega);
        tau_hist[k] = tau;
        h_wheel_hist[k] = actuator.momentum();
        tau_ext_hist[k] = actuator.external_torque();
        if let Some(rw) = actuator.as_reaction_wheel() {
            ww_hist[k] = rw.wheel_speed();
            tau_sat_hist[k] = rw.torque_saturated();
            h_sat_hist[k] = rw.momentum_saturated();
        }
        if let Some((env, orbit)) = &disturbances {
            let state = orbit.state_at(t[k]);
            tau_env_hist[k] = env.tau_body(&q, &state);
        }

        if k + 1 < n {
            let tau_total = tau + cfg.tau_dist + tau_env_hist[k] + tau_ext_hist[k];
            let (qn, on) = step_rigid_body(&body, &q, &omega, &tau_total, cfg.dt);
            q = qn;
            omega = on;
        }
    }

    let euler_deg: Vec<[f64; 3]> = q_hist
        .iter()
        .map(|qi| vec_arr(&(qi.to_euler_321().map(|v| v.to_degrees()))))
        .collect();
    let att_error_deg: Vec<f64> = q_hist
        .iter()
        .map(|qi| qi.geodesic_angle(&q_des).to_degrees())
        .collect();

    let (att_error_hat_deg, est_att_error_deg, mean_nis, nis) = if want_estimator {
        let att_hat: Vec<f64> = qh_hist
            .iter()
            .map(|qi| qi.geodesic_angle(&q_des).to_degrees())
            .collect();
        let est_err: Vec<f64> = (0..n)
            .map(|i| qh_hist[i].geodesic_angle(&q_hist[i]).to_degrees())
            .collect();
        let (mean_nis, nis_samples) = match &estimator {
            Some(Estimator::Mekf(ekf)) => (
                ekf.mean_nis(),
                ekf.innovations
                    .iter()
                    .map(|s| NisSample {
                        t: s.t,
                        sensor: s.sensor.to_string(),
                        nis: s.nis,
                    })
                    .collect(),
            ),
            _ => (None, Vec::new()),
        };
        (Some(att_hat), Some(est_err), mean_nis, nis_samples)
    } else {
        (None, None, None, Vec::new())
    };

    let final_att_error_deg = *att_error_deg.last().unwrap();
    let peak_rate = w_hist.iter().map(|w| w.norm()).fold(0.0_f64, f64::max);
    let peak_tau = tau_hist.iter().map(|w| w.norm()).fold(0.0_f64, f64::max);
    let sat_fraction = if is_rw {
        let hits = (0..n)
            .filter(|&k| tau_sat_hist[k].iter().any(|&b| b) || h_sat_hist[k].iter().any(|&b| b))
            .count();
        Some(hits as f64 / n as f64)
    } else {
        None
    };

    Ok(SimLog {
        q: q_hist.iter().map(quat_arr).collect(),
        omega: w_hist.iter().map(vec_arr).collect(),
        tau: tau_hist.iter().map(vec_arr).collect(),
        q_hat: qh_hist.iter().map(quat_arr).collect(),
        omega_hat: wh_hist.iter().map(vec_arr).collect(),
        q_des: quat_arr(&q_des),
        euler_deg,
        euler_des_deg: vec_arr(&q_des.to_euler_321().map(|v| v.to_degrees())),
        att_error_deg,
        att_error_hat_deg,
        est_att_error_deg,
        tau_env: tau_env_hist.iter().map(vec_arr).collect(),
        h_wheel: h_wheel_hist.iter().map(vec_arr).collect(),
        tau_ext: tau_ext_hist.iter().map(vec_arr).collect(),
        omega_wheel: if is_rw {
            Some(ww_hist.iter().map(vec_arr).collect())
        } else {
            None
        },
        rw_tau_sat: if is_rw { Some(tau_sat_hist) } else { None },
        rw_h_sat: if is_rw { Some(h_sat_hist) } else { None },
        nis,
        summary: SimSummary {
            final_att_error_deg,
            peak_rate,
            peak_tau,
            sat_fraction,
            mean_nis,
        },
        resolved: ResolvedEcho {
            scenario: cfg.scenario.key().to_string(),
            controller: cfg.controller.clone(),
            estimator: cfg.estimator.clone(),
            t_final: cfg.t_final,
            dt: cfg.dt,
            angle_deg: cfg.angle_deg,
            n_samples: n,
            env_active: disturbances.is_some(),
            reaction_wheels_active: is_rw,
        },
        warnings,
        t,
    })
}

/// Static catalog for populating a scenario picker (mirrors `scenario_catalog_text`).
#[derive(Debug, Clone, Serialize)]
pub struct ScenarioInfo {
    pub key: String,
    pub title: String,
    pub blurb: String,
    pub default_t_final: f64,
    pub default_controller: String,
    pub env_by_default: bool,
    pub default_angle_deg: Option<f64>,
}

pub fn scenario_catalog() -> Vec<ScenarioInfo> {
    Scenario::ALL
        .iter()
        .map(|s| ScenarioInfo {
            key: s.key().to_string(),
            title: s.title().to_string(),
            blurb: s.blurb().to_string(),
            default_t_final: s.default_t_final(),
            default_controller: s.default_controller().to_string(),
            env_by_default: s.uses_env_by_default(),
            default_angle_deg: match s {
                Scenario::Slew | Scenario::Eigenaxis => Some(s.default_angle_deg(None)),
                _ => None,
            },
        })
        .collect()
}

/// Convenience default-inertia accessor re-exported for callers/tests.
pub fn default_inertia() -> Mat3 {
    default_cubesat_inertia()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_slew_request_runs_and_settles() {
        let req = SimRequest {
            scenario: Scenario::Slew,
            estimator: "truth".into(),
            t_final: Some(5.0),
            ..Default::default()
        };
        let log = run_sim(&req).unwrap();
        assert_eq!(log.resolved.controller, "pid");
        assert!(log.t.len() > 100);
        assert!(log.att_error_deg[0] > 50.0); // starts near the commanded 75 deg
    }

    #[test]
    fn eigenaxis_defaults_to_lqr() {
        let req = SimRequest {
            scenario: Scenario::Eigenaxis,
            estimator: "truth".into(),
            t_final: Some(1.0),
            ..Default::default()
        };
        let log = run_sim(&req).unwrap();
        assert_eq!(log.resolved.controller, "lqr");
    }

    #[test]
    fn hold_scenario_enables_environment_by_default() {
        let req = SimRequest {
            scenario: Scenario::Hold,
            estimator: "truth".into(),
            t_final: Some(1.0),
            ..Default::default()
        };
        let log = run_sim(&req).unwrap();
        assert!(log.resolved.env_active);
        assert!(log.tau_env.iter().any(|t| t.iter().any(|v| v.abs() > 0.0)));
    }

    #[test]
    fn no_env_overrides_hold_default() {
        let req = SimRequest {
            scenario: Scenario::Hold,
            estimator: "truth".into(),
            t_final: Some(1.0),
            no_env: true,
            ..Default::default()
        };
        let log = run_sim(&req).unwrap();
        assert!(!log.resolved.env_active);
    }

    #[test]
    fn too_many_samples_is_rejected() {
        let req = SimRequest {
            dt: 1e-6,
            t_final: Some(10.0),
            ..Default::default()
        };
        assert!(matches!(run_sim(&req), Err(SimError::TooManySamples(_))));
    }

    #[test]
    fn reaction_wheels_activate_when_h_max_set() {
        let req = SimRequest {
            scenario: Scenario::Slew,
            estimator: "truth".into(),
            t_final: Some(1.0),
            rw_h_max: Some(AxisLimit::Scalar(0.01)),
            ..Default::default()
        };
        let log = run_sim(&req).unwrap();
        assert!(log.resolved.reaction_wheels_active);
        assert!(log.omega_wheel.is_some());
    }
}

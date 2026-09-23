//! Environmental disturbance torques (gravity-gradient, residual dipole,
//! aerodynamic, solar radiation pressure), ported from
//! `attitude_sim.disturbances`. Only the "bound circular orbit + time"
//! evaluation path used by SimLab is ported (the library also accepts an
//! explicit `OrbitState`, which the app never needs).

use crate::plant::{validate_inertia, PlantError};
use crate::quaternion::{Mat3, Quat, Vec3};

pub const MU_EARTH: f64 = 3.986004418e14;
pub const EARTH_MAG_MOMENT: f64 = 7.94e15;
pub const EARTH_DIPOLE_TILT_RAD: f64 = 0.20071286397934787; // 11.5 deg
pub const R_EARTH: f64 = 6.371e6;
pub const OMEGA_EARTH: f64 = 7.292115e-5;

pub const ATM_H_REF: f64 = 400e3;
pub const ATM_RHO_REF: f64 = 2.5e-12;
pub const ATM_SCALE_HEIGHT: f64 = 60e3;
pub const P_SRP_1AU: f64 = 4.56e-6;

const RADIUS_EPS: f64 = 1e-15;
const SPEED_EPS: f64 = 1e-15;

fn unit(v: &Vec3) -> Vec3 {
    let n = v.norm();
    if n < RADIUS_EPS {
        Vec3::zeros()
    } else {
        v / n
    }
}

/// Inertial position/velocity/normal at one instant along a circular orbit.
#[derive(Debug, Clone, Copy)]
pub struct OrbitState {
    pub r_eci: Vec3,
    pub v_eci: Vec3,
    pub mu: f64,
}

impl OrbitState {
    pub fn radius(&self) -> f64 {
        self.r_eci.norm()
    }

    pub fn r_hat(&self) -> Vec3 {
        self.r_eci / self.radius()
    }

    pub fn h_hat(&self) -> Vec3 {
        unit(&self.r_eci.cross(&self.v_eci))
    }

    pub fn mu_over_r3(&self) -> f64 {
        let r = self.radius();
        self.mu / (r * r * r)
    }
}

/// Keplerian circular orbit in ECI, parametrized by argument of latitude.
#[derive(Debug, Clone, Copy)]
pub struct CircularOrbit {
    pub radius: f64,
    pub inclination: f64,
    pub raan: f64,
    pub arg_latitude0: f64,
    pub mu: f64,
}

impl CircularOrbit {
    pub fn new(radius: f64, inclination: f64, raan: f64) -> Self {
        CircularOrbit {
            radius,
            inclination,
            raan,
            arg_latitude0: 0.0,
            mu: MU_EARTH,
        }
    }

    pub fn with_arg_latitude0(mut self, arg_latitude0: f64) -> Self {
        self.arg_latitude0 = arg_latitude0;
        self
    }

    pub fn mean_motion(&self) -> f64 {
        (self.mu / self.radius.powi(3)).sqrt()
    }

    pub fn state_at(&self, t: f64) -> OrbitState {
        let n = self.mean_motion();
        let u = self.arg_latitude0 + n * t;
        let (cu, su) = (u.cos(), u.sin());
        let (ci, si) = (self.inclination.cos(), self.inclination.sin());
        let (co, so) = (self.raan.cos(), self.raan.sin());
        let a = self.radius;
        let r_eci = a * Vec3::new(co * cu - so * su * ci, so * cu + co * su * ci, su * si);
        let v_eci = a * n * Vec3::new(-co * su - so * cu * ci, -so * su + co * cu * ci, cu * si);
        OrbitState {
            r_eci,
            v_eci,
            mu: self.mu,
        }
    }
}

/// Body-frame gravity-gradient torque `3 (mu/r^3) (r_hat_b x J r_hat_b)`.
pub fn gravity_gradient_torque(inertia: &Mat3, r_hat_body: &Vec3, mu_over_r3: f64) -> Vec3 {
    let r_hat = unit(r_hat_body);
    3.0 * mu_over_r3 * r_hat.cross(&(inertia * r_hat))
}

pub fn magnetic_dipole_torque(m_body: &Vec3, b_body: &Vec3) -> Vec3 {
    m_body.cross(b_body)
}

/// Isothermal exponential atmosphere `rho = rho_ref exp(-(h-h_ref)/H)`.
pub fn exponential_density(
    radius: f64,
    rho_ref: f64,
    h_ref: f64,
    scale_height: f64,
    earth_radius: f64,
) -> f64 {
    let altitude = radius - earth_radius;
    rho_ref * (-(altitude - h_ref) / scale_height).exp()
}

pub fn aerodynamic_force(rho: f64, v_rel: &Vec3, cd: f64, area: f64, n_hat: Option<Vec3>) -> Vec3 {
    let speed = v_rel.norm();
    if speed < SPEED_EPS {
        return Vec3::zeros();
    }
    let v_hat = v_rel / speed;
    let n = n_hat.map(|n| unit(&n)).unwrap_or(v_hat);
    -0.5 * rho * speed * speed * cd * area * n
}

pub fn aerodynamic_torque(
    r_cp: &Vec3,
    rho: f64,
    v_rel: &Vec3,
    cd: f64,
    area: f64,
    n_hat: Option<Vec3>,
) -> Vec3 {
    r_cp.cross(&aerodynamic_force(rho, v_rel, cd, area, n_hat))
}

pub fn in_cylindrical_umbra(r_eci: &Vec3, sun_eci: &Vec3, earth_radius: f64) -> bool {
    let sun = unit(sun_eci);
    if r_eci.dot(&sun) >= 0.0 {
        return false;
    }
    r_eci.cross(&sun).norm() < earth_radius
}

pub fn srp_force(
    area: f64,
    cr: f64,
    sun_hat: &Vec3,
    n_hat: Option<Vec3>,
    p_srp: f64,
    eclipse: bool,
) -> Vec3 {
    if eclipse {
        return Vec3::zeros();
    }
    let u_sun = unit(sun_hat);
    let cos_theta = match n_hat {
        None => 1.0,
        Some(n) => unit(&n).dot(&u_sun),
    };
    if cos_theta <= 0.0 {
        return Vec3::zeros();
    }
    p_srp * cr * area * cos_theta * u_sun
}

pub fn srp_torque(
    r_cp: &Vec3,
    area: f64,
    cr: f64,
    sun_hat: &Vec3,
    n_hat: Option<Vec3>,
    p_srp: f64,
    eclipse: bool,
) -> Vec3 {
    r_cp.cross(&srp_force(area, cr, sun_hat, n_hat, p_srp, eclipse))
}

pub fn earth_dipole_moment_eci(moment: f64, tilt_rad: f64, ra_rad: f64) -> Vec3 {
    let (st, ct) = (tilt_rad.sin(), tilt_rad.cos());
    moment * Vec3::new(st * ra_rad.cos(), st * ra_rad.sin(), -ct)
}

pub fn dipole_field_eci(r_eci: &Vec3, moment_eci: &Vec3) -> Vec3 {
    let radius = r_eci.norm();
    let r_hat = r_eci / radius;
    (3.0 * moment_eci.dot(&r_hat) * r_hat - moment_eci) / radius.powi(3)
}

pub fn orbit_normal_dipole_field_eci(orbit: &OrbitState, moment: f64, sign: f64) -> Vec3 {
    (sign * moment / orbit.radius().powi(3)) * orbit.h_hat()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DipoleModel {
    Tilted,
    OrbitNormal,
}

pub fn magnetic_field_eci(
    orbit: &OrbitState,
    model: DipoleModel,
    moment: f64,
    tilt_rad: f64,
    ra_rad: f64,
    sign: f64,
) -> Vec3 {
    match model {
        DipoleModel::Tilted => dipole_field_eci(
            &orbit.r_eci,
            &earth_dipole_moment_eci(moment, tilt_rad, ra_rad),
        ),
        DipoleModel::OrbitNormal => orbit_normal_dipole_field_eci(orbit, moment, sign),
    }
}

#[derive(Debug, Clone, Copy)]
pub enum SrpEclipse {
    Off,
    On,
    Cylindrical,
}

/// Sum of the enabled disturbance models, evaluated at one `(q, orbit-state)`.
#[derive(Debug, Clone)]
pub struct EnvironmentalTorques {
    pub gravity_gradient: Option<Mat3>,
    pub residual_dipole: Option<(Vec3, DipoleModel)>,
    pub aerodynamic: Option<AeroParams>,
    pub srp: Option<SrpParams>,
    pub dipole_moment: f64,
    pub dipole_tilt_rad: f64,
    pub dipole_ra_rad: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct AeroParams {
    pub r_cp_body: Vec3,
    pub area: f64,
    pub cd: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct SrpParams {
    pub r_cp_body: Vec3,
    pub area: f64,
    pub cr: f64,
    pub sun_eci: Vec3,
    pub eclipse: SrpEclipse,
}

impl EnvironmentalTorques {
    pub fn none() -> Self {
        EnvironmentalTorques {
            gravity_gradient: None,
            residual_dipole: None,
            aerodynamic: None,
            srp: None,
            dipole_moment: EARTH_MAG_MOMENT,
            dipole_tilt_rad: EARTH_DIPOLE_TILT_RAD,
            dipole_ra_rad: 0.0,
        }
    }

    pub fn is_active(&self) -> bool {
        self.gravity_gradient.is_some()
            || self.residual_dipole.is_some()
            || self.aerodynamic.is_some()
            || self.srp.is_some()
    }

    pub fn tau_body(&self, q: &Quat, state: &OrbitState) -> Vec3 {
        let mut tau = Vec3::zeros();
        let r_body_frame = |v_eci: &Vec3| q.to_rotation().transpose() * v_eci;

        if let Some(inertia) = &self.gravity_gradient {
            let r_hat_b = r_body_frame(&state.r_hat());
            tau += gravity_gradient_torque(inertia, &r_hat_b, state.mu_over_r3());
        }
        if let Some((m_body, model)) = &self.residual_dipole {
            let b_eci = magnetic_field_eci(
                state,
                *model,
                self.dipole_moment,
                self.dipole_tilt_rad,
                self.dipole_ra_rad,
                1.0,
            );
            let b_body = r_body_frame(&b_eci);
            tau += magnetic_dipole_torque(m_body, &b_body);
        }
        if let Some(p) = &self.aerodynamic {
            let rho = exponential_density(
                state.radius(),
                ATM_RHO_REF,
                ATM_H_REF,
                ATM_SCALE_HEIGHT,
                R_EARTH,
            );
            let v_rel_b = r_body_frame(&state.v_eci);
            tau += aerodynamic_torque(&p.r_cp_body, rho, &v_rel_b, p.cd, p.area, None);
        }
        if let Some(p) = &self.srp {
            let eclipsed = match p.eclipse {
                SrpEclipse::Off => false,
                SrpEclipse::On => true,
                SrpEclipse::Cylindrical => in_cylindrical_umbra(&state.r_eci, &p.sun_eci, R_EARTH),
            };
            let sun_b = r_body_frame(&p.sun_eci);
            tau += srp_torque(
                &p.r_cp_body,
                p.area,
                p.cr,
                &sun_b,
                None,
                P_SRP_1AU,
                eclipsed,
            );
        }
        tau
    }
}

pub fn gravity_gradient_inertia(inertia: &Mat3) -> Result<Mat3, PlantError> {
    validate_inertia(inertia)
}

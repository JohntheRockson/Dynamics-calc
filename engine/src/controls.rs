//! Attitude controllers: quaternion-error PID and linearized LQR.
//!
//! Direct port of `attitude_sim.controls`. The one deliberate numerical
//! deviation is [`solve_care`]: instead of a Hamiltonian-eigendecomposition
//! or LAPACK `care` solver (which would pull in a system BLAS/LAPACK
//! dependency), the continuous algebraic Riccati equation is solved by
//! integrating the associated matrix Riccati *differential* equation
//!
//! ```text
//! dP/dt = A^T P + P A + Q - P B R^-1 B^T P
//! ```
//!
//! forward with RK4 until it reaches steady state. This is a standard,
//! textbook-legitimate way to compute the stabilizing CARE solution when
//! `(A, B)` is stabilizable and `(A, sqrt(Q))` is detectable (true for every
//! attitude-LQR configuration this app can produce), it reuses the same RK4
//! machinery already trusted for the plant, and it needs no external solver.
//! `tests::care_residual_is_near_zero_for_cubesat_plant` checks the
//! resulting `P` against the CARE residual directly, and the Python
//! parity harness cross-checks the resulting gain `K` against
//! `scipy.linalg.solve_continuous_are`.

use crate::actuators::{clip_torque, AxisLimit};
use crate::quaternion::{Mat3, Quat, Vec3};
use nalgebra::{SMatrix, SVector};

pub type Mat6 = SMatrix<f64, 6, 6>;
pub type Mat6x3 = SMatrix<f64, 6, 3>;
pub type Mat3x6 = SMatrix<f64, 3, 6>;
pub type Vec6 = SVector<f64, 6>;

pub const DEFAULT_TORQUE_LIMIT: f64 = 0.02;
pub const CUBESAT_PRINCIPAL_INERTIA: [f64; 3] = [0.05, 0.06, 0.07];
pub const PID_WN: f64 = 0.5;
pub const PID_ZETA: f64 = 1.0;
pub const PID_KI_WN_COEFF: f64 = 0.5;
pub const PID_INTEGRAL_LIMIT: f64 = 3.0;
pub const PID_INTEGRAL_GATE: f64 = 0.10;
pub const PID_KAW: f64 = 0.0;
pub const LQR_THETA_REF: f64 = 0.25;
pub const LQR_OMEGA_REF: f64 = 0.20;
pub const LQR_TAU_REF: f64 = DEFAULT_TORQUE_LIMIT;

pub fn default_cubesat_inertia() -> Mat3 {
    Mat3::from_diagonal(&Vec3::new(
        CUBESAT_PRINCIPAL_INERTIA[0],
        CUBESAT_PRINCIPAL_INERTIA[1],
        CUBESAT_PRINCIPAL_INERTIA[2],
    ))
}

/// Inertia-scaled PID matrices for quaternion-vector error `e_q ~= theta/2`.
///
/// Matching a rotation-vector PD `tau = -wn^2 J theta - 2 zeta wn J omega`
/// requires `Kp = 2 wn^2 J`, `Kd = 2 zeta wn J`, `Ki = ki_wn_coeff wn^3 J`.
pub fn tune_pid_second_order(
    inertia: &Mat3,
    wn: f64,
    zeta: f64,
    ki_wn_coeff: f64,
) -> (Mat3, Mat3, Mat3) {
    let kp = 2.0 * wn * wn * inertia;
    let kd = 2.0 * zeta * wn * inertia;
    let ki = ki_wn_coeff * wn * wn * wn * inertia;
    (kp, kd, ki)
}

/// Bryson `(Q, R)` on `x = [delta_theta, omega]`, `u = tau`.
pub fn bryson_lqr_costs(theta_ref: f64, omega_ref: f64, tau_ref: f64) -> (Mat6, Mat3) {
    let q_att = 1.0 / (theta_ref * theta_ref);
    let q_rate = 1.0 / (omega_ref * omega_ref);
    let r_torque = 1.0 / (tau_ref * tau_ref);
    let mut q = Mat6::zeros();
    for i in 0..3 {
        q[(i, i)] = q_att;
        q[(i + 3, i + 3)] = q_rate;
    }
    let r = Mat3::from_diagonal(&Vec3::new(r_torque, r_torque, r_torque));
    (q, r)
}

/// `(A, B)` for `x = [delta_theta, omega]`, `u = tau` about rest.
pub fn linearize_attitude(inertia: &Mat3) -> (Mat6, Mat6x3) {
    let j_inv = inertia.try_inverse().expect("inertia must be invertible");
    let mut a = Mat6::zeros();
    for i in 0..3 {
        a[(i, i + 3)] = 1.0;
    }
    let mut b = Mat6x3::zeros();
    for i in 0..3 {
        for j in 0..3 {
            b[(i + 3, j)] = j_inv[(i, j)];
        }
    }
    (a, b)
}

/// CARE residual `A^T P + P A - P B R^-1 B^T P + Q` (zero at a solution).
pub fn care_residual(a: &Mat6, b: &Mat6x3, q: &Mat6, r: &Mat3, p: &Mat6) -> Mat6 {
    let r_inv = r.try_inverse().expect("R must be invertible");
    a.transpose() * p + p * a - p * b * r_inv * b.transpose() * p + q
}

/// Solve the CARE `A^T P + P A - P B R^-1 B^T P + Q = 0` for `P = P^T >= 0`
/// by integrating the associated differential Riccati equation to steady
/// state (see module docs for why).
pub fn solve_care(a: &Mat6, b: &Mat6x3, q: &Mat6, r: &Mat3) -> Mat6 {
    let r_inv = r
        .try_inverse()
        .expect("R must be symmetric positive definite");
    let bt = b.transpose();
    let br_inv_bt = b * r_inv * bt;
    let at = a.transpose();
    let deriv = |p: &Mat6| -> Mat6 { at * p + p * a + q - p * br_inv_bt * p };

    let mut p = *q;
    let dt = 0.02;
    let max_steps = 200_000usize;
    let tol = 1e-11 * (1.0 + q.norm());
    for _ in 0..max_steps {
        let k1 = deriv(&p);
        if k1.norm() < tol {
            break;
        }
        let k2 = deriv(&(p + 0.5 * dt * k1));
        let k3 = deriv(&(p + 0.5 * dt * k2));
        let k4 = deriv(&(p + dt * k3));
        p += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
    }
    0.5 * (p + p.transpose())
}

/// `K = R^-1 B^T P` from a CARE solution.
pub fn lqr_gain(b: &Mat6x3, r: &Mat3, p: &Mat6) -> Mat3x6 {
    let bt_p = b.transpose() * p;
    r.lu().solve(&bt_p).expect("R must be invertible")
}

/// Design the rest-linearized attitude LQR. Returns `(K, P, A, B)`.
pub fn design_attitude_lqr(inertia: &Mat3, q: &Mat6, r: &Mat3) -> (Mat3x6, Mat6, Mat6, Mat6x3) {
    let (a, b) = linearize_attitude(inertia);
    let p = solve_care(&a, &b, q, r);
    let k = lqr_gain(&b, r, &p);
    (k, p, a, b)
}

fn saturate(tau: &Vec3, limit: Option<f64>) -> Vec3 {
    match limit {
        Some(l) if l > 0.0 => {
            let n = tau.norm();
            if n > l {
                tau * (l / n)
            } else {
                *tau
            }
        }
        _ => *tau,
    }
}

/// Euclidean `|tau|` clamp, then per-axis box clamp (reaction-wheel limits).
pub fn apply_torque_limits(
    tau: &Vec3,
    torque_limit: Option<f64>,
    tau_max: Option<AxisLimit>,
) -> Vec3 {
    clip_torque(&saturate(tau, torque_limit), tau_max)
}

fn ki_is_active(ki: &Mat3) -> bool {
    ki.norm() > 1e-18
}

fn clamp_integral(z: &Vec3, limit: f64) -> Vec3 {
    let n = z.norm();
    if n > limit && limit > 0.0 {
        z * (limit / n)
    } else {
        *z
    }
}

fn backcalc_z_dot(ki: &Mat3, tau_unsat: &Vec3, tau_sat: &Vec3, kaw: f64) -> Vec3 {
    if kaw <= 0.0 || !ki_is_active(ki) {
        return Vec3::zeros();
    }
    let deficit = tau_unsat - tau_sat;
    match ki.lu().solve(&deficit) {
        Some(sol) => kaw * sol,
        None => Vec3::zeros(),
    }
}

/// PID on quaternion vector error + body rate, with gated conditional
/// anti-windup integration and optional gyroscopic cancellation.
///
/// `tau = -Kp e_q - Kd (omega - omega_des) - Ki z + omega x J omega`. Soft
/// shaping (`shape_pid_command` in the Python reference: `|dtau/dt|` and
/// eigenaxis-rate caps) is not ported because the app never sets those
/// knobs -- with both `None` it is an exact no-op in the original code too.
#[derive(Debug, Clone)]
pub struct PidAttitudeController {
    pub inertia: Mat3,
    pub kp: Mat3,
    pub kd: Mat3,
    pub ki: Mat3,
    pub integral_limit: f64,
    pub integral_gate: f64,
    pub torque_limit: Option<f64>,
    pub kaw: f64,
    pub gyroscopic_cancel: bool,
    z: Vec3,
}

impl PidAttitudeController {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        inertia: Mat3,
        wn: f64,
        zeta: f64,
        ki_wn_coeff: f64,
        gain_scale: f64,
        torque_limit: Option<f64>,
    ) -> Self {
        let (kp0, kd0, ki0) = tune_pid_second_order(&inertia, wn, zeta, ki_wn_coeff);
        PidAttitudeController {
            inertia,
            kp: gain_scale * kp0,
            kd: gain_scale * kd0,
            ki: gain_scale * ki0,
            integral_limit: PID_INTEGRAL_LIMIT,
            integral_gate: PID_INTEGRAL_GATE,
            torque_limit,
            kaw: PID_KAW,
            gyroscopic_cancel: true,
            z: Vec3::zeros(),
        }
    }

    pub fn reset(&mut self) {
        self.z = Vec3::zeros();
    }

    fn limit_torque(&self, tau: &Vec3) -> Vec3 {
        apply_torque_limits(tau, self.torque_limit, None)
    }

    pub fn command(
        &mut self,
        q: &Quat,
        omega: &Vec3,
        q_des: &Quat,
        omega_des: Option<Vec3>,
        dt: f64,
    ) -> Vec3 {
        let omega_des = omega_des.unwrap_or_else(Vec3::zeros);
        let e_q = q.attitude_error_vector(q_des);
        let e_w = omega - omega_des;
        let mut tau_unsat = -(self.kp * e_q) - (self.kd * e_w) - (self.ki * self.z);
        if self.gyroscopic_cancel {
            tau_unsat += omega.cross(&(self.inertia * omega));
        }
        let tau = self.limit_torque(&tau_unsat);
        self.update_integrator(&e_q, &tau_unsat, &tau, dt);
        tau
    }

    fn update_integrator(&mut self, e_q: &Vec3, tau_unsat: &Vec3, tau: &Vec3, dt: f64) {
        if !ki_is_active(&self.ki) || dt <= 0.0 {
            return;
        }
        let gated = e_q.norm() <= self.integral_gate;
        let excess = tau_unsat - tau;
        let sat_tol = 1e-15 + 1e-9 * tau_unsat.norm().max(1e-15);
        let saturated = excess.norm() > sat_tol;
        let mut z_dot = Vec3::zeros();
        if gated {
            let wind = saturated && excess.dot(&(self.ki * e_q)) < 0.0;
            if !wind {
                z_dot = *e_q;
            }
            if saturated {
                z_dot += backcalc_z_dot(&self.ki, tau_unsat, tau, self.kaw);
            }
        }
        self.z = clamp_integral(&(self.z + z_dot * dt), self.integral_limit);
    }
}

/// Continuous LQR on the linearized multiplicative-error model, with
/// optional gyroscopic cancellation and Euclidean torque saturation. No
/// integrator (LQR has no anti-windup state).
#[derive(Debug, Clone)]
pub struct LqrAttitudeController {
    pub inertia: Mat3,
    pub k: Mat3x6,
    pub torque_limit: Option<f64>,
    pub gyroscopic_cancel: bool,
}

impl LqrAttitudeController {
    pub fn new(
        inertia: Mat3,
        q: &Mat6,
        r: &Mat3,
        gain_scale: f64,
        torque_limit: Option<f64>,
    ) -> Self {
        let (k, _p, _a, _b) = design_attitude_lqr(&inertia, q, r);
        LqrAttitudeController {
            inertia,
            k: gain_scale * k,
            torque_limit,
            gyroscopic_cancel: true,
        }
    }

    pub fn reset(&mut self) {}

    fn limit_torque(&self, tau: &Vec3) -> Vec3 {
        apply_torque_limits(tau, self.torque_limit, None)
    }

    pub fn command(
        &mut self,
        q: &Quat,
        omega: &Vec3,
        q_des: &Quat,
        omega_des: Option<Vec3>,
        _dt: f64,
    ) -> Vec3 {
        let omega_des = omega_des.unwrap_or_else(Vec3::zeros);
        let dtheta = q.rotation_vector_error(q_des);
        let werr = omega - omega_des;
        let x = Vec6::new(dtheta.x, dtheta.y, dtheta.z, werr.x, werr.y, werr.z);
        let mut tau = -(self.k * x);
        if self.gyroscopic_cancel {
            tau += omega.cross(&(self.inertia * omega));
        }
        self.limit_torque(&tau)
    }
}

/// Either feedback law, so `sim.rs` can hold one trait-object-free enum.
#[derive(Debug, Clone)]
pub enum Controller {
    Pid(PidAttitudeController),
    Lqr(LqrAttitudeController),
}

impl Controller {
    pub fn reset(&mut self) {
        match self {
            Controller::Pid(c) => c.reset(),
            Controller::Lqr(c) => c.reset(),
        }
    }

    pub fn command(&mut self, q: &Quat, omega: &Vec3, q_des: &Quat, dt: f64) -> Vec3 {
        match self {
            Controller::Pid(c) => c.command(q, omega, q_des, None, dt),
            Controller::Lqr(c) => c.command(q, omega, q_des, None, dt),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn care_residual_is_near_zero_for_cubesat_plant() {
        let j = default_cubesat_inertia();
        let (a, b) = linearize_attitude(&j);
        let (q, r) = bryson_lqr_costs(LQR_THETA_REF, LQR_OMEGA_REF, LQR_TAU_REF);
        let p = solve_care(&a, &b, &q, &r);
        let residual = care_residual(&a, &b, &q, &r, &p);
        assert!(
            residual.norm() < 1e-6,
            "residual norm = {}",
            residual.norm()
        );
        // P must be symmetric positive semidefinite.
        assert_relative_eq!(p, p.transpose(), epsilon = 1e-9);
    }

    #[test]
    fn lqr_gain_is_positive_on_diagonal_for_diagonal_plant() {
        let j = default_cubesat_inertia();
        let (q, r) = bryson_lqr_costs(LQR_THETA_REF, LQR_OMEGA_REF, LQR_TAU_REF);
        let (k, _p, _a, _b) = design_attitude_lqr(&j, &q, &r);
        // Decoupled principal-axis plant => K should be (near-)diagonal blocks.
        for i in 0..3 {
            assert!(k[(i, i)] > 0.0);
            assert!(k[(i, i + 3)] > 0.0);
        }
    }

    #[test]
    fn pid_drives_small_error_to_zero() {
        let j = default_cubesat_inertia();
        let mut pid = PidAttitudeController::new(
            j,
            PID_WN,
            PID_ZETA,
            PID_KI_WN_COEFF,
            1.0,
            Some(DEFAULT_TORQUE_LIMIT),
        );
        let q_des = Quat::IDENTITY;
        let mut q = crate::quaternion::axis_angle_to_quat(&Vec3::new(0.0, 0.0, 1.0), 0.05);
        let mut omega = Vec3::zeros();
        let body = crate::plant::RigidBody::new(j).unwrap();
        let dt = 0.01;
        for _ in 0..3000 {
            let tau = pid.command(&q, &omega, &q_des, None, dt);
            let (qn, on) = crate::plant::step_rigid_body(&body, &q, &omega, &tau, dt);
            q = qn;
            omega = on;
        }
        assert!(q.geodesic_angle(&q_des) < 1e-3);
        assert!(omega.norm() < 1e-3);
    }

    #[test]
    fn integral_gain_is_zero_when_ki_wn_coeff_zero() {
        let j = default_cubesat_inertia();
        let mut pid =
            PidAttitudeController::new(j, PID_WN, PID_ZETA, 0.0, 1.0, Some(DEFAULT_TORQUE_LIMIT));
        assert!(!ki_is_active(&pid.ki));
        let q_des = Quat::IDENTITY;
        let q = Quat::IDENTITY;
        let before = pid.z;
        pid.command(&q, &Vec3::zeros(), &q_des, None, 0.01);
        assert_relative_eq!(pid.z, before);
    }
}

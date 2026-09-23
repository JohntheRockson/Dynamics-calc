//! Rigid-body rotational plant: quaternion kinematics + Euler's equation.
//!
//! State is the pair `(q, omega)`. Kinematics `q_dot = (1/2) q ⊗ [0, omega]`;
//! dynamics `J omega_dot = tau - omega x (J omega)`. Torque is held
//! zero-order over the sample and advanced with classical fourth-order
//! Runge-Kutta, followed by quaternion renormalization -- a direct port of
//! `attitude_sim.plant` (RK4 path only; the optional RKMK4 Lie-group
//! integrator is not exposed by the SimLab CLI/app and is not ported).

use crate::quaternion::{Mat3, Quat, Vec3};
use nalgebra::SymmetricEigen;
use thiserror::Error;

const SYMM_TOL: f64 = 1e-9;
const TRIANGLE_TOL: f64 = 1e-9;

#[derive(Debug, Error, Clone, PartialEq)]
pub enum PlantError {
    #[error("inertia must be finite")]
    NotFinite,
    #[error("inertia must be symmetric")]
    NotSymmetric,
    #[error("inertia must be positive definite")]
    NotPositiveDefinite,
    #[error("principal moments must satisfy I_i + I_j >= I_k (physical rigid-body inertia)")]
    NotPhysical,
}

/// Symmetrize + validate a 3x3 inertia tensor (SPD, triangle inequalities).
pub fn validate_inertia(inertia: &Mat3) -> Result<Mat3, PlantError> {
    if !inertia.iter().all(|v| v.is_finite()) {
        return Err(PlantError::NotFinite);
    }
    let mut max_asym: f64 = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            max_asym = max_asym.max((inertia[(i, j)] - inertia[(j, i)]).abs());
        }
    }
    if max_asym > SYMM_TOL {
        return Err(PlantError::NotSymmetric);
    }
    let j = 0.5 * (inertia + inertia.transpose());
    let eig = SymmetricEigen::new(j);
    let moments = eig.eigenvalues;
    if moments.iter().any(|&m| m <= 0.0) {
        return Err(PlantError::NotPositiveDefinite);
    }
    let total: f64 = moments.iter().sum();
    if moments.iter().any(|&m| total + TRIANGLE_TOL < 2.0 * m) {
        return Err(PlantError::NotPhysical);
    }
    Ok(j)
}

/// Principal moments (ascending) and a right-handed principal-axis DCM.
pub fn principal_moments_and_axes(inertia: &Mat3) -> (Vec3, Mat3) {
    let j = 0.5 * (inertia + inertia.transpose());
    let eig = SymmetricEigen::new(j);
    let mut idx = [0usize, 1, 2];
    idx.sort_by(|&a, &b| eig.eigenvalues[a].partial_cmp(&eig.eigenvalues[b]).unwrap());
    let moments = Vec3::new(
        eig.eigenvalues[idx[0]],
        eig.eigenvalues[idx[1]],
        eig.eigenvalues[idx[2]],
    );
    let mut axes = Mat3::from_columns(&[
        eig.eigenvectors.column(idx[0]).clone_owned(),
        eig.eigenvectors.column(idx[1]).clone_owned(),
        eig.eigenvectors.column(idx[2]).clone_owned(),
    ]);
    if axes.determinant() < 0.0 {
        let flipped = -axes.column(0);
        axes.set_column(0, &flipped);
    }
    (moments, axes)
}

/// A torque-driven rigid body with constant, validated body-frame inertia.
#[derive(Debug, Clone)]
pub struct RigidBody {
    pub inertia: Mat3,
    inertia_inv: Mat3,
}

impl RigidBody {
    pub fn new(inertia: Mat3) -> Result<Self, PlantError> {
        let j = validate_inertia(&inertia)?;
        let inv = j.try_inverse().ok_or(PlantError::NotPositiveDefinite)?;
        Ok(RigidBody {
            inertia: j,
            inertia_inv: inv,
        })
    }

    /// Euler's rotational equation: `omega_dot = J^-1 (tau - omega x (J omega))`.
    pub fn omega_dot(&self, omega: &Vec3, tau: &Vec3) -> Vec3 {
        let h = self.inertia * omega;
        self.inertia_inv * (tau - omega.cross(&h))
    }

    pub fn kinetic_energy(&self, omega: &Vec3) -> f64 {
        0.5 * omega.dot(&(self.inertia * omega))
    }

    pub fn angular_momentum_body(&self, omega: &Vec3) -> Vec3 {
        self.inertia * omega
    }

    pub fn angular_momentum_inertial(&self, q: &Quat, omega: &Vec3) -> Vec3 {
        q.to_rotation() * self.angular_momentum_body(omega)
    }
}

/// Plant state advanced by [`step_rigid_body`].
#[derive(Debug, Clone, Copy)]
pub struct State {
    pub q: Quat,
    pub omega: Vec3,
}

fn state_derivative(body: &RigidBody, s: &State, tau: &Vec3) -> (Quat, Vec3) {
    (s.q.derivative(&s.omega), body.omega_dot(&s.omega, tau))
}

fn combine(a: &State, qd: &Quat, wd: &Vec3, scale: f64) -> State {
    State {
        q: Quat::new(
            a.q.w + scale * qd.w,
            a.q.x + scale * qd.x,
            a.q.y + scale * qd.y,
            a.q.z + scale * qd.z,
        ),
        omega: a.omega + scale * wd,
    }
}

/// One classical RK4 step with zero-order-hold torque, followed by
/// quaternion renormalization.
pub fn step_rigid_body(
    body: &RigidBody,
    q: &Quat,
    omega: &Vec3,
    tau: &Vec3,
    dt: f64,
) -> (Quat, Vec3) {
    let y0 = State {
        q: *q,
        omega: *omega,
    };
    let half = 0.5 * dt;

    let (k1q, k1w) = state_derivative(body, &y0, tau);
    let y1 = combine(&y0, &k1q, &k1w, half);

    let (k2q, k2w) = state_derivative(body, &y1, tau);
    let y2 = combine(&y0, &k2q, &k2w, half);

    let (k3q, k3w) = state_derivative(body, &y2, tau);
    let y3 = combine(&y0, &k3q, &k3w, dt);

    let (k4q, k4w) = state_derivative(body, &y3, tau);

    let sixth = dt / 6.0;
    let q_next = Quat::new(
        y0.q.w + sixth * (k1q.w + 2.0 * k2q.w + 2.0 * k3q.w + k4q.w),
        y0.q.x + sixth * (k1q.x + 2.0 * k2q.x + 2.0 * k3q.x + k4q.x),
        y0.q.y + sixth * (k1q.y + 2.0 * k2q.y + 2.0 * k3q.y + k4q.y),
        y0.q.z + sixth * (k1q.z + 2.0 * k2q.z + 2.0 * k3q.z + k4q.z),
    );
    let omega_next = y0.omega + sixth * (k1w + 2.0 * k2w + 2.0 * k3w + k4w);
    (q_next.normalize(), omega_next)
}

/// Map a body-frame torque into inertial coordinates: `tau_I = R(q) tau_b`.
pub fn inertial_torque(q: &Quat, tau_body: &Vec3) -> Vec3 {
    q.to_rotation() * tau_body
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quaternion::axis_angle_to_quat;
    use approx::assert_relative_eq;

    fn cubesat_inertia() -> Mat3 {
        Mat3::from_diagonal(&Vec3::new(0.05, 0.06, 0.07))
    }

    #[test]
    fn validate_inertia_accepts_cubesat_default() {
        assert!(validate_inertia(&cubesat_inertia()).is_ok());
    }

    #[test]
    fn validate_inertia_rejects_non_physical() {
        // Grossly violates I_i + I_j >= I_k.
        let bad = Mat3::from_diagonal(&Vec3::new(0.001, 0.001, 10.0));
        assert_eq!(validate_inertia(&bad), Err(PlantError::NotPhysical));
    }

    #[test]
    fn principal_axes_are_identity_for_diagonal_inertia() {
        let (moments, axes) = principal_moments_and_axes(&cubesat_inertia());
        assert_relative_eq!(moments, Vec3::new(0.05, 0.06, 0.07), epsilon = 1e-12);
        assert_relative_eq!(axes, Mat3::identity(), epsilon = 1e-10);
    }

    #[test]
    fn torque_free_energy_and_momentum_are_conserved() {
        let body = RigidBody::new(cubesat_inertia()).unwrap();
        let mut q = Quat::IDENTITY;
        let mut omega = Vec3::new(0.2, -0.35, 0.5);
        let e0 = body.kinetic_energy(&omega);
        let h0 = body.angular_momentum_inertial(&q, &omega).norm();
        let dt = 1e-4;
        for _ in 0..2000 {
            let (qn, on) = step_rigid_body(&body, &q, &omega, &Vec3::zeros(), dt);
            q = qn;
            omega = on;
        }
        let e1 = body.kinetic_energy(&omega);
        let h1 = body.angular_momentum_inertial(&q, &omega).norm();
        assert_relative_eq!(e1, e0, epsilon = 1e-8);
        assert_relative_eq!(h1, h0, epsilon = 1e-8);
        assert_relative_eq!(q.norm(), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn constant_torque_matches_spherical_body_closed_form() {
        // Spherical inertia => omega(t) = (tau/I) t exactly (no gyroscopic coupling).
        let inertia_scalar = 0.05;
        let body = RigidBody::new(Mat3::from_diagonal(&Vec3::new(
            inertia_scalar,
            inertia_scalar,
            inertia_scalar,
        )))
        .unwrap();
        let tau = Vec3::new(0.01, 0.0, 0.0);
        let mut q = Quat::IDENTITY;
        let mut omega = Vec3::zeros();
        let dt = 0.001;
        let steps = 1000;
        for _ in 0..steps {
            let (qn, on) = step_rigid_body(&body, &q, &omega, &tau, dt);
            q = qn;
            omega = on;
        }
        let t_final = dt * steps as f64;
        let expected = tau / inertia_scalar * t_final;
        assert_relative_eq!(omega, expected, epsilon = 1e-6);
    }

    #[test]
    fn ninety_degree_slew_about_z_reaches_target() {
        // Pure spin about a principal axis is a torque-free equilibrium
        // (omega x J*omega = 0 exactly when omega is parallel to J*omega), so
        // omega should stay exactly constant and theta(t) = omega * t. Choose
        // omega/dt/steps with no rounding remainder so the only error source
        // under test is RK4 truncation, not a step-count/target-time mismatch.
        let body = RigidBody::new(cubesat_inertia()).unwrap();
        let mut q = Quat::IDENTITY;
        let mut omega = Vec3::new(0.0, 0.0, std::f64::consts::FRAC_PI_2); // rad/s
        let dt = 0.001;
        let steps = 1000; // exactly 1.0 s => 90 deg
        for _ in 0..steps {
            let (qn, on) = step_rigid_body(&body, &q, &omega, &Vec3::zeros(), dt);
            q = qn;
            omega = on;
        }
        let q_des = axis_angle_to_quat(&Vec3::new(0.0, 0.0, 1.0), std::f64::consts::FRAC_PI_2);
        assert_relative_eq!(q.geodesic_angle(&q_des), 0.0, epsilon = 1e-9);
        assert_relative_eq!(
            omega,
            Vec3::new(0.0, 0.0, std::f64::consts::FRAC_PI_2),
            epsilon = 1e-12
        );
    }
}

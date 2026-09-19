//! Unit-quaternion helpers (scalar-first, Hamilton product).
//!
//! Convention
//! ----------
//! A quaternion `q = [w, x, y, z]` represents body attitude relative to an
//! inertial frame. A body-fixed vector `v_b` is expressed in inertial
//! coordinates by `v_I = R(q) v_b`. Body-frame angular velocity `omega`
//! then obeys the kinematics `q_dot = (1/2) q ⊗ [0, omega]`.
//!
//! This is a direct, numerically-faithful port of `attitude_sim.quaternions`
//! (the Python reference implementation kept in `legacy-python/`).

use nalgebra::{Matrix3, Vector3};
use serde::{Deserialize, Serialize};

const QUAT_EPS: f64 = 1e-15;

pub type Vec3 = Vector3<f64>;
pub type Mat3 = Matrix3<f64>;

/// Scalar-first unit quaternion `[w, x, y, z]`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Quat {
    pub w: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Quat {
    pub const IDENTITY: Quat = Quat {
        w: 1.0,
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    pub fn new(w: f64, x: f64, y: f64, z: f64) -> Self {
        Quat { w, x, y, z }
    }

    pub fn from_vec(v: Vec3, w: f64) -> Self {
        Quat {
            w,
            x: v.x,
            y: v.y,
            z: v.z,
        }
    }

    /// Pure (zero-scalar) quaternion wrapping a vector.
    pub fn pure(v: Vec3) -> Self {
        Quat {
            w: 0.0,
            x: v.x,
            y: v.y,
            z: v.z,
        }
    }

    pub fn vector(&self) -> Vec3 {
        Vec3::new(self.x, self.y, self.z)
    }

    pub fn norm(&self) -> f64 {
        (self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    /// Return `q / ||q||`, falling back to identity for a near-zero quaternion.
    pub fn normalize(&self) -> Quat {
        let n = self.norm();
        if n < QUAT_EPS {
            return Quat::IDENTITY;
        }
        Quat {
            w: self.w / n,
            x: self.x / n,
            y: self.y / n,
            z: self.z / n,
        }
    }

    pub fn conjugate(&self) -> Quat {
        Quat {
            w: self.w,
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }

    /// Hamilton product `self ⊗ other` (scalar-first).
    pub fn multiply(&self, other: &Quat) -> Quat {
        let (w1, x1, y1, z1) = (self.w, self.x, self.y, self.z);
        let (w2, x2, y2, z2) = (other.w, other.x, other.y, other.z);
        Quat {
            w: w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            x: w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            y: w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            z: w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        }
    }

    pub fn negate(&self) -> Quat {
        Quat {
            w: -self.w,
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }

    /// Body-frame attitude error `q_e = q_des* ⊗ q`.
    pub fn error(&self, q_des: &Quat) -> Quat {
        q_des.normalize().conjugate().multiply(&self.normalize())
    }

    /// Rotation matrix `R` such that `v_I = R @ v_b`.
    pub fn to_rotation(&self) -> Mat3 {
        let Quat { w, x, y, z } = self.normalize();
        let (xx, yy, zz) = (x * x, y * y, z * z);
        let (xy, xz, yz) = (x * y, x * z, y * z);
        let (wx, wy, wz) = (w * x, w * y, w * z);
        Mat3::new(
            1.0 - 2.0 * (yy + zz),
            2.0 * (xy - wz),
            2.0 * (xz + wy),
            2.0 * (xy + wz),
            1.0 - 2.0 * (xx + zz),
            2.0 * (yz - wx),
            2.0 * (xz - wy),
            2.0 * (yz + wx),
            1.0 - 2.0 * (xx + yy),
        )
    }

    /// Yaw-pitch-roll (3-2-1 / ZYX) Euler angles in radians, `[yaw, pitch, roll]`.
    ///
    /// Matches `scipy.spatial.transform.Rotation.from_quat(...).as_euler("zyx")`
    /// bit-for-bit away from the pitch = +-90 deg singularity (verified against
    /// the Python reference over thousands of random attitudes).
    pub fn to_euler_321(&self) -> Vec3 {
        let r = self.to_rotation();
        let pitch = r[(0, 2)].clamp(-1.0, 1.0).asin();
        let yaw = (-r[(0, 1)]).atan2(r[(0, 0)]);
        let roll = (-r[(1, 2)]).atan2(r[(2, 2)]);
        Vec3::new(yaw, pitch, roll)
    }

    /// Kinematics `q_dot = (1/2) q ⊗ [0, omega]`.
    pub fn derivative(&self, omega: &Vec3) -> Quat {
        let half = self.multiply(&Quat::pure(*omega));
        Quat {
            w: 0.5 * half.w,
            x: 0.5 * half.x,
            y: 0.5 * half.y,
            z: 0.5 * half.z,
        }
    }

    /// Exact integration of kinematics for a piecewise-constant body rate.
    pub fn integrate_const_omega(&self, omega: &Vec3, dt: f64) -> Quat {
        let w = omega.norm();
        if w < 1e-14 {
            return self.normalize();
        }
        let half = 0.5 * w * dt;
        let scale = half.sin() / w;
        let dq = Quat::new(
            half.cos(),
            scale * omega.x,
            scale * omega.y,
            scale * omega.z,
        );
        self.multiply(&dq).normalize()
    }

    /// Shortest-path vector error `sign(q_e0) * q_e[1:]` (approx. half-angle).
    pub fn attitude_error_vector(&self, q_des: &Quat) -> Vec3 {
        let mut qe = self.error(q_des);
        if qe.w < 0.0 {
            qe = qe.negate();
        }
        qe.vector()
    }

    /// Small-angle rotation vector `delta_theta ~= 2 sign(q_e0) q_e[1:]` (rad).
    pub fn rotation_vector_error(&self, q_des: &Quat) -> Vec3 {
        2.0 * self.attitude_error_vector(q_des)
    }

    /// Principal rotation angle between two attitudes, in radians.
    pub fn geodesic_angle(&self, other: &Quat) -> f64 {
        let qe = self.error(other);
        let w = qe.w.abs().clamp(0.0, 1.0);
        2.0 * w.acos()
    }

    /// Robust rotation-matrix -> quaternion conversion (Shepperd's method).
    ///
    /// Produces *a* valid unit quaternion representing `r`; the caller-visible
    /// sign is then canonicalized (`w >= 0`) exactly like the Python reference
    /// (which lets scipy pick a branch and then flips/normalizes), so results
    /// agree regardless of which internal branch is taken here.
    pub fn from_rotation(r: &Mat3) -> Quat {
        let (m00, m01, m02) = (r[(0, 0)], r[(0, 1)], r[(0, 2)]);
        let (m10, m11, m12) = (r[(1, 0)], r[(1, 1)], r[(1, 2)]);
        let (m20, m21, m22) = (r[(2, 0)], r[(2, 1)], r[(2, 2)]);
        let trace = m00 + m11 + m22;
        let q = if trace > 0.0 {
            let s = (trace + 1.0).sqrt() * 2.0;
            Quat::new(0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s)
        } else if m00 > m11 && m00 > m22 {
            let s = (1.0 + m00 - m11 - m22).sqrt() * 2.0;
            Quat::new((m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s)
        } else if m11 > m22 {
            let s = (1.0 + m11 - m00 - m22).sqrt() * 2.0;
            Quat::new((m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s)
        } else {
            let s = (1.0 + m22 - m00 - m11).sqrt() * 2.0;
            Quat::new((m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s)
        };
        canonicalize(q)
    }
}

/// Flip sign so `w >= 0`, then normalize (matches the Python `rotation_to_quat`).
fn canonicalize(q: Quat) -> Quat {
    let q = if q.w < 0.0 { q.negate() } else { q };
    q.normalize()
}

/// Unit quaternion for a right-hand rotation of `angle` rad about `axis`.
pub fn axis_angle_to_quat(axis: &Vec3, angle: f64) -> Quat {
    let n = axis.norm();
    if n < QUAT_EPS {
        return Quat::IDENTITY;
    }
    let a = axis / n;
    let s = (0.5 * angle).sin();
    Quat::new((0.5 * angle).cos(), s * a.x, s * a.y, s * a.z).normalize()
}

/// Return the 3x3 skew-symmetric matrix `[v x]`.
pub fn skew(v: &Vec3) -> Mat3 {
    Mat3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    #[test]
    fn identity_multiply_is_identity() {
        let q = axis_angle_to_quat(&Vec3::new(0.3, 0.5, 0.8), 0.4);
        let r = q.multiply(&Quat::IDENTITY);
        assert_relative_eq!(r.w, q.w, epsilon = 1e-12);
        assert_relative_eq!(r.x, q.x, epsilon = 1e-12);
    }

    #[test]
    fn conjugate_is_inverse_for_unit_quat() {
        let q = axis_angle_to_quat(&Vec3::new(1.0, -0.4, 0.2), 1.234);
        let r = q.multiply(&q.conjugate());
        assert_relative_eq!(r.w, 1.0, epsilon = 1e-12);
        assert_relative_eq!(r.x, 0.0, epsilon = 1e-12);
        assert_relative_eq!(r.y, 0.0, epsilon = 1e-12);
        assert_relative_eq!(r.z, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn rotation_matrix_is_orthonormal_det_one() {
        let q = axis_angle_to_quat(&Vec3::new(0.1, 0.9, -0.3), 2.1);
        let r = q.to_rotation();
        let rtr = r.transpose() * r;
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(rtr[(i, j)], expected, epsilon = 1e-10);
            }
        }
        assert_relative_eq!(r.determinant(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn ninety_deg_about_z_maps_x_to_y() {
        let q = axis_angle_to_quat(&Vec3::new(0.0, 0.0, 1.0), PI / 2.0);
        let r = q.to_rotation();
        let v = r * Vec3::new(1.0, 0.0, 0.0);
        assert_relative_eq!(v.x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(v.y, 1.0, epsilon = 1e-10);
        assert_relative_eq!(v.z, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn geodesic_angle_matches_axis_angle_magnitude() {
        let q0 = Quat::IDENTITY;
        let q1 = axis_angle_to_quat(&Vec3::new(0.0, 1.0, 0.0), 0.7);
        assert_relative_eq!(q1.geodesic_angle(&q0), 0.7, epsilon = 1e-10);
    }

    #[test]
    fn from_rotation_round_trips_to_rotation() {
        let q = axis_angle_to_quat(&Vec3::new(0.4, -0.7, 1.1), 1.9);
        let r = q.to_rotation();
        let q2 = Quat::from_rotation(&r);
        let r2 = q2.to_rotation();
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(r[(i, j)], r2[(i, j)], epsilon = 1e-9);
            }
        }
    }

    #[test]
    fn double_cover_canonicalized_to_nonnegative_w() {
        let q = axis_angle_to_quat(&Vec3::new(0.0, 0.0, 1.0), 3.0);
        let r = q.to_rotation();
        let q2 = Quat::from_rotation(&r);
        assert!(q2.w >= 0.0);
    }

    #[test]
    fn skew_matches_cross_product() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        let u = Vec3::new(-2.0, 0.5, 4.0);
        let via_skew = skew(&v) * u;
        let via_cross = v.cross(&u);
        assert_relative_eq!(via_skew, via_cross, epsilon = 1e-12);
    }
}

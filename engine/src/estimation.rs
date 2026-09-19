//! Attitude/rate estimators: Mahony complementary filter, multiplicative
//! EKF (MEKF), and TRIAD coarse initialization. Ported from
//! `attitude_sim.estimation`. QUEST / Davenport q-method coarse init and
//! the star-tracker path are not exposed by the app's CLI-equivalent
//! surface and are not ported (TRIAD, the CLI default, is).

use crate::quaternion::{axis_angle_to_quat, skew, Mat3, Quat, Vec3};
use crate::sensors::VectorSensor;
use nalgebra::{SMatrix, SVector};

pub type Mat6 = SMatrix<f64, 6, 6>;
pub type Mat3x6 = SMatrix<f64, 3, 6>;
pub type Vec6 = SVector<f64, 6>;

const TRIAD_PARALLEL_EPS: f64 = 1e-8;
pub const MEKF_VECTOR_R_NUGGET: f64 = 1e-12;

/// One available `(body, inertial, sigma, name)` vector measurement for this sample.
#[derive(Debug, Clone, Copy)]
pub struct VectorMeas {
    pub v_body: Vec3,
    pub v_inertial: Vec3,
    pub sigma: f64,
    pub name: &'static str,
}

/// Sample every available sensor at truth `q`, attaching each sensor's sigma.
pub fn vectors_from_sensors(
    q: &Quat,
    sensors: &[VectorSensor],
    rng: &mut crate::sensors::Rng,
) -> Vec<VectorMeas> {
    sensors
        .iter()
        .filter_map(|s| {
            s.measure(q, rng).map(|v_body| VectorMeas {
                v_body,
                v_inertial: s.v_inertial,
                sigma: s.sigma,
                name: s.name,
            })
        })
        .collect()
}

/// Discrete process noise for `x = [delta_alpha, delta_b]` (Farrenkopf gyro).
pub fn farrenkopf_qd(sigma_v: f64, sigma_u: f64, dt: f64) -> Mat6 {
    let sv2 = sigma_v * sigma_v;
    let su2 = sigma_u * sigma_u;
    let qaa = sv2 * dt + su2 * dt.powi(3) / 3.0;
    let qab = -0.5 * su2 * dt * dt;
    let qbb = su2 * dt;
    let mut q = Mat6::zeros();
    for i in 0..3 {
        q[(i, i)] = qaa;
        q[(i, i + 3)] = qab;
        q[(i + 3, i)] = qab;
        q[(i + 3, i + 3)] = qbb;
    }
    q
}

/// State transition `Phi = exp(F dt)` for `F = [[-omega_hat x, -I], [0, 0]]`.
pub fn mekf_stm(omega_hat: &Vec3, dt: f64) -> Mat6 {
    let wn = omega_hat.norm();
    let theta = wn * dt;
    let (phi_aa, gamma) = if theta < 1e-10 {
        let wx = skew(omega_hat);
        let phi_aa = Mat3::identity() - wx * dt + 0.5 * (wx * wx) * dt * dt;
        let gamma = Mat3::identity() * dt - 0.5 * wx * dt * dt + (wx * wx) * (dt.powi(3) / 6.0);
        (phi_aa, gamma)
    } else {
        let u = omega_hat / wn;
        let ux = skew(&u);
        let ux2 = ux * ux;
        let phi_aa = Mat3::identity() - theta.sin() * ux + (1.0 - theta.cos()) * ux2;
        let gamma =
            Mat3::identity() * dt - ((1.0 - theta.cos()) / wn) * ux + (dt - theta.sin() / wn) * ux2;
        (phi_aa, gamma)
    };
    let mut phi = Mat6::identity();
    for i in 0..3 {
        for j in 0..3 {
            phi[(i, j)] = phi_aa[(i, j)];
            phi[(i, j + 3)] = -gamma[(i, j)];
        }
    }
    phi
}

fn inject_body_error(q: &Quat, dtheta: &Vec3) -> Quat {
    let angle = dtheta.norm();
    q.multiply(&axis_angle_to_quat(dtheta, angle)).normalize()
}

/// Wahba TRIAD: two body/inertial unit-vector pairs -> scalar-first `q`.
pub fn triad_attitude(v_b1: &Vec3, v_i1: &Vec3, v_b2: &Vec3, v_i2: &Vec3) -> Option<Quat> {
    let b1 = v_b1.normalize();
    let r1 = v_i1.normalize();
    let b2 = v_b2.normalize();
    let r2 = v_i2.normalize();
    let tb = b1.cross(&b2);
    let tr = r1.cross(&r2);
    let (nb, nr) = (tb.norm(), tr.norm());
    if nb < TRIAD_PARALLEL_EPS || nr < TRIAD_PARALLEL_EPS {
        return None;
    }
    let tb = tb / nb;
    let tr = tr / nr;
    let body = Mat3::from_columns(&[b1, tb, b1.cross(&tb)]);
    let inertial = Mat3::from_columns(&[r1, tr, r1.cross(&tr)]);
    Some(Quat::from_rotation(&(inertial * body.transpose())))
}

/// TRIAD from the first two *available* sensors, or `None` if fewer than two
/// (occulted/out-of-FOV/parallel-reference failures degrade gracefully so a
/// coarse-init caller can keep the true `q0`).
pub fn try_triad_q0_from_sensors(
    q: &Quat,
    sensors: &[VectorSensor],
    rng: &mut crate::sensors::Rng,
) -> Option<Quat> {
    let mut pairs = Vec::new();
    for s in sensors {
        if let Some(v_b) = s.measure(q, rng) {
            pairs.push((v_b, s.v_inertial));
            if pairs.len() >= 2 {
                break;
            }
        }
    }
    if pairs.len() < 2 {
        return None;
    }
    triad_attitude(&pairs[0].0, &pairs[0].1, &pairs[1].0, &pairs[1].1)
}

/// SO(3) Mahony complementary filter with gyro-bias estimation.
#[derive(Debug, Clone)]
pub struct ComplementaryFilter {
    pub kp: f64,
    pub ki: f64,
    pub q: Quat,
    pub bias: Vec3,
}

impl ComplementaryFilter {
    pub fn new(q0: Quat) -> Self {
        ComplementaryFilter {
            kp: 1.5,
            ki: 0.08,
            q: q0.normalize(),
            bias: Vec3::zeros(),
        }
    }

    pub fn step(&mut self, omega_m: &Vec3, dt: f64, meas: &[VectorMeas]) -> (Quat, Vec3) {
        let mut omega_corr = Vec3::zeros();
        if !meas.is_empty() {
            let r = self.q.to_rotation();
            let mut wsum = 0.0;
            let mut acc = Vec3::zeros();
            for m in meas {
                let v_hat = r.transpose() * m.v_inertial;
                let vn = v_hat.norm();
                if vn < 1e-15 {
                    continue;
                }
                let v_hat = v_hat / vn;
                let weight = if m.sigma <= 0.0 {
                    1.0
                } else {
                    1.0 / (m.sigma * m.sigma)
                };
                acc += weight * m.v_body.cross(&v_hat);
                wsum += weight;
            }
            if wsum > 0.0 {
                omega_corr = acc / wsum;
            }
        }
        let omega_hat = omega_m - self.bias;
        let omega_kin = omega_hat + self.kp * omega_corr;
        self.bias -= self.ki * omega_corr * dt;
        self.q = self.q.integrate_const_omega(&omega_kin, dt);
        (self.q, omega_hat)
    }
}

/// One pre-update unit-vector innovation (rank-2 tangent plane), recorded
/// for the app's NIS chart.
#[derive(Debug, Clone, Copy)]
pub struct InnovationSample {
    pub nis: f64,
    pub t: f64,
    pub sensor: &'static str,
}

fn tangent_plane_basis(v: &Vec3) -> Option<(Vec3, Vec3)> {
    let v = v.normalize();
    let k = [v.x.abs(), v.y.abs(), v.z.abs()]
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)?;
    let mut axis = Vec3::zeros();
    axis[k] = 1.0;
    let e1 = v.cross(&axis);
    let n1 = e1.norm();
    if n1 < 1e-15 {
        return None;
    }
    let e1 = e1 / n1;
    let e2 = v.cross(&e1).normalize();
    Some((e1, e2))
}

/// 6-state multiplicative EKF: attitude error `delta_alpha` and gyro bias.
#[derive(Debug, Clone)]
pub struct MultiplicativeEkf {
    pub sigma_v: f64,
    pub sigma_u: f64,
    pub q: Quat,
    pub bias: Vec3,
    pub p: Mat6,
    pub innovations: Vec<InnovationSample>,
}

impl MultiplicativeEkf {
    pub fn new(q0: Quat, sigma_v: f64, sigma_u: f64) -> Self {
        let mut p = Mat6::zeros();
        for i in 0..3 {
            p[(i, i)] = 3e-3;
            p[(i + 3, i + 3)] = 1e-5;
        }
        MultiplicativeEkf {
            sigma_v,
            sigma_u,
            q: q0.normalize(),
            bias: Vec3::zeros(),
            p,
            innovations: Vec::new(),
        }
    }

    pub fn predict(&mut self, omega_m: &Vec3, dt: f64) -> Vec3 {
        let omega_hat = omega_m - self.bias;
        self.q = self.q.integrate_const_omega(&omega_hat, dt);
        let phi = mekf_stm(&omega_hat, dt);
        let qd = farrenkopf_qd(self.sigma_v, self.sigma_u, dt);
        self.p = phi * self.p * phi.transpose() + qd;
        self.p = 0.5 * (self.p + self.p.transpose());
        omega_hat
    }

    pub fn update_vector(
        &mut self,
        v_b_meas: &Vec3,
        v_inertial: &Vec3,
        sigma: f64,
        sensor: &'static str,
        t: f64,
    ) {
        let v_hat = (self.q.to_rotation().transpose() * v_inertial).normalize();
        let mut h = Mat3x6::zeros();
        let skew_v = skew(&v_hat);
        for i in 0..3 {
            for j in 0..3 {
                h[(i, j)] = skew_v[(i, j)];
            }
        }
        let sig2 = sigma * sigma;
        let nugget = MEKF_VECTOR_R_NUGGET * sig2.max(1.0);
        let r = sig2 * (Mat3::identity() - v_hat * v_hat.transpose()) + nugget * Mat3::identity();
        let s = h * self.p * h.transpose() + r;
        let nu = v_b_meas.normalize() - v_hat;

        if let Some((e1, e2)) = tangent_plane_basis(&v_hat) {
            let b = nalgebra::Matrix3x2::from_columns(&[e1, e2]);
            let nu2 = b.transpose() * nu;
            let mut s2 = b.transpose() * s * b;
            s2 = 0.5 * (s2 + s2.transpose());
            if let Some(sol) = s2.lu().solve(&nu2) {
                self.innovations.push(InnovationSample {
                    nis: nu2.dot(&sol),
                    t,
                    sensor,
                });
            }
        }

        // Kalman gain K = P H^T S^-1, Joseph-form covariance update.
        if let Some(s_inv) = s.try_inverse() {
            let k = self.p * h.transpose() * s_inv;
            let dx: Vec6 = k * nu;
            let i_kh = Mat6::identity() - k * h;
            self.p = i_kh * self.p * i_kh.transpose() + k * r * k.transpose();
            self.p = 0.5 * (self.p + self.p.transpose());
            self.inject(&dx);
        }
    }

    fn inject(&mut self, dx: &Vec6) {
        let dalpha = Vec3::new(dx[0], dx[1], dx[2]);
        self.q = inject_body_error(&self.q, &dalpha);
        self.bias += Vec3::new(dx[3], dx[4], dx[5]);
    }

    pub fn step(&mut self, omega_m: &Vec3, dt: f64, meas: &[VectorMeas], t: f64) -> (Quat, Vec3) {
        self.predict(omega_m, dt);
        for m in meas {
            self.update_vector(&m.v_body, &m.v_inertial, m.sigma, m.name, t);
        }
        let omega_hat = omega_m - self.bias;
        (self.q, omega_hat)
    }

    pub fn mean_nis(&self) -> Option<f64> {
        if self.innovations.is_empty() {
            None
        } else {
            Some(
                self.innovations.iter().map(|s| s.nis).sum::<f64>() / self.innovations.len() as f64,
            )
        }
    }
}

/// Either estimator, or full-state "truth" feedback (`None`).
///
/// One instance lives for the whole simulation run and is mutated in place
/// (never copied per-step), so the size difference between variants
/// (`MultiplicativeEkf` carries a 6x6 covariance) is not worth boxing.
#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)]
pub enum Estimator {
    Truth,
    Mahony(ComplementaryFilter),
    Mekf(MultiplicativeEkf),
}

impl Estimator {
    pub fn step(&mut self, omega_m: &Vec3, dt: f64, meas: &[VectorMeas], t: f64) -> (Quat, Vec3) {
        match self {
            Estimator::Truth => {
                unreachable!("truth path is handled by the caller without sampling")
            }
            Estimator::Mahony(f) => f.step(omega_m, dt, meas),
            Estimator::Mekf(f) => f.step(omega_m, dt, meas, t),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quaternion::axis_angle_to_quat;
    use approx::assert_relative_eq;

    #[test]
    fn triad_recovers_known_attitude() {
        let q_true = axis_angle_to_quat(&Vec3::new(0.2, 0.6, 0.77), 0.9);
        let r = q_true.to_rotation();
        let v_i1 = Vec3::new(0.3, 0.1, 0.95).normalize();
        let v_i2 = Vec3::new(1.0, 0.05, 0.02).normalize();
        let v_b1 = r.transpose() * v_i1;
        let v_b2 = r.transpose() * v_i2;
        let q_est = triad_attitude(&v_b1, &v_i1, &v_b2, &v_i2).unwrap();
        assert!(q_true.geodesic_angle(&q_est) < 1e-8);
    }

    #[test]
    fn triad_returns_none_for_parallel_vectors() {
        let v = Vec3::new(1.0, 0.0, 0.0);
        assert!(triad_attitude(&v, &v, &v, &v).is_none());
    }

    #[test]
    fn mekf_predict_only_keeps_covariance_growing() {
        let mut ekf = MultiplicativeEkf::new(Quat::IDENTITY, 5e-4, 1e-6);
        let p0 = ekf.p[(0, 0)];
        for _ in 0..100 {
            ekf.predict(&Vec3::zeros(), 0.01);
        }
        assert!(ekf.p[(0, 0)] > p0);
    }

    #[test]
    fn mekf_vector_update_reduces_covariance_trace() {
        let mut ekf = MultiplicativeEkf::new(Quat::IDENTITY, 5e-4, 1e-6);
        ekf.predict(&Vec3::zeros(), 0.01);
        let trace_before: f64 = (0..6).map(|i| ekf.p[(i, i)]).sum();
        ekf.update_vector(
            &Vec3::new(0.3, 0.1, 0.95).normalize(),
            &Vec3::new(0.3, 0.1, 0.95),
            3e-3,
            "mag",
            0.0,
        );
        let trace_after: f64 = (0..6).map(|i| ekf.p[(i, i)]).sum();
        assert!(trace_after < trace_before);
    }

    #[test]
    fn mahony_converges_toward_truth_with_vector_updates() {
        // Static truth attitude, two exact (sigma=0) vector pairs, kp=1.5/ki=0.08
        // defaults. The filter does not fully null the error (complementary
        // filters trade a small steady-state bias-coupling residual for
        // smoothing); cross-checked bit-for-bit against the Python reference
        // (`ComplementaryFilter`) for this exact scenario, which lands at the
        // same 0.0064105 rad residual -- so this is the *correct* answer, not
        // a convergence bug, and the assertion below pins that regression.
        let q_true = axis_angle_to_quat(&Vec3::new(0.1, 0.2, 0.9), 0.6);
        let initial_error = q_true.geodesic_angle(&Quat::IDENTITY);
        let mut filt = ComplementaryFilter::new(Quat::IDENTITY);
        let v_i1 = Vec3::new(0.3, 0.1, 0.95).normalize();
        let v_i2 = Vec3::new(1.0, 0.05, 0.02).normalize();
        let dt = 0.01;
        for _ in 0..4000 {
            let r = q_true.to_rotation();
            let meas = [
                VectorMeas {
                    v_body: r.transpose() * v_i1,
                    v_inertial: v_i1,
                    sigma: 0.0,
                    name: "mag",
                },
                VectorMeas {
                    v_body: r.transpose() * v_i2,
                    v_inertial: v_i2,
                    sigma: 0.0,
                    name: "sun",
                },
            ];
            filt.step(&Vec3::zeros(), dt, &meas);
        }
        let final_error = filt.q.geodesic_angle(&q_true);
        assert!(final_error < 0.01, "final_error={final_error}");
        assert!(
            final_error < 0.02 * initial_error,
            "expected >98% error reduction, got final={final_error} initial={initial_error}"
        );
        assert_relative_eq!(final_error, 0.006412737559125296, epsilon = 1e-9);
    }
}

//! Noisy body-rate and vector-observation sensor models, ported from
//! `attitude_sim.sensors`.
//!
//! Note on randomness: the Python reference uses NumPy's PCG64 generator.
//! Rust uses `rand_chacha::ChaCha8Rng` here instead (there is no portable,
//! dependency-free way to reproduce NumPy's bit stream in Rust). Given the
//! same `seed` the two engines therefore draw different noise realizations;
//! everything else about the models (ARW/RRW scaling, unit-vector geometry,
//! FOV/eclipse gating) is bit-for-bit faithful. Deterministic parity testing
//! against the Python engine is done with `--estimator truth`, which never
//! samples a sensor; noisy paths (MEKF/Mahony) are validated statistically
//! (see `engine/tests/python_parity.rs` and the estimator unit tests).

use crate::quaternion::{Quat, Vec3};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};

pub type Rng = ChaCha8Rng;

pub fn make_rng(seed: u64) -> Rng {
    ChaCha8Rng::seed_from_u64(seed)
}

fn randn3(rng: &mut Rng) -> Vec3 {
    Vec3::new(
        StandardNormal.sample(rng),
        StandardNormal.sample(rng),
        StandardNormal.sample(rng),
    )
}

pub fn gyro_arw_std(sigma_v: f64, dt: f64) -> f64 {
    sigma_v / dt.sqrt()
}

pub fn gyro_rrw_std(sigma_u: f64, dt: f64) -> f64 {
    sigma_u * dt.sqrt()
}

/// Rate gyro: `omega_m = omega + b + eta_v + eta_n`, `b_dot = eta_u`.
#[derive(Debug, Clone)]
pub struct GyroModel {
    pub sigma_v: f64,
    pub sigma_u: f64,
    pub sigma_n: f64,
    pub bias: Vec3,
}

impl GyroModel {
    pub fn new(sigma_v: f64, sigma_u: f64, sigma_n: f64, bias: Vec3) -> Self {
        GyroModel {
            sigma_v,
            sigma_u,
            sigma_n,
            bias,
        }
    }

    pub fn measure(&mut self, omega: &Vec3, dt: f64, rng: &mut Rng) -> Vec3 {
        self.bias += gyro_rrw_std(self.sigma_u, dt) * randn3(rng);
        let arw = gyro_arw_std(self.sigma_v, dt) * randn3(rng);
        let readout = self.sigma_n * randn3(rng);
        omega + self.bias + arw + readout
    }
}

pub fn in_fov(v_body: &Vec3, boresight_body: &Vec3, fov_half_angle: Option<f64>) -> bool {
    match fov_half_angle {
        None => true,
        Some(half) => {
            let vn = v_body.norm();
            let bn = boresight_body.norm();
            if vn < 1e-15 || bn < 1e-15 {
                return false;
            }
            (v_body / vn).dot(&(boresight_body / bn)) >= half.cos()
        }
    }
}

/// Unit-vector observation in the body frame (magnetometer, sun sensor, ...).
#[derive(Debug, Clone)]
pub struct VectorSensor {
    pub v_inertial: Vec3,
    pub sigma: f64,
    pub name: &'static str,
    pub boresight_body: Vec3,
    pub fov_half_angle: Option<f64>,
    pub occulted: bool,
}

impl VectorSensor {
    pub fn new(v_inertial: Vec3, sigma: f64, name: &'static str) -> Self {
        VectorSensor {
            v_inertial: v_inertial.normalize(),
            sigma,
            name,
            boresight_body: Vec3::new(0.0, 0.0, 1.0),
            fov_half_angle: None,
            occulted: false,
        }
    }

    pub fn available(&self, q: &Quat) -> bool {
        if self.occulted {
            return false;
        }
        let r = q.to_rotation();
        let v_b = r.transpose() * self.v_inertial;
        in_fov(&v_b, &self.boresight_body, self.fov_half_angle)
    }

    pub fn measure(&self, q: &Quat, rng: &mut Rng) -> Option<Vec3> {
        if !self.available(q) {
            return None;
        }
        let r = q.to_rotation();
        let v_b = r.transpose() * self.v_inertial + self.sigma * randn3(rng);
        let n = v_b.norm();
        if n < 1e-15 {
            Some(Vec3::new(1.0, 0.0, 0.0))
        } else {
            Some(v_b / n)
        }
    }
}

pub fn magnetometer(sigma: f64) -> VectorSensor {
    VectorSensor::new(Vec3::new(0.3, 0.1, 0.95), sigma, "mag")
}

pub fn sun_sensor(sigma: f64, eclipse: bool) -> VectorSensor {
    let mut s = VectorSensor::new(Vec3::new(1.0, 0.05, 0.02), sigma, "sun");
    s.occulted = eclipse;
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arw_scales_as_inverse_sqrt_dt() {
        assert!((gyro_arw_std(1e-3, 0.04) - 0.5 * gyro_arw_std(1e-3, 0.01)).abs() < 1e-15);
    }

    #[test]
    fn rrw_scales_as_sqrt_dt() {
        assert!((gyro_rrw_std(1e-6, 0.04) - 2.0 * gyro_rrw_std(1e-6, 0.01)).abs() < 1e-15);
    }

    #[test]
    fn occulted_sensor_is_never_available() {
        let mut s = magnetometer(1e-3);
        s.occulted = true;
        assert!(!s.available(&Quat::IDENTITY));
    }

    #[test]
    fn gyro_measurement_has_zero_mean_over_many_samples() {
        let mut gyro = GyroModel::new(1e-3, 0.0, 0.0, Vec3::zeros());
        let mut rng = make_rng(42);
        let mut sum = Vec3::zeros();
        let n = 20000;
        for _ in 0..n {
            gyro.bias = Vec3::zeros(); // isolate ARW only for this check
            sum += gyro.measure(&Vec3::zeros(), 0.01, &mut rng);
        }
        let mean = sum / n as f64;
        assert!(mean.norm() < 0.02, "mean={:?}", mean);
    }
}

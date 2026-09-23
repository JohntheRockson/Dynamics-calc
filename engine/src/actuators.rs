//! Reaction-wheel / torque-limit actuator: per-axis clip, optional first-order
//! lag, optional momentum dump. Direct port of `attitude_sim.actuators`
//! (`TorqueActuator` half; the reaction-wheel assembly lives in
//! `reaction_wheels.rs`).

use crate::quaternion::Vec3;
use serde::{Deserialize, Serialize};

pub const DEFAULT_DUMP_GAIN: f64 = 1.0;

/// Per-axis (or isotropic) non-negative limit, e.g. wheel torque/momentum box.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AxisLimit {
    Scalar(f64),
    Vector([f64; 3]),
}

impl AxisLimit {
    pub fn as_vec3(&self) -> Vec3 {
        match self {
            AxisLimit::Scalar(s) => Vec3::new(*s, *s, *s),
            AxisLimit::Vector(v) => Vec3::new(v[0], v[1], v[2]),
        }
    }
}

/// Clip body torque to per-axis limits `+-tau_max`. `None` is unlimited.
pub fn clip_torque(tau: &Vec3, tau_max: Option<AxisLimit>) -> Vec3 {
    match tau_max {
        None => *tau,
        Some(lim) => {
            let l = lim.as_vec3();
            Vec3::new(
                tau.x.clamp(-l.x, l.x),
                tau.y.clamp(-l.y, l.y),
                tau.z.clamp(-l.z, l.z),
            )
        }
    }
}

/// Deadzone dump torque from a wheel-momentum proxy `h` (N*m*s).
pub fn momentum_dump_torque(momentum: &Vec3, h_dump: Option<AxisLimit>, dump_gain: f64) -> Vec3 {
    if dump_gain == 0.0 {
        return Vec3::zeros();
    }
    match h_dump {
        None => Vec3::zeros(),
        Some(lim) => {
            let t = lim.as_vec3();
            let excess = Vec3::new(
                momentum.x.signum() * (momentum.x.abs() - t.x).max(0.0),
                momentum.y.signum() * (momentum.y.abs() - t.y).max(0.0),
                momentum.z.signum() * (momentum.z.abs() - t.z).max(0.0),
            );
            dump_gain * excess
        }
    }
}

/// Per-axis saturation + optional first-order lag + optional momentum dump.
#[derive(Debug, Clone)]
pub struct TorqueActuator {
    pub tau_max: Option<AxisLimit>,
    pub time_constant: Option<f64>,
    pub h_dump: Option<AxisLimit>,
    pub dump_gain: f64,
    tau: Vec3,
    h: Vec3,
    tau_dump: Vec3,
}

impl TorqueActuator {
    pub fn new(
        tau_max: Option<AxisLimit>,
        time_constant: Option<f64>,
        h_dump: Option<AxisLimit>,
        dump_gain: f64,
    ) -> Self {
        TorqueActuator {
            tau_max,
            time_constant,
            h_dump,
            dump_gain,
            tau: Vec3::zeros(),
            h: Vec3::zeros(),
            tau_dump: Vec3::zeros(),
        }
    }

    pub fn torque(&self) -> Vec3 {
        self.tau
    }

    pub fn momentum(&self) -> Vec3 {
        self.h
    }

    /// External cancel of the dump so the spacecraft net dump is zero.
    pub fn external_torque(&self) -> Vec3 {
        -self.tau_dump
    }

    /// Advance one sample: clip command, apply leftover-authority dump, lag, clip.
    /// Returns the applied wheel torque.
    pub fn apply(&mut self, command: &Vec3, dt: f64) -> Vec3 {
        let u_cmd = clip_torque(command, self.tau_max);
        let raw_dump = momentum_dump_torque(&self.h, self.h_dump, self.dump_gain);
        let tau_dump = clip_torque(&(u_cmd + raw_dump), self.tau_max) - u_cmd;
        self.tau_dump = tau_dump;
        let u = u_cmd + tau_dump;
        match self.time_constant {
            None => self.tau = u,
            Some(t) if t <= 0.0 => self.tau = u,
            Some(t) => {
                let alpha = (-dt / t).exp();
                self.tau = clip_torque(&(alpha * self.tau + (1.0 - alpha) * u), self.tau_max);
            }
        }
        if dt > 0.0 {
            self.h -= self.tau * dt;
        }
        self.tau
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn clip_torque_none_is_passthrough() {
        let tau = Vec3::new(1.0, -2.0, 3.0);
        assert_relative_eq!(clip_torque(&tau, None), tau);
    }

    #[test]
    fn clip_torque_scalar_box() {
        let tau = Vec3::new(1.0, -2.0, 0.05);
        let out = clip_torque(&tau, Some(AxisLimit::Scalar(0.5)));
        assert_relative_eq!(out, Vec3::new(0.5, -0.5, 0.05));
    }

    #[test]
    fn unlimited_actuator_is_identity() {
        let mut act = TorqueActuator::new(None, None, None, DEFAULT_DUMP_GAIN);
        let cmd = Vec3::new(0.01, -0.02, 0.005);
        let out = act.apply(&cmd, 0.01);
        assert_relative_eq!(out, cmd);
    }

    #[test]
    fn lag_converges_to_command_at_steady_state() {
        let mut act = TorqueActuator::new(None, Some(0.1), None, DEFAULT_DUMP_GAIN);
        let cmd = Vec3::new(0.02, 0.0, 0.0);
        let mut out = Vec3::zeros();
        for _ in 0..5000 {
            out = act.apply(&cmd, 0.001);
        }
        assert_relative_eq!(out, cmd, epsilon = 1e-6);
    }

    #[test]
    fn momentum_integrates_negative_torque() {
        let mut act = TorqueActuator::new(None, None, None, DEFAULT_DUMP_GAIN);
        let cmd = Vec3::new(0.01, 0.0, 0.0);
        for _ in 0..100 {
            act.apply(&cmd, 0.01);
        }
        // h_dot = -tau => h(1s) = -0.01
        assert_relative_eq!(act.momentum().x, -0.01, epsilon = 1e-12);
    }
}

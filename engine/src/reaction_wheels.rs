//! Three-axis reaction-wheel assembly: motor torque, momentum storage,
//! per-axis |tau|/|h| saturation, viscous + smoothed-Coulomb friction, and
//! the omega x h_w gyroscopic couple. Additive actuator stage sitting after
//! the controller command, matching `attitude_sim.reaction_wheels`.

use crate::actuators::{clip_torque, AxisLimit};
use crate::quaternion::Vec3;

pub const DEFAULT_WHEEL_INERTIA: f64 = 2.0e-4;
const COULOMB_EPS: f64 = 0.05;
const SAT_FRAC: f64 = 1.0 - 1e-9;

fn gate_momentum_saturation(tau_m: &Vec3, h: &Vec3, h_max: Option<AxisLimit>) -> Vec3 {
    match h_max {
        None => *tau_m,
        Some(lim) => {
            let l = lim.as_vec3();
            let mut out = *tau_m;
            for i in 0..3 {
                let at_hi = h[i] >= l[i] * SAT_FRAC;
                let at_lo = h[i] <= -l[i] * SAT_FRAC;
                if (at_hi && out[i] > 0.0) || (at_lo && out[i] < 0.0) {
                    out[i] = 0.0;
                }
            }
            out
        }
    }
}

fn friction_torque(omega_w: &Vec3, visc: f64, coulomb: f64) -> Vec3 {
    Vec3::new(
        -visc * omega_w.x - coulomb * (omega_w.x / COULOMB_EPS).tanh(),
        -visc * omega_w.y - coulomb * (omega_w.y / COULOMB_EPS).tanh(),
        -visc * omega_w.z - coulomb * (omega_w.z / COULOMB_EPS).tanh(),
    )
}

fn at_limit(value: &Vec3, limit: Option<AxisLimit>) -> [bool; 3] {
    match limit {
        None => [false, false, false],
        Some(lim) => {
            let l = lim.as_vec3();
            [
                value.x.abs() >= l.x * SAT_FRAC,
                value.y.abs() >= l.y * SAT_FRAC,
                value.z.abs() >= l.z * SAT_FRAC,
            ]
        }
    }
}

#[derive(Debug, Clone)]
pub struct ReactionWheelAssembly {
    pub tau_max: Option<AxisLimit>,
    pub h_max: Option<AxisLimit>,
    pub wheel_inertia: Vec3,
    pub visc_friction: f64,
    pub coulomb_friction: f64,
    pub time_constant: Option<f64>,
    pub gyroscopic: bool,
    h: Vec3,
    tau_m: Vec3,
    tau: Vec3,
    tau_sat: [bool; 3],
    h_sat: [bool; 3],
}

impl ReactionWheelAssembly {
    pub fn new(
        tau_max: Option<AxisLimit>,
        h_max: Option<AxisLimit>,
        wheel_inertia: Vec3,
        visc_friction: f64,
        coulomb_friction: f64,
        time_constant: Option<f64>,
        gyroscopic: bool,
    ) -> Self {
        ReactionWheelAssembly {
            tau_max,
            h_max,
            wheel_inertia,
            visc_friction,
            coulomb_friction,
            time_constant,
            gyroscopic,
            h: Vec3::zeros(),
            tau_m: Vec3::zeros(),
            tau: Vec3::zeros(),
            tau_sat: [false; 3],
            h_sat: [false; 3],
        }
    }

    pub fn torque(&self) -> Vec3 {
        self.tau
    }

    pub fn momentum(&self) -> Vec3 {
        self.h
    }

    pub fn wheel_speed(&self) -> Vec3 {
        Vec3::new(
            self.h.x / self.wheel_inertia.x,
            self.h.y / self.wheel_inertia.y,
            self.h.z / self.wheel_inertia.z,
        )
    }

    pub fn torque_saturated(&self) -> [bool; 3] {
        self.tau_sat
    }

    pub fn momentum_saturated(&self) -> [bool; 3] {
        self.h_sat
    }

    /// Advance the wheels one sample. Returns the body torque applied to `RigidBody`.
    pub fn apply(&mut self, command: &Vec3, dt: f64, omega: &Vec3) -> Vec3 {
        let tau_m_des = clip_torque(&(-command), self.tau_max);
        let tau_m = match self.time_constant {
            Some(t) if t > 0.0 => {
                let alpha = (-dt / t).exp();
                clip_torque(
                    &(alpha * self.tau_m + (1.0 - alpha) * tau_m_des),
                    self.tau_max,
                )
            }
            _ => tau_m_des,
        };
        let tau_m = gate_momentum_saturation(&tau_m, &self.h, self.h_max);
        let omega_w = self.wheel_speed();
        let tau_f = friction_torque(&omega_w, self.visc_friction, self.coulomb_friction);
        let h_dot_unsat = tau_m + tau_f;
        let mut h_next = self.h + h_dot_unsat * dt;
        if self.h_max.is_some() {
            h_next = clip_torque(&h_next, self.h_max);
        }
        let h_dot = (h_next - self.h) / dt;
        let gyro = if self.gyroscopic {
            omega.cross(&self.h)
        } else {
            Vec3::zeros()
        };
        let tau_body = -h_dot - gyro;

        self.h = h_next;
        self.tau_m = tau_m;
        self.tau = tau_body;
        self.tau_sat = at_limit(command, self.tau_max);
        self.h_sat = at_limit(&self.h, self.h_max);
        tau_body
    }
}

pub fn make_reaction_wheels(
    tau_max: Option<AxisLimit>,
    h_max: Option<AxisLimit>,
    wheel_inertia: Option<AxisLimit>,
    visc_friction: f64,
    coulomb_friction: f64,
    time_constant: Option<f64>,
    gyroscopic: bool,
) -> ReactionWheelAssembly {
    let inertia = wheel_inertia.map(|w| w.as_vec3()).unwrap_or_else(|| {
        Vec3::new(
            DEFAULT_WHEEL_INERTIA,
            DEFAULT_WHEEL_INERTIA,
            DEFAULT_WHEEL_INERTIA,
        )
    });
    ReactionWheelAssembly::new(
        tau_max,
        h_max,
        inertia,
        visc_friction,
        coulomb_friction,
        time_constant,
        gyroscopic,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn matches_torque_actuator_when_unlimited_no_gyro() {
        // With gyroscopic=false and no h_max, applied tau should equal -tau_cmd's
        // rate contribution reaction i.e. tracks the box actuator on first sample.
        let mut rw = make_reaction_wheels(None, None, None, 0.0, 0.0, None, false);
        let cmd = Vec3::new(0.01, -0.02, 0.0);
        let tau = rw.apply(&cmd, 0.01, &Vec3::zeros());
        // h_dot = tau_m = -cmd (clip=None) => tau_body = -h_dot = cmd
        assert_relative_eq!(tau, cmd, epsilon = 1e-12);
    }

    #[test]
    fn per_axis_torque_never_exceeds_limit_without_gyroscopic_coupling() {
        // Matches the Python reference's own invariant (`test_reaction_wheels.py`
        // asserts this with `gyroscopic=False`): the *motor* reaction is boxed
        // to tau_max, but with gyroscopic=True the total body torque also
        // carries -omega x h_w, which is a physical consequence of body
        // rotation and is not itself clamped (see the module docs).
        let mut rw = make_reaction_wheels(
            Some(AxisLimit::Scalar(0.02)),
            None,
            None,
            0.0,
            0.0,
            None,
            false,
        );
        let cmd = Vec3::new(1.0, -1.0, 0.5);
        for _ in 0..50 {
            let tau = rw.apply(&cmd, 0.01, &Vec3::new(0.1, -0.1, 0.05));
            assert!(tau.x.abs() <= 0.02 + 1e-9);
            assert!(tau.y.abs() <= 0.02 + 1e-9);
            assert!(tau.z.abs() <= 0.02 + 1e-9);
        }
    }

    #[test]
    fn gyroscopic_coupling_can_exceed_motor_torque_limit() {
        let mut rw = make_reaction_wheels(
            Some(AxisLimit::Scalar(0.02)),
            None,
            None,
            0.0,
            0.0,
            None,
            true,
        );
        let cmd = Vec3::new(1.0, -1.0, 0.5);
        // Build up wheel momentum first so omega x h is non-negligible.
        for _ in 0..2000 {
            rw.apply(&cmd, 0.01, &Vec3::new(0.1, -0.1, 0.05));
        }
        let tau = rw.apply(&cmd, 0.01, &Vec3::new(0.1, -0.1, 0.05));
        assert!(
            tau.norm() > 0.02,
            "expected gyroscopic coupling to push |tau| above tau_max, got {}",
            tau.norm()
        );
    }

    #[test]
    fn momentum_saturates_at_h_max() {
        let mut rw = make_reaction_wheels(
            None,
            Some(AxisLimit::Scalar(0.01)),
            None,
            0.0,
            0.0,
            None,
            false,
        );
        let cmd = Vec3::new(-1.0, 0.0, 0.0); // drives h positive continuously
        for _ in 0..10000 {
            rw.apply(&cmd, 0.01, &Vec3::zeros());
        }
        assert!(rw.momentum().x <= 0.01 + 1e-9);
        assert!(rw.momentum().x >= 0.01 - 1e-6);
    }
}

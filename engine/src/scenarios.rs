//! Named SimLab closed-loop scenario pack, ported from `attitude_sim.scenarios`.
//!
//! Plant/controller/estimator cores are unchanged; this module only names
//! initial conditions, attitude commands, default durations/controllers, and
//! the `hold` environmental-torque preset.

use crate::disturbances::{CircularOrbit, DipoleModel, EnvironmentalTorques};
use crate::quaternion::{axis_angle_to_quat, Quat, Vec3};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Scenario {
    Slew,
    Detumble,
    Hold,
    Eigenaxis,
}

impl Scenario {
    pub const ALL: [Scenario; 4] = [
        Scenario::Slew,
        Scenario::Detumble,
        Scenario::Hold,
        Scenario::Eigenaxis,
    ];

    pub fn key(&self) -> &'static str {
        match self {
            Scenario::Slew => "slew",
            Scenario::Detumble => "detumble",
            Scenario::Hold => "hold",
            Scenario::Eigenaxis => "eigenaxis",
        }
    }

    pub fn title(&self) -> &'static str {
        match self {
            Scenario::Slew => "Rest-to-rest slew",
            Scenario::Detumble => "Detumble to rest",
            Scenario::Hold => "Hold under environmental torques",
            Scenario::Eigenaxis => "Eigenaxis slew (LQR)",
        }
    }

    pub fn blurb(&self) -> &'static str {
        match self {
            Scenario::Slew => "75 deg rest-to-rest slew about a skewed body axis (default PID)",
            Scenario::Detumble => "tumbling initial rate, dump omega and recover identity (default PID)",
            Scenario::Hold => "identity hold under EnvironmentalTorques (GG + demo-scale residual dipole, default PID)",
            Scenario::Eigenaxis => "principal-axis (body z) rest-to-rest slew (default LQR, angle default 30 deg)",
        }
    }

    pub fn default_t_final(&self) -> f64 {
        match self {
            Scenario::Slew => 40.0,
            Scenario::Detumble => 30.0,
            Scenario::Hold => 30.0,
            Scenario::Eigenaxis => 20.0,
        }
    }

    pub fn default_controller(&self) -> &'static str {
        match self {
            Scenario::Eigenaxis => "lqr",
            _ => "pid",
        }
    }

    pub fn uses_env_by_default(&self) -> bool {
        matches!(self, Scenario::Hold)
    }

    pub fn default_angle_deg(&self, angle_deg: Option<f64>) -> f64 {
        if let Some(a) = angle_deg {
            return a;
        }
        match self {
            Scenario::Eigenaxis => EIGENAXIS_ANGLE_DEG,
            _ => SLEW_ANGLE_DEG,
        }
    }
}

pub const SLEW_ANGLE_DEG: f64 = 75.0;
pub const EIGENAXIS_ANGLE_DEG: f64 = 30.0;

fn slew_axis() -> Vec3 {
    Vec3::new(0.2, 0.5, 0.84)
}
fn detumble_q0_axis() -> Vec3 {
    Vec3::new(0.4, 0.2, 0.9)
}
fn detumble_omega0() -> Vec3 {
    Vec3::new(0.55, -0.40, 0.30)
}
fn eigenaxis_axis() -> Vec3 {
    Vec3::new(0.0, 0.0, 1.0)
}

pub const HOLD_ORBIT_RADIUS_M: f64 = 7.0e6;
pub const HOLD_ORBIT_INCLINATION_DEG: f64 = 51.6;
const HOLD_ARG_LATITUDE0_DEG: f64 = 40.0;

pub fn hold_residual_dipole_a_m2() -> Vec3 {
    Vec3::new(40.0, -15.0, 8.0)
}

pub fn default_hold_orbit() -> CircularOrbit {
    CircularOrbit::new(
        HOLD_ORBIT_RADIUS_M,
        HOLD_ORBIT_INCLINATION_DEG.to_radians(),
        0.0,
    )
    .with_arg_latitude0(HOLD_ARG_LATITUDE0_DEG.to_radians())
}

/// `(q0, omega0, q_des)` for a named scenario.
pub fn scenario_state(scenario: Scenario, angle_deg: Option<f64>) -> (Quat, Vec3, Quat) {
    let ang = scenario.default_angle_deg(angle_deg);
    match scenario {
        Scenario::Detumble => (
            axis_angle_to_quat(&detumble_q0_axis(), 40f64.to_radians()),
            detumble_omega0(),
            Quat::IDENTITY,
        ),
        Scenario::Hold => (Quat::IDENTITY, Vec3::zeros(), Quat::IDENTITY),
        Scenario::Eigenaxis => (
            Quat::IDENTITY,
            Vec3::zeros(),
            axis_angle_to_quat(&eigenaxis_axis(), ang.to_radians()),
        ),
        Scenario::Slew => (
            Quat::IDENTITY,
            Vec3::zeros(),
            axis_angle_to_quat(&slew_axis(), ang.to_radians()),
        ),
    }
}

/// Gravity-gradient + residual-dipole pack used by `Scenario::Hold`.
pub fn make_hold_environmental_torques(inertia: &crate::quaternion::Mat3) -> EnvironmentalTorques {
    EnvironmentalTorques {
        gravity_gradient: Some(*inertia),
        residual_dipole: Some((hold_residual_dipole_a_m2(), DipoleModel::Tilted)),
        ..EnvironmentalTorques::none()
    }
}

//! `attitude-engine`: a Rust port of the `attitude_sim` Python package
//! (kept at `legacy-python/`) -- 6DOF rigid-body attitude dynamics,
//! control (PID/LQR), and estimation (MEKF/Mahony/TRIAD), plus the
//! SimLab closed-loop orchestration that powers the web app.
//!
//! Module layout mirrors the Python package one file at a time so the two
//! can be diffed side by side:
//!
//! | Rust module      | Python module                  |
//! |-------------------|---------------------------------|
//! | `quaternion`       | `attitude_sim.quaternions`      |
//! | `plant`            | `attitude_sim.plant`            |
//! | `controls`         | `attitude_sim.controls`         |
//! | `actuators`        | `attitude_sim.actuators`        |
//! | `reaction_wheels`  | `attitude_sim.reaction_wheels`  |
//! | `disturbances`     | `attitude_sim.disturbances`     |
//! | `sensors`          | `attitude_sim.sensors`          |
//! | `estimation`       | `attitude_sim.estimation`       |
//! | `scenarios`        | `attitude_sim.scenarios`        |
//! | `sim`              | `attitude_sim.sim`              |
//!
//! Deliberately not ported (see `README.md` for the full rationale): the
//! RKMK4 Lie-group plant step, gyrostat/flex/polhode library extras, the
//! magnetic-torquer library module, QUEST/Davenport coarse init and the
//! star-tracker sensor, MRP charting, and the Monte Carlo harness. None of
//! these are reachable from the documented SimLab CLI surface that this
//! app reproduces.

pub mod actuators;
pub mod controls;
pub mod disturbances;
pub mod estimation;
pub mod plant;
pub mod quaternion;
pub mod reaction_wheels;
pub mod scenarios;
pub mod sensors;
pub mod sim;

pub use sim::{run_sim, scenario_catalog, SimError, SimLog, SimRequest};

//! Cross-validates the Rust engine against golden trajectories captured
//! from the Python reference implementation (`legacy-python/`) for every
//! `--estimator truth` configuration in
//! `engine/tests/golden/generate_golden.py`. Truth feedback never samples a
//! sensor, so both engines are exercising the exact same deterministic
//! math (RK4 plant, PID/LQR, actuators, reaction wheels, disturbances) and
//! should agree far tighter than any physically meaningful tolerance.
//!
//! Regenerate fixtures after an intentional engine change with:
//! `pip install -e legacy-python[dev] && python3 engine/tests/golden/generate_golden.py`

use attitude_engine::sim::{run_sim, SimRequest};
use serde_json::Value;
use std::fs;
use std::path::PathBuf;

fn golden_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden")
        .join(format!("{name}.json"))
}

fn load_case(name: &str) -> (SimRequest, Value) {
    let text = fs::read_to_string(golden_path(name))
        .unwrap_or_else(|e| panic!("reading golden fixture {name}: {e}"));
    let value: Value = serde_json::from_str(&text).unwrap();
    let req: SimRequest = serde_json::from_value(value["request"].clone())
        .unwrap_or_else(|e| panic!("parsing request for {name}: {e}"));
    (req, value)
}

fn as_f64_matrix(v: &Value) -> Vec<Vec<f64>> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|row| {
            row.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
                .collect()
        })
        .collect()
}

fn as_f64_vec(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

fn assert_matrix_close(
    case: &str,
    field: &str,
    got: &[impl AsRef<[f64]>],
    want: &[Vec<f64>],
    tol: f64,
) {
    assert_eq!(
        got.len(),
        want.len(),
        "[{case}] {field}: sample count mismatch"
    );
    let mut max_diff = 0.0f64;
    let mut worst = (0usize, 0usize);
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        let g = g.as_ref();
        assert_eq!(g.len(), w.len(), "[{case}] {field}[{i}]: width mismatch");
        for (j, (&gv, &wv)) in g.iter().zip(w.iter()).enumerate() {
            let diff = (gv - wv).abs();
            if diff > max_diff {
                max_diff = diff;
                worst = (i, j);
            }
        }
    }
    assert!(
        max_diff < tol,
        "[{case}] {field}: max abs diff {max_diff:e} at sample {} component {} exceeds tol {tol:e}",
        worst.0,
        worst.1
    );
}

fn assert_vec_close(case: &str, field: &str, got: &[f64], want: &[f64], tol: f64) {
    assert_eq!(
        got.len(),
        want.len(),
        "[{case}] {field}: sample count mismatch"
    );
    let mut max_diff = 0.0f64;
    let mut worst = 0usize;
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        let diff = (g - w).abs();
        if diff > max_diff {
            max_diff = diff;
            worst = i;
        }
    }
    assert!(
        max_diff < tol,
        "[{case}] {field}: max abs diff {max_diff:e} at sample {worst} exceeds tol {tol:e}"
    );
}

/// Run one golden case through the Rust engine and compare every logged
/// series against the Python fixture. `tol` covers state trajectories
/// (q/omega/tau/euler/att_error); torque-derived quantities that
/// differentiate a lagged/near-zero signal (h_wheel, omega_wheel, tau_env)
/// get a slightly looser multiple since they amplify the same underlying
/// sub-ULP-scale numerical noise.
fn check_case(name: &str, tol: f64) {
    let (req, golden) = load_case(name);
    let log = run_sim(&req).unwrap_or_else(|e| panic!("[{name}] run_sim failed: {e}"));

    let q: Vec<[f64; 4]> = log.q;
    let omega: Vec<[f64; 3]> = log.omega;
    let tau: Vec<[f64; 3]> = log.tau;
    let euler: Vec<[f64; 3]> = log.euler_deg;
    let tau_env: Vec<[f64; 3]> = log.tau_env;
    let h_wheel: Vec<[f64; 3]> = log.h_wheel;

    assert_matrix_close(name, "q", &q, &as_f64_matrix(&golden["q"]), tol);
    assert_matrix_close(name, "omega", &omega, &as_f64_matrix(&golden["omega"]), tol);
    assert_matrix_close(name, "tau", &tau, &as_f64_matrix(&golden["tau"]), tol);
    assert_matrix_close(
        name,
        "euler_deg",
        &euler,
        &as_f64_matrix(&golden["euler_deg"]),
        tol * 1e4,
    );
    assert_vec_close(
        name,
        "att_error_deg",
        &log.att_error_deg,
        &as_f64_vec(&golden["att_error_deg"]),
        tol * 1e4,
    );
    assert_matrix_close(
        name,
        "tau_env",
        &tau_env,
        &as_f64_matrix(&golden["tau_env"]),
        tol.max(1e-9),
    );
    assert_matrix_close(
        name,
        "h_wheel",
        &h_wheel,
        &as_f64_matrix(&golden["h_wheel"]),
        tol.max(1e-9),
    );

    let final_err_want = golden["final_att_error_deg"].as_f64().unwrap();
    assert!(
        (log.summary.final_att_error_deg - final_err_want).abs() < tol * 1e4,
        "[{name}] final_att_error_deg: got {} want {}",
        log.summary.final_att_error_deg,
        final_err_want
    );
}

// PID cases: closed-form gains, no iterative solver anywhere in the loop,
// so these should (and do) agree with Python to near machine precision.
const PID_TOL: f64 = 1e-9;
// LQR cases: the Riccati-via-integration CARE solver agrees with SciPy's
// `solve_continuous_are` to ~1e-10 on K itself (see controls.rs docs); a
// few hundred closed-loop RK4 steps can amplify that slightly, so this is
// still an extremely tight physical tolerance, not a loosened-for-convenience one.
const LQR_TOL: f64 = 1e-6;

#[test]
fn slew_pid_truth_matches_python() {
    check_case("slew_pid_truth", PID_TOL);
}

#[test]
fn slew_lqr_truth_matches_python() {
    check_case("slew_lqr_truth", LQR_TOL);
}

#[test]
fn detumble_pid_truth_matches_python() {
    check_case("detumble_pid_truth", PID_TOL);
}

#[test]
fn hold_pid_truth_matches_python() {
    check_case("hold_pid_truth", PID_TOL);
}

#[test]
fn eigenaxis_lqr_truth_matches_python() {
    check_case("eigenaxis_lqr_truth", LQR_TOL);
}

#[test]
fn eigenaxis_custom_angle_matches_python() {
    check_case("eigenaxis_custom_angle", LQR_TOL);
}

#[test]
fn slew_tau_dist_matches_python() {
    check_case("slew_tau_dist", PID_TOL);
}

#[test]
fn slew_actuator_lag_matches_python() {
    check_case("slew_actuator_lag", PID_TOL);
}

#[test]
fn slew_reaction_wheels_matches_python() {
    check_case("slew_reaction_wheels", PID_TOL);
}

#[test]
fn slew_actuator_dump_matches_python() {
    check_case("slew_actuator_dump", PID_TOL);
}

#[test]
fn hold_aero_srp_matches_python() {
    check_case("hold_aero_srp", PID_TOL);
}

#[test]
fn slew_env_gg_dipole_matches_python() {
    check_case("slew_env_gg_dipole", PID_TOL);
}

#[test]
fn slew_no_env_override_matches_python() {
    check_case("slew_no_env_override", PID_TOL);
}

#[test]
fn eigenaxis_srp_cylindrical_matches_python() {
    check_case("eigenaxis_srp_cylindrical", LQR_TOL);
}

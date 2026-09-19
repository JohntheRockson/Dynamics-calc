# SimLab — Attitude Control Console

An interactive 6DOF rigid-body **attitude dynamics, control, and estimation** simulator: a Rust physics/controls engine behind a small HTTP API, driving a React + Three.js console in the browser. Pick a scenario, tune the spacecraft/controller/disturbances, hit **Run**, and watch the satellite fly while telemetry charts update live.

This is a from-scratch **Rust + web rewrite** of a Python/NumPy simulator (`legacy-python/`, kept as the reference implementation — see [Relationship to the Python reference](#relationship-to-the-python-reference)).

## Quickstart

```bash
# 1. Build the frontend once (or after any web/ change)
cd web && npm install && npm run build && cd ..

# 2. Run the server — serves the API *and* the built frontend on one port
cargo run --release -p attitude-server
```

Open **http://localhost:8080**. `make build && make run` does the same two steps.

Dev loop (hot-reloading frontend, separate terminals):

```bash
cargo run -p attitude-server          # API on :8080
cd web && npm run dev                 # Vite dev server on :5173, proxies /api to :8080
```

## Architecture

```
sensors → estimator (MEKF / Mahony / truth)
              ↓
        controller (PID | LQR) → τ_cmd
              ↓
        actuator (clip + lag, or reaction-wheel assembly)  → τ
              ↓
        + τ_d + τ_env (gravity-gradient / dipole / aero / SRP)
              ↓
        rigid-body plant (RK4)  →  q, ω
              ↓
            sensors
```

| Path | Role |
| --- | --- |
| `engine/` | Pure Rust library: quaternion kinematics, Euler dynamics + RK4, PID/LQR control, MEKF/Mahony estimation, TRIAD coarse init, actuators + reaction wheels, environmental disturbances, scenario presets, and the `run_sim` closed-loop orchestration. No I/O. |
| `server/` | `axum` HTTP API (`GET /api/scenarios`, `POST /api/simulate`) wrapping `engine::run_sim`, plus static-file serving for the built frontend so the whole app is one binary on one port. |
| `web/` | React + TypeScript + Three.js console: a control panel for every simulation parameter, a live 3D attitude viewer, canvas telemetry charts, and playback transport. |
| `legacy-python/` | The original Python/NumPy reference implementation (`attitude_sim`), kept as the correctness oracle — see below. |

There is no session/streaming state: one request runs a few thousand RK4 steps (sub-millisecond in Rust) and returns the full trajectory as JSON; the frontend animates and charts it client-side. That's simple, and — measured, not assumed — plenty fast: the default 40 s slew round-trips in **~15 ms**, and the heaviest request the UI can construct (finest `dt`, longest duration, every disturbance/estimator/actuator feature on at once) round-trips in **~300 ms** for a **~2.4 MB gzip-compressed** payload (see [Performance notes](#performance-notes)).

## The app

**Scenarios** (`attitude_sim.scenarios` → `engine::scenarios`):

| Scenario | What it is | Default duration | Default controller |
| --- | --- | --- | --- |
| Slew | 75° rest-to-rest slew about a skewed body axis | 40 s | PID |
| Detumble | Tumbling initial rate, dump ω and recover identity | 30 s | PID |
| Hold | Identity hold under gravity-gradient + demo-scale residual dipole | 30 s | PID |
| Eigenaxis | Rest-to-rest principal-axis (body *z*) slew | 20 s | LQR |

**Control panel** (left sidebar) exposes essentially everything the original CLI did, plus a few extras the engine already supported but the CLI never wired up:

- **Scenario** — pick a preset, commanded angle, duration, sample step, RNG seed.
- **Guidance & control** — controller (auto/PID/LQR), estimator (truth/MEKF/Mahony), coarse TRIAD init, principal inertia, and *(bonus, not CLI-exposed)* gain scale / torque limit / PID ωn,ζ sliders.
- **Disturbances** — constant body torque, gravity-gradient, residual dipole (tilted or orbit-normal Earth field), aerodynamic drag, solar radiation pressure (with a cylindrical-Earth-shadow eclipse model), and the shared circular-orbit geometry.
- **Actuator** — per-axis wheel torque limit, first-order lag, a full reaction-wheel assembly (momentum storage, viscous/Coulomb friction, ω×h gyroscopic coupling), and momentum dump.
- **Sensors** — gyro ARW/RRW noise densities, magnetometer/sun-sensor noise and enable toggles, sun eclipse.

**Results** — a live 3D viewer (satellite mesh with RGB body axes, a cyan wireframe of the commanded attitude, and an amber wireframe of the estimator's attitude when running MEKF/Mahony), playback transport (play/pause/scrub/speed), run-summary stat cards, and tabbed telemetry charts (attitude error, Euler angles, body rate, control torque, reaction-wheel state, and MEKF innovation NIS with its theoretical χ² mean reference line).

## Testing

```bash
cargo test --workspace     # 57 tests: unit tests + Python-parity integration tests
cargo clippy --workspace --all-targets
cd web && npx tsc -b && npm run lint
```

### Cross-validation against the Python reference

The whole point of a rewrite is that it computes the *same physics*. `engine/tests/python_parity.rs` loads golden trajectories captured from `legacy-python` (`engine/tests/golden/generate_golden.py`) for every named scenario, both controllers, and every disturbance/actuator/reaction-wheel combination, and asserts the Rust engine reproduces them:

- **PID paths: agree to ~1e-9** (closed-form gains, no iterative solver anywhere in the loop).
- **LQR paths: agree to ~1e-6.** The one deliberate numerical difference from the Python reference is `engine::controls::solve_care`: instead of a LAPACK/SciPy Riccati solver (which would need a system BLAS/LAPACK dependency), it integrates the associated matrix Riccati *differential* equation to steady state with the same RK4 already trusted for the plant. That's a standard, textbook-legitimate way to get the stabilizing CARE solution, and empirically it matches `scipy.linalg.solve_continuous_are` to **~10 significant figures** on this problem (checked by hand during development; `care_residual` is asserted near-zero in `engine/src/controls.rs` tests).

All of this uses `--estimator truth` (no sensor sampling). **MEKF/Mahony are not bit-parity-tested against Python** — the noisy paths use `rand_chacha::ChaCha8Rng` instead of NumPy's PCG64, since there's no dependency-free way to reproduce NumPy's exact bit stream in Rust. Every model (ARW/RRW scaling, filter equations, TRIAD geometry, FOV/eclipse gating) is still a faithful line-for-line port; it's validated with Rust-side unit tests (filter convergence, covariance behavior, TRIAD exactness) rather than cross-language bit-parity. Given the same seed, the app is fully reproducible in its own right — just with a different (equally valid) noise realization than the Python CLI would draw.

Regenerate the golden fixtures after an intentional engine change:

```bash
pip install -e legacy-python[dev]
python3 engine/tests/golden/generate_golden.py
```

## Performance notes

Measured on this dev box (release build), not assumed:

| Request | Samples | Round trip | Gzip payload |
| --- | --- | --- | --- |
| Default slew (`estimator=truth`) | 4,001 | ~15 ms | ~250 KB |
| Default slew (`estimator=mekf`) | 4,001 | ~30 ms | ~1.0 MB |
| Heaviest UI-reachable request (finest dt, longest duration, MEKF, aero+SRP+GG+dipole, reaction wheels) | 8,000 | ~300 ms | ~2.4 MB |

Two things make that possible:

1. **A sample budget, not just a compute budget.** The Rust computation itself stays fast even at far larger sample counts, so the real constraint for "feels instant" is JSON payload size, not CPU. `engine::sim::MAX_SAMPLES = 8_000` keeps every response a few MB at most; the frontend (`web/src/limits.ts`) mirrors that budget so the duration slider's range shrinks automatically as `dt` gets finer, and a scenario switch re-clamps the same way — the UI cannot construct a request the server would reject.
2. **Gzip on API responses.** `attitude-server` wraps the whole router in `tower_http::compression::CompressionLayer`, which shrinks the (very repetitive, highly compressible) float-heavy JSON substantially — the table above already reflects post-compression sizes.

The frontend also code-splits the Three.js viewer (`React.lazy`) so the control panel and charts are interactive before the ~140 KB gzipped 3D chunk finishes loading.

## Scope: what was ported, and what wasn't

Ported (this *is* the SimLab CLI's feature set, faithfully): quaternion kinematics, classical RK4 rigid-body plant, PID with gated anti-windup, LQR via CARE, MEKF, Mahony complementary filter, TRIAD coarse init, gyro + magnetometer + sun-sensor models, per-axis actuator clip/lag/dump, the full reaction-wheel assembly, gravity-gradient / residual-dipole / aerodynamic / SRP disturbances, and all four named scenarios.

Deliberately **not** ported, because the original CLI never exposed them either (so the app loses nothing relative to `python -m attitude_sim`) or they're research-library extras:

- The optional Munthe–Kaas **RKMK4** Lie-group plant integrator (`step_rigid_body(..., method="rkmk4")` in Python) — the CLI always used classical RK4.
- **QUEST / Davenport** q-method coarse init and the **star-tracker** sensor stub — the CLI's `--coarse-init` always defaulted to TRIAD; QUEST/Davenport and the star tracker are `attitude_sim` library surface the CLI didn't wire up. TRIAD is ported.
- **Gyrostat, hinged-appendage/flex, polhode/energy-Casimir, magnetic-torquer** — all explicitly "library plant, not a SimLab CLI flag" in the Python docstrings.
- **Monte Carlo sweep** (`attitude_sim.monte_carlo`) — a batch statistics tool; out of scope for a single-run interactive app. A future iteration could add it as a "sweep" mode.
- **MRP charting** (`--mrp-plot`) — a post-process chart derived from logged quaternions; superseded by the live Euler/rate/error charts.

## Relationship to the Python reference

`legacy-python/` is the original implementation this app replaces as the primary product, kept in the repo because it's (a) the correctness oracle the Rust engine is validated against, (b) the source of the golden fixtures, and (c) a complete, independently useful CLI/library with its own docs, tests, and committed figures — see `legacy-python/README.md`. It is not being actively extended going forward.

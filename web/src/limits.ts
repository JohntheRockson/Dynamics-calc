// Mirrors `engine::sim::MAX_SAMPLES` (engine/src/sim.rs). Kept in sync by
// hand since the two live in different languages/build systems; the server
// is the source of truth and will reject anything this constant lets
// through incorrectly, so a drift here only loosens/tightens the client-side
// UX guard, not correctness.
export const MAX_SAMPLES = 8000

// The server computes `n = round(t_final / dt) + 1` and rejects n >
// MAX_SAMPLES (see `run_sim` in engine/src/sim.rs), so the naive inverse
// `MAX_SAMPLES * dt` is exactly one sample over budget at the boundary
// (e.g. dt=0.002, t_final=16 -> n=8001). Independent JS-vs-Rust
// floating-point rounding of `t_final / dt` could also disagree by a ULP
// right at a boundary. Reserve a small margin so the client's suggested
// cap is always strictly inside the server's actual limit.
const SAMPLE_MARGIN = 8

/** Longest `t_final` (s) that stays within the sample budget at this `dt`. */
export function maxTFinalForDt(dt: number): number {
  return Math.floor((MAX_SAMPLES - SAMPLE_MARGIN) * dt)
}

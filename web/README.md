# SimLab web frontend

React + TypeScript + Three.js console for the `attitude-server` API. See the [repository root README](../README.md) for the full picture (architecture, quickstart, testing).

```bash
npm install
npm run dev      # Vite dev server on :5173, proxies /api/* to :8080
npm run build    # production build -> dist/ (served by attitude-server)
npm run lint     # oxlint
```

## Layout

- `src/types.ts` / `src/api.ts` — wire types and fetch wrappers matching the Rust `SimRequest`/`SimLog` JSON exactly (field names included), so the two stay easy to diff against `engine/src/sim.rs`.
- `src/limits.ts` — mirrors `engine::sim::MAX_SAMPLES` so the duration/dt controls can never construct a request the server would reject.
- `src/components/sections/*` — one file per control-panel section (Scenario, Guidance & Control, Disturbances, Actuator, Sensors).
- `src/components/Viewer3D/` — the Three.js satellite scene (code-split via `React.lazy`, since it's most of the bundle size).
- `src/components/Charts/` — a small dependency-free canvas line-chart component plus the tabbed telemetry panel built on top of it.

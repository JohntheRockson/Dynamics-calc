import { parseExpr, type CompiledExpr } from './expr'
import type { RectilinearResult } from './physics'

const EPS = 1e-10
const RESIDUAL_TOL = 1e-4

export interface SolverKnowns {
  s0: number
  v0: number | null
  s: number | null
  v: number | null
  a: number | null
  t: number | null
}

export interface SolverPoint {
  t: number
  s: number
  v: number
  a: number
}

export interface SolverOk {
  ok: true
  primary: SolverPoint
  alternatives: SolverPoint[]
  method: string[]
  notes: string[]
  computed: { t: boolean; s: boolean; v: boolean; a: boolean }
  history: RectilinearResult
}

export interface SolverErr {
  ok: false
  error: string
}

export type SolverResult = SolverOk | SolverErr

export type RelationKind = 'const-a' | 'a-expr' | 'v-of-t' | 'v-of-s'

export type TargetKind = 't' | 's' | 'v'

export interface VariableInput {
  kind: 'a-expr' | 'v-of-t' | 'v-of-s'
  expr: CompiledExpr
  t0: number
  s0: number
  v0: number
  target: { kind: TargetKind; value: number }
  tHorizon: number
}

function nearly(a: number, b: number, rel = RESIDUAL_TOL): boolean {
  return Math.abs(a - b) <= rel * (1 + Math.abs(a) + Math.abs(b))
}

function finite(n: number): boolean {
  return Number.isFinite(n)
}

interface Five {
  ds: number
  u: number
  v: number
  a: number
  t: number
}

type Slot = 'ds' | 'u' | 'v' | 'a' | 't'

function residual(got: Five, given: { ds: number | null; u: number | null; v: number | null; a: number | null; t: number | null }): number {
  let max = 0
  const pairs: [number, number | null][] = [
    [got.ds, given.ds],
    [got.u, given.u],
    [got.v, given.v],
    [got.a, given.a],
    [got.t, given.t],
  ]
  for (const [g, k] of pairs) {
    if (k === null) continue
    const scale = 1 + Math.abs(k) + Math.abs(g)
    max = Math.max(max, Math.abs(g - k) / scale)
  }
  return max
}

function solveTrio(ds: number | null, u: number | null, v: number | null, a: number | null, t: number | null): { sols: Five[]; error?: string } {
  const U = u !== null
  const V = v !== null
  const A = a !== null
  const T = t !== null
  const S = ds !== null
  const n = [U, V, A, T, S].filter(Boolean).length
  if (n !== 3) return { sols: [], error: 'internal: trio must have 3 knowns' }

  const out: Five[] = []

  const push = (sol: Five) => {
    if ([sol.ds, sol.u, sol.v, sol.a, sol.t].every(finite)) out.push(sol)
  }

  // u, a, t
  if (U && A && T) {
    push({ ds: u * t + 0.5 * a * t * t, u, v: u + a * t, a, t })
    return { sols: out }
  }

  // u, v, t
  if (U && V && T) {
    if (Math.abs(t) < EPS) {
      if (!nearly(u, v)) return { sols: [], error: 't = 0 requires v = v₀' }
      return { sols: [], error: 't = 0 leaves acceleration undetermined — give a or a displacement' }
    }
    const aa = (v - u) / t
    push({ ds: 0.5 * (u + v) * t, u, v, a: aa, t })
    return { sols: out }
  }

  // u, v, a
  if (U && V && A) {
    if (Math.abs(a) < EPS) {
      if (!nearly(u, v)) return { sols: [], error: 'a = 0 requires v = v₀ (no speed change)' }
      return { sols: [], error: 'a = 0 and v = v₀: time is undetermined without position' }
    }
    const tt = (v - u) / a
    push({ ds: u * tt + 0.5 * a * tt * tt, u, v, a, t: tt })
    return { sols: out }
  }

  // u, v, ds
  if (U && V && S) {
    if (Math.abs(ds) < EPS) {
      if (nearly(u, v)) return { sols: [], error: 'no displacement and no speed change: a and t are not unique (t = 0 is trivial)' }
      return { sols: [], error: 'returning to the same position with v = −v₀ leaves a and t underdetermined' }
    }
    const aa = (v * v - u * u) / (2 * ds)
    let tt: number
    if (Math.abs(aa) < EPS) {
      if (Math.abs(u) < EPS) return { sols: [], error: 'a = 0 and v₀ = 0 cannot produce a displacement' }
      tt = ds / u
    } else {
      tt = (v - u) / aa
    }
    push({ ds, u, v, a: aa, t: tt })
    return { sols: out }
  }

  // u, a, ds
  if (U && A && S) {
    if (Math.abs(a) < EPS) {
      if (Math.abs(u) < EPS) {
        if (Math.abs(ds) < EPS) return { sols: [], error: 'particle is at rest at s₀ — t is undetermined' }
        return { sols: [], error: 'a = 0 and v₀ = 0 cannot produce a displacement' }
      }
      const tt = ds / u
      push({ ds, u, v: u, a: 0, t: tt })
      return { sols: out }
    }
    const disc = u * u + 2 * a * ds
    if (disc < -1e-12) {
      return { sols: [], error: `no real motion: v² = v₀² + 2aΔs = ${disc.toPrecision(4)} < 0` }
    }
    const w = Math.sqrt(Math.max(0, disc))
    const tPlus = (-u + w) / a
    const tMinus = (-u - w) / a
    push({ ds, u, v: w, a, t: tPlus })
    if (w > EPS) push({ ds, u, v: -w, a, t: tMinus })
    if (!out.length) return { sols: [], error: 'no finite time for this (v₀, a, Δs)' }
    return { sols: out }
  }

  // u, t, ds
  if (U && T && S) {
    if (Math.abs(t) < EPS) {
      if (!nearly(ds, 0)) return { sols: [], error: 't = 0 requires Δs = 0' }
      return { sols: [], error: 't = 0 leaves acceleration undetermined' }
    }
    const aa = (2 * (ds - u * t)) / (t * t)
    push({ ds, u, v: u + aa * t, a: aa, t })
    return { sols: out }
  }

  // v, a, t
  if (V && A && T) {
    const uu = v - a * t
    push({ ds: uu * t + 0.5 * a * t * t, u: uu, v, a, t })
    return { sols: out }
  }

  // v, a, ds
  if (V && A && S) {
    if (Math.abs(a) < EPS) {
      if (Math.abs(v) < EPS) {
        if (Math.abs(ds) < EPS) return { sols: [], error: 'at rest with a = 0 — t is undetermined' }
        return { sols: [], error: 'a = 0 and v = 0 cannot produce a displacement' }
      }
      const uu = v
      const tt = ds / v
      push({ ds, u: uu, v, a: 0, t: tt })
      return { sols: out }
    }
    const disc = v * v - 2 * a * ds
    if (disc < -1e-12) {
      return { sols: [], error: `no real motion: v₀² = v² − 2aΔs = ${disc.toPrecision(4)} < 0` }
    }
    const w = Math.sqrt(Math.max(0, disc))
    const candidates = w < EPS ? [0] : [w, -w]
    for (const uu of candidates) {
      const tt = (v - uu) / a
      push({ ds, u: uu, v, a, t: tt })
    }
    if (!out.length) return { sols: [], error: 'no finite time for this (v, a, Δs)' }
    return { sols: out }
  }

  // v, t, ds
  if (V && T && S) {
    if (Math.abs(t) < EPS) {
      if (!nearly(ds, 0)) return { sols: [], error: 't = 0 requires Δs = 0' }
      return { sols: [], error: 't = 0 leaves v₀ and a undetermined' }
    }
    const uu = (2 * ds) / t - v
    const aa = (v - uu) / t
    push({ ds, u: uu, v, a: aa, t })
    return { sols: out }
  }

  // a, t, ds
  if (A && T && S) {
    if (Math.abs(t) < EPS) {
      if (!nearly(ds, 0)) return { sols: [], error: 't = 0 requires Δs = 0' }
      return { sols: [], error: 't = 0 leaves v₀ and v undetermined' }
    }
    const uu = ds / t - 0.5 * a * t
    push({ ds, u: uu, v: uu + a * t, a, t })
    return { sols: out }
  }

  return { sols: [], error: 'internal: unhandled trio' }
}

function constAMethod(slots: Slot[]): string[] {
  const set = new Set(slots)
  const has = (k: Slot) => set.has(k)
  if (has('u') && has('a') && has('t')) {
    return ['v = v_0 + a t', 's = s_0 + v_0 t + \\tfrac12 a t^2']
  }
  if (has('u') && has('v') && has('t')) {
    return ['a = \\dfrac{v - v_0}{t}', 's = s_0 + \\dfrac{v_0 + v}{2} t']
  }
  if (has('u') && has('v') && has('a')) {
    return ['t = \\dfrac{v - v_0}{a}', 's = s_0 + v_0 t + \\tfrac12 a t^2']
  }
  if (has('u') && has('v') && has('ds')) {
    return ['v^2 = v_0^2 + 2 a (s - s_0)', 't = \\dfrac{v - v_0}{a}']
  }
  if (has('u') && has('a') && has('ds')) {
    return ['v^2 = v_0^2 + 2 a (s - s_0)', '\\tfrac12 a t^2 + v_0 t - (s - s_0) = 0']
  }
  if (has('u') && has('t') && has('ds')) {
    return ['s = s_0 + v_0 t + \\tfrac12 a t^2 \\ \\Rightarrow\\ a = \\dfrac{2\\big((s-s_0) - v_0 t\\big)}{t^2}', 'v = v_0 + a t']
  }
  if (has('v') && has('a') && has('t')) {
    return ['v_0 = v - a t', 's = s_0 + v_0 t + \\tfrac12 a t^2']
  }
  if (has('v') && has('a') && has('ds')) {
    return ['v_0^2 = v^2 - 2 a (s - s_0)', 't = \\dfrac{v - v_0}{a}']
  }
  if (has('v') && has('t') && has('ds')) {
    return ['s = s_0 + \\dfrac{v_0 + v}{2} t \\ \\Rightarrow\\ v_0 = \\dfrac{2(s-s_0)}{t} - v', 'a = \\dfrac{v - v_0}{t}']
  }
  if (has('a') && has('t') && has('ds')) {
    return ['s = s_0 + v_0 t + \\tfrac12 a t^2 \\ \\Rightarrow\\ v_0 = \\dfrac{s-s_0}{t} - \\tfrac12 a t', 'v = v_0 + a t']
  }
  return ['v = v_0 + a t', 's = s_0 + v_0 t + \\tfrac12 a t^2', 'v^2 = v_0^2 + 2 a (s - s_0)']
}

function orientHistory(h: RectilinearResult): RectilinearResult {
  if (h.t.length < 2 || h.t[h.t.length - 1] >= h.t[0]) return h
  return { t: h.t.slice().reverse(), s: h.s.slice().reverse(), v: h.v.slice().reverse(), a: h.a.slice().reverse() }
}

function historyConstA(s0: number, u: number, a: number, tEnd: number, steps = 400): RectilinearResult {
  const t: number[] = []
  const s: number[] = []
  const v: number[] = []
  const aa: number[] = []
  const t0 = 0
  const t1 = tEnd
  if (!finite(t1) || Math.abs(t1 - t0) < 1e-15) {
    return { t: [0], s: [s0], v: [u], a: [a] }
  }
  for (let i = 0; i <= steps; i++) {
    const ti = t0 + ((t1 - t0) * i) / steps
    const tau = ti // elapsed from the known state at t=0
    t.push(ti)
    s.push(s0 + u * tau + 0.5 * a * tau * tau)
    v.push(u + a * tau)
    aa.push(a)
  }
  return orientHistory({ t, s, v, a: aa })
}

/** Closed-form s(t), v(t) from a known state at t = 0, for charting a selected root. */
export function constAHistory(s0: number, v0: number, a: number, tEnd: number): RectilinearResult {
  return historyConstA(s0, v0, a, tEnd)
}

function rankTime(t: number): number {
  // Prefer the first interesting instant: skip the trivial t = 0 root when
  // another exists (e.g. "when does it return to s₀"), then nonnegative, then |t|.
  if (Math.abs(t) < 1e-9) return 5e5
  return (t < -1e-12 ? 1e6 : 0) + Math.abs(t)
}

export function solveConstA(knowns: SolverKnowns): SolverResult {
  const given = {
    ds: knowns.s === null ? null : knowns.s - knowns.s0,
    u: knowns.v0,
    v: knowns.v,
    a: knowns.a,
    t: knowns.t,
  }
  const filled: Slot[] = (['ds', 'u', 'v', 'a', 't'] as Slot[]).filter((k) => given[k] !== null)
  if (filled.length < 3) {
    return {
      ok: false,
      error: `Need 3 of {Δs, v₀, v, a, t}. You have ${filled.length}. Fill another known, or switch to a = f(·) if acceleration isn't constant.`,
    }
  }

  const trios: Slot[][] = []
  for (let i = 0; i < filled.length; i++) {
    for (let j = i + 1; j < filled.length; j++) {
      for (let k = j + 1; k < filled.length; k++) {
        trios.push([filled[i], filled[j], filled[k]])
      }
    }
  }

  let best: { sols: Five[]; res: number; trio: Slot[] } | null = null
  let lastError = 'could not solve'

  for (const trio of trios) {
    const ds = trio.includes('ds') ? given.ds : null
    const u = trio.includes('u') ? given.u : null
    const v = trio.includes('v') ? given.v : null
    const a = trio.includes('a') ? given.a : null
    const t = trio.includes('t') ? given.t : null
    const r = solveTrio(ds, u, v, a, t)
    if (!r.sols.length) {
      lastError = r.error ?? lastError
      continue
    }
    const res = Math.min(...r.sols.map((sol) => residual(sol, given)))
    if (!best || res < best.res) best = { sols: r.sols, res, trio }
  }

  if (!best) return { ok: false, error: lastError }

  if (filled.length > 3 && best.res > RESIDUAL_TOL) {
    return {
      ok: false,
      error: 'Those values are inconsistent with constant-a kinematics (they do not satisfy v = v₀ + a t and s = s₀ + v₀ t + ½ a t² together). Uncheck one that was measured, or a is not constant.',
    }
  }

  const sols = [...best.sols].sort((p, q) => rankTime(p.t) - rankTime(q.t))
  const notes: string[] = []
  if (sols.length > 1) {
    notes.push('Two roots (the v² = v₀² + 2aΔs square root, or the quadratic in t). Showing the smallest t ≥ 0 first.')
  }
  if (filled.length > 3 && best.res <= RESIDUAL_TOL) {
    notes.push('Extra knowns match the constant-a equations (consistent).')
  }

  const toPoint = (f: Five): SolverPoint => ({
    t: f.t,
    s: knowns.s0 + f.ds,
    v: f.v,
    a: f.a,
  })

  const primary = toPoint(sols[0])
  const alternatives = sols.slice(1).map(toPoint)

  return {
    ok: true,
    primary,
    alternatives,
    method: constAMethod(best.trio),
    notes,
    computed: {
      t: knowns.t === null,
      s: knowns.s === null,
      v: knowns.v === null,
      a: knowns.a === null,
    },
    history: historyConstA(knowns.s0, sols[0].u, sols[0].a, sols[0].t),
  }
}

function downsample(h: RectilinearResult, maxPoints = 500): RectilinearResult {
  const oriented = orientHistory(h)
  const n = oriented.t.length
  if (n <= maxPoints) return oriented
  const t: number[] = []
  const s: number[] = []
  const v: number[] = []
  const a: number[] = []
  const stride = (n - 1) / (maxPoints - 1)
  for (let i = 0; i < maxPoints; i++) {
    const idx = i === maxPoints - 1 ? n - 1 : Math.round(i * stride)
    t.push(oriented.t[idx])
    s.push(oriented.s[idx])
    v.push(oriented.v[idx])
    a.push(oriented.a[idx])
  }
  return { t, s, v, a }
}

function aMethodLatex(vars: Set<'t' | 's' | 'v'>): string[] {
  const onlyT = vars.has('t') && !vars.has('s') && !vars.has('v')
  const onlyS = vars.has('s') && !vars.has('t') && !vars.has('v')
  const onlyV = vars.has('v') && !vars.has('t') && !vars.has('s')
  if (onlyT) {
    return [
      'a = \\dfrac{dv}{dt} = f(t)',
      'v(t) = v_0 + \\displaystyle\\int_{t_0}^{t} a(\\tau)\\,d\\tau',
      's(t) = s_0 + \\displaystyle\\int_{t_0}^{t} v(\\tau)\\,d\\tau',
    ]
  }
  if (onlyS) {
    return [
      'a\\,ds = v\\,dv \\quad (a = f(s))',
      '\\tfrac12\\big(v^2 - v_0^2\\big) = \\displaystyle\\int_{s_0}^{s} a(\\sigma)\\,d\\sigma',
      'dt = \\dfrac{ds}{v}',
    ]
  }
  if (onlyV) {
    return ['a = \\dfrac{dv}{dt} = f(v)', 'dt = \\dfrac{dv}{a(v)}', 'ds = \\dfrac{v\\,dv}{a(v)}']
  }
  return ['\\dot s = v', '\\dot v = a(t,s,v)', '\\text{RK4 on the coupled first-order system}']
}

interface State {
  t: number
  s: number
  v: number
}

function rk4Step(st: State, dt: number, aFn: (t: number, s: number, v: number) => number): State {
  const a1 = aFn(st.t, st.s, st.v)
  const k1s = st.v
  const k1v = a1

  const a2 = aFn(st.t + 0.5 * dt, st.s + 0.5 * dt * k1s, st.v + 0.5 * dt * k1v)
  const k2s = st.v + 0.5 * dt * k1v
  const k2v = a2

  const a3 = aFn(st.t + 0.5 * dt, st.s + 0.5 * dt * k2s, st.v + 0.5 * dt * k2v)
  const k3s = st.v + 0.5 * dt * k2v
  const k3v = a3

  const a4 = aFn(st.t + dt, st.s + dt * k3s, st.v + dt * k3v)
  const k4s = st.v + dt * k3v
  const k4v = a4

  return {
    t: st.t + dt,
    s: st.s + (dt / 6) * (k1s + 2 * k2s + 2 * k3s + k4s),
    v: st.v + (dt / 6) * (k1v + 2 * k2v + 2 * k3v + k4v),
  }
}

function qty(st: State, kind: TargetKind): number {
  if (kind === 't') return st.t
  if (kind === 's') return st.s
  return st.v
}

function crossed(a: number, b: number, target: number): boolean {
  if (!finite(a) || !finite(b)) return false
  return (a - target) * (b - target) <= 0
}

function integrateIVP(
  start: State,
  aFn: (t: number, s: number, v: number) => number,
  target: { kind: TargetKind; value: number },
  tHorizon: number,
  project?: (st: State) => State,
): { hit: State | null; history: RectilinearResult; error?: string } {
  const t: number[] = []
  const s: number[] = []
  const v: number[] = []
  const a: number[] = []

  const push = (st: State) => {
    const acc = aFn(st.t, st.s, st.v)
    t.push(st.t)
    s.push(st.s)
    v.push(st.v)
    a.push(acc)
  }

  const tMax = Math.max(1e-6, tHorizon)
  const dtMag = Math.min(0.004, Math.max(2e-4, tMax / 8000))

  const tryDirection = (dir: 1 | -1): State | null => {
    t.length = 0
    s.length = 0
    v.length = 0
    a.length = 0
    let st = { ...start }
    push(st)
    if (target.kind === 't' && nearly(st.t, target.value, 1e-12)) return st
    if (target.kind !== 't' && nearly(qty(st, target.kind), target.value, 1e-10)) return st

    const dt = dir * dtMag
    const tLimit = start.t + dir * tMax
    const pastLimit = (time: number) => (dir > 0 ? time >= tLimit : time <= tLimit)

    // Direct time target: last step lands on it.
    if (target.kind === 't') {
      const span = target.value - start.t
      if (Math.sign(span) !== dir && Math.abs(span) > EPS) return null
      const nSteps = Math.max(1, Math.ceil(Math.abs(span) / dtMag))
      const h = span / nSteps
      for (let i = 0; i < nSteps; i++) {
        st = rk4Step(st, h, aFn)
        if (project) st = project(st)
        if (!finite(st.s) || !finite(st.v)) return null
        push(st)
      }
      return st
    }

    let guard = 0
    const guardMax = Math.ceil(tMax / dtMag) + 4
    while (guard++ < guardMax && !pastLimit(st.t)) {
      const prev = st
      const qPrev = qty(prev, target.kind)
      let next = rk4Step(prev, dt, aFn)
      if (project) next = project(next)
      if (!finite(next.s) || !finite(next.v) || !finite(aFn(next.t, next.s, next.v))) {
        return null
      }
      const qNext = qty(next, target.kind)
      if (crossed(qPrev, qNext, target.value) && Math.abs(qNext - qPrev) > 1e-18) {
        let lo = 0
        let hi = dt
        for (let k = 0; k < 18; k++) {
          const mid = 0.5 * (lo + hi)
          let trial = rk4Step(prev, mid, aFn)
          if (project) trial = project(trial)
          if (crossed(qPrev, qty(trial, target.kind), target.value)) hi = mid
          else lo = mid
        }
        let hit = rk4Step(prev, 0.5 * (lo + hi), aFn)
        if (project) hit = project(hit)
        push(hit)
        return hit
      }
      st = next
      push(st)
    }
    return null
  }

  let hit = tryDirection(1)
  let hist: RectilinearResult = { t: [...t], s: [...s], v: [...v], a: [...a] }
  if (!hit) {
    hit = tryDirection(-1)
    hist = { t: [...t], s: [...s], v: [...v], a: [...a] }
  }
  if (!hit) {
    return {
      hit: null,
      history: downsample(hist),
      error: `Never reached ${target.kind} = ${target.value} within ±${tMax} s of t₀. Increase the search time, or the motion may not get there (wrong direction, or a turning point).`,
    }
  }
  // `tryDirection` left t/s/v/a as the successful run.
  hist = { t: [...t], s: [...s], v: [...v], a: [...a] }
  return { hit, history: downsample(hist) }
}

export function solveVariable(input: VariableInput): SolverResult {
  const { expr, t0, s0, v0, target, tHorizon } = input
  const notes: string[] = []

  let aFn: (t: number, s: number, v: number) => number
  let project: ((st: State) => State) | undefined
  let method: string[]
  const computed = { t: target.kind !== 't', s: target.kind !== 's', v: target.kind !== 'v', a: true }

  if (input.kind === 'a-expr') {
    aFn = (t, s, v) => expr.eval({ t, s, v })
    method = aMethodLatex(expr.vars)
    method = [`a = ${expr.latex}`, ...method]
  } else if (input.kind === 'v-of-t') {
    if (expr.vars.has('s') || expr.vars.has('v')) {
      return { ok: false, error: 'v(t) should depend only on t (not s or v).' }
    }
    let dvdt: CompiledExpr
    try {
      dvdt = expr.deriv('t')
    } catch (err) {
      return { ok: false, error: err instanceof Error ? err.message : String(err) }
    }
    const v0Expr = expr.eval({ t: t0, s: s0, v: v0 })
    if (finite(v0Expr) && !nearly(v0Expr, v0, 1e-6)) {
      notes.push(`v(t₀) from the expression is ${v0Expr.toPrecision(5)}, which differs from the entered v₀ — using the expression.`)
    }
    aFn = (t, s, v) => dvdt.eval({ t, s, v })
    project = (st) => ({ ...st, v: expr.eval({ t: st.t, s: st.s, v: st.v }) })
    method = [`v(t) = ${expr.latex}`, `a = \\dfrac{dv}{dt} = ${dvdt.latex}`, 's(t) = s_0 + \\displaystyle\\int_{t_0}^{t} v(\\tau)\\,d\\tau']
  } else {
    if (expr.vars.has('t') || expr.vars.has('v')) {
      return { ok: false, error: 'v(s) should depend only on s (not t or v).' }
    }
    let dvds: CompiledExpr
    try {
      dvds = expr.deriv('s')
    } catch (err) {
      return { ok: false, error: err instanceof Error ? err.message : String(err) }
    }
    const v0Expr = expr.eval({ t: t0, s: s0, v: v0 })
    if (finite(v0Expr) && !nearly(v0Expr, v0, 1e-6)) {
      notes.push(`v(s₀) from the expression is ${v0Expr.toPrecision(5)}, which differs from the entered v₀ — using the expression.`)
    }
    aFn = (t, s, v) => {
      const vv = expr.eval({ t, s, v })
      return vv * dvds.eval({ t, s, v })
    }
    project = (st) => ({ ...st, v: expr.eval({ t: st.t, s: st.s, v: st.v }) })
    method = [`v(s) = ${expr.latex}`, `a = v\\,\\dfrac{dv}{ds} = ${dvds.latex}\\cdot v`, 'dt = \\dfrac{ds}{v}']
  }

  const vStart = input.kind === 'a-expr' ? v0 : expr.eval({ t: t0, s: s0, v: v0 })
  if (!finite(vStart)) return { ok: false, error: 'starting velocity is not finite — check the expression at (t₀, s₀).' }

  const start: State = { t: t0, s: s0, v: vStart }
  const a0 = aFn(t0, s0, vStart)
  if (!finite(a0) && input.kind === 'a-expr') {
    return { ok: false, error: 'acceleration is not finite at the starting state — check for division by zero or sqrt of a negative.' }
  }

  const integ = integrateIVP(start, aFn, target, tHorizon, project)
  if (!integ.hit) {
    return { ok: false, error: integ.error ?? 'did not reach the target' }
  }

  const hit = integ.hit
  const aHit = aFn(hit.t, hit.s, hit.v)
  if (!finite(hit.s) || !finite(hit.v) || !finite(aHit)) {
    return { ok: false, error: 'the solution became non-finite (division by zero or a domain error in the expression).' }
  }

  if (hit.t < t0 - 1e-9) {
    notes.push('Reached the target going backward in time from t₀ (the particle was there earlier).')
  }

  return {
    ok: true,
    primary: { t: hit.t, s: hit.s, v: hit.v, a: aHit },
    alternatives: [],
    method,
    notes,
    computed,
    history: integ.history,
  }
}

export function solveFromExpression(
  kind: 'a-expr' | 'v-of-t' | 'v-of-s',
  source: string,
  t0: number,
  s0: number,
  v0: number,
  target: { kind: TargetKind; value: number },
  tHorizon: number,
): SolverResult {
  const parsed = parseExpr(source)
  if (!parsed.ok) return { ok: false, error: parsed.error }
  return solveVariable({ kind, expr: parsed.expr, t0, s0, v0, target, tHorizon })
}

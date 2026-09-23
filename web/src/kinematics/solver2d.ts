/** 2D particle with constant acceleration — the projectile / snowblower class. */

const EPS = 1e-10

export const G_SI = 9.81
export const G_FT = 32.2
export const FT_TO_M = 0.3048

export type LengthUnit = 'm' | 'ft'

export function gDefault(unit: LengthUnit): number {
  return unit === 'ft' ? G_FT : G_SI
}

export function convertLength(v: number, from: LengthUnit, to: LengthUnit): number {
  if (from === to) return v
  return from === 'ft' ? v * FT_TO_M : v / FT_TO_M
}

export interface Known2D {
  xA: number
  yA: number
  ax: number
  ay: number
  xB: number | null
  yB: number | null
  v0: number | null
  thetaDeg: number | null
  t: number | null
}

export interface Sol2D {
  xA: number
  yA: number
  xB: number
  yB: number
  v0: number
  thetaDeg: number
  t: number
  vxA: number
  vyA: number
  vxB: number
  vyB: number
  ax: number
  ay: number
}

export interface Hist2D {
  t: number[]
  x: number[]
  y: number[]
  vx: number[]
  vy: number[]
}

export interface Ok2D {
  ok: true
  primary: Sol2D
  alternatives: Sol2D[]
  method: string[]
  notes: string[]
  computed: { v0: boolean; theta: boolean; xB: boolean; yB: boolean; t: boolean }
  history: Hist2D
}

export interface Err2D {
  ok: false
  error: string
}

export type Result2D = Ok2D | Err2D

function finite(n: number): boolean {
  return Number.isFinite(n)
}

function nearly(a: number, b: number, rel = 1e-4): boolean {
  return Math.abs(a - b) <= rel * (1 + Math.abs(a) + Math.abs(b))
}

function pack(xA: number, yA: number, vxA: number, vyA: number, ax: number, ay: number, t: number): Sol2D | null {
  if (!finite(t) || !finite(vxA) || !finite(vyA)) return null
  const xB = xA + vxA * t + 0.5 * ax * t * t
  const yB = yA + vyA * t + 0.5 * ay * t * t
  const vxB = vxA + ax * t
  const vyB = vyA + ay * t
  const v0 = Math.hypot(vxA, vyA)
  const thetaDeg = (Math.atan2(vyA, vxA) * 180) / Math.PI
  if (![xB, yB, vxB, vyB, v0, thetaDeg].every(finite)) return null
  return { xA, yA, xB, yB, v0, thetaDeg, t, vxA, vyA, vxB, vyB, ax, ay }
}

function fromAngle(v0: number, thetaDeg: number): { vxA: number; vyA: number } {
  const th = (thetaDeg * Math.PI) / 180
  return { vxA: v0 * Math.cos(th), vyA: v0 * Math.sin(th) }
}

function rankT(t: number): number {
  if (Math.abs(t) < 1e-9) return 5e5
  return (t < -1e-12 ? 1e6 : 0) + Math.abs(t)
}

export function trajectory2d(sol: Sol2D, steps = 400): Hist2D {
  const t: number[] = []
  const x: number[] = []
  const y: number[] = []
  const vx: number[] = []
  const vy: number[] = []
  const tEnd = sol.t
  if (Math.abs(tEnd) < 1e-15) {
    return { t: [0], x: [sol.xA], y: [sol.yA], vx: [sol.vxA], vy: [sol.vyA] }
  }
  for (let i = 0; i <= steps; i++) {
    const ti = (tEnd * i) / steps
    t.push(ti)
    x.push(sol.xA + sol.vxA * ti + 0.5 * sol.ax * ti * ti)
    y.push(sol.yA + sol.vyA * ti + 0.5 * sol.ay * ti * ti)
    vx.push(sol.vxA + sol.ax * ti)
    vy.push(sol.vyA + sol.ay * ti)
  }
  if (t[t.length - 1] < t[0]) {
    t.reverse()
    x.reverse()
    y.reverse()
    vx.reverse()
    vy.reverse()
  }
  return { t, x, y, vx, vy }
}

/** θ + Δr + a  →  v0, t. The snowblower / "pass through B" case. */
function fromAngleAndDisplacement(k: Known2D): { sols: Sol2D[]; error?: string; method: string[] } {
  const theta = k.thetaDeg!
  const dx = k.xB! - k.xA
  const dy = k.yB! - k.yA
  const th = (theta * Math.PI) / 180
  const tan = Math.tan(th)
  const cos = Math.cos(th)
  const denom = k.ay - k.ax * tan
  const numer = dy - dx * tan
  const method = [
    'x_B = x_A + v_0\\cos\\theta\\, t + \\tfrac12 a_x t^2',
    'y_B = y_A + v_0\\sin\\theta\\, t + \\tfrac12 a_y t^2',
    't^2 = \\dfrac{2\\big(\\Delta y - \\Delta x\\,\\tan\\theta\\big)}{a_y - a_x\\tan\\theta}',
    'v_0 = \\dfrac{\\Delta x - \\tfrac12 a_x t^2}{t\\cos\\theta}',
  ]
  if (Math.abs(cos) < EPS) {
    // Vertical launch: Δx = 0.5 ax t², Δy = v0 t + 0.5 ay t² (sin = ±1)
    if (Math.abs(dx) > 1e-9 && Math.abs(k.ax) < EPS) {
      return { sols: [], error: 'θ = ±90° cannot produce a horizontal displacement when a_x = 0', method }
    }
  }
  if (Math.abs(denom) < EPS) {
    if (Math.abs(numer) > 1e-8) {
      return { sols: [], error: 'no real t: the launch direction is parallel to a in a way that never hits B', method }
    }
    return { sols: [], error: 'underdetermined: Δr is along the launch line and a does not constrain t', method }
  }
  const t2 = (2 * numer) / denom
  if (t2 < -1e-10) {
    return {
      sols: [],
      error: 'no real time: gravity pulls the path away from B at this angle (v₀² would be negative)',
      method,
    }
  }
  const tAbs = Math.sqrt(Math.max(0, t2))
  const candidates = tAbs < EPS ? [0] : [tAbs, -tAbs]
  const sols: Sol2D[] = []
  for (const t of candidates) {
    if (Math.abs(t) < EPS) continue
    const vxA = (dx - 0.5 * k.ax * t * t) / t
    const vyA = (dy - 0.5 * k.ay * t * t) / t
    // Must match the given angle (same ray, not 180° opposite).
    const ang = (Math.atan2(vyA, vxA) * 180) / Math.PI
    let dAng = ((ang - theta + 180) % 360) - 180
    if (Math.abs(dAng) > 90) continue
    const packed = pack(k.xA, k.yA, vxA, vyA, k.ax, k.ay, t)
    if (packed) sols.push(packed)
  }
  // If the angle-match filter emptied things but t is valid, reconstruct from θ.
  if (!sols.length) {
    for (const t of candidates) {
      if (Math.abs(t) < EPS || Math.abs(cos) < EPS) continue
      const v0 = (dx - 0.5 * k.ax * t * t) / (t * cos)
      if (v0 < -EPS) continue
      const { vxA, vyA } = fromAngle(Math.abs(v0), theta)
      const packed = pack(k.xA, k.yA, vxA, vyA, k.ax, k.ay, t)
      if (packed && nearly(packed.xB, k.xB!) && nearly(packed.yB, k.yB!)) sols.push(packed)
    }
  }
  if (!sols.length) return { sols: [], error: 'no physical v₀ hits B at that angle (check the figure: is B below the θ-line?)', method }
  sols.sort((a, b) => rankT(a.t) - rankT(b.t))
  return { sols, method }
}

/** t + Δr + a  →  v0, θ */
function fromTimeAndDisplacement(k: Known2D): { sols: Sol2D[]; error?: string; method: string[] } {
  const t = k.t!
  const method = [
    'v_{0x} = \\dfrac{\\Delta x - \\tfrac12 a_x t^2}{t}',
    'v_{0y} = \\dfrac{\\Delta y - \\tfrac12 a_y t^2}{t}',
    'v_0 = \\sqrt{v_{0x}^2 + v_{0y}^2},\\quad \\theta = \\mathrm{atan2}(v_{0y}, v_{0x})',
  ]
  if (Math.abs(t) < EPS) return { sols: [], error: 't = 0: A and B must be the same point, and v₀ is undetermined', method }
  const dx = k.xB! - k.xA
  const dy = k.yB! - k.yA
  const vxA = (dx - 0.5 * k.ax * t * t) / t
  const vyA = (dy - 0.5 * k.ay * t * t) / t
  const packed = pack(k.xA, k.yA, vxA, vyA, k.ax, k.ay, t)
  if (!packed) return { sols: [], error: 'could not form v₀ from Δr, a, t', method }
  return { sols: [packed], method }
}

/** v0 + θ + t + a  →  B */
function fromLaunchAndTime(k: Known2D): { sols: Sol2D[]; error?: string; method: string[] } {
  const method = [
    'v_{0x} = v_0\\cos\\theta,\\quad v_{0y} = v_0\\sin\\theta',
    'x(t) = x_A + v_{0x} t + \\tfrac12 a_x t^2',
    'y(t) = y_A + v_{0y} t + \\tfrac12 a_y t^2',
  ]
  const { vxA, vyA } = fromAngle(k.v0!, k.thetaDeg!)
  const packed = pack(k.xA, k.yA, vxA, vyA, k.ax, k.ay, k.t!)
  if (!packed) return { sols: [], error: 'could not step to time t', method }
  return { sols: [packed], method }
}

/** v0 + θ + xB + a  →  t, yB */
function fromLaunchAndX(k: Known2D): { sols: Sol2D[]; error?: string; method: string[] } {
  const method = [
    '\\Delta x = v_0\\cos\\theta\\, t + \\tfrac12 a_x t^2',
    '\\tfrac12 a_x t^2 + v_0\\cos\\theta\\, t - \\Delta x = 0',
  ]
  const dx = k.xB! - k.xA
  const { vxA, vyA } = fromAngle(k.v0!, k.thetaDeg!)
  const A = 0.5 * k.ax
  const B = vxA
  const C = -dx
  const ts: number[] = []
  if (Math.abs(A) < EPS) {
    if (Math.abs(B) < EPS) {
      if (Math.abs(dx) > 1e-9) return { sols: [], error: 'no x-motion: cannot reach that x_B', method }
      return { sols: [], error: 'x_B = x_A already; give y_B or t instead', method }
    }
    ts.push(-C / B)
  } else {
    const disc = B * B - 4 * A * C
    if (disc < -1e-12) return { sols: [], error: 'no real time to that x_B', method }
    const s = Math.sqrt(Math.max(0, disc))
    ts.push((-B + s) / (2 * A), (-B - s) / (2 * A))
  }
  const sols = ts.map((t) => pack(k.xA, k.yA, vxA, vyA, k.ax, k.ay, t)).filter((p): p is Sol2D => p !== null)
  sols.sort((a, b) => rankT(a.t) - rankT(b.t))
  if (!sols.length) return { sols: [], error: 'no finite time to that x_B', method }
  return { sols, method }
}

/** v0 + θ + yB + a  →  t, xB */
function fromLaunchAndY(k: Known2D): { sols: Sol2D[]; error?: string; method: string[] } {
  const method = [
    '\\Delta y = v_0\\sin\\theta\\, t + \\tfrac12 a_y t^2',
    '\\tfrac12 a_y t^2 + v_0\\sin\\theta\\, t - \\Delta y = 0',
  ]
  const dy = k.yB! - k.yA
  const { vxA, vyA } = fromAngle(k.v0!, k.thetaDeg!)
  const A = 0.5 * k.ay
  const B = vyA
  const C = -dy
  const ts: number[] = []
  if (Math.abs(A) < EPS) {
    if (Math.abs(B) < EPS) {
      if (Math.abs(dy) > 1e-9) return { sols: [], error: 'no y-motion: cannot reach that y_B', method }
      return { sols: [], error: 'y_B = y_A already; give x_B or t instead', method }
    }
    ts.push(-C / B)
  } else {
    const disc = B * B - 4 * A * C
    if (disc < -1e-12) return { sols: [], error: 'no real time to that y_B (never gets that high/low)', method }
    const s = Math.sqrt(Math.max(0, disc))
    ts.push((-B + s) / (2 * A), (-B - s) / (2 * A))
  }
  const sols = ts.map((t) => pack(k.xA, k.yA, vxA, vyA, k.ax, k.ay, t)).filter((p): p is Sol2D => p !== null)
  sols.sort((a, b) => rankT(a.t) - rankT(b.t))
  if (!sols.length) return { sols: [], error: 'no finite time to that y_B', method }
  return { sols, method }
}

/** v0 + Δr + a_x=0  →  θ (quadratic in tan θ), t */
function fromSpeedAndDisplacement(k: Known2D): { sols: Sol2D[]; error?: string; method: string[] } {
  const dx = k.xB! - k.xA
  const dy = k.yB! - k.yA
  const v0 = k.v0!
  const method = [
    '\\Delta y = \\Delta x\\,\\tan\\theta + \\dfrac{a_y \\Delta x^2}{2 v_0^2}\\big(1+\\tan^2\\theta\\big)',
    '\\text{(uses }a_x=0\\text{; two launch angles are common)}',
  ]
  if (Math.abs(k.ax) > 1e-9) {
    return { sols: [], error: 'finding θ from v₀ and Δr is implemented for gravity problems (a_x = 0). Set a_x = 0 or give θ.', method }
  }
  if (Math.abs(v0) < EPS) return { sols: [], error: 'v₀ = 0 cannot reach a different point', method }
  if (Math.abs(dx) < EPS) {
    // Vertical: θ = ±90°
    const thetaDeg = dy >= 0 ? 90 : -90
    const { vxA, vyA } = fromAngle(v0, thetaDeg)
    const A = 0.5 * k.ay
    const B = vyA
    const C = -dy
    const disc = B * B - 4 * A * C
    if (disc < -1e-12 && Math.abs(A) > EPS) return { sols: [], error: 'cannot reach that height with this v₀', method }
    const ts =
      Math.abs(A) < EPS
        ? Math.abs(B) < EPS
          ? []
          : [-C / B]
        : [(-B + Math.sqrt(Math.max(0, disc))) / (2 * A), (-B - Math.sqrt(Math.max(0, disc))) / (2 * A)]
    const sols = ts.map((t) => pack(k.xA, k.yA, vxA, vyA, k.ax, k.ay, t)).filter((p): p is Sol2D => p !== null)
    if (!sols.length) return { sols: [], error: 'no time for vertical shot', method }
    return { sols, method }
  }
  const coeff = (k.ay * dx * dx) / (2 * v0 * v0) // k in k(1+u²)
  // coeff u² + dx u + coeff - dy = 0
  const A = coeff
  const B = dx
  const C = coeff - dy
  const us: number[] = []
  if (Math.abs(A) < EPS) {
    us.push(-C / B)
  } else {
    const disc = B * B - 4 * A * C
    if (disc < -1e-12) {
      return { sols: [], error: 'no real launch angle: v₀ is too small to reach B', method }
    }
    const s = Math.sqrt(Math.max(0, disc))
    us.push((-B + s) / (2 * A), (-B - s) / (2 * A))
  }
  const sols: Sol2D[] = []
  for (const u of us) {
    const thetaDeg = (Math.atan(u) * 180) / Math.PI
    // atan loses the quadrant when cos < 0; for a projectile to +Δx, prefer the angle with cos>0.
    // Try θ and θ+180 if needed.
    for (const th of [thetaDeg, thetaDeg + 180]) {
      const { vxA, vyA } = fromAngle(v0, th)
      if (vxA * dx < -1e-8 && Math.abs(dx) > 1e-8) continue
      const t = dx / vxA
      const packed = pack(k.xA, k.yA, vxA, vyA, k.ax, k.ay, t)
      if (packed && nearly(packed.yB, k.yB!) && nearly(packed.xB, k.xB!)) sols.push(packed)
    }
  }
  // Dedup
  const uniq: Sol2D[] = []
  for (const s of sols) {
    if (!uniq.some((u) => nearly(u.thetaDeg, s.thetaDeg, 1e-6) && nearly(u.t, s.t, 1e-6))) uniq.push(s)
  }
  uniq.sort((a, b) => rankT(a.t) - rankT(b.t))
  if (!uniq.length) return { sols: [], error: 'no launch angle with this v₀ hits B', method }
  return { sols: uniq, method }
}

function residual(sol: Sol2D, k: Known2D): number {
  let m = 0
  const check = (got: number, want: number | null) => {
    if (want === null) return
    m = Math.max(m, Math.abs(got - want) / (1 + Math.abs(want) + Math.abs(got)))
  }
  check(sol.xB, k.xB)
  check(sol.yB, k.yB)
  check(sol.v0, k.v0)
  check(sol.t, k.t)
  if (k.thetaDeg !== null) {
    let d = ((sol.thetaDeg - k.thetaDeg + 180) % 360) - 180
    m = Math.max(m, Math.abs(d) / 180)
  }
  return m
}

export function solve2D(k: Known2D): Result2D {
  const hasXB = k.xB !== null
  const hasYB = k.yB !== null
  const hasV0 = k.v0 !== null
  const hasTh = k.thetaDeg !== null
  const hasT = k.t !== null
  const hasB = hasXB && hasYB

  const attempts: { run: () => { sols: Sol2D[]; error?: string; method: string[] }; need: boolean }[] = [
    { need: hasTh && hasB, run: () => fromAngleAndDisplacement(k) },
    { need: hasT && hasB, run: () => fromTimeAndDisplacement(k) },
    { need: hasV0 && hasTh && hasT, run: () => fromLaunchAndTime(k) },
    { need: hasV0 && hasTh && hasXB, run: () => fromLaunchAndX(k) },
    { need: hasV0 && hasTh && hasYB, run: () => fromLaunchAndY(k) },
    { need: hasV0 && hasB, run: () => fromSpeedAndDisplacement(k) },
  ]

  let best: { sols: Sol2D[]; method: string[]; res: number } | null = null
  let lastError = ''

  for (const a of attempts) {
    if (!a.need) continue
    const r = a.run()
    if (!r.sols.length) {
      lastError = r.error ?? lastError
      continue
    }
    const res = Math.min(...r.sols.map((s) => residual(s, k)))
    if (!best || res < best.res) best = { sols: r.sols, method: r.method, res }
  }

  if (!best) {
    if (!hasB && !(hasV0 && hasTh && hasT) && !(hasV0 && hasTh && (hasXB || hasYB))) {
      return {
        ok: false,
        error:
          'Need Δr = B − A (both x and y of the other point) and constant a, plus one more about the launch: the angle θ (from the figure), or v₀, or t.',
      }
    }
    return {
      ok: false,
      error: lastError || 'not enough knowns. Typical textbook set: a = g downward, point A, point B, and launch angle θ → solves v₀ and t.',
    }
  }

  if (best.res > 2e-3) {
    return {
      ok: false,
      error: 'Those values are inconsistent with constant-a projectile motion (they do not lie on one parabola). Uncheck one.',
    }
  }

  const sols = [...best.sols].sort((p, q) => rankT(p.t) - rankT(q.t))
  const primary = sols[0]
  const notes: string[] = []
  if (sols.length > 1) {
    notes.push('Two roots (going up and coming down, or two launch angles). Showing the first t > 0.')
  }

  return {
    ok: true,
    primary,
    alternatives: sols.slice(1),
    method: best.method,
    notes,
    computed: {
      v0: !hasV0,
      theta: !hasTh,
      xB: !hasXB,
      yB: !hasYB,
      t: !hasT,
    },
    history: trajectory2d(primary),
  }
}

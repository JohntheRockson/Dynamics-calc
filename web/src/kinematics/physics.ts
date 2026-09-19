// Particle kinematics: rectilinear motion, curvilinear (projectile) motion,
// and relative motion of two particles on translating axes. Pure functions,
// SI units throughout (m, s, kg, N). This mirrors the "governing equation
// in, full time history out" shape of `engine::run_sim` on the Rust side,
// but stays entirely client-side: these are 1-4 state ODEs with a handful
// of parameters, so a network round trip would only add latency for no
// benefit.

export interface Series {
  t: number[]
}

// ---------------------------------------------------------------------
// Rectilinear motion (1D)
// ---------------------------------------------------------------------

export interface RectilinearResult {
  t: number[]
  s: number[]
  v: number[]
  a: number[]
}

/** Constant acceleration: s(t) = s0 + v0 t + 1/2 a t^2 (exact, closed form). */
export function rectilinearConstantAccel(s0: number, v0: number, a: number, tFinal: number, steps = 400): RectilinearResult {
  const t: number[] = []
  const s: number[] = []
  const v: number[] = []
  const aOut: number[] = []
  for (let i = 0; i <= steps; i++) {
    const ti = (tFinal * i) / steps
    t.push(ti)
    s.push(s0 + v0 * ti + 0.5 * a * ti * ti)
    v.push(v0 + a * ti)
    aOut.push(a)
  }
  return { t, s, v, a: aOut }
}

export interface DragParams {
  mass: number // kg
  cd: number // drag coefficient, dimensionless
  area: number // m^2, reference cross-sectional area
  airDensity: number // kg/m^3
}

/** Quadratic-drag coefficient k such that F_drag = k * v^2 (opposing motion). */
function dragK(p: DragParams): number {
  return 0.5 * p.airDensity * p.cd * p.area
}

/** Closed-form speed of a body dropped from rest under gravity + quadratic drag (v0=0). */
export function terminalSpeed(g: number, p: DragParams): number {
  const k = dragK(p)
  if (k <= 0) return Infinity
  return Math.sqrt((p.mass * g) / k)
}

/**
 * 1D motion under constant gravity (acting in the -s direction) plus
 * quadratic drag opposing velocity: m dv/dt = -m g - sign(v) * k * v^2.
 * Solved by RK4 since the closed form differs by initial-velocity sign
 * and phase (rising vs falling); RK4 is uniformly correct for any v0.
 */
export function rectilinearWithDrag(s0: number, v0: number, g: number, drag: DragParams, tFinal: number, steps = 800): RectilinearResult {
  const k = drag.mass > 0 ? dragK(drag) / drag.mass : 0
  const deriv = (v: number): number => -g - Math.sign(v) * k * v * v

  const t: number[] = [0]
  const s: number[] = [s0]
  const v: number[] = [v0]
  const a: number[] = [deriv(v0)]
  const dt = tFinal / steps

  let sCur = s0
  let vCur = v0
  for (let i = 0; i < steps; i++) {
    // RK4 on the coupled (s, v) system; ds/dt = v is trivial but kept
    // explicit for symmetry with the engine's own RK4 style.
    const k1v = deriv(vCur)
    const k1s = vCur
    const k2v = deriv(vCur + 0.5 * dt * k1v)
    const k2s = vCur + 0.5 * dt * k1v
    const k3v = deriv(vCur + 0.5 * dt * k2v)
    const k3s = vCur + 0.5 * dt * k2v
    const k4v = deriv(vCur + dt * k3v)
    const k4s = vCur + dt * k3v

    vCur += (dt / 6) * (k1v + 2 * k2v + 2 * k3v + k4v)
    sCur += (dt / 6) * (k1s + 2 * k2s + 2 * k3s + k4s)

    t.push((i + 1) * dt)
    s.push(sCur)
    v.push(vCur)
    a.push(deriv(vCur))
  }
  return { t, s, v, a }
}

// ---------------------------------------------------------------------
// Curvilinear motion (2D projectile)
// ---------------------------------------------------------------------

export interface CurvilinearResult {
  t: number[]
  x: number[]
  y: number[]
  vx: number[]
  vy: number[]
  ax: number[]
  ay: number[]
  landingIndex: number // index of the last sample (== length-1 if it never lands within tFinal)
}

export interface ProjectileParams {
  x0: number
  y0: number
  speed0: number
  angleDeg: number
  g: number
  groundY: number
  drag: DragParams | null // null => vacuum (closed-form) trajectory
}

/** No-drag projectile: exact closed-form rectangular components. */
function projectileNoDrag(p: ProjectileParams, tFinal: number, steps: number): CurvilinearResult {
  const theta = (p.angleDeg * Math.PI) / 180
  const vx0 = p.speed0 * Math.cos(theta)
  const vy0 = p.speed0 * Math.sin(theta)
  const t: number[] = []
  const x: number[] = []
  const y: number[] = []
  const vx: number[] = []
  const vy: number[] = []
  const ax: number[] = []
  const ay: number[] = []
  let landingIndex = steps
  let landed = false
  for (let i = 0; i <= steps; i++) {
    const ti = (tFinal * i) / steps
    const xi = p.x0 + vx0 * ti
    const yi = p.y0 + vy0 * ti - 0.5 * p.g * ti * ti
    t.push(ti)
    x.push(xi)
    y.push(yi)
    vx.push(vx0)
    vy.push(vy0 - p.g * ti)
    ax.push(0)
    ay.push(-p.g)
    if (!landed && yi <= p.groundY && i > 0) {
      landingIndex = i
      landed = true
    }
  }
  return { t, x, y, vx, vy, ax, ay, landingIndex }
}

/** Projectile with quadratic air drag opposing the velocity vector: RK4. */
function projectileWithDrag(p: ProjectileParams, drag: DragParams, tFinal: number, steps: number): CurvilinearResult {
  const theta = (p.angleDeg * Math.PI) / 180
  const kOverM = drag.mass > 0 ? dragK(drag) / drag.mass : 0
  const g = p.g

  const deriv = (vx: number, vy: number): [number, number] => {
    const speed = Math.hypot(vx, vy)
    return [-kOverM * speed * vx, -g - kOverM * speed * vy]
  }

  const dt = tFinal / steps
  const t: number[] = [0]
  const x: number[] = [p.x0]
  const y: number[] = [p.y0]
  const vx: number[] = [p.speed0 * Math.cos(theta)]
  const vy: number[] = [p.speed0 * Math.sin(theta)]
  const [ax0, ay0] = deriv(vx[0], vy[0])
  const ax: number[] = [ax0]
  const ay: number[] = [ay0]

  let xCur = x[0]
  let yCur = y[0]
  let vxCur = vx[0]
  let vyCur = vy[0]
  let landingIndex = steps
  let landed = false

  for (let i = 0; i < steps; i++) {
    const [k1vx, k1vy] = deriv(vxCur, vyCur)
    const k1x = vxCur
    const k1y = vyCur

    const [k2vx, k2vy] = deriv(vxCur + 0.5 * dt * k1vx, vyCur + 0.5 * dt * k1vy)
    const k2x = vxCur + 0.5 * dt * k1vx
    const k2y = vyCur + 0.5 * dt * k1vy

    const [k3vx, k3vy] = deriv(vxCur + 0.5 * dt * k2vx, vyCur + 0.5 * dt * k2vy)
    const k3x = vxCur + 0.5 * dt * k2vx
    const k3y = vyCur + 0.5 * dt * k2vy

    const [k4vx, k4vy] = deriv(vxCur + dt * k3vx, vyCur + dt * k3vy)
    const k4x = vxCur + dt * k3vx
    const k4y = vyCur + dt * k3vy

    const yPrev = yCur
    vxCur += (dt / 6) * (k1vx + 2 * k2vx + 2 * k3vx + k4vx)
    vyCur += (dt / 6) * (k1vy + 2 * k2vy + 2 * k3vy + k4vy)
    xCur += (dt / 6) * (k1x + 2 * k2x + 2 * k3x + k4x)
    yCur += (dt / 6) * (k1y + 2 * k2y + 2 * k3y + k4y)

    t.push((i + 1) * dt)
    x.push(xCur)
    y.push(yCur)
    vx.push(vxCur)
    vy.push(vyCur)
    const [axi, ayi] = deriv(vxCur, vyCur)
    ax.push(axi)
    ay.push(ayi)

    if (!landed && yCur <= p.groundY && yPrev > p.groundY) {
      landingIndex = i + 1
      landed = true
    }
  }
  return { t, x, y, vx, vy, ax, ay, landingIndex }
}

export function simulateProjectile(p: ProjectileParams, tFinal: number, steps = 600): CurvilinearResult {
  return p.drag ? projectileWithDrag(p, p.drag, tFinal, steps) : projectileNoDrag(p, tFinal, steps)
}

/** Exact (no-drag) time of flight to `groundY`, or null if the projectile never reaches it going up. */
export function noDragTimeOfFlight(p: ProjectileParams): number | null {
  const theta = (p.angleDeg * Math.PI) / 180
  const vy0 = p.speed0 * Math.sin(theta)
  const dy = p.y0 - p.groundY
  // 0 = dy + vy0 t - 1/2 g t^2  =>  1/2 g t^2 - vy0 t - dy = 0
  const a = 0.5 * p.g
  const b = -vy0
  const c = -dy
  const disc = b * b - 4 * a * c
  if (disc < 0 || p.g <= 0) return null
  const sqrtDisc = Math.sqrt(disc)
  const t1 = (-b + sqrtDisc) / (2 * a)
  const t2 = (-b - sqrtDisc) / (2 * a)
  const candidates = [t1, t2].filter((v) => v > 1e-9)
  if (candidates.length === 0) return null
  return Math.min(...candidates)
}

export interface NormalTangential {
  speed: number
  aTangential: number
  aNormal: number
  radiusOfCurvature: number
}

/** Tangential/normal acceleration components and radius of curvature at one instant. */
export function normalTangentialAt(vx: number, vy: number, ax: number, ay: number): NormalTangential {
  const speed = Math.hypot(vx, vy)
  if (speed < 1e-9) {
    return { speed: 0, aTangential: 0, aNormal: Math.hypot(ax, ay), radiusOfCurvature: Infinity }
  }
  const aTangential = (vx * ax + vy * ay) / speed
  const aMagSq = ax * ax + ay * ay
  const aNormal = Math.sqrt(Math.max(0, aMagSq - aTangential * aTangential))
  const radiusOfCurvature = aNormal > 1e-9 ? (speed * speed) / aNormal : Infinity
  return { speed, aTangential, aNormal, radiusOfCurvature }
}

// ---------------------------------------------------------------------
// Relative motion of two particles (translating axes)
// ---------------------------------------------------------------------

export interface ParticleIC {
  x0: number
  y0: number
  vx0: number
  vy0: number
  ax: number // constant acceleration (0 for pure constant-velocity motion)
  ay: number
}

export interface RelativeMotionResult {
  t: number[]
  aPos: { x: number[]; y: number[] }
  bPos: { x: number[]; y: number[] }
  relPos: { x: number[]; y: number[] } // r_A/B = r_A - r_B
  relSpeed: number[]
  closestApproach: { t: number; distance: number }
}

function positionAt(ic: ParticleIC, t: number): { x: number; y: number; vx: number; vy: number } {
  return {
    x: ic.x0 + ic.vx0 * t + 0.5 * ic.ax * t * t,
    y: ic.y0 + ic.vy0 * t + 0.5 * ic.ay * t * t,
    vx: ic.vx0 + ic.ax * t,
    vy: ic.vy0 + ic.ay * t,
  }
}

export function simulateRelativeMotion(a: ParticleIC, b: ParticleIC, tFinal: number, steps = 400): RelativeMotionResult {
  const t: number[] = []
  const ax: number[] = []
  const ay: number[] = []
  const bx: number[] = []
  const by: number[] = []
  const rx: number[] = []
  const ry: number[] = []
  const relSpeed: number[] = []

  let closestT = 0
  let closestD = Infinity

  for (let i = 0; i <= steps; i++) {
    const ti = (tFinal * i) / steps
    const pa = positionAt(a, ti)
    const pb = positionAt(b, ti)
    const relX = pa.x - pb.x
    const relY = pa.y - pb.y
    const dist = Math.hypot(relX, relY)
    if (dist < closestD) {
      closestD = dist
      closestT = ti
    }
    t.push(ti)
    ax.push(pa.x)
    ay.push(pa.y)
    bx.push(pb.x)
    by.push(pb.y)
    rx.push(relX)
    ry.push(relY)
    relSpeed.push(Math.hypot(pa.vx - pb.vx, pa.vy - pb.vy))
  }

  return {
    t,
    aPos: { x: ax, y: ay },
    bPos: { x: bx, y: by },
    relPos: { x: rx, y: ry },
    relSpeed,
    closestApproach: { t: closestT, distance: closestD },
  }
}

/** Linear interpolation helper for reading a series at an arbitrary time. */
export function sampleAt(t: number[], series: number[], time: number): number {
  if (t.length === 0) return NaN
  if (time <= t[0]) return series[0]
  if (time >= t[t.length - 1]) return series[series.length - 1]
  let lo = 0
  let hi = t.length - 1
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1
    if (t[mid] <= time) lo = mid
    else hi = mid
  }
  const frac = (time - t[lo]) / (t[hi] - t[lo] || 1)
  return series[lo] + frac * (series[hi] - series[lo])
}

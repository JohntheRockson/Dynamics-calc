// Instantaneous curvilinear-kinematics solver. Given any mix of known
// rectangular, normal-tangential, and polar (radial-transverse) scalars at
// one instant — plus optional constant-accel time relations — fill every
// quantity that the standard identities determine uniquely. Multiple
// particles are solved independently, then relative r, v, a are formed.
// SI internally (m, s, rad). Pure functions; no I/O.

import { formatNumber } from '../format'

export const QTY_KEYS = [
  't',
  'x',
  'y',
  'vx',
  'vy',
  'ax',
  'ay',
  'speed',
  'aMag',
  'psi',
  'at',
  'an',
  'rho',
  'omega',
  'alpha',
  'r',
  'theta',
  'rdot',
  'thetadot',
  'rddot',
  'thetaddot',
  'vr',
  'vtheta',
  'ar',
  'atheta',
  's',
  's0',
  'v0',
  'omega0',
  'theta0',
  'x0',
  'y0',
  'vx0',
  'vy0',
  'r0',
  'rdot0',
] as const

export type Qty = (typeof QTY_KEYS)[number]

export type QtyGroup = 'time' | 'rect' | 'nt' | 'polar' | 'initial'

export interface QtyMeta {
  key: Qty
  label: string
  tex: string
  unit: string
  group: QtyGroup
  /** Stored in radians; UI edits degrees. */
  angle?: boolean
}

export const QTY_META: QtyMeta[] = [
  { key: 't', label: 't', tex: 't', unit: 's', group: 'time' },

  { key: 'x', label: 'x', tex: 'x', unit: 'm', group: 'rect' },
  { key: 'y', label: 'y', tex: 'y', unit: 'm', group: 'rect' },
  { key: 'vx', label: 'vx', tex: 'v_x', unit: 'm/s', group: 'rect' },
  { key: 'vy', label: 'vy', tex: 'v_y', unit: 'm/s', group: 'rect' },
  { key: 'ax', label: 'ax', tex: 'a_x', unit: 'm/s²', group: 'rect' },
  { key: 'ay', label: 'ay', tex: 'a_y', unit: 'm/s²', group: 'rect' },

  { key: 'speed', label: 'v', tex: 'v', unit: 'm/s', group: 'nt' },
  { key: 'psi', label: 'ψ', tex: '\\psi', unit: 'deg', group: 'nt', angle: true },
  { key: 'at', label: 'at', tex: 'a_t', unit: 'm/s²', group: 'nt' },
  { key: 'an', label: 'an', tex: 'a_n', unit: 'm/s²', group: 'nt' },
  { key: 'rho', label: 'ρ', tex: '\\rho', unit: 'm', group: 'nt' },
  { key: 'omega', label: 'ω', tex: '\\omega', unit: 'rad/s', group: 'nt' },
  { key: 'alpha', label: 'α', tex: '\\alpha', unit: 'rad/s²', group: 'nt' },
  { key: 'aMag', label: '|a|', tex: '|\\vec a|', unit: 'm/s²', group: 'nt' },

  { key: 'r', label: 'r', tex: 'r', unit: 'm', group: 'polar' },
  { key: 'theta', label: 'θ', tex: '\\theta', unit: 'deg', group: 'polar', angle: true },
  { key: 'rdot', label: 'ṙ', tex: '\\dot r', unit: 'm/s', group: 'polar' },
  { key: 'thetadot', label: 'θ̇', tex: '\\dot\\theta', unit: 'rad/s', group: 'polar' },
  { key: 'rddot', label: 'r̈', tex: '\\ddot r', unit: 'm/s²', group: 'polar' },
  { key: 'thetaddot', label: 'θ̈', tex: '\\ddot\\theta', unit: 'rad/s²', group: 'polar' },
  { key: 'vr', label: 'vr', tex: 'v_r', unit: 'm/s', group: 'polar' },
  { key: 'vtheta', label: 'vθ', tex: 'v_\\theta', unit: 'm/s', group: 'polar' },
  { key: 'ar', label: 'ar', tex: 'a_r', unit: 'm/s²', group: 'polar' },
  { key: 'atheta', label: 'aθ', tex: 'a_\\theta', unit: 'm/s²', group: 'polar' },

  { key: 's', label: 's', tex: 's', unit: 'm', group: 'initial' },
  { key: 's0', label: 's₀', tex: 's_0', unit: 'm', group: 'initial' },
  { key: 'v0', label: 'v₀', tex: 'v_0', unit: 'm/s', group: 'initial' },
  { key: 'omega0', label: 'ω₀', tex: '\\omega_0', unit: 'rad/s', group: 'initial' },
  { key: 'theta0', label: 'θ₀', tex: '\\theta_0', unit: 'deg', group: 'initial', angle: true },
  { key: 'x0', label: 'x₀', tex: 'x_0', unit: 'm', group: 'initial' },
  { key: 'y0', label: 'y₀', tex: 'y_0', unit: 'm', group: 'initial' },
  { key: 'vx0', label: 'vx₀', tex: 'v_{x0}', unit: 'm/s', group: 'initial' },
  { key: 'vy0', label: 'vy₀', tex: 'v_{y0}', unit: 'm/s', group: 'initial' },
  { key: 'r0', label: 'r₀', tex: 'r_0', unit: 'm', group: 'initial' },
  { key: 'rdot0', label: 'ṙ₀', tex: '\\dot r_0', unit: 'm/s', group: 'initial' },
]

export const QTY_META_BY_KEY: Record<Qty, QtyMeta> = Object.fromEntries(QTY_META.map((m) => [m.key, m])) as Record<Qty, QtyMeta>

export type PathConstraint = 'general' | 'circular' | 'circular-origin'
export type Frame = 'rect' | 'nt' | 'polar'
export type SlotSource = 'input' | 'solved' | 'constraint'

export interface Slot {
  value: number | null
  source: SlotSource | null
  formula: string | null
}

export interface Vec2 {
  x: number
  y: number
}

export interface UnitVectors {
  i: Vec2
  j: Vec2
  e_r: Vec2 | null
  e_theta: Vec2 | null
  e_t: Vec2 | null
  e_n: Vec2 | null
}

export interface ParticleSpec {
  id: string
  label: string
  color: string
  path: PathConstraint
  /** ê_n is 90° CCW from ê_t (path curves left). Used only when n-t → cartesian. */
  curveCCW: boolean
  knowns: Partial<Record<Qty, number>>
}

export interface Conflict {
  key: Qty
  existing: number
  computed: number
  formula: string
}

export interface ParticleSolution {
  id: string
  label: string
  color: string
  slots: Record<Qty, Slot>
  unitVectors: UnitVectors
  conflicts: Conflict[]
  notes: string[]
}

export interface RelativeResult {
  fromId: string
  toId: string
  fromLabel: string
  toLabel: string
  r: Vec2 | null
  v: Vec2 | null
  a: Vec2 | null
  rMag: number | null
  vMag: number | null
  aMag: number | null
}

export interface SolverOutput {
  particles: ParticleSolution[]
  relatives: RelativeResult[]
}

const EPS = 1e-10
const MAX_ITERS = 60

function isValue(v: number): boolean {
  return Number.isFinite(v) || v === Infinity || v === -Infinity
}

function nearlyEqual(a: number, b: number): boolean {
  if (a === b) return true
  if (!Number.isFinite(a) || !Number.isFinite(b)) return false
  return Math.abs(a - b) <= 1e-6 * Math.max(1, Math.abs(a), Math.abs(b))
}

function emptySlots(): Record<Qty, Slot> {
  const slots = {} as Record<Qty, Slot>
  for (const k of QTY_KEYS) slots[k] = { value: null, source: null, formula: null }
  return slots
}

function makeSet(slots: Record<Qty, Slot>, conflicts: Conflict[]) {
  return (key: Qty, value: number, formula: string, source: SlotSource = 'solved'): boolean => {
    if (!isValue(value)) return false
    if (key === 'speed' && value < -EPS) return false
    if (key === 'an' && value < -EPS) return false
    if (key === 'rho' && value < -EPS) return false
    if (key === 'r' && value < -EPS) return false
    if (key === 'aMag' && value < -EPS) return false
    const clamped =
      key === 'speed' || key === 'an' || key === 'aMag' || key === 'r'
        ? Math.max(0, value)
        : key === 'rho'
          ? value < 0
            ? Math.abs(value)
            : value
          : value
    const cur = slots[key]
    if (cur.value === null) {
      slots[key] = { value: clamped, source, formula }
      return true
    }
    if (!nearlyEqual(cur.value, clamped)) {
      const already = conflicts.some((c) => c.key === key && nearlyEqual(c.computed, clamped))
      if (!already) conflicts.push({ key, existing: cur.value, computed: clamped, formula })
    }
    return false
  }
}

type Get = (k: Qty) => number | null
type SetFn = (key: Qty, value: number, formula: string, source?: SlotSource) => boolean

function mul(get: Get, set: SetFn, a: Qty, b: Qty, c: Qty, formula: string): boolean {
  // c = a * b
  const av = get(a)
  const bv = get(b)
  const cv = get(c)
  let ch = false
  if (av !== null && bv !== null) ch = set(c, av * bv, formula) || ch
  if (cv !== null && av !== null && Math.abs(av) > EPS) ch = set(b, cv / av, formula) || ch
  if (cv !== null && bv !== null && Math.abs(bv) > EPS) ch = set(a, cv / bv, formula) || ch
  return ch
}

function hypMag(get: Get, set: SetFn, a: Qty, b: Qty, mag: Qty, formula: string): boolean {
  const av = get(a)
  const bv = get(b)
  if (av !== null && bv !== null) return set(mag, Math.hypot(av, bv), formula)
  return false
}

/** Constant-acceleration 1D (signed): v = v0 + a t, x = x0 + v0 t + ½ a t², v² = v0² + 2 a (x − x0). */
function suvat(get: Get, set: SetFn, keys: { x0: Qty; x: Qty; v0: Qty; v: Qty; a: Qty; t: Qty }, speedLikeV: boolean): boolean {
  const x0 = get(keys.x0)
  const x = get(keys.x)
  const v0 = get(keys.v0)
  const v = get(keys.v)
  const a = get(keys.a)
  const t = get(keys.t)
  let ch = false

  const putV = (value: number, formula: string) => {
    if (speedLikeV && value < -EPS) return false
    return set(keys.v, speedLikeV ? Math.max(0, value) : value, formula)
  }

  if (v0 !== null && a !== null && t !== null) ch = putV(v0 + a * t, `${keys.v} = ${keys.v0} + ${keys.a}\\,t`) || ch
  if (v !== null && a !== null && t !== null) ch = set(keys.v0, v - a * t, `${keys.v0} = ${keys.v} - ${keys.a}\\,t`) || ch
  if (v !== null && v0 !== null && t !== null && Math.abs(t) > EPS) ch = set(keys.a, (v - v0) / t, `${keys.a} = (${keys.v}-${keys.v0})/t`) || ch
  if (v !== null && v0 !== null && a !== null && Math.abs(a) > EPS) {
    const tCand = (v - v0) / a
    if (!speedLikeV || tCand >= -EPS) ch = set(keys.t, tCand, `t = (${keys.v}-${keys.v0})/${keys.a}`) || ch
  }

  if (x0 !== null && v0 !== null && a !== null && t !== null) {
    ch = set(keys.x, x0 + v0 * t + 0.5 * a * t * t, `${keys.x} = ${keys.x0} + ${keys.v0}t + \\tfrac12 ${keys.a}t^2`) || ch
  }
  if (x !== null && v0 !== null && a !== null && t !== null) {
    ch = set(keys.x0, x - v0 * t - 0.5 * a * t * t, `${keys.x0} = ${keys.x} - ${keys.v0}t - \\tfrac12 ${keys.a}t^2`) || ch
  }
  if (x !== null && x0 !== null && v0 !== null && t !== null && Math.abs(t) > EPS) {
    ch = set(keys.a, (2 * (x - x0 - v0 * t)) / (t * t), `${keys.a} = 2(${keys.x}-${keys.x0}-${keys.v0}t)/t^2`) || ch
  }
  if (x !== null && x0 !== null && a !== null && t !== null && Math.abs(t) > EPS) {
    ch = set(keys.v0, (x - x0 - 0.5 * a * t * t) / t, `${keys.v0} = (${keys.x}-${keys.x0}-\\tfrac12 ${keys.a}t^2)/t`) || ch
  }

  if (v0 !== null && a !== null && x0 !== null && x !== null) {
    const rad = v0 * v0 + 2 * a * (x - x0)
    if (rad >= -1e-8) {
      const mag = Math.sqrt(Math.max(0, rad))
      if (v === null) {
        const fromTime = t !== null ? v0 + a * t : null
        let signed = mag
        if (fromTime !== null) signed = fromTime >= 0 ? mag : -mag
        else if (v0 < 0 || (v0 === 0 && a * (x - x0) < 0)) signed = -mag
        ch = putV(signed, `${keys.v}^2 = ${keys.v0}^2 + 2${keys.a}(${keys.x}-${keys.x0})`) || ch
      }
    }
  }

  if (v !== null && v0 !== null && a !== null && Math.abs(a) > EPS && x0 !== null) {
    ch = set(keys.x, x0 + (v * v - v0 * v0) / (2 * a), `${keys.x} = ${keys.x0} + (${keys.v}^2-${keys.v0}^2)/(2${keys.a})`) || ch
  }
  if (v !== null && v0 !== null && a !== null && Math.abs(a) > EPS && x !== null) {
    ch = set(keys.x0, x - (v * v - v0 * v0) / (2 * a), `${keys.x0} = ${keys.x} - (${keys.v}^2-${keys.v0}^2)/(2${keys.a})`) || ch
  }
  if (v !== null && v0 !== null && x !== null && x0 !== null) {
    const dx = x - x0
    if (Math.abs(dx) > EPS) ch = set(keys.a, (v * v - v0 * v0) / (2 * dx), `${keys.a} = (${keys.v}^2-${keys.v0}^2)/(2\\Delta ${keys.x})`) || ch
  }

  // Quadratic for t from x = x0 + v0 t + ½ a t².
  if (t === null && x !== null && x0 !== null && v0 !== null && a !== null) {
    const dx = x - x0
    if (Math.abs(a) < EPS) {
      if (Math.abs(v0) > EPS) {
        const tCand = dx / v0
        if (!speedLikeV || tCand >= -EPS) ch = set(keys.t, tCand, `t = (${keys.x}-${keys.x0})/${keys.v0}`) || ch
      }
    } else {
      const disc = v0 * v0 + 2 * a * dx
      if (disc >= -1e-8) {
        const root = Math.sqrt(Math.max(0, disc))
        const t1 = (-v0 + root) / a
        const t2 = (-v0 - root) / a
        const cands = [t1, t2].filter((ti) => Number.isFinite(ti) && (!speedLikeV || ti >= -1e-9))
        let pick: number | null = null
        if (v !== null) {
          pick = cands.find((ti) => nearlyEqual(v0 + a * ti, v)) ?? null
        }
        if (pick === null) {
          const pos = cands.filter((ti) => ti >= -1e-9)
          pick = pos.length ? Math.min(...pos) : cands.length ? cands[0] : null
        }
        if (pick !== null) ch = set(keys.t, pick, `\\tfrac12 ${keys.a}t^2 + ${keys.v0}t + (${keys.x0}-${keys.x}) = 0`) || ch
      }
    }
  }

  return ch
}

function applyPathConstraints(spec: ParticleSpec, set: SetFn, notes: string[]): void {
  if (spec.path === 'circular' || spec.path === 'circular-origin') {
    notes.push('Circular path: radius of curvature ρ is constant.')
  }
  if (spec.path === 'circular-origin') {
    set('rdot', 0, '\\dot r = 0 \\;\\text{(circular about origin)}', 'constraint')
    set('rddot', 0, '\\ddot r = 0 \\;\\text{(circular about origin)}', 'constraint')
    set('vr', 0, 'v_r = 0 \\;\\text{(circular about origin)}', 'constraint')
    notes.push('Polar origin is the center: r = ρ, ṙ = r̈ = 0.')
  }
}

function propagate(get: Get, set: SetFn, spec: ParticleSpec): boolean {
  let ch = false

  if (spec.path === 'circular-origin') {
    const r = get('r')
    const rho = get('rho')
    if (r !== null) ch = set('rho', r, '\\rho = r \\;\\text{(center at origin)}') || ch
    if (rho !== null) ch = set('r', rho, 'r = \\rho \\;\\text{(center at origin)}') || ch
    // ω = θ̇ and α = θ̈ when the polar origin is the curvature center.
    const om = get('omega')
    const thd = get('thetadot')
    if (om !== null) ch = set('thetadot', om, '\\dot\\theta = \\omega') || ch
    if (thd !== null) ch = set('omega', thd, '\\omega = \\dot\\theta') || ch
    const al = get('alpha')
    const thdd = get('thetaddot')
    if (al !== null) ch = set('thetaddot', al, '\\ddot\\theta = \\alpha') || ch
    if (thdd !== null) ch = set('alpha', thdd, '\\alpha = \\ddot\\theta') || ch
  }

  // Polar: vr = ṙ, vθ = r θ̇.
  {
    const rdot = get('rdot')
    const vr = get('vr')
    if (rdot !== null) ch = set('vr', rdot, 'v_r = \\dot r') || ch
    if (vr !== null) ch = set('rdot', vr, '\\dot r = v_r') || ch
  }
  ch = mul(get, set, 'r', 'thetadot', 'vtheta', 'v_\\theta = r\\dot\\theta') || ch

  // ar = r̈ − r θ̇² ,  aθ = r θ̈ + 2 ṙ θ̇
  {
    const r = get('r')
    const rddot = get('rddot')
    const thd = get('thetadot')
    const ar = get('ar')
    if (r !== null && thd !== null && rddot !== null) ch = set('ar', rddot - r * thd * thd, 'a_r = \\ddot r - r\\dot\\theta^2') || ch
    if (r !== null && thd !== null && ar !== null) ch = set('rddot', ar + r * thd * thd, '\\ddot r = a_r + r\\dot\\theta^2') || ch
    if (ar !== null && rddot !== null && thd !== null && thd * thd > EPS) {
      ch = set('r', (rddot - ar) / (thd * thd), 'r = (\\ddot r - a_r)/\\dot\\theta^2') || ch
    }

    const rdot = get('rdot')
    const thdd = get('thetaddot')
    const atheta = get('atheta')
    if (r !== null && thdd !== null && rdot !== null && thd !== null) {
      ch = set('atheta', r * thdd + 2 * rdot * thd, 'a_\\theta = r\\ddot\\theta + 2\\dot r\\dot\\theta') || ch
    }
    if (atheta !== null && r !== null && rdot !== null && thd !== null && Math.abs(r) > EPS) {
      ch = set('thetaddot', (atheta - 2 * rdot * thd) / r, '\\ddot\\theta = (a_\\theta - 2\\dot r\\dot\\theta)/r') || ch
    }
    if (atheta !== null && r !== null && thdd !== null && thd !== null && Math.abs(thd) > EPS) {
      ch = set('rdot', (atheta - r * thdd) / (2 * thd), '\\dot r = (a_\\theta - r\\ddot\\theta)/(2\\dot\\theta)') || ch
    }
    if (atheta !== null && thdd !== null && rdot !== null && thd !== null && Math.abs(thdd) > EPS) {
      ch = set('r', (atheta - 2 * rdot * thd) / thdd, 'r = (a_\\theta - 2\\dot r\\dot\\theta)/\\ddot\\theta') || ch
    }
  }

  ch = hypMag(get, set, 'vr', 'vtheta', 'speed', 'v = \\sqrt{v_r^2 + v_\\theta^2}') || ch
  ch = hypMag(get, set, 'ar', 'atheta', 'aMag', '|\\vec a| = \\sqrt{a_r^2 + a_\\theta^2}') || ch

  // Polar ↔ rectangular.
  {
    const r = get('r')
    const th = get('theta')
    const x = get('x')
    const y = get('y')
    if (r !== null && th !== null) {
      ch = set('x', r * Math.cos(th), 'x = r\\cos\\theta') || ch
      ch = set('y', r * Math.sin(th), 'y = r\\sin\\theta') || ch
    }
    if (x !== null && y !== null) {
      ch = set('r', Math.hypot(x, y), 'r = \\sqrt{x^2+y^2}') || ch
      if (Math.hypot(x, y) > EPS) ch = set('theta', Math.atan2(y, x), '\\theta = \\operatorname{atan2}(y,x)') || ch
    }

    const vr = get('vr')
    const vth = get('vtheta')
    const ar = get('ar')
    const ath = get('atheta')
    if (th !== null && vr !== null && vth !== null) {
      ch = set('vx', vr * Math.cos(th) - vth * Math.sin(th), 'v_x = v_r\\cos\\theta - v_\\theta\\sin\\theta') || ch
      ch = set('vy', vr * Math.sin(th) + vth * Math.cos(th), 'v_y = v_r\\sin\\theta + v_\\theta\\cos\\theta') || ch
    }
    if (th !== null && ar !== null && ath !== null) {
      ch = set('ax', ar * Math.cos(th) - ath * Math.sin(th), 'a_x = a_r\\cos\\theta - a_\\theta\\sin\\theta') || ch
      ch = set('ay', ar * Math.sin(th) + ath * Math.cos(th), 'a_y = a_r\\sin\\theta + a_\\theta\\cos\\theta') || ch
    }
    const vx = get('vx')
    const vy = get('vy')
    const ax = get('ax')
    const ay = get('ay')
    if (th !== null && vx !== null && vy !== null) {
      ch = set('vr', vx * Math.cos(th) + vy * Math.sin(th), 'v_r = v_x\\cos\\theta + v_y\\sin\\theta') || ch
      ch = set('vtheta', -vx * Math.sin(th) + vy * Math.cos(th), 'v_\\theta = -v_x\\sin\\theta + v_y\\cos\\theta') || ch
    }
    if (th !== null && ax !== null && ay !== null) {
      ch = set('ar', ax * Math.cos(th) + ay * Math.sin(th), 'a_r = a_x\\cos\\theta + a_y\\sin\\theta') || ch
      ch = set('atheta', -ax * Math.sin(th) + ay * Math.cos(th), 'a_\\theta = -a_x\\sin\\theta + a_y\\cos\\theta') || ch
    }
  }

  // Rectangular magnitudes.
  ch = hypMag(get, set, 'vx', 'vy', 'speed', 'v = \\sqrt{v_x^2 + v_y^2}') || ch
  ch = hypMag(get, set, 'ax', 'ay', 'aMag', '|\\vec a| = \\sqrt{a_x^2 + a_y^2}') || ch
  ch = hypMag(get, set, 'at', 'an', 'aMag', '|\\vec a| = \\sqrt{a_t^2 + a_n^2}') || ch

  // n-t from cartesian.
  {
    const vx = get('vx')
    const vy = get('vy')
    const ax = get('ax')
    const ay = get('ay')
    const speed = get('speed')
    if (vx !== null && vy !== null && Math.hypot(vx, vy) > EPS) {
      ch = set('psi', Math.atan2(vy, vx), '\\psi = \\operatorname{atan2}(v_y,v_x)') || ch
    }
    if (vx !== null && vy !== null && ax !== null && ay !== null) {
      const sp = Math.hypot(vx, vy)
      if (sp > EPS) {
        const at = (vx * ax + vy * ay) / sp
        ch = set('at', at, 'a_t = \\vec a\\cdot\\hat e_t') || ch
        const aMagSq = ax * ax + ay * ay
        const an = Math.sqrt(Math.max(0, aMagSq - at * at))
        ch = set('an', an, 'a_n = \\sqrt{|\\vec a|^2 - a_t^2}') || ch
      } else {
        ch = set('at', 0, 'a_t = 0 \\;(v=0)') || ch
        ch = set('an', Math.hypot(ax, ay), 'a_n = |\\vec a| \\;(v=0)') || ch
      }
    }
    if (speed !== null && get('psi') !== null) {
      const psi = get('psi')!
      ch = set('vx', speed * Math.cos(psi), 'v_x = v\\cos\\psi') || ch
      ch = set('vy', speed * Math.sin(psi), 'v_y = v\\sin\\psi') || ch
    }
  }

  // an = v²/ρ , ω = ±v/ρ , α = ±a_t/ρ
  {
    const speed = get('speed')
    const an = get('an')
    const rho = get('rho')
    const at = get('at')
    const sign = spec.curveCCW ? 1 : -1
    if (speed !== null && rho !== null && Number.isFinite(rho) && Math.abs(rho) > EPS) {
      ch = set('an', (speed * speed) / Math.abs(rho), 'a_n = v^2/\\rho') || ch
      ch = set('omega', (sign * speed) / rho, '\\omega = \\pm v/\\rho') || ch
    }
    if (speed !== null && an !== null) {
      if (an < EPS) {
        if (speed > EPS) ch = set('rho', Infinity, '\\rho \\to \\infty \\;(a_n=0)') || ch
        ch = set('omega', 0, '\\omega = 0 \\;(straight)') || ch
      } else {
        ch = set('rho', (speed * speed) / an, '\\rho = v^2/a_n') || ch
      }
    }
    if (an !== null && rho !== null && Number.isFinite(rho) && Math.abs(rho) > EPS) {
      ch = set('speed', Math.sqrt(an * Math.abs(rho)), 'v = \\sqrt{a_n\\rho}') || ch
    }
    if (at !== null && rho !== null && Number.isFinite(rho) && Math.abs(rho) > EPS) {
      ch = set('alpha', (sign * at) / rho, '\\alpha = a_t/\\rho') || ch
    }
    if (get('alpha') !== null && rho !== null && Number.isFinite(rho) && Math.abs(rho) > EPS) {
      ch = set('at', sign * get('alpha')! * rho, 'a_t = \\alpha\\rho') || ch
    }
    if (get('omega') !== null && rho !== null && Number.isFinite(rho) && Math.abs(rho) > EPS) {
      const om = get('omega')!
      ch = set('speed', Math.abs(om * rho), 'v = |\\omega|\\rho') || ch
    }
    if (get('omega') !== null && speed !== null && Math.abs(speed) > EPS && Number.isFinite(get('omega')!)) {
      const om = get('omega')!
      if (Math.abs(om) > EPS) ch = set('rho', speed / Math.abs(om), '\\rho = v/|\\omega|') || ch
    }
  }

  // n-t → cartesian using ê_t, ê_n.
  {
    const psi = get('psi')
    const at = get('at')
    const an = get('an')
    if (psi !== null && at !== null && an !== null) {
      const etx = Math.cos(psi)
      const ety = Math.sin(psi)
      const enx = spec.curveCCW ? -ety : ety
      const eny = spec.curveCCW ? etx : -etx
      ch = set('ax', at * etx + an * enx, '\\vec a = a_t\\hat e_t + a_n\\hat e_n') || ch
      ch = set('ay', at * ety + an * eny, '\\vec a = a_t\\hat e_t + a_n\\hat e_n') || ch
    }
  }

  // Constant-accel time relations.
  ch =
    suvat(get, set, { x0: 's0', x: 's', v0: 'v0', v: 'speed', a: 'at', t: 't' }, true) || ch
  ch = suvat(get, set, { x0: 'x0', x: 'x', v0: 'vx0', v: 'vx', a: 'ax', t: 't' }, false) || ch
  ch = suvat(get, set, { x0: 'y0', x: 'y', v0: 'vy0', v: 'vy', a: 'ay', t: 't' }, false) || ch
  ch =
    suvat(get, set, { x0: 'theta0', x: 'theta', v0: 'omega0', v: 'omega', a: 'alpha', t: 't' }, false) || ch
  ch =
    suvat(get, set, { x0: 'r0', x: 'r', v0: 'rdot0', v: 'rdot', a: 'rddot', t: 't' }, false) || ch

  return ch
}

function unitVectorsOf(get: Get, spec: ParticleSpec): UnitVectors {
  const theta = get('theta')
  const psi = get('psi')
  const vx = get('vx')
  const vy = get('vy')
  const ax = get('ax')
  const ay = get('ay')

  let e_r: Vec2 | null = null
  let e_theta: Vec2 | null = null
  if (theta !== null) {
    e_r = { x: Math.cos(theta), y: Math.sin(theta) }
    e_theta = { x: -Math.sin(theta), y: Math.cos(theta) }
  }

  let e_t: Vec2 | null = null
  if (psi !== null) e_t = { x: Math.cos(psi), y: Math.sin(psi) }
  else if (vx !== null && vy !== null) {
    const sp = Math.hypot(vx, vy)
    if (sp > EPS) e_t = { x: vx / sp, y: vy / sp }
  }

  let e_n: Vec2 | null = null
  if (e_t && ax !== null && ay !== null) {
    const at = ax * e_t.x + ay * e_t.y
    const nx = ax - at * e_t.x
    const ny = ay - at * e_t.y
    const nmag = Math.hypot(nx, ny)
    if (nmag > EPS) e_n = { x: nx / nmag, y: ny / nmag }
  }
  if (!e_n && e_t) {
    e_n = spec.curveCCW ? { x: -e_t.y, y: e_t.x } : { x: e_t.y, y: -e_t.x }
  }

  return {
    i: { x: 1, y: 0 },
    j: { x: 0, y: 1 },
    e_r,
    e_theta,
    e_t,
    e_n,
  }
}

export function solveParticle(spec: ParticleSpec): ParticleSolution {
  const slots = emptySlots()
  const conflicts: Conflict[] = []
  const notes: string[] = []
  const set = makeSet(slots, conflicts)
  const get: Get = (k) => slots[k].value

  for (const key of QTY_KEYS) {
    const v = spec.knowns[key]
    if (v !== undefined && isValue(v)) {
      slots[key] = { value: v, source: 'input', formula: null }
    }
  }
  applyPathConstraints(spec, set, notes)

  for (let i = 0; i < MAX_ITERS; i++) {
    if (!propagate(get, set, spec)) break
  }

  return {
    id: spec.id,
    label: spec.label,
    color: spec.color,
    slots,
    unitVectors: unitVectorsOf(get, spec),
    conflicts,
    notes,
  }
}

function slotVec(sol: ParticleSolution, xk: Qty, yk: Qty): Vec2 | null {
  const x = sol.slots[xk].value
  const y = sol.slots[yk].value
  if (x === null || y === null) return null
  return { x, y }
}

export function relativeOf(a: ParticleSolution, b: ParticleSolution): RelativeResult {
  const rA = slotVec(a, 'x', 'y')
  const rB = slotVec(b, 'x', 'y')
  const vA = slotVec(a, 'vx', 'vy')
  const vB = slotVec(b, 'vx', 'vy')
  const aA = slotVec(a, 'ax', 'ay')
  const aB = slotVec(b, 'ax', 'ay')
  const r = rA && rB ? { x: rA.x - rB.x, y: rA.y - rB.y } : null
  const v = vA && vB ? { x: vA.x - vB.x, y: vA.y - vB.y } : null
  const acc = aA && aB ? { x: aA.x - aB.x, y: aA.y - aB.y } : null
  return {
    fromId: a.id,
    toId: b.id,
    fromLabel: a.label,
    toLabel: b.label,
    r,
    v,
    a: acc,
    rMag: r ? Math.hypot(r.x, r.y) : null,
    vMag: v ? Math.hypot(v.x, v.y) : null,
    aMag: acc ? Math.hypot(acc.x, acc.y) : null,
  }
}

export function solveCurvilinear(particles: ParticleSpec[]): SolverOutput {
  const solved = particles.map(solveParticle)
  const relatives: RelativeResult[] = []
  for (let i = 0; i < solved.length; i++) {
    for (let j = 0; j < solved.length; j++) {
      if (i === j) continue
      relatives.push(relativeOf(solved[i], solved[j]))
    }
  }
  return { particles: solved, relatives }
}

export function formatVectorTex(x: number, y: number, iHat = '\\hat{\\imath}', jHat = '\\hat{\\jmath}', precision = 3): string {
  const ax = Math.abs(x) < 1e-12 ? 0 : x
  const ay = Math.abs(y) < 1e-12 ? 0 : y
  if (ax === 0 && ay === 0) return `0`
  if (ay === 0) return `${formatNumber(ax, precision)}\\,${iHat}`
  if (ax === 0) return `${ay < 0 ? '-' : ''}${formatNumber(Math.abs(ay), precision)}\\,${jHat}`
  const ySign = ay < 0 ? '-' : '+'
  return `${formatNumber(ax, precision)}\\,${iHat} ${ySign} ${formatNumber(Math.abs(ay), precision)}\\,${jHat}`
}

export function formatUnitVecTex(v: Vec2 | null, name: string): string | null {
  if (!v) return null
  return `${name} = ${formatVectorTex(v.x, v.y)}`
}

export function slotNumber(sol: ParticleSolution, key: Qty): number | null {
  return sol.slots[key].value
}

const CHECK_TOL = 1e-6

function expectClose(name: string, got: number | null, want: number, errors: string[]): void {
  if (got === null || !Number.isFinite(got) || Math.abs(got - want) > CHECK_TOL * Math.max(1, Math.abs(want))) {
    errors.push(`${name}: expected ${want}, got ${got}`)
  }
}

/** Textbook-style identities used as a regression harness (called from a node runner). */
export function runCurvilinearSolverChecks(): string[] {
  const errors: string[] = []

  const circular = solveParticle({
    id: 'A',
    label: 'A',
    color: '#29d3f5',
    path: 'circular',
    curveCCW: true,
    knowns: { rho: 200, speed: 25, at: 3, psi: 0 },
  })
  expectClose('circ an', circular.slots.an.value, (25 * 25) / 200, errors)
  expectClose('circ omega', circular.slots.omega.value, 25 / 200, errors)
  expectClose('circ alpha', circular.slots.alpha.value, 3 / 200, errors)
  expectClose('circ vx', circular.slots.vx.value, 25, errors)
  expectClose('circ vy', circular.slots.vy.value, 0, errors)
  const aMag = Math.hypot(3, (25 * 25) / 200)
  expectClose('circ |a|', circular.slots.aMag.value, aMag, errors)

  const polar = solveParticle({
    id: 'P',
    label: 'P',
    color: '#f5a524',
    path: 'general',
    curveCCW: true,
    knowns: { r: 4, theta: Math.PI / 6, rdot: 1.5, thetadot: 2, rddot: 0, thetaddot: 0 },
  })
  expectClose('polar vr', polar.slots.vr.value, 1.5, errors)
  expectClose('polar vθ', polar.slots.vtheta.value, 8, errors)
  expectClose('polar ar', polar.slots.ar.value, -16, errors)
  expectClose('polar aθ', polar.slots.atheta.value, 6, errors)
  expectClose('polar x', polar.slots.x.value, 4 * Math.cos(Math.PI / 6), errors)
  expectClose('polar y', polar.slots.y.value, 4 * Math.sin(Math.PI / 6), errors)

  const rect = solveParticle({
    id: 'R',
    label: 'R',
    color: '#a78bfa',
    path: 'general',
    curveCCW: true,
    knowns: { vx: 3, vy: 4, ax: 0, ay: -10 },
  })
  expectClose('rect v', rect.slots.speed.value, 5, errors)
  expectClose('rect at', rect.slots.at.value, -8, errors)

  const timed = solveParticle({
    id: 'T',
    label: 'T',
    color: '#59d67f',
    path: 'general',
    curveCCW: true,
    knowns: { v0: 10, at: 2, t: 3, s0: 0 },
  })
  expectClose('time v', timed.slots.speed.value, 16, errors)
  expectClose('time s', timed.slots.s.value, 10 * 3 + 0.5 * 2 * 9, errors)

  const originCirc = solveParticle({
    id: 'C',
    label: 'C',
    color: '#fb6a6a',
    path: 'circular-origin',
    curveCCW: true,
    knowns: { r: 2, thetadot: 3, thetaddot: 1, theta: 0 },
  })
  expectClose('oc vθ', originCirc.slots.vtheta.value, 6, errors)
  expectClose('oc speed', originCirc.slots.speed.value, 6, errors)
  expectClose('oc ar', originCirc.slots.ar.value, -18, errors)
  expectClose('oc aθ', originCirc.slots.atheta.value, 2, errors)
  expectClose('oc an', originCirc.slots.an.value, 18, errors)
  expectClose('oc rho', originCirc.slots.rho.value, 2, errors)

  const a = solveParticle({
    id: 'A',
    label: 'A',
    color: '#29d3f5',
    path: 'general',
    curveCCW: true,
    knowns: { x: 0, y: 0, vx: 8, vy: 0, ax: 0, ay: 2 },
  })
  const b = solveParticle({
    id: 'B',
    label: 'B',
    color: '#f5a524',
    path: 'general',
    curveCCW: true,
    knowns: { x: 4, y: 3, vx: 2, vy: 4, ax: -1, ay: 0 },
  })
  const rel = relativeOf(a, b)
  expectClose('rel rx', rel.r?.x ?? null, -4, errors)
  expectClose('rel ry', rel.r?.y ?? null, -3, errors)
  expectClose('rel vx', rel.v?.x ?? null, 6, errors)
  expectClose('rel vy', rel.v?.y ?? null, -4, errors)
  expectClose('rel ax', rel.a?.x ?? null, 1, errors)
  expectClose('rel ay', rel.a?.y ?? null, 2, errors)

  const tFromSuvat = solveParticle({
    id: 'S',
    label: 'S',
    color: '#fff',
    path: 'general',
    curveCCW: true,
    knowns: { s0: 0, s: 20, v0: 0, at: 4 },
  })
  expectClose('suvat t', tFromSuvat.slots.t.value, Math.sqrt(10), errors)

  return errors
}

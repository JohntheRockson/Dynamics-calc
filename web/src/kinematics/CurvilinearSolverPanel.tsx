import { useMemo, useState } from 'react'
import { OptionalNumberField, Segmented, SelectField, ToggleField } from '../components/fields'
import { formatNumber } from '../format'
import { Eq, EqList } from './Eq'
import { Scene2D, type Scene2DBody, type Scene2DVector } from './Scene2D'
import {
  formatUnitVecTex,
  formatVectorTex,
  QTY_META,
  QTY_META_BY_KEY,
  solveCurvilinear,
  type Frame,
  type ParticleSolution,
  type ParticleSpec,
  type PathConstraint,
  type Qty,
  type QtyGroup,
  type RelativeResult,
  type Slot,
  type Vec2,
} from './curvilinearSolver'

const COLORS = ['#29d3f5', '#f5a524', '#a78bfa', '#fb6a6a', '#59d67f', '#5aa8ff']
const LABELS = ['A', 'B', 'C', 'D', 'E', 'F']
const MAX_PARTICLES = 6

interface ParticleState {
  id: string
  label: string
  color: string
  path: PathConstraint
  curveCCW: boolean
  frame: Frame
  showAll: boolean
  knowns: Partial<Record<Qty, number>>
}

let nextId = 1

function newParticle(index: number, knowns: Partial<Record<Qty, number>> = {}, extras: Partial<ParticleState> = {}): ParticleState {
  return {
    id: `p${nextId++}`,
    label: LABELS[index] ?? `P${index + 1}`,
    color: COLORS[index % COLORS.length],
    path: 'general',
    curveCCW: true,
    frame: 'nt',
    showAll: false,
    knowns,
    ...extras,
  }
}

const PRESETS: { key: string; label: string; particles: () => ParticleState[] }[] = [
  {
    key: 'circular',
    label: 'Circular track',
    particles: () => [
      newParticle(0, { rho: 200, speed: 25, at: 3, psi: 0 }, { frame: 'nt', path: 'circular' }),
    ],
  },
  {
    key: 'polar',
    label: 'Polar r–θ',
    particles: () => [
      newParticle(
        0,
        { r: 4, theta: Math.PI / 6, rdot: 1.5, thetadot: 2, rddot: 0, thetaddot: 0 },
        { frame: 'polar', path: 'general', showAll: false },
      ),
    ],
  },
  {
    key: 'rect',
    label: 'Rectangular',
    particles: () => [newParticle(0, { x: 2, y: 3, vx: 3, vy: 4, ax: 1, ay: -2 }, { frame: 'rect' })],
  },
  {
    key: 'relative',
    label: 'Two points',
    particles: () => [
      newParticle(0, { x: 0, y: 0, vx: 8, vy: 0, ax: 0, ay: 2 }, { frame: 'rect' }),
      newParticle(1, { x: 4, y: 3, vx: 2, vy: 4, ax: -1, ay: 0 }, { frame: 'rect' }),
    ],
  },
  {
    key: 'orbit',
    label: 'Circle about origin',
    particles: () => [
      newParticle(0, { r: 2, theta: 0, thetadot: 3, thetaddot: 1 }, { frame: 'polar', path: 'circular-origin' }),
    ],
  },
]

function toSpec(p: ParticleState): ParticleSpec {
  return {
    id: p.id,
    label: p.label,
    color: p.color,
    path: p.path,
    curveCCW: p.curveCCW,
    knowns: p.knowns,
  }
}

function displayOf(key: Qty, si: number | null): number | null {
  if (si === null) return null
  if (!Number.isFinite(si)) return si
  return QTY_META_BY_KEY[key].angle ? (si * 180) / Math.PI : si
}

function siOf(key: Qty, display: number | null): number | null {
  if (display === null) return null
  return QTY_META_BY_KEY[key].angle ? (display * Math.PI) / 180 : display
}

function setKnown(p: ParticleState, key: Qty, display: number | null): ParticleState {
  const knowns = { ...p.knowns }
  const si = siOf(key, display)
  if (si === null) delete knowns[key]
  else knowns[key] = si
  return { ...p, knowns }
}

function groupsFor(frame: Frame, showAll: boolean): QtyGroup[] {
  if (showAll) return ['time', 'rect', 'nt', 'polar', 'initial']
  if (frame === 'rect') return ['time', 'rect', 'initial']
  if (frame === 'polar') return ['time', 'polar', 'initial']
  return ['time', 'nt', 'initial']
}

function circlePoly(cx: number, cy: number, rho: number, n = 72): { x: number; y: number }[] {
  if (!Number.isFinite(rho) || rho <= 0) return []
  const pts: { x: number; y: number }[] = []
  for (let i = 0; i <= n; i++) {
    const a = (2 * Math.PI * i) / n
    pts.push({ x: cx + rho * Math.cos(a), y: cy + rho * Math.sin(a) })
  }
  return pts
}

function qtyUnit(key: Qty): string {
  return QTY_META_BY_KEY[key].unit
}

function formatSlotValue(key: Qty, slot: Slot): string {
  if (slot.value === null) return '—'
  if (slot.value === Infinity) return '∞'
  if (slot.value === -Infinity) return '-∞'
  const shown = displayOf(key, slot.value)
  if (shown === null || !Number.isFinite(shown)) return '—'
  return formatNumber(shown, QTY_META_BY_KEY[key].angle ? 2 : 4)
}

function ParticleCard({
  particle,
  canRemove,
  onChange,
  onRemove,
}: {
  particle: ParticleState
  canRemove: boolean
  onChange: (p: ParticleState) => void
  onRemove: () => void
}) {
  const groups = groupsFor(particle.frame, particle.showAll)
  return (
    <div className="panel solver-particle" style={{ borderTop: `3px solid ${particle.color}` }}>
      <div className="panel-header">
        <h2>
          Point {particle.label}
        </h2>
        {canRemove && (
          <button type="button" className="btn btn-ghost" onClick={onRemove} aria-label={`Remove point ${particle.label}`}>
            Remove
          </button>
        )}
      </div>
      <div className="panel-body" style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        <Segmented
          value={particle.frame}
          onChange={(frame: Frame) => onChange({ ...particle, frame })}
          options={[
            { value: 'rect', label: 'Rectangular' },
            { value: 'nt', label: 'Normal–tangential' },
            { value: 'polar', label: 'Radial–transverse' },
          ]}
        />
        <SelectField
          label="Path"
          value={particle.path}
          onChange={(path: PathConstraint) => onChange({ ...particle, path })}
          options={[
            { value: 'general', label: 'General curvilinear' },
            { value: 'circular', label: 'Circular (constant ρ)' },
            { value: 'circular-origin', label: 'Circular about origin (r = ρ)' },
          ]}
        />
        <ToggleField
          label="Path curves left (êₙ 90° CCW from êₜ)"
          checked={particle.curveCCW}
          onChange={(curveCCW) => onChange({ ...particle, curveCCW })}
        />
        <ToggleField label="Show every component set" checked={particle.showAll} onChange={(showAll) => onChange({ ...particle, showAll })} />
        <p className="hint">Leave a box empty to treat it as unknown. Mix frames if the problem gives mixed data.</p>
        {groups.map((group) => (
          <QtyGroupFields key={group} group={group} particle={particle} onChange={onChange} />
        ))}
      </div>
    </div>
  )
}

function QtyGroupFields({
  group,
  particle,
  onChange,
}: {
  group: QtyGroup
  particle: ParticleState
  onChange: (p: ParticleState) => void
}) {
  const titles: Record<QtyGroup, string> = {
    time: 'Time',
    rect: 'Rectangular (x, y)',
    nt: 'Normal–tangential',
    polar: 'Radial & transverse (r, θ)',
    initial: 'Constant-accel initials (optional)',
  }
  const keys = QTY_META.filter((m) => m.group === group)
  const body = (
    <div className="solver-qty-grid">
      {keys.map((m) => (
        <OptionalNumberField
          key={m.key}
          label={m.label}
          unit={m.unit}
          value={displayOf(m.key, particle.knowns[m.key] ?? null)}
          onChange={(v) => onChange(setKnown(particle, m.key, v))}
        />
      ))}
    </div>
  )
  if (group === 'initial') {
    return (
      <details className="solver-group">
        <summary className="solver-group-title">{titles[group]}</summary>
        {body}
      </details>
    )
  }
  return (
    <div className="solver-group">
      <div className="solver-group-title">{titles[group]}</div>
      {body}
    </div>
  )
}

function sourceBadge(source: Slot['source']): string {
  if (source === 'input') return 'given'
  if (source === 'constraint') return 'path'
  if (source === 'solved') return 'solved'
  return ''
}

function ScalarTable({ sol, group }: { sol: ParticleSolution; group: QtyGroup }) {
  const rows = QTY_META.filter((m) => m.group === group && sol.slots[m.key].value !== null)
  if (rows.length === 0) return null
  return (
    <div className="solver-table-wrap">
      <table className="solver-table">
        <thead>
          <tr>
            <th>Qty</th>
            <th>Value</th>
            <th>Unit</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {rows.map((m) => {
            const slot = sol.slots[m.key]
            return (
              <tr key={m.key} className={slot.source === 'input' ? 'is-given' : 'is-solved'}>
                <td>
                  <Eq tex={m.tex} />
                </td>
                <td className="num">{formatSlotValue(m.key, slot)}</td>
                <td className="unit">{qtyUnit(m.key)}</td>
                <td>
                  <span className={`badge ${slot.source === 'input' ? 'badge-given' : 'badge-solved'}`}>{sourceBadge(slot.source)}</span>
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

function vectorEqs(sol: ParticleSolution): string[] {
  const x = sol.slots.x.value
  const y = sol.slots.y.value
  const vx = sol.slots.vx.value
  const vy = sol.slots.vy.value
  const ax = sol.slots.ax.value
  const ay = sol.slots.ay.value
  const speed = sol.slots.speed.value
  const at = sol.slots.at.value
  const an = sol.slots.an.value
  const vr = sol.slots.vr.value
  const vth = sol.slots.vtheta.value
  const ar = sol.slots.ar.value
  const ath = sol.slots.atheta.value
  const items: string[] = []
  if (x !== null && y !== null) items.push(`\\vec r = ${formatVectorTex(x, y)}`)
  if (vx !== null && vy !== null) {
    let rhs = formatVectorTex(vx, vy)
    if (speed !== null) rhs += ` = ${formatNumber(speed, 3)}\\,\\hat e_t`
    if (vr !== null && vth !== null) {
      rhs += ` = ${formatNumber(vr, 3)}\\,\\hat e_r + ${formatNumber(vth, 3)}\\,\\hat e_\\theta`
    }
    items.push(`\\vec v = ${rhs}`)
  }
  if (ax !== null && ay !== null) {
    let rhs = formatVectorTex(ax, ay)
    if (at !== null && an !== null) rhs += ` = ${formatNumber(at, 3)}\\,\\hat e_t + ${formatNumber(an, 3)}\\,\\hat e_n`
    if (ar !== null && ath !== null) rhs += ` = ${formatNumber(ar, 3)}\\,\\hat e_r + ${formatNumber(ath, 3)}\\,\\hat e_\\theta`
    items.push(`\\vec a = ${rhs}`)
  }
  return items
}

function unitEqs(sol: ParticleSolution): string[] {
  const u = sol.unitVectors
  const items = [
    formatUnitVecTex(u.i, '\\hat{\\imath}'),
    formatUnitVecTex(u.j, '\\hat{\\jmath}'),
    formatUnitVecTex(u.e_t, '\\hat e_t'),
    formatUnitVecTex(u.e_n, '\\hat e_n'),
    formatUnitVecTex(u.e_r, '\\hat e_r'),
    formatUnitVecTex(u.e_theta, '\\hat e_\\theta'),
  ]
  return items.filter((s): s is string => s !== null)
}

function RelativeCard({ rel }: { rel: RelativeResult }) {
  const tag = `${rel.fromLabel}/${rel.toLabel}`
  const items: string[] = []
  if (rel.r) items.push(`\\vec r_{${tag}} = ${formatVectorTex(rel.r.x, rel.r.y)}`)
  if (rel.v) items.push(`\\vec v_{${tag}} = ${formatVectorTex(rel.v.x, rel.v.y)}`)
  if (rel.a) items.push(`\\vec a_{${tag}} = ${formatVectorTex(rel.a.x, rel.a.y)}`)
  if (items.length === 0) {
    return (
      <p className="hint">
        Need rectangular (or convertible) position/velocity/acceleration on both {rel.fromLabel} and {rel.toLabel}.
      </p>
    )
  }
  return (
    <div className="solver-rel-card">
      <div className="solver-group-title">
        {rel.fromLabel} relative to {rel.toLabel}
      </div>
      <EqList items={items} />
      <div className="summary-grid">
        {rel.rMag !== null && (
          <div className="stat-card">
            <div className="label">|r_{tag}|</div>
            <div className="val">
              {formatNumber(rel.rMag, 3)}
              <span className="unit">m</span>
            </div>
          </div>
        )}
        {rel.vMag !== null && (
          <div className="stat-card">
            <div className="label">|v_{tag}|</div>
            <div className="val">
              {formatNumber(rel.vMag, 3)}
              <span className="unit">m/s</span>
            </div>
          </div>
        )}
        {rel.aMag !== null && (
          <div className="stat-card">
            <div className="label">|a_{tag}|</div>
            <div className="val">
              {formatNumber(rel.aMag, 3)}
              <span className="unit">m/s²</span>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

function vec(sol: ParticleSolution, xk: Qty, yk: Qty): Vec2 | null {
  const x = sol.slots[xk].value
  const y = sol.slots[yk].value
  if (x === null || y === null || !Number.isFinite(x) || !Number.isFinite(y)) return null
  return { x, y }
}

function sceneBodies(
  particles: ParticleState[],
  solved: ParticleSolution[],
  relatives: RelativeResult[],
  show: { v: boolean; a: boolean; et: boolean; en: boolean; er: boolean; eth: boolean; rel: boolean },
): Scene2DBody[] {
  const bodies: Scene2DBody[] = []
  for (const sol of solved) {
    const spec = particles.find((p) => p.id === sol.id)
    const pos = vec(sol, 'x', 'y')
    const displayPos = pos ?? { x: 0, y: 0 }
    const rho = sol.slots.rho.value
    const pathKind = spec?.path ?? 'general'
    if (pathKind === 'circular-origin' && rho !== null && Number.isFinite(rho) && rho > 0) {
      bodies.push({
        label: `${sol.label} path`,
        color: sol.color,
        dashed: true,
        hideMarker: true,
        path: circlePoly(0, 0, rho),
        currentIndex: 0,
      })
    } else if (pathKind === 'circular' && sol.unitVectors.e_n && rho !== null && Number.isFinite(rho) && rho > 0) {
      const cx = displayPos.x + rho * sol.unitVectors.e_n.x
      const cy = displayPos.y + rho * sol.unitVectors.e_n.y
      bodies.push({
        label: `${sol.label} path`,
        color: sol.color,
        dashed: true,
        hideMarker: true,
        path: circlePoly(cx, cy, rho),
        currentIndex: 0,
      })
    }
    const extraVectors: Scene2DVector[] = []
    const vel = vec(sol, 'vx', 'vy')
    const acc = vec(sol, 'ax', 'ay')
    if (show.v && vel) extraVectors.push({ vx: vel.x, vy: vel.y, color: sol.color })
    if (show.a && acc) extraVectors.push({ vx: acc.x, vy: acc.y, color: '#fb7185' })
    if (show.et && sol.unitVectors.e_t) extraVectors.push({ vx: sol.unitVectors.e_t.x, vy: sol.unitVectors.e_t.y, color: '#29d3f5', unit: true })
    if (show.en && sol.unitVectors.e_n) extraVectors.push({ vx: sol.unitVectors.e_n.x, vy: sol.unitVectors.e_n.y, color: '#a78bfa', unit: true })
    if (show.er && sol.unitVectors.e_r) extraVectors.push({ vx: sol.unitVectors.e_r.x, vy: sol.unitVectors.e_r.y, color: '#59d67f', unit: true, dashed: true })
    if (show.eth && sol.unitVectors.e_theta) extraVectors.push({ vx: sol.unitVectors.e_theta.x, vy: sol.unitVectors.e_theta.y, color: '#f5a524', unit: true, dashed: true })
    const hasAnything = pos !== null || vel !== null || extraVectors.length > 0 || pathKind !== 'general'
    if (hasAnything) {
      bodies.push({
        label: sol.label,
        color: sol.color,
        path: [displayPos],
        currentIndex: 0,
        extraVectors,
      })
    }
  }
  if (show.rel) {
    for (const rel of relatives) {
      if (!rel.r) continue
      const from = solved.find((p) => p.id === rel.fromId)
      const to = solved.find((p) => p.id === rel.toId)
      if (!from || !to) continue
      const aPos = vec(from, 'x', 'y')
      const bPos = vec(to, 'x', 'y')
      if (!aPos || !bPos) continue
      bodies.push({
        label: `r_${rel.fromLabel}/${rel.toLabel}`,
        color: '#93a2b6',
        dashed: true,
        hideMarker: true,
        path: [bPos, aPos],
        currentIndex: 1,
      })
    }
  }
  return bodies
}

export function CurvilinearSolverPanel() {
  const [particles, setParticles] = useState<ParticleState[]>(() => PRESETS[0].particles())
  const [presetKey, setPresetKey] = useState('circular')
  const [showV, setShowV] = useState(true)
  const [showA, setShowA] = useState(true)
  const [showEt, setShowEt] = useState(true)
  const [showEn, setShowEn] = useState(true)
  const [showEr, setShowEr] = useState(false)
  const [showEth, setShowEth] = useState(false)
  const [showRel, setShowRel] = useState(true)

  const output = useMemo(() => solveCurvilinear(particles.map(toSpec)), [particles])
  const pairRelatives = output.relatives.filter((_, i) => {
    // Keep A/B but drop B/A for the unique unordered pair i<j equivalent.
    const rel = output.relatives[i]
    const fromIdx = output.particles.findIndex((p) => p.id === rel.fromId)
    const toIdx = output.particles.findIndex((p) => p.id === rel.toId)
    return fromIdx < toIdx
  })

  const includeOrigin = particles.some((p) => p.frame === 'polar' || p.path === 'circular-origin' || p.knowns.r !== undefined || p.knowns.theta !== undefined)
  const bodies = sceneBodies(particles, output.particles, pairRelatives, {
    v: showV,
    a: showA,
    et: showEt,
    en: showEn,
    er: showEr,
    eth: showEth,
    rel: showRel,
  })

  const addParticle = () => {
    if (particles.length >= MAX_PARTICLES) return
    setParticles((ps) => [...ps, newParticle(ps.length)])
  }

  return (
    <div className="kin-solver">
      <div className="solver-toolbar panel">
        <div className="panel-body solver-toolbar-row">
          <SelectField
            label="Example"
            value={presetKey}
            onChange={(key) => {
              setPresetKey(key)
              const preset = PRESETS.find((p) => p.key === key)
              if (preset) setParticles(preset.particles())
            }}
            options={PRESETS.map((p) => ({ value: p.key, label: p.label }))}
          />
          <button type="button" className="btn btn-primary" onClick={addParticle} disabled={particles.length >= MAX_PARTICLES}>
            Add point
          </button>
          <p className="hint" style={{ margin: 0, flex: 1 }}>
            Instantaneous solver: enter knowns, leave the rest blank. Works in rectangular, n–t, and r–θ, including several points and relative v, a.
          </p>
        </div>
      </div>

      <div className="solver-particle-grid">
        {particles.map((p) => (
          <ParticleCard
            key={p.id}
            particle={p}
            canRemove={particles.length > 1}
            onChange={(next) => setParticles((ps) => ps.map((q) => (q.id === p.id ? next : q)))}
            onRemove={() =>
              setParticles((ps) =>
                ps.filter((q) => q.id !== p.id).map((q, idx) => ({ ...q, label: LABELS[idx] ?? q.label, color: COLORS[idx % COLORS.length] })),
              )
            }
          />
        ))}
      </div>

      <div className="panel">
        <div className="panel-header">
          <h2>Governing identities</h2>
        </div>
        <div className="panel-body">
          <EqList
            items={[
              '\\vec v = v_x\\hat{\\imath} + v_y\\hat{\\jmath} = v\\hat e_t = v_r\\hat e_r + v_\\theta\\hat e_\\theta',
              '\\vec a = a_x\\hat{\\imath} + a_y\\hat{\\jmath} = a_t\\hat e_t + a_n\\hat e_n = a_r\\hat e_r + a_\\theta\\hat e_\\theta',
              'a_t = \\dot v,\\quad a_n = v^2/\\rho,\\quad \\omega = v/\\rho,\\quad \\alpha = a_t/\\rho',
              'v_r = \\dot r,\\quad v_\\theta = r\\dot\\theta,\\quad a_r = \\ddot r - r\\dot\\theta^2,\\quad a_\\theta = r\\ddot\\theta + 2\\dot r\\dot\\theta',
              '\\vec v_A = \\vec v_B + \\vec v_{A/B},\\quad \\vec a_A = \\vec a_B + \\vec a_{A/B}',
            ]}
          />
        </div>
      </div>

      {output.particles.map((sol) => {
        const vecs = vectorEqs(sol)
        const units = unitEqs(sol)
        return (
          <div className="panel" key={sol.id}>
            <div className="panel-header">
              <h2>
                Point {sol.label} — scalars, vectors, unit vectors
              </h2>
            </div>
            <div className="panel-body" style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              {sol.conflicts.length > 0 && (
                <div className="banner banner-error">
                  Over-specified: {sol.conflicts.map((c) => `${QTY_META_BY_KEY[c.key].label} given ${formatNumber(c.existing, 4)} vs ${formatNumber(c.computed, 4)}`).join('; ')}
                </div>
              )}
              {sol.notes.map((n) => (
                <p className="hint" key={n}>
                  {n}
                </p>
              ))}
              <div className="solver-result-grid">
                <ScalarTable sol={sol} group="time" />
                <ScalarTable sol={sol} group="rect" />
                <ScalarTable sol={sol} group="nt" />
                <ScalarTable sol={sol} group="polar" />
                <ScalarTable sol={sol} group="initial" />
              </div>
              {vecs.length > 0 && (
                <div>
                  <div className="solver-group-title">Vectors</div>
                  <EqList items={vecs} />
                </div>
              )}
              {units.length > 0 && (
                <div>
                  <div className="solver-group-title">Unit vectors</div>
                  <EqList items={units} />
                </div>
              )}
              <details className="solver-steps">
                <summary>How each unknown was obtained</summary>
                <ul>
                  {QTY_KEYS_SOLVED(sol).map((key) => (
                    <li key={key}>
                      <Eq tex={QTY_META_BY_KEY[key].tex} />
                      {': '}
                      <Eq tex={sol.slots[key].formula ?? ''} />
                    </li>
                  ))}
                </ul>
              </details>
            </div>
          </div>
        )
      })}

      {particles.length > 1 && (
        <div className="panel">
          <div className="panel-header">
            <h2>Relative position, velocity, acceleration</h2>
          </div>
          <div className="panel-body" style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            <p className="hint">
              <Eq tex="\vec r_{A/B} = \vec r_A - \vec r_B" />, same for velocity and acceleration. Add more points to compare every pair.
            </p>
            {pairRelatives.map((rel) => (
              <RelativeCard key={`${rel.fromId}-${rel.toId}`} rel={rel} />
            ))}
          </div>
        </div>
      )}

      <div className="panel">
        <div className="panel-header">
          <h2>Geometry at this instant</h2>
        </div>
        <div className="panel-body">
          <div className="solver-show-row">
            <ToggleField label="v" checked={showV} onChange={setShowV} />
            <ToggleField label="a" checked={showA} onChange={setShowA} />
            <ToggleField label="êₜ" checked={showEt} onChange={setShowEt} />
            <ToggleField label="êₙ" checked={showEn} onChange={setShowEn} />
            <ToggleField label="êᵣ" checked={showEr} onChange={setShowEr} />
            <ToggleField label="êθ" checked={showEth} onChange={setShowEth} />
            {particles.length > 1 && <ToggleField label="r_A/B" checked={showRel} onChange={setShowRel} />}
          </div>
          {output.particles.some((p) => p.slots.x.value === null || p.slots.y.value === null) && (
            <p className="hint">If x and y are unknown, the sketch places that point at the origin so the path and unit vectors still draw.</p>
          )}
          <Scene2D height={320} aspectEqual includeOrigin={includeOrigin} bodies={bodies} />
        </div>
      </div>
    </div>
  )
}

function QTY_KEYS_SOLVED(sol: ParticleSolution): Qty[] {
  return QTY_META.map((m) => m.key).filter((k) => sol.slots[k].source === 'solved' && sol.slots[k].formula)
}

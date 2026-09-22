// Turn a statement document into the rows and figure for one instant.
// `compileDocument` reads the statements once. `viewAt` solves the kinematics
// at a playback time so the figure and the solved rows stay in step.

import { formatNumber } from '../../format'
import {
  formatVectorTex,
  QTY_META_BY_KEY,
  solveCurvilinear,
  type ParticleSolution,
  type ParticleSpec,
  type Qty,
} from '../curvilinearSolver'
import {
  exampleDocument,
  PROPERTY_INFO,
  type PathKind,
  type PropertyKey,
  type Statement,
  type WorkspaceDocument,
} from './document'
import { compileMath, type CurvePlot, type SurfacePlot } from './math/eval'

export interface RowModel {
  id: string
  statementId: string | null
  label: string
  text: string
  tex: string | null
  source: 'given' | 'solved' | 'error'
  formula: string | null
  /** Set when this row draws a graph. */
  plotKind?: 'curve' | 'surface' | null
  visible?: boolean
  /** Plain text is kept for checks. Hide it when TeX already shows the same line. */
  showText?: boolean
}

export interface BlockModel {
  id: string
  title: string
  color: string
  rows: RowModel[]
  note: string | null
  /** Statement removed by the block's own Remove control. */
  removeId: string
}

export interface FigureBody {
  label: string
  color: string
  path: { x: number; y: number; z: number }[]
  index: number
  dashed?: boolean
  hideMarker?: boolean
  velocity?: { x: number; y: number; z: number }
  role?: 'plot'
}

export interface FigureSurface {
  label: string
  color: string
  grid: { x: number; y: number; z: number }[][]
}

export interface WorkspaceView {
  dimension: 2 | 3
  duration: number
  fitKey: string
  blocks: BlockModel[]
  bodies: FigureBody[]
  surfaces: FigureSurface[]
}

interface Sample {
  x: number
  y: number
  z: number
  vx: number
  vy: number
  vz: number
  speed: number
  psi: number
}

type Motion =
  | { type: 'circle-origin'; radius: number; theta0: number; omega: number; alpha: number; z0: number; vz: number; az: number }
  | {
      type: 'circle'
      cx: number
      cy: number
      relX: number
      relY: number
      radius: number
      v0: number
      at: number
      psi0: number
      z0: number
      vz: number
      az: number
    }
  | { type: 'rect'; x0: number; y0: number; z0: number; vx: number; vy: number; vz: number; ax: number; ay: number; az: number }

interface PointModel {
  statementId: string
  name: string
  color: string
  path: PathKind | null
  given: RowModel[]
  baseKnowns: Partial<Record<Qty, number>>
  givenValues: Partial<Record<PropertyKey, number>>
  hasZ: boolean
  simulate: number | null
  motion: Motion | null
  trace: Sample[] | null
}

interface RelativeModel {
  id: string
  from: string
  to: string
}

export interface CompiledDocument {
  points: PointModel[]
  relatives: RelativeModel[]
  mathRows: RowModel[]
  plots: Array<CurvePlot | SurfacePlot>
  dimension: 2 | 3
  duration: number
  fitKey: string
}

const SCALAR_KEYS = ['speed', 'at', 'an', 'rho', 'omega', 'alpha'] as const
const TRACE_STEPS = 180

function isSolverKey(key: PropertyKey): key is Extract<PropertyKey, Qty> {
  return key !== 'z' && key !== 'vz' && key !== 'az'
}

function nearlyEqual(a: number, b: number): boolean {
  return Math.abs(a - b) <= 1e-6 * Math.max(1, Math.abs(a), Math.abs(b))
}

function propsOf(statements: Statement[], name: string): Partial<Record<PropertyKey, number>> {
  const props: Partial<Record<PropertyKey, number>> = {}
  for (const statement of statements) {
    if (statement.type === 'property' && statement.point === name) props[statement.key] = statement.value
  }
  return props
}

function formatProperty(key: PropertyKey, value: number): string {
  const info = PROPERTY_INFO[key]
  if (value === Infinity) return `∞ ${info.unit}`
  const shown = info.angle ? (value * 180) / Math.PI : value
  return `${formatNumber(shown, info.angle ? 2 : 4)} ${info.unit}`
}

function givenRows(statements: Statement[], name: string): RowModel[] {
  const rows: RowModel[] = []
  for (const statement of statements) {
    if (statement.type === 'property' && statement.point === name) {
      rows.push({
        id: statement.id,
        statementId: statement.id,
        label: PROPERTY_INFO[statement.key].label,
        text: formatProperty(statement.key, statement.value),
        tex: null,
        source: 'given',
        formula: null,
      })
    } else if (statement.type === 'path' && statement.point === name) {
      rows.push({
        id: statement.id,
        statementId: statement.id,
        label: 'Path',
        text: statement.path === 'circle' ? 'circle' : 'circle about the origin',
        tex: null,
        source: 'given',
        formula: null,
      })
    } else if (statement.type === 'simulate' && statement.point === name) {
      rows.push({
        id: statement.id,
        statementId: statement.id,
        label: 'Simulate',
        text: `${formatNumber(statement.duration, 2)} s`,
        tex: null,
        source: 'given',
        formula: null,
      })
    }
  }
  return rows
}

function zState(props: Partial<Record<PropertyKey, number>>): { z0: number; vz: number; az: number; hasZ: boolean } {
  const hasZ = props.z !== undefined || props.vz !== undefined || props.az !== undefined
  return { z0: props.z ?? 0, vz: props.vz ?? 0, az: props.az ?? 0, hasZ }
}

function buildMotion(path: PathKind | null, props: Partial<Record<PropertyKey, number>>): Motion | null {
  const z = zState(props)
  if (path === 'circle-origin') {
    const radius = props.rho ?? (props.x !== undefined || props.y !== undefined ? Math.hypot(props.x ?? 0, props.y ?? 0) : undefined)
    if (radius === undefined || radius <= 0) return null
    const omega = props.omega ?? (props.speed !== undefined ? props.speed / radius : undefined)
    if (omega === undefined) return null
    const alpha = props.alpha ?? (props.at !== undefined ? props.at / radius : 0)
    const theta0 = props.x !== undefined || props.y !== undefined ? Math.atan2(props.y ?? 0, props.x ?? 0) : 0
    return { type: 'circle-origin', radius, theta0, omega, alpha, ...z }
  }
  if (path === 'circle') {
    const radius = props.rho
    const v0 = props.speed ?? (props.omega !== undefined && radius !== undefined ? Math.abs(props.omega * radius) : undefined)
    if (radius === undefined || radius <= 0 || v0 === undefined) return null
    const psi0 = props.psi ?? 0
    const x0 = props.x ?? 0
    const y0 = props.y ?? 0
    const enx = -Math.sin(psi0)
    const eny = Math.cos(psi0)
    return {
      type: 'circle',
      cx: x0 + radius * enx,
      cy: y0 + radius * eny,
      relX: x0 - (x0 + radius * enx),
      relY: y0 - (y0 + radius * eny),
      radius,
      v0,
      at: props.at ?? 0,
      psi0,
      ...z,
    }
  }
  let vx = props.vx
  let vy = props.vy
  if (vx === undefined && vy === undefined && props.speed !== undefined) {
    const psi = props.psi ?? 0
    vx = props.speed * Math.cos(psi)
    vy = props.speed * Math.sin(psi)
  }
  if (vx === undefined && vy === undefined) return null
  return {
    type: 'rect',
    x0: props.x ?? 0,
    y0: props.y ?? 0,
    vx: vx ?? 0,
    vy: vy ?? 0,
    ax: props.ax ?? 0,
    ay: props.ay ?? 0,
    ...z,
  }
}

function rotate(x: number, y: number, phi: number): [number, number] {
  const c = Math.cos(phi)
  const s = Math.sin(phi)
  return [x * c - y * s, x * s + y * c]
}

export function sampleMotion(motion: Motion, time: number): Sample {
  const t = Math.max(0, time)
  const z = motion.z0 + motion.vz * t + 0.5 * motion.az * t * t
  const vz = motion.vz + motion.az * t
  if (motion.type === 'circle-origin') {
    const theta = motion.theta0 + motion.omega * t + 0.5 * motion.alpha * t * t
    const thetaDot = motion.omega + motion.alpha * t
    const x = motion.radius * Math.cos(theta)
    const y = motion.radius * Math.sin(theta)
    const vx = -motion.radius * Math.sin(theta) * thetaDot
    const vy = motion.radius * Math.cos(theta) * thetaDot
    return { x, y, z, vx, vy, vz, speed: Math.hypot(vx, vy), psi: Math.atan2(vy, vx) }
  }
  if (motion.type === 'circle') {
    const arc = motion.v0 * t + 0.5 * motion.at * t * t
    const phi = arc / motion.radius
    const [rx, ry] = rotate(motion.relX, motion.relY, phi)
    const [tx, ty] = rotate(Math.cos(motion.psi0), Math.sin(motion.psi0), phi)
    const speed = motion.v0 + motion.at * t
    const vx = speed * tx
    const vy = speed * ty
    return { x: motion.cx + rx, y: motion.cy + ry, z, vx, vy, vz, speed, psi: Math.atan2(vy, vx) }
  }
  const x = motion.x0 + motion.vx * t + 0.5 * motion.ax * t * t
  const y = motion.y0 + motion.vy * t + 0.5 * motion.ay * t * t
  const vx = motion.vx + motion.ax * t
  const vy = motion.vy + motion.ay * t
  return { x, y, z, vx, vy, vz, speed: Math.hypot(vx, vy), psi: Math.atan2(vy, vx) }
}

function traceFor(motion: Motion, duration: number): Sample[] {
  const samples: Sample[] = []
  for (let i = 0; i <= TRACE_STEPS; i++) samples.push(sampleMotion(motion, (duration * i) / TRACE_STEPS))
  return samples
}

function circleGuide(motion: Motion): { x: number; y: number; z: number }[] | null {
  if (motion.type === 'rect') return null
  const cx = motion.type === 'circle' ? motion.cx : 0
  const cy = motion.type === 'circle' ? motion.cy : 0
  const z = motion.z0
  const pts: { x: number; y: number; z: number }[] = []
  for (let i = 0; i <= 96; i++) {
    const angle = (2 * Math.PI * i) / 96
    pts.push({ x: cx + motion.radius * Math.cos(angle), y: cy + motion.radius * Math.sin(angle), z })
  }
  return pts
}

export function compileDocument(doc: WorkspaceDocument): CompiledDocument {
  const points: PointModel[] = []
  for (const statement of doc.statements) {
    if (statement.type !== 'point') continue
    const props = propsOf(doc.statements, statement.name)
    const path = doc.statements.find((s): s is Extract<Statement, { type: 'path' }> => s.type === 'path' && s.point === statement.name)?.path ?? null
    const simulate = doc.statements.find((s): s is Extract<Statement, { type: 'simulate' }> => s.type === 'simulate' && s.point === statement.name)?.duration ?? null
    const motion = buildMotion(path, props)
    const knowns: Partial<Record<Qty, number>> = {}
    for (const key of Object.keys(props) as PropertyKey[]) {
      const value = props[key]
      if (value !== undefined && isSolverKey(key)) knowns[key] = value
    }
    points.push({
      statementId: statement.id,
      name: statement.name,
      color: statement.color,
      path,
      given: givenRows(doc.statements, statement.name),
      baseKnowns: knowns,
      givenValues: props,
      hasZ: zState(props).hasZ,
      simulate,
      motion,
      trace: motion && simulate ? traceFor(motion, simulate) : null,
    })
  }
  const relatives: RelativeModel[] = doc.statements
    .filter((s): s is Extract<Statement, { type: 'relative' }> => s.type === 'relative')
    .map((s) => ({ id: s.id, from: s.from, to: s.to }))
  const math = compileMath(doc.statements)
  const mathRows: RowModel[] = math.rows.map((row) => ({
    id: row.statementId,
    statementId: row.statementId,
    label: row.label,
    text: row.text,
    tex: row.tex,
    source: row.tex ? (row.plotKind ? 'given' : 'solved') : 'error',
    formula: null,
    plotKind: row.plotKind,
    visible: row.visible,
    showText: row.plotKind ? Boolean(row.warn) : !row.tex,
  }))
  const duration = points.reduce((max, point) => Math.max(max, point.simulate ?? 0), 0)
  const surfaceVisible = math.plots.some((plot) => plot.kind === 'surface' && plot.visible && plot.grid.some((row) => row.some((point) => Number.isFinite(point.z))))
  const dimension: 2 | 3 = points.some((point) => point.hasZ) || surfaceVisible ? 3 : 2
  const fitKey = `${dimension}:${duration}:${points.map((point) => point.name).join(',')}:${relatives.map((rel) => rel.id).join(',')}:${math.plots.map((plot) => `${plot.statementId}${plot.visible ? '1' : '0'}`).join(',')}`
  return { points, relatives, mathRows, plots: math.plots, dimension, duration, fitKey }
}

function knownsAt(point: PointModel, time: number): Partial<Record<Qty, number>> {
  if (!point.simulate || !point.motion) return point.baseKnowns
  const sample = sampleMotion(point.motion, Math.min(time, point.simulate))
  return { ...point.baseKnowns, x: sample.x, y: sample.y, vx: sample.vx, vy: sample.vy, speed: sample.speed, psi: sample.psi }
}

function scalarChanged(point: PointModel, key: PropertyKey, value: number): boolean {
  const given = point.givenValues[key]
  return given === undefined || !nearlyEqual(given, value)
}

function vecTex(x: number, y: number, z: number, hasZ: boolean): string {
  if (!hasZ || Math.abs(z) < 1e-9) return formatVectorTex(x, y)
  const planar = formatVectorTex(x, y)
  const zText = `${formatNumber(Math.abs(z), 3)}\\,\\hat{k}`
  if (planar === '0') return `${z < 0 ? '-' : ''}${zText}`
  return `${planar} ${z < 0 ? '-' : '+'} ${zText}`
}

function solvedRows(point: PointModel, solSlots: ParticleSolution['slots'], sample: Sample | null): RowModel[] {
  const rows: RowModel[] = []
  for (const key of SCALAR_KEYS) {
    const slot = solSlots[key]
    if (slot.value === null) continue
    if (!Number.isFinite(slot.value) && slot.value !== Infinity) continue
    if (!scalarChanged(point, key, slot.value)) continue
    rows.push({
      id: `solved:${point.name}:${key}`,
      statementId: null,
      label: PROPERTY_INFO[key].label,
      text: formatProperty(key, slot.value),
      tex: null,
      source: 'solved',
      formula: slot.formula,
    })
  }

  const x = sample?.x ?? solSlots.x.value
  const y = sample?.y ?? solSlots.y.value
  const z = sample?.z ?? point.givenValues.z ?? 0
  if (x !== null && y !== null && (scalarChanged(point, 'x', x) || scalarChanged(point, 'y', y) || (point.hasZ && scalarChanged(point, 'z', z)))) {
    const parts = [`(${formatNumber(x, 3)}, ${formatNumber(y, 3)}${point.hasZ ? `, ${formatNumber(z, 3)}` : ''}) m`]
    rows.push({
      id: `solved:${point.name}:position`,
      statementId: null,
      label: 'Position',
      text: parts[0],
      tex: null,
      source: 'solved',
      formula: solSlots.x.formula,
    })
  }

  const vx = sample?.vx ?? solSlots.vx.value
  const vy = sample?.vy ?? solSlots.vy.value
  const vz = sample?.vz ?? point.givenValues.vz ?? 0
  if (vx !== null && vy !== null && (scalarChanged(point, 'vx', vx) || scalarChanged(point, 'vy', vy) || (point.hasZ && scalarChanged(point, 'vz', vz)))) {
    rows.push({
      id: `solved:${point.name}:velocity`,
      statementId: null,
      label: 'Velocity',
      text: '',
      tex: vecTex(vx, vy, vz, point.hasZ),
      source: 'solved',
      formula: solSlots.vx.formula,
    })
  }

  const ax = solSlots.ax.value
  const ay = solSlots.ay.value
  const az = point.givenValues.az ?? 0
  if (ax !== null && ay !== null && (scalarChanged(point, 'ax', ax) || scalarChanged(point, 'ay', ay) || (point.hasZ && scalarChanged(point, 'az', az)))) {
    rows.push({
      id: `solved:${point.name}:acceleration`,
      statementId: null,
      label: 'Acceleration',
      text: '',
      tex: vecTex(ax, ay, az, point.hasZ),
      source: 'solved',
      formula: solSlots.ax.formula,
    })
  }
  return rows
}

function traceIndex(point: PointModel, time: number): number {
  if (!point.trace || !point.simulate) return 0
  const clamped = Math.min(Math.max(time, 0), point.simulate)
  return Math.round((clamped / point.simulate) * (point.trace.length - 1))
}

export function viewAt(compiled: CompiledDocument, time: number): WorkspaceView {
  const specs: ParticleSpec[] = compiled.points.map((point) => ({
    id: point.name,
    label: point.name,
    color: point.color,
    path: point.path === 'circle' ? 'circular' : point.path === 'circle-origin' ? 'circular-origin' : 'general',
    curveCCW: true,
    knowns: knownsAt(point, time),
  }))
  const solved = solveCurvilinear(specs)
  const byName = new Map(solved.particles.map((particle) => [particle.label, particle]))
  const placed = new Map<string, Sample>()

  const blocks: BlockModel[] = []
  const bodies: FigureBody[] = []

  for (const point of compiled.points) {
    const particle = byName.get(point.name)
    const sample = point.motion ? sampleMotion(point.motion, point.simulate ? Math.min(time, point.simulate) : 0) : null
    if (sample) placed.set(point.name, sample)
    else if (particle && particle.slots.x.value !== null && particle.slots.y.value !== null) {
      placed.set(point.name, {
        x: particle.slots.x.value,
        y: particle.slots.y.value,
        z: point.givenValues.z ?? 0,
        vx: particle.slots.vx.value ?? 0,
        vy: particle.slots.vy.value ?? 0,
        vz: point.givenValues.vz ?? 0,
        speed: particle.slots.speed.value ?? 0,
        psi: particle.slots.psi.value ?? 0,
      })
    }

    const rows = [...point.given]
    if (particle) rows.push(...solvedRows(point, particle.slots, sample))
    if (point.simulate && !point.motion) {
      rows.push({
        id: `solved:${point.name}:stuck`,
        statementId: null,
        label: 'Simulate',
        text: 'needs a speed or a velocity',
        tex: null,
        source: 'solved',
        formula: null,
      })
    }
    const note = particle && particle.conflicts.length > 0
      ? particle.conflicts.map((conflict) => `${QTY_META_BY_KEY[conflict.key].label} disagrees (${formatNumber(conflict.existing, 3)} and ${formatNumber(conflict.computed, 3)})`).join('; ')
      : null
    blocks.push({ id: point.name, title: `Point ${point.name}`, color: point.color, rows, note, removeId: point.statementId })

    if (point.motion) {
      const guide = circleGuide(point.motion)
      if (guide) {
        bodies.push({ label: `${point.name} path`, color: point.color, path: guide, index: 0, dashed: true, hideMarker: true })
      }
    }
    if (point.trace) {
      const index = traceIndex(point, time)
      const here = point.trace[index]
      bodies.push({
        label: point.name,
        color: point.color,
        path: point.trace.map((p) => ({ x: p.x, y: p.y, z: p.z })),
        index,
        velocity: here ? { x: here.vx, y: here.vy, z: here.vz } : undefined,
      })
    } else {
      const here = placed.get(point.name)
      if (here) {
        bodies.push({
          label: point.name,
          color: point.color,
          path: [{ x: here.x, y: here.y, z: here.z }],
          index: 0,
          velocity: { x: here.vx, y: here.vy, z: here.vz },
        })
      }
    }
  }

  for (const relative of compiled.relatives) {
    const from = compiled.points.find((point) => point.name === relative.from)
    const to = compiled.points.find((point) => point.name === relative.to)
    const color = from?.color ?? '#93a2b6'
    const rows: RowModel[] = [
      { id: relative.id, statementId: relative.id, label: 'Pair', text: `${relative.from} to ${relative.to}`, tex: null, source: 'given', formula: null },
    ]
    const a = placed.get(relative.from)
    const b = placed.get(relative.to)
    let note: string | null = null
    if (!from || !to) note = 'Those points are not on the page.'
    else if (!a || !b) note = 'Both points need a position.'
    else {
      const rx = a.x - b.x
      const ry = a.y - b.y
      const rz = a.z - b.z
      const rvx = a.vx - b.vx
      const rvy = a.vy - b.vy
      const rvz = a.vz - b.vz
      const hasZ = Boolean(from.hasZ || to.hasZ)
      const ax = (byName.get(from.name)?.slots.ax.value ?? from.givenValues.ax ?? 0) - (byName.get(to.name)?.slots.ax.value ?? to.givenValues.ax ?? 0)
      const ay = (byName.get(from.name)?.slots.ay.value ?? from.givenValues.ay ?? 0) - (byName.get(to.name)?.slots.ay.value ?? to.givenValues.ay ?? 0)
      const az = (from.givenValues.az ?? 0) - (to.givenValues.az ?? 0)
      const tag = `${relative.from}/${relative.to}`
      rows.push(
        { id: `${relative.id}:r`, statementId: null, label: 'Relative position', text: `${formatNumber(Math.hypot(rx, ry, rz), 3)} m`, tex: vecTex(rx, ry, rz, hasZ), source: 'solved', formula: `\\vec r_{${tag}} = \\vec r_{${relative.from}} - \\vec r_{${relative.to}}` },
        { id: `${relative.id}:v`, statementId: null, label: 'Relative velocity', text: `${formatNumber(Math.hypot(rvx, rvy, rvz), 3)} m/s`, tex: vecTex(rvx, rvy, rvz, hasZ), source: 'solved', formula: `\\vec v_{${tag}} = \\vec v_{${relative.from}} - \\vec v_{${relative.to}}` },
        { id: `${relative.id}:a`, statementId: null, label: 'Relative acceleration', text: `${formatNumber(Math.hypot(ax, ay, az), 3)} m/s²`, tex: vecTex(ax, ay, az, hasZ), source: 'solved', formula: `\\vec a_{${tag}} = \\vec a_{${relative.from}} - \\vec a_{${relative.to}}` },
      )
      bodies.push({
        label: `${relative.from} to ${relative.to}`,
        color: '#93a2b6',
        path: [
          { x: b.x, y: b.y, z: b.z },
          { x: a.x, y: a.y, z: a.z },
        ],
        index: 1,
        dashed: true,
        hideMarker: true,
      })
    }
    blocks.push({ id: relative.id, title: `${relative.from} relative to ${relative.to}`, color, rows, note, removeId: relative.id })
  }

  if (compiled.mathRows.length > 0) {
    blocks.push({ id: 'math', title: 'Math', color: '#59d67f', rows: compiled.mathRows, note: null, removeId: '' })
  }

  for (const plot of compiled.plots) {
    if (!plot.visible || plot.kind !== 'curve') continue
    if (!plot.path.some((point) => Number.isFinite(point.x) && Number.isFinite(point.y))) continue
    bodies.push({ label: plot.label, color: plot.color, path: plot.path, index: 0, hideMarker: true, role: 'plot' })
  }

  const surfaces: FigureSurface[] = []
  for (const plot of compiled.plots) {
    if (!plot.visible || plot.kind !== 'surface') continue
    if (!plot.grid.some((row) => row.some((point) => Number.isFinite(point.z)))) continue
    surfaces.push({ label: plot.label, color: plot.color, grid: plot.grid })
  }

  return { dimension: compiled.dimension, duration: compiled.duration, fitKey: compiled.fitKey, blocks, bodies, surfaces }
}

export function evaluateDocument(doc: WorkspaceDocument, time = 0): WorkspaceView {
  return viewAt(compileDocument(doc), time)
}

export function runWorkspaceChecks(): string[] {
  const errors: string[] = []
  const circular = compileDocument(exampleDocument('circular'))
  const at0 = viewAt(circular, 0)
  const an0 = at0.blocks[0]?.rows.find((row) => row.label === 'Normal acceleration')
  if (!an0 || !an0.text.startsWith('3.125')) errors.push(`circular an at 0: ${an0?.text}`)
  const an1 = viewAt(circular, 1).blocks[0]?.rows.find((row) => row.label === 'Normal acceleration')
  if (!an1 || !an1.text.startsWith('3.92')) errors.push(`circular an at 1: ${an1?.text}`)
  if (at0.dimension !== 2) errors.push('circular dimension')

  const pair = evaluateDocument(exampleDocument('two-points'), 0)
  const rel = pair.blocks.find((block) => block.title.includes('relative'))
  const mag = rel?.rows.find((row) => row.label === 'Relative position')
  if (!mag || !mag.text.startsWith('5')) errors.push(`relative |r|: ${mag?.text}`)

  const projectile = compileDocument(exampleDocument('projectile'))
  const motion = projectile.points[0]?.motion
  if (!motion) errors.push('projectile has no motion')
  else {
    const y1 = sampleMotion(motion, 1).y
    if (Math.abs(y1 - (1 + 15 - 0.5 * 9.81)) > 1e-6) errors.push(`projectile y(1): ${y1}`)
  }
  if (projectile.dimension !== 2) errors.push('projectile dimension')

  const helix = compileDocument(exampleDocument('helix'))
  if (helix.dimension !== 3) errors.push('helix dimension')
  const helixMotion = helix.points[0]?.motion
  if (!helixMotion) errors.push('helix has no motion')
  else {
    const mid = sampleMotion(helixMotion, Math.PI / 2)
    if (Math.abs(mid.x) > 1e-6 || Math.abs(mid.y - 2) > 1e-6) errors.push(`helix xy: ${mid.x}, ${mid.y}`)
    if (Math.abs(mid.z - 0.4 * (Math.PI / 2)) > 1e-6) errors.push(`helix z: ${mid.z}`)
  }

  return errors
}

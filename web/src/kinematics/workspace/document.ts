// Workspace document: the list of statements on the left.
// `commitCommand` and `removeStatement` are the operations a later agent
// can call. Units are SI; angles on the document are radians.

import { commandById } from './commands'
import { convertAngleInput, type AngleMode } from './math/expr'

export const PROPERTY_KEYS = [
  'x',
  'y',
  'z',
  'vx',
  'vy',
  'vz',
  'ax',
  'ay',
  'az',
  'speed',
  'psi',
  'at',
  'an',
  'rho',
  'omega',
  'alpha',
] as const

export type PropertyKey = (typeof PROPERTY_KEYS)[number]

export interface PropertyInfo {
  label: string
  unit: string
  /** Stored in radians. Slots and the statement list show degrees. */
  angle?: boolean
}

export const PROPERTY_INFO: Record<PropertyKey, PropertyInfo> = {
  x: { label: 'Position x', unit: 'm' },
  y: { label: 'Position y', unit: 'm' },
  z: { label: 'Position z', unit: 'm' },
  vx: { label: 'Velocity x', unit: 'm/s' },
  vy: { label: 'Velocity y', unit: 'm/s' },
  vz: { label: 'Velocity z', unit: 'm/s' },
  ax: { label: 'Acceleration x', unit: 'm/s²' },
  ay: { label: 'Acceleration y', unit: 'm/s²' },
  az: { label: 'Acceleration z', unit: 'm/s²' },
  speed: { label: 'Speed', unit: 'm/s' },
  psi: { label: 'Direction', unit: 'deg', angle: true },
  at: { label: 'Tangential acceleration', unit: 'm/s²' },
  an: { label: 'Normal acceleration', unit: 'm/s²' },
  rho: { label: 'Radius', unit: 'm' },
  omega: { label: 'Angular velocity', unit: 'rad/s' },
  alpha: { label: 'Angular acceleration', unit: 'rad/s²' },
}

const POINT_COLORS = ['#29d3f5', '#f5a524', '#a78bfa', '#fb6a6a', '#59d67f', '#5aa8ff']

export type PathKind = 'circle' | 'circle-origin'

export type Statement =
  | { id: string; type: 'point'; name: string; color: string }
  | { id: string; type: 'property'; point: string; key: PropertyKey; value: number }
  | { id: string; type: 'path'; point: string; path: PathKind }
  | { id: string; type: 'relative'; from: string; to: string }
  | { id: string; type: 'simulate'; point: string; duration: number }
  | { id: string; type: 'math'; input: string; visible: boolean }

export interface WorkspaceDocument {
  nextId: number
  statements: Statement[]
}

export type ExampleId = 'circular' | 'two-points' | 'projectile' | 'helix'

export function emptyDocument(): WorkspaceDocument {
  return { nextId: 1, statements: [] }
}

export function pointNames(doc: WorkspaceDocument): string[] {
  return doc.statements.filter((s): s is Extract<Statement, { type: 'point' }> => s.type === 'point').map((s) => s.name)
}

export function nextPointName(doc: WorkspaceDocument): string {
  const used = new Set(pointNames(doc))
  const letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
  for (const letter of letters) {
    if (!used.has(letter)) return letter
  }
  let n = 27
  while (used.has(`P${n}`)) n += 1
  return `P${n}`
}

function allocate(doc: WorkspaceDocument): { doc: WorkspaceDocument; id: string } {
  return { doc: { ...doc, nextId: doc.nextId + 1 }, id: `s${doc.nextId}` }
}

function normalizeName(raw: string): string {
  const name = raw.trim()
  if (/^[a-zA-Z]$/.test(name)) return name.toUpperCase()
  return name
}

function parseRequired(raw: string | undefined): number | null {
  if (raw === undefined) return null
  const text = raw.trim()
  if (text === '') return null
  const value = Number(text)
  return Number.isFinite(value) ? value : null
}

function parseOptional(raw: string | undefined): number | undefined | null {
  if (raw === undefined) return undefined
  const text = raw.trim()
  if (text === '') return undefined
  const value = Number(text)
  return Number.isFinite(value) ? value : null
}

function addPoint(doc: WorkspaceDocument, rawName: string): { doc: WorkspaceDocument; error: string | null } {
  const name = normalizeName(rawName)
  if (!name) return { doc, error: 'Name the point.' }
  if (pointNames(doc).includes(name)) return { doc, error: `Point ${name} already exists.` }
  const slot = allocate(doc)
  const color = POINT_COLORS[(pointNames(doc).length) % POINT_COLORS.length]
  const statement: Statement = { id: slot.id, type: 'point', name, color }
  return { doc: { ...slot.doc, statements: [...slot.doc.statements, statement] }, error: null }
}

function upsertProperty(doc: WorkspaceDocument, point: string, key: PropertyKey, value: number): WorkspaceDocument {
  const existing = doc.statements.find((s) => s.type === 'property' && s.point === point && s.key === key)
  if (existing) {
    return {
      ...doc,
      statements: doc.statements.map((s) => (s.id === existing.id ? { ...existing, value } : s)),
    }
  }
  const slot = allocate(doc)
  const statement: Statement = { id: slot.id, type: 'property', point, key, value }
  return { ...slot.doc, statements: [...slot.doc.statements, statement] }
}

function upsertPath(doc: WorkspaceDocument, point: string, path: PathKind): WorkspaceDocument {
  const existing = doc.statements.find((s) => s.type === 'path' && s.point === point)
  if (existing && existing.type === 'path') {
    return { ...doc, statements: doc.statements.map((s) => (s.id === existing.id ? { ...existing, path } : s)) }
  }
  const slot = allocate(doc)
  const statement: Statement = { id: slot.id, type: 'path', point, path }
  return { ...slot.doc, statements: [...slot.doc.statements, statement] }
}

function upsertRelative(doc: WorkspaceDocument, from: string, to: string): WorkspaceDocument {
  const existing = doc.statements.find((s) => s.type === 'relative' && s.from === from && s.to === to)
  if (existing) return doc
  const slot = allocate(doc)
  const statement: Statement = { id: slot.id, type: 'relative', from, to }
  return { ...slot.doc, statements: [...slot.doc.statements, statement] }
}

function upsertSimulate(doc: WorkspaceDocument, point: string, duration: number): WorkspaceDocument {
  const existing = doc.statements.find((s) => s.type === 'simulate' && s.point === point)
  if (existing && existing.type === 'simulate') {
    return { ...doc, statements: doc.statements.map((s) => (s.id === existing.id ? { ...existing, duration } : s)) }
  }
  const slot = allocate(doc)
  const statement: Statement = { id: slot.id, type: 'simulate', point, duration }
  return { ...slot.doc, statements: [...slot.doc.statements, statement] }
}

function requirePoint(doc: WorkspaceDocument, raw: string | undefined): { name: string; error: string | null } {
  const name = normalizeName(raw ?? '')
  if (!name) return { name: '', error: 'Choose a point.' }
  if (!pointNames(doc).includes(name)) return { name, error: `Add point ${name} first.` }
  return { name, error: null }
}

function setScalar(doc: WorkspaceDocument, point: string, key: PropertyKey, raw: string | undefined, angle = false): { doc: WorkspaceDocument; error: string | null } {
  const parsed = parseRequired(raw)
  if (parsed === null) return { doc, error: `Enter ${PROPERTY_INFO[key].label.toLowerCase()}.` }
  const value = angle ? (parsed * Math.PI) / 180 : parsed
  return { doc: upsertProperty(doc, point, key, value), error: null }
}

/** Apply one catalog command. Unknown command ids and bad slots return the same document plus an error. */
export function commitCommand(doc: WorkspaceDocument, commandId: string, args: Record<string, string>): { doc: WorkspaceDocument; error: string | null } {
  const command = commandById(commandId)
  if (!command) return { doc, error: 'Unknown statement.' }

  if (commandId === 'point') return addPoint(doc, args.name ?? '')

  if (commandId === 'relative') {
    const from = requirePoint(doc, args.from)
    if (from.error) return { doc, error: from.error }
    const to = requirePoint(doc, args.to)
    if (to.error) return { doc, error: to.error }
    if (from.name === to.name) return { doc, error: 'Pick two different points.' }
    return { doc: upsertRelative(doc, from.name, to.name), error: null }
  }

  const point = requirePoint(doc, args.point)
  if (point.error) return { doc, error: point.error }
  const name = point.name

  if (commandId === 'circle') return { doc: upsertPath(doc, name, 'circle'), error: null }
  if (commandId === 'circle-origin') return { doc: upsertPath(doc, name, 'circle-origin'), error: null }

  if (commandId === 'simulate') {
    const duration = parseRequired(args.duration)
    if (duration === null || duration <= 0) return { doc, error: 'Enter a duration greater than zero.' }
    return { doc: upsertSimulate(doc, name, duration), error: null }
  }

  if (commandId === 'position' || commandId === 'velocity' || commandId === 'acceleration') {
    const triples =
      commandId === 'position'
        ? ([['x', args.x, true], ['y', args.y, true], ['z', args.z, false]] as const)
        : commandId === 'velocity'
          ? ([['vx', args.vx, true], ['vy', args.vy, true], ['vz', args.vz, false]] as const)
          : ([['ax', args.ax, true], ['ay', args.ay, true], ['az', args.az, false]] as const)
    let next = doc
    for (const [key, raw, required] of triples) {
      if (required) {
        const value = parseRequired(raw)
        if (value === null) return { doc, error: `Enter ${PROPERTY_INFO[key].label.toLowerCase()}.` }
        next = upsertProperty(next, name, key, value)
      } else {
        const value = parseOptional(raw)
        if (value === null) return { doc, error: `Enter ${PROPERTY_INFO[key].label.toLowerCase()}.` }
        if (value !== undefined) next = upsertProperty(next, name, key, value)
      }
    }
    return { doc: next, error: null }
  }

  const scalarKey: PropertyKey | null =
    commandId === 'speed'
      ? 'speed'
      : commandId === 'direction'
        ? 'psi'
        : commandId === 'tangential-acceleration'
          ? 'at'
          : commandId === 'normal-acceleration'
            ? 'an'
            : commandId === 'radius'
              ? 'rho'
              : commandId === 'angular-velocity'
                ? 'omega'
                : commandId === 'angular-acceleration'
                  ? 'alpha'
                  : null
  if (!scalarKey) return { doc, error: 'Unknown statement.' }
  return setScalar(doc, name, scalarKey, args.value, scalarKey === 'psi')
}

/** Rewrite stored math so a radians/degrees switch changes angle inputs and keeps the values. */
export function convertDocumentAngles(doc: WorkspaceDocument, from: AngleMode, to: AngleMode): WorkspaceDocument {
  if (from === to) return doc
  let changed = false
  const statements = doc.statements.map((statement) => {
    if (statement.type !== 'math') return statement
    const input = convertAngleInput(statement.input, from, to)
    if (input === statement.input) return statement
    changed = true
    return { ...statement, input }
  })
  return changed ? { ...doc, statements } : doc
}

export function replaceMath(doc: WorkspaceDocument, id: string, input: string): WorkspaceDocument {
  const text = input.trim()
  if (!text) return removeStatement(doc, id)
  return {
    ...doc,
    statements: doc.statements.map((statement) => (statement.id === id && statement.type === 'math' ? { ...statement, input: text } : statement)),
  }
}

export function appendMath(doc: WorkspaceDocument, input: string): WorkspaceDocument {
  const text = input.trim()
  const slot = allocate(doc)
  const statement: Statement = { id: slot.id, type: 'math', input: text, visible: true }
  return { ...slot.doc, statements: [...slot.doc.statements, statement] }
}

export function setMathVisible(doc: WorkspaceDocument, id: string, visible: boolean): WorkspaceDocument {
  return {
    ...doc,
    statements: doc.statements.map((statement) => (statement.id === id && statement.type === 'math' ? { ...statement, visible } : statement)),
  }
}

export function removeStatement(doc: WorkspaceDocument, id: string): WorkspaceDocument {
  const target = doc.statements.find((s) => s.id === id)
  if (!target) return doc
  if (target.type === 'point') {
    const name = target.name
    return {
      ...doc,
      statements: doc.statements.filter((s) => {
        if (s.id === id) return false
        if (s.type === 'property' || s.type === 'path' || s.type === 'simulate') return s.point !== name
        if (s.type === 'relative') return s.from !== name && s.to !== name
        return true
      }),
    }
  }
  return { ...doc, statements: doc.statements.filter((s) => s.id !== id) }
}

function applyAll(steps: [string, Record<string, string>][]): WorkspaceDocument {
  let doc = emptyDocument()
  for (const [id, args] of steps) {
    const result = commitCommand(doc, id, args)
    if (result.error) throw new Error(`${id}: ${result.error}`)
    doc = result.doc
  }
  return doc
}

export function exampleDocument(example: ExampleId): WorkspaceDocument {
  if (example === 'circular') {
    return applyAll([
      ['point', { name: 'A' }],
      ['circle', { point: 'A' }],
      ['radius', { point: 'A', value: '200' }],
      ['speed', { point: 'A', value: '25' }],
      ['tangential-acceleration', { point: 'A', value: '3' }],
      ['simulate', { point: 'A', duration: '4' }],
    ])
  }
  if (example === 'two-points') {
    return applyAll([
      ['point', { name: 'A' }],
      ['position', { point: 'A', x: '0', y: '0' }],
      ['velocity', { point: 'A', vx: '8', vy: '0' }],
      ['acceleration', { point: 'A', ax: '0', ay: '2' }],
      ['point', { name: 'B' }],
      ['position', { point: 'B', x: '4', y: '3' }],
      ['velocity', { point: 'B', vx: '2', vy: '4' }],
      ['acceleration', { point: 'B', ax: '-1', ay: '0' }],
      ['relative', { from: 'A', to: 'B' }],
      ['simulate', { point: 'A', duration: '4' }],
      ['simulate', { point: 'B', duration: '4' }],
    ])
  }
  if (example === 'projectile') {
    return applyAll([
      ['point', { name: 'A' }],
      ['position', { point: 'A', x: '0', y: '1' }],
      ['velocity', { point: 'A', vx: '20', vy: '15' }],
      ['acceleration', { point: 'A', ax: '0', ay: '-9.81' }],
      ['simulate', { point: 'A', duration: '3' }],
    ])
  }
  return applyAll([
    ['point', { name: 'A' }],
    ['circle-origin', { point: 'A' }],
    ['radius', { point: 'A', value: '2' }],
    ['angular-velocity', { point: 'A', value: '1' }],
    ['position', { point: 'A', x: '2', y: '0', z: '0' }],
    ['velocity', { point: 'A', vx: '0', vy: '2', vz: '0.4' }],
    ['simulate', { point: 'A', duration: '8' }],
  ])
}

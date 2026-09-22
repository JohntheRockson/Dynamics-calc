// Evaluate math statements in document order. A one-input function is a curve.
// A two-input function is a surface. Everything else stays in the console.

import type { Statement } from '../document'
import {
  MathError,
  applyEnv,
  freeSymbols,
  normalize,
  numericValue,
  parseMathInput,
  plain,
  present,
  solveEquation,
  tex,
  type Expr,
  type MathEnv,
} from './expr'

const PLOT_COLORS = ['#59d67f', '#f5a524', '#a78bfa', '#fb6a6a', '#5aa8ff', '#e879f9']
const CURVE_SAMPLES = 201
const SURFACE_SAMPLES = 21
const WINDOW = 10
const CLIP = 1e4

export interface MathConsoleRow {
  statementId: string
  label: string
  text: string
  tex: string | null
  plotKind: 'curve' | 'surface' | null
  visible: boolean
  warn: string | null
}

export interface CurvePlot {
  kind: 'curve'
  statementId: string
  label: string
  color: string
  visible: boolean
  path: { x: number; y: number; z: number }[]
}

export interface SurfacePlot {
  kind: 'surface'
  statementId: string
  label: string
  color: string
  visible: boolean
  grid: { x: number; y: number; z: number }[][]
}

export interface MathCompilation {
  rows: MathConsoleRow[]
  plots: Array<CurvePlot | SurfacePlot>
}

export function compileMath(statements: Statement[]): MathCompilation {
  const env: MathEnv = new Map()
  const rows: MathConsoleRow[] = []
  const plots: Array<CurvePlot | SurfacePlot> = []
  let colorIndex = 0

  for (const statement of statements) {
    if (statement.type !== 'math') continue
    try {
      const parsed = parseMathInput(statement.input)
      if (parsed.kind === 'fn') {
        env.set(parsed.name, { kind: 'fn', params: parsed.params, body: parsed.body })
        const label = `${parsed.name}(${parsed.params.join(', ')})`
        const formula = { tex: `${labelTex(parsed.name, parsed.params)} = ${tex(parsed.body)}`, text: `${label} = ${plain(parsed.body)}` }
        const missing = freeSymbols(parsed.body).filter((name) => !parsed.params.includes(name) && !env.has(name))
        const color = PLOT_COLORS[colorIndex % PLOT_COLORS.length]
        colorIndex += 1
        const plotKind = parsed.params.length === 2 ? 'surface' : 'curve'
        let warn: string | null = null
        let plot: CurvePlot | SurfacePlot
        if (missing.length > 0) {
          warn = `Give ${missing.join(', ')} a value above this line to draw the graph.`
          plot = plotKind === 'surface'
            ? { kind: 'surface', statementId: statement.id, label, color, visible: statement.visible, grid: [] }
            : { kind: 'curve', statementId: statement.id, label, color, visible: statement.visible, path: [] }
        } else if (parsed.params.length === 2) {
          const grid = sampleSurface(parsed.body, parsed.params[0], parsed.params[1], env)
          if (!grid.some((row) => row.some((point) => Number.isFinite(point.z)))) warn = 'No real values to plot on [-10, 10].'
          plot = { kind: 'surface', statementId: statement.id, label, color, visible: statement.visible, grid }
        } else {
          const path = sampleCurve(parsed.body, parsed.params[0], env)
          if (!path.some((point) => Number.isFinite(point.y))) warn = 'No real values to plot on [-10, 10].'
          plot = { kind: 'curve', statementId: statement.id, label, color, visible: statement.visible, path }
        }
        plots.push(plot)
        rows.push({
          statementId: statement.id,
          label,
          text: warn ?? formula.text,
          tex: formula.tex,
          plotKind,
          visible: statement.visible,
          warn,
        })
        continue
      }
      if (parsed.kind === 'assign') {
        const value = normalize(applyEnv(parsed.expr, env, 0))
        env.set(parsed.name, { kind: 'expr', expr: value })
        const shown = present(value)
        rows.push({
          statementId: statement.id,
          label: parsed.name,
          text: `${parsed.name} = ${shown.text}`,
          tex: `${tex({ type: 'sym', name: parsed.name })} = ${shown.tex}`,
          plotKind: null,
          visible: true,
          warn: null,
        })
        continue
      }
      if (parsed.kind === 'solve') {
        const equation = applyEnv(parsed.equation, env, 0)
        const solved = solveEquation(equation, parsed.variable)
        rows.push({ statementId: statement.id, label: 'Solve', text: solved.text, tex: solved.tex, plotKind: null, visible: true, warn: null })
        continue
      }
      const value = normalize(applyEnv(parsed.expr, env, 0))
      if (value.type === 'eq') {
        const left = present(normalize(value.left))
        const right = present(normalize(value.right))
        rows.push({
          statementId: statement.id,
          label: 'Result',
          text: `${left.text} = ${right.text}`,
          tex: `${left.tex} = ${right.tex}`,
          plotKind: null,
          visible: true,
          warn: null,
        })
        continue
      }
      const line = echo(parsed.expr, present(value))
      rows.push({ statementId: statement.id, label: 'Result', text: line.text, tex: line.tex, plotKind: null, visible: true, warn: null })
    } catch (error) {
      const message = error instanceof MathError ? error.message : 'Could not read that.'
      rows.push({ statementId: statement.id, label: 'Math', text: message, tex: null, plotKind: null, visible: statement.visible, warn: message })
    }
  }

  return { rows, plots }
}

function echo(input: Expr, shown: { tex: string; text: string }): { tex: string; text: string } {
  const leftText = plain(input)
  if (leftText === shown.text) return shown
  return { text: `${leftText} = ${shown.text}`, tex: `${tex(input)} = ${shown.tex}` }
}

function labelTex(name: string, params: string[]): string {
  return `${name}\\left(${params.join(', ')}\\right)`
}

function sampleNumber(index: number, count: number): number {
  if (count === 1) return 0
  return -WINDOW + (2 * WINDOW * index) / (count - 1)
}

function bind(env: MathEnv, name: string, value: number): MathEnv {
  const next: MathEnv = new Map(env)
  next.set(name, { kind: 'expr', expr: { type: 'dec', text: String(value), value } })
  return next
}

function sampleCurve(body: Expr, param: string, env: MathEnv): { x: number; y: number; z: number }[] {
  const path: { x: number; y: number; z: number }[] = []
  let previous = false
  for (let i = 0; i < CURVE_SAMPLES; i += 1) {
    const x = sampleNumber(i, CURVE_SAMPLES)
    const y = numericValue(body, bind(env, param, x))
    if (y === null || Math.abs(y) > CLIP) {
      if (previous) path.push({ x: Number.NaN, y: Number.NaN, z: Number.NaN })
      previous = false
      continue
    }
    path.push({ x, y, z: 0 })
    previous = true
  }
  return path
}

function sampleSurface(body: Expr, xName: string, yName: string, env: MathEnv): { x: number; y: number; z: number }[][] {
  const grid: { x: number; y: number; z: number }[][] = []
  for (let row = 0; row < SURFACE_SAMPLES; row += 1) {
    const y = sampleNumber(row, SURFACE_SAMPLES)
    const line: { x: number; y: number; z: number }[] = []
    for (let col = 0; col < SURFACE_SAMPLES; col += 1) {
      const x = sampleNumber(col, SURFACE_SAMPLES)
      let local = bind(env, xName, x)
      local = bind(local, yName, y)
      const z = numericValue(body, local)
      line.push({ x, y, z: z === null || Math.abs(z) > CLIP ? Number.NaN : z })
    }
    grid.push(line)
  }
  return grid
}

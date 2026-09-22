// Evaluate math statements in document order. A one-input function is a curve.
// y = and x = are curves too. z = and z^2 = are surfaces. Other results stay in the console.

import type { Statement } from '../document'
import {
  MathError,
  applyEnv,
  approximate,
  freeSymbols,
  normalize,
  numericValue,
  parseMathInput,
  plain,
  present,
  solveEquation,
  tex,
  type AngleMode,
  type Expr,
  type MathEnv,
} from './expr'

const PLOT_COLORS = ['#59d67f', '#f5a524', '#a78bfa', '#fb6a6a', '#5aa8ff', '#e879f9']
const CURVE_SAMPLES = 201
const SURFACE_SAMPLES = 61
const CLIP = 1e4
const AXES = new Set(['x', 'y', 'z'])

export interface PlotWindow {
  xMin: number
  xMax: number
  yMin: number
  yMax: number
}

export const DEFAULT_WINDOW: PlotWindow = { xMin: -10, xMax: 10, yMin: -10, yMax: 10 }

export interface MathConsoleRow {
  statementId: string
  label: string
  /** The line the user typed. */
  input: string
  text: string
  tex: string | null
  exactTex: string | null
  exactText: string | null
  approxTex: string | null
  approxText: string | null
  preferDecimal: boolean
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
  sample: (window: PlotWindow) => { x: number; y: number; z: number }[]
}

export interface SurfacePlot {
  kind: 'surface'
  statementId: string
  label: string
  color: string
  visible: boolean
  grid: { x: number; y: number; z: number }[][]
  sheets: { x: number; y: number; z: number }[][][]
}

export interface MathCompilation {
  rows: MathConsoleRow[]
  plots: Array<CurvePlot | SurfacePlot>
}

export function compileMath(statements: Statement[], angles: AngleMode = 'rad'): MathCompilation {
  const env: MathEnv = new Map()
  const rows: MathConsoleRow[] = []
  const plots: Array<CurvePlot | SurfacePlot> = []
  let colorIndex = 0

  const nextColor = () => {
    const color = PLOT_COLORS[colorIndex % PLOT_COLORS.length]
    colorIndex += 1
    return color
  }

  for (const statement of statements) {
    if (statement.type !== 'math') continue
    try {
      const parsed = parseMathInput(statement.input)
      if (parsed.kind === 'fn') {
        env.set(parsed.name, { kind: 'fn', params: parsed.params, body: parsed.body })
        const label = `${parsed.name}(${parsed.params.join(', ')})`
        const formulaTex = `${labelTex(parsed.name, parsed.params)} = ${tex(parsed.body)}`
        const formulaText = `${label} = ${plain(parsed.body)}`
        const missing = freeSymbols(parsed.body).filter((name) => !parsed.params.includes(name) && !env.has(name))
        const color = nextColor()
        const plot = parsed.params.length === 2
          ? surfacePlot(statement.id, label, color, statement.visible, missing.length ? [] : [parsed.body], parsed.params[0], parsed.params[1], env, angles)
          : curvePlot(statement.id, label, color, statement.visible, missing.length ? null : { bodies: [parsed.body], param: parsed.params[0], along: 'x' }, env, angles)
        if (missing.length > 0) plot.warn = `Give ${missing.join(', ')} a value above this line to draw the graph.`
        else if (!hasGeometry(plot.plot)) plot.warn = 'No real values to plot on [-10, 10].'
        plots.push(plot.plot)
        rows.push(rowBase(statement.id, statement.input, label, formulaText, formulaTex, plot.plot.kind, statement.visible, plot.warn))
        continue
      }
      const equation: { left: Expr; right: Expr } | null = parsed.kind === 'assign'
        ? { left: { type: 'sym', name: parsed.name }, right: parsed.expr }
        : parsed.kind === 'expr' && parsed.expr.type === 'eq'
          ? parsed.expr
          : null
      if (equation) {
        const dep = dependentAxis(equation.left)
        const plotted = dep ? explicitPlot(dep, equation.right, env, angles) : null
        if (parsed.kind === 'assign' && !plotted) {
          const value = normalize(applyEnv(parsed.expr, env, 0), angles)
          env.set(parsed.name, { kind: 'expr', expr: value })
          const shown = described(parsed.expr, value, statement.input, angles)
          const nameTex = tex({ type: 'sym', name: parsed.name })
          const exactText = `${parsed.name} = ${shown.exactText ?? ''}`
          const exactTex = `${nameTex} = ${shown.exactTex ?? ''}`
          const approxText = shown.approxText ? `${parsed.name} = ${shown.approxText}` : null
          const approxTex = shown.approxTex ? `${nameTex} = ${shown.approxTex}` : null
          const useDecimal = Boolean(shown.preferDecimal && approxText && approxTex)
          rows.push({
            statementId: statement.id,
            label: parsed.name,
            input: statement.input,
            text: useDecimal && approxText ? approxText : exactText,
            tex: useDecimal && approxTex ? approxTex : exactTex,
            exactTex,
            exactText,
            approxTex,
            approxText,
            preferDecimal: useDecimal,
            plotKind: null,
            visible: true,
            warn: null,
          })
          continue
        }
        if (plotted) {
          if (parsed.kind === 'assign') {
            const value = normalize(applyEnv(parsed.expr, env, 0), angles)
            env.set(parsed.name, { kind: 'expr', expr: value })
          }
          const color = nextColor()
          const label = dep && dep.power === 1 ? dep.name : plain(equation.left)
          const formulaTex = `${tex(equation.left)} = ${tex(equation.right)}`
          const formulaText = `${plain(equation.left)} = ${plain(equation.right)}`
          const plot = plotted.kind === 'surface'
            ? surfacePlot(statement.id, label, color, statement.visible, plotted.bodies, 'x', 'y', env, angles)
            : curvePlot(statement.id, label, color, statement.visible, { bodies: plotted.bodies, param: plotted.param, along: plotted.along }, env, angles)
          if (plotted.warn) plot.warn = plotted.warn
          else if (!hasGeometry(plot.plot)) plot.warn = 'No real values to plot on [-10, 10].'
          plots.push(plot.plot)
          rows.push(rowBase(statement.id, statement.input, label, formulaText, formulaTex, plot.plot.kind, statement.visible, plot.warn))
          continue
        }
      }
      if (parsed.kind === 'solve') {
        const equation = applyEnv(parsed.equation, env, 0)
        const solved = solveEquation(equation, parsed.variable)
        rows.push({
          statementId: statement.id,
          label: 'Solve',
          input: statement.input,
          text: solved.text,
          tex: solved.tex,
          exactTex: solved.tex,
          exactText: solved.text,
          approxTex: null,
          approxText: null,
          preferDecimal: false,
          plotKind: null,
          visible: true,
          warn: null,
        })
        continue
      }
      const expr = parsed.kind === 'expr' ? parsed.expr : null
      if (!expr) continue
      const value = normalize(applyEnv(expr, env, 0), angles)
      if (value.type === 'eq') {
        const left = present(normalize(value.left, angles), angles)
        const right = present(normalize(value.right, angles), angles)
        rows.push({
          statementId: statement.id,
          label: 'Result',
          input: statement.input,
          text: `${left.text} = ${right.text}`,
          tex: `${left.tex} = ${right.tex}`,
          exactTex: `${left.tex} = ${right.tex}`,
          exactText: `${left.text} = ${right.text}`,
          approxTex: null,
          approxText: null,
          preferDecimal: false,
          plotKind: null,
          visible: true,
          warn: null,
        })
        continue
      }
      const shown = described(expr, value, statement.input, angles)
      rows.push({ ...shown, statementId: statement.id, label: 'Result', plotKind: null, visible: true, warn: null })
    } catch (error) {
      const message = error instanceof MathError ? error.message : 'Could not read that.'
      rows.push({
        statementId: statement.id,
        label: 'Math',
        input: statement.input,
        text: message,
        tex: null,
        exactTex: null,
        exactText: null,
        approxTex: null,
        approxText: null,
        preferDecimal: false,
        plotKind: null,
        visible: statement.visible,
        warn: message,
      })
    }
  }

  return { rows, plots }
}

function rowBase(id: string, input: string, label: string, text: string, formula: string, plotKind: 'curve' | 'surface', visible: boolean, warn: string | null): MathConsoleRow {
  return {
    statementId: id,
    label,
    input,
    text: warn ?? text,
    tex: formula,
    exactTex: formula,
    exactText: text,
    approxTex: null,
    approxText: null,
    preferDecimal: false,
    plotKind,
    visible,
    warn,
  }
}

function described(input: Expr, value: Expr, raw: string, angles: AngleMode): Omit<MathConsoleRow, 'statementId' | 'label' | 'plotKind' | 'visible' | 'warn'> {
  const exact = present(value, angles)
  const approx = approximate(value, angles)
  const same = !approx || approx.text === exact.text
  const unchanged = plain(input) === exact.text
  const preferDecimal = Boolean(approx && !same && unchanged && !/^-?\d+(?:\/\d+)?$/.test(exact.text))
  const shown = preferDecimal && approx ? approx : exact
  const text = plain(input) === shown.text ? shown.text : `${plain(input)} = ${shown.text}`
  return {
    input: raw,
    text,
    tex: shown.tex,
    exactTex: exact.tex,
    exactText: exact.text,
    approxTex: approx && !same ? approx.tex : null,
    approxText: approx && !same ? approx.text : null,
    preferDecimal,
  }
}

function dependentAxis(left: Expr): { name: 'x' | 'y' | 'z'; power: number } | null {
  if (left.type === 'sym' && AXES.has(left.name)) return { name: left.name as 'x' | 'y' | 'z', power: 1 }
  if (left.type === 'pow' && left.base.type === 'sym' && AXES.has(left.base.name) && left.exp.type === 'rat' && left.exp.d === 1n && left.exp.n >= 1n && left.exp.n <= 6n) {
    return { name: left.base.name as 'x' | 'y' | 'z', power: Number(left.exp.n) }
  }
  return null
}

function explicitPlot(dep: { name: 'x' | 'y' | 'z'; power: number }, rhs: Expr, env: MathEnv, angles: AngleMode): { kind: 'curve'; bodies: Expr[]; param: string; along: 'x' | 'y'; warn: string | null } | { kind: 'surface'; bodies: Expr[]; warn: string | null } | null {
  const value = normalize(applyEnv(rhs, env, 0), angles)
  if (freeSymbols(value).includes(dep.name)) return null
  const roots = rootBodies(value, dep.power)
  if (dep.name === 'z') {
    const unknown = freeSymbols(value).filter((name) => name !== 'x' && name !== 'y')
    if (unknown.length > 0) return { kind: 'surface', bodies: [], warn: `Give ${unknown.join(', ')} a value above this line to draw the graph.` }
    return { kind: 'surface', bodies: roots, warn: null }
  }
  const param = dep.name === 'y' ? 'x' : 'y'
  const unknown = freeSymbols(value).filter((name) => name !== param)
  if (unknown.length > 0) return { kind: 'curve', bodies: [], param, along: param, warn: `Give ${unknown.join(', ')} a value above this line to draw the graph.` }
  return { kind: 'curve', bodies: roots, param, along: param, warn: null }
}

function rootBodies(rhs: Expr, power: number): Expr[] {
  if (power === 1) return [rhs]
  const root: Expr = { type: 'pow', base: rhs, exp: { type: 'rat', n: 1n, d: BigInt(power) } }
  if (power % 2 === 0) return [root, { type: 'mul', args: [{ type: 'rat', n: -1n, d: 1n }, root] }]
  return [root]
}

function hasGeometry(plot: CurvePlot | SurfacePlot): boolean {
  if (plot.kind === 'curve') return plot.path.some((point) => Number.isFinite(point.x) && Number.isFinite(point.y))
  return plot.sheets.some((grid) => grid.some((row) => row.some((point) => Number.isFinite(point.z))))
}

function curvePlot(id: string, label: string, color: string, visible: boolean, spec: { bodies: Expr[]; param: string; along: 'x' | 'y' } | null, env: MathEnv, angles: AngleMode): { plot: CurvePlot; warn: string | null } {
  const sample = (window: PlotWindow) => (spec && spec.bodies.length > 0 ? sampleBranches(spec.bodies, spec.param, spec.along, env, window, angles) : [])
  return { warn: null, plot: { kind: 'curve', statementId: id, label, color, visible, sample, path: sample(DEFAULT_WINDOW) } }
}

function surfacePlot(id: string, label: string, color: string, visible: boolean, bodies: Expr[], xName: string, yName: string, env: MathEnv, angles: AngleMode): { plot: SurfacePlot; warn: string | null } {
  const sheets = bodies.length > 0 ? bodies.map((body) => sampleSurface(body, xName, yName, env, DEFAULT_WINDOW, angles)) : []
  return { warn: null, plot: { kind: 'surface', statementId: id, label, color, visible, sheets, grid: sheets[0] ?? [] } }
}

function labelTex(name: string, params: string[]): string {
  return `${name}\\left(${params.join(', ')}\\right)`
}

function sampleAt(index: number, count: number, min: number, max: number): number {
  if (count === 1) return (min + max) / 2
  return min + ((max - min) * index) / (count - 1)
}

function bind(env: MathEnv, name: string, value: number): MathEnv {
  const next: MathEnv = new Map(env)
  next.set(name, { kind: 'expr', expr: { type: 'dec', text: String(value), value } })
  return next
}

function sampleBranches(bodies: Expr[], param: string, along: 'x' | 'y', env: MathEnv, window: PlotWindow, angles: AngleMode): { x: number; y: number; z: number }[] {
  const path: { x: number; y: number; z: number }[] = []
  const min = along === 'x' ? window.xMin : window.yMin
  const max = along === 'x' ? window.xMax : window.yMax
  const clip = Math.max(CLIP, 40 * Math.abs(window.xMax - window.xMin), 40 * Math.abs(window.yMax - window.yMin))
  for (const body of bodies) {
    let previous = false
    for (let i = 0; i < CURVE_SAMPLES; i += 1) {
      const t = sampleAt(i, CURVE_SAMPLES, min, max)
      const value = numericValue(body, bind(env, param, t), angles)
      const x = along === 'x' ? t : value
      const y = along === 'x' ? value : t
      if (x === null || y === null || !Number.isFinite(x) || !Number.isFinite(y) || Math.abs(x) > clip || Math.abs(y) > clip) {
        if (previous) path.push({ x: Number.NaN, y: Number.NaN, z: Number.NaN })
        previous = false
        continue
      }
      path.push({ x, y, z: 0 })
      previous = true
    }
    path.push({ x: Number.NaN, y: Number.NaN, z: Number.NaN })
  }
  return path
}

function sampleSurface(body: Expr, xName: string, yName: string, env: MathEnv, window: PlotWindow, angles: AngleMode): { x: number; y: number; z: number }[][] {
  const grid: { x: number; y: number; z: number }[][] = []
  for (let row = 0; row < SURFACE_SAMPLES; row += 1) {
    const y = sampleAt(row, SURFACE_SAMPLES, window.yMin, window.yMax)
    const line: { x: number; y: number; z: number }[] = []
    for (let col = 0; col < SURFACE_SAMPLES; col += 1) {
      const x = sampleAt(col, SURFACE_SAMPLES, window.xMin, window.xMax)
      let local = bind(env, xName, x)
      local = bind(local, yName, y)
      const z = numericValue(body, local, angles)
      line.push({ x, y, z: z === null || Math.abs(z) > CLIP ? Number.NaN : z })
    }
    grid.push(line)
  }
  return grid
}

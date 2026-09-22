// Evaluate math statements in document order. A one-input function is a curve.
// y = and x = are curves too. z = and z^2 = are surfaces. Other results stay in the console.

import type { Statement } from '../document'
import { containsCas, evaluateCas, rewriteAll, solveLinearSystem, type CasArrow, type CasCurve, type CasParametric, type CasPoint } from './cas'
import {
  MathError,
  applyEnv,
  approximate,
  freeSymbols,
  normalize,
  numericConstant,
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
  dashed?: boolean
  /** A single point, drawn with a marker instead of a stroke. */
  marker?: boolean
  /** Draw an arrowhead at the end of the path. */
  arrow?: boolean
  /** Fill between the curve and the axis. */
  shade?: { from: number; to: number }
  /** The curve is only the shaded region because the same function is already drawn. */
  hideStroke?: boolean
  /** Which coordinate the shade interval is measured along. */
  along?: 'x' | 'y'
  /** Plain expression, used to avoid drawing the same curve twice. */
  exprKey?: string
}

export interface SurfacePlot {
  kind: 'surface'
  statementId: string
  label: string
  color: string
  visible: boolean
  grid: { x: number; y: number; z: number }[][]
  sheets: { x: number; y: number; z: number }[][][]
  sample: (window: PlotWindow) => { x: number; y: number; z: number }[][][]
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

  const pushVisual = (id: string, input: string, label: string, text: string, formula: string, curves: CasCurve[], points: CasPoint[], parametrics: CasParametric[], arrows: CasArrow[], visible: boolean, warn: string | null) => {
    const color = nextColor()
    let plotKind: 'curve' | null = null
    for (const curve of curves) {
      const hideStroke = Boolean(curve.shade && curve.exprKey && plots.some((plot) => plot.kind === 'curve' && plot.visible && !plot.hideStroke && plot.exprKey === curve.exprKey))
      const plot = curvePlot(id, curve.label, color, visible, { bodies: [curve.expr], param: curve.along, along: curve.along }, env, angles, { dashed: curve.dashed, shade: curve.shade, exprKey: curve.exprKey, hideStroke, along: curve.along })
      plots.push(plot.plot)
      plotKind = 'curve'
    }
    for (const parametric of parametrics) {
      plots.push(parametricPlot(id, parametric.label, color, visible, parametric.components, parametric.param, env, angles, parametric.dashed))
      plotKind = 'curve'
    }
    for (const arrow of arrows) {
      plots.push(arrowPlot(id, arrow.label, color, visible, arrow.x, arrow.y, arrow.z))
      plotKind = 'curve'
    }
    for (const point of points) {
      plots.push(pointPlot(id, point.label, color, visible, point.x, point.y))
      plotKind = 'curve'
    }
    rows.push({
      statementId: id,
      label,
      input,
      text: warn ? `${text} (${warn})` : text,
      tex: formula,
      exactTex: formula,
      exactText: text,
      approxTex: null,
      approxText: null,
      preferDecimal: false,
      plotKind,
      visible,
      warn,
    })
  }

  for (const statement of statements) {
    if (statement.type !== 'math') continue
    try {
      const parsed = parseMathInput(statement.input)
      if (parsed.kind === 'fn') {
        const label = `${parsed.name}(${parsed.params.join(', ')})`
        const formulaTex = `${labelTex(parsed.name, parsed.params)} = ${tex(parsed.body)}`
        const formulaText = `${label} = ${plain(parsed.body)}`
        const body = rewriteAll(parsed.body, angles)
        env.set(parsed.name, { kind: 'fn', params: parsed.params, body })
        if (parsed.params.length === 1 && body.type === 'vec' && body.args.length >= 2 && body.args.length <= 3) {
          const color = nextColor()
          const plot = parametricPlot(statement.id, label, color, statement.visible, body.args, parsed.params[0], env, angles)
          const warn = hasGeometry(plot) ? null : 'No real values to plot for t from -10 to 10.'
          plots.push(plot)
          rows.push(rowBase(statement.id, statement.input, label, formulaText, formulaTex, 'curve', statement.visible, warn))
          continue
        }
        const missing = freeSymbols(body).filter((name) => !parsed.params.includes(name) && !env.has(name))
        const color = nextColor()
        const plot = parsed.params.length === 2
          ? surfacePlot(statement.id, label, color, statement.visible, missing.length ? [] : [body], parsed.params[0], parsed.params[1], env, angles)
          : curvePlot(statement.id, label, color, statement.visible, missing.length ? null : { bodies: [body], param: parsed.params[0], along: 'x' }, env, angles)
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
          const picture = vectorPicture(value)
          if (picture) {
            const color = nextColor()
            plots.push(picture.kind === 'arrow' ? arrowPlot(statement.id, parsed.name, color, statement.visible, picture.x, picture.y, picture.z) : parametricPlot(statement.id, parsed.name, color, statement.visible, picture.components, picture.param, env, angles))
          }
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
            plotKind: picture ? 'curve' : null,
            visible: statement.visible,
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
      if (parsed.kind === 'system') {
        const solved = solveLinearSystem(parsed.equations.map((equation) => applyEnv(equation, env, 0)), angles)
        pushVisual(statement.id, statement.input, 'Solve', solved.text, solved.tex, solved.curves, solved.points, solved.parametrics, solved.arrows, statement.visible, solved.warn)
        continue
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
      const applied = applyEnv(expr, env, 0)
      const cas = evaluateCas(applied, angles)
      if (cas) {
        pushVisual(statement.id, statement.input, casLabel(applied), cas.text, cas.tex, cas.curves, cas.points, cas.parametrics, cas.arrows, statement.visible, cas.warn)
        continue
      }
      const value = normalize(rewriteAll(applied, angles), angles)
      const picture = vectorPicture(value)
      if (picture) {
        const color = nextColor()
        const drawn = picture.kind === 'arrow' ? arrowPlot(statement.id, 'vector', color, statement.visible, picture.x, picture.y, picture.z) : parametricPlot(statement.id, picture.param, color, statement.visible, picture.components, picture.param, env, angles)
        plots.push(drawn)
        const shown = described(expr, value, statement.input, angles)
        rows.push({ ...shown, statementId: statement.id, label: picture.kind === 'arrow' ? 'Vector' : picture.param, plotKind: 'curve', visible: statement.visible, warn: null })
        continue
      }
      if (containsCas(expr)) {
        const curve = singleCurve(value)
        if (curve) {
          const color = nextColor()
          const plot = curvePlot(statement.id, curve.label, color, statement.visible, { bodies: [curve.expr], param: curve.along, along: curve.along }, env, angles)
          plots.push(plot.plot)
          const shown = described(expr, value, statement.input, angles)
          rows.push({ ...shown, statementId: statement.id, label: curve.label, plotKind: 'curve', visible: statement.visible, warn: plot.warn })
          continue
        }
      }
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

function curvePlot(id: string, label: string, color: string, visible: boolean, spec: { bodies: Expr[]; param: string; along: 'x' | 'y' } | null, env: MathEnv, angles: AngleMode, options?: { dashed?: boolean; shade?: { from: number; to: number }; exprKey?: string; hideStroke?: boolean; along?: 'x' | 'y' }): { plot: CurvePlot; warn: string | null } {
  const sample = (window: PlotWindow) => (spec && spec.bodies.length > 0 ? sampleBranches(spec.bodies, spec.param, spec.along, env, window, angles) : [])
  const exprKey = options?.exprKey ?? (spec && spec.bodies.length === 1 ? plain(spec.bodies[0]) : undefined)
  return { warn: null, plot: { kind: 'curve', statementId: id, label, color, visible, sample, path: sample(DEFAULT_WINDOW), dashed: options?.dashed, shade: options?.shade, hideStroke: options?.hideStroke, along: options?.along ?? spec?.along, exprKey } }
}

const PARAM_SAMPLES = 241

function parametricPlot(id: string, label: string, color: string, visible: boolean, components: Expr[], param: string, env: MathEnv, angles: AngleMode, dashed?: boolean): CurvePlot {
  const sample = (_window: PlotWindow) => {
    const path: { x: number; y: number; z: number }[] = []
    for (let i = 0; i < PARAM_SAMPLES; i += 1) {
      const t = sampleAt(i, PARAM_SAMPLES, -10, 10)
      const local = bind(env, param, t)
      const values = components.map((component) => numericValue(component, local, angles))
      if (values.some((value) => value === null || !Number.isFinite(value))) {
        path.push({ x: Number.NaN, y: Number.NaN, z: Number.NaN })
        continue
      }
      path.push({ x: values[0] ?? Number.NaN, y: values[1] ?? Number.NaN, z: values[2] ?? 0 })
    }
    return path
  }
  return { kind: 'curve', statementId: id, label, color, visible, sample, path: sample(DEFAULT_WINDOW), dashed, along: 'x' }
}

function arrowPlot(id: string, label: string, color: string, visible: boolean, x: number, y: number, z: number): CurvePlot {
  const sample = () => [
    { x: 0, y: 0, z: 0 },
    { x, y, z },
  ]
  return { kind: 'curve', statementId: id, label, color, visible, sample, path: sample(), arrow: true }
}

function vectorPicture(value: Expr): { kind: 'arrow'; x: number; y: number; z: number } | { kind: 'parametric'; components: Expr[]; param: string } | null {
  if (value.type !== 'vec' || value.args.length < 2 || value.args.length > 3) return null
  const symbols = freeSymbols(value)
  if (symbols.length > 1) return null
  if (symbols.length === 1) return { kind: 'parametric', components: value.args, param: symbols[0] }
  const nums = value.args.map((arg) => numericConstant(arg))
  if (nums.some((item) => item === null)) return null
  return { kind: 'arrow', x: nums[0] ?? 0, y: nums[1] ?? 0, z: nums[2] ?? 0 }
}

function pointPlot(id: string, label: string, color: string, visible: boolean, x: number, y: number): CurvePlot {
  const sample = (window: PlotWindow) => (x < window.xMin || x > window.xMax || y < window.yMin || y > window.yMax ? [] : [{ x, y, z: 0 }])
  return { kind: 'curve', statementId: id, label, color, visible, sample, path: sample(DEFAULT_WINDOW), marker: true }
}

function surfacePlot(id: string, label: string, color: string, visible: boolean, bodies: Expr[], xName: string, yName: string, env: MathEnv, angles: AngleMode): { plot: SurfacePlot; warn: string | null } {
  const sample = (window: PlotWindow) => (bodies.length > 0 ? bodies.map((body) => sampleSurface(body, xName, yName, env, window, angles)) : [])
  const sheets = sample(DEFAULT_WINDOW)
  return { warn: null, plot: { kind: 'surface', statementId: id, label, color, visible, sheets, grid: sheets[0] ?? [], sample } }
}

function casLabel(expr: Expr): string {
  if (expr.type !== 'call') return 'Result'
  if (expr.name === 'diff') return 'Derivative'
  if (expr.name === 'integrate') return 'Integral'
  if (expr.name === 'zeros') return 'Zeros'
  if (expr.name === 'tangent') return 'Tangent'
  if (expr.name === 'normal') return 'Normal'
  if (expr.name === 'series') return 'Series'
  if (expr.name === 'fmin') return 'Minimum'
  if (expr.name === 'fmax') return 'Maximum'
  if (expr.name === 'dsolve') return 'Solution'
  return expr.name
}

function singleCurve(expr: Expr): CasCurve | null {
  const symbols = freeSymbols(expr)
  if (symbols.length !== 1) return null
  if (symbols[0] !== 'x' && symbols[0] !== 'y') return null
  return { expr, along: symbols[0], label: symbols[0] === 'x' ? 'y' : 'x' }
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

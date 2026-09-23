// Evaluate math statements in document order. A one-input function is a curve.
// y = and x = are curves too. z = and z^2 = are surfaces. Other results stay in the console.

import type { Statement } from '../document'
import { containsCas, evaluateCas, rewriteAll, solveSystem, type CasArrow, type CasCurve, type CasParametric, type CasPoint } from './cas'
import { sampleCurveNative } from './plotter'
import {
  DEFAULT_PLOT,
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
  texName,
  type AngleMode,
  type Expr,
  type MathEnv,
  type PlotOptions,
} from './expr'

const PLOT_COLORS = ['#59d67f', '#f5a524', '#a78bfa', '#fb6a6a', '#5aa8ff', '#e879f9']
const CLIP = 1e4
const MAX_CURVE_POINTS = 3600
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
          const plot = parametricPlot(statement.id, label, color, statement.visible, body.args, parsed.params[0], env, angles, undefined, parsed.plot)
          const warn = domainWarning(parsed.plot, angles, parsed.params[0]) ?? (hasGeometry(plot) ? null : missingDomain(parsed.plot, angles, parsed.params[0]))
          plots.push(plot)
          rows.push(rowBase(statement.id, statement.input, label, formulaText, formulaTex, 'curve', statement.visible, warn))
          continue
        }
        const missing = freeSymbols(body).filter((name) => !parsed.params.includes(name) && !env.has(name))
        const color = nextColor()
        const plot = parsed.params.length === 2
          ? surfacePlot(statement.id, label, color, statement.visible, missing.length ? [] : [body], parsed.params[0], parsed.params[1], env, angles, parsed.plot)
          : curvePlot(statement.id, label, color, statement.visible, missing.length ? null : { bodies: [body], param: parsed.params[0], along: 'x' }, env, angles, { plot: parsed.plot })
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
            plots.push(picture.kind === 'arrow' ? arrowPlot(statement.id, parsed.name, color, statement.visible, picture.x, picture.y, picture.z) : parametricPlot(statement.id, parsed.name, color, statement.visible, picture.components, picture.param, env, angles, undefined, parsed.plot))
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
            ? surfacePlot(statement.id, label, color, statement.visible, plotted.bodies, 'x', 'y', env, angles, parsed.plot)
            : curvePlot(statement.id, label, color, statement.visible, { bodies: plotted.bodies, param: plotted.param, along: plotted.along }, env, angles, { plot: parsed.plot })
          if (plotted.warn) plot.warn = plotted.warn
          else if (!hasGeometry(plot.plot)) plot.warn = 'No real values to plot on [-10, 10].'
          plots.push(plot.plot)
          rows.push(rowBase(statement.id, statement.input, label, formulaText, formulaTex, plot.plot.kind, statement.visible, plot.warn))
          continue
        }
      }
      if (parsed.kind === 'system') {
        const solved = solveSystem(parsed.equations.map((equation) => applyEnv(equation, env, 0)), parsed.domains, angles)
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
        const drawn = picture.kind === 'arrow' ? arrowPlot(statement.id, 'vector', color, statement.visible, picture.x, picture.y, picture.z) : parametricPlot(statement.id, picture.param, color, statement.visible, picture.components, picture.param, env, angles, undefined, parsed.plot)
        plots.push(drawn)
        const shown = described(expr, value, statement.input, angles)
        const warn = picture.kind === 'arrow' ? null : (domainWarning(parsed.plot, angles, picture.param) ?? (hasGeometry(drawn) ? null : missingDomain(parsed.plot, angles, picture.param)))
        rows.push({ ...shown, text: warn ? `${shown.text} (${warn})` : shown.text, statementId: statement.id, label: picture.kind === 'arrow' ? 'Vector' : picture.param, plotKind: 'curve', visible: statement.visible, warn })
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

function curvePlot(id: string, label: string, color: string, visible: boolean, spec: { bodies: Expr[]; param: string; along: 'x' | 'y' } | null, env: MathEnv, angles: AngleMode, options?: { dashed?: boolean; shade?: { from: number; to: number }; exprKey?: string; hideStroke?: boolean; along?: 'x' | 'y'; plot?: PlotOptions }): { plot: CurvePlot; warn: string | null } {
  const style = options?.plot ?? DEFAULT_PLOT
  const sample = (window: PlotWindow) => (spec && spec.bodies.length > 0 ? sampleBranches(spec.bodies, spec.param, spec.along, env, window, angles, style) : [])
  const exprKey = options?.exprKey ?? (spec && spec.bodies.length === 1 ? plain(spec.bodies[0]) : undefined)
  return { warn: null, plot: { kind: 'curve', statementId: id, label, color, visible, sample, path: sample(DEFAULT_WINDOW), dashed: options?.dashed, shade: options?.shade, hideStroke: options?.hideStroke, along: options?.along ?? spec?.along, exprKey } }
}

function parametricPlot(id: string, label: string, color: string, visible: boolean, components: Expr[], param: string, env: MathEnv, angles: AngleMode, dashed?: boolean, plot: PlotOptions = DEFAULT_PLOT): CurvePlot {
  const sample = (window: PlotWindow) => sampleParametric(components, param, env, window, angles, plot)
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

function surfacePlot(id: string, label: string, color: string, visible: boolean, bodies: Expr[], xName: string, yName: string, env: MathEnv, angles: AngleMode, plot: PlotOptions = DEFAULT_PLOT): { plot: SurfacePlot; warn: string | null } {
  const sample = (window: PlotWindow) => (bodies.length > 0 ? bodies.map((body) => sampleSurface(body, xName, yName, env, window, angles, plot)) : [])
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
  return `${texName(name)}\\left(${params.map((param) => texName(param)).join(', ')}\\right)`
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

interface CurveNode {
  t: number
  y: number | null
  cut: boolean
}

function sampleBranches(bodies: Expr[], param: string, along: 'x' | 'y', env: MathEnv, window: PlotWindow, angles: AngleMode, style: PlotOptions): { x: number; y: number; z: number }[] {
  const path: { x: number; y: number; z: number }[] = []
  const min = along === 'x' ? window.xMin : window.yMin
  const max = along === 'x' ? window.xMax : window.yMax
  const ySpan = Math.max(1e-6, along === 'x' ? window.yMax - window.yMin : window.xMax - window.xMin)
  const clip = Math.max(CLIP, 40 * Math.abs(window.xMax - window.xMin), 40 * Math.abs(window.yMax - window.yMin))
  for (const body of bodies) {
    const at = (t: number) => {
      const value = numericValue(body, bind(env, param, t), angles)
      return value === null || !Number.isFinite(value) ? null : value
    }
    const native = sampleCurveNative(body, param, env, min, max, ySpan, angles === 'deg', style)
    const nodes = native ?? refineSamples(min, max, { points: Math.min(style.points, 160), recursion: Math.min(style.recursion, 4), exclusions: style.exclusions, domain: null }, ySpan, at)
    let previous = false
    for (const node of nodes) {
      const value = node.y
      const x = along === 'x' ? node.t : value
      const y = along === 'x' ? value : node.t
      const bad = x === null || y === null || !Number.isFinite(x) || !Number.isFinite(y) || Math.abs(x) > clip || Math.abs(y) > clip
      if (bad) {
        if (previous) path.push({ x: Number.NaN, y: Number.NaN, z: Number.NaN })
        previous = false
        continue
      }
      path.push({ x, y, z: 0 })
      previous = true
      if (node.cut) {
        path.push({ x: Number.NaN, y: Number.NaN, z: Number.NaN })
        previous = false
      }
    }
    path.push({ x: Number.NaN, y: Number.NaN, z: Number.NaN })
  }
  return path
}

function refineSamples(min: number, max: number, style: PlotOptions, ySpan: number, at: (t: number) => number | null): CurveNode[] {
  const count = Math.max(12, style.points)
  const span = Math.max(1e-12, max - min)
  const times: number[] = []
  for (let i = 0; i < count; i += 1) times.push(sampleAt(i, count, min, max))
  if (min < 0 && max > 0) {
    let nearest = 0
    for (let i = 1; i < times.length; i += 1) if (Math.abs(times[i] ?? 0) < Math.abs(times[nearest] ?? 0)) nearest = i
    if (Math.abs(times[nearest] ?? 0) > 1e-12) {
      times[nearest] = 0
      times.sort((left, right) => left - right)
    }
  }
  let nodes: CurveNode[] = times.map((t) => ({ t, y: at(t), cut: false }))
  for (let level = 0; level <= style.recursion; level += 1) {
    const last = level === style.recursion
    const next: CurveNode[] = []
    let grew = false
    for (let i = 0; i < nodes.length - 1; i += 1) {
      const left = nodes[i]
      const right = nodes[i + 1]
      if (!left || !right) continue
      next.push({ ...left, cut: false })
      const midT = (left.t + right.t) / 2
      const tooFine = right.t - left.t < span / 10000
      const midY = at(midT)
      if (!last && !tooFine && nodes.length < MAX_CURVE_POINTS && shouldSplit(left.y, midY, right.y, ySpan)) {
        next.push({ t: midT, y: midY, cut: false })
        grew = true
      } else if (style.exclusions && isDiscontinuity(left.y, midY, right.y, ySpan)) next[next.length - 1].cut = true
    }
    const tail = nodes[nodes.length - 1]
    if (tail) next.push({ ...tail, cut: false })
    nodes = next
    if (!grew) break
  }
  if (style.exclusions) markSpikes(nodes, ySpan)
  return nodes
}

function markSpikes(nodes: CurveNode[], ySpan: number): void {
  for (let index = 1; index < nodes.length - 1; index += 1) {
    const left = nodes[index - 1]?.y
    const mid = nodes[index]?.y
    const right = nodes[index + 1]?.y
    if (left === null || left === undefined || mid === null || mid === undefined || right === null || right === undefined) continue
    const peak = Math.max(Math.abs(left), Math.abs(right))
    if (Math.abs(mid) > peak * 3 + ySpan && Math.abs(mid) > ySpan * 2) {
      const before = nodes[index - 1]
      const at = nodes[index]
      if (before) before.cut = true
      if (at) at.cut = true
    }
  }
}

function shouldSplit(left: number | null, mid: number | null, right: number | null, ySpan: number): boolean {
  if (left === null || right === null || mid === null) return true
  const bend = Math.abs(mid - (left + right) / 2)
  const spike = Math.abs(mid) > Math.max(Math.abs(left), Math.abs(right)) * 3 + ySpan * 0.35
  return bend > Math.max(ySpan * 0.012, 1e-4) || spike
}

function isDiscontinuity(left: number | null, mid: number | null, right: number | null, ySpan: number): boolean {
  if (left === null || right === null || mid === null) return true
  const between = (mid - left) * (mid - right) <= 0
  if (between) return false
  const peak = Math.max(Math.abs(left), Math.abs(right))
  return Math.abs(mid) > peak * 1.5 + ySpan * 0.25
}

function parametricRange(plot: PlotOptions, angles: AngleMode): { min: number; max: number } | null {
  if (!plot.domain) return { min: -10, max: 10 }
  const min = numericConstant(plot.domain.min, angles)
  const max = numericConstant(plot.domain.max, angles)
  if (min === null || max === null || !Number.isFinite(min) || !Number.isFinite(max) || min === max) return null
  return min < max ? { min, max } : { min: max, max: min }
}

function domainWarning(plot: PlotOptions, angles: AngleMode, param: string): string | null {
  if (!plot.domain) return null
  if (parametricRange(plot, angles)) return null
  return `The domain of ${param} needs two different numbers, for example ${param} = 0..2*pi.`
}

function missingDomain(plot: PlotOptions, angles: AngleMode, param: string): string {
  const range = parametricRange(plot, angles) ?? { min: -10, max: 10 }
  const bound = (n: number) => String(Math.round(n * 1000) / 1000)
  return `No real values to plot for ${param} from ${bound(range.min)} to ${bound(range.max)}.`
}

function sampleParametric(components: Expr[], param: string, env: MathEnv, window: PlotWindow, angles: AngleMode, style: PlotOptions): { x: number; y: number; z: number }[] {
  const ySpan = Math.max(1e-6, window.yMax - window.yMin, window.xMax - window.xMin)
  const range = parametricRange(style, angles) ?? { min: -10, max: 10 }
  const at = (t: number) => {
    const local = bind(env, param, t)
    const values = components.map((component) => numericValue(component, local, angles))
    if (values.some((value) => value === null || !Number.isFinite(value))) return null
    return values[0] ?? null
  }
  const nodes = refineSamples(range.min, range.max, style, ySpan, at)
  const path: { x: number; y: number; z: number }[] = []
  for (const node of nodes) {
    if (node.y === null || node.cut) {
      path.push({ x: Number.NaN, y: Number.NaN, z: Number.NaN })
      continue
    }
    const local = bind(env, param, node.t)
    const values = components.map((component) => numericValue(component, local, angles))
    if (values.some((value) => value === null || !Number.isFinite(value))) {
      path.push({ x: Number.NaN, y: Number.NaN, z: Number.NaN })
      continue
    }
    path.push({ x: values[0] ?? Number.NaN, y: values[1] ?? Number.NaN, z: values[2] ?? 0 })
  }
  return path
}

function sampleSurface(body: Expr, xName: string, yName: string, env: MathEnv, window: PlotWindow, angles: AngleMode, plot: PlotOptions): { x: number; y: number; z: number }[][] {
  const count = Math.max(24, Math.min(80, Math.round(plot.points / 2)))
  const grid: { x: number; y: number; z: number }[][] = []
  const zSpan = Math.max(1, window.yMax - window.yMin)
  for (let row = 0; row < count; row += 1) {
    const y = sampleAt(row, count, window.yMin, window.yMax)
    const line: { x: number; y: number; z: number }[] = []
    for (let col = 0; col < count; col += 1) {
      const x = sampleAt(col, count, window.xMin, window.xMax)
      let local = bind(env, xName, x)
      local = bind(local, yName, y)
      const z = numericValue(body, local, angles)
      const previous = line[line.length - 1]
      const jump = previous && z !== null && Number.isFinite(previous.z) && Number.isFinite(z) && Math.abs(z - previous.z) > zSpan * 6
      line.push({ x, y, z: z === null || Math.abs(z) > CLIP || (plot.exclusions && jump) ? Number.NaN : z })
    }
    grid.push(line)
  }
  return grid
}

// Evaluate math statements in document order. A one-input function is a curve.
// y = and x = are curves too. z = and z^2 = are surfaces. Other results stay in the console.

import type { Statement } from '../document'
import { containsCas, evaluateCas, rewriteAll, solveSingle, solveSystem, type CasCurve, type CasVisual } from './cas'
import { functionByKernel } from './functions'
import { sampleCurveNative } from './plotter'
import { INFER_NAME, parseMathInput, previewTex } from './syntax'
import {
  DEFAULT_PLOT,
  MathError,
  applyEnv,
  approximate,
  freeSymbols,
  normalize,
  numericConstant,
  numericValue,
  plain,
  present,
  tex,
  texName,
  type AngleMode,
  type Expr,
  type MathEnv,
  type PlotDomain,
  type PlotOptions,
  type SearchDomain,
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
  /** Which coordinate the shade interval is measured along. */
  along?: 'x' | 'y'
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

type MathStatement = Extract<Statement, { type: 'math' }>

interface Interval {
  min: number
  max: number
}

const DEFAULT_RANGE: Interval = { min: -10, max: 10 }

function guessNames(expr: Expr, label: string, count: number): string[] {
  const names = freeSymbols(expr).filter((name) => name !== INFER_NAME)
  if (count === 1 && names.length <= 1) return [names[0] ?? 'x']
  if (names.length === count) {
    if (count === 2 && names.includes('x') && names.includes('y')) return ['x', 'y']
    return names
  }
  const sample = count === 2 ? 'Plot3D(x^2 - y^2, x, y)' : `${label}(x^2, x)`
  throw new MathError(names.length === 0 ? `${label} needs a variable, for example ${sample}.` : `${label} needs ${count === 1 ? 'a variable' : `${count} variables`}. This one has ${names.join(' and ')}, so name ${count === 1 ? 'it' : 'them'}, as in ${sample}.`)
}

function mapExpr(e: Expr, visit: (node: Expr) => Expr): Expr {
  switch (e.type) {
    case 'add':
    case 'mul':
    case 'vec':
      return { ...e, args: e.args.map(visit) }
    case 'div':
      return { ...e, num: visit(e.num), den: visit(e.den) }
    case 'pow':
      return { ...e, base: visit(e.base), exp: visit(e.exp) }
    case 'call':
      return { ...e, args: e.args.map(visit) }
    case 'mat':
      return { ...e, rows: e.rows.map((row) => row.map(visit)) }
    case 'eq':
      return { ...e, left: visit(e.left), right: visit(e.right) }
    case 'group':
      return { ...e, body: visit(e.body) }
    case 'caret':
      return { ...e, body: e.body ? visit(e.body) : null }
    default:
      return e
  }
}

/** Fill a variable that was left blank when the expression uses only one. */
function fillInferred(e: Expr, env?: MathEnv): Expr {
  const node = mapExpr(e, (child) => fillInferred(child, env))
  if (node.type !== 'call') return node
  const spec = functionByKernel(node.name)
  const args = node.args.map((arg) => {
    if (arg.type !== 'sym' || arg.name !== INFER_NAME) return arg
    const host = node.args[0] ?? arg
    return { type: 'sym' as const, name: guessNames(env ? applyEnv(host, env, 0) : host, spec?.name ?? 'This', 1)[0] ?? 'x' }
  })
  return args.some((arg, index) => arg !== node.args[index]) ? { ...node, args } : node
}

function quietRow(row: MathConsoleRow, input: string): void {
  const echo = previewTex(input)
  row.text = input
  row.tex = echo
  row.exactTex = echo
  row.exactText = input
  row.approxTex = null
  row.approxText = null
  row.plotKind = null
  row.warn = null
  row.visible = true
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
  const colorFor = (plot: PlotOptions) => plot.color ?? nextColor()

  const pushVisual = (statement: MathStatement, label: string, visual: CasVisual, plot: PlotOptions) => {
    const color = colorFor(plot)
    let plotKind: 'curve' | null = null
    let warn = visual.warn
    const shaded = visual.curves.find((curve) => !curve.dashed) ?? visual.curves[0]
    if (!shaded) warn = warn ?? curveOnlyShade(plot)
    for (const curve of visual.curves) {
      const clip = rangeOf(plot, curve.along, env, angles)
      const fill = curve === shaded ? shadeOf(plot, clip.interval, env, angles) : NO_SHADE
      warn = warn ?? clip.warn ?? fill.warn
      plots.push(curvePlot(statement.id, curve.label, color, statement.visible, { bodies: [curve.expr], param: curve.along, along: curve.along }, env, angles, { dashed: curve.dashed || plot.dashed, shade: fill.shade, along: curve.along, plot, clip: clip.interval }))
      plotKind = 'curve'
    }
    for (const parametric of visual.parametrics) {
      const range = rangeOf(plot, parametric.param, env, angles)
      warn = warn ?? range.warn
      plots.push(parametricPlot(statement.id, parametric.label, color, statement.visible, parametric.components, parametric.param, env, angles, plot, range.interval ?? DEFAULT_RANGE, parametric.dashed || plot.dashed))
      plotKind = 'curve'
    }
    for (const arrow of visual.arrows) {
      plots.push(arrowPlot(statement.id, arrow.label, color, statement.visible, arrow.x, arrow.y, arrow.z))
      plotKind = 'curve'
    }
    for (const point of visual.points) {
      plots.push(pointPlot(statement.id, point.label, color, statement.visible, point.x, point.y))
      plotKind = 'curve'
    }
    rows.push({
      statementId: statement.id,
      label,
      input: statement.input,
      text: warn ? `${visual.text} (${warn})` : visual.text,
      tex: visual.tex,
      exactTex: visual.tex,
      exactText: visual.text,
      approxTex: null,
      approxText: null,
      preferDecimal: false,
      plotKind,
      visible: statement.visible,
      warn,
    })
  }

  for (const statement of statements) {
    if (statement.type !== 'math') continue
    const plotMark = plots.length
    const rowMark = rows.length
    let silent = false
    let failed = false
    try {
      const parsed = parseMathInput(statement.input)
      silent = parsed.silent
      const echo = previewTex(statement.input)
      if (parsed.kind === 'fn') {
        const label = `${parsed.name}(${parsed.params.join(', ')})`
        const body = fillInferred(rewriteAll(parsed.body, angles), env)
        const computed = containsCas(parsed.body)
        const shownBody = computed ? tidy(body, angles) : parsed.body
        const formulaTex = computed || !echo ? `${labelTex(parsed.name, parsed.params)} = ${tex(shownBody)}` : echo
        const formulaText = `${label} = ${plain(shownBody)}`
        env.set(parsed.name, { kind: 'fn', params: parsed.params, body })
        const param = parsed.params[0] ?? 'x'
        if (parsed.params.length === 1 && body.type === 'vec' && body.args.length >= 2 && body.args.length <= 3) {
          const range = rangeOf(parsed.plot, param, env, angles)
          const drawn = parametricPlot(statement.id, label, colorFor(parsed.plot), statement.visible, body.args, param, env, angles, parsed.plot, range.interval ?? DEFAULT_RANGE, parsed.plot.dashed)
          const warn = range.warn ?? curveOnlyShade(parsed.plot) ?? (hasGeometry(drawn) ? null : emptyWarning(param, range.interval ?? DEFAULT_RANGE))
          plots.push(drawn)
          rows.push(rowBase(statement.id, statement.input, label, formulaText, formulaTex, 'curve', statement.visible, warn))
          continue
        }
        const missing = freeSymbols(body).filter((name) => !parsed.params.includes(name) && !env.has(name))
        const color = colorFor(parsed.plot)
        if (parsed.params.length === 2) {
          const names: [string, string] = [param, parsed.params[1] ?? 'y']
          const clips = surfaceRanges(parsed.plot, names, env, angles)
          const drawn = surfacePlot(statement.id, label, color, statement.visible, missing.length ? [] : [body], names, env, angles, parsed.plot, clips)
          const warn = missing.length > 0 ? giveValue(missing) : (clips.warn ?? curveOnlyShade(parsed.plot) ?? (hasGeometry(drawn) ? null : emptyWarning(param, null)))
          plots.push(drawn)
          rows.push(rowBase(statement.id, statement.input, label, formulaText, formulaTex, 'surface', statement.visible, warn))
          continue
        }
        const clip = rangeOf(parsed.plot, param, env, angles)
        const fill = shadeOf(parsed.plot, clip.interval, env, angles)
        const drawn = curvePlot(statement.id, label, color, statement.visible, missing.length ? null : { bodies: [body], param, along: 'x' }, env, angles, { plot: parsed.plot, clip: clip.interval, dashed: parsed.plot.dashed, shade: fill.shade })
        const warn = missing.length > 0 ? giveValue(missing) : (clip.warn ?? fill.warn ?? (hasGeometry(drawn) ? null : emptyWarning(param, clip.interval)))
        plots.push(drawn)
        rows.push(rowBase(statement.id, statement.input, label, formulaText, formulaTex, 'curve', statement.visible, warn))
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
          const value = normalize(rewriteAll(applyEnv(parsed.expr, env, 0), angles), angles)
          env.set(parsed.name, { kind: 'expr', expr: value })
          const shown = described(parsed.expr, value, statement.input, angles)
          const picture = vectorPicture(value)
          let warn: string | null = null
          if (picture) {
            const color = colorFor(parsed.plot)
            warn = curveOnlyShade(parsed.plot)
            if (picture.kind === 'arrow') plots.push(arrowPlot(statement.id, parsed.name, color, statement.visible, picture.x, picture.y, picture.z))
            else {
              const range = rangeOf(parsed.plot, picture.param, env, angles)
              warn = range.warn ?? warn
              plots.push(parametricPlot(statement.id, parsed.name, color, statement.visible, picture.components, picture.param, env, angles, parsed.plot, range.interval ?? DEFAULT_RANGE, parsed.plot.dashed))
            }
          }
          const nameTex = tex({ type: 'sym', name: parsed.name })
          const exactText = `${parsed.name} = ${shown.exactText ?? ''}`
          const exactTex = `${nameTex} = ${shown.exactTex ?? ''}`
          const approxText = shown.approxText ? `${parsed.name} = ${shown.approxText}` : null
          const approxTex = shown.approxTex ? `${nameTex} = ${shown.approxTex}` : null
          const useDecimal = Boolean(shown.preferDecimal && approxText && approxTex)
          const text = useDecimal && approxText ? approxText : exactText
          rows.push({
            statementId: statement.id,
            label: parsed.name,
            input: statement.input,
            text: warn ? `${text} (${warn})` : text,
            tex: useDecimal && approxTex ? approxTex : exactTex,
            exactTex,
            exactText,
            approxTex,
            approxText,
            preferDecimal: useDecimal,
            plotKind: picture ? 'curve' : null,
            visible: statement.visible,
            warn,
          })
          continue
        }
        if (plotted) {
          if (parsed.kind === 'assign') env.set(parsed.name, { kind: 'expr', expr: plotted.value })
          const color = colorFor(parsed.plot)
          const label = dep && dep.power === 1 ? dep.name : plain(equation.left)
          const computed = containsCas(equation.right)
          const formulaTex = computed || !echo ? `${tex(equation.left)} = ${tex(computed ? plotted.value : equation.right)}` : echo
          const formulaText = `${plain(equation.left)} = ${plain(computed ? plotted.value : equation.right)}`
          if (plotted.kind === 'surface') {
            const clips = surfaceRanges(parsed.plot, ['x', 'y'], env, angles)
            const drawn = surfacePlot(statement.id, label, color, statement.visible, plotted.bodies, ['x', 'y'], env, angles, parsed.plot, clips)
            const warn = plotted.warn ?? clips.warn ?? curveOnlyShade(parsed.plot) ?? (hasGeometry(drawn) ? null : emptyWarning('x', null))
            plots.push(drawn)
            rows.push(rowBase(statement.id, statement.input, label, formulaText, formulaTex, 'surface', statement.visible, warn))
            continue
          }
          const clip = rangeOf(parsed.plot, plotted.param, env, angles)
          const fill = shadeOf(parsed.plot, clip.interval, env, angles)
          const drawn = curvePlot(statement.id, label, color, statement.visible, { bodies: plotted.bodies, param: plotted.param, along: plotted.along }, env, angles, { plot: parsed.plot, clip: clip.interval, dashed: parsed.plot.dashed, shade: fill.shade })
          const warn = plotted.warn ?? clip.warn ?? fill.warn ?? (hasGeometry(drawn) ? null : emptyWarning(plotted.param, clip.interval))
          plots.push(drawn)
          rows.push(rowBase(statement.id, statement.input, label, formulaText, formulaTex, 'curve', statement.visible, warn))
          continue
        }
      }
      if (parsed.kind === 'system') {
        const local = withoutNames(env, parsed.variables ?? [])
        const solved = solveSystem(parsed.equations.map((item) => applyEnv(item, local, 0)), searchIn(parsed.domains, env), angles, parsed.variables)
        pushVisual(statement, 'Solve', solved, parsed.plot)
        continue
      }
      if (parsed.kind === 'solve') {
        const local = withoutNames(env, parsed.variable ? [parsed.variable] : [])
        const solved = solveSingle(applyEnv(parsed.equation, local, 0), parsed.variable, searchIn(parsed.domains, env), angles)
        pushVisual(statement, 'Solve', solved, parsed.plot)
        continue
      }
      if (parsed.kind === 'plot') {
        const variable = parsed.variable ?? guessNames(applyEnv(parsed.expr, env, 0), 'Plot', 1)[0] ?? 'x'
        if (parsed.plot.domain && !parsed.plot.domain.name) parsed.plot.domain = { ...parsed.plot.domain, name: variable }
        const local = withoutNames(env, [variable])
        const value = normalize(rewriteAll(applyEnv(parsed.expr, local, 0), angles), angles)
        const missing = freeSymbols(value).filter((name) => name !== variable)
        const text = `Plot(${plain(parsed.expr)}, ${variable})`
        const formula = echo ?? `\\operatorname{Plot}\\left(${tex(parsed.expr)}, ${texName(variable)}\\right)`
        const label = shortLabel(parsed.expr, 'Plot')
        const range = rangeOf(parsed.plot, variable, env, angles)
        if (value.type === 'vec') {
          if (value.args.length < 2 || value.args.length > 3) throw new MathError('Plot draws a list of two or three expressions as a curve, for example Plot([cos(t), sin(t)], t).')
          const drawn = parametricPlot(statement.id, label, colorFor(parsed.plot), statement.visible, missing.length ? [] : value.args, variable, local, angles, parsed.plot, range.interval ?? DEFAULT_RANGE, parsed.plot.dashed)
          const warn = missing.length > 0 ? giveValue(missing) : (range.warn ?? curveOnlyShade(parsed.plot) ?? (hasGeometry(drawn) ? null : emptyWarning(variable, range.interval ?? DEFAULT_RANGE)))
          plots.push(drawn)
          rows.push(rowBase(statement.id, statement.input, 'Plot', text, formula, 'curve', statement.visible, warn))
          continue
        }
        const fill = shadeOf(parsed.plot, range.interval, env, angles)
        const drawn = curvePlot(statement.id, label, colorFor(parsed.plot), statement.visible, missing.length ? null : { bodies: [value], param: variable, along: 'x' }, local, angles, { plot: parsed.plot, clip: range.interval, dashed: parsed.plot.dashed, shade: fill.shade })
        const warn = missing.length > 0 ? giveValue(missing) : (range.warn ?? fill.warn ?? (hasGeometry(drawn) ? null : emptyWarning(variable, range.interval)))
        plots.push(drawn)
        rows.push(rowBase(statement.id, statement.input, 'Plot', text, formula, 'curve', statement.visible, warn))
        continue
      }
      if (parsed.kind === 'plot3d') {
        const names = parsed.variables ?? (guessNames(applyEnv(parsed.expr, env, 0), 'Plot3D', 2) as [string, string])
        const local = withoutNames(env, names)
        const value = normalize(rewriteAll(applyEnv(parsed.expr, local, 0), angles), angles)
        const missing = freeSymbols(value).filter((name) => !names.includes(name))
        const clips = surfaceRanges(parsed.plot, names, env, angles)
        const drawn = surfacePlot(statement.id, shortLabel(parsed.expr, 'Plot3D'), colorFor(parsed.plot), statement.visible, missing.length ? [] : [value], names, local, angles, parsed.plot, clips)
        const warn = missing.length > 0 ? giveValue(missing) : (clips.warn ?? curveOnlyShade(parsed.plot) ?? (hasGeometry(drawn) ? null : emptyWarning(names[0], null)))
        const formula = echo ?? `\\operatorname{Plot3D}\\left(${tex(parsed.expr)}, ${texName(names[0])}, ${texName(names[1])}\\right)`
        plots.push(drawn)
        rows.push(rowBase(statement.id, statement.input, 'Plot3D', `Plot3D(${plain(parsed.expr)}, ${names[0]}, ${names[1]})`, formula, 'surface', statement.visible, warn))
        continue
      }
      if (parsed.kind !== 'expr') continue
      const expr = fillInferred(parsed.expr, env)
      const applied = fillInferred(applyEnv(parsed.expr, env, 0))
      const cas = evaluateCas(applied, angles)
      if (cas) {
        pushVisual(statement, casLabel(applied), cas, parsed.plot)
        continue
      }
      const value = normalize(rewriteAll(applied, angles), angles)
      const picture = vectorPicture(value)
      if (picture) {
        const color = colorFor(parsed.plot)
        const range = picture.kind === 'arrow' ? null : rangeOf(parsed.plot, picture.param, env, angles)
        const drawn = picture.kind === 'arrow' ? arrowPlot(statement.id, 'vector', color, statement.visible, picture.x, picture.y, picture.z) : parametricPlot(statement.id, picture.param, color, statement.visible, picture.components, picture.param, env, angles, parsed.plot, range?.interval ?? DEFAULT_RANGE, parsed.plot.dashed)
        plots.push(drawn)
        const shown = described(expr, value, statement.input, angles)
        const warn = picture.kind === 'arrow' ? curveOnlyShade(parsed.plot) : (range?.warn ?? curveOnlyShade(parsed.plot) ?? (hasGeometry(drawn) ? null : emptyWarning(picture.param, range?.interval ?? DEFAULT_RANGE)))
        rows.push({ ...shown, text: warn ? `${shown.text} (${warn})` : shown.text, statementId: statement.id, label: picture.kind === 'arrow' ? 'Vector' : picture.param, plotKind: 'curve', visible: statement.visible, warn })
        continue
      }
      if (containsCas(expr)) {
        const curve = singleCurve(value)
        if (curve) {
          const clip = rangeOf(parsed.plot, curve.along, env, angles)
          const fill = shadeOf(parsed.plot, clip.interval, env, angles)
          const drawn = curvePlot(statement.id, curve.label, colorFor(parsed.plot), statement.visible, { bodies: [curve.expr], param: curve.along, along: curve.along }, env, angles, { plot: parsed.plot, clip: clip.interval, dashed: parsed.plot.dashed, shade: fill.shade })
          plots.push(drawn)
          const shown = described(expr, value, statement.input, angles)
          const warn = clip.warn ?? fill.warn
          rows.push({ ...shown, text: warn ? `${shown.text} (${warn})` : shown.text, statementId: statement.id, label: curve.label, plotKind: 'curve', visible: statement.visible, warn })
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
      failed = true
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
    } finally {
      if (silent && !failed) {
        plots.splice(plotMark)
        for (const row of rows.slice(rowMark)) quietRow(row, statement.input)
      }
    }
  }

  return { rows, plots }
}

function tidy(e: Expr, angles: AngleMode): Expr {
  try {
    return normalize(e, angles)
  } catch {
    return e
  }
}

function withoutNames(env: MathEnv, names: string[]): MathEnv {
  if (!names.some((name) => env.has(name))) return env
  const local: MathEnv = new Map(env)
  for (const name of names) local.delete(name)
  return local
}

function searchIn(domains: SearchDomain[], env: MathEnv): SearchDomain[] {
  return domains.map((domain) => ({ ...domain, min: applyEnv(domain.min, env, 0), max: applyEnv(domain.max, env, 0) }))
}

function giveValue(missing: string[]): string {
  return `Give ${missing.join(', ')} a value above this line to draw the graph.`
}

function shortLabel(expr: Expr, fallback: string): string {
  const text = plain(expr)
  return text.length <= 24 ? text : fallback
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

function explicitPlot(dep: { name: 'x' | 'y' | 'z'; power: number }, rhs: Expr, env: MathEnv, angles: AngleMode): { kind: 'curve'; value: Expr; bodies: Expr[]; param: string; along: 'x' | 'y'; warn: string | null } | { kind: 'surface'; value: Expr; bodies: Expr[]; warn: string | null } | null {
  const value = normalize(rewriteAll(applyEnv(rhs, env, 0), angles), angles)
  if (freeSymbols(value).includes(dep.name)) return null
  const roots = rootBodies(value, dep.power)
  if (dep.name === 'z') {
    const unknown = freeSymbols(value).filter((name) => name !== 'x' && name !== 'y')
    if (unknown.length > 0) return { kind: 'surface', value, bodies: [], warn: giveValue(unknown) }
    return { kind: 'surface', value, bodies: roots, warn: null }
  }
  const param = dep.name === 'y' ? 'x' : 'y'
  const unknown = freeSymbols(value).filter((name) => name !== param)
  if (unknown.length > 0) return { kind: 'curve', value, bodies: [], param, along: param, warn: giveValue(unknown) }
  return { kind: 'curve', value, bodies: roots, param, along: param, warn: null }
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

interface CurveOptions {
  dashed?: boolean
  shade?: { from: number; to: number }
  along?: 'x' | 'y'
  plot?: PlotOptions
  /** Only draw this interval of the input, as a Domain setting asks. */
  clip?: Interval | null
}

function curvePlot(id: string, label: string, color: string, visible: boolean, spec: { bodies: Expr[]; param: string; along: 'x' | 'y' } | null, env: MathEnv, angles: AngleMode, options: CurveOptions = {}): CurvePlot {
  const style = options.plot ?? DEFAULT_PLOT
  const clip = options.clip ?? null
  const sample = (window: PlotWindow) => (spec && spec.bodies.length > 0 ? sampleBranches(spec.bodies, spec.param, spec.along, env, window, angles, style, clip) : [])
  return { kind: 'curve', statementId: id, label, color, visible, sample, path: sample(DEFAULT_WINDOW), dashed: options.dashed, shade: options.shade, along: options.along ?? spec?.along }
}

function parametricPlot(id: string, label: string, color: string, visible: boolean, components: Expr[], param: string, env: MathEnv, angles: AngleMode, style: PlotOptions, range: Interval, dashed?: boolean): CurvePlot {
  const sample = (window: PlotWindow) => (components.length > 0 ? sampleParametric(components, param, env, window, angles, style, range) : [])
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

function surfacePlot(id: string, label: string, color: string, visible: boolean, bodies: Expr[], names: [string, string], env: MathEnv, angles: AngleMode, plot: PlotOptions, clips: { x: Interval | null; y: Interval | null }): SurfacePlot {
  const sample = (window: PlotWindow) => (bodies.length > 0 ? bodies.map((body) => sampleSurface(body, names, env, window, angles, plot, clips)) : [])
  const sheets = sample(DEFAULT_WINDOW)
  return { kind: 'surface', statementId: id, label, color, visible, sheets, grid: sheets[0] ?? [], sample }
}

const CAS_NOUNS: Record<string, string> = { integrate: 'Integral', fmin: 'Minimum', fmax: 'Maximum', dsolve: 'Solution' }

function casLabel(expr: Expr): string {
  if (expr.type !== 'call') return 'Result'
  return CAS_NOUNS[expr.name] ?? functionByKernel(expr.name)?.name ?? expr.name
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

function sampleBranches(bodies: Expr[], param: string, along: 'x' | 'y', env: MathEnv, window: PlotWindow, angles: AngleMode, style: PlotOptions, clip: Interval | null): { x: number; y: number; z: number }[] {
  const path: { x: number; y: number; z: number }[] = []
  const min = Math.max(along === 'x' ? window.xMin : window.yMin, clip?.min ?? -Infinity)
  const max = Math.min(along === 'x' ? window.xMax : window.yMax, clip?.max ?? Infinity)
  if (max < min) return path
  const ySpan = Math.max(1e-6, along === 'x' ? window.yMax - window.yMin : window.xMax - window.xMin)
  const limit = Math.max(CLIP, 40 * Math.abs(window.xMax - window.xMin), 40 * Math.abs(window.yMax - window.yMin))
  for (const body of bodies) {
    const at = (t: number) => {
      const value = numericValue(body, bind(env, param, t), angles)
      return value === null || !Number.isFinite(value) ? null : value
    }
    const native = sampleCurveNative(body, param, env, min, max, ySpan, angles === 'deg', style)
    const nodes = native ?? refineSamples(min, max, { ...style, points: Math.min(style.points, 160), recursion: Math.min(style.recursion, 4) }, ySpan, at)
    let previous = false
    for (const node of nodes) {
      const value = node.y
      const x = along === 'x' ? node.t : value
      const y = along === 'x' ? value : node.t
      const bad = x === null || y === null || !Number.isFinite(x) || !Number.isFinite(y) || Math.abs(x) > limit || Math.abs(y) > limit
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

function readInterval(domain: PlotDomain, env: MathEnv, angles: AngleMode): Interval | null {
  const min = numericValue(domain.min, env, angles)
  const max = numericValue(domain.max, env, angles)
  if (min === null || max === null || min === max) return null
  return min < max ? { min, max } : { min: max, max: min }
}

/** The interval a setting gives one input: {t: 0..2*pi} or {Domain: 0..2*pi}. */
function rangeOf(plot: PlotOptions, param: string, env: MathEnv, angles: AngleMode, inputs: string[] = [param]): { interval: Interval | null; warn: string | null } {
  const domain = plot.ranges.find((range) => range.name === param) ?? plot.domain
  if (!domain) {
    const stray = plot.ranges.find((range) => !inputs.includes(range.name))
    return { interval: null, warn: stray ? `This graph's input is ${param}, so write {${param}: …} or {Domain: …}.` : null }
  }
  const interval = readInterval(domain, env, angles)
  return { interval, warn: interval ? null : `The domain of ${param} needs two different numbers, for example {${param}: 0..2*pi}.` }
}

const NO_SHADE: { shade: undefined; warn: null } = { shade: undefined, warn: null }

/** Shade: a..b fills that interval. Shade: true fills the drawn interval, or the whole curve. */
function shadeOf(plot: PlotOptions, clip: Interval | null, env: MathEnv, angles: AngleMode): { shade: { from: number; to: number } | undefined; warn: string | null } {
  if (!plot.shade) return NO_SHADE
  if (plot.shade === 'all') return { shade: { from: clip?.min ?? -Infinity, to: clip?.max ?? Infinity }, warn: null }
  const interval = readInterval({ name: 'shade', ...plot.shade }, env, angles)
  if (!interval) return { shade: undefined, warn: 'Shade needs two different numbers, for example {Shade: 0..2}.' }
  return { shade: { from: interval.min, to: interval.max }, warn: null }
}

function curveOnlyShade(plot: PlotOptions): string | null {
  return plot.shade ? 'Shade only fills under a curve such as y = f(x), not a parametric curve, vector, or surface.' : null
}

/** A surface can trim either input by name. A plain Domain trims both. */
function surfaceRanges(plot: PlotOptions, names: [string, string], env: MathEnv, angles: AngleMode): { x: Interval | null; y: Interval | null; warn: string | null } {
  const stray = plot.ranges.find((range) => !names.includes(range.name))
  if (stray) return { x: null, y: null, warn: `This surface's inputs are ${names[0]} and ${names[1]}, so name one of those: {${names[0]}: -2..2}.` }
  const [x, y] = names.map((name) => rangeOf(plot, name, env, angles, names))
  return { x: x?.interval ?? null, y: y?.interval ?? null, warn: x?.warn ?? y?.warn ?? null }
}

function emptyWarning(param: string, range: Interval | null): string {
  if (!range) return 'No real values to plot on [-10, 10].'
  const bound = (n: number) => String(Math.round(n * 1000) / 1000)
  return `No real values to plot for ${param} from ${bound(range.min)} to ${bound(range.max)}.`
}

function sampleParametric(components: Expr[], param: string, env: MathEnv, window: PlotWindow, angles: AngleMode, style: PlotOptions, range: Interval): { x: number; y: number; z: number }[] {
  const ySpan = Math.max(1e-6, window.yMax - window.yMin, window.xMax - window.xMin)
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

function sampleSurface(body: Expr, names: [string, string], env: MathEnv, window: PlotWindow, angles: AngleMode, plot: PlotOptions, clips: { x: Interval | null; y: Interval | null }): { x: number; y: number; z: number }[][] {
  const count = Math.max(24, Math.min(80, Math.round(plot.points / 2)))
  const grid: { x: number; y: number; z: number }[][] = []
  const zSpan = Math.max(1, window.yMax - window.yMin)
  const xMin = Math.max(window.xMin, clips.x?.min ?? -Infinity)
  const xMax = Math.min(window.xMax, clips.x?.max ?? Infinity)
  const yMin = Math.max(window.yMin, clips.y?.min ?? -Infinity)
  const yMax = Math.min(window.yMax, clips.y?.max ?? Infinity)
  if (xMax < xMin || yMax < yMin) return grid
  for (let row = 0; row < count; row += 1) {
    const y = sampleAt(row, count, yMin, yMax)
    const line: { x: number; y: number; z: number }[] = []
    for (let col = 0; col < count; col += 1) {
      const x = sampleAt(col, count, xMin, xMax)
      let local = bind(env, names[0], x)
      local = bind(local, names[1], y)
      const z = numericValue(body, local, angles)
      const previous = line[line.length - 1]
      const jump = previous && z !== null && Number.isFinite(previous.z) && Number.isFinite(z) && Math.abs(z - previous.z) > zSpan * 6
      line.push({ x, y, z: z === null || Math.abs(z) > CLIP || (plot.exclusions && jump) ? Number.NaN : z })
    }
    grid.push(line)
  }
  return grid
}

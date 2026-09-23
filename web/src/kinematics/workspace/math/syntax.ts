// The console line syntax. A call is Name(required inputs){Setting: value, ...}: parentheses
// hold what the function needs, in order, and braces after the closing parenthesis hold
// optional settings in any order. Settings at the end of a definition or an equation style its
// graph. Reading is lenient so a half-typed line still previews; binding checks every input and
// setting against the function table and hands the kernel plain positional calls.

import { GRAPH_OPTIONS, findFunction, findOption, functionByKernel, isGraphOption, optionKey, optionsFor, parseColor, suggestName, type FunctionSpec, type OptionSpec } from './functions'
import {
  CARET_TEX,
  MathError,
  blockTex,
  convertAngles,
  exprKey,
  isReservedName,
  naming,
  numericConstant,
  parseExpr,
  parseTracked,
  plain,
  plotDefaults,
  readOptionBlock,
  splitRange,
  tex,
  texName,
  type AngleMode,
  type CallExpr,
  type Expr,
  type OptionBlock,
  type OptionEntry,
  type PlotDomain,
  type PlotOptions,
  type SearchDomain,
} from './expr'
import { MATH_CARET, latexToSource } from './inputView'

const IDENT = '[A-Za-zαβγδεζηθικλμνξπρστυφχψω][A-Za-z0-9αβγδεζηθικλμνξπρστυφχψω]*'
const VARIABLE_KEY = /^(?:[A-Za-z][A-Za-z0-9]?|alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|lambda|mu|nu|xi|rho|sigma|tau|upsilon|phi|chi|psi|omega|[αβγδεζηθικλμνξρστυφχψω])$/

export type MathInput =
  | { kind: 'fn'; name: string; params: string[]; body: Expr; raw: string; plot: PlotOptions }
  | { kind: 'assign'; name: string; expr: Expr; raw: string; plot: PlotOptions }
  | { kind: 'solve'; equation: Expr; variable: string | null; domains: SearchDomain[]; raw: string; plot: PlotOptions }
  | { kind: 'system'; equations: Expr[]; variables: string[] | null; domains: SearchDomain[]; raw: string; plot: PlotOptions }
  | { kind: 'plot'; expr: Expr; variable: string; raw: string; plot: PlotOptions }
  | { kind: 'plot3d'; expr: Expr; variables: [string, string]; raw: string; plot: PlotOptions }
  | { kind: 'expr'; expr: Expr; raw: string; plot: PlotOptions }

interface LineRead {
  kind: 'fn' | 'assign' | 'expr'
  name: string
  params: string[]
  /** The left side of an assignment, kept for the preview. */
  left: Expr | null
  expr: Expr
  /** The settings block at the very end of the line. */
  block: OptionBlock | null
  /** The call that block directly follows, if it follows one. */
  tail: CallExpr | null
  caretAfter: boolean
  legacy: PlotOptions
  raw: string
}

export function parseMathInput(input: string): MathInput {
  return bindLine(readLine(input))
}

export function validateMath(input: string): string | null {
  try {
    parseMathInput(input)
    return null
  } catch (error) {
    return error instanceof MathError ? error.message : 'Could not read that.'
  }
}

function readLine(input: string): LineRead {
  const closed = closeGroups(latexToSource(input).trim())
  const split = splitTrailingBlock(closed)
  const { source, plot } = takePlotOptions(split.source)
  const raw = source.trim()
  if (!raw.replaceAll(MATH_CARET, '').trim()) throw new MathError(split.block ? 'Write the calculation before the settings in { }.' : 'Enter a calculation.')
  const base = { block: split.block, caretAfter: split.caretAfter, legacy: plot, raw, left: null }
  const fn2 = new RegExp(`^(${IDENT})\\s*\\(\\s*(${IDENT})\\s*,\\s*(${IDENT})\\s*\\)\\s*=\\s*([\\s\\S]+)$`).exec(raw)
  if (fn2) {
    const name = naming(fn2[1] ?? '')
    const params = [naming(fn2[2] ?? ''), naming(fn2[3] ?? '')]
    assertDefinable(name)
    if (params[0] === params[1]) throw new MathError('Use two different inputs.')
    const body = parseTracked(fn2[4] ?? '')
    return { ...base, kind: 'fn', name, params, expr: body.expr, tail: body.tail }
  }
  const fn1 = new RegExp(`^(${IDENT})\\s*\\(\\s*(${IDENT})\\s*\\)\\s*=\\s*([\\s\\S]+)$`).exec(raw)
  if (fn1) {
    const name = naming(fn1[1] ?? '')
    assertDefinable(name)
    const body = parseTracked(fn1[3] ?? '')
    return { ...base, kind: 'fn', name, params: [naming(fn1[2] ?? '')], expr: body.expr, tail: body.tail }
  }
  const read = parseTracked(raw)
  const expr = read.expr
  if (expr.type === 'eq' && expr.left.type === 'call') throw new MathError('Define a curve as f(x) = … or a surface as f(x, y) = ….')
  if (expr.type === 'eq' && expr.left.type === 'sym') {
    if (findFunction(expr.left.name)) throw new MathError(`${expr.left.name} is a built-in function. Pick another name.`)
    if (!isReservedName(expr.left.name)) return { ...base, kind: 'assign', name: expr.left.name, params: [], left: expr.left, expr: expr.right, tail: read.tail }
  }
  return { ...base, kind: 'expr', name: '', params: [], expr, tail: read.tail }
}

function assertDefinable(name: string): void {
  const spec = findFunction(name)
  if (spec) throw new MathError(spec.name.toLowerCase() === name.toLowerCase() ? `${name} is built in.` : `${name} is built in (it means ${spec.name}).`)
  if (isReservedName(name)) throw new MathError(`${name} is built in.`)
}

/** Close brackets that are still open, innermost first, so Enter on `Solve(x^2 = 4, x` still runs. */
function closeGroups(input: string): string {
  const stack: string[] = []
  const closer: Record<string, string> = { '(': ')', '[': ']', '{': '}' }
  const opener: Record<string, string> = { ')': '(', ']': '[', '}': '{' }
  for (const ch of input) {
    if (ch === '(' || ch === '[' || ch === '{') stack.push(ch)
    else if (ch === ')' || ch === ']' || ch === '}') {
      if (stack[stack.length - 1] !== opener[ch]) return input
      stack.pop()
    }
  }
  let out = input
  for (let i = stack.length - 1; i >= 0; i -= 1) out += closer[stack[i] ?? ''] ?? ''
  return out
}

function matchOpenBrace(text: string, close: number): number {
  let depth = 0
  for (let i = close; i >= 0; i -= 1) {
    if (text[i] === '}') depth += 1
    else if (text[i] === '{') {
      depth -= 1
      if (depth === 0) return i
    }
  }
  return -1
}

/**
 * Take the `{…}` at the end of the line off before anything else reads it. A brace right after
 * `_` or `^` is a subscript or an exponent, not settings. A live caret on either side of the
 * block is remembered so the preview can draw it there.
 */
function splitTrailingBlock(text: string): { source: string; block: OptionBlock | null; caretAfter: boolean } {
  const none = { source: text, block: null, caretAfter: false }
  let end = text.length
  while (end > 0 && /\s/.test(text[end - 1] ?? '')) end -= 1
  let caretAfter = false
  if (text[end - 1] === MATH_CARET) {
    caretAfter = true
    end -= 1
    while (end > 0 && /\s/.test(text[end - 1] ?? '')) end -= 1
  }
  if (text[end - 1] !== '}') return none
  const open = matchOpenBrace(text, end - 1)
  if (open < 0) return none
  let before = open - 1
  while (before >= 0 && (/\s/.test(text[before] ?? '') || text[before] === MATH_CARET)) before -= 1
  if (before >= 0 && (text[before] === '_' || text[before] === '^')) return none
  const block = readOptionBlock(text.slice(open + 1, end - 1))
  if (text.slice(before + 1, open).includes(MATH_CARET)) block.caretBefore = true
  return { source: text.slice(0, before + 1), block, caretAfter }
}

/**
 * The older comma tails, `, plotpoints = 20` and `, t = 0..2*pi`, still work. A live caret marker
 * can land inside or right after such a tail while it is being typed; it is stripped before
 * matching and put back only if it belonged to the main expression.
 */
function takePlotOptions(source: string): { source: string; plot: PlotOptions } {
  const caretAt = source.indexOf(MATH_CARET)
  const plain = caretAt < 0 ? source : source.slice(0, caretAt) + source.slice(caretAt + 1)
  let rest = plain.trim()
  const plot = plotDefaults()
  const pattern = /,\s*(plotpoints|maxrecursion|exclusions)\s*=\s*([^\s,]+)\s*$/i
  const domainPattern = new RegExp(`(?:,|\\n)\\s*(domain|${IDENT})\\s*=\\s*([^\\s,]+)\\s*\\.\\.\\s*([^\\s,]+)\\s*$`, 'i')
  const parenDomain = new RegExp(`(?:,|\\n)\\s*(domain|${IDENT})\\s*=\\s*\\(([^,]+),\\s*([^)]+)\\)\\s*$`, 'i')
  for (let n = 0; n < 8; n += 1) {
    const domain = domainPattern.exec(rest) ?? parenDomain.exec(rest)
    const domainName = (domain?.[1] ?? '').toLowerCase()
    if (domain && domainName !== 'plotpoints' && domainName !== 'maxrecursion' && domainName !== 'exclusions') {
      if (!plot.domain) {
        try {
          plot.domain = { name: domain[1] ?? 't', min: parseExpr((domain[2] ?? '').trim()), max: parseExpr((domain[3] ?? '').trim()) }
        } catch {
          throw new MathError('Use a domain such as {t: 0..2*pi}.')
        }
      }
      rest = rest.slice(0, domain.index).trim()
      continue
    }
    const match = pattern.exec(rest)
    if (!match) break
    const key = (match[1] ?? '').toLowerCase()
    const raw = match[2] ?? ''
    if (key === 'exclusions') {
      const flag = raw.toLowerCase()
      plot.exclusions = flag !== 'false' && flag !== '0' && flag !== 'off' && flag !== 'none'
    } else {
      const value = Number(raw)
      if (!Number.isFinite(value)) throw new MathError(key === 'plotpoints' ? 'PlotPoints needs a number of samples.' : 'MaxRecursion needs a whole number.')
      if (key === 'plotpoints') plot.points = Math.max(12, Math.min(800, Math.round(value)))
      else plot.recursion = Math.max(0, Math.min(8, Math.round(value)))
    }
    rest = rest.slice(0, match.index).trim()
  }
  if (caretAt >= 0 && caretAt <= rest.length) rest = `${rest.slice(0, caretAt)}${MATH_CARET}${rest.slice(caretAt)}`
  return { source: rest, plot }
}

interface BindContext {
  /** The call the whole line, or the right side of its definition, consists of. */
  main: CallExpr | null
  /** Graph settings given to the main call, which style the line's graph. */
  graph: OptionEntry[]
}

interface Settings {
  values: Map<string, OptionEntry>
  ranges: { name: string; entry: OptionEntry }[]
  graph: OptionEntry[]
}

function realEntry(entry: OptionEntry): boolean {
  return Boolean(entry.key) || entry.value !== null
}

function bindLine(line: LineRead): MathInput {
  const plot: PlotOptions = { ...line.legacy, ranges: [...line.legacy.ranges] }
  const main = line.expr.type === 'call' ? line.expr : null
  const draws = line.kind !== 'expr' || line.expr.type === 'eq'
  const tail = line.tail
  const tailSpec = tail ? functionByKernel(tail.name) : null
  const lineEntries: OptionEntry[] = []
  if (line.block) {
    const own: OptionEntry[] = []
    for (const entry of line.block.entries) {
      if (!realEntry(entry)) continue
      if (!entry.key) throw new MathError(`Give each setting a name, for example {Domain: ${entry.value}}.`)
      const styleOnly = draws && isGraphOption(entry.key) && optionKey(entry.key) !== 'domain'
      if (tail && tailSpec && !styleOnly && accepts(tailSpec, tail, entry)) own.push(entry)
      else lineEntries.push(entry)
    }
    if (tail && own.length > 0) tail.options = { raw: '', entries: [...(tail.options?.entries ?? []), ...own] }
  }
  const lineOwner = { spec: tailSpec, bare: line.kind === 'expr' && main !== null && main === tail, expr: line.expr }
  const context: BindContext = { main, graph: [] }
  if (line.kind === 'expr' && main && (main.name === 'solve' || main.name === 'plot' || main.name === 'plot3d')) {
    const bound = main.name === 'solve' ? bindSolve(main, line, plot, context) : bindPlot(main, line, plot, context)
    for (const entry of lineEntries) applyLineSetting(plot, entry, lineOwner)
    return bound
  }
  const expr = bindExpr(line.expr, context)
  for (const entry of context.graph) applyGraphSetting(plot, entry)
  for (const entry of lineEntries) applyLineSetting(plot, entry, lineOwner)
  if (line.kind === 'fn') return { kind: 'fn', name: line.name, params: line.params, body: expr, raw: line.raw, plot }
  if (line.kind === 'assign') return { kind: 'assign', name: line.name, expr, raw: line.raw, plot }
  return { kind: 'expr', expr, raw: line.raw, plot }
}

function variablesOf(spec: FunctionSpec, call: CallExpr): string[] {
  const names: string[] = []
  spec.params.forEach((param, index) => {
    if (param.kind !== 'variable') return
    const arg = call.args[index]
    if (arg?.type === 'sym') names.push(arg.name)
    else if (arg?.type === 'vec') arg.args.forEach((item) => item.type === 'sym' && names.push(item.name))
  })
  return names
}

/** `{x: 0..5}` names one of the call's variables, for a function that takes a domain. */
function rangeTarget(spec: FunctionSpec, call: CallExpr, entry: OptionEntry): string | null {
  const takesDomain = spec.options.some((option) => optionKey(option.name) === 'domain')
  if (!spec.variableRanges && !takesDomain) return null
  if (entry.value === null || !splitRange(entry.value)) return null
  const key = entry.key.trim()
  if (!VARIABLE_KEY.test(key)) return null
  const name = naming(key)
  const names = variablesOf(spec, call)
  if (names.includes(name)) return name
  return spec.variableRanges && names.length === 0 ? name : null
}

function accepts(spec: FunctionSpec, call: CallExpr, entry: OptionEntry): boolean {
  return findOption(optionsFor(spec), entry.key) !== null || rangeTarget(spec, call, entry) !== null
}

function bindExpr(e: Expr, context: BindContext): Expr {
  const visit = (child: Expr) => bindExpr(child, context)
  switch (e.type) {
    case 'call':
      return bindCall(e, context)
    case 'add':
    case 'mul':
      return { ...e, args: e.args.map(visit) }
    case 'div':
      return { type: 'div', num: visit(e.num), den: visit(e.den) }
    case 'pow':
      return { type: 'pow', base: visit(e.base), exp: visit(e.exp) }
    case 'vec':
      return { type: 'vec', args: e.args.map(visit) }
    case 'mat':
      return { type: 'mat', rows: e.rows.map((row) => row.map(visit)) }
    case 'eq':
      return { type: 'eq', left: visit(e.left), right: visit(e.right) }
    case 'group':
      return { type: 'group', body: visit(e.body) }
    case 'caret':
      return e.body ? { ...e, body: visit(e.body) } : e
    default:
      return e
  }
}

function bindCall(e: CallExpr, context: BindContext): Expr {
  const args = e.args.map((arg) => bindExpr(arg, context))
  const spec = functionByKernel(e.name)
  if (!spec) {
    if (e.options?.entries.some(realEntry)) throw new MathError(`${e.name} has no settings.`)
    if (e.name === 'Dt' && args.length !== 1) throw new MathError('Use Dt(expression) for a time derivative.')
    return { type: 'call', name: e.name, args }
  }
  if (spec.kernel === 'solve' || spec.kernel === 'plot' || spec.kernel === 'plot3d') throw new MathError(`${spec.name} goes on a line of its own, for example ${spec.examples[0]}`)
  const settings = readSettings(spec, e)
  if (settings.graph.length > 0) {
    if (e !== context.main) throw new MathError(`Graph settings such as ${settings.graph[0]?.key.trim()} go at the end of the line, on the function the line draws.`)
    context.graph.push(...settings.graph)
  }
  return kernelCall(spec, args, settings)
}

function readSettings(spec: FunctionSpec, call: CallExpr): Settings {
  const settings: Settings = { values: new Map(), ranges: [], graph: [] }
  for (const entry of call.options?.entries ?? []) {
    if (!realEntry(entry)) continue
    if (!entry.key) throw new MathError(`Give each setting a name, for example {${spec.options[0]?.name ?? 'Color'}: ${entry.value}}.`)
    const own = findOption(spec.options, entry.key)
    if (own) {
      if (settings.values.has(own.name)) throw new MathError(`${own.name} is set twice.`)
      settings.values.set(own.name, entry)
      continue
    }
    if (spec.draws && isGraphOption(entry.key)) {
      settings.graph.push(entry)
      continue
    }
    const target = rangeTarget(spec, call, entry)
    if (target) {
      settings.ranges.push({ name: target, entry })
      continue
    }
    throw unknownSetting(spec.name, entry.key, optionsFor(spec), spec.section === 'basic')
  }
  return settings
}

function listWords(words: string[]): string {
  if (words.length <= 1) return words[0] ?? ''
  if (words.length === 2) return `${words[0]} and ${words[1]}`
  return `${words.slice(0, -1).join(', ')}, and ${words[words.length - 1]}`
}

function unknownSetting(owner: string, key: string, options: OptionSpec[], plotHint = false, pointer: string | null = null): MathError {
  const typed = key.trim()
  const head = options.length === 0 ? `${owner} has no settings.` : `${owner} has no setting ${typed}.`
  if (pointer) return new MathError(`${head} ${pointer}`)
  if (options.length === 0) {
    const hint = plotHint && isGraphOption(typed) ? ` To draw it, use Plot(${owner}(x), x){${typed}: …}.` : ''
    return new MathError(`${head}${hint}`)
  }
  const names = options.map((option) => option.name)
  const guess = suggestName(typed, names)
  return new MathError(`${head} ${guess ? `Did you mean ${guess}?` : `It takes ${listWords(names)}.`}`)
}

function childrenOf(e: Expr): Expr[] {
  switch (e.type) {
    case 'call':
    case 'add':
    case 'mul':
    case 'vec':
      return e.args
    case 'div':
      return [e.num, e.den]
    case 'pow':
      return [e.base, e.exp]
    case 'mat':
      return e.rows.flat()
    case 'eq':
      return [e.left, e.right]
    case 'group':
      return [e.body]
    case 'caret':
      return e.body ? [e.body] : []
    default:
      return []
  }
}

function callTaking(e: Expr, key: string): FunctionSpec | null {
  const spec = e.type === 'call' ? functionByKernel(e.name) : null
  if (spec && findOption(spec.options, key)) return spec
  for (const child of childrenOf(e)) {
    const found = callTaking(child, key)
    if (found) return found
  }
  return null
}

/** A block at the end of the line belongs to the outer call, so `Expand(Derivative(x^3, x)){Order: 2}` gets pointed at Derivative. */
function innerOwner(expr: Expr, key: string): string | null {
  const inner = callTaking(expr, key)
  const option = inner ? findOption(inner.options, key) : null
  if (!inner || !option) return null
  const sample = inner.examples.find((example) => example.includes(`${option.name}:`)) ?? `${inner.name}(…){${option.name}: ${option.example}}`
  return `${option.name} belongs to ${inner.name}, so put it right after ${inner.name}(…), as in ${sample}.`
}

const ORDINALS = ['first', 'second', 'third', 'fourth', 'fifth']

function arityError(spec: FunctionSpec, count: number): MathError {
  const phrases = listWords(spec.params.map((param) => param.phrase))
  if (count < spec.params.length) return new MathError(`${spec.name} needs ${phrases}: ${spec.examples[0]}`)
  const withSettings = spec.examples.find((example) => example.includes('{'))
  if (withSettings) return new MathError(`${spec.name} takes ${phrases}. Other settings go in { } after the ), as in ${withSettings}`)
  return new MathError(`${spec.name} takes ${phrases}, for example ${spec.examples[0]}`)
}

function checkArity(spec: FunctionSpec, args: Expr[], legacy: number[] = []): void {
  const need = spec.params.length
  const repeats = spec.params[need - 1]?.repeats === true
  if (repeats ? args.length >= need : args.length === need || legacy.includes(args.length)) return
  throw arityError(spec, args.length)
}

function isVariable(arg: Expr | undefined): arg is Extract<Expr, { type: 'sym' }> {
  return Boolean(arg && arg.type === 'sym' && !arg.sub && !((arg.dots ?? 0) > 0) && arg.name !== 'pi' && arg.name !== 'e' && arg.name !== '?')
}

function requireVariable(spec: FunctionSpec, args: Expr[], index: number): string {
  const arg = args[index]
  if (isVariable(arg)) return arg.name
  const param = spec.params[index]
  throw new MathError(`${spec.name} needs ${param?.phrase ?? 'a variable'} as its ${ORDINALS[index] ?? 'next'} input, for example ${spec.examples[0]}`)
}

function readRange(name: string, entry: OptionEntry, example: string): { min: Expr; max: Expr } {
  const text = entry.value ?? ''
  const range = splitRange(text)
  const paren = /^\(\s*([^,]+),\s*([^)]+)\)$/.exec(text.trim())
  const sides = range ?? (paren ? { min: paren[1] ?? '', max: paren[2] ?? '' } : null)
  if (!sides || !sides.min.trim() || !sides.max.trim()) throw new MathError(`${name} needs a range such as ${example}.`)
  try {
    return { min: parseExpr(sides.min), max: parseExpr(sides.max) }
  } catch {
    throw new MathError(`${name} needs a range such as ${example}.`)
  }
}

function readWhole(option: OptionSpec, entry: OptionEntry): number {
  const low = option.min ?? 0
  const message = option.max === undefined ? `${option.name} needs a whole number of at least ${low}.` : `${option.name} needs a whole number from ${low} to ${option.max}.`
  if (entry.value === null) throw new MathError(message)
  let value: number | null = null
  try {
    value = numericConstant(parseExpr(entry.value))
  } catch {
    value = null
  }
  if (value === null || !Number.isInteger(value) || value < low || (option.max !== undefined && value > option.max)) throw new MathError(message)
  return value
}

function readBoolean(option: OptionSpec, entry: OptionEntry): boolean {
  if (entry.value === null) return true
  const word = entry.value.trim().toLowerCase()
  if (['true', 'yes', 'on', '1'].includes(word)) return true
  if (['false', 'no', 'off', '0', 'none'].includes(word)) return false
  throw new MathError(`${option.name} is true or false, for example {${option.name}: false}.`)
}

function readColor(entry: OptionEntry): string {
  const color = entry.value === null ? null : parseColor(entry.value)
  if (!color) throw new MathError('Color needs a name such as red, blue, or green, or a hex code such as #ff8800.')
  return color
}

function readValue(option: OptionSpec, entry: OptionEntry): Expr {
  if (entry.value === null || !entry.value.trim()) throw new MathError(`${option.name} needs a value, for example {${option.name}: ${option.example}}.`)
  try {
    return parseExpr(entry.value)
  } catch {
    throw new MathError(`${option.name} needs a value, for example {${option.name}: ${option.example}}.`)
  }
}

function whole(n: number): Expr {
  return { type: 'rat', n: BigInt(n), d: 1n }
}

/** `Limit(f, x = 0)` is `Limit(f, x, 0)`. */
function pointSugar(args: Expr[]): Expr[] {
  const last = args[1]
  if (args.length === 2 && last?.type === 'eq' && last.left.type === 'sym') return [args[0] as Expr, last.left, last.right]
  return args
}

function kernelCall(spec: FunctionSpec, args: Expr[], settings: Settings): Expr {
  const call = (list: Expr[]): Expr => ({ type: 'call', name: spec.kernel, args: list })
  const option = (name: string) => findOption(spec.options, name) as OptionSpec
  const entry = (name: string) => settings.values.get(name) ?? null
  const domain = (): { min: Expr; max: Expr } | null => {
    const given = entry('Domain')
    if (given) return readRange('Domain', given, option('Domain').example)
    const named = settings.ranges[0]
    return named ? readRange(named.name, named.entry, '0..5') : null
  }
  const positional = (count: number) => {
    if (args.length > count && (settings.values.size > 0 || settings.ranges.length > 0)) throw arityError(spec, args.length)
  }
  switch (spec.kernel) {
    case 'diff': {
      positional(2)
      checkArity(spec, args, [3, 4])
      requireVariable(spec, args, 1)
      const order = entry('Order')
      const at = entry('At')
      if (!order && !at) return call(args)
      const count = whole(order ? readWhole(option('Order'), order) : 1)
      return call(at ? [args[0] as Expr, args[1] as Expr, count, readValue(option('At'), at)] : [args[0] as Expr, args[1] as Expr, count])
    }
    case 'integrate': {
      positional(2)
      checkArity(spec, args, [4])
      requireVariable(spec, args, 1)
      const range = domain()
      return call(range ? [args[0] as Expr, args[1] as Expr, range.min, range.max] : args)
    }
    case 'zeros': {
      positional(2)
      checkArity(spec, args, [1])
      if (args.length > 1) requireVariable(spec, args, 1)
      const range = domain()
      if (range && args.length < 2) throw arityError(spec, args.length)
      return call(range ? [args[0] as Expr, args[1] as Expr, range.min, range.max] : args)
    }
    case 'fmin':
    case 'fmax': {
      positional(2)
      checkArity(spec, args, [4])
      requireVariable(spec, args, 1)
      if (args.length === 4) return call(args)
      const range = domain() ?? { min: whole(-10), max: whole(10) }
      return call([args[0] as Expr, args[1] as Expr, range.min, range.max])
    }
    case 'series': {
      positional(2)
      checkArity(spec, args, [4])
      requireVariable(spec, args, 1)
      if (args.length === 4) return call(args)
      const at = entry('Point')
      const order = entry('Order')
      return call([args[0] as Expr, args[1] as Expr, at ? readValue(option('Point'), at) : whole(0), whole(order ? readWhole(option('Order'), order) : 3)])
    }
    case 'decimal': {
      checkArity(spec, args)
      const digits = entry('Digits')
      return call(digits ? [args[0] as Expr, whole(readWhole(option('Digits'), digits))] : args)
    }
    case 'log': {
      positional(1)
      checkArity(spec, args, [2])
      const base = entry('Base')
      return call(base ? [args[0] as Expr, readValue(option('Base'), base)] : args)
    }
    case 'limit':
    case 'tangent':
    case 'normal': {
      const list = pointSugar(args)
      checkArity(spec, list)
      requireVariable(spec, list, 1)
      return call(list)
    }
    default:
      checkArity(spec, args)
      spec.params.forEach((param, index) => {
        if (param.kind === 'variable') requireVariable(spec, args, index)
      })
      return call(args)
  }
}

function searchDomains(settings: Settings): SearchDomain[] {
  const domains: SearchDomain[] = []
  const shared = settings.values.get('Domain')
  if (shared) domains.push({ name: null, ...readRange('Domain', shared, '0..2*pi') })
  for (const named of settings.ranges) domains.push({ name: named.name, ...readRange(named.name, named.entry, '0..5') })
  return domains
}

function bindSolve(call: CallExpr, line: LineRead, plot: PlotOptions, context: BindContext): MathInput {
  const spec = functionByKernel('solve') as FunctionSpec
  const args = call.args.map((arg) => bindExpr(arg, context))
  const domains = searchDomains(readSettings(spec, call))
  const first = args[0]
  const second = args[1]
  if (args.length === 2 && first?.type === 'vec') {
    const variables = second?.type === 'vec' ? second.args : [second]
    const names = variables.map((item) => {
      if (!isVariable(item)) throw new MathError('Solve needs a list of variables as its second input, for example Solve([x + y = 3, x - y = 1], [x, y])')
      return item.name
    })
    return { kind: 'system', equations: first.args, variables: names, domains, raw: line.raw, plot }
  }
  if (args.length >= 2 && first?.type === 'eq' && second?.type === 'eq') {
    let index = 0
    const equations: Expr[] = []
    while (index < args.length && args[index]?.type === 'eq') {
      equations.push(args[index] as Expr)
      index += 1
    }
    return { kind: 'system', equations, variables: null, domains: [...legacyDomains(args.slice(index)), ...domains], raw: line.raw, plot }
  }
  if (args.length === 1 && first) return { kind: 'solve', equation: first, variable: null, domains, raw: line.raw, plot }
  if (args.length === 2 && first) {
    if (second?.type === 'vec') throw new MathError('Solve one equation for one variable, or a list of equations for a list of variables: Solve([x + y = 3, x - y = 1], [x, y])')
    return { kind: 'solve', equation: first, variable: requireVariable(spec, args, 1), domains, raw: line.raw, plot }
  }
  throw arityError(spec, args.length)
}

function legacyDomains(args: Expr[]): SearchDomain[] {
  const zero: Expr = { type: 'rat', n: 0n, d: 1n }
  const domains: SearchDomain[] = []
  let index = 0
  while (index < args.length) {
    const arg = args[index]
    if (arg?.type === 'vec' && arg.args.length === 2) {
      domains.push({ name: null, min: arg.args[0] ?? zero, max: arg.args[1] ?? zero })
      index += 1
      continue
    }
    if (arg?.type === 'vec' && arg.args.length === 3 && arg.args[0]?.type === 'sym') {
      domains.push({ name: arg.args[0].name, min: arg.args[1] ?? zero, max: arg.args[2] ?? zero })
      index += 1
      continue
    }
    if (arg?.type === 'sym' && args[index + 1] && args[index + 2]) {
      domains.push({ name: arg.name, min: args[index + 1] ?? zero, max: args[index + 2] ?? zero })
      index += 3
      continue
    }
    throw new MathError('Give the search range as a setting: Solve([eq1, eq2], [x, y]){Domain: 0..2*pi}')
  }
  return domains
}

function bindPlot(call: CallExpr, line: LineRead, plot: PlotOptions, context: BindContext): MathInput {
  const spec = functionByKernel(call.name) as FunctionSpec
  const args = call.args.map((arg) => bindExpr(arg, context))
  const settings = readSettings(spec, call)
  checkArity(spec, args)
  for (const entry of settings.graph) applyGraphSetting(plot, entry)
  if (spec.kernel === 'plot') {
    const variable = requireVariable(spec, args, 1)
    const own = settings.values.get('Domain')
    if (own) plot.domain = { name: variable, ...readRange('Domain', own, '0..2*pi') }
    for (const named of settings.ranges) plot.domain = { name: variable, ...readRange(named.name, named.entry, '0..2*pi') }
    return { kind: 'plot', expr: args[0] as Expr, variable, raw: line.raw, plot }
  }
  const x = requireVariable(spec, args, 1)
  const y = requireVariable(spec, args, 2)
  if (x === y) throw new MathError('Plot3D needs two different variables, for example Plot3D(x^2 - y^2, x, y)')
  for (const named of settings.ranges) {
    if (named.name !== x && named.name !== y) throw new MathError(`Plot3D draws over ${x} and ${y}, so name one of those: {${x}: -2..2}.`)
    plot.ranges.push({ name: named.name, ...readRange(named.name, named.entry, '-2..2') })
  }
  return { kind: 'plot3d', expr: args[0] as Expr, variables: [x, y], raw: line.raw, plot }
}

function applyGraphSetting(plot: PlotOptions, entry: OptionEntry): void {
  const option = findOption(GRAPH_OPTIONS, entry.key)
  if (!option) return
  switch (optionKey(option.name)) {
    case 'color':
      plot.color = readColor(entry)
      break
    case 'plotpoints':
      plot.points = readWhole(option, entry)
      break
    case 'maxrecursion':
      plot.recursion = readWhole(option, entry)
      break
    case 'exclusions':
      plot.exclusions = readBoolean(option, entry)
      break
    case 'dashed':
      plot.dashed = readBoolean(option, entry)
      break
    case 'domain':
      plot.domain = { name: 'domain', ...readRange('Domain', entry, option.example) }
      break
  }
}

/** A setting at the end of the line that the line's function did not take styles the graph. */
function applyLineSetting(plot: PlotOptions, entry: OptionEntry, owner: { spec: FunctionSpec | null; bare: boolean; expr: Expr }): void {
  const spec = owner.spec
  if (owner.bare && spec && !spec.draws) throw unknownSetting(spec.name, entry.key, optionsFor(spec), spec.section === 'basic', innerOwner(owner.expr, entry.key))
  if (isGraphOption(entry.key)) {
    applyGraphSetting(plot, entry)
    return
  }
  const key = entry.key.trim()
  if (entry.value !== null && splitRange(entry.value) && VARIABLE_KEY.test(key)) {
    plot.ranges.push({ name: naming(key), ...readRange(key, entry, '0..2*pi') })
    return
  }
  const pointer = innerOwner(owner.expr, key)
  const owned = spec ? optionsFor(spec) : []
  if (spec && owned.length > 0) throw unknownSetting(spec.name, key, [...owned, ...GRAPH_OPTIONS.filter((option) => !findOption(owned, option.name))], false, pointer)
  throw unknownSetting('This line', key, GRAPH_OPTIONS, false, pointer)
}

/** Turn a half-typed line into something the preview can draw: `x^` and `sqrt(` become placeholders. */
function cookPreview(input: string): string {
  let source = closeGroups(input.trim())
  source = source.replace(/\(\s*\)/g, '(?)')
  source = source.replace(/\[\s*\]/g, '[?]')
  source = source.replace(/([_^])\{\s*\}/g, '$1{?}')
  source = source.replace(/([([])\s*,/g, '$1?,')
  source = source.replace(/,\s*(?=[)\]])/g, ', ?')
  const held = source.endsWith(MATH_CARET)
  if (!held && /[+\-*/^,]$/.test(source)) source += '?'
  else if (!held && /_$/.test(source)) source += '?'
  else if (!held && /=\s*$/.test(source)) source += '?'
  return source
}

const IDENT_PART = /[A-Za-z0-9αβγδεζηθικλμνξπρστυφχψω]/
const IDENT_START = /[A-Za-zαβγδεζηθικλμνξπρστυφχψω]/

/** A caret sitting inside sigma still draws one symbol, with the bar after it. */
function snapIdentCaret(input: string, cursor: number): number {
  if (cursor <= 0 || cursor >= input.length) return cursor
  const prev = input[cursor - 1] ?? ''
  const next = input[cursor] ?? ''
  if (!IDENT_PART.test(prev) || !IDENT_PART.test(next)) return cursor
  if (insideBraces(input, cursor)) return cursor
  let start = cursor
  while (start > 0 && IDENT_PART.test(input[start - 1] ?? '')) start -= 1
  if (!IDENT_START.test(input[start] ?? '')) return cursor
  let end = cursor
  while (end < input.length && IDENT_PART.test(input[end] ?? '')) end += 1
  return end
}

function insideBraces(input: string, cursor: number): boolean {
  let depth = 0
  for (let i = 0; i < cursor; i += 1) {
    if (input[i] === '{') depth += 1
    else if (input[i] === '}') depth = Math.max(0, depth - 1)
  }
  return depth > 0
}

function domainClauseTex(input: string, cursor?: number): string | null {
  const match = new RegExp(`^\\s*(domain|${IDENT})\\s*=\\s*([^\\s,]+)\\s*\\.\\.\\s*([^\\s,]+)\\s*$`, 'i').exec(input)
  if (!match) return null
  try {
    const min = parseExpr((match[2] ?? '').trim())
    const max = parseExpr((match[3] ?? '').trim())
    const name = (match[1] ?? '').toLowerCase() === 'domain' ? '' : `${texName(match[1] ?? 't')} = `
    const body = `${name}${tex(min)} \\ldots ${tex(max)}`
    return cursor === undefined ? body : `${body}${CARET_TEX}`
  } catch {
    return null
  }
}

/** The typed line as KaTeX, with a caret bar where the cursor is. Null when it cannot be drawn yet. */
export function previewTex(input: string, cursor?: number): string | null {
  const clause = domainClauseTex(input, cursor)
  if (clause) return clause
  try {
    const at = cursor === undefined ? null : snapIdentCaret(input, Math.max(0, Math.min(cursor, input.length)))
    const marked = at === null ? input : `${input.slice(0, at)}${MATH_CARET}${input.slice(at)}`
    const line = readLine(cookPreview(marked))
    let body = lineTex(line)
    const domain = line.legacy.domain
    if (domain) body = `${body},\\ ${domain.name.toLowerCase() === 'domain' ? '' : `${texName(domain.name)} = `}${tex(domain.min)} \\ldots ${tex(domain.max)}`
    if (line.block) body = `${body}${blockTex(line.block, line.caretAfter)}`
    if (cursor !== undefined && !body.includes('\\rule')) body = `${body}${CARET_TEX}`
    return body
  } catch {
    return null
  }
}

function lineTex(line: LineRead): string {
  if (line.kind === 'fn') return `${texName(line.name)}\\left(${line.params.map((name) => texName(name)).join(', ')}\\right) = ${tex(line.expr)}`
  if (line.kind === 'assign') return `${tex(line.left ?? { type: 'sym', name: line.name })} = ${tex(line.expr)}`
  return tex(line.expr)
}

/**
 * Switching radians and degrees rewrites constant trig inputs so the values stay put.
 * `sin(pi)` becomes `sin(180)`, and `sin(180)` becomes `sin(pi)`. A free variable such as
 * `sin(x)` is left alone, because that variable is read in the unit you just picked.
 * Settings blocks and the older comma tails are kept as typed.
 */
export function convertAngleInput(input: string, from: AngleMode, to: AngleMode): string {
  if (from === to) return input
  try {
    const line = readLine(input)
    const converted = convertAngles(line.expr, from, to)
    if (exprKey(converted) === exprKey(line.expr)) return input
    const head = line.kind === 'fn' ? `${line.name}(${line.params.join(', ')}) = ` : line.kind === 'assign' ? `${plain(line.left ?? { type: 'sym', name: line.name })} = ` : ''
    const block = line.block ? `{${line.block.raw}}` : ''
    const next = `${head}${plain(converted)}${plotSuffix(line.legacy)}${block}`
    parseMathInput(next)
    return next
  } catch {
    return input
  }
}

function plotSuffix(plot: PlotOptions): string {
  const parts: string[] = []
  if (plot.points !== 128) parts.push(`plotpoints = ${plot.points}`)
  if (plot.recursion !== 5) parts.push(`maxrecursion = ${plot.recursion}`)
  if (!plot.exclusions) parts.push('exclusions = false')
  const domain: PlotDomain | null = plot.domain
  if (domain) parts.push(`${domain.name} = ${plain(domain.min)}..${plain(domain.max)}`)
  return parts.length ? `, ${parts.join(', ')}` : ''
}

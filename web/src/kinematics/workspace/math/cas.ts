// Number, algebra, and calculus on the rational kernel.
// Forms the kernel cannot do exactly say so instead of returning a wrong answer.

import {
  MathError,
  freeSymbols,
  normalize,
  numericConstant,
  plain,
  polynomialCoefficients,
  present,
  substitute,
  tex,
  texDecimal,
  type AngleMode,
  type Expr,
  type SearchDomain,
} from './expr'

export interface CasCurve {
  expr: Expr
  along: 'x' | 'y'
  dashed?: boolean
  label: string
  /** Fill between this curve and the axis on the parameter interval. */
  shade?: { from: number; to: number }
  /** Plain integrand, so a second copy is not stroked when the curve is already drawn. */
  exprKey?: string
}

export interface CasPoint {
  x: number
  y: number
  label: string
}

export interface CasParametric {
  components: Expr[]
  param: string
  label: string
  dashed?: boolean
}

export interface CasArrow {
  x: number
  y: number
  z: number
  label: string
}

export interface CasVisual {
  text: string
  tex: string
  curves: CasCurve[]
  points: CasPoint[]
  parametrics: CasParametric[]
  arrows: CasArrow[]
  warn: string | null
}

const ZERO: Expr = { type: 'rat', n: 0n, d: 1n }
const ONE: Expr = { type: 'rat', n: 1n, d: 1n }
const REDUCIBLE = new Set(['diff', 'integrate', 'limit', 'sum', 'prod', 'expand', 'decimal', 'fraction', 'gcd', 'lcm', 'mod', 'rem', 'tangent', 'normal', 'series', 'Dt'])

function R(n: bigint, d = 1n): Expr {
  return { type: 'rat', n, d }
}

function S(name: string): Expr {
  return { type: 'sym', name }
}

function add(args: Expr[]): Expr {
  return args.length === 1 ? args[0] : { type: 'add', args }
}

function mul(args: Expr[]): Expr {
  return args.length === 1 ? args[0] : { type: 'mul', args }
}

function div(num: Expr, den: Expr): Expr {
  return { type: 'div', num, den }
}

function pow(base: Expr, exp: Expr): Expr {
  return { type: 'pow', base, exp }
}

function call(name: string, args: Expr[]): Expr {
  return { type: 'call', name, args }
}

function neg(e: Expr): Expr {
  return mul([R(-1n), e])
}

function sub(a: Expr, b: Expr): Expr {
  return add([a, neg(b)])
}

function isSym(e: Expr, name: string): boolean {
  return e.type === 'sym' && e.name === name
}

function mapChildren(e: Expr, visit: (child: Expr) => Expr): Expr {
  switch (e.type) {
    case 'add':
    case 'mul':
      return { ...e, args: e.args.map(visit) }
    case 'div':
      return { type: 'div', num: visit(e.num), den: visit(e.den) }
    case 'pow':
      return { type: 'pow', base: visit(e.base), exp: visit(e.exp) }
    case 'call':
      return { type: 'call', name: e.name, args: e.args.map(visit) }
    case 'vec':
      return { type: 'vec', args: e.args.map(visit) }
    case 'mat':
      return { type: 'mat', rows: e.rows.map((row) => row.map(visit)) }
    case 'eq':
      return { type: 'eq', left: visit(e.left), right: visit(e.right) }
    case 'group':
      return visit(e.body)
    case 'caret':
      return e.body ? visit(e.body) : e
    default:
      return e
  }
}

function walk(e: Expr, visit: (node: Expr) => void): void {
  visit(e)
  mapChildren(e, (child) => {
    walk(child, visit)
    return child
  })
}

export function containsCas(e: Expr): boolean {
  let found = false
  walk(e, (node) => {
    if (node.type === 'call' && (REDUCIBLE.has(node.name) || node.name === 'factor' || node.name === 'zeros' || node.name === 'fmin' || node.name === 'fmax' || node.name === 'dsolve' || node.name === 'idiff')) found = true
  })
  return found
}

function whole(e: Expr, angles: AngleMode): bigint | null {
  const n = numericConstant(normalize(e, angles), angles)
  if (n === null || !Number.isFinite(n)) return null
  const rounded = Math.round(n)
  if (Math.abs(n - rounded) > 1e-8 || Math.abs(rounded) > 1e15) return null
  return BigInt(rounded)
}

function gcdBig(a: bigint, b: bigint): bigint {
  let x = a < 0n ? -a : a
  let y = b < 0n ? -b : b
  while (y !== 0n) {
    const next = x % y
    x = y
    y = next
  }
  return x
}

function isZero(e: Expr, angles: AngleMode): boolean {
  if (e.type === 'rat') return e.n === 0n
  const n = numericConstant(e, angles)
  return n !== null && Math.abs(n) <= 1e-8 * Math.max(1, Math.abs(n))
}

function symbolOf(e: Expr, fallback: string): string {
  if (e.type !== 'sym' || e.sub || (e.dots ?? 0) > 0 || e.name === 'pi' || e.name === 'e') throw new MathError(`Use a variable such as ${fallback}.`)
  return e.name
}

function decimalText(n: number): string {
  if (!Number.isFinite(n)) throw new MathError('Not a real number.')
  if (n === 0) return '0'
  const abs = Math.abs(n)
  const text = abs >= 1e6 || abs < 1e-4 ? n.toExponential(4) : n.toPrecision(8)
  return text.replace(/(\.\d*?)0+(e|$)/, '$1$2').replace(/\.(e|$)/, '$1')
}

function shown(value: Expr, angles: AngleMode, source: Expr | null): { text: string; tex: string; value: Expr } {
  const simplified = normalize(value, angles)
  const display = present(simplified, angles)
  const text = source && plain(source) !== display.text ? `${plain(source)} = ${display.text}` : display.text
  return { text, tex: display.tex, value: simplified }
}

function curveFor(expr: Expr, variable: string, label: string, dashed = false): CasCurve | null {
  if (expr.type === 'vec' || expr.type === 'mat') return null
  const symbols = freeSymbols(expr).filter((name) => name !== 'C' && name !== 'C1' && name !== 'C2' && name !== 'C3')
  if (symbols.length > 1) return null
  if (symbols.length === 1 && symbols[0] !== variable) return null
  if (variable !== 'x' && variable !== 'y') return null
  return { expr, along: variable, dashed, label }
}

function vectorGraphics(expr: Expr, param: string, label: string): { parametrics: CasParametric[]; arrows: CasArrow[] } {
  if (expr.type !== 'vec' || expr.args.length < 2 || expr.args.length > 3) return { parametrics: [], arrows: [] }
  const symbols = freeSymbols(expr).filter((name) => name !== 'C' && !/^C\d+$/.test(name))
  if (symbols.length > 1 || (symbols.length === 1 && symbols[0] !== param)) return { parametrics: [], arrows: [] }
  if (symbols.length === 1) return { parametrics: [{ components: expr.args, param, label }], arrows: [] }
  const nums = expr.args.map((arg) => numericConstant(arg))
  if (nums.some((value) => value === null)) return { parametrics: [], arrows: [] }
  return { arrows: [{ x: nums[0] ?? 0, y: nums[1] ?? 0, z: nums[2] ?? 0, label }], parametrics: [] }
}

function pointAt(x: number, y: number, label: string): CasPoint | null {
  if (!Number.isFinite(x) || !Number.isFinite(y)) return null
  return { x, y, label }
}

function angleIn(angles: AngleMode): Expr {
  return angles === 'deg' ? div(S('pi'), R(180n)) : ONE
}

function angleOut(angles: AngleMode): Expr {
  return angles === 'deg' ? div(R(180n), S('pi')) : ONE
}

function derivative(e: Expr, variable: string, angles: AngleMode, dependent?: { name: string; symbol: string }, total = false): Expr {
  switch (e.type) {
    case 'rat':
    case 'dec':
      return ZERO
    case 'sym':
      return derivativeSymbol(e, variable, dependent, total)
    case 'vec':
      return { type: 'vec', args: e.args.map((arg) => derivative(arg, variable, angles, dependent, total)) }
    case 'mat':
      return { type: 'mat', rows: e.rows.map((row) => row.map((arg) => derivative(arg, variable, angles, dependent, total))) }
    case 'add':
      return add(e.args.map((arg) => derivative(arg, variable, angles, dependent, total)))
    case 'mul': {
      const parts: Expr[] = []
      for (let index = 0; index < e.args.length; index += 1) {
        parts.push(mul(e.args.map((item, itemIndex) => (itemIndex === index ? derivative(item, variable, angles, dependent, total) : item))))
      }
      return add(parts)
    }
    case 'div': {
      const num = derivative(e.num, variable, angles, dependent, total)
      const den = derivative(e.den, variable, angles, dependent, total)
      return div(sub(mul([num, e.den]), mul([e.num, den])), pow(e.den, R(2n)))
    }
    case 'pow':
      return diffPow(e, variable, angles, dependent, total)
    case 'call':
      if (total && e.name === 'Dt' && e.args.length === 1) return derivative(derivative(e.args[0], 't', angles, undefined, true), variable, angles, dependent, true)
      return diffCall(e, variable, angles, dependent, total)
    case 'eq':
      throw new MathError('Differentiate an expression, or use idiff for an equation.')
    case 'group':
      return derivative(e.body, variable, angles, dependent, total)
    case 'caret':
      return e.body ? derivative(e.body, variable, angles, dependent, total) : ZERO
  }
}

function derivativeSymbol(e: Extract<Expr, { type: 'sym' }>, variable: string, dependent: { name: string; symbol: string } | undefined, total: boolean): Expr {
  const marked = Boolean(e.sub) || (e.dots ?? 0) > 0
  if (!marked && e.name === variable) return ONE
  if (total) {
    if (e.name === 'pi' || (e.name === 'e' && !e.sub)) return ZERO
    if (!marked && e.name === 't') return ONE
    if (e.name === 't') return ZERO
    return { type: 'sym', name: e.name, sub: e.sub, dots: (e.dots ?? 0) + 1 }
  }
  if (dependent && !marked && e.name === dependent.name) return S(dependent.symbol)
  return ZERO
}

function diffPow(e: Extract<Expr, { type: 'pow' }>, variable: string, angles: AngleMode, dependent: { name: string; symbol: string } | undefined, total: boolean): Expr {
  const base = derivative(e.base, variable, angles, dependent, total)
  const exp = derivative(e.exp, variable, angles, dependent, total)
  const expConstant = isZero(exp, angles)
  const baseConstant = isZero(base, angles)
  if (expConstant && baseConstant) return ZERO
  if (expConstant) return mul([e.exp, pow(e.base, sub(e.exp, ONE)), base])
  if (baseConstant) return mul([e, call('ln', [e.base]), exp])
  return mul([e, add([mul([exp, call('ln', [e.base])]), div(mul([e.exp, base]), e.base)])])
}

function diffCall(e: Extract<Expr, { type: 'call' }>, variable: string, angles: AngleMode, dependent: { name: string; symbol: string } | undefined, total: boolean): Expr {
  if (e.args.length === 0) throw new MathError(`Cannot differentiate ${e.name} yet.`)
  const arg = e.args[0]
  const inner = derivative(arg, variable, angles, dependent, total)
  const scale = angleIn(angles)
  const chain = mul([inner, scale])
  const inv = mul([inner, angleOut(angles)])
  switch (e.name) {
    case 'sin':
      return mul([call('cos', [arg]), chain])
    case 'cos':
      return mul([neg(call('sin', [arg])), chain])
    case 'tan':
      return mul([div(ONE, pow(call('cos', [arg]), R(2n))), chain])
    case 'asin':
      return mul([div(ONE, call('sqrt', [sub(ONE, pow(arg, R(2n)))])), inv])
    case 'acos':
      return neg(mul([div(ONE, call('sqrt', [sub(ONE, pow(arg, R(2n)))])), inv]))
    case 'atan':
      return mul([div(ONE, add([ONE, pow(arg, R(2n))])), inv])
    case 'ln':
      return div(inner, arg)
    case 'log':
      if (e.args.length === 1) return div(inner, mul([arg, call('ln', [R(10n)])]))
      return div(inner, mul([arg, call('ln', [e.args[1]])]))
    case 'exp':
      return mul([e, inner])
    case 'sqrt':
      return div(inner, mul([R(2n), e]))
    case 'abs':
      return mul([div(arg, e), inner])
    default:
      throw new MathError(`Cannot differentiate ${e.name} yet.`)
  }
}

function diffValue(args: Expr[], angles: AngleMode): { value: Expr; variable: string; at: Expr | null } {
  if (args.length < 2 || args[1].type !== 'sym') throw new MathError('Use diff(expr, x).')
  const variable = args[1].name
  let order = 1
  let at: Expr | null = null
  if (args.length >= 3) {
    const asOrder = whole(args[2], angles)
    if (args.length === 3 && (asOrder === null || asOrder < 1 || asOrder > 6)) at = args[2]
    else {
      if (asOrder === null || asOrder < 1 || asOrder > 6) throw new MathError('The derivative order should be a whole number from 1 to 6.')
      order = Number(asOrder)
      at = args[3] ?? null
    }
  }
  let value = args[0]
  const total = variable === 't'
  for (let i = 0; i < order; i += 1) value = derivative(value, variable, angles, undefined, total)
  if (at) value = substitute(value, new Map([[variable, at]]))
  return { value, variable, at }
}

function hasDot(e: Expr): boolean {
  if (e.type === 'sym') return (e.dots ?? 0) > 0
  if (e.type === 'call' && e.name === 'Dt') return true
  if (e.type === 'add' || e.type === 'mul' || e.type === 'call') return e.args.some(hasDot)
  if (e.type === 'vec') return e.args.some(hasDot)
  if (e.type === 'mat') return e.rows.some((row) => row.some(hasDot))
  if (e.type === 'div') return hasDot(e.num) || hasDot(e.den)
  if (e.type === 'pow') return hasDot(e.base) || hasDot(e.exp)
  if (e.type === 'eq') return hasDot(e.left) || hasDot(e.right)
  if ((e.type === 'group' || e.type === 'caret') && e.body) return hasDot(e.body)
  return false
}

function constantFactor(e: Expr): boolean {
  if (e.type === 'rat' || e.type === 'dec') return true
  if (e.type === 'sym') return e.name === 'pi' || (e.name === 'e' && !e.sub && !(e.dots && e.dots > 0))
  if (e.type === 'mul') return e.args.every(constantFactor)
  if ((e.type === 'group' || e.type === 'caret') && e.body) return constantFactor(e.body)
  return false
}

/** Undo one time dot. A number times a dotted symbol integrates; two dotted factors do not. */
function undot(e: Expr): Expr | null {
  if (e.type === 'sym' && (e.dots ?? 0) > 0) {
    const dots = (e.dots ?? 0) - 1
    return dots > 0 ? { ...e, dots } : { type: 'sym', name: e.name, sub: e.sub }
  }
  if (e.type === 'call' && e.name === 'Dt' && e.args.length === 1) return e.args[0]
  if (e.type === 'add') {
    const args = e.args.map(undot)
    if (args.some((arg) => arg === null)) return null
    return add(args as Expr[])
  }
  if ((e.type === 'group' || e.type === 'caret') && e.body) return undot(e.body)
  if (e.type === 'mul') {
    const index = e.args.findIndex((arg) => hasDot(arg))
    if (index < 0) return null
    if (e.args.some((arg, i) => i !== index && hasDot(arg))) return null
    if (e.args.some((arg, i) => i !== index && !constantFactor(arg))) return null
    const next = undot(e.args[index])
    if (!next) return null
    return mul(e.args.map((arg, i) => (i === index ? next : arg)))
  }
  return null
}

function antiderivative(e: Expr, variable: string, angles: AngleMode): Expr {
  if (e.type === 'group' || e.type === 'caret') return e.body ? antiderivative(e.body, variable, angles) : ZERO
  if (e.type === 'vec') return { type: 'vec', args: e.args.map((arg) => antiderivative(arg, variable, angles)) }
  if (e.type === 'mat') return { type: 'mat', rows: e.rows.map((row) => row.map((arg) => antiderivative(arg, variable, angles))) }
  if (variable === 't') {
    const undone = undot(e)
    if (undone) return undone
    if (hasDot(e)) throw new MathError('Cannot integrate that yet.')
  }
  if (!freeSymbols(e).includes(variable)) return mul([e, S(variable)])
  if (e.type === 'add') return add(e.args.map((arg) => antiderivative(arg, variable, angles)))
  if (e.type === 'mul') {
    const constants = e.args.filter((arg) => !freeSymbols(arg).includes(variable))
    const varying = e.args.filter((arg) => freeSymbols(arg).includes(variable))
    if (constants.length > 0 && varying.length === 1) return mul([...constants, antiderivative(varying[0], variable, angles)])
  }
  const polynomial = integratePolynomial(e, variable, angles)
  if (polynomial) return polynomial
  const quotient = integrateQuotient(e, variable, angles)
  if (quotient) return quotient
  const elementary = integrateElementary(e, variable, angles)
  if (elementary) return elementary
  throw new MathError('Cannot integrate that yet.')
}

function integratePolynomial(e: Expr, variable: string, angles: AngleMode): Expr | null {
  const coeffs = polynomialCoefficients(e, variable)
  if (!coeffs) return null
  const terms: Expr[] = []
  for (let power = 0; power < coeffs.length; power += 1) {
    const coeff = coeffs[power] ?? ZERO
    if (isZero(coeff, angles)) continue
    if (power === 0 && coeff.type === 'div' && freeSymbols(coeff).includes(variable)) return null
    terms.push(div(mul([coeff, pow(S(variable), R(BigInt(power + 1)))]), R(BigInt(power + 1))))
  }
  return terms.length === 0 ? ZERO : add(terms)
}

function integrateQuotient(e: Expr, variable: string, angles: AngleMode): Expr | null {
  if (e.type !== 'div' || freeSymbols(e.num).includes(variable)) return null
  const linear = linearOf(e.den, variable, angles)
  if (!linear || isZero(linear.slope, angles)) return null
  return mul([div(e.num, linear.slope), call('ln', [call('abs', [e.den])])])
}

function integrateElementary(e: Expr, variable: string, angles: AngleMode): Expr | null {
  if (e.type !== 'call' || e.args.length !== 1) return null
  const linear = linearOf(e.args[0], variable, angles)
  if (!linear || isZero(linear.slope, angles)) return null
  const scale = e.name === 'sin' || e.name === 'cos' ? angleIn(angles) : ONE
  const denom = mul([linear.slope, scale])
  if (e.name === 'sin') return div(neg(call('cos', [e.args[0]])), denom)
  if (e.name === 'cos') return div(call('sin', [e.args[0]]), denom)
  if (e.name === 'exp') return div(call('exp', [e.args[0]]), linear.slope)
  return null
}

function linearOf(e: Expr, variable: string, angles: AngleMode): { slope: Expr; intercept: Expr } | null {
  const coeffs = polynomialCoefficients(e, variable)
  if (!coeffs || coeffs.length > 2) return null
  const slope = coeffs[1] ?? ZERO
  const intercept = coeffs[0] ?? ZERO
  if (freeSymbols(slope).includes(variable) || freeSymbols(intercept).includes(variable)) return null
  if (isZero(slope, angles) && isZero(intercept, angles)) return null
  return { slope, intercept }
}

function simpson(expr: Expr, variable: string, a: number, b: number, angles: AngleMode): number {
  const count = 200
  const at = (t: number) => {
    const value = numericConstant(substitute(expr, new Map([[variable, { type: 'dec', text: String(t), value: t }]])), angles)
    if (value === null || !Number.isFinite(value)) throw new MathError('Cannot integrate that on this interval.')
    return value
  }
  const step = (b - a) / count
  let total = at(a) + at(b)
  for (let i = 1; i < count; i += 1) total += (i % 2 === 0 ? 2 : 4) * at(a + i * step)
  return (step / 3) * total
}

function limitValue(expr: Expr, variable: string, point: Expr, angles: AngleMode, depth = 0): Expr {
  const substituted = substitute(expr, new Map([[variable, point]]))
  let simplified = substituted
  try {
    simplified = normalize(substituted, angles)
  } catch {
    simplified = substituted
  }
  const value = numericConstant(simplified, angles)
  if (value !== null && Number.isFinite(value) && depth === 0) return simplified
  if (value !== null && Number.isFinite(value) && depth > 0) return simplified
  if (depth < 4 && expr.type === 'div') {
    try {
      return limitValue(div(derivative(expr.num, variable, angles), derivative(expr.den, variable, angles)), variable, point, angles, depth + 1)
    } catch {
      // Fall through to a numeric approach.
    }
  }
  const at = numericConstant(point, angles)
  if (at === null) throw new MathError('Cannot find that limit yet.')
  let previous: number | null = null
  for (const step of [1e-3, 1e-4, 1e-5, 1e-6]) {
    const left = numericConstant(substitute(expr, new Map([[variable, { type: 'dec', text: String(at - step), value: at - step }]])), angles)
    const right = numericConstant(substitute(expr, new Map([[variable, { type: 'dec', text: String(at + step), value: at + step }]])), angles)
    if (left === null || right === null || !Number.isFinite(left) || !Number.isFinite(right)) continue
    if (Math.abs(left - right) > 1e-3 * Math.max(1, Math.abs(left), Math.abs(right))) throw new MathError('The two sides of that limit do not agree.')
    const average = (left + right) / 2
    if (previous !== null && Math.abs(average - previous) <= 1e-4 * Math.max(1, Math.abs(average))) {
      return { type: 'dec', text: decimalText(average), value: average }
    }
    previous = average
  }
  if (previous !== null) return { type: 'dec', text: decimalText(previous), value: previous }
  throw new MathError('Cannot find that limit yet.')
}

function finiteFold(args: Expr[], angles: AngleMode, product: boolean): Expr {
  const index = symbolOf(args[1], 'i')
  const start = whole(args[2], angles)
  const end = whole(args[3], angles)
  if (start === null || end === null) throw new MathError('The bounds need to be whole numbers.')
  if (end < start || end - start > 400) throw new MathError('Use a range of at most 400 terms.')
  let acc: Expr = product ? ONE : ZERO
  for (let i = start; i <= end; i += 1n) {
    const term = substitute(args[0], new Map([[index, R(i)]]))
    acc = product ? mul([acc, term]) : add([acc, term])
  }
  return acc
}

function quadraticRoots(a: Expr, b: Expr, c: Expr, angles: AngleMode): Expr[] {
  const disc = normalize(sub(pow(b, R(2n)), mul([R(4n), a, c])), angles)
  const value = numericConstant(disc, angles)
  if (value === null || value < -1e-8) return []
  if (Math.abs(value) <= 1e-8) {
    const root = normalize(div(neg(b), mul([R(2n), a])), angles)
    return [root, root]
  }
  let radical: Expr
  try {
    radical = normalize(call('sqrt', [disc]), angles)
  } catch {
    return []
  }
  const twoA = mul([R(2n), a])
  const plus = normalize(div(add([neg(b), radical]), twoA), angles)
  const minus = normalize(div(sub(neg(b), radical), twoA), angles)
  return plain(plus) === plain(minus) ? [plus] : [plus, minus]
}

function rationalRoots(expr: Expr, variable: string, angles: AngleMode): Expr[] {
  const normalized = normalize(expr, angles)
  const coeffs = polynomialCoefficients(normalized, variable)
  if (!coeffs) return numericRoots(normalized, variable, angles, -10, 10)
  let degree = coeffs.length - 1
  while (degree > 0 && isZero(coeffs[degree] ?? ZERO, angles)) degree -= 1
  if (degree <= 0) return []
  if (degree === 1) return [normalize(div(neg(coeffs[0] ?? ZERO), coeffs[1] ?? ONE), angles)]
  if (degree === 2) return quadraticRoots(coeffs[2] ?? ONE, coeffs[1] ?? ZERO, coeffs[0] ?? ZERO, angles)
  const integer = coeffs.slice(0, degree + 1).map((coeff) => (coeff.type === 'rat' && coeff.d === 1n ? coeff.n : null))
  if (integer.some((coeff) => coeff === null)) return numericRoots(normalized, variable, angles, -10, 10)
  const values = integer as bigint[]
  const found: Expr[] = []
  let rest = values
  for (let guard = 0; guard < 6 && rest.length > 3; guard += 1) {
    const root = integerRoot(rest)
    if (root === null) break
    found.push(R(root))
    rest = deflate(rest, root)
  }
  if (rest.length === 3) found.push(...quadraticRoots(R(rest[2] ?? 1n), R(rest[1] ?? 0n), R(rest[0] ?? 0n), angles))
  else if (rest.length === 2) found.push(normalize(div(neg(R(rest[0] ?? 0n)), R(rest[1] ?? 1n)), angles))
  return found
}

function integerRoot(coeffs: bigint[]): bigint | null {
  const constant = coeffs[0] ?? 0n
  if (constant === 0n) return 0n
  const lead = coeffs[coeffs.length - 1] ?? 1n
  const candidates = divisors(constant < 0n ? -constant : constant)
  for (const candidate of candidates) {
    for (const root of [candidate, -candidate]) {
      if (polyAt(coeffs, root) === 0n && lead !== 0n) return root
    }
  }
  return null
}

function divisors(n: bigint): bigint[] {
  const values: bigint[] = []
  for (let i = 1n; i * i <= n && i < 100000n; i += 1n) {
    if (n % i === 0n) {
      values.push(i)
      if (i * i !== n) values.push(n / i)
    }
  }
  return values
}

function polyAt(coeffs: bigint[], x: bigint): bigint {
  let acc = 0n
  for (let i = coeffs.length - 1; i >= 0; i -= 1) acc = acc * x + (coeffs[i] ?? 0n)
  return acc
}

function deflate(coeffs: bigint[], root: bigint): bigint[] {
  const out: bigint[] = []
  let acc = 0n
  for (let i = coeffs.length - 1; i >= 1; i -= 1) {
    acc = acc * root + (coeffs[i] ?? 0n)
    out.push(acc)
  }
  return out.reverse()
}

function numericRoots(expr: Expr, variable: string, angles: AngleMode, min: number, max: number): Expr[] {
  const at = (t: number) => numericConstant(substitute(expr, new Map([[variable, { type: 'dec', text: String(t), value: t }]])), angles)
  const roots: number[] = []
  let previousT = min
  let previous = at(min)
  const steps = 64
  for (let i = 1; i <= steps; i += 1) {
    const t = min + ((max - min) * i) / steps
    const value = at(t)
    if (previous !== null && value !== null && Number.isFinite(previous) && Number.isFinite(value) && previous === 0) roots.push(previousT)
    if (previous !== null && value !== null && Number.isFinite(previous) && Number.isFinite(value) && previous * value < 0) {
      let lo = previousT
      let hi = t
      let flo = previous
      for (let step = 0; step < 24; step += 1) {
        const mid = (lo + hi) / 2
        const fmid = at(mid)
        if (fmid === null || !Number.isFinite(fmid) || flo === null) break
        if (flo * fmid <= 0) hi = mid
        else {
          lo = mid
          flo = fmid
        }
      }
      roots.push((lo + hi) / 2)
    }
    previousT = t
    previous = value
  }
  const unique = roots.filter((root, index) => roots.findIndex((other) => Math.abs(other - root) < 1e-3) === index)
  return unique.map((root) => {
    const nearest = Math.round(root)
    if (Math.abs(root - nearest) < 1e-4) return R(BigInt(nearest))
    return { type: 'dec', text: decimalText(root), value: root }
  })
}

function factorInteger(n: bigint): Expr {
  if (n < 0n) return mul([R(-1n), factorInteger(-n)])
  if (n === 0n || n === 1n) return R(n)
  const pieces: Expr[] = []
  let rest = n
  const take = (prime: bigint) => {
    let exp = 0
    while (rest % prime === 0n) {
      rest /= prime
      exp += 1
    }
    if (exp === 1) pieces.push(R(prime))
    else if (exp > 1) pieces.push(pow(R(prime), R(BigInt(exp))))
  }
  take(2n)
  for (let prime = 3n; prime * prime <= rest && prime <= 1000000n; prime += 2n) take(prime)
  if (rest > 1n) pieces.push(R(rest))
  return pieces.length === 1 ? pieces[0] : mul(pieces)
}

function factorPolynomial(expr: Expr, variable: string, angles: AngleMode): Expr {
  const normalized = normalize(expr, angles)
  const coeffs = polynomialCoefficients(normalized, variable)
  if (!coeffs) throw new MathError('Factor a whole number or a polynomial in one variable.')
  let degree = coeffs.length - 1
  while (degree > 0 && isZero(coeffs[degree] ?? ZERO, angles)) degree -= 1
  if (degree <= 0) {
    const value = whole(coeffs[0] ?? ZERO, angles)
    if (value === null) throw new MathError('Factor a whole number or a polynomial in one variable.')
    return factorInteger(value)
  }
    const roots = rationalRoots(normalized, variable, angles)
    if (roots.length === 0) throw new MathError('No rational factors.')
    const grouped = new Map<string, { root: Expr; count: number }>()
    for (const root of roots) {
      const key = plain(root)
      const prev = grouped.get(key)
      if (prev) prev.count += 1
      else grouped.set(key, { root, count: 1 })
    }
    const factors = [...grouped.values()].map(({ root, count }) => {
      const linear = sub(S(variable), root)
      return count === 1 ? linear : pow(linear, R(BigInt(count)))
    })
  const lead = coeffs[degree] ?? ONE
  const body = factors.length === 1 ? factors[0] : mul(factors)
  if (isZero(sub(lead, ONE), angles)) return body
  return mul([lead, body])
}

function approximateFraction(n: number): { n: bigint; d: bigint } {
  const sign = n < 0 ? -1n : 1n
  const target = Math.abs(n)
  let rest = target
  let prevN = 0n
  let currN = 1n
  let prevD = 1n
  let currD = 0n
  let bestN = 0n
  let bestD = 1n
  for (let i = 0; i < 24; i += 1) {
    const term = Math.floor(rest + 1e-10)
    if (!Number.isFinite(term) || term > 1e8) break
    const nextN = BigInt(term) * currN + prevN
    const nextD = BigInt(term) * currD + prevD
    if (nextD <= 0n || nextD > 10000n) break
    prevN = currN
    prevD = currD
    currN = nextN
    currD = nextD
    bestN = currN
    bestD = currD
    const error = Math.abs(Number(currN) / Number(currD) - target)
    if (error <= 2e-3 && currD <= 200n) break
    if (error <= 1e-9 * Math.max(1, target)) break
    const frac = rest - term
    if (frac < 1e-12) break
    rest = 1 / frac
  }
  return { n: sign * bestN, d: bestD }
}

function numbersOf(args: Expr[], angles: AngleMode): bigint[] {
  return args.map((arg) => {
    const value = whole(arg, angles)
    if (value === null) throw new MathError('Use whole numbers.')
    return value
  })
}

function extremum(args: Expr[], angles: AngleMode, kind: 'min' | 'max'): { value: Expr; at: Expr; variable: string; body: Expr } {
  const variable = symbolOf(args[1], 'x')
  const start = numericConstant(normalize(args[2], angles), angles)
  const end = numericConstant(normalize(args[3], angles), angles)
  if (start === null || end === null || end <= start) throw new MathError('Use an interval with the right end greater than the left.')
  const body = args[0]
  const sample = (t: number) => numericConstant(substitute(body, new Map([[variable, { type: 'dec', text: String(t), value: t }]])), angles)
  const candidates = [start, end]
  try {
    const slope = derivative(body, variable, angles)
    for (const root of rationalRoots(slope, variable, angles)) {
      const at = numericConstant(root, angles)
      if (at !== null && at >= start && at <= end) candidates.push(at)
    }
  } catch {
    // A derivative that is not available still leaves the endpoints and the sample.
  }
  const steps = 80
  for (let i = 0; i <= steps; i += 1) candidates.push(start + ((end - start) * i) / steps)
  let bestT = start
  let best = sample(start)
  if (best === null) throw new MathError('Cannot evaluate that on the interval.')
  for (const candidate of candidates) {
    const value = sample(candidate)
    if (value === null || !Number.isFinite(value)) continue
    if (kind === 'min' ? value < best : value > best) {
      best = value
      bestT = candidate
    }
  }
  const at = Math.abs(bestT - Math.round(bestT)) < 1e-4 ? R(BigInt(Math.round(bestT))) : { type: 'dec' as const, text: decimalText(bestT), value: bestT }
  const exact = numericConstant(substitute(body, new Map([[variable, at]])), angles)
  const height = exact !== null && Number.isFinite(exact) ? normalize(substitute(body, new Map([[variable, at]])), angles) : { type: 'dec' as const, text: decimalText(best), value: best }
  return { value: height, at, variable, body }
}

function taylor(args: Expr[], angles: AngleMode): Expr {
  const variable = symbolOf(args[1], 'x')
  const point = args[2]
  const order = whole(args[3], angles)
  if (order === null || order < 0 || order > 8) throw new MathError('The series order should be a whole number from 0 to 8.')
  let deriv = args[0]
  let factorial = 1
  const terms: Expr[] = [substitute(args[0], new Map([[variable, point]]))]
  for (let k = 1; k <= Number(order); k += 1) {
    deriv = derivative(deriv, variable, angles)
    factorial *= k
    const coeff = div(substitute(deriv, new Map([[variable, point]])), R(BigInt(factorial)))
    terms.push(mul([coeff, pow(sub(S(variable), point), R(BigInt(k)))]))
  }
  return add(terms)
}

function replaceDerivative(e: Expr, dependent: string, independent: string): Expr {
  if (e.type === 'call' && e.name === 'diff' && e.args.length >= 2 && isSym(e.args[1], independent)) {
    if (isSym(e.args[0], dependent)) return S('__d')
    if (e.args[0].type === 'call') {
      const inner = replaceDerivative(e.args[0], dependent, independent)
      if (isSym(inner, '__d')) return S('__d2')
    }
  }
  return mapChildren(e, (child) => replaceDerivative(child, dependent, independent))
}

function equationZero(eq: Expr): Expr {
  if (eq.type !== 'eq') throw new MathError('That needs an equation.')
  return sub(eq.left, eq.right)
}

function solveDe(args: Expr[], angles: AngleMode): Expr {
  const dependent = symbolOf(args[1], 'y')
  const independent = symbolOf(args[2], 'x')
  if (dependent === independent) throw new MathError('Use two different names, for example dsolve(diff(y, x) = y, y, x).')
  const zero = normalize(replaceDerivative(equationZero(args[0]), dependent, independent), angles)
  if (freeSymbols(zero).includes('__d2')) return solveSecondOrder(zero, dependent, independent, angles)
  if (!freeSymbols(zero).includes('__d')) throw new MathError("The equation needs y', for example dsolve(diff(y, x) = y, y, x).")
  const poly = polynomialCoefficients(zero, '__d')
  if (!poly || poly.length > 2 || isZero(poly[1] ?? ZERO, angles)) throw new MathError('Cannot solve that differential equation yet.')
  const rhs = normalize(div(neg(poly[0] ?? ZERO), poly[1] ?? ONE), angles)
  if (!freeSymbols(rhs).includes(dependent)) return add([antiderivative(rhs, independent, angles), S('C')])
  const growth = normalize(div(rhs, S(dependent)), angles)
  if (freeSymbols(growth).includes(dependent) || freeSymbols(growth).includes(independent)) {
    throw new MathError("Cannot solve that differential equation yet. It can be y' = f(x) or y' = ky.")
  }
  return mul([S('C'), call('exp', [normalize(mul([growth, S(independent)]), angles)])])
}

function solveSecondOrder(zero: Expr, dependent: string, independent: string, angles: AngleMode): Expr {
  const second = polynomialCoefficients(zero, '__d2')
  if (!second || second.length !== 2 || isZero(second[1] ?? ZERO, angles)) throw new MathError('Cannot solve that differential equation yet.')
  const first = polynomialCoefficients(second[0] ?? ZERO, '__d')
  const slope = first && first.length > 1 ? (first[1] ?? ZERO) : ZERO
  const rest = first ? (first[0] ?? ZERO) : (second[0] ?? ZERO)
  const base = polynomialCoefficients(rest, dependent)
  if (!base || base.length > 2 || !isZero(base[0] ?? ZERO, angles)) throw new MathError('Cannot solve that differential equation yet.')
  const A = normalize(div(slope, second[1] ?? ONE), angles)
  const B = normalize(div(base[1] ?? ZERO, second[1] ?? ONE), angles)
  if (A.type !== 'rat' || B.type !== 'rat') throw new MathError('Cannot solve that differential equation yet.')
  const disc = normalize(sub(pow(A, R(2n)), mul([R(4n), B])), angles)
  const discValue = numericConstant(disc, angles)
  if (discValue === null) throw new MathError('Cannot solve that differential equation yet.')
  const x = S(independent)
  if (discValue < -1e-8) {
    const alpha = normalize(div(neg(A), R(2n)), angles)
    const beta = normalize(div(call('sqrt', [neg(disc)]), R(2n)), angles)
    const wave = normalize(mul([beta, x]), angles)
    const osc = add([mul([S('C1'), call('cos', [wave])]), mul([S('C2'), call('sin', [wave])])])
    return isZero(alpha, angles) ? osc : mul([call('exp', [normalize(mul([alpha, x]), angles)]), osc])
  }
  const radical = normalize(call('sqrt', [disc]), angles)
  const half = div(ONE, R(2n))
  const r1 = normalize(mul([half, add([neg(A), radical])]), angles)
  const r2 = normalize(mul([half, sub(neg(A), radical)]), angles)
  if (plain(r1) === plain(r2)) return mul([add([S('C1'), mul([S('C2'), x])]), call('exp', [normalize(mul([r1, x]), angles)])])
  return add([mul([S('C1'), call('exp', [normalize(mul([r1, x]), angles)])]), mul([S('C2'), call('exp', [normalize(mul([r2, x]), angles)])])])
}

function implicitDerivative(args: Expr[], angles: AngleMode): Expr {
  const dependent = symbolOf(args[1], 'y')
  const independent = symbolOf(args[2], 'x')
  const slope = derivative(equationZero(args[0]), independent, angles, { name: dependent, symbol: '__yp' })
  const poly = polynomialCoefficients(normalize(slope, angles), '__yp')
  if (!poly || poly.length > 2 || isZero(poly[1] ?? ZERO, angles)) throw new MathError('Cannot solve that for the derivative yet.')
  return div(neg(poly[0] ?? ZERO), poly[1] ?? ONE)
}

function solveLinear(equations: Expr[], angles: AngleMode): { names: string[]; values: Expr[]; curves: CasCurve[] } {
  if (equations.length !== 2) throw new MathError('Use two equations, for example solve(x + y = 3, x - y = 1).')
  const zeros = equations.map((equation) => normalize(equationZero(equation), angles))
  const names = [...new Set(zeros.flatMap((zero) => freeSymbols(zero)))]
  if (names.length !== 2) throw new MathError('A system of two equations needs two unknowns.')
  const rows = zeros.map((zero) => affine(zero, names, angles))
  const a1 = rows[0]?.a[0] ?? ZERO
  const b1 = rows[0]?.a[1] ?? ZERO
  const c1 = rows[0]?.c ?? ZERO
  const a2 = rows[1]?.a[0] ?? ZERO
  const b2 = rows[1]?.a[1] ?? ZERO
  const c2 = rows[1]?.c ?? ZERO
  const det = normalize(sub(mul([a1, b2]), mul([a2, b1])), angles)
  if (isZero(det, angles)) throw new MathError('That system does not have one solution.')
  const first = normalize(div(sub(mul([c2, b1]), mul([c1, b2])), det), angles)
  const second = normalize(div(sub(mul([a2, c1]), mul([a1, c2])), det), angles)
  const curves = rows.flatMap((row, index) => lineFrom(row, names, `line ${index + 1}`, angles))
  return { names, values: [first, second], curves }
}

function affine(expr: Expr, names: string[], angles: AngleMode): { a: Expr[]; c: Expr } {
  let rest = expr
  const coefficients: Expr[] = []
  for (const name of names) {
    const poly = polynomialCoefficients(rest, name)
    if (!poly || poly.length > 2) throw new MathError('That system is not linear.')
    const coeff = poly[1] ?? ZERO
    if (freeSymbols(coeff).some((symbol) => names.includes(symbol))) throw new MathError('That system is not linear.')
    coefficients.push(coeff)
    rest = poly[0] ?? ZERO
  }
  if (freeSymbols(rest).some((symbol) => names.includes(symbol))) throw new MathError('That system is not linear.')
  if (isZero(coefficients[0] ?? ZERO, angles) && isZero(coefficients[1] ?? ZERO, angles)) throw new MathError('That system is not linear.')
  return { a: coefficients, c: rest }
}

function lineFrom(row: { a: Expr[]; c: Expr }, names: string[], label: string, angles: AngleMode): CasCurve[] {
  const xIndex = names.indexOf('x')
  const yIndex = names.indexOf('y')
  if (xIndex < 0 || yIndex < 0) return []
  const a = row.a[xIndex] ?? ZERO
  const b = row.a[yIndex] ?? ZERO
  const c = row.c
  if (!isZero(b, angles)) return [{ expr: div(neg(add([mul([a, S('x')]), c])), b), along: 'x', label }]
  if (!isZero(a, angles)) return [{ expr: div(neg(c), a), along: 'y', label }]
  return []
}

function presentValue(value: Expr, angles: AngleMode, source: Expr, curves: CasCurve[] = [], points: CasPoint[] = [], warn: string | null = null): CasVisual {
  const display = shown(value, angles, source)
  return { text: display.text, tex: display.tex, curves, points, parametrics: [], arrows: [], warn }
}

function withOriginal(body: Expr, variable: string, curves: CasCurve[]): CasCurve[] {
  const original = curveFor(body, variable, 'curve', true)
  return original ? [original, ...curves] : curves
}

export function rewriteAll(e: Expr, angles: AngleMode, depth = 0): Expr {
  if (depth > 24) return e
  if (e.type === 'call' && (e.name === 'dsolve' || e.name === 'idiff')) {
    const solved = e.name === 'dsolve' ? solveDe(e.args, angles) : implicitDerivative(e.args, angles)
    return rewriteAll(solved, angles, depth + 1)
  }
  const mapped = mapChildren(e, (child) => rewriteAll(child, angles, depth + 1))
  if (mapped.type !== 'call' || !REDUCIBLE.has(mapped.name)) return mapped
  return rewriteAll(reduceNamed(mapped.name, mapped.args, angles), angles, depth + 1)
}

function reduceNamed(name: string, args: Expr[], angles: AngleMode): Expr {
  switch (name) {
    case 'Dt':
      if (args.length !== 1) throw new MathError('Use Dt(expr) for a time derivative.')
      return derivative(args[0], 't', angles, undefined, true)
    case 'diff':
      return diffValue(args, angles).value
    case 'integrate': {
      const variable = symbolOf(args[1], 'x')
      if (args.length === 2) return antiderivative(args[0], variable, angles)
      const upper = substitute(antiderivative(args[0], variable, angles), new Map([[variable, args[3]]]))
      const lower = substitute(antiderivative(args[0], variable, angles), new Map([[variable, args[2]]]))
      return sub(upper, lower)
    }
    case 'limit':
      return limitValue(args[0], symbolOf(args[1], 'x'), args[2], angles)
    case 'sum':
      return finiteFold(args, angles, false)
    case 'prod':
      return finiteFold(args, angles, true)
    case 'expand':
    case 'decimal':
    case 'fraction':
    case 'gcd':
    case 'lcm':
    case 'mod':
    case 'rem':
    case 'tangent':
    case 'normal':
    case 'series':
      return valueNamed(name, args, angles)
    default:
      throw new MathError(`Cannot do ${name} yet.`)
  }
}

function valueNamed(name: string, args: Expr[], angles: AngleMode): Expr {
  switch (name) {
    case 'expand':
      return args[0]
    case 'decimal': {
      const value = numericConstant(normalize(args[0], angles), angles)
      if (value === null) throw new MathError('decimal needs a number.')
      return { type: 'dec', text: decimalText(value), value }
    }
    case 'fraction': {
      if (args[0].type === 'rat' && args[0].d <= 20n) return args[0]
      const value = numericConstant(normalize(args[0], angles), angles)
      if (value === null) throw new MathError('fraction needs a number.')
      const approx = approximateFraction(value)
      return R(approx.n, approx.d)
    }
    case 'gcd': {
      const values = numbersOf(args, angles)
      return R(values.reduce((acc, value) => gcdBig(acc, value)))
    }
    case 'lcm': {
      const values = numbersOf(args, angles)
      const result = values.reduce((acc, value) => (acc === 0n || value === 0n ? 0n : (acc / gcdBig(acc, value)) * value))
      return R(result < 0n ? -result : result)
    }
    case 'mod':
    case 'rem': {
      const [left, right] = numbersOf(args, angles)
      if (right === 0n) throw new MathError('Cannot divide by zero.')
      const remainder = left % right
      const positive = remainder < 0n ? remainder + (right < 0n ? -right : right) : remainder
      return R(positive)
    }
    case 'series':
      return taylor(args, angles)
    case 'tangent':
    case 'normal': {
      const variable = symbolOf(args[1], 'x')
      const point = args[2]
      const slope = normalize(derivative(substitute(args[0], new Map()), variable, angles), angles)
      const atSlope = normalize(substitute(slope, new Map([[variable, point]])), angles)
      const height = normalize(substitute(args[0], new Map([[variable, point]])), angles)
      if (name === 'normal' && isZero(atSlope, angles)) return { type: 'eq', left: S(variable), right: point }
      const grade = name === 'normal' ? div(R(-1n), atSlope) : atSlope
      return add([height, mul([grade, sub(S(variable), point)])])
    }
    default:
      throw new MathError(`Cannot do ${name} yet.`)
  }
}

export function solveSystem(equations: Expr[], domains: SearchDomain[], angles: AngleMode): CasVisual {
  try {
    const solved = solveLinear(equations, angles)
    if (domains.length === 0) return presentLinear(solved)
    const numbers = solved.values.map((value) => numericConstant(normalize(value, angles), angles))
    if (numbers.every((value) => value !== null) && fitsDomains(solved.names, numbers as number[], domains, equations, angles)) return presentLinear(solved)
    throw new MathError('The solution is outside that domain.')
  } catch (error) {
    const nonlinear = error instanceof MathError && error.message === 'That system is not linear.'
    if (!nonlinear) throw error
    if (domains.length === 0) throw new MathError('That system is not linear. Add a domain to search, for example solve(eq1, eq2, (0, 2*pi)).')
    return solveNumeric(equations, domains, angles)
  }
}

function presentLinear(solved: { names: string[]; values: Expr[]; curves: CasCurve[] }): CasVisual {
  const parts = solved.names.map((name, index) => `${name} = ${plain(solved.values[index] ?? ZERO)}`)
  const texParts = solved.names.map((name, index) => `${tex(S(name))} = ${tex(solved.values[index] ?? ZERO)}`)
  return { text: parts.join(', '), tex: texParts.join(', '), curves: solved.curves, points: [], parametrics: [], arrows: [], warn: null }
}

const TRIG = new Set(['sin', 'cos', 'tan', 'asin', 'acos', 'atan'])

function usesTrig(expr: Expr, name: string): boolean {
  if (expr.type === 'call' && TRIG.has(expr.name) && expr.args.some((arg) => freeSymbols(arg).includes(name))) return true
  if (expr.type === 'add' || expr.type === 'mul' || expr.type === 'call' || expr.type === 'vec') return expr.args.some((arg) => usesTrig(arg, name))
  if (expr.type === 'mat') return expr.rows.some((row) => row.some((arg) => usesTrig(arg, name)))
  if (expr.type === 'div') return usesTrig(expr.num, name) || usesTrig(expr.den, name)
  if (expr.type === 'pow') return usesTrig(expr.base, name) || usesTrig(expr.exp, name)
  if (expr.type === 'eq') return usesTrig(expr.left, name) || usesTrig(expr.right, name)
  if (expr.type === 'group') return usesTrig(expr.body, name)
  if (expr.type === 'caret') return expr.body ? usesTrig(expr.body, name) : false
  return false
}

function domainEnds(domain: SearchDomain, angles: AngleMode): { min: number; max: number } {
  const min = numericConstant(normalize(domain.min, angles), angles)
  const max = numericConstant(normalize(domain.max, angles), angles)
  if (min === null || max === null || !(max > min)) throw new MathError('Use a domain such as (0, 2*pi), with the right end greater than the left.')
  return { min, max }
}

function boxesFor(names: string[], equations: Expr[], domains: SearchDomain[], angles: AngleMode): { name: string; min: number; max: number }[] {
  const trig = names.filter((name) => equations.some((equation) => usesTrig(equation, name)))
  const unnamed = domains.filter((domain) => domain.name === null)
  if (unnamed.length > 1) throw new MathError('Name each domain, for example solve(eq1, eq2, (theta, 0, 2*pi)).')
  return names.map((name) => {
    const named = domains.find((domain) => domain.name === name)
    if (named) return { name, ...domainEnds(named, angles) }
    const shared = unnamed[0]
    if (shared && (trig.length === 0 || trig.includes(name))) return { name, ...domainEnds(shared, angles) }
    return { name, min: -40, max: 40 }
  })
}

function fitsDomains(names: string[], values: number[], domains: SearchDomain[], equations: Expr[], angles: AngleMode): boolean {
  const boxes = boxesFor(names, equations, domains, angles)
  return boxes.every((box, index) => {
    const value = values[index]
    return value !== undefined && value >= box.min - 1e-6 && value <= box.max + 1e-6
  })
}

function solveNumeric(equations: Expr[], domains: SearchDomain[], angles: AngleMode): CasVisual {
  if (equations.length !== 2) throw new MathError('Use two equations, for example solve(eq1, eq2, (0, 2*pi)).')
  const zeros = equations.map((equation) => normalize(equationZero(equation), angles))
  const names = [...new Set(zeros.flatMap((zero) => freeSymbols(zero)))].filter((name) => name !== 'pi' && name !== 'e')
  if (names.length !== 2) throw new MathError('A system of two equations needs two unknowns.')
  const boxes = boxesFor(names, equations, domains, angles)
  const evaluate = (xs: number[]) => {
    const map = new Map(names.map((name, index) => [name, { type: 'dec' as const, text: String(xs[index] ?? 0), value: xs[index] ?? 0 }]))
    const values = zeros.map((zero) => numericConstant(substitute(zero, map), angles))
    if (values.some((value) => value === null || !Number.isFinite(value))) return null
    return values as number[]
  }
  const seeds: { xs: number[]; err: number }[] = []
  const steps = boxes.map((box) => (box.max - box.min < 20 ? 48 : 16))
  const count0 = steps[0] ?? 16
  const count1 = steps[1] ?? 16
  for (let i = 0; i <= count0; i += 1) {
    for (let j = 0; j <= count1; j += 1) {
      const xs = boxes.map((box, index) => box.min + ((box.max - box.min) * (index === 0 ? i : j)) / (steps[index] ?? 1))
      const value = evaluate(xs)
      if (!value) continue
      seeds.push({ xs, err: Math.hypot(value[0] ?? 0, value[1] ?? 0) })
    }
  }
  seeds.sort((a, b) => a.err - b.err)
  const found: number[][] = []
  for (const seed of seeds.slice(0, 36)) {
    const hit = newtonSystem(evaluate, seed.xs, boxes)
    if (!hit) continue
    if (found.some((other) => other.every((value, index) => Math.abs(value - (hit[index] ?? 0)) <= 2e-3 * Math.max(1, Math.abs(value))))) continue
    found.push(hit)
    if (found.length >= 8) break
  }
  if (found.length === 0) throw new MathError('No solution on that domain.')
  const pieces = found.map((xs) => names.map((name, index) => `${name} = ${plain(shownRoot(xs[index] ?? 0))}`).join(', '))
  const texPieces = found.map((xs) => names.map((name, index) => `${tex(S(name))} = ${tex(shownRoot(xs[index] ?? 0))}`).join(', '))
  const where = boxes.map((box) => `${box.name} from ${decimalText(box.min)} to ${decimalText(box.max)}`).join(', ')
  return { text: pieces.join(' or '), tex: texPieces.join(' \\;\\text{or}\\; '), curves: [], points: [], parametrics: [], arrows: [], warn: `Searched ${where}.` }
}

function shownRoot(value: number): Expr {
  const nearest = Math.round(value)
  if (Math.abs(value - nearest) < 1e-5 && Math.abs(nearest) < 1e9) return R(BigInt(nearest))
  return { type: 'dec', text: decimalText(value), value }
}

function newtonSystem(evaluate: (xs: number[]) => number[] | null, start: number[], boxes: { min: number; max: number }[]): number[] | null {
  let xs = start.slice()
  for (let step = 0; step < 18; step += 1) {
    const value = evaluate(xs)
    if (!value) return null
    const error = Math.hypot(value[0] ?? 0, value[1] ?? 0)
    if (error < 1e-8) return insideBox(xs, boxes) ? xs : null
    const columns: number[][] = []
    for (let index = 0; index < xs.length; index += 1) {
      const span = Math.max(1e-4, Math.abs(boxes[index]?.max ?? 1) - (boxes[index]?.min ?? 0))
      const h = Math.max(1e-6, span * 1e-6)
      const shifted = xs.slice()
      shifted[index] = (shifted[index] ?? 0) + h
      const next = evaluate(shifted)
      if (!next) return null
      columns.push(next.map((item, row) => (item - (value[row] ?? 0)) / h))
    }
    const delta = solve2(columns[0]?.[0] ?? 0, columns[1]?.[0] ?? 0, columns[0]?.[1] ?? 0, columns[1]?.[1] ?? 0, value[0] ?? 0, value[1] ?? 0)
    if (!delta) return null
    let scale = 1
    let moved = xs
    for (let damp = 0; damp < 5; damp += 1) {
      moved = xs.map((item, index) => item - (delta[index] ?? 0) * scale)
      const trial = evaluate(moved)
      if (!trial) {
        scale *= 0.5
        continue
      }
      if (Math.hypot(trial[0] ?? 0, trial[1] ?? 0) <= error * 1.2) break
      scale *= 0.5
    }
    if (moved.some((item) => !Number.isFinite(item) || Math.abs(item) > 1e6)) return null
    xs = moved
  }
  const final = evaluate(xs)
  if (!final || Math.hypot(final[0] ?? 0, final[1] ?? 0) > 1e-6 || !insideBox(xs, boxes)) return null
  return xs
}

function insideBox(xs: number[], boxes: { min: number; max: number }[]): boolean {
  return xs.every((value, index) => {
    const box = boxes[index]
    if (!box) return false
    const slack = 1e-5 * Math.max(1, box.max - box.min)
    return value >= box.min - slack && value <= box.max + slack
  })
}

function solve2(a: number, b: number, c: number, d: number, fx: number, fy: number): [number, number] | null {
  const det = a * d - b * c
  if (!Number.isFinite(det) || Math.abs(det) < 1e-12) return null
  return [(d * fx - b * fy) / det, (a * fy - c * fx) / det]
}

export function evaluateCas(call: Expr, angles: AngleMode): CasVisual | null {
  if (call.type !== 'call') return null
  if (call.name === 'dsolve') {
    const value = normalize(solveDe(call.args, angles), angles)
    const variable = symbolOf(call.args[2], 'x')
    const plotted = substitute(value, new Map([['C', ONE], ['C1', ONE], ['C2', ONE]]))
    const curve = curveFor(normalize(plotted, angles), variable, 'solution')
    return { ...presentValue(value, angles, call, curve ? [curve] : []), warn: 'Graph uses C = 1.' }
  }
  if (call.name === 'idiff') return presentValue(implicitDerivative(call.args, angles), angles, call)
  if (!REDUCIBLE.has(call.name) && call.name !== 'factor' && call.name !== 'zeros' && call.name !== 'fmin' && call.name !== 'fmax') return null
  const args = call.args.map((arg) => rewriteAll(arg, angles))
  if (call.name === 'factor') {
    const arg = normalize(args[0], angles)
    const value = whole(arg, angles)
    let factored: Expr
    if (arg.type === 'rat' && arg.d !== 1n) factored = div(factorInteger(arg.n), factorInteger(arg.d))
    else if (value !== null) factored = factorInteger(value)
    else {
      const symbols = freeSymbols(arg)
      if (symbols.length !== 1) throw new MathError('Factor a whole number or a polynomial in one variable.')
      factored = factorPolynomial(arg, symbols[0], angles)
    }
    return { text: `${plain(call)} = ${plain(factored)}`, tex: tex(factored), curves: [], points: [], parametrics: [], arrows: [], warn: null }
  }
  if (call.name === 'zeros') {
    const variable = args[1] ? symbolOf(args[1], 'x') : (freeSymbols(args[0]).find((name) => name !== 'pi' && name !== 'e') ?? 'x')
    const roots = rationalRoots(args[0], variable, angles).filter((root, index, all) => all.findIndex((other) => plain(other) === plain(root)) === index)
    if (roots.length === 0) throw new MathError('No real roots on [-10, 10].')
    const pieces = roots.map((root) => `${variable} = ${plain(root)}`)
    const formula = roots.map((root) => `${tex(S(variable))} = ${tex(root)}`).join(' \\;\\text{or}\\; ')
    const points = roots.flatMap((root) => {
      const at = numericConstant(root, angles)
      if (at === null) return []
      const marker = variable === 'y' ? pointAt(0, at, `${variable} = ${plain(root)}`) : pointAt(at, 0, `${variable} = ${plain(root)}`)
      return marker ? [marker] : []
    })
    const curve = curveFor(normalize(args[0], angles), variable, 'curve', true)
    return { text: pieces.join(' or '), tex: formula, curves: curve ? [curve] : [], points, parametrics: [], arrows: [], warn: null }
  }
  if (call.name === 'fmin' || call.name === 'fmax') {
    const found = extremum(args, angles, call.name === 'fmin' ? 'min' : 'max')
    const word = call.name === 'fmin' ? 'minimum' : 'maximum'
    const display = present(found.value, angles)
    const at = present(found.at, angles)
    const atNumber = numericConstant(found.at, angles) ?? Number.NaN
    const height = numericConstant(found.value, angles) ?? Number.NaN
    const marker = found.variable === 'y' ? pointAt(height, atNumber, word) : pointAt(atNumber, height, word)
    const curve = curveFor(normalize(found.body, angles), found.variable, 'curve', true)
    return {
      text: `${word} ${display.text} at ${found.variable} = ${at.text}`,
      tex: `\\text{${word} }${display.tex}\\text{ at }${tex(S(found.variable))} = ${at.tex}`,
      curves: curve ? [curve] : [],
      points: marker ? [marker] : [],
      parametrics: [],
      arrows: [],
      warn: null,
    }
  }
  if (call.name === 'decimal') {
    const value = valueNamed('decimal', args, angles)
    const text = value.type === 'dec' ? value.text : plain(value)
    const shownAsTex = value.type === 'dec' ? texDecimal(value.text) : text
    return { text: `${plain(call)} = ${text}`, tex: shownAsTex, curves: [], points: [], parametrics: [], arrows: [], warn: null }
  }
  if (call.name === 'integrate' && args[0].type === 'vec') return vectorIntegral(args, angles, call)
  if (call.name === 'integrate' && args.length === 4) {
    const variable = symbolOf(args[1], 'x')
    const integrand = normalize(args[0], angles)
    const start = numericConstant(normalize(args[2], angles), angles)
    const end = numericConstant(normalize(args[3], angles), angles)
    const curves = shadedIntegrand(integrand, variable, start, end)
    try {
      return presentValue(reduceNamed('integrate', args, angles), angles, call, curves)
    } catch (error) {
      if (start === null || end === null) throw error
      const value = simpson(args[0], variable, start, end, angles)
      const text = decimalText(value)
      return { text: `${plain(call)} ≈ ${text}`, tex: texDecimal(text), curves, points: [], parametrics: [], arrows: [], warn: 'Decimal approximation.' }
    }
  }
  if (call.name === 'integrate') {
    const variable = symbolOf(args[1], 'x')
    const value = add([antiderivative(args[0], variable, angles), S('C')])
    const plotted = curveFor(normalize(antiderivative(args[0], variable, angles), angles), variable, 'integral')
    return { ...presentValue(value, angles, call, plotted ? [plotted] : []), warn: plotted ? 'Graph uses C = 0.' : null }
  }
  if (call.name === 'tangent' || call.name === 'normal') {
    const variable = symbolOf(args[1], 'x')
    const value = valueNamed(call.name, args, angles)
    const height = numericConstant(substitute(args[0], new Map([[variable, args[2]]])), angles)
    const at = numericConstant(normalize(args[2], angles), angles)
    const marker = at === null || height === null ? null : variable === 'y' ? pointAt(height, at, call.name) : pointAt(at, height, call.name)
    const curves = withOriginal(args[0], variable, [])
    if (value.type === 'eq') {
      const place = normalize(value.right, angles)
      if (variable === 'x') curves.unshift({ expr: place, along: 'y', label: call.name })
      else if (variable === 'y') curves.unshift({ expr: place, along: 'x', label: call.name })
    } else {
      const line = curveFor(normalize(value, angles), variable, call.name)
      if (line) curves.unshift(line)
    }
    return { ...presentValue(value, angles, call, curves, marker ? [marker] : []), warn: null }
  }
  const reduced = reduceNamed(call.name, args, angles)
  const simplified = normalize(reduced, angles)
  if (call.name === 'diff' || call.name === 'series') {
    const variable = symbolOf(args[1], 'x')
    const atPoint = call.name === 'diff' && diffValue(args, angles).at !== null
    const curve = atPoint ? null : curveFor(simplified, variable, call.name)
    const curves = call.name === 'series' ? withOriginal(args[0], variable, curve ? [curve] : []) : curve ? [curve] : []
    const graphics = atPoint ? { parametrics: [], arrows: [] } : vectorGraphics(simplified, variable, call.name)
    return { ...presentValue(simplified, angles, call, curves), ...graphics }
  }
  return presentValue(simplified, angles, call)
}

function shadedIntegrand(integrand: Expr, variable: string, start: number | null, end: number | null): CasCurve[] {
  const label = plain(integrand).length > 24 ? 'integrand' : plain(integrand)
  const curve = curveFor(integrand, variable, label)
  if (!curve) return []
  curve.exprKey = plain(integrand)
  if (start !== null && end !== null) curve.shade = { from: start, to: end }
  return [curve]
}

function vectorIntegral(args: Expr[], angles: AngleMode, source: Expr): CasVisual {
  const variable = symbolOf(args[1], 't')
  const body = args[0]
  if (body.type !== 'vec') throw new MathError('Integrate a vector one component at a time.')
  if (args.length === 4) {
    const parts = body.args.map((component) => reduceNamed('integrate', [component, args[1], args[2], args[3]], angles))
    const value = normalize({ type: 'vec', args: parts }, angles)
    const field = vectorGraphics(normalize(body, angles), variable, 'integrand')
    const result = vectorGraphics(value, variable, 'integral')
    return { ...presentValue(value, angles, source), parametrics: field.parametrics, arrows: result.arrows }
  }
  const antiderivatives = body.args.map((component) => antiderivative(component, variable, angles))
  const value = normalize({ type: 'vec', args: antiderivatives.map((component, index) => add([component, S(`C${index + 1}`)])) }, angles)
  const graphics = vectorGraphics(normalize({ type: 'vec', args: antiderivatives }, angles), variable, 'integral')
  const drawn = graphics.parametrics.length > 0 || graphics.arrows.length > 0
  return { ...presentValue(value, angles, source), ...graphics, warn: drawn ? 'Graph uses C = 0.' : null }
}

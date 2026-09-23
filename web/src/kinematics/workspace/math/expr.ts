// Small exact-rational kernel: parse, draw as TeX, evaluate, and solve
// linear or quadratic equations. Trig angles follow the active unit.

import { MATH_FUNCTION_NAMES } from './catalog'
import { containsAggregate, reduceAlgebra } from './linear'

export type AngleMode = 'rad' | 'deg'

export class MathError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'MathError'
  }
}

export type Expr =
  | { type: 'rat'; n: bigint; d: bigint }
  | { type: 'dec'; text: string; value: number }
  | { type: 'sym'; name: string; sub?: string; dots?: number }
  | { type: 'add'; args: Expr[] }
  | { type: 'mul'; args: Expr[] }
  | { type: 'div'; num: Expr; den: Expr }
  | { type: 'pow'; base: Expr; exp: Expr }
  | { type: 'call'; name: string; args: Expr[] }
  | { type: 'vec'; args: Expr[] }
  | { type: 'mat'; rows: Expr[][] }
  | { type: 'eq'; left: Expr; right: Expr }

export type Binding =
  | { kind: 'expr'; expr: Expr }
  | { kind: 'fn'; params: string[]; body: Expr }

export type MathEnv = Map<string, Binding>

type Atom =
  | { kind: 'var'; name: string; sub?: string; dots?: number; exp: number }
  | { kind: 'expr'; expr: Expr; exp: number }

interface Term {
  coeff: Rat
  atoms: Atom[]
}

interface Rat {
  n: bigint
  d: bigint
}

const BUILTIN_CALLS = new Set(['sqrt', 'ln', 'log', 'sin', 'cos', 'tan', 'abs', 'exp', 'asin', 'acos', 'atan'])
const CAS_CALLS = new Set(MATH_FUNCTION_NAMES)
const RESERVED = new Set([...BUILTIN_CALLS, ...CAS_CALLS, 'pi', 'e'])

const ONE: Rat = { n: 1n, d: 1n }
const ZERO_EXPR: Expr = { type: 'rat', n: 0n, d: 1n }

function rat(n: bigint, d: bigint = 1n): Expr {
  if (d === 0n) return { type: 'div', num: { type: 'rat', n, d: 1n }, den: ZERO_EXPR }
  if (d < 0n) {
    n = -n
    d = -d
  }
  const g = gcd(n < 0n ? -n : n, d)
  return { type: 'rat', n: n / g, d: d / g }
}

function asRat(e: Expr): Rat | null {
  if (e.type !== 'rat') return null
  return { n: e.n, d: e.d }
}

function ratExpr(r: Rat): Expr {
  return rat(r.n, r.d)
}

function gcd(a: bigint, b: bigint): bigint {
  let x = a < 0n ? -a : a
  let y = b < 0n ? -b : b
  while (y !== 0n) {
    const t = y
    y = x % y
    x = t
  }
  return x || 1n
}

function addRat(a: Rat, b: Rat): Rat {
  const n = a.n * b.d + b.n * a.d
  const d = a.d * b.d
  const out = asRat(rat(n, d))
  return out ?? ONE
}

function mulRat(a: Rat, b: Rat): Rat {
  const out = asRat(rat(a.n * b.n, a.d * b.d))
  return out ?? ONE
}

function divRat(a: Rat, b: Rat): Rat {
  const out = asRat(rat(a.n * b.d, a.d * b.n))
  return out ?? ONE
}

export function parseMathInput(input: string): MathInput {
  const raw = autoClose(input.trim())
  if (!raw) throw new MathError('Enter a calculation.')
  const fn2 = /^([A-Za-z][A-Za-z0-9]*)\s*\(\s*([A-Za-z][A-Za-z0-9]*)\s*,\s*([A-Za-z][A-Za-z0-9]*)\s*\)\s*=\s*([\s\S]+)$/.exec(raw)
  if (fn2) {
    assertDefinable(fn2[1])
    if (fn2[2] === fn2[3]) throw new MathError('Use two different inputs.')
    return { kind: 'fn', name: fn2[1], params: [fn2[2], fn2[3]], body: parseExpr(fn2[4]), raw }
  }
  const fn1 = /^([A-Za-z][A-Za-z0-9]*)\s*\(\s*([A-Za-z][A-Za-z0-9]*)\s*\)\s*=\s*([\s\S]+)$/.exec(raw)
  if (fn1) {
    assertDefinable(fn1[1])
    return { kind: 'fn', name: fn1[1], params: [fn1[2]], body: parseExpr(fn1[3]), raw }
  }
  const expr = parseExpr(raw)
  if (expr.type === 'eq' && expr.left.type === 'call') {
    throw new MathError('Define a curve as f(x) = … or a surface as f(x, y) = ….')
  }
  if (expr.type === 'eq' && expr.left.type === 'sym' && !RESERVED.has(expr.left.name)) {
    return { kind: 'assign', name: expr.left.name, expr: expr.right, raw }
  }
  if (expr.type === 'call' && expr.name === 'solve') return parseSolve(expr, raw)
  return { kind: 'expr', expr, raw }
}

export type MathInput =
  | { kind: 'fn'; name: string; params: string[]; body: Expr; raw: string }
  | { kind: 'assign'; name: string; expr: Expr; raw: string }
  | { kind: 'solve'; equation: Expr; variable: string | null; raw: string }
  | { kind: 'system'; equations: Expr[]; raw: string }
  | { kind: 'expr'; expr: Expr; raw: string }

function assertDefinable(name: string): void {
  if (RESERVED.has(name)) throw new MathError(`${name} is built in.`)
}

function parseSolve(expr: Expr, raw: string): MathInput {
  if (expr.type !== 'call') throw new MathError('Use solve(equation) or solve(equation, x).')
  if (expr.args.length >= 2 && expr.args.every((arg) => arg.type === 'eq')) return { kind: 'system', equations: expr.args, raw }
  if (expr.args.length === 1) return { kind: 'solve', equation: expr.args[0], variable: null, raw }
    if (expr.args.length === 2 && expr.args[1].type === 'sym') {
    if (bareConstant(expr.args[1])) throw new MathError(`${expr.args[1].name} is a constant.`)
    return { kind: 'solve', equation: expr.args[0], variable: expr.args[1].name, raw }
  }
  throw new MathError('Use solve(equation), solve(equation, x), or solve(eq1, eq2).')
}

function autoClose(input: string): string {
  let balance = 0
  for (const ch of input) {
    if (ch === '(') balance += 1
    else if (ch === ')') balance -= 1
    if (balance < 0) return input
  }
  if (balance > 0) return input + ')'.repeat(balance)
  return input
}

export function parseExpr(input: string): Expr {
  const parser = new Parser(input.trim())
  const expr = parser.parseEquation()
  parser.skip()
  if (parser.i < parser.src.length) throw new MathError('Could not read that. Check the operators and parentheses.')
  return expr
}

class Parser {
  src: string
  i = 0

  constructor(src: string) {
    this.src = src
  }

  skip(): void {
    while (this.i < this.src.length && /\s/.test(this.src[this.i])) this.i += 1
  }

  peek(): string {
    this.skip()
    return this.src[this.i] ?? ''
  }

  eat(ch: string): boolean {
    if (this.peek() !== ch) return false
    this.i += 1
    return true
  }

  parseEquation(): Expr {
    const left = this.parseSum()
    if (!this.eat('=')) return left
    const right = this.parseSum()
    return { type: 'eq', left, right }
  }

  parseSum(): Expr {
    let left = this.parseProduct()
    for (;;) {
      if (this.eat('+')) left = { type: 'add', args: [left, this.parseProduct()] }
      else if (this.eat('-')) left = { type: 'add', args: [left, { type: 'mul', args: [rat(-1n), this.parseProduct()] }] }
      else break
    }
    return left
  }

  parseProduct(): Expr {
    let left = this.parseUnary()
    for (;;) {
      if (this.eat('*')) {
        left = { type: 'mul', args: [left, this.parseUnary()] }
        continue
      }
      if (this.eat('/')) {
        left = { type: 'div', num: left, den: this.parseUnary() }
        continue
      }
      if (!this.startsImplicit()) break
      const before = this.i
      const right = this.parseUnary()
      if (this.i === before) break
      left = { type: 'mul', args: [left, right] }
    }
    return left
  }

  canStartPrimary(): boolean {
    const c = this.peek()
    return c === '(' || c === '[' || /[A-Za-z0-9.]/.test(c) || greekCharName(c) !== null
  }

  startsImplicit(): boolean {
    const c = this.peek()
    if (c === '(' || c === '[' || /[A-Za-z]/.test(c) || greekCharName(c) !== null) return true
    if (/[0-9.]/.test(c)) return this.prevNonSpace() === ')'
    return false
  }

  prevNonSpace(): string {
    let j = this.i - 1
    while (j >= 0 && /\s/.test(this.src[j])) j -= 1
    return this.src[j] ?? ''
  }

  parseUnary(): Expr {
    if (this.eat('+')) return this.parseUnary()
    if (this.eat('-')) return { type: 'mul', args: [rat(-1n), this.parseUnary()] }
    return this.consumePrimes(this.parsePower())
  }

  parsePower(): Expr {
    const base = this.parsePostfix(this.parsePrimary())
    if (!this.eat('^')) return base
    return { type: 'pow', base, exp: this.parseUnary() }
  }

  /** Primes and underscores bind to the name just typed: theta'^2 is (theta-dot)^2, and e_theta is e with a subscript. */
  parsePostfix(expr: Expr): Expr {
    let current = expr
    for (;;) {
      if (this.eat("'")) {
        current = this.withPrime(current)
        continue
      }
      if (this.peek() === '_') {
        current = this.withSubscript(current, this.parseSubscript())
        continue
      }
      break
    }
    return current
  }

  /** A prime after a finished power wraps that power: theta^2' is d/dt of theta^2. */
  consumePrimes(expr: Expr): Expr {
    let current = expr
    while (this.eat("'")) current = this.withPrime(current)
    return current
  }

  withPrime(expr: Expr): Expr {
    if (expr.type === 'sym') return { ...expr, dots: (expr.dots ?? 0) + 1 }
    return { type: 'call', name: 'Dt', args: [expr] }
  }

  withSubscript(expr: Expr, sub: string): Expr {
    if (expr.type !== 'sym') throw new MathError('Use _ after a name, for example e_r or theta_0.')
    if (expr.sub) throw new MathError('That name already has a subscript.')
    return { ...expr, sub }
  }

  parseSubscript(): string {
    if (!this.eat('_')) throw new MathError('Use _ after a name, for example e_r or theta_0.')
    if (this.eat('{')) {
      const start = this.i
      while (this.i < this.src.length && this.src[this.i] !== '}') this.i += 1
      const body = this.src.slice(start, this.i).trim()
      if (!this.eat('}')) throw new MathError('Close the subscript with }.')
      return subscriptName(body)
    }
    const ch = this.src[this.i] ?? ''
    const fromChar = greekCharName(ch)
    if (fromChar) {
      this.i += 1
      return fromChar
    }
    if (ch === '?') {
      this.i += 1
      return '?'
    }
    if (!/[A-Za-z0-9]/.test(ch)) throw new MathError('Use _ after a name, for example e_r or theta_0.')
    const start = this.i
    while (this.i < this.src.length && /[A-Za-z0-9]/.test(this.src[this.i] ?? '')) this.i += 1
    return subscriptName(this.src.slice(start, this.i))
  }

  parsePrimary(): Expr {
    const c = this.peek()
    if (!c) throw new MathError('Could not read that. Check the operators and parentheses.')
    if (/[0-9.]/.test(c)) return this.parseNumber()
    if (greekCharName(c) || /[A-Za-z]/.test(c)) {
      const ident = this.parseIdent()
      if (this.peek() === '(') {
        this.i += 1
        const args: Expr[] = []
        if (this.peek() !== ')') {
          args.push(this.parseEquation())
          while (this.eat(',')) args.push(this.parseEquation())
        }
        if (!this.eat(')')) throw new MathError('Could not read that. Check the operators and parentheses.')
        checkCall(ident.name, args)
        return { type: 'call', name: ident.name, args }
      }
      if (BUILTIN_CALLS.has(ident.name) && this.canStartPrimary()) return { type: 'call', name: ident.name, args: [this.parseProduct()] }
      return { type: 'sym', name: ident.name }
    }
    if (c === '?') {
      this.i += 1
      return { type: 'sym', name: '?' }
    }
    if (c === '[') return this.parseBracket()
    if (c === '(') {
      this.i += 1
      const inner = this.parseSum()
      if (this.peek() === ',') {
        const args = [inner]
        while (this.eat(',')) args.push(this.parseSum())
        if (!this.eat(')')) throw new MathError('Could not read that. Check the operators and parentheses.')
        return { type: 'vec', args }
      }
      if (!this.eat(')')) throw new MathError('Could not read that. Check the operators and parentheses.')
      return inner
    }
    throw new MathError('Could not read that. Check the operators and parentheses.')
  }

  parseBracket(): Expr {
    if (!this.eat('[')) throw new MathError('Could not read that vector.')
    if (this.peek() === '[') {
      const rows: Expr[][] = []
      while (this.peek() === '[') {
        rows.push(this.parseRow())
        if (!this.eat(',')) break
      }
      if (!this.eat(']')) throw new MathError('Close the matrix with ].')
      const width = rows[0]?.length ?? 0
      if (rows.length === 0 || width === 0 || rows.some((row) => row.length !== width)) throw new MathError('Matrix rows should be the same length.')
      return { type: 'mat', rows }
    }
    if (this.eat(']')) return { type: 'vec', args: [] }
    const args = [this.parseSum()]
    while (this.eat(',')) args.push(this.parseSum())
    if (!this.eat(']')) throw new MathError('Close the vector with ].')
    return { type: 'vec', args }
  }

  parseRow(): Expr[] {
    if (!this.eat('[')) throw new MathError('A matrix row starts with [.')
    if (this.eat(']')) return []
    const items = [this.parseSum()]
    while (this.eat(',')) items.push(this.parseSum())
    if (!this.eat(']')) throw new MathError('Close the matrix row with ].')
    return items
  }

  parseIdent(): { name: string } {
    const fromChar = greekCharName(this.src[this.i] ?? '')
    if (fromChar) {
      this.i += 1
      return { name: fromChar }
    }
    const start = this.i
    this.i += 1
    while (this.i < this.src.length && /[A-Za-z0-9]/.test(this.src[this.i])) this.i += 1
    return { name: canonicalGreek(this.src.slice(start, this.i)) }
  }

  parseNumber(): Expr {
    const m = /^(\d+\.?\d*|\.\d+)(?:e([+-]?\d+))?/i.exec(this.src.slice(this.i))
    if (!m) throw new MathError('Could not read that number.')
    this.i += m[0].length
    const exp = m[2] ? Number(m[2]) : 0
    if (!Number.isFinite(exp)) throw new MathError('Could not read that number.')
    const [whole, frac = ''] = m[1].split('.')
    const digits = `${whole}${frac}`.replace(/^0+(?=\d)/, '') || '0'
    const scale = frac.length - exp
    if (scale >= 0) return rat(BigInt(digits), 10n ** BigInt(scale))
    return rat(BigInt(digits) * 10n ** BigInt(-scale), 1n)
  }
}

/** Turn a half-typed line into something the preview can draw: `x^` and `sqrt(` become placeholders. */
function cookPreview(input: string): string {
  let source = input.trim()
  const stack: string[] = []
  const closer: Record<string, string> = { '(': ')', '[': ']', '{': '}' }
  const opener: Record<string, string> = { ')': '(', ']': '[', '}': '{' }
  for (const ch of source) {
    if (ch === '(' || ch === '[' || ch === '{') stack.push(ch)
    else if (ch === ')' || ch === ']' || ch === '}') {
      const open = stack.pop()
      if (open !== opener[ch]) return source
    }
  }
  while (stack.length > 0) {
    const open = stack.pop()
    if (open) source += closer[open]
  }
  source = source.replace(/\(\s*\)/g, '(?)')
  source = source.replace(/\[\s*\]/g, '[?]')
  source = source.replace(/\{\s*\}/g, '{?}')
  if (/[+\-*/^,]$/.test(source)) source += '?'
  else if (/_$/.test(source)) source += '?'
  else if (/=\s*$/.test(source)) source += '?'
  return source
}

export function previewTex(input: string): string | null {
  try {
    const parsed = parseMathInput(cookPreview(input))
    if (parsed.kind === 'fn') {
      const params = parsed.params.join(', ')
      return `${parsed.name}\\left(${params}\\right) = ${tex(parsed.body)}`
    }
    if (parsed.kind === 'assign') return `${texSymbol(parsed.name)} = ${tex(parsed.expr)}`
    if (parsed.kind === 'solve') {
      const eq = parsed.equation.type === 'eq' ? parsed.equation : { type: 'eq' as const, left: parsed.equation, right: ZERO_EXPR }
      return `${tex(eq.left)} = ${tex(eq.right)}`
    }
    if (parsed.kind === 'system') {
      return parsed.equations
        .map((eq) => (eq.type === 'eq' ? `${tex(eq.left)} = ${tex(eq.right)}` : tex(eq)))
        .join(', ')
    }
    return tex(parsed.expr)
  } catch {
    return null
  }
}

export function validateMath(input: string): string | null {
  try {
    parseMathInput(input)
    return null
  } catch (error) {
    return error instanceof MathError ? error.message : 'Could not read that.'
  }
}

export function normalize(e: Expr, angles: AngleMode = 'rad'): Expr {
  const folded = fold(e, angles)
  if (!containsAggregate(folded)) return fromTerms(toSum(folded))
  const reduced = reduceAlgebra(folded, angles, normalize)
  if (reduced.type === 'vec' || reduced.type === 'mat' || containsAggregate(reduced)) return reduced
  return fromTerms(toSum(fold(reduced, angles)))
}

export function present(e: Expr, angles: AngleMode = 'rad'): { tex: string; text: string } {
  const unknown = firstUnknown(e)
  if (unknown) throw new MathError(`${unknown} is not defined.`)
  if (e.type === 'sym' && (e.name === 'e' || e.name === 'pi')) {
    const n = evalConst(e, angles)
    if (n === null) return { tex: tex(e), text: plain(e) }
    const dec: Expr = { type: 'dec', text: trimNum(n), value: n }
    return { tex: tex(dec), text: plain(dec) }
  }
  if (hasFreeSymbol(e) || keepSymbolic(e) || containsConstantSym(e)) {
    const n = evalConst(e, angles)
    if (!hasFreeSymbol(e) && !keepSymbolic(e) && n !== null && Number.isFinite(n)) {
      const snapped = snap(n)
      if (snapped) return { tex: tex(snapped), text: plain(snapped) }
    }
    if (!hasFreeSymbol(e) && n !== null && !Number.isFinite(n)) throw new MathError('Not a real number.')
    return { tex: tex(e), text: plain(e) }
  }
  const n = evalConst(e, angles)
  if (n === null) return { tex: tex(e), text: plain(e) }
  if (!Number.isFinite(n)) throw new MathError('Not a real number.')
  const snapped = snap(n)
  if (snapped) return { tex: tex(snapped), text: plain(snapped) }
  const dec: Expr = { type: 'dec', text: trimNum(n), value: n }
  return { tex: tex(dec), text: plain(dec) }
}

/** A decimal for a fully numeric value. Radicals that did not simplify still have one. */
export function approximate(e: Expr, angles: AngleMode = 'rad'): { tex: string; text: string } | null {
  if (hasFreeSymbol(e)) return null
  const n = evalConst(e, angles)
  if (n === null || !Number.isFinite(n)) return null
  const text = trimNum(n)
  return { tex: text, text }
}

export function numericConstant(e: Expr, angles: AngleMode = 'rad'): number | null {
  try {
    return evalConst(e, angles)
  } catch {
    return null
  }
}

export function polynomialCoefficients(e: Expr, variable: string): Expr[] | null {
  return toPoly(e, variable)
}

export function numericValue(e: Expr, env: MathEnv, angles: AngleMode = 'rad'): number | null {
  try {
    const n = evalConst(applyEnv(e, env, 0), angles)
    if (n === null || !Number.isFinite(n)) return null
    return n
  } catch {
    return null
  }
}

export function applyEnv(e: Expr, env: MathEnv, depth: number): Expr {
  if (depth > 24) throw new MathError('That calculation repeats without ending.')
  switch (e.type) {
    case 'rat':
    case 'dec':
      return e
    case 'sym': {
      if (e.sub || (e.dots ?? 0) > 0) return e
      const binding = env.get(e.name)
      if (binding?.kind === 'expr') return applyEnv(binding.expr, env, depth + 1)
      return e
    }
    case 'add':
    case 'mul':
      return { ...e, args: e.args.map((arg) => applyEnv(arg, env, depth)) }
    case 'div':
      return { type: 'div', num: applyEnv(e.num, env, depth), den: applyEnv(e.den, env, depth) }
    case 'pow':
      return { type: 'pow', base: applyEnv(e.base, env, depth), exp: applyEnv(e.exp, env, depth) }
    case 'eq':
      return { type: 'eq', left: applyEnv(e.left, env, depth), right: applyEnv(e.right, env, depth) }
    case 'call': {
      const args = e.args.map((arg) => applyEnv(arg, env, depth))
      const binding = env.get(e.name)
      if (binding?.kind === 'fn') {
        if (binding.params.length !== args.length) {
          const count = binding.params.length
          throw new MathError(`${e.name} takes ${count} input${count === 1 ? '' : 's'}.`)
        }
        const map = new Map(binding.params.map((param, index) => [param, args[index]]))
        return applyEnv(substitute(binding.body, map), env, depth + 1)
      }
      return { type: 'call', name: e.name, args }
    }
    case 'vec':
      return { type: 'vec', args: e.args.map((arg) => applyEnv(arg, env, depth)) }
    case 'mat':
      return { type: 'mat', rows: e.rows.map((row) => row.map((arg) => applyEnv(arg, env, depth))) }
  }
}

export function freeSymbols(e: Expr): string[] {
  const names = new Set<string>()
  walk(e, (node) => {
    if (node.type === 'sym' && !bareConstant(node)) names.add(symbolKey(node))
  })
  return [...names]
}

export function solveEquation(equation: Expr, variable: string | null): { tex: string; text: string } {
  const zero = equation.type === 'eq'
    ? normalize({ type: 'add', args: [equation.left, { type: 'mul', args: [rat(-1n), equation.right] }] })
    : normalize(equation)
  const symbols = freeSymbols(zero)
  let name = variable
  if (!name) {
    if (symbols.length === 1) name = symbols[0]
    else if (symbols.includes('x')) name = 'x'
    else if (symbols.length === 0) {
      if (isZeroExpr(zero)) return { tex: '\\text{true}', text: 'true' }
      return { tex: '\\text{no solution}', text: 'no solution' }
    } else {
      throw new MathError('Say which variable to solve for, for example solve(x + y = 3, x).')
    }
  }
  const poly = toPoly(zero, name)
  if (!poly) throw new MathError('Cannot solve that algebraically yet. It can be linear or quadratic.')
  const deg = polyDegree(poly)
  if (deg > 2) throw new MathError('Cannot solve that algebraically yet. It can be linear or quadratic.')
  if (deg === 0) {
    if (isZeroExpr(poly[0] ?? ZERO_EXPR)) return { tex: `\\text{true for every }${texSymbol(name)}`, text: `true for every ${name}` }
    return { tex: '\\text{no solution}', text: 'no solution' }
  }
  if (deg === 1) {
    const root = normalize({ type: 'div', num: { type: 'mul', args: [rat(-1n), poly[0] ?? ZERO_EXPR] }, den: poly[1] ?? ZERO_EXPR })
    return { tex: `${texSymbol(name)} = ${tex(root)}`, text: `${name} = ${plain(root)}` }
  }
  const a = poly[2] ?? ZERO_EXPR
  const b = poly[1] ?? ZERO_EXPR
  const c = poly[0] ?? ZERO_EXPR
  const disc = normalize({
    type: 'add',
    args: [
      { type: 'pow', base: b, exp: rat(2n) },
      { type: 'mul', args: [rat(-4n), a, c] },
    ],
  })
  if (disc.type === 'rat' && disc.n < 0n) return { tex: `\\text{no real solution for }${texSymbol(name)}`, text: `no real solution for ${name}` }
  const radical = sqrtOf(disc)
  const twoA = normalize({ type: 'mul', args: [rat(2n), a] })
  const negB = normalize({ type: 'mul', args: [rat(-1n), b] })
  const plus = normalize({ type: 'div', num: { type: 'add', args: [negB, radical] }, den: twoA })
  const minus = normalize({ type: 'div', num: { type: 'add', args: [negB, { type: 'mul', args: [rat(-1n), radical] }] }, den: twoA })
  if (exprKey(plus) === exprKey(minus)) return { tex: `${texSymbol(name)} = ${tex(plus)}`, text: `${name} = ${plain(plus)}` }
  return {
    tex: `${texSymbol(name)} = ${tex(plus)} \\;\\text{or}\\; ${texSymbol(name)} = ${tex(minus)}`,
    text: `${name} = ${plain(plus)} or ${name} = ${plain(minus)}`,
  }
}

function sqrtOf(e: Expr): Expr {
  if (e.type === 'rat') {
    if (e.n < 0n) throw new MathError('No real solution.')
    return exactSqrt(e)
  }
  return { type: 'call', name: 'sqrt', args: [e] }
}

export function substitute(e: Expr, map: Map<string, Expr>): Expr {
  switch (e.type) {
    case 'sym':
      if (e.sub || (e.dots ?? 0) > 0) return e
      return map.get(e.name) ?? e
    case 'rat':
    case 'dec':
      return e
    case 'add':
    case 'mul':
      return { ...e, args: e.args.map((arg) => substitute(arg, map)) }
    case 'div':
      return { type: 'div', num: substitute(e.num, map), den: substitute(e.den, map) }
    case 'pow':
      return { type: 'pow', base: substitute(e.base, map), exp: substitute(e.exp, map) }
    case 'call':
      return { type: 'call', name: e.name, args: e.args.map((arg) => substitute(arg, map)) }
    case 'vec':
      return { type: 'vec', args: e.args.map((arg) => substitute(arg, map)) }
    case 'mat':
      return { type: 'mat', rows: e.rows.map((row) => row.map((arg) => substitute(arg, map))) }
    case 'eq':
      return { type: 'eq', left: substitute(e.left, map), right: substitute(e.right, map) }
  }
}

function fold(e: Expr, angles: AngleMode = 'rad'): Expr {
  switch (e.type) {
    case 'rat':
      return rat(e.n, e.d)
    case 'dec':
    case 'sym':
      return e
    case 'add': {
      const args = e.args.map((arg) => fold(arg, angles))
      if (args.every((arg) => arg.type === 'rat')) {
        return args.reduce<Expr>((acc, arg) => {
          if (acc.type !== 'rat' || arg.type !== 'rat') return arg
          return rat(acc.n * arg.d + arg.n * acc.d, acc.d * arg.d)
        }, rat(0n))
      }
      return { type: 'add', args }
    }
    case 'mul': {
      const args = e.args.map((arg) => fold(arg, angles))
      if (args.every((arg) => arg.type === 'rat')) {
        return args.reduce<Expr>((acc, arg) => {
          if (acc.type !== 'rat' || arg.type !== 'rat') return arg
          return rat(acc.n * arg.n, acc.d * arg.d)
        }, rat(1n))
      }
      return { type: 'mul', args }
    }
    case 'div': {
      const num = fold(e.num, angles)
      const den = fold(e.den, angles)
      if (num.type === 'rat' && den.type === 'rat' && den.n !== 0n) return rat(num.n * den.d, num.d * den.n)
      return { type: 'div', num, den }
    }
    case 'pow': {
      const base = fold(e.base, angles)
      const exp = fold(e.exp, angles)
      if (exp.type === 'rat' && exp.n === 0n) return rat(1n)
      if (exp.type === 'rat' && exp.n === 1n && exp.d === 1n) return base
      if (base.type === 'rat' && exp.type === 'rat' && exp.d === 1n) return powRat(base, exp.n)
      if (base.type === 'rat' && exp.type === 'rat' && exp.n === 1n && exp.d === 2n) return exactSqrt(base)
      return snapConstant({ type: 'pow', base, exp }, angles) ?? { type: 'pow', base, exp }
    }
    case 'call': {
      const args = e.args.map((arg) => fold(arg, angles))
      if (e.name === 'sqrt' && args.length === 1 && args[0].type === 'rat') return exactSqrt(args[0])
      if (e.name === 'ln' && args.length === 1 && args[0].type === 'sym' && bareConstant(args[0]) && args[0].name === 'e') return rat(1n)
      if (e.name === 'log' && args.length === 1 && args[0].type === 'rat') {
        const exact = log10Rat(args[0])
        if (exact) return exact
      }
      if (angles === 'deg' && args.length === 1 && (e.name === 'sin' || e.name === 'cos' || e.name === 'tan')) {
        const exact = exactDegreeTrig(e.name, args[0])
        if (exact) return fold(exact, 'rad')
      }
      return snapConstant({ type: 'call', name: e.name, args }, angles) ?? { type: 'call', name: e.name, args }
    }
    case 'vec':
      return { type: 'vec', args: e.args.map((arg) => fold(arg, angles)) }
    case 'mat':
      return { type: 'mat', rows: e.rows.map((row) => row.map((arg) => fold(arg, angles))) }
    case 'eq':
      return { type: 'eq', left: fold(e.left, angles), right: fold(e.right, angles) }
  }
}

function snapConstant(e: Expr, angles: AngleMode): Expr | null {
  if (hasFreeSymbol(e) || keepSymbolic(e) || containsConstantSym(e)) {
    const n = evalConst(e, angles)
    if (n !== null && Number.isFinite(n)) return snap(n)
    return null
  }
  const n = evalConst(e, angles)
  if (n === null || !Number.isFinite(n)) return null
  return snap(n)
}

const DEGREE_MARKS = [0, 30, 45, 60, 90, 120, 135, 150, 180, 210, 225, 240, 270, 300, 315, 330]

function exactDegreeTrig(name: 'sin' | 'cos' | 'tan', angle: Expr): Expr | null {
  if (angle.type !== 'rat') return null
  let n = angle.n % (360n * angle.d)
  if (n < 0n) n += 360n * angle.d
  const match = DEGREE_MARKS.find((deg) => n === BigInt(deg) * angle.d)
  if (match === undefined) return null
  if (name === 'tan' && (match === 90 || match === 270)) throw new MathError('Not a real number.')
  return name === 'sin' ? sinDegrees(match) : name === 'cos' ? cosDegrees(match) : tanDegrees(match)
}

function halfRoot(k: 2 | 3): Expr {
  return { type: 'div', num: { type: 'call', name: 'sqrt', args: [rat(BigInt(k))] }, den: rat(2n) }
}

function negExpr(e: Expr): Expr {
  return { type: 'mul', args: [rat(-1n), e] }
}

function sinDegrees(deg: number): Expr {
  switch (deg) {
    case 0:
    case 180:
      return rat(0n)
    case 30:
    case 150:
      return rat(1n, 2n)
    case 45:
    case 135:
      return halfRoot(2)
    case 60:
    case 120:
      return halfRoot(3)
    case 90:
      return rat(1n)
    case 210:
    case 330:
      return negExpr(rat(1n, 2n))
    case 225:
    case 315:
      return negExpr(halfRoot(2))
    case 240:
    case 300:
      return negExpr(halfRoot(3))
    default:
      return rat(-1n)
  }
}

function cosDegrees(deg: number): Expr {
  switch (deg) {
    case 0:
      return rat(1n)
    case 30:
    case 330:
      return halfRoot(3)
    case 45:
    case 315:
      return halfRoot(2)
    case 60:
    case 300:
      return rat(1n, 2n)
    case 90:
    case 270:
      return rat(0n)
    case 120:
    case 240:
      return negExpr(rat(1n, 2n))
    case 135:
    case 225:
      return negExpr(halfRoot(2))
    case 150:
    case 210:
      return negExpr(halfRoot(3))
    default:
      return rat(-1n)
  }
}

function tanDegrees(deg: number): Expr {
  const root3: Expr = { type: 'call', name: 'sqrt', args: [rat(3n)] }
  const over3: Expr = { type: 'div', num: root3, den: rat(3n) }
  switch (deg) {
    case 0:
    case 180:
      return rat(0n)
    case 30:
    case 210:
      return over3
    case 45:
    case 225:
      return rat(1n)
    case 60:
    case 240:
      return root3
    case 120:
    case 300:
      return negExpr(root3)
    case 135:
    case 315:
      return rat(-1n)
    case 150:
    case 330:
      return negExpr(over3)
    default:
      return rat(0n)
  }
}

function log10Rat(r: Expr): Expr | null {
  if (r.type !== 'rat' || r.n <= 0n || r.d !== 1n) return null
  let n = r.n
  let k = 0n
  while (n % 10n === 0n) {
    n /= 10n
    k += 1n
  }
  if (n === 1n) return rat(k)
  return null
}

function powRat(base: Expr, exp: bigint): Expr {
  if (base.type !== 'rat') return base
  if (exp < 0n) {
    if (base.n === 0n) return { type: 'div', num: rat(1n), den: ZERO_EXPR }
    return rat(base.d ** -exp, base.n ** -exp)
  }
  return rat(base.n ** exp, base.d ** exp)
}

function exactSqrt(r: Expr): Expr {
  if (r.type !== 'rat') return { type: 'call', name: 'sqrt', args: [r] }
  if (r.n < 0n) return { type: 'call', name: 'sqrt', args: [r] }
  const reduced = asRat(rat(r.n, r.d)) ?? { n: r.n, d: r.d }
  const num = squareParts(reduced.n)
  const den = squareParts(reduced.d)
  const inside = num.rest * den.rest
  const coeff = asRat(rat(num.root, den.root * den.rest)) ?? ONE
  if (inside === 1n) return ratExpr(coeff)
  const radical: Expr = { type: 'call', name: 'sqrt', args: [rat(inside)] }
  if (coeff.n === 1n && coeff.d === 1n) return radical
  return { type: 'mul', args: [ratExpr(coeff), radical] }
}

function squareParts(n: bigint): { root: bigint; rest: bigint } {
  if (n < 0n) throw new MathError('Not a real number.')
  if (n === 0n) return { root: 0n, rest: 1n }
  let root = 1n
  let free = 1n
  let rest = n
  let factor = 2n
  while (factor * factor <= rest) {
    let count = 0n
    while (rest % factor === 0n) {
      rest /= factor
      count += 1n
    }
    if (count > 0n) {
      root *= factor ** (count / 2n)
      if (count % 2n === 1n) free *= factor
    }
    factor = factor === 2n ? 3n : factor + 2n
  }
  if (rest > 1n) free *= rest
  return { root, rest: free }
}

function snap(n: number): Expr | null {
  if (!Number.isFinite(n)) return null
  const nearest = Math.round(n)
  if (Math.abs(n - nearest) <= 1e-9 * Math.max(1, Math.abs(n))) return rat(BigInt(nearest))
  for (let d = 2; d <= 12; d += 1) {
    const num = Math.round(n * d)
    if (Math.abs(n - num / d) <= 1e-8 * Math.max(1, Math.abs(n))) return rat(BigInt(num), BigInt(d))
  }
  return null
}

function trimNum(n: number): string {
  if (Object.is(n, -0) || n === 0) return '0'
  const abs = Math.abs(n)
  const text = abs >= 1e6 || abs < 1e-4 ? n.toExponential(4) : n.toPrecision(8)
  return text.replace(/(\.\d*?)0+(e|$)/, '$1$2').replace(/\.(e|$)/, '$1')
}

function toSum(e: Expr): Term[] {
  switch (e.type) {
    case 'rat': {
      const coeff = asRat(rat(e.n, e.d)) ?? ONE
      if (coeff.n === 0n) return []
      return [{ coeff, atoms: [] }]
    }
    case 'dec':
    case 'sym':
      return [{ coeff: ONE, atoms: [atomOf(e)] }]
    case 'add':
      return mergeTerms(e.args.flatMap(toSum))
    case 'mul':
      return e.args.reduce<Term[]>((acc, arg) => mulSums(acc, toSum(arg)), [{ coeff: ONE, atoms: [] }])
    case 'div': {
      const num = toSum(e.num)
      const den = toSum(e.den)
      if (den.length === 1 && den[0].coeff.n !== 0n) {
        const factor = den[0]
        if (exprKey(fromTerms(num)) === exprKey(fromTerms(den))) return [{ coeff: ONE, atoms: [] }]
        const inv: Term = {
          coeff: divRat(ONE, factor.coeff),
          atoms: factor.atoms.map((atom) => ({ ...atom, exp: -atom.exp })),
        }
        return mulSums(num, [inv])
      }
      return [{ coeff: ONE, atoms: [{ kind: 'expr', expr: e, exp: 1 }] }]
    }
    case 'pow': {
      if (e.exp.type === 'rat' && e.exp.d === 1n && e.exp.n >= 0n && e.exp.n <= 12n) {
        let acc: Term[] = [{ coeff: ONE, atoms: [] }]
        const base = toSum(e.base)
        for (let i = 0n; i < e.exp.n; i += 1n) {
          acc = mulSums(acc, base)
          if (acc.length > 400) return [{ coeff: ONE, atoms: [{ kind: 'expr', expr: e, exp: 1 }] }]
        }
        return acc
      }
      if (e.base.type === 'sym' && e.exp.type === 'rat' && e.exp.d === 1n && e.exp.n > -12n && e.exp.n < 12n) {
        return [{ coeff: ONE, atoms: [{ kind: 'var', name: e.base.name, sub: e.base.sub, dots: e.base.dots, exp: Number(e.exp.n) }] }]
      }
      return [{ coeff: ONE, atoms: [atomOf(e)] }]
    }
    case 'call':
    case 'eq':
    case 'vec':
    case 'mat':
      return [{ coeff: ONE, atoms: [atomOf(e)] }]
  }
}

function atomOf(e: Expr): Atom {
  if (e.type === 'sym') return { kind: 'var', name: e.name, sub: e.sub, dots: e.dots, exp: 1 }
  return { kind: 'expr', expr: e, exp: 1 }
}

function mulSums(a: Term[], b: Term[]): Term[] {
  if (a.length === 0 || b.length === 0) return []
  const out: Term[] = []
  for (const left of a) {
    for (const right of b) {
      out.push({
        coeff: mulRat(left.coeff, right.coeff),
        atoms: [...left.atoms, ...right.atoms],
      })
    }
  }
  return mergeTerms(out)
}

function mergeTerms(terms: Term[]): Term[] {
  const map = new Map<string, Term>()
  for (const term of terms) {
    const atoms = mergeAtoms(term.atoms)
    const key = atoms.map(atomKey).join('*')
    const prev = map.get(key)
    if (!prev) map.set(key, { coeff: term.coeff, atoms })
    else prev.coeff = addRat(prev.coeff, term.coeff)
  }
  return [...map.values()].filter((term) => term.coeff.n !== 0n)
}

function mergeAtoms(atoms: Atom[]): Atom[] {
  const map = new Map<string, Atom>()
  for (const atom of atoms) {
    const key = atom.kind === 'var' ? `v:${atom.name}:${atom.sub ?? ''}:${atom.dots ?? 0}` : `e:${exprKey(atom.expr)}`
    const prev = map.get(key)
    if (!prev) map.set(key, { ...atom })
    else prev.exp += atom.exp
  }
  return [...map.values()].filter((atom) => atom.exp !== 0).sort((a, b) => atomKey(a).localeCompare(atomKey(b)))
}

function atomKey(atom: Atom): string {
  if (atom.kind === 'var') return `v:${atom.name}:${atom.sub ?? ''}:${atom.dots ?? 0}^${atom.exp}`
  return `e:${exprKey(atom.expr)}^${atom.exp}`
}

function fromTerms(terms: Term[]): Expr {
  const merged = mergeTerms(terms)
  if (merged.length === 0) return ZERO_EXPR
  const sorted = merged.sort((a, b) => degreeOf(b) - degreeOf(a) || atomListKey(a).localeCompare(atomListKey(b)))
  const args = sorted.map(termToExpr)
  const expr: Expr = args.length === 1 ? args[0] : { type: 'add', args }
  return preferConstantFirst(expr)
}

function degreeOf(term: Term): number {
  return term.atoms.reduce((sum, atom) => sum + Math.abs(atom.exp), 0)
}

function atomListKey(term: Term): string {
  return term.atoms.map(atomKey).join('*')
}

function requireSymbol(arg: Expr | undefined, example: string): void {
  if (!arg || arg.type !== 'sym' || arg.sub || (arg.dots ?? 0) > 0 || arg.name === 'pi' || arg.name === 'e' || arg.name === '?') throw new MathError(example)
}

function bareConstant(e: { name: string; sub?: string; dots?: number }): boolean {
  return !e.sub && !(e.dots && e.dots > 0) && (e.name === 'pi' || e.name === 'e')
}

function symbolKey(e: { name: string; sub?: string; dots?: number }): string {
  const base = e.sub ? `${e.name}_${e.sub}` : e.name
  return base + "'".repeat(e.dots ?? 0)
}

function checkCall(name: string, args: Expr[]): void {
  const needsOne = new Set(['sqrt', 'ln', 'sin', 'cos', 'tan', 'abs', 'exp', 'asin', 'acos', 'atan', 'decimal', 'fraction', 'factor', 'expand'])
  if (needsOne.has(name) && args.length !== 1) throw new MathError(`${name} needs one value.`)
  if (name === 'log' && args.length !== 1 && args.length !== 2) throw new MathError('log takes a value, or a value and a base.')
  if (name === 'solve' && (args.length < 1 || args.length > 4)) throw new MathError('Use solve(equation), solve(equation, x), or solve(eq1, eq2).')
  if ((name === 'gcd' || name === 'lcm') && args.length < 2) throw new MathError(`${name} needs at least two whole numbers.`)
  if ((name === 'mod' || name === 'rem') && args.length !== 2) throw new MathError('Use mod(a, b) for the remainder.')
  if (name === 'zeros' && args.length !== 1 && args.length !== 2) throw new MathError('Use zeros(expr) or zeros(expr, x).')
  if (name === 'zeros' && args.length === 2) requireSymbol(args[1], 'Use zeros(expr, x).')
  if (name === 'diff') {
    if (args.length < 2 || args.length > 4) throw new MathError('Use diff(expr, x), diff(expr, x, 2), or diff(expr, x, 1, a).')
    requireSymbol(args[1], 'Say which variable to differentiate, for example diff(x^2, x).')
  }
  if (name === 'integrate' && args.length !== 2 && args.length !== 4) throw new MathError('Use integrate(expr, x) or integrate(expr, x, a, b).')
  if (name === 'integrate') requireSymbol(args[1], 'Say which variable to integrate, for example integrate(x^2, x).')
  if (name === 'limit' && args.length !== 3) throw new MathError('Use limit(expr, x, a).')
  if (name === 'limit') requireSymbol(args[1], 'Use limit(expr, x, a).')
  if ((name === 'sum' || name === 'prod') && args.length !== 4) throw new MathError(`Use ${name}(expr, i, start, end).`)
  if (name === 'sum' || name === 'prod') requireSymbol(args[1], `Use ${name}(expr, i, start, end).`)
  if ((name === 'tangent' || name === 'normal') && args.length !== 3) throw new MathError(`Use ${name}(expr, x, a).`)
  if (name === 'tangent' || name === 'normal') requireSymbol(args[1], `Use ${name}(expr, x, a).`)
  if (name === 'fmin' || name === 'fmax') {
    if (args.length !== 4) throw new MathError(`Use ${name}(expr, x, a, b).`)
    requireSymbol(args[1], `Use ${name}(expr, x, a, b).`)
  }
  if (name === 'series' && args.length !== 4) throw new MathError('Use series(expr, x, a, order).')
  if (name === 'series') requireSymbol(args[1], 'Use series(expr, x, a, order).')
  if ((name === 'dsolve' || name === 'idiff') && args.length !== 3) throw new MathError(name === 'dsolve' ? 'Use dsolve(equation, y, x).' : 'Use idiff(equation, y, x).')
  if (name === 'dsolve' || name === 'idiff') {
    requireSymbol(args[1], name === 'dsolve' ? 'Use dsolve(equation, y, x).' : 'Use idiff(equation, y, x).')
    requireSymbol(args[2], name === 'dsolve' ? 'Use dsolve(equation, y, x).' : 'Use idiff(equation, y, x).')
  }
  if ((name === 'dot' || name === 'cross') && args.length !== 2) throw new MathError(`Use ${name}(u, v).`)
  if ((name === 'unit' || name === 'norm' || name === 'mag') && args.length !== 1) throw new MathError(`Use ${name}(v) on a vector.`)
  if ((name === 'det' || name === 'transpose' || name === 'inv' || name === 'trace') && args.length !== 1) throw new MathError(`Use ${name}(matrix).`)
  if (name === 'Dt' && args.length !== 1) throw new MathError('Use Dt(expr) for a time derivative.')
}

function termToExpr(term: Term): Expr {
  const negative = term.coeff.n < 0n
  const coeff = negative ? { n: -term.coeff.n, d: term.coeff.d } : term.coeff
  const num: Expr[] = []
  const den: Expr[] = []
  if (coeff.d !== 1n) den.push(rat(coeff.d))
  if (coeff.n !== 1n) num.push(rat(coeff.n))
  for (const atom of term.atoms) {
    const base: Expr = atom.kind === 'var' ? { type: 'sym', name: atom.name, sub: atom.sub, dots: atom.dots } : atom.expr
    const exp = atom.exp
    const piece: Expr = Math.abs(exp) === 1 ? base : { type: 'pow', base, exp: rat(BigInt(Math.abs(exp))) }
    if (exp > 0) num.push(piece)
    else den.push(piece)
  }
  let expr: Expr = num.length === 0 ? rat(1n) : num.length === 1 ? num[0] : { type: 'mul', args: num }
  if (den.length === 1) expr = { type: 'div', num: expr, den: den[0] }
  else if (den.length > 1) expr = { type: 'div', num: expr, den: { type: 'mul', args: den } }
  if (!negative) return expr
  if (expr.type === 'rat') return rat(-expr.n, expr.d)
  return { type: 'mul', args: [rat(-1n), expr] }
}

function preferConstantFirst(e: Expr): Expr {
  if (e.type !== 'add') return e
  const constants = e.args.filter((arg) => arg.type === 'rat' && arg.n > 0n)
  const rest = e.args.filter((arg) => !(arg.type === 'rat' && arg.n > 0n))
  if (constants.length === 0 || rest.length === 0 || !isNegative(rest[0])) return e
  return { type: 'add', args: [...constants, ...rest] }
}

function isNegative(e: Expr): boolean {
  if (e.type === 'rat') return e.n < 0n
  if (e.type === 'mul' && e.args[0]?.type === 'rat' && e.args[0].n < 0n) return true
  if (e.type === 'div') return isNegative(e.num)
  return false
}

function toPoly(e: Expr, variable: string): Expr[] | null {
  const buckets = new Map<number, Term[]>()
  for (const term of toSum(fold(e))) {
    let power = 0
    const rest: Atom[] = []
    for (const atom of term.atoms) {
      if (atom.kind === 'var' && atom.name === variable) {
        if (atom.exp < 0) return null
        power += atom.exp
      } else if (atom.kind === 'expr' && depends(atom.expr, variable)) return null
      else rest.push(atom)
    }
    const list = buckets.get(power) ?? []
    list.push({ coeff: term.coeff, atoms: rest })
    buckets.set(power, list)
  }
  if (buckets.size === 0) return [ZERO_EXPR]
  const max = Math.max(...buckets.keys())
  const coeffs: Expr[] = []
  for (let i = 0; i <= max; i += 1) coeffs.push(normalize(fromTerms(buckets.get(i) ?? [])))
  return coeffs
}

function polyDegree(coeffs: Expr[]): number {
  let deg = coeffs.length - 1
  while (deg > 0 && isZeroExpr(coeffs[deg])) deg -= 1
  return deg
}

function isZeroExpr(e: Expr): boolean {
  return toSum(fold(e)).every((term) => term.coeff.n === 0n)
}

function depends(e: Expr, variable: string): boolean {
  if (e.type === 'sym') return symbolKey(e) === variable || (!e.sub && !(e.dots && e.dots > 0) && e.name === variable)
  if (e.type === 'rat' || e.type === 'dec') return false
  if (e.type === 'vec') return e.args.some((arg) => depends(arg, variable))
  if (e.type === 'mat') return e.rows.some((row) => row.some((arg) => depends(arg, variable)))
  if (e.type === 'add' || e.type === 'mul') return e.args.some((arg) => depends(arg, variable))
  if (e.type === 'div') return depends(e.num, variable) || depends(e.den, variable)
  if (e.type === 'pow') return depends(e.base, variable) || depends(e.exp, variable)
  if (e.type === 'call') return e.args.some((arg) => depends(arg, variable))
  if (e.type === 'eq') return depends(e.left, variable) || depends(e.right, variable)
  return false
}

function hasFreeSymbol(e: Expr): boolean {
  return freeSymbols(e).length > 0
}

function containsConstantSym(e: Expr): boolean {
  let found = false
  walk(e, (node) => {
    if (node.type === 'sym' && bareConstant(node)) found = true
  })
  return found
}

function keepSymbolic(e: Expr): boolean {
  if (e.type === 'call' && e.name === 'sqrt') {
    const arg = e.args[0]
    if (arg?.type === 'rat' && arg.n >= 0n) return !isPerfectSquare(arg)
    return true
  }
  if (e.type === 'pow' && e.exp.type === 'rat' && e.exp.d !== 1n) return true
  if (e.type === 'add' || e.type === 'mul') return e.args.some(keepSymbolic)
  if (e.type === 'div') return keepSymbolic(e.num) || keepSymbolic(e.den)
  if (e.type === 'pow') return keepSymbolic(e.base) || keepSymbolic(e.exp)
  if (e.type === 'call') return e.args.some(keepSymbolic)
  if (e.type === 'vec') return e.args.some(keepSymbolic)
  if (e.type === 'mat') return e.rows.some((row) => row.some(keepSymbolic))
  if (e.type === 'eq') return keepSymbolic(e.left) || keepSymbolic(e.right)
  return false
}

function isPerfectSquare(r: Expr): boolean {
  if (r.type !== 'rat' || r.n < 0n) return false
  const num = squareParts(r.n)
  const den = squareParts(r.d)
  return num.rest === 1n && den.rest === 1n
}

function firstUnknown(e: Expr): string | null {
  let name: string | null = null
  walk(e, (node) => {
    if (!name && node.type === 'call' && !BUILTIN_CALLS.has(node.name) && !CAS_CALLS.has(node.name)) name = node.name
  })
  return name
}

function walk(e: Expr, visit: (node: Expr) => void): void {
  visit(e)
  switch (e.type) {
    case 'add':
    case 'mul':
      e.args.forEach((arg) => walk(arg, visit))
      break
    case 'div':
      walk(e.num, visit)
      walk(e.den, visit)
      break
    case 'pow':
      walk(e.base, visit)
      walk(e.exp, visit)
      break
    case 'call':
      e.args.forEach((arg) => walk(arg, visit))
      break
    case 'vec':
      e.args.forEach((arg) => walk(arg, visit))
      break
    case 'mat':
      e.rows.forEach((row) => row.forEach((arg) => walk(arg, visit)))
      break
    case 'eq':
      walk(e.left, visit)
      walk(e.right, visit)
      break
    default:
      break
  }
}

function evalConst(e: Expr, angles: AngleMode = 'rad'): number | null {
  switch (e.type) {
    case 'rat':
      return Number(e.n) / Number(e.d)
    case 'dec':
      return e.value
    case 'sym':
      if (!bareConstant(e)) return null
      if (e.name === 'pi') return Math.PI
      if (e.name === 'e') return Math.E
      return null
    case 'add':
      return reduceNums(e.args, 0, (sum, n) => sum + n, angles)
    case 'mul':
      return reduceNums(e.args, 1, (product, n) => product * n, angles)
    case 'div': {
      const num = evalConst(e.num, angles)
      const den = evalConst(e.den, angles)
      if (num === null || den === null) return null
      if (den === 0) return Number.NaN
      return num / den
    }
    case 'pow': {
      const base = evalConst(e.base, angles)
      const exp = evalConst(e.exp, angles)
      if (base === null || exp === null) return null
      if (base < 0 && !Number.isInteger(exp)) return Number.NaN
      return base ** exp
    }
    case 'call': {
      const args: number[] = []
      for (const arg of e.args) {
        const n = evalConst(arg, angles)
        if (n === null) return null
        args.push(n)
      }
      return callNumber(e.name, args, angles)
    }
    case 'vec':
    case 'mat':
    case 'eq':
      return null
  }
}

function reduceNums(args: Expr[], start: number, step: (acc: number, n: number) => number, angles: AngleMode): number | null {
  let acc = start
  for (const arg of args) {
    const n = evalConst(arg, angles)
    if (n === null) return null
    acc = step(acc, n)
  }
  return acc
}

function callNumber(name: string, args: number[], angles: AngleMode): number | null {
  const x = args[0]
  const y = args[1]
  const toRad = (n: number) => (angles === 'deg' ? (n * Math.PI) / 180 : n)
  const fromRad = (n: number) => (angles === 'deg' ? (n * 180) / Math.PI : n)
  switch (name) {
    case 'sqrt':
      return args.length === 1 ? Math.sqrt(x) : null
    case 'ln':
      return args.length === 1 ? Math.log(x) : null
    case 'log':
      if (args.length === 1) return Math.log10(x)
      if (args.length === 2 && y !== 0) return Math.log(x) / Math.log(y)
      return null
    case 'sin':
      return args.length === 1 ? Math.sin(toRad(x)) : null
    case 'cos':
      return args.length === 1 ? Math.cos(toRad(x)) : null
    case 'tan':
      return args.length === 1 ? Math.tan(toRad(x)) : null
    case 'abs':
      return args.length === 1 ? Math.abs(x) : null
    case 'exp':
      return args.length === 1 ? Math.exp(x) : null
    case 'asin':
      return args.length === 1 ? fromRad(Math.asin(x)) : null
    case 'acos':
      return args.length === 1 ? fromRad(Math.acos(x)) : null
    case 'atan':
      return args.length === 1 ? fromRad(Math.atan(x)) : null
    default:
      return null
  }
}

const P_ADD = 1
const P_MUL = 2
const P_POW = 3
const P_ATOM = 5

export function tex(e: Expr): string {
  return texAt(e, 0)
}

function texAt(e: Expr, parent: number): string {
  const [text, prec] = texPrec(e)
  return prec < parent ? `\\left(${text}\\right)` : text
}

function texPrec(e: Expr): [string, number] {
  switch (e.type) {
    case 'rat':
      if (e.d === 1n) return [e.n.toString(), P_ATOM]
      return [`${e.n < 0n ? '-' : ''}\\frac{${e.n < 0n ? -e.n : e.n}}{${e.d}}`, P_ATOM]
    case 'dec':
      return [e.text, P_ATOM]
    case 'sym':
      return [texNamed(e), P_ATOM]
    case 'add':
      return [texAdd(e.args), P_ADD]
    case 'mul':
      return [texMul(e.args), P_MUL]
    case 'div':
      return [`\\frac{${texAt(e.num, 0)}}{${texAt(e.den, 0)}}`, P_ATOM]
    case 'pow':
      if (e.exp.type === 'rat' && e.exp.n === 1n && e.exp.d === 2n) return [`\\sqrt{${texAt(e.base, 0)}}`, P_ATOM]
      return [`${texAt(e.base, P_POW + 1)}^{${texAt(e.exp, 0)}}`, P_POW]
    case 'call':
      return [texCall(e.name, e.args), P_ATOM]
    case 'vec':
      return [`\\left[${e.args.map((arg) => texAt(arg, 0)).join(', ')}\\right]`, P_ATOM]
    case 'mat':
      return [`\\begin{bmatrix}${e.rows.map((row) => row.map((arg) => texAt(arg, 0)).join(' & ')).join(' \\\\ ')}\\end{bmatrix}`, P_ATOM]
    case 'eq':
      return [`${texAt(e.left, 0)} = ${texAt(e.right, 0)}`, 0]
  }
}

const GREEK: Record<string, string> = {
  alpha: '\\alpha',
  beta: '\\beta',
  gamma: '\\gamma',
  delta: '\\delta',
  epsilon: '\\epsilon',
  zeta: '\\zeta',
  eta: '\\eta',
  theta: '\\theta',
  iota: '\\iota',
  kappa: '\\kappa',
  lambda: '\\lambda',
  mu: '\\mu',
  nu: '\\nu',
  xi: '\\xi',
  pi: '\\pi',
  rho: '\\rho',
  sigma: '\\sigma',
  tau: '\\tau',
  upsilon: '\\upsilon',
  phi: '\\phi',
  chi: '\\chi',
  psi: '\\psi',
  omega: '\\omega',
}

const GREEK_CHAR: Record<string, string> = {
  α: 'alpha',
  β: 'beta',
  γ: 'gamma',
  δ: 'delta',
  ε: 'epsilon',
  ζ: 'zeta',
  η: 'eta',
  θ: 'theta',
  ι: 'iota',
  κ: 'kappa',
  λ: 'lambda',
  μ: 'mu',
  ν: 'nu',
  ξ: 'xi',
  π: 'pi',
  ρ: 'rho',
  σ: 'sigma',
  τ: 'tau',
  υ: 'upsilon',
  φ: 'phi',
  χ: 'chi',
  ψ: 'psi',
  ω: 'omega',
}

/** True when the whole word is a Greek letter name, the way `pi` is. */
export function isGreekName(name: string): boolean {
  return Object.prototype.hasOwnProperty.call(GREEK, name.toLowerCase())
}

/** True when more letters could still finish a Greek name: `th` is on the way to `theta`. */
export function isGreekPrefix(name: string): boolean {
  const lower = name.toLowerCase()
  if (!lower) return false
  return Object.keys(GREEK).some((greek) => greek.startsWith(lower))
}

export function containsGreekLetter(text: string): boolean {
  return [...text].some((ch) => greekCharName(ch) !== null)
}

function greekCharName(char: string): string | null {
  return GREEK_CHAR[char] ?? null
}

function canonicalGreek(name: string): string {
  const lower = name.toLowerCase()
  return Object.prototype.hasOwnProperty.call(GREEK, lower) ? lower : name
}

function subscriptName(body: string): string {
  if (body === '?') return '?'
  if (!/^[A-Za-z0-9]+$/.test(body)) throw new MathError('Use _ after a name, for example e_r or theta_0.')
  return canonicalGreek(body)
}

function texSymbol(name: string): string {
  if (name === '?') return '\\square'
  return GREEK[name] ?? name
}

function texNamed(e: { name: string; sub?: string; dots?: number }): string {
  let body = texSymbol(e.name)
  if (e.name === 'e' && e.sub) body = `\\mathbf{${body}}`
  if (e.sub) body = `${body}_{${texSymbol(e.sub)}}`
  const dots = e.dots ?? 0
  if (dots === 1) return `\\dot{${body}}`
  if (dots === 2) return `\\ddot{${body}}`
  if (dots === 3) return `\\dddot{${body}}`
  if (dots > 3) return `\\overset{(${dots})}{${body}}`
  return body
}

function texAdd(args: Expr[]): string {
  let out = ''
  args.forEach((arg, index) => {
    const neg = isNegative(arg)
    const body = texAt(neg ? stripNegative(arg) : arg, P_ADD)
    if (index === 0) out = neg ? `-${body}` : body
    else out += neg ? ` - ${body}` : ` + ${body}`
  })
  return out
}

function stripNegative(e: Expr): Expr {
  if (e.type === 'rat') return rat(e.n < 0n ? -e.n : e.n, e.d)
  if (e.type === 'mul' && e.args[0]?.type === 'rat' && e.args[0].n < 0n) {
    const coeff = rat(-e.args[0].n, e.args[0].d)
    const rest = e.args.slice(1)
    if (coeff.type === 'rat' && coeff.n === 1n && coeff.d === 1n) return rest.length === 1 ? rest[0] : { type: 'mul', args: rest }
    return { type: 'mul', args: [coeff, ...rest] }
  }
  if (e.type === 'div' && isNegative(e.num)) return { type: 'div', num: stripNegative(e.num), den: e.den }
  return e
}

function texMul(args: Expr[]): string {
  if (args.length >= 2 && args[0].type === 'rat' && args[0].n === -1n && args[0].d === 1n) {
    const rest = args.slice(1)
    const body = rest.length === 1 ? texAt(rest[0], P_MUL) : texMul(rest)
    return `-${body}`
  }
  return args
    .map((arg, index) => {
      const piece = texAt(arg, P_MUL)
      if (index === 0) return piece
      const prev = args[index - 1]
      const juxtapose = (prev.type === 'rat' || prev.type === 'dec') && (arg.type === 'sym' || arg.type === 'call' || arg.type === 'pow')
      return juxtapose ? piece : ` \\cdot ${piece}`
    })
    .join('')
}

function texTimeDerivative(arg: Expr, order: number): string {
  let body = arg
  let count = order
  while (body.type === 'call' && body.name === 'Dt' && body.args.length === 1) {
    body = body.args[0]
    count += 1
  }
  const inner = texAt(body, 0)
  if (count === 1) return `\\frac{d}{dt}\\left(${inner}\\right)`
  return `\\frac{d^{${count}}}{dt^{${count}}}\\left(${inner}\\right)`
}

function texCall(name: string, args: Expr[]): string {
  if (name === 'Dt' && args.length === 1) return texTimeDerivative(args[0], 1)
  if (name === 'sqrt' && args.length === 1) return `\\sqrt{${texAt(args[0], 0)}}`
  if (name === 'abs' && args.length === 1) return `\\left|${texAt(args[0], 0)}\\right|`
  const macro: Record<string, string> = { sin: '\\sin', cos: '\\cos', tan: '\\tan', ln: '\\ln', log: '\\log', exp: '\\exp', asin: '\\arcsin', acos: '\\arccos', atan: '\\arctan' }
  if (name === 'log' && args.length === 2) return `\\log_{${texAt(args[1], 0)}}\\left(${texAt(args[0], 0)}\\right)`
  const head = macro[name] ?? name
  return `${head}\\left(${args.map((arg) => texAt(arg, 0)).join(', ')}\\right)`
}

export function plain(e: Expr): string {
  return plainAt(e, 0)
}

function plainAt(e: Expr, parent: number): string {
  const [text, prec] = plainPrec(e)
  return prec < parent ? `(${text})` : text
}

function plainPrec(e: Expr): [string, number] {
  switch (e.type) {
    case 'rat':
      if (e.d === 1n) return [e.n.toString(), P_ATOM]
      return [`${e.n < 0n ? '-' : ''}${e.n < 0n ? -e.n : e.n}/${e.d}`, P_ATOM]
    case 'dec':
      return [e.text, P_ATOM]
    case 'sym':
      return [symbolKey(e), P_ATOM]
    case 'add':
      return [plainAdd(e.args), P_ADD]
    case 'mul':
      return [plainMul(e.args), P_MUL]
    case 'div':
      return [`${plainAt(e.num, P_MUL)}/${plainAt(e.den, P_MUL)}`, P_MUL]
    case 'pow':
      if (e.exp.type === 'rat' && e.exp.n === 1n && e.exp.d === 2n) return [`sqrt(${plainAt(e.base, 0)})`, P_ATOM]
      return [`${plainAt(e.base, P_POW + 1)}^${plainAt(e.exp, P_POW)}`, P_POW]
    case 'call':
      if (e.name === 'sqrt' && e.args.length === 1) return [`sqrt(${plainAt(e.args[0], 0)})`, P_ATOM]
      return [`${e.name}(${e.args.map((arg) => plainAt(arg, 0)).join(', ')})`, P_ATOM]
    case 'vec':
      return [`[${e.args.map((arg) => plainAt(arg, 0)).join(', ')}]`, P_ATOM]
    case 'mat':
      return [`[${e.rows.map((row) => `[${row.map((arg) => plainAt(arg, 0)).join(', ')}]`).join(', ')}]`, P_ATOM]
    case 'eq':
      return [`${plainAt(e.left, 0)} = ${plainAt(e.right, 0)}`, 0]
  }
}

function plainAdd(args: Expr[]): string {
  let out = ''
  args.forEach((arg, index) => {
    const neg = isNegative(arg)
    const body = plainAt(neg ? stripNegative(arg) : arg, P_ADD)
    if (index === 0) out = neg ? `-${body}` : body
    else out += neg ? ` - ${body}` : ` + ${body}`
  })
  return out
}

function plainMul(args: Expr[]): string {
  if (args.length >= 2 && args[0].type === 'rat' && args[0].n === -1n && args[0].d === 1n) {
    const rest = args.slice(1)
    const body = rest.length === 1 ? plainAt(rest[0], P_MUL) : plainMul(rest)
    return `-${body}`
  }
  return args
    .map((arg, index) => {
      const piece = plainAt(arg, P_MUL)
      if (index === 0) return piece
      const prev = args[index - 1]
      const juxtapose = (prev.type === 'rat' || prev.type === 'dec') && (arg.type === 'sym' || arg.type === 'call' || arg.type === 'pow')
      return juxtapose ? piece : `*${piece}`
    })
    .join('')
}

function exprKey(e: Expr): string {
  switch (e.type) {
    case 'rat':
      return `r${e.n}/${e.d}`
    case 'dec':
      return `d${e.text}`
    case 'sym':
      return `s${symbolKey(e)}`
    case 'add':
      return `a(${e.args.map(exprKey).join(',')})`
    case 'mul':
      return `m(${e.args.map(exprKey).join(',')})`
    case 'div':
      return `v(${exprKey(e.num)}/${exprKey(e.den)})`
    case 'pow':
      return `p(${exprKey(e.base)}^${exprKey(e.exp)})`
    case 'call':
      return `c${e.name}(${e.args.map(exprKey).join(',')})`
    case 'vec':
      return `u(${e.args.map(exprKey).join(',')})`
    case 'mat':
      return `M(${e.rows.map((row) => row.map(exprKey).join(',')).join(';')})`
    case 'eq':
      return `q(${exprKey(e.left)}=${exprKey(e.right)})`
  }
}

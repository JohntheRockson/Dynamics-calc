// Small exact-rational kernel: parse, draw as TeX, evaluate, and solve
// linear or quadratic equations. Trig angles follow the active unit.

import { FUNCTIONS, GRAPH_OPTIONS, canonicalOptionName, findFunction, findOption, functionByKernel, parseColor, suggestName, type FunctionSpec, type OptionSpec } from './functions'
import { MATH_CARET } from './inputView'
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
  | { type: 'call'; name: string; args: Expr[]; options?: OptionBlock }
  | { type: 'vec'; args: Expr[] }
  | { type: 'mat'; rows: Expr[][] }
  | { type: 'eq'; left: Expr; right: Expr }
  | { type: 'group'; body: Expr }
  | { type: 'caret'; place: 'pre' | 'post' | 'sub'; body: Expr | null; split?: number }

export type CallExpr = Extract<Expr, { type: 'call' }>

/** One `Name: value` inside a settings block. A bare flag such as `Dashed` has no value. */
export interface OptionEntry {
  key: string
  value: string | null
  /** The entry as typed, including a live caret marker. */
  raw: string
}

/** The `{…}` after a call or at the end of a line, read but not yet checked. */
export interface OptionBlock {
  raw: string
  entries: OptionEntry[]
  /** A live caret sits just before the opening brace. */
  caretBefore?: boolean
}

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

/** Built-in names ignore case and have aliases: Derivative, derivative, and diff are one function. */
export function resolveCallName(name: string): string {
  return findFunction(name)?.kernel ?? name
}

function isElementary(name: string): boolean {
  return functionByKernel(name)?.section === 'basic'
}

export function isKnownCall(name: string): boolean {
  return name === 'Dt' || functionByKernel(name) !== null
}

export function isReservedName(name: string): boolean {
  return name === 'pi' || name === 'e' || name === 'Dt' || findFunction(name) !== null
}

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

export interface PlotDomain {
  name: string
  min: Expr
  max: Expr
}

export interface PlotOptions {
  points: number
  recursion: number
  exclusions: boolean
  domain: PlotDomain | null
  /** Ranges named after an input, as in {t: 0..2*pi} or Plot3D(…){x: -2..2}. */
  ranges: PlotDomain[]
  color: string | null
  dashed: boolean
}

export interface SearchDomain {
  name: string | null
  min: Expr
  max: Expr
}

export const DEFAULT_PLOT: PlotOptions = { points: 128, recursion: 5, exclusions: true, domain: null, ranges: [], color: null, dashed: false }

export function plotDefaults(): PlotOptions {
  return { ...DEFAULT_PLOT, ranges: [] }
}

export function naming(raw: string): string {
  const chars = [...raw]
  if (chars.length === 1) {
    const greek = greekCharName(chars[0] ?? '')
    if (greek) return greek
  }
  return canonicalGreek(raw)
}

const UNREADABLE = 'Could not read that. Check the operators and parentheses.'
const RANGE_HINT = 'Put a range such as 0..2*pi in { } after the closing parenthesis, for example Plot(sin(x), x){Domain: 0..2*pi}.'

export function parseExpr(input: string): Expr {
  return parseTracked(input).expr
}

/**
 * Parse an expression and report the call it ends with, if any. In `2*Derivative(x^3, x)` that
 * is the Derivative, which is where a settings block typed at the end of the line belongs.
 */
export function parseTracked(input: string): { expr: Expr; tail: CallExpr | null } {
  const text = input.trim()
  const parser = new Parser(text)
  let expr: Expr
  try {
    expr = parser.parseEquation()
    parser.skip()
  } catch (error) {
    if (error instanceof MathError && error.message === UNREADABLE && text.includes('..')) throw new MathError(RANGE_HINT)
    throw error
  }
  if (parser.i < text.length) {
    if (text[parser.i] === '{') throw new MathError('Settings in { } go right after a function’s closing parenthesis, or at the end of the line.')
    throw new MathError(text.includes('..') ? RANGE_HINT : UNREADABLE)
  }
  let end = text.length
  while (end > 0 && (/\s/.test(text[end - 1] ?? '') || text[end - 1] === MATH_CARET)) end -= 1
  const tail = parser.calls.find((call) => call.end === end)?.node ?? null
  return { expr, tail }
}

class Parser {
  src: string
  i = 0
  /** Every call read so far and where it ends, settings included. */
  calls: { node: CallExpr; end: number }[] = []

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
    return c === '(' || c === '[' || c === '?' || /[A-Za-z0-9.]/.test(c) || greekCharName(c) !== null
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
    if (this.eat('{')) {
      const exp = this.parseSum()
      if (!this.eat('}')) throw new MathError('Close the exponent with }.')
      return { type: 'pow', base, exp: { type: 'group', body: exp } }
    }
    return { type: 'pow', base, exp: this.parseUnary() }
  }

  /** Primes and underscores bind to the name just typed: theta'^2 is (theta-dot)^2, and e_theta is e with a subscript. */
  parsePostfix(expr: Expr): Expr {
    let current = expr
    let trailing = false
    for (;;) {
      if (this.src[this.i] === MATH_CARET) {
        this.i += 1
        trailing = true
        continue
      }
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
    return trailing ? { type: 'caret', place: 'post', body: current } : current
  }

  /** A prime after a finished power wraps that power: theta^2' is d/dt of theta^2. */
  consumePrimes(expr: Expr): Expr {
    let current = expr
    let trailing = false
    for (;;) {
      if (this.src[this.i] === MATH_CARET) {
        this.i += 1
        trailing = true
        continue
      }
      if (!this.eat("'")) break
      current = this.withPrime(current)
    }
    return trailing ? { type: 'caret', place: 'post', body: current } : current
  }

  withPrime(expr: Expr): Expr {
    const target = expr.type === 'group' ? expr.body : expr
    if (target.type === 'sym') return { ...target, dots: (target.dots ?? 0) + 1 }
    return { type: 'call', name: 'Dt', args: [expr] }
  }

  withSubscript(expr: Expr, sub: { name: string; split: number | null }): Expr {
    if (expr.type !== 'sym') throw new MathError('Use _ after a name, for example e_r or theta_0.')
    if (expr.sub) throw new MathError('That name already has a subscript.')
    if (sub.split === null) return { ...expr, sub: sub.name }
    return { type: 'caret', place: 'sub', body: { ...expr, sub: sub.name }, split: sub.split }
  }

  parseSubscript(): { name: string; split: number | null } {
    if (!this.eat('_')) throw new MathError('Use _ after a name, for example e_r or theta_0.')
    if (this.eat('{')) {
      const start = this.i
      while (this.i < this.src.length && this.src[this.i] !== '}') this.i += 1
      const raw = this.src.slice(start, this.i)
      if (!this.eat('}')) throw new MathError('Close the subscript with }.')
      const caretAt = raw.indexOf(MATH_CARET)
      if (caretAt >= 0) {
        const name = raw.replaceAll(MATH_CARET, '').trim()
        const before = raw.slice(0, caretAt).replaceAll(MATH_CARET, '').trim()
        if (!name) return { name: '', split: 0 }
        return { name: subscriptName(name), split: before.length }
      }
      return { name: subscriptName(raw.trim()), split: null }
    }
    const ch = this.src[this.i] ?? ''
    const fromChar = greekCharName(ch)
    if (fromChar) {
      this.i += 1
      return { name: fromChar, split: null }
    }
    if (ch === '?' || ch === MATH_CARET) {
      this.i += 1
      return { name: ch === MATH_CARET ? 'CARET' : '?', split: null }
    }
    if (!/[A-Za-z0-9]/.test(ch)) throw new MathError('Use _ after a name, for example e_r or theta_0.')
    const start = this.i
    while (this.i < this.src.length && /[A-Za-z0-9]/.test(this.src[this.i] ?? '')) this.i += 1
    return { name: subscriptName(this.src.slice(start, this.i)), split: null }
  }

  parsePrimary(): Expr {
    const c = this.peek()
    if (c === MATH_CARET) {
      this.i += 1
      this.skip()
      if (!this.canStartPrimary()) return { type: 'caret', place: 'pre', body: null }
      return { type: 'caret', place: 'pre', body: this.parsePostfix(this.parsePrimary()) }
    }
    if (!c) throw new MathError(UNREADABLE)
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
        if (!this.eat(')')) throw new MathError(UNREADABLE)
        const node: CallExpr = { type: 'call', name: resolveCallName(ident.name), args }
        const options = this.parseCallOptions()
        if (options) node.options = options
        this.calls.push({ node, end: this.i })
        return node
      }
      const name = resolveCallName(ident.name)
      if (isElementary(name) && this.canStartPrimary()) return { type: 'call', name, args: [this.parseProduct()] }
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
        if (!this.eat(')')) throw new MathError(UNREADABLE)
        return { type: 'vec', args }
      }
      if (!this.eat(')')) throw new MathError(UNREADABLE)
      return { type: 'group', body: inner }
    }
    throw new MathError(UNREADABLE)
  }

  /** A settings block right after a call's closing parenthesis. */
  parseCallOptions(): OptionBlock | null {
    let at = this.i
    while (at < this.src.length && /\s/.test(this.src[at] ?? '')) at += 1
    let caretBefore = false
    if (this.src[at] === MATH_CARET) {
      let next = at + 1
      while (next < this.src.length && /\s/.test(this.src[next] ?? '')) next += 1
      if (this.src[next] !== '{') return null
      caretBefore = true
      at = next
    }
    if (this.src[at] !== '{') return null
    const close = matchBrace(this.src, at)
    if (close < 0) throw new MathError('Close the settings with }.')
    const block = readOptionBlock(this.src.slice(at + 1, close))
    if (caretBefore) block.caretBefore = true
    this.i = close + 1
    return block
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
    const args = [this.parseEquation()]
    while (this.eat(',')) args.push(this.parseEquation())
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
    const s = this.src
    let i = this.i
    let raw = ''
    let split: number | null = null
    const pullDigits = () => {
      while (i < s.length && /[0-9]/.test(s[i] ?? '')) {
        raw += s[i]
        i += 1
      }
    }
    const pullInteriorCaret = () => {
      if (split !== null || s[i] !== MATH_CARET) return
      if (!/[0-9.]/.test(s[i + 1] ?? '')) return
      split = raw.length
      i += 1
    }
    if (s[i] === '.') {
      raw = '.'
      i += 1
      pullInteriorCaret()
      pullDigits()
    } else {
      pullDigits()
      pullInteriorCaret()
      pullDigits()
      if (s[i] === '.') {
        raw += '.'
        i += 1
        pullInteriorCaret()
        pullDigits()
      }
    }
    if (!/\d/.test(raw)) throw new MathError('Could not read that number.')
    if ((s[i] === 'e' || s[i] === 'E') && /[0-9+-]/.test(s[i + 1] ?? '')) {
      const expStart = i
      let j = i + 1
      if (s[j] === '+' || s[j] === '-') j += 1
      if (/[0-9]/.test(s[j] ?? '')) {
        while (/[0-9]/.test(s[j] ?? '')) j += 1
        raw += s.slice(expStart, j)
        i = j
      }
    }
    this.i = i
    const expr = literalNumber(raw)
    if (split === null) return expr
    return { type: 'caret', place: 'post', body: expr, split }
  }
}

function literalNumber(literal: string): Expr {
  if (/[eE.]/.test(literal)) {
    const value = Number(literal)
    if (!Number.isFinite(value)) throw new MathError('Could not read that number.')
    return { type: 'dec', text: literal, value }
  }
  const digits = literal.replace(/^0+(?=\d)/, '') || '0'
  return rat(BigInt(digits), 1n)
}

function matchBrace(text: string, open: number): number {
  let depth = 0
  for (let i = open; i < text.length; i += 1) {
    if (text[i] === '{') depth += 1
    else if (text[i] === '}') {
      depth -= 1
      if (depth === 0) return i
    }
  }
  return -1
}

function splitTopLevel(text: string): string[] {
  const pieces: string[] = []
  let depth = 0
  let start = 0
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i]
    if (ch === '(' || ch === '[' || ch === '{') depth += 1
    else if (ch === ')' || ch === ']' || ch === '}') depth = Math.max(0, depth - 1)
    else if (ch === ',' && depth === 0) {
      pieces.push(text.slice(start, i))
      start = i + 1
    }
  }
  pieces.push(text.slice(start))
  return pieces
}

const OPTION_PARTS = /^(\s*)([A-Za-zα-ω][A-Za-z0-9α-ω_\s-]*?)(\s*(?:->|:|=)\s*)([\s\S]*)$/
const OPTION_FLAG = /^\s*[A-Za-zα-ω][A-Za-z0-9α-ω_\s-]*$/

/** Split a settings block into entries. Nothing is checked here, so a half-typed block still previews. */
export function readOptionBlock(raw: string): OptionBlock {
  const entries = splitTopLevel(raw).map((piece): OptionEntry => {
    const clean = piece.replaceAll(MATH_CARET, '')
    if (!clean.trim()) return { key: '', value: null, raw: piece }
    const pair = OPTION_PARTS.exec(clean)
    if (pair) return { key: (pair[2] ?? '').trim(), value: (pair[4] ?? '').trim(), raw: piece }
    if (OPTION_FLAG.test(clean)) return { key: clean.trim(), value: null, raw: piece }
    return { key: '', value: clean.trim(), raw: piece }
  })
  return { raw, entries }
}

/** `0..2*pi` as its two ends, split at the `..` that is not inside brackets. */
export function splitRange(text: string): { min: string; max: string } | null {
  let depth = 0
  for (let i = 0; i < text.length - 1; i += 1) {
    const ch = text[i]
    if (ch === '(' || ch === '[' || ch === '{') depth += 1
    else if (ch === ')' || ch === ']' || ch === '}') depth -= 1
    else if (depth === 0 && ch === '.' && text[i + 1] === '.') return { min: text.slice(0, i).trim(), max: text.slice(i + 2).trim() }
  }
  return null
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
  if (unknown) {
    const guess = suggestName(unknown, FUNCTIONS.map((spec) => spec.name))
    throw new MathError(guess ? `${unknown} is not defined. Did you mean ${guess}?` : `${unknown} is not defined.`)
  }
  if (e.type === 'dec') return { tex: texDecimal(e.text), text: e.text }
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
  return { tex: texDecimal(text), text }
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
    case 'group':
      return applyEnv(e.body, env, depth)
    case 'caret':
      return e.body ? applyEnv(e.body, env, depth) : e
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
      throw new MathError('Say which variable to solve for, for example Solve(x + y = 3, x).')
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
    case 'group':
      return substitute(e.body, map)
    case 'caret':
      return e.body ? substitute(e.body, map) : e
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
    case 'group':
      return fold(e.body, angles)
    case 'caret':
      return e.body ? fold(e.body, angles) : e
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

/** Exact powers past this many bits would stall the page, so they turn into decimals instead. */
const EXACT_POWER_BITS = 3000n

function powRat(base: Expr, exp: bigint): Expr {
  if (base.type !== 'rat') return base
  if (exp < 0n && base.n === 0n) return { type: 'div', num: rat(1n), den: ZERO_EXPR }
  const size = BigInt(Math.max((base.n < 0n ? -base.n : base.n).toString(2).length, base.d.toString(2).length) - 1)
  if (size * (exp < 0n ? -exp : exp) > EXACT_POWER_BITS) {
    const value = Math.pow(Number(base.n) / Number(base.d), Number(exp))
    return Number.isFinite(value) ? { type: 'dec', text: trimNum(value), value } : { type: 'pow', base, exp: rat(exp) }
  }
  if (exp < 0n) return rat(base.d ** -exp, base.n ** -exp)
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
    case 'group':
      return toSum(e.body)
    case 'caret':
      return e.body ? toSum(e.body) : []
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

function bareConstant(e: { name: string; sub?: string; dots?: number }): boolean {
  return !e.sub && !(e.dots && e.dots > 0) && (e.name === 'pi' || e.name === 'e')
}

function symbolKey(e: { name: string; sub?: string; dots?: number }): string {
  const base = e.sub ? `${e.name}_${e.sub}` : e.name
  return base + "'".repeat(e.dots ?? 0)
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
  if ((e.type === 'group' || e.type === 'caret') && e.body) return isNegative(e.body)
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
  if ((e.type === 'group' || e.type === 'caret') && e.body) return depends(e.body, variable)
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
  if ((e.type === 'group' || e.type === 'caret') && e.body) return keepSymbolic(e.body)
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
    if (!name && node.type === 'call' && !isKnownCall(node.name)) name = node.name
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
    case 'group':
      walk(e.body, visit)
      break
    case 'caret':
      if (e.body) walk(e.body, visit)
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
    case 'group':
      return evalConst(e.body, angles)
    case 'caret':
      return e.body ? evalConst(e.body, angles) : null
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
      return [texDecimal(e.text), P_ATOM]
    case 'sym':
      return [texNamed(e), P_ATOM]
    case 'add':
      return [texAdd(e.args), P_ADD]
    case 'mul':
      return [texMul(e.args), P_MUL]
    case 'div':
      return [`\\frac{${texAt(peelGroup(e.num), 0)}}{${texAt(peelGroup(e.den), 0)}}`, P_ATOM]
    case 'pow': {
      const exp = peelGroup(e.exp)
      if (exp.type === 'rat' && exp.n === 1n && exp.d === 2n) return [`\\sqrt{${texAt(e.base, 0)}}`, P_ATOM]
      return [`${texAt(e.base, P_POW + 1)}^{${texAt(exp, 0)}}`, P_POW]
    }
    case 'call':
      return [texCall(e), P_ATOM]
    case 'vec':
      return [`\\left[${e.args.map((arg) => texAt(arg, 0)).join(', ')}\\right]`, P_ATOM]
    case 'mat':
      return [`\\begin{bmatrix}${e.rows.map((row) => row.map((arg) => texAt(arg, 0)).join(' & ')).join(' \\\\ ')}\\end{bmatrix}`, P_ATOM]
    case 'eq':
      return [`${texAt(e.left, 0)} = ${texAt(e.right, 0)}`, 0]
    case 'group':
      if (isBareCaret(e.body)) return [`(${CARET_TEX}\\vphantom{0})`, P_ATOM]
      return [`\\left(${texAt(e.body, 0)}\\right)`, P_ATOM]
    case 'caret':
      return [texCaret(e), P_ATOM]
  }
}

function peelGroup(e: Expr): Expr {
  return e.type === 'group' ? e.body : e
}

const SCI_NOTATION = /^(-?\d+(?:\.\d+)?)e([+-]?\d+)$/i

/** `4.7947e+23` (a raw JS exponential string) becomes `4.7947 \times 10^{23}`, spaced like other math platforms. */
export function texDecimal(text: string): string {
  const match = SCI_NOTATION.exec(text)
  if (!match) return text
  const mantissa = match[1]
  const exponent = Number(match[2])
  return `${mantissa} \\times 10^{${exponent}}`
}

// The rule's height uses mu, a unit that scales with the current script style, so the caret
// shrinks inside a nested fraction instead of staying the size of the outer formula (em would
// hold it to the *text*-style size no matter how deep the nesting goes; mu tracks the style KaTeX
// is actually drawing right now). The width stays a fixed physical size, like a real text caret.
export const CARET_TEX = '\\textcolor{#d6dee8}{\\smash{\\rule{1.4px}{16.2mu}}}'

function isBareCaret(e: Expr): boolean {
  return e.type === 'caret' && e.body === null
}

function texCaret(e: Extract<Expr, { type: 'caret' }>): string {
  if (e.place === 'sub' && e.body?.type === 'sym') return texNamed(e.body, subscriptCaretTex(e.body.sub ?? '', e.split ?? 0))
  if (!e.body) return CARET_TEX
  if (e.split !== undefined && (e.body.type === 'dec' || (e.body.type === 'rat' && e.body.d === 1n))) {
    const text = e.body.type === 'dec' ? e.body.text : e.body.n.toString()
    const at = Math.max(0, Math.min(e.split, text.length))
    return `${text.slice(0, at)}${CARET_TEX}${text.slice(at)}`
  }
  const inner = texAt(e.body, 0)
  return e.place === 'pre' ? `${CARET_TEX}${inner}` : `${inner}${CARET_TEX}`
}

function subscriptCaretTex(sub: string, split: number): string {
  if (!sub) return CARET_TEX
  if (isGreekName(sub)) return split !== 0 ? `${texSymbol(sub)}${CARET_TEX}` : `${CARET_TEX}${texSymbol(sub)}`
  const at = Math.max(0, Math.min(split, sub.length))
  return `${sub.slice(0, at)}${CARET_TEX}${sub.slice(at)}`
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
  if (body.length === 1 && greekCharName(body)) return greekCharName(body) ?? body
  if (!/^[A-Za-z0-9]+$/.test(body)) throw new MathError('Use _ after a name, for example e_r or theta_0.')
  return canonicalGreek(body)
}

export function texName(name: string): string {
  return texSymbol(name)
}

function texSymbol(name: string): string {
  if (name === '?') return '\\square'
  if (name === 'CARET') return CARET_TEX
  return GREEK[name] ?? name
}

function texNamed(e: { name: string; sub?: string; dots?: number }, subTex?: string): string {
  let body = texSymbol(e.name)
  if (e.name === 'e' && (Boolean(e.sub) || subTex !== undefined)) body = `\\mathbf{${body}}`
  if (subTex !== undefined) body = `${body}_{${subTex}}`
  else if (e.sub) body = `${body}_{${texSymbol(e.sub)}}`
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
      if (arg.type === 'caret' && arg.body === null) return ` \\cdot ${piece}`
      const prev = args[index - 1]
      const prevCore = prev.type === 'caret' && prev.body ? prev.body : prev
      const juxtapose = (prevCore.type === 'rat' || prevCore.type === 'dec') && (arg.type === 'sym' || arg.type === 'call' || arg.type === 'pow' || arg.type === 'group')
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

const ELEMENTARY_TEX: Record<string, string> = { sin: '\\sin', cos: '\\cos', tan: '\\tan', ln: '\\ln', log: '\\log', exp: '\\exp', asin: '\\arcsin', acos: '\\arccos', atan: '\\arctan' }

function texCall(e: CallExpr): string {
  return `${texCallBody(e.name, e.args)}${e.options ? blockTex(e.options) : ''}`
}

function texCallBody(name: string, args: Expr[]): string {
  if (name === 'Dt' && args.length === 1) return texTimeDerivative(args[0], 1)
  if (name === 'sqrt' && args.length === 1) return `\\sqrt{${texAt(args[0], 0)}}`
  if (name === 'abs' && args.length === 1) return `\\left|${texAt(args[0], 0)}\\right|`
  if (name === 'log' && args.length === 2) return `\\log_{${texAt(args[1], 0)}}\\left(${texAt(args[0], 0)}\\right)`
  const spec = functionByKernel(name)
  const head = ELEMENTARY_TEX[name] ?? (spec ? `\\operatorname{${spec.name}}` : texSymbol(name))
  if (args.length === 1 && isBareCaret(args[0])) return `${head}(${CARET_TEX}\\vphantom{0})`
  return `${head}\\left(${args.map((arg) => texAt(arg, 0)).join(', ')}\\right)`
}

const SETTINGS_COLOR = '#8d9db3'
const ALL_OPTIONS: OptionSpec[] = [...GRAPH_OPTIONS, ...FUNCTIONS.flatMap((spec) => spec.options)]

/** A settings block as KaTeX, muted so the calculation stays in front: `{Domain: 0 … 5}`. */
export function blockTex(block: OptionBlock, caretAfter = false): string {
  const parts = block.entries.map(entryTex).filter((part) => part !== '')
  const body = parts.length > 0 ? parts.join(',\\ ') : '\\,'
  return `\\;${block.caretBefore ? CARET_TEX : ''}\\textcolor{${SETTINGS_COLOR}}{\\{${body}\\}}${caretAfter ? CARET_TEX : ''}`
}

function entryTex(entry: OptionEntry): string {
  const clean = entry.raw.replaceAll(MATH_CARET, '')
  const found = entry.raw.indexOf(MATH_CARET)
  const caret = found < 0 ? null : found
  if (!clean.trim()) return caret === null ? '' : CARET_TEX
  const pair = OPTION_PARTS.exec(clean)
  if (pair) {
    const keyStart = (pair[1] ?? '').length
    const key = pair[2] ?? ''
    const valueStart = keyStart + key.length + (pair[3] ?? '').length
    const keyCaret = caret !== null && caret <= keyStart + key.length ? Math.max(0, caret - keyStart) : null
    const valueCaret = caret !== null && keyCaret === null ? Math.max(0, caret - valueStart) : null
    return `${keyTex(key, keyCaret)}\\colon ${valueTex(key, pair[4] ?? '', valueCaret)}`
  }
  const lead = clean.length - clean.trimStart().length
  if (OPTION_FLAG.test(clean)) return keyTex(clean.trim(), caret === null ? null : Math.max(0, caret - lead))
  return valueTex('', clean, caret)
}

function keyTex(key: string, caret: number | null): string {
  if (caret !== null) return withCaret(key, caret, (part) => `\\text{${escapeText(part)}}`)
  const known = canonicalOptionName(key)
  if (known) return `\\mathrm{${known}}`
  if (/^[A-Za-z][A-Za-z0-9]?$/.test(key) || isGreekName(key) || greekCharName(key)) return texSymbol(naming(key))
  return `\\text{${escapeText(key)}}`
}

function valueTex(key: string, value: string, caret: number | null): string {
  const text = (caret === null ? value : `${value.slice(0, caret)}${MATH_CARET}${value.slice(caret)}`).trim()
  const bare = text.replaceAll(MATH_CARET, '')
  if (!bare) return caret === null ? '\\square' : CARET_TEX
  const kind = findOption(ALL_OPTIONS, key)?.kind ?? null
  if (kind === 'color') {
    const color = parseColor(bare)
    return color ? `\\textcolor{${color}}{\\blacksquare}\\,${textWithCaret(text)}` : textWithCaret(text)
  }
  if (kind === 'boolean') return textWithCaret(text)
  const range = splitRange(text)
  if (range) return `${rangeSideTex(range.min)} \\ldots ${rangeSideTex(range.max)}`
  return exprOrText(text)
}

function rangeSideTex(text: string): string {
  if (!text.replaceAll(MATH_CARET, '').trim()) return text.includes(MATH_CARET) ? CARET_TEX : '\\square'
  return exprOrText(text)
}

function exprOrText(text: string): string {
  try {
    return tex(parseExpr(text))
  } catch {
    return textWithCaret(text)
  }
}

function textWithCaret(text: string): string {
  const at = text.indexOf(MATH_CARET)
  const bare = text.replaceAll(MATH_CARET, '')
  if (at < 0) return `\\text{${escapeText(bare)}}`
  return withCaret(bare, at, (part) => `\\text{${escapeText(part)}}`)
}

function withCaret(text: string, caret: number, wrap: (part: string) => string): string {
  const at = Math.max(0, Math.min(caret, text.length))
  const before = text.slice(0, at)
  const after = text.slice(at)
  return `${before ? wrap(before) : ''}${CARET_TEX}${after ? wrap(after) : ''}`
}

function escapeText(text: string): string {
  return text.replace(/[\\{}$&#^_%~]/g, (ch) => (ch === '\\' ? '\\textbackslash{}' : ch === '~' ? '\\textasciitilde{}' : ch === '^' ? '\\textasciicircum{}' : `\\${ch}`))
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
      return [`${plainAt(peelGroup(e.num), P_MUL)}/${plainAt(peelGroup(e.den), P_MUL)}`, P_MUL]
    case 'pow':
      if (e.exp.type === 'rat' && e.exp.n === 1n && e.exp.d === 2n) return [`sqrt(${plainAt(e.base, 0)})`, P_ATOM]
      return [`${plainAt(e.base, P_POW + 1)}^${plainAt(e.exp, P_POW)}`, P_POW]
    case 'call':
      return [plainCall(e), P_ATOM]
    case 'vec':
      return [`[${e.args.map((arg) => plainAt(arg, 0)).join(', ')}]`, P_ATOM]
    case 'mat':
      return [`[${e.rows.map((row) => `[${row.map((arg) => plainAt(arg, 0)).join(', ')}]`).join(', ')}]`, P_ATOM]
    case 'eq':
      return [`${plainAt(e.left, 0)} = ${plainAt(e.right, 0)}`, 0]
    case 'group':
      return [`(${plainAt(e.body, 0)})`, P_ATOM]
    case 'caret':
      return [e.body ? plainAt(e.body, 0) : '', P_ATOM]
  }
}

function plainCall(e: CallExpr): string {
  if (e.name === 'sqrt' && e.args.length === 1 && !e.options) return `sqrt(${plainAt(e.args[0], 0)})`
  const spec = functionByKernel(e.name)
  const shown = spec && !e.options ? unbind(spec, e.args) : { args: e.args, settings: [] }
  const block = e.options ? `{${e.options.raw}}` : shown.settings.length > 0 ? `{${shown.settings.join(', ')}}` : ''
  return `${spec?.name ?? e.name}(${shown.args.map((arg) => plainAt(arg, 0)).join(', ')})${block}`
}

/** A bound call written the way it is typed: diff(x^3, x, 2) is Derivative(x^3, x){Order: 2}. */
function unbind(spec: FunctionSpec, args: Expr[]): { args: Expr[]; settings: string[] } {
  const settings: string[] = []
  const put = (name: string, value: string) => {
    const option = spec.options.find((item) => item.name === name)
    if (option && String(option.default) === value) return
    settings.push(`${name}: ${value}`)
  }
  const text = (e: Expr | undefined) => (e ? plainAt(e, 0) : '')
  switch (spec.kernel) {
    case 'diff': {
      if (args.length === 3) {
        const order = args[2]?.type === 'rat' && args[2].d === 1n ? Number(args[2].n) : null
        if (order !== null && order >= 1 && order <= 6) put('Order', String(order))
        else put('At', text(args[2]))
        return { args: args.slice(0, 2), settings }
      }
      if (args.length !== 4) break
      put('Order', text(args[2]))
      put('At', text(args[3]))
      return { args: args.slice(0, 2), settings }
    }
    case 'integrate':
    case 'zeros':
    case 'fmin':
    case 'fmax':
      if (args.length !== 4) break
      put('Domain', `${text(args[2])}..${text(args[3])}`)
      return { args: args.slice(0, 2), settings }
    case 'series':
      if (args.length !== 4) break
      put('Point', text(args[2]))
      put('Order', text(args[3]))
      return { args: args.slice(0, 2), settings }
    case 'decimal':
      if (args.length !== 2) break
      put('Digits', text(args[1]))
      return { args: args.slice(0, 1), settings }
    case 'log':
      if (args.length !== 2) break
      put('Base', text(args[1]))
      return { args: args.slice(0, 1), settings }
  }
  return { args, settings: [] }
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

export function exprKey(e: Expr): string {
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
    case 'group':
      return exprKey(e.body)
    case 'caret':
      return e.body ? exprKey(e.body) : 'caret'
  }
}

const DIRECT_TRIG = new Set(['sin', 'cos', 'tan'])
const ANGLE_LITERAL_CALLS = new Set(['sqrt', 'abs', 'ln', 'log', 'exp'])

/** Rewrite constant trig inputs from one angle unit to the other, so `sin(pi)` becomes `sin(180)`. */
export function convertAngles(e: Expr, from: AngleMode, to: AngleMode): Expr {
  const walked = convertChildren(e, from, to)
  if (walked.type !== 'call' || walked.args.length !== 1 || !DIRECT_TRIG.has(walked.name)) return walked
  const arg = walked.args[0]
  if (!arg) return walked
  return { ...walked, args: [rescaleAngle(arg, from, to)] }
}

function convertChildren(e: Expr, from: AngleMode, to: AngleMode): Expr {
  const visit = (child: Expr) => convertAngles(child, from, to)
  switch (e.type) {
    case 'rat':
    case 'dec':
    case 'sym':
      return e
    case 'add':
    case 'mul':
      return { ...e, args: e.args.map(visit) }
    case 'div':
      return { type: 'div', num: visit(e.num), den: visit(e.den) }
    case 'pow':
      return { type: 'pow', base: visit(e.base), exp: visit(e.exp) }
    case 'call':
      return { ...e, args: e.args.map(visit) }
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
  }
}

/** Numbers, pi, and arithmetic are an angle written in the current unit. A trig call already yields a pure number. */
function isAngleLiteral(e: Expr): boolean {
  switch (e.type) {
    case 'rat':
    case 'dec':
      return true
    case 'sym':
      return bareConstant(e)
    case 'add':
    case 'mul':
      return e.args.every(isAngleLiteral)
    case 'div':
      return isAngleLiteral(e.num) && isAngleLiteral(e.den)
    case 'pow':
      return isAngleLiteral(e.base) && isAngleLiteral(e.exp)
    case 'group':
      return isAngleLiteral(e.body)
    case 'caret':
      return e.body ? isAngleLiteral(e.body) : false
    case 'call':
      return ANGLE_LITERAL_CALLS.has(e.name) && e.args.every(isAngleLiteral)
    default:
      return false
  }
}

function rescaleAngle(arg: Expr, from: AngleMode, to: AngleMode): Expr {
  const body = peelGroup(arg)
  if (body.type === 'add') return normalize({ type: 'add', args: body.args.map((term) => rescaleTerm(term, from, to)) }, 'rad')
  if (isAngleLiteral(body)) return scaleAngle(body, from)
  return rescaleNegativeSum(body, from, to) ?? arg
}

function rescaleTerm(term: Expr, from: AngleMode, to: AngleMode): Expr {
  const body = peelGroup(term)
  if (isAngleLiteral(body)) return scaleAngle(body, from)
  return rescaleNegativeSum(body, from, to) ?? term
}

function rescaleNegativeSum(e: Expr, from: AngleMode, to: AngleMode): Expr | null {
  if (e.type !== 'mul' || e.args.length !== 2) return null
  const sign = e.args[0]
  if (sign.type !== 'rat' || sign.n !== -1n || sign.d !== 1n) return null
  const inner = peelGroup(e.args[1])
  if (inner.type !== 'add') return null
  return normalize({ type: 'mul', args: [rat(-1n), { type: 'add', args: inner.args.map((part) => rescaleTerm(part, from, to)) }] }, 'rad')
}

function scaleAngle(e: Expr, from: AngleMode): Expr {
  const pi: Expr = { type: 'sym', name: 'pi' }
  const factor: Expr = from === 'rad' ? { type: 'div', num: rat(180n), den: pi } : { type: 'div', num: pi, den: rat(180n) }
  const scaled = normalize({ type: 'mul', args: [peelGroup(e), factor] }, 'rad')
  if (!containsDecimal(e) || scaled.type === 'rat') return scaled
  const n = evalConst(scaled, 'rad')
  if (n === null || !Number.isFinite(n)) return scaled
  return snap(n) ?? scaled
}

function containsDecimal(e: Expr): boolean {
  let found = false
  walk(e, (node) => {
    if (node.type === 'dec') found = true
  })
  return found
}

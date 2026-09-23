// Turn pasted LaTeX into the console's ascii, and move the caret through a fraction
// the way a structural editor does: up and down between the two sides, sideways across a side.

import { ALL_OPTION_NAMES, FUNCTION_WORDS, optionKey } from './functions'

export const MATH_CARET = '\u0001'

const GREEK = [
  'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta', 'iota', 'kappa', 'lambda', 'mu',
  'nu', 'xi', 'pi', 'rho', 'sigma', 'tau', 'upsilon', 'phi', 'chi', 'psi', 'omega',
]

const FUNC_WORDS: Record<string, string> = {
  sin: 'sin',
  cos: 'cos',
  tan: 'tan',
  ln: 'ln',
  log: 'log',
  exp: 'exp',
  abs: 'abs',
  arcsin: 'asin',
  arccos: 'acos',
  arctan: 'atan',
  asin: 'asin',
  acos: 'acos',
  atan: 'atan',
}

const SKIP_WORDS = new Set([
  'displaystyle', 'textstyle', 'scriptstyle', 'scriptscriptstyle', 'limits', 'nolimits',
  'bigl', 'bigr', 'Bigl', 'Bigr', 'biggl', 'biggr', 'Biggl', 'Biggr', 'big', 'Big', 'bigg', 'Bigg',
  'quad', 'qquad', 'blacksquare', 'square',
])

const KNOWN = new Set<string>([
  ...GREEK,
  ...Object.keys(FUNC_WORDS),
  ...FUNCTION_WORDS,
  ...SKIP_WORDS,
  'frac', 'dfrac', 'tfrac', 'sqrt', 'cdot', 'times', 'ast', 'div', 'left', 'right',
  'dot', 'ddot', 'dddot', 'mathrm', 'mathbf', 'text', 'operatorname', 'infty',
  'colon', 'ldots', 'dots', 'cdots', 'textcolor', 'color', 'lbrace', 'rbrace',
])

const SETTING_FLAGS = new Set(ALL_OPTION_NAMES.map(optionKey))

/** `Domain: 0..5` or a lone flag such as `Dashed`: the inside of a settings block, not a TeX group. */
function looksLikeSettings(body: string): boolean {
  const text = body.trim()
  if (/^[A-Za-zα-ω][A-Za-z0-9α-ω _-]*\s*(?::|->|=)/.test(text)) return true
  const words = text.split(',').map((part) => optionKey(part))
  return words.length > 0 && words.every((word) => SETTING_FLAGS.has(word))
}

class IncompleteLatex extends Error {}

/** Convert a pasted LaTeX formula into ascii the kernel can parse. A string with no commands is unchanged. */
export function latexToSource(input: string): string {
  if (!input.includes('\\')) return input
  try {
    return tidySettings(convertLatex(input))
  } catch (error) {
    if (error instanceof IncompleteLatex) return input
    return input
  }
}

/** The preview spaces a settings block out with `\;` and `\colon`; pasted back, it should read the way it is typed. */
function tidySettings(source: string): string {
  return source.replace(/([)\]])\s+\{/g, '$1{').replace(/\s*\.\.\s*/g, '..').replace(/:\s+/g, ': ')
}

function convertLatex(src: string): string {
  let i = 0
  let out = ''
  while (i < src.length) {
    const ch = src[i] ?? ''
    if (ch === '\\') {
      const cmd = readCommand(src, i)
      if (cmd.next <= i) {
        i += 1
        continue
      }
      if (isOpenPrefix(cmd.name, cmd.next >= src.length)) throw new IncompleteLatex()
      const applied = applyCommand(cmd.name, src, cmd.next)
      out += applied.text
      i = applied.next
      continue
    }
    if (ch === '^' || ch === '_') {
      const arg = readArg(src, i + 1)
      if (!arg) throw new IncompleteLatex()
      const inner = convertLatex(arg.body).trim()
      out += /[+\-*/^,=]/.test(inner) ? `${ch}(${inner})` : `${ch}${inner}`
      i = arg.next
      continue
    }
    if (ch === '{') {
      const body = readBrace(src, i)
      if (!body) throw new IncompleteLatex()
      const inner = convertLatex(body.body)
      out += looksLikeSettings(inner) ? `{${inner}}` : `(${inner})`
      i = body.next
      continue
    }
    out += ch
    i += 1
  }
  return out
}

function isOpenPrefix(name: string, atEnd: boolean): boolean {
  if (!atEnd || !name || KNOWN.has(name)) return false
  return [...KNOWN].some((cmd) => cmd.startsWith(name) && cmd.length > name.length)
}

function readCommand(src: string, i: number): { name: string; next: number } {
  let j = i + 1
  if (j >= src.length) return { name: '', next: j }
  if (/[A-Za-z]/.test(src[j] ?? '')) {
    const start = j
    while (j < src.length && /[A-Za-z]/.test(src[j] ?? '')) j += 1
    return { name: src.slice(start, j), next: j }
  }
  return { name: src[j] ?? '', next: j + 1 }
}

function readBrace(src: string, i: number): { body: string; next: number } | null {
  if (src[i] !== '{') return null
  let depth = 0
  for (let j = i; j < src.length; j += 1) {
    if (src[j] === '\\') {
      j += 1
      continue
    }
    if (src[j] === '{') depth += 1
    else if (src[j] === '}') {
      depth -= 1
      if (depth === 0) return { body: src.slice(i + 1, j), next: j + 1 }
    }
  }
  return null
}

function readArg(src: string, i: number): { body: string; next: number } | null {
  while (src[i] === ' ') i += 1
  if (i >= src.length) return null
  if (src[i] === '{') return readBrace(src, i)
  if (src[i] === '\\') {
    const cmd = readCommand(src, i)
    return { body: src.slice(i, cmd.next), next: cmd.next }
  }
  return { body: src[i] ?? '', next: i + 1 }
}

function applyCommand(name: string, src: string, i: number): { text: string; next: number } {
  if (name === 'frac' || name === 'dfrac' || name === 'tfrac') {
    const num = readArg(src, i)
    if (!num) throw new IncompleteLatex()
    const den = readArg(src, num.next)
    if (!den) throw new IncompleteLatex()
    return { text: `(${convertLatex(num.body)})/(${convertLatex(den.body)})`, next: den.next }
  }
  if (name === 'sqrt') {
    let root = ''
    if (src[i] === '[') {
      const close = src.indexOf(']', i + 1)
      if (close < 0) throw new IncompleteLatex()
      root = src.slice(i + 1, close)
      i = close + 1
    }
    const arg = readArg(src, i)
    if (!arg) throw new IncompleteLatex()
    const body = convertLatex(arg.body)
    if (root.trim()) return { text: `(${body})^(1/(${convertLatex(root)}))`, next: arg.next }
    return { text: `sqrt(${body})`, next: arg.next }
  }
  if (name === 'cdot' || name === 'times' || name === 'ast') return { text: '*', next: i }
  if (name === 'div') return { text: '/', next: i }
  if (name === ',' || name === ';' || name === ':' || name === ' ' || name === '\\' || name === 'quad' || name === 'qquad') return { text: ' ', next: i }
  if (name === '!') return { text: '', next: i }
  if (name === 'left' || name === 'right') {
    if (i >= src.length) throw new IncompleteLatex()
    const delim = readDelim(src, i)
    return { text: delim.emit, next: delim.next }
  }
  if (name === 'dot' || name === 'ddot' || name === 'dddot') {
    const arg = readArg(src, i)
    if (!arg) throw new IncompleteLatex()
    const inner = convertLatex(arg.body).trim()
    const primes = name === 'dot' ? "'" : name === 'ddot' ? "''" : "'''"
    const text = /^[A-Za-z][A-Za-z0-9]*$/.test(inner) ? `${inner}${primes}` : `(${inner})${primes}`
    return { text, next: arg.next }
  }
  if (name === 'mathrm' || name === 'mathbf' || name === 'text' || name === 'operatorname') {
    const arg = readArg(src, i)
    if (!arg) throw new IncompleteLatex()
    return { text: convertLatex(arg.body), next: arg.next }
  }
  if (name === 'textcolor' || name === 'color') {
    const color = readArg(src, i)
    if (!color) throw new IncompleteLatex()
    if (name === 'color') return { text: '', next: color.next }
    const arg = readArg(src, color.next)
    if (!arg) throw new IncompleteLatex()
    return { text: convertLatex(arg.body), next: arg.next }
  }
  if (name === 'colon') return { text: ':', next: i }
  if (name === 'ldots' || name === 'dots' || name === 'cdots') return { text: '..', next: i }
  if (SKIP_WORDS.has(name)) return { text: '', next: i }
  if (FUNC_WORDS[name]) return { text: FUNC_WORDS[name], next: i }
  if (GREEK.includes(name) || FUNCTION_WORDS.includes(name.toLowerCase())) return { text: name, next: i }
  if (name === '{' || name === 'lbrace') {
    const close = matchEscapedBrace(src, i)
    if (close) {
      const inner = convertLatex(src.slice(i, close.start))
      if (looksLikeSettings(inner)) return { text: `{${inner}}`, next: close.next }
    }
    return { text: '(', next: i }
  }
  if (name === '}' || name === 'rbrace') return { text: ')', next: i }
  return { text: name, next: i }
}

/** The `\}` that closes a `\{` whose body starts at `i`, skipping nested escaped pairs. */
function matchEscapedBrace(src: string, i: number): { start: number; next: number } | null {
  let depth = 0
  let j = i
  while (j < src.length) {
    if (src[j] !== '\\') {
      j += 1
      continue
    }
    const cmd = readCommand(src, j)
    if (cmd.name === '{' || cmd.name === 'lbrace') depth += 1
    else if (cmd.name === '}' || cmd.name === 'rbrace') {
      if (depth === 0) return { start: j, next: cmd.next }
      depth -= 1
    }
    j = Math.max(cmd.next, j + 1)
  }
  return null
}

function readDelim(src: string, i: number): { emit: string; next: number } {
  while (src[i] === ' ') i += 1
  if (src[i] === '\\') {
    const cmd = readCommand(src, i)
    if (cmd.name === '{' || cmd.name === 'lbrace') return { emit: '(', next: cmd.next }
    if (cmd.name === '}' || cmd.name === 'rbrace') return { emit: ')', next: cmd.next }
    if (cmd.name === 'langle') return { emit: '(', next: cmd.next }
    if (cmd.name === 'rangle') return { emit: ')', next: cmd.next }
    if (cmd.name === 'vert' || cmd.name === '|') return { emit: '|', next: cmd.next }
    return { emit: '', next: cmd.next }
  }
  const ch = src[i] ?? ''
  const next = i + 1
  if (ch === '.' || ch === '') return { emit: '', next }
  if (ch === '{') return { emit: '(', next }
  if (ch === '}') return { emit: ')', next }
  return { emit: ch, next }
}

interface FractionSpan {
  slash: number
  numStart: number
  denStart: number
  denExit: number
  end: number
}

const IDENT_START = /[A-Za-zαβγδεζηθικλμνξπρστυφχψω]/
const IDENT_PART = /[A-Za-z0-9αβγδεζηθικλμνξπρστυφχψω]/

interface Slot {
  start: number
  end: number
  kind: 'fraction' | 'exponent' | 'subscript'
  move: (at: number, dir: 'left' | 'right' | 'up' | 'down') => number | null
}

/**
 * Move through the innermost fraction, exponent, or subscript.
 * A name such as sigma is one step. Other arrows step one character.
 */
export function moveMathCursor(source: string, cursor: number, dir: 'left' | 'right' | 'up' | 'down'): number {
  const at = Math.max(0, Math.min(cursor, source.length))
  const slots = findSlots(source)
    .filter((slot) => at >= slot.start && at <= slot.end)
    .sort((a, b) => a.end - a.start - (b.end - b.start))
  for (const slot of slots) {
    const next = slot.move(at, dir)
    if (next !== null) return next
  }
  if (dir === 'left' || dir === 'right') return jumpToken(source, at, dir)
  return at
}

/**
 * The editor opens `^()`, `_{}`, and `/()` as soon as `^`, `_`, or `/` is typed, and the preview
 * hides those brackets. A comma, an `=`, or a `)` never belongs inside one: typing `1/2, x` gives
 * `1/(2), x` and `x^2 = 4` gives `x^(2) = 4`. While the innermost open bracket at the caret is
 * one of those slots, step past its closer. A call, a list, a block, or a plain `(...)` group
 * stops the walk, so `x^(Mod(7, 3))` still types.
 */
export function exitSlots(source: string, cursor: number): number {
  const lineStart = source.lastIndexOf('\n', cursor - 1) + 1
  const lineEndAt = source.indexOf('\n', cursor)
  const lineEnd = lineEndAt < 0 ? source.length : lineEndAt
  let at = cursor
  for (;;) {
    const stack: number[] = []
    for (let i = lineStart; i < at; i += 1) {
      const ch = source[i]
      if (ch === '(' || ch === '[' || ch === '{') stack.push(i)
      else if (ch === ')' || ch === ']' || ch === '}') stack.pop()
    }
    const open = stack[stack.length - 1]
    if (open === undefined) return at
    const before = source[open - 1]
    const slot = before === '^' || (before === '_' && source[open] === '{') || (before === '/' && source[open] === '(')
    if (!slot) return at
    let depth = 0
    let close = -1
    for (let i = open; i < lineEnd && close < 0; i += 1) {
      const ch = source[i]
      if (ch === '(' || ch === '[' || ch === '{') depth += 1
      else if (ch === ')' || ch === ']' || ch === '}') {
        depth -= 1
        if (depth === 0) close = i
      }
    }
    if (close < 0) return at
    at = close + 1
  }
}

/** Append the closers for groups that are still open, so Right can leave cos(30. */
export function closeOpenGroups(source: string): string {
  const stack: string[] = []
  const closer: Record<string, string> = { '(': ')', '[': ']', '{': '}' }
  const opener: Record<string, string> = { ')': '(', ']': '[', '}': '{' }
  for (const ch of source) {
    if (ch === '(' || ch === '[' || ch === '{') stack.push(ch)
    else if (ch === ')' || ch === ']' || ch === '}') {
      if (stack[stack.length - 1] !== opener[ch]) return source
      stack.pop()
    }
  }
  if (stack.length === 0) return source
  let out = source
  for (let i = stack.length - 1; i >= 0; i -= 1) out += closer[stack[i] ?? ''] ?? ''
  return out
}

function findSlots(source: string): Slot[] {
  return [...findFractions(source), ...findExponents(source), ...findSubscripts(source)]
}

function findFractions(source: string): Slot[] {
  const spans: Slot[] = []
  for (let i = 0; i < source.length; i += 1) {
    if (source[i] !== '/' || source[i + 1] === '/' || source[i - 1] === '/') continue
    const den = denominatorSpan(source, i)
    const frac: FractionSpan = { slash: i, numStart: numeratorStart(source, i), denStart: den.start, denExit: den.exit, end: den.end }
    spans.push({
      start: frac.numStart,
      end: frac.end,
      kind: 'fraction',
      move: (at, dir) => moveFraction(frac, at, dir),
    })
  }
  return spans
}

function moveFraction(frac: FractionSpan, at: number, dir: 'left' | 'right' | 'up' | 'down'): number | null {
  const inNumerator = at >= frac.numStart && at <= frac.slash
  const inDenominator = at >= frac.denStart && at <= frac.denExit
  if (dir === 'down' && inNumerator) return frac.denExit
  if (dir === 'up' && (inDenominator || (at > frac.slash && at < frac.denStart))) return frac.slash
  if (dir === 'right' && at === frac.slash) return frac.end
  if (dir === 'right' && at === frac.denExit && frac.denExit !== frac.end) return frac.end
  if (dir === 'right' && at > frac.slash && at < frac.denStart) return frac.denStart
  if (dir === 'left' && at === frac.end) return frac.slash
  if (dir === 'left' && (at === frac.denStart || (at > frac.slash && at < frac.denStart))) return frac.slash
  return null
}

function findExponents(source: string): Slot[] {
  const spans: Slot[] = []
  for (let i = 0; i < source.length; i += 1) {
    if (source[i] !== '^') continue
    let j = i + 1
    while (j < source.length && /\s/.test(source[j] ?? '')) j += 1
    if (j >= source.length) continue
    if (source[j] === '(') {
      const close = matchDelim(source, j, '(', ')')
      const end = close < 0 ? source.length : close + 1
      const contentStart = j + 1
      const closeAt = close < 0 ? source.length : close
      spans.push({
        start: i,
        end,
        kind: 'exponent',
        move: (at, dir) => moveWrapped(i, contentStart, closeAt, end, at, dir),
      })
      continue
    }
    const end = atomEnd(source, j)
    spans.push({
      start: i,
      end,
      kind: 'exponent',
      move: (at, dir) => moveWrapped(i, j, end, end, at, dir),
    })
  }
  return spans
}

function findSubscripts(source: string): Slot[] {
  const spans: Slot[] = []
  for (let i = 0; i < source.length; i += 1) {
    if (source[i] !== '_') continue
    let j = i + 1
    while (j < source.length && /\s/.test(source[j] ?? '')) j += 1
    if (j >= source.length) continue
    if (source[j] === '{') {
      const close = matchDelim(source, j, '{', '}')
      const end = close < 0 ? source.length : close + 1
      const closeAt = close < 0 ? source.length : close
      spans.push({
        start: i,
        end,
        kind: 'subscript',
        move: (at, dir) => moveWrapped(i, j + 1, closeAt, end, at, dir),
      })
      continue
    }
    let end = j
    if (IDENT_START.test(source[j] ?? '')) {
      while (end < source.length && IDENT_PART.test(source[end] ?? '')) end += 1
    } else end += 1
    spans.push({
      start: i,
      end,
      kind: 'subscript',
      move: (at, dir) => moveWrapped(i, j, end, end, at, dir),
    })
  }
  return spans
}

/** Right at the opener enters the slot. Right at the closer leaves it. Left at the content returns to the opener. */
function moveWrapped(opener: number, contentStart: number, closeAt: number, end: number, at: number, dir: 'left' | 'right' | 'up' | 'down'): number | null {
  if (dir === 'right' && (at === opener || (at > opener && at < contentStart))) return contentStart
  if (dir === 'right' && closeAt !== end && at === closeAt) return end
  if (dir === 'left' && at === contentStart) return opener
  return null
}

function jumpToken(source: string, at: number, dir: 'left' | 'right'): number {
  if (dir === 'right') {
    const run = identRun(source, at)
    if (run && at < run.end) {
      if (source[run.end] === '}' && source[run.start - 1] === '{') return Math.min(source.length, run.end + 1)
      return run.end
    }
    return Math.min(source.length, at + 1)
  }
  if (at > 0) {
    const run = identRun(source, at - 1)
    if (run && at > run.start && at <= run.end) return run.start
  }
  return Math.max(0, at - 1)
}

function identRun(source: string, index: number): { start: number; end: number } | null {
  if (!IDENT_PART.test(source[index] ?? '')) return null
  let start = index
  while (start > 0 && IDENT_PART.test(source[start - 1] ?? '')) start -= 1
  if (!IDENT_START.test(source[start] ?? '')) return null
  let end = index + 1
  while (end < source.length && IDENT_PART.test(source[end] ?? '')) end += 1
  return { start, end }
}

function numeratorStart(source: string, slash: number): number {
  let i = slash - 1
  let depth = 0
  while (i >= 0) {
    const ch = source[i] ?? ''
    if (ch === ')' || ch === ']' || ch === '}') {
      depth += 1
      i -= 1
      continue
    }
    if (ch === '(' || ch === '[' || ch === '{') {
      if (depth === 0) break
      depth -= 1
      i -= 1
      continue
    }
    if (depth === 0 && (ch === '+' || ch === '=' || ch === ',')) break
    if (depth === 0 && ch === '-' && binaryMinus(source, i)) break
    i -= 1
  }
  let start = i + 1
  while (start < slash && /\s/.test(source[start] ?? '')) start += 1
  return start
}

function binaryMinus(source: string, index: number): boolean {
  let j = index - 1
  while (j >= 0 && /\s/.test(source[j] ?? '')) j -= 1
  if (j < 0) return false
  return !'+-*/^=(,['.includes(source[j] ?? '')
}

function denominatorSpan(source: string, slash: number): { start: number; exit: number; end: number } {
  let i = slash + 1
  while (i < source.length && /\s/.test(source[i] ?? '')) i += 1
  if (i >= source.length) return { start: i, exit: i, end: i }
  if (source[i] === '(') {
    const close = matchDelim(source, i, '(', ')')
    if (close < 0) return { start: i + 1, exit: source.length, end: source.length }
    return { start: i + 1, exit: close, end: close + 1 }
  }
  const end = atomEnd(source, i)
  return { start: i, exit: end, end }
}

function matchDelim(source: string, open: number, left: string, right: string): number {
  let depth = 0
  for (let i = open; i < source.length; i += 1) {
    if (source[i] === left) depth += 1
    else if (source[i] === right) {
      depth -= 1
      if (depth === 0) return i
    }
  }
  return -1
}

function atomEnd(source: string, start: number): number {
  let i = start
  if ((source[i] === '+' || source[i] === '-') && i + 1 < source.length) i += 1
  if (source[i] === '(' || source[i] === '[') {
    const right = source[i] === '(' ? ')' : ']'
    const close = matchDelim(source, i, source[i] ?? '(', right)
    i = close < 0 ? source.length : close + 1
  } else if (/[0-9.]/.test(source[i] ?? '')) {
    const matched = /^(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?/i.exec(source.slice(i))
    i += matched ? matched[0].length : 1
  } else if (IDENT_START.test(source[i] ?? '')) {
    while (i < source.length && IDENT_PART.test(source[i] ?? '')) i += 1
    if (source[i] === '(') {
      const close = matchDelim(source, i, '(', ')')
      i = close < 0 ? source.length : close + 1
    }
  } else if (i < source.length) i += 1
  while (source[i] === "'") i += 1
  if (source[i] === '_') {
    i += 1
    if (source[i] === '{') {
      const close = matchDelim(source, i, '{', '}')
      i = close < 0 ? source.length : close + 1
    } else {
      while (i < source.length && IDENT_PART.test(source[i] ?? '')) i += 1
    }
    while (source[i] === "'") i += 1
  }
  if (source[i] === '^') return atomEnd(source, i + 1)
  return i
}

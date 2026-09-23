// Turn pasted LaTeX into the console's ascii, and move the caret through a fraction
// the way a structural editor does: up and down between the two sides, sideways across a side.

import { MATH_FUNCTION_NAMES } from './catalog'

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
  'quad', 'qquad',
])

const KNOWN = new Set<string>([
  ...GREEK,
  ...Object.keys(FUNC_WORDS),
  ...MATH_FUNCTION_NAMES,
  ...SKIP_WORDS,
  'frac', 'dfrac', 'tfrac', 'sqrt', 'cdot', 'times', 'ast', 'div', 'left', 'right',
  'dot', 'ddot', 'dddot', 'mathrm', 'mathbf', 'text', 'operatorname', 'infty',
])

class IncompleteLatex extends Error {}

/** Convert a pasted LaTeX formula into ascii the kernel can parse. A string with no commands is unchanged. */
export function latexToSource(input: string): string {
  if (!input.includes('\\')) return input
  try {
    return convertLatex(input)
  } catch (error) {
    if (error instanceof IncompleteLatex) return input
    return input
  }
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
      out += `(${convertLatex(body.body)})`
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
  if (SKIP_WORDS.has(name)) return { text: '', next: i }
  if (FUNC_WORDS[name]) return { text: FUNC_WORDS[name], next: i }
  if (GREEK.includes(name) || MATH_FUNCTION_NAMES.includes(name)) return { text: name, next: i }
  if (name === '{' || name === 'lbrace') return { text: '(', next: i }
  if (name === '}' || name === 'rbrace') return { text: ')', next: i }
  return { text: name, next: i }
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

/** Move through the fraction that contains the caret. Other arrows step one character. */
export function moveMathCursor(source: string, cursor: number, dir: 'left' | 'right' | 'up' | 'down'): number {
  const at = Math.max(0, Math.min(cursor, source.length))
  const frac = innermostFraction(source, at)
  if (!frac) {
    if (dir === 'left') return Math.max(0, at - 1)
    if (dir === 'right') return Math.min(source.length, at + 1)
    return at
  }
  const inNumerator = at >= frac.numStart && at <= frac.slash
  const inDenominator = at >= frac.denStart && at <= frac.denExit
  if (dir === 'down' && inNumerator) return frac.denExit
  if (dir === 'up' && (inDenominator || (at > frac.slash && at < frac.denStart))) return frac.slash
  if (dir === 'right' && at === frac.slash) return frac.end
  if (dir === 'right' && at === frac.denExit && frac.denExit !== frac.end) return frac.end
  if (dir === 'right' && at > frac.slash && at < frac.denStart) return frac.denStart
  if (dir === 'left' && at === frac.end) return frac.slash
  if (dir === 'left' && (at === frac.denStart || (at > frac.slash && at < frac.denStart))) return frac.slash
  if (dir === 'left') return Math.max(0, at - 1)
  if (dir === 'right') return Math.min(source.length, at + 1)
  return at
}

function innermostFraction(source: string, cursor: number): FractionSpan | null {
  const hits = findFractions(source).filter((frac) => cursor >= frac.numStart && cursor <= frac.end)
  if (hits.length === 0) return null
  hits.sort((a, b) => a.end - a.numStart - (b.end - b.numStart) || b.slash - a.slash)
  return hits[0] ?? null
}

function findFractions(source: string): FractionSpan[] {
  const spans: FractionSpan[] = []
  for (let i = 0; i < source.length; i += 1) {
    if (source[i] !== '/' || source[i + 1] === '/' || source[i - 1] === '/') continue
    const den = denominatorSpan(source, i)
    spans.push({ slash: i, numStart: numeratorStart(source, i), denStart: den.start, denExit: den.exit, end: den.end })
  }
  return spans
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
  } else if (/[A-Za-z]/.test(source[i] ?? '')) {
    while (i < source.length && /[A-Za-z0-9]/.test(source[i] ?? '')) i += 1
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
      while (i < source.length && /[A-Za-z0-9]/.test(source[i] ?? '')) i += 1
    }
    while (source[i] === "'") i += 1
  }
  if (source[i] === '^') return atomEnd(source, i + 1)
  return i
}

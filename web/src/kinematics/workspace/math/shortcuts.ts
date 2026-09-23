import { COMMANDS } from '../commands'
import { MATH_FUNCTION_NAMES } from './catalog'
import { containsGreekLetter, isGreekName, isGreekPrefix } from './expr'

const CALLS = new Set(['sqrt', 'ln', 'log', 'sin', 'cos', 'tan', 'abs', 'exp', 'asin', 'acos', 'atan', ...MATH_FUNCTION_NAMES])
const MATH_WORDS = new Set([...CALLS, 'pi', 'e'])
const NAME_BEFORE = /[A-Za-z0-9'αβγδεζηθικλμνξπρστυφχψω)]$/

function exactCommand(word: string): boolean {
  const lower = word.toLowerCase()
  return COMMANDS.some((command) => [command.title, ...command.aliases].some((name) => name.toLowerCase() === lower))
}

/** A whole statement word, such as point, stays a statement. Short names and Greek letters stay math. */
function blocksSlot(left: string): boolean {
  const word = /^[A-Za-z][A-Za-z0-9]*$/.exec(left.trim())?.[0]?.toLowerCase() ?? ''
  if (!word || word.length < 3) return false
  if (isGreekName(word) || MATH_WORDS.has(word)) return false
  return exactCommand(word)
}

function slotLeft(left: string): boolean {
  if (blocksSlot(left)) return false
  if (looksLikeMath(left)) return true
  return NAME_BEFORE.test(left.trim())
}

function isStatementCall(text: string): boolean {
  const word = /^([A-Za-z][A-Za-z0-9]*)\s*\(/.exec(text.trim())?.[1]?.toLowerCase() ?? ''
  if (!word || word.length < 3 || isGreekName(word) || MATH_WORDS.has(word)) return false
  return exactCommand(word)
}

function commandPrefix(word: string): boolean {
  return COMMANDS.some((command) => [command.title, ...command.aliases].some((name) => name.toLowerCase().startsWith(word)))
}

let lastFunctionSlash = 0

/** True when the composer should evaluate the line instead of picking a kinematics statement. */
export function looksLikeMath(query: string): boolean {
  const text = query.trim()
  if (!text) return false
  if (isStatementCall(text)) return false
  if (/[0-9=+\-*/^(),'\\_]/.test(text) || containsGreekLetter(text)) return true
  const match = /^[A-Za-z]+/.exec(text)
  if (!match) return false
  const word = match[0].toLowerCase()
  const rest = text.slice(match[0].length)
  const boundary = rest === '' || /^[^A-Za-z]/.test(rest)
  if (boundary && (MATH_WORDS.has(word) || isGreekName(word))) return true
  if (/^[A-Za-z][A-Za-z0-9]*\s*\(/.test(text)) return true
  if (boundary && word.length >= 2 && isGreekPrefix(word) && !commandPrefix(word)) return true
  return false
}

/** `/` with nothing before the caret starts `f(x) = `. A second slash makes `f(x, y) = `. */
export function emptyFunctionShortcut(value: string, cursor: number, key: string): { value: string; cursor: number } | null {
  if (key !== '/') return null
  const left = value.slice(0, cursor)
  const right = value.slice(cursor)
  const now = Date.now()
  const rapid = now - lastFunctionSlash < 500
  if (value === 'f(x) = ' && cursor === value.length) {
    lastFunctionSlash = 0
    const inserted = 'f(x, y) = '
    return { value: inserted, cursor: inserted.length }
  }
  if (left.trim() !== '') return null
  lastFunctionSlash = now
  const inserted = rapid && right.trim() === '' ? 'f(x, y) = ' : 'f(x) = '
  return { value: `${left}${inserted}${right}`, cursor: left.length + inserted.length }
}

/**
 * Turn a finished function name into a call.
 * `sqrt` then space or Tab becomes `sqrt()`, with the caret inside.
 * A digit becomes `sqrt(4)`, still inside the parentheses.
 */
export function expandMathShortcut(value: string, cursor: number, key: string): { value: string; cursor: number } | null {
  if (key === '(' || key === '/') return null
  const typed = key === ' ' || key === 'Tab' || /^[0-9.+\-*,]$/.test(key)
  if (!typed) return null
  const match = /([A-Za-z]+)$/.exec(value.slice(0, cursor))
  if (!match || !CALLS.has(match[1].toLowerCase())) return null
  const start = cursor - match[1].length
  if (start > 0 && /[A-Za-z0-9]/.test(value[start - 1] ?? '')) return null
  const word = match[1].toLowerCase()
  const rest = value.slice(cursor)
  if (key === ' ' || key === 'Tab') {
    const head = `${value.slice(0, start)}${word}()`
    return { value: `${head}${rest}`, cursor: head.length - 1 }
  }
  const head = `${value.slice(0, start)}${word}(`
  return { value: `${head}${key})${rest}`, cursor: head.length + key.length }
}

/**
 * `^` opens an exponent, `_` after a name opens a subscript, and `(` opens a pair.
 * A typed `)` that is already there moves past it. Statement words such as point stay text.
 */
export function insertMathSlot(value: string, cursor: number, end: number, key: string): { value: string; cursor: number } | null {
  const left = value.slice(0, cursor)
  const right = value.slice(Math.max(cursor, end))
  if (key === '^' && slotLeft(left)) {
    const head = `${left}^()`
    return { value: `${head}${right}`, cursor: head.length - 1 }
  }
  if (key === '_' && /[A-Za-z0-9'αβγδεζηθικλμνξπρστυφχψω]$/.test(left) && slotLeft(left)) {
    const head = `${left}_{}`
    return { value: `${head}${right}`, cursor: head.length - 1 }
  }
  if (key === '(' && blocksSlot(left)) return { value, cursor }
  if (key === '(' && (left.trim() === '' || slotLeft(left))) {
    const head = `${left}()`
    return { value: `${head}${right}`, cursor: head.length - 1 }
  }
  if (key === ')' && value[cursor] === ')') return { value, cursor: cursor + 1 }
  return null
}

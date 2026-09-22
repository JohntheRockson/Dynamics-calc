const CALLS = new Set(['sqrt', 'ln', 'log', 'sin', 'cos', 'tan', 'abs', 'exp', 'asin', 'acos', 'atan'])
const MATH_WORDS = new Set([...CALLS, 'pi', 'e', 'solve'])

/** True when the composer should evaluate the line instead of picking a kinematics statement. */
export function looksLikeMath(query: string): boolean {
  const text = query.trim()
  if (!text) return false
  if (/[0-9=+\-*/^(),π]/.test(text)) return true
  const match = /^[A-Za-z]+/.exec(text)
  if (!match) return false
  const word = match[0].toLowerCase()
  const rest = text.slice(match[0].length)
  if (MATH_WORDS.has(word) && (rest === '' || /^[^A-Za-z]/.test(rest))) return true
  if (/^[A-Za-z][A-Za-z0-9]*\s*\(/.test(text)) return true
  return false
}

/** `/` with nothing before the caret starts a blank curve, `f(x) = `. */
export function emptyFunctionShortcut(value: string, cursor: number, key: string): { value: string; cursor: number } | null {
  if (key !== '/') return null
  if (value.slice(0, cursor).trim() !== '') return null
  const inserted = 'f(x) = '
  return { value: `${value.slice(0, cursor)}${inserted}${value.slice(cursor)}`, cursor: cursor + inserted.length }
}

/**
 * Turn a finished function name into a call.
 * `sqrt` then space, Tab, or a digit becomes `sqrt(`.
 */
export function expandMathShortcut(value: string, cursor: number, key: string): { value: string; cursor: number } | null {
  if (key === '(') return null
  const typed = key === ' ' || key === 'Tab' || /^[0-9.+\-*/^,]$/.test(key)
  if (!typed) return null
  const match = /([A-Za-z]+)$/.exec(value.slice(0, cursor))
  if (!match || !CALLS.has(match[1].toLowerCase())) return null
  const start = cursor - match[1].length
  if (start > 0 && /[A-Za-z0-9]/.test(value[start - 1] ?? '')) return null
  const word = match[1].toLowerCase()
  const head = `${value.slice(0, start)}${word}(`
  if (key === ' ' || key === 'Tab') return { value: `${head}${value.slice(cursor)}`, cursor: head.length }
  return { value: `${head}${key}${value.slice(cursor)}`, cursor: head.length + key.length }
}

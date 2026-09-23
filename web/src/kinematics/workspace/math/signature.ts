// What the caret is in the middle of, for the hint card above the editor: which input of which
// function, the settings a closed call takes, which setting inside { }, or which value after
// `Name:`. Plain string scanning, so it is cheap on every keystroke and easy to check.

import { GRAPH_OPTIONS, PLOT_PALETTE, completeFunctionName, findFunction, findOption, optionKey, optionsFor, signatureText, type FunctionSpec, type OptionSpec } from './functions'

export type SignatureHelp =
  /** Inside the parentheses. `active` can pass the last input, which means too many inputs. */
  | { kind: 'params'; spec: FunctionSpec; active: number }
  /** Just after the closing parenthesis of a call that takes settings. */
  | { kind: 'after'; spec: FunctionSpec }
  /** Inside { }, where a setting name goes. */
  | { kind: 'options'; owner: string | null; options: OptionSpec[]; used: string[]; prefix: string; start: number; end: number }
  /** Inside { }, after `Name:`. */
  | { kind: 'value'; owner: string | null; options: OptionSpec[]; key: string; option: OptionSpec | null; prefix: string; start: number; end: number }

export interface NameCompletion {
  start: number
  end: number
  prefix: string
  items: FunctionSpec[]
}

/** One pickable line in the hint card and the edit that picking it makes. */
export interface AssistItem {
  id: string
  label: string
  detail: string
  note: string | null
  swatch: string | null
  edit: { start: number; end: number; text: string; caret: number }
}

export type Assist =
  | { kind: 'complete'; items: AssistItem[] }
  | { kind: 'params'; spec: FunctionSpec; active: number }
  | { kind: 'after'; spec: FunctionSpec }
  | { kind: 'options'; owner: string | null; prefix: string; options: OptionSpec[]; items: AssistItem[] }
  | { kind: 'value'; owner: string | null; options: OptionSpec[]; key: string; option: OptionSpec | null; items: AssistItem[] }

const IDENT_START = /[A-Za-zαβγδεζηθικλμνξπρστυφχψω]/
const IDENT_PART = /[A-Za-z0-9αβγδεζηθικλμνξπρστυφχψω]/

interface Frame {
  kind: 'call' | 'group' | 'list' | 'block' | 'slot'
  open: number
  name: string
  commas: number
  entryStart: number
  owner: BlockOwner | null
}

interface BlockOwner {
  label: string | null
  options: OptionSpec[]
}

interface ClosedCall {
  name: string
  open: number
  close: number
}

function identBefore(line: string, index: number): string | null {
  let start = index
  while (start > 0 && IDENT_PART.test(line[start - 1] ?? '')) start -= 1
  while (start < index && !IDENT_START.test(line[start] ?? '')) start += 1
  return start < index ? line.slice(start, index) : null
}

function previousSolid(line: string, index: number): number {
  let i = index - 1
  while (i >= 0 && /\s/.test(line[i] ?? '')) i -= 1
  return i
}

function nextSolid(line: string, index: number): number {
  let i = index
  while (i < line.length && /\s/.test(line[i] ?? '')) i += 1
  return i
}

/** The comma-separated pieces between two brackets, split only at the top level. */
function topLevelParts(text: string): string[] {
  const parts: string[] = []
  let depth = 0
  let start = 0
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i]
    if (ch === '(' || ch === '[' || ch === '{') depth += 1
    else if (ch === ')' || ch === ']' || ch === '}') depth -= 1
    else if (ch === ',' && depth === 0) {
      parts.push(text.slice(start, i))
      start = i + 1
    }
  }
  parts.push(text.slice(start))
  return parts.map((part) => part.trim())
}

function rangeOption(name: string, description: string): OptionSpec {
  return { name, kind: 'range', description, default: null, example: '0..2*pi' }
}

/** Variables a call names in its inputs, which {x: 0..5} can give a range. */
function callVariables(spec: FunctionSpec, args: string[]): string[] {
  if (!spec.variableRanges) return []
  const names: string[] = []
  spec.params.forEach((param, index) => {
    if (param.kind !== 'variable') return
    const arg = args[index] ?? ''
    const list = /^\[(.*)\]$/.exec(arg)
    for (const item of list ? topLevelParts(list[1] ?? '') : [arg]) {
      if (/^[A-Za-zα-ω][A-Za-z0-9α-ω]*$/.test(item) && !names.includes(item)) names.push(item)
    }
  })
  return names
}

function mergeOptions(first: OptionSpec[], second: OptionSpec[]): OptionSpec[] {
  const seen = new Set(first.map((option) => optionKey(option.name)))
  return [...first, ...second.filter((option) => !seen.has(optionKey(option.name)))]
}

function blockOwner(line: string, open: number, closed: ClosedCall | null, nested: boolean, drawsLine: boolean): BlockOwner {
  const before = previousSolid(line, open)
  const call = closed && closed.close === before ? findFunction(closed.name) : null
  let options: OptionSpec[] = []
  if (call && closed) {
    const variables = callVariables(call, topLevelParts(line.slice(closed.open + 1, closed.close)))
    const takesDomain = call.options.some((option) => optionKey(option.name) === 'domain')
    const ranges = variables.length > 1 || !takesDomain ? variables : []
    options = mergeOptions([...call.options, ...ranges.map((name) => rangeOption(name, call.kernel === 'solve' ? `Search for ${name} in this range` : `Only draw ${name} in this range`))], optionsFor(call))
  }
  if (!nested && drawsLine) {
    const head = /^\s*[A-Za-zα-ω][A-Za-z0-9α-ω]*\s*\(\s*([^()=]*)\)\s*=/.exec(line)
    const params = head ? topLevelParts(head[1] ?? '').filter((name) => /^[A-Za-zα-ω][A-Za-z0-9α-ω]*$/.test(name)) : []
    options = mergeOptions(options, [...params.map((name) => rangeOption(name, `Only draw ${name} in this range`)), ...GRAPH_OPTIONS])
  }
  const takesSettings = call !== null && (call.options.length > 0 || call.draws)
  return { label: takesSettings ? call.name : drawsLine && !nested ? 'Graph' : null, options }
}

/** Setting names already written in the block that holds the caret, apart from the one being typed. */
function usedKeys(line: string, frame: Frame, at: number): string[] {
  const keys: string[] = []
  let depth = 0
  let start = frame.open + 1
  for (let i = frame.open + 1; i <= line.length; i += 1) {
    const ch = line[i] ?? ''
    const closing = i === line.length || (depth === 0 && ch === '}')
    if (closing || (depth === 0 && ch === ',')) {
      if (at < start || at > i) {
        const part = line.slice(start, i)
        const key = /^([^:=]*?)\s*(?::|->|=)/.exec(part)?.[1] ?? part
        if (key.trim()) keys.push(optionKey(key))
      }
      if (closing) break
      start = i + 1
    } else if (ch === '(' || ch === '[' || ch === '{') depth += 1
    else if (ch === ')' || ch === ']' || ch === '}') depth -= 1
  }
  return keys
}

/** The hint for the caret at `cursor`: which input, which setting, or which value it is on. */
export function signatureAt(source: string, cursor: number): SignatureHelp | null {
  const lineStart = source.lastIndexOf('\n', cursor - 1) + 1
  const lineEndAt = source.indexOf('\n', cursor)
  const line = source.slice(lineStart, lineEndAt < 0 ? source.length : lineEndAt)
  const at = Math.max(0, Math.min(cursor - lineStart, line.length))
  const stack: Frame[] = []
  let closed: ClosedCall | null = null
  let drawsLine = false
  for (let i = 0; i < at; i += 1) {
    const ch = line[i] ?? ''
    const top = stack[stack.length - 1]
    if (ch === '(') {
      const name = identBefore(line, i)
      stack.push({ kind: name ? 'call' : 'group', open: i, name: name ?? '', commas: 0, entryStart: i + 1, owner: null })
    } else if (ch === '[') {
      stack.push({ kind: 'list', open: i, name: '', commas: 0, entryStart: i + 1, owner: null })
    } else if (ch === '{') {
      const before = line[previousSolid(line, i)] ?? ''
      if (before === '_' || before === '^') stack.push({ kind: 'slot', open: i, name: '', commas: 0, entryStart: i + 1, owner: null })
      else stack.push({ kind: 'block', open: i, name: '', commas: 0, entryStart: i + 1, owner: blockOwner(line, i, closed, stack.length > 0, drawsLine) })
    } else if (ch === ')' || ch === ']' || ch === '}') {
      const frame = stack.pop()
      if (ch === ')' && frame?.kind === 'call') closed = { name: frame.name, open: frame.open, close: i }
    } else if (ch === ',' && top) {
      top.commas += 1
      top.entryStart = i + 1
    } else if (ch === '=' && stack.length === 0) {
      drawsLine = true
    }
  }
  const top = stack[stack.length - 1]
  if (top?.kind === 'block' && top.owner) {
    const entry = line.slice(top.entryStart, at)
    const pair = /^(\s*)([^:=]*?)(\s*(?::|->|=)\s*)([\s\S]*)$/.exec(entry)
    let end = at
    while (end < line.length && !/[,}\n]/.test(line[end] ?? '')) end += 1
    if (pair) {
      const key = (pair[2] ?? '').trim()
      const start = top.entryStart + (pair[1] ?? '').length + (pair[2] ?? '').length + (pair[3] ?? '').length
      return { kind: 'value', owner: top.owner.label, options: top.owner.options, key, option: findOption(top.owner.options, key), prefix: (pair[4] ?? '').trimEnd(), start: lineStart + start, end: lineStart + end }
    }
    const lead = entry.length - entry.trimStart().length
    let keyEnd = at
    while (keyEnd < line.length && IDENT_PART.test(line[keyEnd] ?? '')) keyEnd += 1
    return { kind: 'options', owner: top.owner.label, options: top.owner.options, used: usedKeys(line, top, at), prefix: entry.trim(), start: lineStart + top.entryStart + lead, end: lineStart + keyEnd }
  }
  const before = previousSolid(line, at)
  if (closed && closed.close === before && line[nextSolid(line, at)] !== '{') {
    const spec = findFunction(closed.name)
    if (spec && (spec.options.length > 0 || spec.draws)) return { kind: 'after', spec }
  }
  for (let depth = stack.length - 1; depth >= 0; depth -= 1) {
    const frame = stack[depth]
    if (!frame || frame.kind === 'block') return null
    if (frame.kind !== 'call') continue
    const spec = findFunction(frame.name)
    if (!spec) continue
    const last = spec.params.length - 1
    const active = spec.params[last]?.repeats ? Math.min(frame.commas, last) : frame.commas
    return { kind: 'params', spec, active }
  }
  return null
}

/** Function names that finish the word before the caret, once two letters are typed. */
export function completionAt(source: string, cursor: number): NameCompletion | null {
  const match = /[A-Za-z][A-Za-z0-9]*$/.exec(source.slice(0, cursor))
  if (!match) return null
  const prefix = match[0]
  const start = cursor - prefix.length
  if (prefix.length < 2 || /[A-Za-z_'.αβγδεζηθικλμνξπρστυφχψω]/.test(source[start - 1] ?? '')) return null
  if (/[A-Za-z0-9(]/.test(source[cursor] ?? '')) return null
  const help = signatureAt(source, cursor)
  if (help?.kind === 'options') return null
  if (help?.kind === 'value' && (help.option?.kind === 'color' || help.option?.kind === 'boolean')) return null
  const items = completeFunctionName(prefix)
  return items.length > 0 ? { start, end: cursor, prefix, items } : null
}

function describeDefault(option: OptionSpec): string | null {
  if (option.default === null) return null
  if (option.default === 'auto') return 'automatic'
  return `default ${String(option.default)}`
}

function optionItems(help: Extract<SignatureHelp, { kind: 'options' }>): AssistItem[] {
  const typed = optionKey(help.prefix)
  if (help.prefix && !/^[A-Za-zα-ω][A-Za-z0-9α-ω _-]*$/.test(help.prefix)) return []
  return help.options
    .filter((option) => !help.used.includes(optionKey(option.name)) && optionKey(option.name).startsWith(typed))
    .map((option) => {
      const flag = option.kind === 'boolean' && option.default === false
      const text = flag ? option.name : `${option.name}: `
      return { id: option.name, label: option.name, detail: option.description, note: describeDefault(option), swatch: null, edit: { start: help.start, end: help.end, text, caret: text.length } }
    })
}

function valueItems(help: Extract<SignatureHelp, { kind: 'value' }>): AssistItem[] {
  const option = help.option
  if (!option) return []
  const typed = help.prefix.trim().toLowerCase()
  const choices = option.kind === 'color' ? Object.keys(PLOT_PALETTE) : option.kind === 'boolean' ? ['true', 'false'] : []
  return choices
    .filter((choice) => choice.startsWith(typed) && choice !== typed)
    .map((choice) => ({ id: choice, label: choice, detail: '', note: null, swatch: option.kind === 'color' ? (PLOT_PALETTE[choice] ?? null) : null, edit: { start: help.start, end: help.end, text: choice, caret: choice.length } }))
}

/** Everything the hint card shows for this caret, with the edit each pickable line makes. */
export function assistAt(source: string, cursor: number): Assist | null {
  const completion = completionAt(source, cursor)
  if (completion) {
    return {
      kind: 'complete',
      items: completion.items.map((spec) => {
        const text = `${spec.name}()`
        return { id: spec.name, label: signatureText(spec), detail: spec.summary, note: null, swatch: null, edit: { start: completion.start, end: completion.end, text, caret: spec.name.length + 1 } }
      }),
    }
  }
  const help = signatureAt(source, cursor)
  if (!help) return null
  if (help.kind === 'options') return { kind: 'options', owner: help.owner, prefix: help.prefix, options: help.options, items: optionItems(help) }
  if (help.kind === 'value') return { kind: 'value', owner: help.owner, options: help.options, key: help.key, option: help.option, items: valueItems(help) }
  return help
}

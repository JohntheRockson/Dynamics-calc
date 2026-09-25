// The tools an agent is allowed to use. Both the MCP server (Cursor) and the
// in-app Grok assistant call these. They do not invent a second calculator:
// they run the same console the page uses.

import { appendMath, emptyDocument } from '../src/kinematics/workspace/document.ts'
import { evaluateDocument } from '../src/kinematics/workspace/evaluate.ts'
import { FUNCTIONS, SYNTAX_RULES, type FunctionSpec } from '../src/kinematics/workspace/math/functions.ts'
import type { AngleMode } from '../src/kinematics/workspace/math/expr.ts'

export interface MathLineResult {
  input: string
  output: string
  error: boolean
  plotted: boolean
}

export interface MathRun {
  lines: MathLineResult[]
  curves: number
  surfaces: number
}

export interface FunctionCard {
  name: string
  summary: string
  params: string[]
  options: string[]
  examples: string[]
}

function card(spec: FunctionSpec): FunctionCard {
  return {
    name: spec.name,
    summary: spec.summary,
    params: spec.params.map((param) => param.phrase),
    options: spec.options.map((option) => option.name),
    examples: spec.examples,
  }
}

/** The console's vocabulary, small enough to hand to a model. */
export function listMathFunctions(): { syntax: string[]; functions: FunctionCard[] } {
  return { syntax: SYNTAX_RULES, functions: FUNCTIONS.map(card) }
}

/**
 * Run one or more console lines. A newline starts a new line, and a later
 * line can see a name defined above it. Paths are not returned: a plot is
 * only reported as a count, so a tool result stays small.
 */
export function evaluateMath(script: string, angles: AngleMode = 'rad'): MathRun {
  const written = script.split('\n').map((line) => line.trim()).filter((line) => line.length > 0)
  let doc = emptyDocument()
  for (const line of written) doc = appendMath(doc, line)
  const view = evaluateDocument(doc, 0, angles)
  const lines = view.blocks.flatMap((block) => block.rows).flatMap((row): MathLineResult[] => {
    if (row.input === undefined && row.source !== 'error') return []
    const silent = !row.input || (row.source !== 'error' && row.text.trim() === row.input.trim())
    return [{ input: row.input ?? '', output: silent ? '' : row.text, error: row.source === 'error', plotted: Boolean(row.plotKind) }]
  })
  return { lines, curves: view.bodies.length, surfaces: view.surfaces.length }
}

export function runMathTool(name: string, args: Record<string, unknown>, pageAngles: AngleMode): unknown {
  if (name === 'list_math_functions') return listMathFunctions()
  if (name === 'evaluate_math') {
    const script = typeof args.script === 'string' ? args.script : ''
    if (!script.trim()) return { error: 'evaluate_math needs a script.' }
    const requested = args.angles === 'deg' || args.angles === 'rad' ? args.angles : pageAngles
    return evaluateMath(script, requested)
  }
  return { error: `Unknown tool ${name}.` }
}

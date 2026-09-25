// One turn with Grok. The model may ask to run a tool. We run it here, send
// the result back, and repeat until Grok answers in words. The API key stays
// in this process. The browser never sees it.

import type { AngleMode } from '../src/kinematics/workspace/math/expr.ts'
import { runMathTool } from './mathTools.ts'

export const GROK_MODEL = 'grok-4.7'
const RESPONSES_URL = 'https://api.x.ai/v1/responses'
const MAX_ROUNDS = 6

const SYSTEM = [
  'You are the assistant inside a dynamics class calculator.',
  'The student already has a console. You can run that same console. You cannot see its graphs, only the text it returns and whether a line drew a curve.',
  'Never invent a numeric or symbolic result. If the answer depends on a calculation, call evaluate_math and use what it returns.',
  'Write math in the console syntax: Name(required inputs){Setting: value}. Settings go in braces after the ), not as extra commas.',
  'Integrate only computes. Its limits are Bounds, as in Integrate(x, x){Bounds: 0..1}. Domain clips a graph. Shade fills under a drawn curve, as in Plot(x^2, x){Shade: 0..2}.',
  'A trailing semicolon runs a line and shows nothing. Leave it off when the student should see the result.',
  'If an expression uses only one variable, that variable may be omitted.',
  'Pass angles as rad or deg to match the page.',
  'A script is one or more lines separated by newlines. Later lines see earlier definitions.',
  'Call list_math_functions when you are not sure of a name or a setting.',
  'After the tool result, explain it in plain language. Quote the line you ran. Do not say you changed the page; the student adds lines themselves.',
].join(' ')

export interface AgentStep {
  tool: string
  arguments: Record<string, unknown>
  result: unknown
}

export interface AgentReply {
  model: string
  reply: string
  steps: AgentStep[]
  /** Console lines the model actually ran, in order, for the page to offer. */
  lines: string[]
}

interface ResponseContent {
  type?: string
  text?: string
}

interface ResponseItem {
  type?: string
  name?: string
  arguments?: string
  call_id?: string
  content?: ResponseContent[] | string
}

interface ModelResponse {
  id?: string
  output?: ResponseItem[]
  error?: { message?: string }
}

export interface AskInput {
  question: string
  angles?: AngleMode
  page?: string[]
  apiKey: string
  fetchImpl?: typeof fetch
}

function messageText(items: ResponseItem[]): string {
  const parts: string[] = []
  for (const item of items) {
    if (item.type !== 'message') continue
    if (typeof item.content === 'string') parts.push(item.content)
    else for (const block of item.content ?? []) if (block.text) parts.push(block.text)
  }
  return parts.join('\n').trim()
}

function scriptsFrom(steps: AgentStep[]): string[] {
  const lines: string[] = []
  for (const step of steps) {
    if (step.tool !== 'evaluate_math') continue
    const script = step.arguments.script
    if (typeof script !== 'string') continue
    for (const line of script.split('\n')) {
      const trimmed = line.trim()
      if (trimmed) lines.push(trimmed)
    }
  }
  return lines
}

const TOOLS = [
  {
    type: 'function',
    name: 'list_math_functions',
    description: 'List the calculator functions, the inputs they require, and their optional settings.',
    parameters: { type: 'object', properties: {}, required: [] },
  },
  {
    type: 'function',
    name: 'evaluate_math',
    description: 'Run one or more calculator lines. Later lines see names defined above them. Returns each line\'s text result and whether it drew a curve. Does not change the student\'s page.',
    parameters: {
      type: 'object',
      properties: {
        script: { type: 'string', description: 'Console lines separated by newlines, such as "Integrate(x^2, x){Bounds: 0..1}".' },
        angles: { type: 'string', enum: ['rad', 'deg'], description: 'Angle unit. Use the unit the page is in.' },
      },
      required: ['script'],
    },
  },
]

/** Ask Grok 4.7, running calculator tools locally until it answers in words. */
export async function askGrok(input: AskInput): Promise<AgentReply> {
  const key = input.apiKey.trim()
  if (!key) throw new Error('Set XAI_API_KEY in the environment before starting the agent. The page does not take the key.')
  const question = input.question.trim()
  if (!question) throw new Error('Ask a question first.')
  const angles: AngleMode = input.angles === 'deg' ? 'deg' : 'rad'
  const page = (input.page ?? []).map((line) => line.trim()).filter(Boolean)
  const pageNote = page.length ? `Lines already on the page:\n${page.join('\n')}` : 'The page has no math lines yet.'
  const fetchImpl = input.fetchImpl ?? fetch
  const steps: AgentStep[] = []
  let previousId: string | undefined
  let payload: unknown[] = [
    { role: 'system', content: SYSTEM },
    { role: 'user', content: `${pageNote}\nThe page is in ${angles === 'deg' ? 'degrees' : 'radians'}.\n\n${question}` },
  ]

  for (let round = 0; round < MAX_ROUNDS; round += 1) {
    const body: Record<string, unknown> = { model: GROK_MODEL, input: payload, tools: TOOLS, tool_choice: 'auto' }
    if (previousId) body.previous_response_id = previousId
    const response = await fetchImpl(RESPONSES_URL, {
      method: 'POST',
      headers: { Authorization: `Bearer ${key}`, 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
    const raw = await response.text()
    if (!response.ok) throw new Error(raw.slice(0, 500) || `Grok returned ${response.status}.`)
    const data = JSON.parse(raw) as ModelResponse
    if (data.error?.message) throw new Error(data.error.message)
    if (!data.id) throw new Error('Grok returned no response id.')
    previousId = data.id
    const output = data.output ?? []
    const calls = output.filter((item) => item.type === 'function_call')
    if (calls.length === 0) {
      const reply = messageText(output)
      if (!reply) throw new Error('Grok returned no answer.')
      return { model: GROK_MODEL, reply, steps, lines: scriptsFrom(steps) }
    }
    const returned: unknown[] = []
    for (const call of calls) {
      const name = call.name ?? ''
      let args: Record<string, unknown> = {}
      try {
        const parsed = JSON.parse(call.arguments || '{}') as unknown
        if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) args = parsed as Record<string, unknown>
      } catch {
        args = {}
      }
      const result = runMathTool(name, args, angles)
      steps.push({ tool: name, arguments: args, result })
      returned.push({ type: 'function_call_output', call_id: call.call_id ?? '', output: JSON.stringify(result) })
    }
    payload = returned
  }
  return { model: GROK_MODEL, reply: 'I stopped after several calculations. Ask a narrower question if you still need a result.', steps, lines: scriptsFrom(steps) }
}

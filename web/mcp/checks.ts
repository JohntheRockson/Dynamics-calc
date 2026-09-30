// Checks for the tool layer and the Grok loop. No network: the model is a fake
// response that asks for one calculation, then answers from that result.

import type { AngleMode } from '../src/kinematics/workspace/math/expr.ts'
import { askGrok } from './agent.ts'
import { evaluateMath, listMathFunctions } from './mathTools.ts'

export function runAgentChecks(): string[] {
  const errors: string[] = []
  const expect = (cond: boolean, message: string) => {
    if (!cond) errors.push(message)
  }

  const catalog = listMathFunctions()
  const integrate = catalog.functions.find((fn) => fn.name === 'Integrate')
  expect(Boolean(integrate?.options.includes('Bounds')), `Integrate is listed with Bounds: ${integrate?.options.join(', ')}`)
  expect(catalog.syntax.some((rule) => rule.includes('Parentheses')), 'the syntax rules are part of the catalog')

  const area = evaluateMath('Integrate(x, x){Bounds: 0..1}')
  expect(area.lines.length === 1 && area.lines[0]?.output.includes('1/2') && area.curves === 0, `a definite integral is a value: ${JSON.stringify(area)}`)
  const quiet = evaluateMath('a = 4;\na + 1')
  expect(quiet.lines[0]?.output === '' && quiet.lines[1]?.output.includes('5'), `a semicolon hides the first line and the second still sees it: ${JSON.stringify(quiet)}`)
  const guessed = evaluateMath('Plot(sin(t))')
  expect(guessed.curves === 1 && guessed.lines[0]?.output.includes(', t'), `Plot guesses the only variable: ${JSON.stringify(guessed)}`)

  const rounds: string[] = [
    JSON.stringify({
      id: 'resp-1',
      output: [{ type: 'function_call', name: 'evaluate_math', call_id: 'call-1', arguments: JSON.stringify({ script: 'Integrate(x, x){Bounds: 0..1}' }) }],
    }),
    JSON.stringify({
      id: 'resp-2',
      output: [{ type: 'message', content: [{ type: 'output_text', text: 'The integral from 0 to 1 is 1/2.' }] }],
    }),
  ]
  const seen: unknown[] = []
  const fetchImpl: typeof fetch = async (_url, init) => {
    seen.push(JSON.parse(String(init?.body ?? '{}')))
    const body = rounds.shift() ?? '{}'
    return new Response(body, { status: 200 })
  }

  return askGrok({ question: 'What is the area under y = x from 0 to 1?', angles: 'rad' as AngleMode, apiKey: 'test-key', fetchImpl }).then(
    (reply) => {
      expect(reply.reply.includes('1/2'), `Grok's answer uses the tool result: ${reply.reply}`)
      expect(reply.lines[0] === 'Integrate(x, x){Bounds: 0..1}', `the ran line is offered back: ${reply.lines.join(' | ')}`)
      const second = seen[1] as { previous_response_id?: string; input?: { type?: string; output?: string }[] }
      expect(second.previous_response_id === 'resp-1' && second.input?.[0]?.type === 'function_call_output', `the tool result is sent back to Grok: ${JSON.stringify(second)}`)
      expect(String(second.input?.[0]?.output ?? '').includes('1/2'), 'the calculator result is what Grok sees')
      return errors
    },
    (error: unknown) => {
      errors.push(error instanceof Error ? error.message : 'askGrok failed')
      return errors
    },
  )
}

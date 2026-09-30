import { useEffect, useState } from 'react'
import type { AngleMode } from './math/expr'

interface AgentStep {
  tool: string
  arguments: { script?: string }
  result: { lines?: { input: string; output: string; error: boolean }[]; error?: string }
}

interface AgentReply {
  model?: string
  reply?: string
  steps?: AgentStep[]
  lines?: string[]
  error?: string
}

interface Health {
  ok: boolean
  hasKey: boolean
}

/**
 * The in-app assistant. The browser only sends the question. A local process
 * calls Grok 4.7 and runs the calculator. The API key never comes here.
 */
export function AskGrok({ angles, page, onInsert }: { angles: AngleMode; page: string[]; onInsert: (lines: string[]) => void }) {
  const [open, setOpen] = useState(false)
  const [question, setQuestion] = useState('')
  const [health, setHealth] = useState<Health | null>(null)
  const [busy, setBusy] = useState(false)
  const [reply, setReply] = useState<AgentReply | null>(null)

  useEffect(() => {
    if (!open) return
    let cancelled = false
    void fetch('/api/agent/health')
      .then((response) => (response.ok ? response.json() : Promise.reject(new Error('down'))))
      .then((body: Health) => {
        if (!cancelled) setHealth(body)
      })
      .catch(() => {
        if (!cancelled) setHealth(null)
      })
    return () => {
      cancelled = true
    }
  }, [open])

  const ask = () => {
    const text = question.trim()
    if (!text || busy) return
    setBusy(true)
    setReply(null)
    void fetch('/api/agent', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: text, angles, page }),
    })
      .then(async (response) => {
        const body = (await response.json()) as AgentReply
        setReply(response.ok ? body : { error: body.error ?? 'The agent could not answer.' })
      })
      .catch(() => setReply({ error: 'The agent is not running. From web/, run npm run agent with XAI_API_KEY set.' }))
      .finally(() => setBusy(false))
  }

  const lines = reply?.lines ?? []

  return (
    <section className="ask-grok">
      <button type="button" className="ask-grok-toggle" aria-expanded={open} onClick={() => setOpen((current) => !current)}>
        {open ? 'Hide Grok' : 'Ask Grok'}
      </button>
      {open && (
        <div className="ask-grok-body">
          <p className="hint">Grok 4.7 can run this console. It calls the same math the page uses, and it does not guess a result.</p>
          {health === null && <p className="statement-note">The agent process is not running. From web/, run npm run agent.</p>}
          {health && !health.hasKey && <p className="statement-note">The agent is up, but XAI_API_KEY is not set, so Grok cannot be called.</p>}
          <div className="ask-grok-row">
            <input
              className="num-input"
              value={question}
              placeholder="Ask about the page, or for a calculation"
              aria-label="Question for Grok"
              onChange={(event) => setQuestion(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter') ask()
              }}
            />
            <button type="button" className="btn btn-primary" disabled={busy || !question.trim()} onClick={ask}>
              {busy ? 'Working' : 'Ask'}
            </button>
          </div>
          {reply?.error && <p className="statement-note">{reply.error}</p>}
          {reply?.reply && <p className="ask-grok-reply">{reply.reply}</p>}
          {reply?.steps && reply.steps.length > 0 && (
            <ul className="ask-grok-steps">
              {reply.steps.map((step, index) => (
                <li key={`${step.tool}-${index}`}>
                  {step.tool === 'evaluate_math' ? (step.arguments.script ?? 'evaluate_math') : 'Looked up the function list'}
                </li>
              ))}
            </ul>
          )}
          {lines.length > 0 && (
            <button type="button" className="btn btn-ghost" onClick={() => onInsert(lines)}>
              Add these lines to the page
            </button>
          )}
        </div>
      )}
    </section>
  )
}

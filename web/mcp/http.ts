// HTTP front for the in-app assistant. The page posts a question here.
// Grok 4.7 answers, and if it needs a calculation this process runs it.
// Start with XAI_API_KEY set: `npm run agent` from web/.

import { createServer, type IncomingMessage, type ServerResponse } from 'node:http'
import type { AngleMode } from '../src/kinematics/workspace/math/expr.ts'
import { askGrok } from './agent.ts'

const port = Number(process.env.MATH_AGENT_PORT ?? 8788)

function send(res: ServerResponse, status: number, body: unknown): void {
  const payload = JSON.stringify(body)
  res.writeHead(status, { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(payload) })
  res.end(payload)
}

function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = []
    req.on('data', (chunk: Buffer) => {
      chunks.push(chunk)
      if (chunks.reduce((sum, part) => sum + part.length, 0) > 100_000) {
        reject(new Error('That question is too long.'))
        req.destroy()
      }
    })
    req.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')))
    req.on('error', reject)
  })
}

const server = createServer((req, res) => {
  const url = req.url ?? ''
  if (req.method === 'GET' && (url === '/api/agent/health' || url === '/health')) {
    send(res, 200, { ok: true, model: 'grok-4.7', hasKey: Boolean(process.env.XAI_API_KEY?.trim()) })
    return
  }
  if (req.method !== 'POST' || (url !== '/api/agent' && url !== '/')) {
    send(res, 404, { error: 'Not found.' })
    return
  }
  void (async () => {
    try {
      const raw = await readBody(req)
      const body = raw ? (JSON.parse(raw) as { question?: unknown; angles?: unknown; page?: unknown }) : {}
      const question = typeof body.question === 'string' ? body.question : ''
      const angles: AngleMode = body.angles === 'deg' ? 'deg' : 'rad'
      const page = Array.isArray(body.page) ? body.page.filter((line): line is string => typeof line === 'string') : []
      const reply = await askGrok({ question, angles, page, apiKey: process.env.XAI_API_KEY ?? '' })
      send(res, 200, reply)
    } catch (error) {
      const message = error instanceof Error ? error.message : 'The agent could not answer.'
      send(res, 400, { error: message })
    }
  })()
})

server.listen(port, '127.0.0.1', () => {
  console.error(`Math agent listening on http://127.0.0.1:${port} (model grok-4.7)`)
  if (!process.env.XAI_API_KEY?.trim()) console.error('XAI_API_KEY is not set. Questions will fail until it is.')
})

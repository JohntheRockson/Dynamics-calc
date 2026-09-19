import type { ScenarioInfo, SimLog, SimRequest } from './types'

export class ApiError extends Error {}

async function parseJsonOrThrow<T>(res: Response): Promise<T> {
  const text = await res.text()
  let data: unknown
  try {
    data = text ? JSON.parse(text) : {}
  } catch {
    throw new ApiError(`Server returned invalid JSON (status ${res.status}).`)
  }
  if (!res.ok) {
    const message = (data as { error?: string } | null)?.error ?? `Request failed with status ${res.status}`
    throw new ApiError(message)
  }
  return data as T
}

export async function fetchScenarios(): Promise<ScenarioInfo[]> {
  const res = await fetch('/api/scenarios')
  const data = await parseJsonOrThrow<{ scenarios: ScenarioInfo[] }>(res)
  return data.scenarios
}

export async function runSimulation(req: SimRequest, signal?: AbortSignal): Promise<SimLog> {
  const res = await fetch('/api/simulate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
    signal,
  })
  return parseJsonOrThrow<SimLog>(res)
}

// The adaptive sampler runs in Rust (WebAssembly) so a large plot-point count
// and a deep recursion do not block the page. Until that module has loaded,
// the console samples in TypeScript with a smaller cap.

import { applyEnv, freeSymbols, type Expr, type MathEnv, type PlotOptions } from './expr'

export interface PlotNode {
  t: number
  y: number | null
  cut: boolean
}

interface PlotExports {
  memory: WebAssembly.Memory
  plot_input_ptr: (len: number) => number
  plot_sample: (ptr: number, len: number) => number
  plot_output_ptr: () => number
}

let wasm: PlotExports | null = null
let ready = false
let loading = false
const waiters = new Set<() => void>()

export function plotterReady(): boolean {
  return ready
}

export function onPlotterReady(listener: () => void): () => void {
  if (ready) listener()
  else waiters.add(listener)
  return () => waiters.delete(listener)
}

function bind(instance: WebAssembly.Instance): void {
  wasm = instance.exports as unknown as PlotExports
  ready = true
  waiters.forEach((listener) => listener())
  waiters.clear()
}

export function loadPlotterBytes(bytes: BufferSource): Promise<void> {
  return WebAssembly.instantiate(bytes).then(({ instance }) => bind(instance))
}

export function loadPlotter(): void {
  if (ready || loading || typeof fetch === 'undefined' || typeof WebAssembly === 'undefined') return
  loading = true
  fetch('/plotter.wasm')
    .then((response) => {
      if (!response.ok) throw new Error('plotter missing')
      return response.arrayBuffer()
    })
    .then((bytes) => loadPlotterBytes(bytes))
    .catch(() => {
      wasm = null
    })
}

function encodeExpr(expr: Expr): unknown | null {
  switch (expr.type) {
    case 'rat':
      return { t: 'rat', n: Number(expr.n), d: Number(expr.d) }
    case 'dec':
      return { t: 'dec', v: expr.value }
    case 'sym':
      return { t: 'sym', name: expr.name }
    case 'add':
    case 'mul': {
      const args = expr.args.map((arg) => encodeExpr(arg))
      if (args.some((arg) => arg === null)) return null
      return { t: expr.type, args }
    }
    case 'div': {
      const num = encodeExpr(expr.num)
      const den = encodeExpr(expr.den)
      if (!num || !den) return null
      return { t: 'div', num, den }
    }
    case 'pow': {
      const base = encodeExpr(expr.base)
      const exp = encodeExpr(expr.exp)
      if (!base || !exp) return null
      return { t: 'pow', base, exp }
    }
    case 'call': {
      const args = expr.args.map((arg) => encodeExpr(arg))
      if (args.some((arg) => arg === null)) return null
      return { t: 'call', name: expr.name, args }
    }
    case 'group':
      return encodeExpr(expr.body)
    case 'caret':
      return expr.body ? encodeExpr(expr.body) : null
    default:
      return null
  }
}

function closedExpr(body: Expr, param: string, env: MathEnv): unknown | null {
  const frozen: MathEnv = new Map(env)
  frozen.delete(param)
  let expr: Expr
  try {
    expr = applyEnv(body, frozen, 0)
  } catch {
    return null
  }
  const free = freeSymbols(expr).filter((name) => name !== param && name !== 'pi' && name !== 'e')
  if (free.length > 0) return null
  return encodeExpr(expr)
}

/** Sample one explicit curve in Rust. Null means the page should use the TypeScript sampler. */
export function sampleCurveNative(body: Expr, param: string, env: MathEnv, min: number, max: number, ySpan: number, degrees: boolean, style: PlotOptions): PlotNode[] | null {
  if (!wasm) return null
  const expr = closedExpr(body, param, env)
  if (!expr) return null
  const payload = JSON.stringify({ min, max, points: style.points, recursion: style.recursion, exclusions: style.exclusions, yspan: ySpan, degrees, expr })
  const bytes = new TextEncoder().encode(payload)
  const ptr = wasm.plot_input_ptr(bytes.length)
  if (!ptr) return null
  new Uint8Array(wasm.memory.buffer, ptr, bytes.length).set(bytes)
  const size = wasm.plot_sample(ptr, bytes.length)
  if (size < 0) return null
  const outputPtr = wasm.plot_output_ptr()
  const text = new TextDecoder().decode(new Uint8Array(wasm.memory.buffer, outputPtr, size))
  let parsed: unknown
  try {
    parsed = JSON.parse(text)
  } catch {
    return null
  }
  if (!Array.isArray(parsed)) return null
  const nodes: PlotNode[] = []
  for (const item of parsed) {
    if (!Array.isArray(item) || item.length < 3 || typeof item[0] !== 'number') return null
    const y = typeof item[1] === 'number' && Number.isFinite(item[1]) ? item[1] : null
    nodes.push({ t: item[0], y, cut: Boolean(item[2]) })
  }
  return nodes
}

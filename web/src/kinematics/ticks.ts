/** Nice 1-2-5 steps so grid lines sit on world coordinates and travel with a pan. */

export function niceStep(span: number, target: number): number {
  if (!(span > 0) || !(target > 0)) return 1
  const rough = span / target
  const pow = 10 ** Math.floor(Math.log10(rough))
  const n = rough / pow
  const base = n < 1.5 ? 1 : n < 3.5 ? 2 : n < 7.5 ? 5 : 10
  return base * pow
}

export function tickMarks(min: number, max: number, step: number): number[] {
  if (!(step > 0) || !(max > min)) return []
  const start = Math.ceil(min / step - 1e-8) * step
  const marks: number[] = []
  for (let i = 0; i < 48; i += 1) {
    const value = Math.round((start + i * step) / step) * step
    if (value > max + step * 1e-6) break
    if (value < min - step * 1e-6) continue
    marks.push(Math.abs(value) <= step * 1e-6 ? 0 : value)
  }
  return marks
}

export function formatTick(value: number, step: number): string {
  if (!Number.isFinite(value) || Math.abs(value) <= Math.abs(step) * 1e-6) return '0'
  const abs = Math.abs(value)
  if (abs >= 1e5 || abs < 1e-4) return value.toExponential(1).replace('e+', 'e')
  const exp = Math.floor(Math.log10(step))
  const mant = step / 10 ** exp
  const tenth = Math.abs(mant * 10 - Math.round(mant * 10))
  const mantDec = Math.abs(mant - Math.round(mant)) < 1e-6 ? 0 : tenth < 1e-6 ? 1 : 2
  const decimals = Math.min(6, Math.max(0, mantDec - exp))
  const snapped = Math.round(value / step) * step
  const text = snapped.toFixed(decimals)
  if (!text.includes('.')) return text
  return text.replace(/0+$/, '').replace(/\.$/, '')
}

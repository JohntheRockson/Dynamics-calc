/** Compact numeric display shared by the control panel, HUD, and charts. */
export function formatNumber(v: number, precision = 4): string {
  if (!Number.isFinite(v)) return '-'
  if (v !== 0 && (Math.abs(v) < 1e-3 || Math.abs(v) >= 1e5)) return v.toExponential(2)
  let s = v.toFixed(precision)
  if (s.includes('.')) {
    s = s.replace(/0+$/, '').replace(/\.$/, '')
  }
  return s === '' || s === '-' ? '0' : s
}

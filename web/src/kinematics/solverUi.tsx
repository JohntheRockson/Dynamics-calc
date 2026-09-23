import { useId } from 'react'
import { formatNumber } from '../format'
import { Eq } from './Eq'

export function parseNum(raw: string): number | null {
  const t = raw.trim().replace(',', '.')
  if (t === '' || t === '-' || t === '.' || t === '-.') return null
  const n = Number(t)
  return Number.isFinite(n) ? n : null
}

export function KnownRow({
  symbol,
  name,
  unit,
  known,
  value,
  onKnown,
  onValue,
  locked,
}: {
  symbol: string
  name: string
  unit: string
  known: boolean
  value: number | null
  onKnown: (k: boolean) => void
  onValue: (v: number | null) => void
  locked?: boolean
}) {
  const id = useId()
  return (
    <div className={`known-row ${known ? '' : 'is-unknown'}`}>
      <input
        id={id}
        type="checkbox"
        checked={known}
        disabled={locked}
        aria-label={`${name} known`}
        onChange={(e) => onKnown(e.target.checked)}
      />
      <label className="known-sym" htmlFor={`${id}-val`} title={name}>
        <Eq tex={symbol} />
      </label>
      <input
        id={`${id}-val`}
        className="num-input"
        inputMode="decimal"
        placeholder={known ? '' : 'unknown'}
        value={known && value !== null ? String(value) : ''}
        onChange={(e) => {
          const n = parseNum(e.target.value)
          onValue(n)
          if (e.target.value.trim() !== '') onKnown(true)
          else onKnown(false)
        }}
      />
      <span className="known-unit">{unit}</span>
    </div>
  )
}

export function PlainRow({
  symbol,
  name,
  unit,
  value,
  onChange,
}: {
  symbol: string
  name: string
  unit: string
  value: number | null
  onChange: (v: number | null) => void
}) {
  const id = useId()
  return (
    <div className="known-row">
      <span className="known-lock" title="always known">
        ✓
      </span>
      <label className="known-sym" htmlFor={id} title={name}>
        <Eq tex={symbol} />
      </label>
      <input
        id={id}
        className="num-input"
        inputMode="decimal"
        value={value === null ? '' : String(value)}
        aria-label={name}
        onChange={(e) => onChange(parseNum(e.target.value))}
      />
      <span className="known-unit">{unit}</span>
    </div>
  )
}

export function ResultCard({
  label,
  value,
  unit,
  computed,
}: {
  label: string
  value: number
  unit: string
  computed: boolean
}) {
  return (
    <div className={`stat-card ${computed ? 'computed' : ''}`}>
      <div className="label">
        {label} {computed ? '· solved' : '· given'}
      </div>
      <div className="val">
        {formatNumber(value, 5)}
        <span className="unit">{unit}</span>
      </div>
    </div>
  )
}

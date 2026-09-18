import { useId } from 'react'
import { formatNumber } from '../format'

function clamp(v: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, v))
}

interface NumberFieldProps {
  label: string
  value: number
  onChange: (v: number) => void
  min: number
  max: number
  step?: number
  unit?: string
  precision?: number
  logScale?: boolean
  disabled?: boolean
  hint?: string
}

export function NumberField({ label, value, onChange, min, max, step, unit, precision = 4, logScale = false, disabled, hint }: NumberFieldProps) {
  const id = useId()
  const toSlider = (v: number) => (logScale ? Math.log10(Math.max(v, min)) : v)
  const fromSlider = (s: number) => (logScale ? 10 ** s : s)
  const sliderStep = step ?? (logScale ? (Math.log10(max) - Math.log10(min)) / 200 : (max - min) / 200)

  return (
    <div className="field-row">
      <label className="field-label" htmlFor={id}>
        <span>{label}</span>
        <span className="value">
          {formatNumber(value, precision)}
          {unit ? ` ${unit}` : ''}
        </span>
      </label>
      <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
        <input
          id={id}
          name={id}
          type="range"
          aria-label={label}
          min={toSlider(min)}
          max={toSlider(max)}
          step={sliderStep}
          value={toSlider(clamp(value, min, max))}
          disabled={disabled}
          onChange={(e) => onChange(clamp(fromSlider(parseFloat(e.target.value)), min, max))}
        />
      </div>
      {hint && <p className="hint">{hint}</p>}
    </div>
  )
}

interface Vec3FieldProps {
  label: string
  value: [number, number, number]
  onChange: (v: [number, number, number]) => void
  min: number
  max: number
  step?: number
  unit?: string
  disabled?: boolean
}

const AXIS_CLASSES = ['axis-x', 'axis-y', 'axis-z']
const AXIS_LABELS = ['x', 'y', 'z']

export function Vec3Field({ label, value, onChange, min, max, step, unit, disabled }: Vec3FieldProps) {
  const id = useId()
  return (
    <div className="field-row">
      <span className="field-label">
        <span>{label}</span>
        <span className="value">{unit ?? ''}</span>
      </span>
      <div className="vec3-input">
        {value.map((v, i) => (
          <input
            key={AXIS_LABELS[i]}
            id={`${id}-${AXIS_LABELS[i]}`}
            name={`${id}-${AXIS_LABELS[i]}`}
            className={`num-input ${AXIS_CLASSES[i]}`}
            type="number"
            step={step ?? 'any'}
            min={min}
            max={max}
            disabled={disabled}
            value={Number.isFinite(v) ? v : 0}
            onChange={(e) => {
              const next = [...value] as [number, number, number]
              next[i] = parseFloat(e.target.value)
              onChange(next)
            }}
            aria-label={`${label} ${AXIS_LABELS[i]}`}
          />
        ))}
      </div>
    </div>
  )
}

interface ToggleFieldProps {
  label: string
  checked: boolean
  onChange: (v: boolean) => void
  disabled?: boolean
}

export function ToggleField({ label, checked, onChange, disabled }: ToggleFieldProps) {
  const id = useId()
  return (
    <label className="toggle-row" style={{ opacity: disabled ? 0.5 : 1 }} htmlFor={id}>
      <span className="toggle">
        <input id={id} name={id} type="checkbox" checked={checked} disabled={disabled} onChange={(e) => onChange(e.target.checked)} />
        <span className="track">
          <span className="thumb" />
        </span>
      </span>
      <span>{label}</span>
    </label>
  )
}

interface SelectFieldProps<T extends string> {
  label: string
  value: T
  options: { value: T; label: string }[]
  onChange: (v: T) => void
  disabled?: boolean
}

export function SelectField<T extends string>({ label, value, options, onChange, disabled }: SelectFieldProps<T>) {
  const id = useId()
  return (
    <div className="field-row">
      <label className="field-label" htmlFor={id}>
        <span>{label}</span>
      </label>
      <select id={id} name={id} className="num-input" value={value} disabled={disabled} onChange={(e) => onChange(e.target.value as T)}>
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </div>
  )
}

interface SegmentedProps<T extends string> {
  value: T
  options: { value: T; label: string }[]
  onChange: (v: T) => void
  disabled?: boolean
}

export function Segmented<T extends string>({ value, options, onChange, disabled }: SegmentedProps<T>) {
  return (
    <div className="segmented">
      {options.map((o) => (
        <button key={o.value} type="button" className={o.value === value ? 'active' : ''} disabled={disabled} onClick={() => onChange(o.value)}>
          {o.label}
        </button>
      ))}
    </div>
  )
}


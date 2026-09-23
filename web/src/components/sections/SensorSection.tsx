import type { SimRequest } from '../../types'
import { NumberField, ToggleField } from '../fields'
import { Section } from '../Section'

interface Props {
  req: SimRequest
  patch: (p: Partial<SimRequest>) => void
}

export function SensorSection({ req, patch }: Props) {
  const disabled = req.estimator === 'truth'

  return (
    <Section title="Sensors" icon="📡" defaultOpen={false} rightSlot={disabled ? <span className="hint">n/a for truth</span> : undefined}>
      <div style={{ opacity: disabled ? 0.5 : 1, display: 'flex', flexDirection: 'column', gap: 10, pointerEvents: disabled ? 'none' : 'auto' }}>
        <NumberField
          label="Gyro ARW density σv"
          value={req.gyro_sigma_v}
          onChange={(v) => patch({ gyro_sigma_v: v })}
          min={1e-5}
          max={1e-2}
          unit="rad/s/√Hz"
          precision={6}
          logScale
        />
        <NumberField
          label="Gyro RRW density σu"
          value={req.gyro_sigma_u}
          onChange={(v) => patch({ gyro_sigma_u: v })}
          min={1e-8}
          max={1e-4}
          unit="rad/s²/√Hz"
          precision={8}
          logScale
        />
        <ToggleField label="Magnetometer" checked={req.use_mag} onChange={(v) => patch({ use_mag: v })} />
        {req.use_mag && (
          <NumberField label="Magnetometer σ" value={req.mag_sigma} onChange={(v) => patch({ mag_sigma: v })} min={1e-4} max={5e-2} unit="unit vector" precision={4} logScale />
        )}
        <ToggleField label="Sun sensor" checked={req.use_sun} onChange={(v) => patch({ use_sun: v })} />
        {req.use_sun && (
          <>
            <NumberField label="Sun sensor σ" value={req.sun_sigma} onChange={(v) => patch({ sun_sigma: v })} min={1e-4} max={5e-2} unit="unit vector" precision={4} logScale />
            <ToggleField label="Force eclipse (sun sensor occluded)" checked={req.sun_eclipse} onChange={(v) => patch({ sun_eclipse: v })} />
          </>
        )}
        <p className="hint">
          MEKF/Mahony run on ChaCha8-seeded noise (the Rust RNG), not NumPy's generator, so exact sample-by-sample values differ
          from the Python CLI for a given seed even though every model (ARW/RRW scaling, filter equations, gating) is ported
          faithfully &mdash; see the README for why.
        </p>
      </div>
    </Section>
  )
}

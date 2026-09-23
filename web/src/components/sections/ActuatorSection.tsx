import type { SimRequest } from '../../types'
import { NumberField, ToggleField } from '../fields'
import { Section } from '../Section'

interface Props {
  req: SimRequest
  patch: (p: Partial<SimRequest>) => void
}

function scalarOf(limit: SimRequest['actuator_tau_max'], fallback: number): number {
  if (limit === null) return fallback
  return typeof limit === 'number' ? limit : limit[0]
}

export function ActuatorSection({ req, patch }: Props) {
  const wheelsEnabled = req.rw_h_max !== null
  const dumpEnabled = req.actuator_h_dump !== null

  return (
    <Section title="Actuator" icon="⚙️" defaultOpen={false}>
      <ToggleField label="Per-axis wheel torque limit" checked={req.actuator_tau_max !== null} onChange={(v) => patch({ actuator_tau_max: v ? 0.02 : null })} />
      {req.actuator_tau_max !== null && (
        <NumberField
          label="Wheel torque limit"
          value={scalarOf(req.actuator_tau_max, 0.02)}
          onChange={(v) => patch({ actuator_tau_max: v })}
          min={0.002}
          max={0.08}
          step={0.001}
          unit="N·m"
          precision={3}
        />
      )}

      <ToggleField label="First-order actuator lag" checked={req.actuator_tau !== null} onChange={(v) => patch({ actuator_tau: v ? 0.05 : null })} />
      {req.actuator_tau !== null && (
        <NumberField label="Lag time constant" value={req.actuator_tau} onChange={(v) => patch({ actuator_tau: v })} min={0.005} max={0.5} step={0.005} unit="s" precision={3} />
      )}

      <div style={{ borderTop: '1px solid var(--border-soft)', paddingTop: 8, marginTop: 2 }}>
        <ToggleField
          label="Reaction-wheel assembly (momentum storage)"
          checked={wheelsEnabled}
          onChange={(v) => patch({ rw_h_max: v ? 0.006 : null, rw_inertia: v ? req.rw_inertia : null })}
        />
        {wheelsEnabled && (
          <>
            <NumberField
              label="Wheel momentum limit h_max"
              value={scalarOf(req.rw_h_max, 0.006)}
              onChange={(v) => patch({ rw_h_max: v })}
              min={0.0005}
              max={0.05}
              step={0.0005}
              unit="N·m·s"
              precision={4}
            />
            <NumberField
              label="Wheel spin inertia I_w"
              value={scalarOf(req.rw_inertia, 2e-4)}
              onChange={(v) => patch({ rw_inertia: v })}
              min={2e-5}
              max={2e-3}
              unit="kg·m²"
              precision={5}
              logScale
            />
            <NumberField label="Viscous friction b" value={req.rw_visc} onChange={(v) => patch({ rw_visc: v })} min={0} max={1e-4} unit="N·m·s" precision={6} />
            <NumberField label="Coulomb friction c" value={req.rw_coulomb} onChange={(v) => patch({ rw_coulomb: v })} min={0} max={5e-4} unit="N·m" precision={6} />
            <ToggleField label="Include ω×h gyroscopic coupling" checked={!req.rw_no_gyro} onChange={(v) => patch({ rw_no_gyro: !v })} />
          </>
        )}
      </div>

      <div style={{ borderTop: '1px solid var(--border-soft)', paddingTop: 8 }}>
        <ToggleField label="Wheel-momentum dump" checked={dumpEnabled} onChange={(v) => patch({ actuator_h_dump: v ? 0.01 : null })} />
        {dumpEnabled && (
          <>
            <NumberField
              label="Dump deadzone threshold"
              value={scalarOf(req.actuator_h_dump, 0.01)}
              onChange={(v) => patch({ actuator_h_dump: v })}
              min={0.001}
              max={0.05}
              step={0.001}
              unit="N·m·s"
              precision={3}
            />
            <NumberField label="Dump gain" value={req.actuator_dump_gain} onChange={(v) => patch({ actuator_dump_gain: v })} min={0.1} max={5} step={0.1} unit="1/s" precision={1} />
          </>
        )}
      </div>
    </Section>
  )
}

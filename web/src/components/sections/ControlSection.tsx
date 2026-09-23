import { useState } from 'react'
import type { ControllerKind, EstimatorKind, ScenarioInfo, SimRequest } from '../../types'
import { NumberField, Segmented, ToggleField, Vec3Field } from '../fields'
import { Section } from '../Section'

interface Props {
  req: SimRequest
  patch: (p: Partial<SimRequest>) => void
  scenarioInfo?: ScenarioInfo
}

export function ControlSection({ req, patch, scenarioInfo }: Props) {
  const [showAdvanced, setShowAdvanced] = useState(false)
  const effectiveController: ControllerKind = req.controller ?? scenarioInfo?.default_controller ?? 'pid'

  return (
    <Section title="Guidance &amp; Control" icon="🎯">
      <div className="field-row">
        <span className="field-label">
          <span>Controller</span>
          <span className="value">{req.controller === null ? `auto (${effectiveController})` : effectiveController}</span>
        </span>
        <Segmented
          value={req.controller ?? 'auto'}
          onChange={(v) => patch({ controller: v === 'auto' ? null : (v as ControllerKind) })}
          options={[
            { value: 'auto', label: 'Auto' },
            { value: 'pid', label: 'PID' },
            { value: 'lqr', label: 'LQR' },
          ]}
        />
      </div>

      <div className="field-row">
        <span className="field-label">Estimator</span>
        <Segmented
          value={req.estimator}
          onChange={(v) => patch({ estimator: v as EstimatorKind })}
          options={[
            { value: 'truth', label: 'Truth' },
            { value: 'mekf', label: 'MEKF' },
            { value: 'mahony', label: 'Mahony' },
          ]}
        />
        <p className="hint">
          Truth feeds the controller the exact plant state (no sensors). MEKF/Mahony run the closed loop on a noisy gyro + mag/sun
          vector filter.
        </p>
      </div>

      {req.estimator !== 'truth' && (
        <ToggleField label="Coarse TRIAD init at t=0 (else start at true q0)" checked={req.coarse_init} onChange={(v) => patch({ coarse_init: v })} />
      )}

      <Vec3Field label="Principal inertia (kg·m²)" value={req.inertia} onChange={(v) => patch({ inertia: v })} min={0.01} max={0.3} step={0.001} />

      <button type="button" className="btn btn-ghost" style={{ justifyContent: 'flex-start', padding: '4px 0' }} onClick={() => setShowAdvanced((s) => !s)}>
        {showAdvanced ? '−' : '+'} Advanced tuning
      </button>

      {showAdvanced && (
        <>
          <NumberField
            label="Gain scale"
            value={req.gain_scale}
            onChange={(v) => patch({ gain_scale: v })}
            min={0.1}
            max={3}
            step={0.05}
            precision={2}
            hint="Multiplies the implemented Kp/Kd/Ki (PID) or K (LQR). 1.0 = design value."
          />
          <ToggleField
            label="Unlimited controller torque ball"
            checked={req.torque_limit === null}
            onChange={(v) => patch({ torque_limit: v ? null : 0.02 })}
          />
          {req.torque_limit !== null && (
            <NumberField
              label="Torque limit (Euclidean ball)"
              value={req.torque_limit}
              onChange={(v) => patch({ torque_limit: v })}
              min={0.002}
              max={0.08}
              step={0.001}
              unit="N·m"
              precision={3}
            />
          )}
          {effectiveController === 'pid' && (
            <>
              <NumberField label="PID ωn (bandwidth)" value={req.wn} onChange={(v) => patch({ wn: v })} min={0.1} max={1.5} step={0.01} unit="rad/s" precision={2} />
              <NumberField label="PID ζ (damping)" value={req.zeta} onChange={(v) => patch({ zeta: v })} min={0.3} max={2.0} step={0.01} precision={2} />
            </>
          )}
          {effectiveController === 'lqr' && (
            <p className="hint">LQR gains are re-solved from the Bryson Q/R weights every run (theta_ref/omega_ref fixed; R follows the torque limit above).</p>
          )}
        </>
      )}
    </Section>
  )
}

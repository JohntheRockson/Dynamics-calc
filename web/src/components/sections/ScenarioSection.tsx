import { useEffect } from 'react'
import { maxTFinalForDt } from '../../limits'
import type { Scenario, ScenarioInfo, SimRequest } from '../../types'
import { NumberField, Segmented, SelectField } from '../fields'
import { Section } from '../Section'

const DT_OPTIONS = [0.02, 0.01, 0.005, 0.002].map((v) => ({ value: String(v), label: `${v * 1000} ms` }))
const DURATION_CEILING = 120

interface Props {
  req: SimRequest
  patch: (p: Partial<SimRequest>) => void
  scenarios: ScenarioInfo[]
}

export function ScenarioSection({ req, patch, scenarios }: Props) {
  const info = scenarios.find((s) => s.key === req.scenario)
  const hasAngle = req.scenario === 'slew' || req.scenario === 'eigenaxis'
  const effectiveAngle = req.angle_deg ?? info?.default_angle_deg ?? 0
  const effectiveTFinal = req.t_final ?? info?.default_t_final ?? 20
  // The sample-count budget (dt x t_final) is enforced server-side, but we
  // also shrink the duration slider's ceiling client-side so a finer dt
  // can never silently produce a request the server would reject.
  const maxDuration = Math.min(DURATION_CEILING, maxTFinalForDt(req.dt))

  // `t_final: null` means "use the scenario's default duration", and that
  // default can outgrow the current dt's budget from *either* side of the
  // interaction -- picking a finer dt while a long scenario is selected, or
  // switching to a longer-default scenario while a fine dt is selected.
  // A plain onChange clamp only catches one direction, so enforce the
  // invariant here instead: whenever the *resolved* duration exceeds what
  // this dt allows, pin `t_final` down to the cap explicitly.
  useEffect(() => {
    if (effectiveTFinal > maxDuration) {
      patch({ t_final: maxDuration })
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [effectiveTFinal, maxDuration])

  return (
    <Section title="Scenario" icon="🛰️" defaultOpen>
      <Segmented
        value={req.scenario}
        onChange={(v: Scenario) => patch({ scenario: v, angle_deg: null, t_final: null })}
        options={scenarios.map((s) => ({ value: s.key, label: s.title.split(' ')[0] }))}
      />
      {info && <p className="hint">{info.blurb}</p>}

      {hasAngle && (
        <NumberField
          label="Commanded angle"
          value={effectiveAngle}
          onChange={(v) => patch({ angle_deg: v })}
          min={0}
          max={180}
          step={1}
          unit="deg"
          precision={0}
        />
      )}

      <NumberField
        label="Duration"
        value={Math.min(effectiveTFinal, maxDuration)}
        onChange={(v) => patch({ t_final: v })}
        min={2}
        max={maxDuration}
        step={1}
        unit="s"
        precision={0}
        hint={maxDuration < DURATION_CEILING ? `Capped at ${maxDuration}s for this dt (~${Math.round(maxDuration / req.dt).toLocaleString()} samples/run).` : undefined}
      />

      <div className="field-row">
        <SelectField label="Sample step (dt)" value={String(req.dt)} options={DT_OPTIONS} onChange={(v) => patch({ dt: parseFloat(v) })} />
      </div>

      <div className="field-row inline">
        <span className="field-label" style={{ flex: 1 }}>
          RNG seed
        </span>
        <input
          className="num-input"
          id="rng-seed"
          name="rng-seed"
          style={{ width: 90 }}
          type="number"
          aria-label="RNG seed"
          value={req.seed}
          onChange={(e) => patch({ seed: Math.max(0, Math.floor(parseFloat(e.target.value) || 0)) })}
        />
        <button type="button" className="btn btn-ghost btn-icon" title="Randomize seed" onClick={() => patch({ seed: Math.floor(Math.random() * 1_000_000) })}>
          &#127922;
        </button>
      </div>
    </Section>
  )
}

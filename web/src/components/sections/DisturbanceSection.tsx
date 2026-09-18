import type { DipoleModel, ScenarioInfo, SimRequest, SrpEclipseMode } from '../../types'
import { NumberField, SelectField, Segmented, ToggleField, Vec3Field } from '../fields'
import { Section } from '../Section'

interface Props {
  req: SimRequest
  patch: (p: Partial<SimRequest>) => void
  scenarioInfo?: ScenarioInfo
}

type EnvMode = 'auto' | 'on' | 'off'

function envModeOf(req: SimRequest): EnvMode {
  if (req.no_env) return 'off'
  if (req.env) return 'on'
  return 'auto'
}

export function DisturbanceSection({ req, patch, scenarioInfo }: Props) {
  const envMode = envModeOf(req)
  const showOrbitParams = req.gravity_gradient || req.residual_dipole || req.aerodynamic || req.srp || envMode !== 'auto'
  const showPanelParams = req.aerodynamic || req.srp

  return (
    <Section title="Disturbances" icon="🌌" defaultOpen={false}>
      <Vec3Field label="Constant body torque τ_d" value={req.tau_dist} onChange={(v) => patch({ tau_dist: v })} min={-0.02} max={0.02} step={0.0001} unit="N·m" />

      <div className="field-row">
        <span className="field-label">
          <span>Environmental torques (GG + dipole)</span>
          <span className="value">{scenarioInfo?.env_by_default ? 'hold: on by default' : 'off by default'}</span>
        </span>
        <Segmented
          value={envMode}
          onChange={(v: EnvMode) => patch({ env: v === 'on', no_env: v === 'off' })}
          options={[
            { value: 'auto', label: 'Auto' },
            { value: 'on', label: 'Force on' },
            { value: 'off', label: 'Force off' },
          ]}
        />
        <p className="hint">Force off overrides everything below; force on enables gravity-gradient + residual dipole regardless of their toggles.</p>
      </div>

      <ToggleField label="Gravity-gradient torque" checked={req.gravity_gradient} onChange={(v) => patch({ gravity_gradient: v })} disabled={envMode === 'off'} />
      <ToggleField label="Residual magnetic dipole" checked={req.residual_dipole} onChange={(v) => patch({ residual_dipole: v })} disabled={envMode === 'off'} />
      <ToggleField label="Aerodynamic drag torque" checked={req.aerodynamic} onChange={(v) => patch({ aerodynamic: v })} />
      <ToggleField label="Solar radiation pressure" checked={req.srp} onChange={(v) => patch({ srp: v })} />

      {showOrbitParams && (
        <>
          <div style={{ borderTop: '1px solid var(--border-soft)', paddingTop: 8, marginTop: 2 }}>
            <p className="hint" style={{ marginBottom: 6 }}>
              Circular-orbit geometry (shared by all environment models)
            </p>
            <NumberField
              label="Orbit radius"
              value={req.orbit_radius}
              onChange={(v) => patch({ orbit_radius: v })}
              min={6.6e6}
              max={4.2e7}
              unit="m"
              precision={0}
              logScale
            />
            <NumberField
              label="Inclination"
              value={req.orbit_inc_deg ?? (scenarioInfo?.env_by_default ? 51.6 : 0)}
              onChange={(v) => patch({ orbit_inc_deg: v })}
              min={0}
              max={98}
              step={0.1}
              unit="deg"
              precision={1}
            />
            <NumberField label="RAAN" value={req.orbit_raan_deg} onChange={(v) => patch({ orbit_raan_deg: v })} min={0} max={360} step={1} unit="deg" precision={0} />
          </div>
        </>
      )}

      {req.residual_dipole && (
        <div style={{ borderTop: '1px solid var(--border-soft)', paddingTop: 8 }}>
          <Vec3Field
            label="Residual body dipole m"
            value={req.dipole_m ?? [40, -15, 8]}
            onChange={(v) => patch({ dipole_m: v })}
            min={-100}
            max={100}
            step={0.5}
            unit="A·m²"
          />
          <SelectField
            label="Earth field model"
            value={req.dipole_model}
            onChange={(v: DipoleModel) => patch({ dipole_model: v })}
            options={[
              { value: 'tilted', label: 'Tilted dipole (11.5°)' },
              { value: 'orbit_normal', label: 'Orbit-normal (equatorial)' },
            ]}
          />
        </div>
      )}

      {showPanelParams && (
        <div style={{ borderTop: '1px solid var(--border-soft)', paddingTop: 8 }}>
          <p className="hint" style={{ marginBottom: 6 }}>
            Panel / ram surface (shared by aero + SRP)
          </p>
          <NumberField label="Panel area" value={req.panel_area} onChange={(v) => patch({ panel_area: v })} min={0.01} max={2} step={0.01} unit="m²" precision={2} />
          <Vec3Field label="Centre of pressure r_cp" value={req.panel_rcp} onChange={(v) => patch({ panel_rcp: v })} min={-0.3} max={0.3} step={0.005} unit="m" />
          {req.aerodynamic && (
            <NumberField label="Drag coefficient Cd" value={req.aero_cd} onChange={(v) => patch({ aero_cd: v })} min={1.0} max={3.5} step={0.05} precision={2} />
          )}
          {req.srp && (
            <>
              <NumberField label="Reflectivity Cr" value={req.srp_cr} onChange={(v) => patch({ srp_cr: v })} min={0} max={2} step={0.05} precision={2} />
              <SelectField
                label="Eclipse"
                value={req.srp_eclipse}
                onChange={(v: SrpEclipseMode) => patch({ srp_eclipse: v })}
                options={[
                  { value: 'off', label: 'Sunlit (off)' },
                  { value: 'on', label: 'Forced eclipse' },
                  { value: 'cylindrical', label: 'Cylindrical Earth shadow' },
                ]}
              />
            </>
          )}
        </div>
      )}
    </Section>
  )
}

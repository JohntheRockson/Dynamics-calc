import type { ScenarioInfo, SimRequest } from '../types'
import { ActuatorSection } from './sections/ActuatorSection'
import { ControlSection } from './sections/ControlSection'
import { DisturbanceSection } from './sections/DisturbanceSection'
import { ScenarioSection } from './sections/ScenarioSection'
import { SensorSection } from './sections/SensorSection'

interface Props {
  req: SimRequest
  setReq: (updater: (prev: SimRequest) => SimRequest) => void
  scenarios: ScenarioInfo[]
  onRun: () => void
  onReset: () => void
  loading: boolean
}

export function ControlPanel({ req, setReq, scenarios, onRun, onReset, loading }: Props) {
  const patch = (p: Partial<SimRequest>) => setReq((prev) => ({ ...prev, ...p }))
  const scenarioInfo = scenarios.find((s) => s.key === req.scenario)

  return (
    <aside className="sidebar">
      <div style={{ display: 'flex', gap: 8 }}>
        <button type="button" className="btn btn-primary" style={{ flex: 1 }} onClick={onRun} disabled={loading}>
          {loading ? (
            <>
              <span className="spinner" /> Running&hellip;
            </>
          ) : (
            <>&#9654; Run simulation</>
          )}
        </button>
        <button type="button" className="btn btn-ghost btn-icon" title="Reset all settings" onClick={onReset} disabled={loading}>
          &#8635;
        </button>
      </div>

      <ScenarioSection req={req} patch={patch} scenarios={scenarios} />
      <ControlSection req={req} patch={patch} scenarioInfo={scenarioInfo} />
      <DisturbanceSection req={req} patch={patch} scenarioInfo={scenarioInfo} />
      <ActuatorSection req={req} patch={patch} />
      <SensorSection req={req} patch={patch} />
    </aside>
  )
}

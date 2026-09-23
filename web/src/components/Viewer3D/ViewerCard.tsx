import { useMemo, useState } from 'react'
import type { SimLog } from '../../types'
import { formatNumber } from '../../format'
import { SatelliteViewer } from './SatelliteViewer'

interface Props {
  log: SimLog | null
  time: number
}

function sampleAt(log: SimLog, time: number) {
  const dt = log.resolved.dt
  const idx = Math.min(log.t.length - 1, Math.max(0, Math.round(time / dt)))
  return idx
}

export function ViewerCard({ log, time }: Props) {
  const [showEstimate, setShowEstimate] = useState(true)
  const idx = useMemo(() => (log ? sampleAt(log, time) : 0), [log, time])

  const attErr = log ? log.att_error_deg[idx] : null
  const rate = log ? Math.hypot(...log.omega[idx]) : null
  const tau = log ? Math.hypot(...log.tau[idx]) : null
  const hasEstimator = log ? log.resolved.estimator !== 'truth' : false

  return (
    <div className="viewer-card">
      <SatelliteViewer log={log} time={time} showEstimate={showEstimate} />
      <div className="viewer-overlay">
        <div className="viewer-hud">
          <div className="legend-axes">
            <span>
              <span className="dot" style={{ background: '#fb6a6a' }} />x
            </span>
            <span>
              <span className="dot" style={{ background: '#59d67f' }} />y
            </span>
            <span>
              <span className="dot" style={{ background: '#5aa8ff' }} />z
            </span>
            <span style={{ color: 'var(--text-faint)' }}>body axes &middot; cyan wireframe = target &middot; amber = estimate</span>
          </div>
          <div className="hud-card">
            <div className="row">
              <span>t</span>
              <b>{log ? `${time.toFixed(2)} / ${log.resolved.t_final.toFixed(1)} s` : '-'}</b>
            </div>
            <div className="row">
              <span>err</span>
              <b>{attErr !== null ? `${formatNumber(attErr, 2)}°` : '-'}</b>
            </div>
            <div className="row">
              <span>|ω|</span>
              <b>{rate !== null ? `${formatNumber(rate, 3)} r/s` : '-'}</b>
            </div>
            <div className="row">
              <span>|τ|</span>
              <b>{tau !== null ? `${formatNumber(tau, 4)} N·m` : '-'}</b>
            </div>
            {hasEstimator && (
              <label style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 6, cursor: 'pointer' }} htmlFor="show-estimate-toggle">
                <input id="show-estimate-toggle" name="show-estimate-toggle" type="checkbox" checked={showEstimate} onChange={(e) => setShowEstimate(e.target.checked)} />
                <span>show estimate</span>
              </label>
            )}
          </div>
        </div>
        {!log && (
          <div className="empty-state" style={{ pointerEvents: 'none' }}>
            <span className="big">&#128752;</span>
            <span>Configure a scenario on the left and run the simulation to see the spacecraft fly.</span>
          </div>
        )}
      </div>
    </div>
  )
}

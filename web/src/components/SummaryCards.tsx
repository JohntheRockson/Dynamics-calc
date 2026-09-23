import type { SimLog } from '../types'
import { formatNumber } from '../format'

interface Props {
  log: SimLog
}

export function SummaryCards({ log }: Props) {
  const s = log.summary
  const r = log.resolved
  return (
    <div className="panel">
      <div className="panel-header">
        <h2>Run summary</h2>
        <div style={{ display: 'flex', gap: 6 }}>
          <span className={`badge badge-${r.controller}`}>{r.controller}</span>
          <span className={`badge badge-${r.estimator}`}>{r.estimator}</span>
          {r.env_active && <span className="badge">env</span>}
          {r.reaction_wheels_active && <span className="badge">wheels</span>}
        </div>
      </div>
      <div className="panel-body">
        <div className="summary-grid">
          <div className="stat-card">
            <div className="label">Final error</div>
            <div className="val">
              {formatNumber(s.final_att_error_deg, 3)}
              <span className="unit">deg</span>
            </div>
          </div>
          <div className="stat-card">
            <div className="label">Peak |ω|</div>
            <div className="val">
              {formatNumber(s.peak_rate, 3)}
              <span className="unit">rad/s</span>
            </div>
          </div>
          <div className="stat-card">
            <div className="label">Peak |τ|</div>
            <div className="val">
              {formatNumber(s.peak_tau, 4)}
              <span className="unit">N·m</span>
            </div>
          </div>
          {s.sat_fraction !== null && (
            <div className={`stat-card ${s.sat_fraction > 0.2 ? 'warn' : ''}`}>
              <div className="label">RW saturation</div>
              <div className="val">{formatNumber(s.sat_fraction * 100, 1)}%</div>
            </div>
          )}
          {s.mean_nis !== null && (
            <div className="stat-card">
              <div className="label">Mean NIS (E=2)</div>
              <div className="val">{formatNumber(s.mean_nis, 2)}</div>
            </div>
          )}
          <div className="stat-card">
            <div className="label">Samples</div>
            <div className="val" style={{ fontSize: 15 }}>
              {r.n_samples.toLocaleString()}
              <span className="unit">@ {r.dt * 1000}ms</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

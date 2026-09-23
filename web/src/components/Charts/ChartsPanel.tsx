import { useMemo, useState } from 'react'
import type { SimLog } from '../../types'
import { TimeSeriesChart, type ChartSeries } from './TimeSeriesChart'

const AXIS_COLORS = ['#fb6a6a', '#59d67f', '#5aa8ff']

function column(data: [number, number, number][], i: number): number[] {
  return data.map((row) => row[i])
}

function buildNisSeries(log: SimLog): ChartSeries[] {
  const bySensor = new Map<string, Map<number, number>>()
  for (const sample of log.nis) {
    if (!bySensor.has(sample.sensor)) bySensor.set(sample.sensor, new Map())
    const idx = Math.round(sample.t / log.resolved.dt)
    bySensor.get(sample.sensor)!.set(idx, sample.nis)
  }
  const colors = ['#f5a524', '#29d3f5', '#a78bfa']
  let ci = 0
  const series: ChartSeries[] = []
  for (const [sensor, m] of bySensor) {
    const data = log.t.map((_, i) => m.get(i) ?? NaN)
    series.push({ label: sensor, color: colors[ci % colors.length], data, dotted: true })
    ci++
  }
  return series
}

interface TabDef {
  id: string
  label: string
  build: (log: SimLog) => { series: ChartSeries[]; referenceLines?: { value: number; color: string; label: string }[] }
  show: (log: SimLog) => boolean
}

const TABS: TabDef[] = [
  {
    id: 'error',
    label: 'Attitude error',
    show: () => true,
    build: (log) => {
      const series: ChartSeries[] = [{ label: 'true error (deg)', color: '#29d3f5', data: log.att_error_deg }]
      if (log.est_att_error_deg) {
        series.push({ label: 'estimator error vs truth (deg)', color: '#f5a524', data: log.est_att_error_deg, dashed: true })
      }
      if (log.att_error_hat_deg) {
        series.push({ label: 'controller-perceived error (deg)', color: '#a78bfa', data: log.att_error_hat_deg, dashed: true })
      }
      return { series }
    },
  },
  {
    id: 'euler',
    label: 'Euler angles',
    show: () => true,
    build: (log) => ({
      series: ['yaw', 'pitch', 'roll'].map((label, i) => ({ label: `${label} (deg)`, color: AXIS_COLORS[i], data: column(log.euler_deg, i) })),
    }),
  },
  {
    id: 'rate',
    label: 'Body rate',
    show: () => true,
    build: (log) => ({
      series: ['ωx', 'ωy', 'ωz'].map((label, i) => ({ label: `${label} (rad/s)`, color: AXIS_COLORS[i], data: column(log.omega, i) })),
    }),
  },
  {
    id: 'torque',
    label: 'Control torque',
    show: () => true,
    build: (log) => {
      const series: ChartSeries[] = ['τx', 'τy', 'τz'].map((label, i) => ({
        label: `${label} (N·m)`,
        color: AXIS_COLORS[i],
        data: column(log.tau, i),
      }))
      if (log.resolved.env_active) {
        ;['τenv,x', 'τenv,y', 'τenv,z'].forEach((label, i) =>
          series.push({ label, color: AXIS_COLORS[i], data: column(log.tau_env, i), dashed: true }),
        )
      }
      return { series }
    },
  },
  {
    id: 'wheels',
    label: 'Reaction wheels',
    show: (log) => log.resolved.reaction_wheels_active,
    build: (log) => ({
      series: ['hx', 'hy', 'hz'].map((label, i) => ({ label: `${label} (N·m·s)`, color: AXIS_COLORS[i], data: column(log.h_wheel, i) })),
    }),
  },
  {
    id: 'nis',
    label: 'Estimator NIS',
    show: (log) => log.nis.length > 0,
    build: (log) => ({ series: buildNisSeries(log), referenceLines: [{ value: 2, color: '#8b98a9', label: 'E[NIS]=2 (rank-2)' }] }),
  },
]

interface Props {
  log: SimLog | null
  time: number
}

export function ChartsPanel({ log, time }: Props) {
  const [tab, setTab] = useState('error')
  const visibleTabs = useMemo(() => (log ? TABS.filter((tdef) => tdef.show(log)) : []), [log])
  const active = visibleTabs.find((tdef) => tdef.id === tab) ?? visibleTabs[0]
  const built = useMemo(() => (log && active ? active.build(log) : null), [log, active])

  return (
    <div className="panel">
      <div className="panel-header">
        <h2>Telemetry</h2>
      </div>
      <div className="charts-tabs">
        {visibleTabs.map((tdef) => (
          <button key={tdef.id} type="button" className={`chart-tab ${tdef.id === active?.id ? 'active' : ''}`} onClick={() => setTab(tdef.id)}>
            {tdef.label}
          </button>
        ))}
      </div>
      <div className="chart-wrap">
        {log && built ? (
          <TimeSeriesChart t={log.t} series={built.series} currentTime={time} referenceLines={built.referenceLines} />
        ) : (
          <div style={{ height: 190, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-faint)', fontSize: 12.5 }}>
            Run a simulation to see telemetry charts.
          </div>
        )}
      </div>
    </div>
  )
}

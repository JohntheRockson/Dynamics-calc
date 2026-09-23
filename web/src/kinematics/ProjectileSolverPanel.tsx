import { useEffect, useMemo, useState } from 'react'
import { Segmented, SelectField } from '../components/fields'
import { PlaybackBar } from '../components/PlaybackBar'
import { TimeSeriesChart } from '../components/Charts/TimeSeriesChart'
import { formatNumber } from '../format'
import { usePlayback } from '../hooks/usePlayback'
import { Eq, EqList } from './Eq'
import { Scene2D } from './Scene2D'
import { sampleAt } from './physics'
import {
  convertLength,
  gDefault,
  solve2D,
  trajectory2d,
  type LengthUnit,
  type Sol2D,
} from './solver2d'
import { KnownRow, PlainRow, ResultCard } from './solverUi'

interface Preset2D {
  id: string
  label: string
  unit: LengthUnit
  xA: number
  yA: number
  xB: number | null
  yB: number | null
  v0: number | null
  thetaDeg: number | null
  t: number | null
  useXB: boolean
  useYB: boolean
  useV0: boolean
  useTh: boolean
  useT: boolean
}

const PRESETS: Preset2D[] = [
  {
    id: 'snowblower',
    label: 'Hibbeler 11.105 — snowblower (find v₀)',
    unit: 'ft',
    xA: 0,
    yA: 2,
    xB: 14,
    yB: 3.5,
    v0: null,
    thetaDeg: 40,
    t: null,
    useXB: true,
    useYB: true,
    useV0: false,
    useTh: true,
    useT: false,
  },
  {
    id: 'window',
    label: 'Throw through a window — find v₀',
    unit: 'm',
    xA: 0,
    yA: 1.5,
    xB: 12,
    yB: 4,
    v0: null,
    thetaDeg: 50,
    t: null,
    useXB: true,
    useYB: true,
    useV0: false,
    useTh: true,
    useT: false,
  },
  {
    id: 'range',
    label: 'Given v₀ and θ, where is it at t',
    unit: 'm',
    xA: 0,
    yA: 0,
    xB: null,
    yB: null,
    v0: 20,
    thetaDeg: 35,
    t: 2,
    useXB: false,
    useYB: false,
    useV0: true,
    useTh: true,
    useT: true,
  },
  {
    id: 'two-angles',
    label: 'Given v₀ and B, find the two θ',
    unit: 'm',
    xA: 0,
    yA: 0,
    xB: 30,
    yB: 8,
    v0: 25,
    thetaDeg: null,
    t: null,
    useXB: true,
    useYB: true,
    useV0: true,
    useTh: false,
    useT: false,
  },
]

export function ProjectileSolverPanel() {
  const [presetId, setPresetId] = useState('snowblower')
  const [unit, setUnit] = useState<LengthUnit>('ft')
  const [g, setG] = useState(gDefault('ft'))
  const [gravityDown, setGravityDown] = useState(true)
  const [ax, setAx] = useState<number | null>(0)
  const [ay, setAy] = useState<number | null>(-gDefault('ft'))

  const [xA, setXA] = useState<number | null>(0)
  const [yA, setYA] = useState<number | null>(2)
  const [xB, setXB] = useState<number | null>(14)
  const [yB, setYB] = useState<number | null>(3.5)
  const [v0, setV0] = useState<number | null>(null)
  const [theta, setTheta] = useState<number | null>(40)
  const [t, setT] = useState<number | null>(null)

  const [useXB, setUseXB] = useState(true)
  const [useYB, setUseYB] = useState(true)
  const [useV0, setUseV0] = useState(false)
  const [useTh, setUseTh] = useState(true)
  const [useT, setUseT] = useState(false)
  const [altIndex, setAltIndex] = useState(0)

  const Lu = unit
  const Vu = unit === 'ft' ? 'ft/s' : 'm/s'
  const Au = unit === 'ft' ? 'ft/s²' : 'm/s²'

  const loadPreset = (id: string) => {
    const p = PRESETS.find((x) => x.id === id)
    if (!p) return
    setPresetId(id)
    const next = p
    if (unit !== next.unit) {
      setUnit(next.unit)
      setG(gDefault(next.unit))
    } else {
      setG(gDefault(next.unit))
    }
    setGravityDown(true)
    setAx(0)
    setAy(-gDefault(next.unit))
    setXA(next.xA)
    setYA(next.yA)
    setXB(next.xB)
    setYB(next.yB)
    setV0(next.v0)
    setTheta(next.thetaDeg)
    setT(next.t)
    setUseXB(next.useXB)
    setUseYB(next.useYB)
    setUseV0(next.useV0)
    setUseTh(next.useTh)
    setUseT(next.useT)
    setAltIndex(0)
  }

  const switchUnit = (to: LengthUnit) => {
    if (to === unit) return
    const conv = (v: number | null) => (v === null ? null : convertLength(v, unit, to))
    const convG = convertLength(g, unit, to)
    const snapped = Math.abs(convG - gDefault(to)) / gDefault(to) < 0.02 ? gDefault(to) : convG
    setXA(conv(xA))
    setYA(conv(yA))
    setXB(conv(xB))
    setYB(conv(yB))
    setV0(conv(v0))
    setAx(conv(ax))
    setAy(gravityDown ? -snapped : conv(ay))
    setG(snapped)
    setUnit(to)
    setPresetId('_custom')
  }

  const ayUsed = gravityDown ? -g : (ay ?? 0)
  const axUsed = gravityDown ? 0 : (ax ?? 0)

  const result = useMemo(
    () =>
      solve2D({
        xA: xA ?? 0,
        yA: yA ?? 0,
        ax: axUsed,
        ay: ayUsed,
        xB: useXB ? xB : null,
        yB: useYB ? yB : null,
        v0: useV0 ? v0 : null,
        thetaDeg: useTh ? theta : null,
        t: useT ? t : null,
      }),
    [xA, yA, axUsed, ayUsed, useXB, xB, useYB, yB, useV0, v0, useTh, theta, useT, t],
  )

  const points = result.ok ? [result.primary, ...result.alternatives] : []
  const shown: Sol2D | undefined = points[Math.min(altIndex, Math.max(0, points.length - 1))]
  const history = useMemo(() => {
    if (!shown) return { t: [0], x: [0], y: [0], vx: [0], vy: [0] }
    return trajectory2d(shown)
  }, [shown])

  const tStart = history.t[0] ?? 0
  const tEnd = history.t[history.t.length - 1] ?? 0
  const duration = Math.max(0, tEnd - tStart)
  const playback = usePlayback(Math.max(duration, 1e-6))
  useEffect(() => {
    playback.reset(Math.max(duration, 1e-6))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [duration, shown?.t, shown?.v0, shown?.thetaDeg])
  useEffect(() => {
    setAltIndex(0)
  }, [unit, xA, yA, xB, yB, v0, theta, t, useXB, useYB, useV0, useTh, useT, axUsed, ayUsed])

  const clock = tStart + playback.time
  const xNow = sampleAt(history.t, history.x, clock)
  const yNow = sampleAt(history.t, history.y, clock)
  const vxNow = sampleAt(history.t, history.vx, clock)
  const vyNow = sampleAt(history.t, history.vy, clock)
  const idx = useMemo(() => {
    if (history.t.length < 2) return 0
    const dt = (history.t[history.t.length - 1] - history.t[0]) / (history.t.length - 1)
    return Math.min(history.t.length - 1, Math.max(0, Math.round((clock - history.t[0]) / (dt || 1))))
  }, [history.t, clock])

  const len = (n: number) => `${formatNumber(n, 4)} ${Lu}`

  return (
    <div className="kin-grid kin-grid-solver">
      <div className="kin-controls panel">
        <div className="panel-header">
          <h2>Put in the figure</h2>
        </div>
        <div className="panel-body" style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <p className="hint">
            Constant <Eq tex="\mathbf{a}" /> (usually gravity) and the displacement <Eq tex="\Delta\mathbf{r}" /> to
            point B, plus the launch angle from the figure, determine <Eq tex="v_0" /> and <Eq tex="t" />. Leave
            unknowns blank.
          </p>

          <SelectField
            label="Load example"
            value={presetId}
            onChange={loadPreset}
            options={[{ value: '_custom', label: 'Custom' }, ...PRESETS.map((p) => ({ value: p.id, label: p.label }))]}
          />

          <Segmented
            value={unit}
            onChange={switchUnit}
            options={[
              { value: 'ft', label: 'US (ft)' },
              { value: 'm', label: 'SI (m)' },
            ]}
          />

          <div className="known-block">
            <div className="known-heading">Constant acceleration</div>
            <label className="toggle-row" htmlFor="g-down">
              <span className="toggle">
                <input
                  id="g-down"
                  type="checkbox"
                  checked={gravityDown}
                  onChange={(e) => {
                    setGravityDown(e.target.checked)
                    if (e.target.checked) {
                      setAx(0)
                      setAy(-g)
                    }
                    setPresetId('_custom')
                  }}
                />
                <span className="track">
                  <span className="thumb" />
                </span>
              </span>
              <span>
                Gravity down, <Eq tex="a_x=0" />, <Eq tex={`a_y=-g`} />
              </span>
            </label>
            {gravityDown ? (
              <PlainRow
                symbol="g"
                name="gravity magnitude"
                unit={Au}
                value={g}
                onChange={(n) => {
                  if (n === null) return
                  setG(n)
                  setAy(-n)
                  setPresetId('_custom')
                }}
              />
            ) : (
              <div className="known-pair">
                <PlainRow symbol="a_x" name="ax" unit={Au} value={ax} onChange={setAx} />
                <PlainRow symbol="a_y" name="ay" unit={Au} value={ay} onChange={setAy} />
              </div>
            )}
            <p className="hint">{unit === 'ft' ? 'Hibbeler US customary: g = 32.2 ft/s²' : 'SI: g = 9.81 m/s²'}</p>
          </div>

          <div className="known-block">
            <div className="known-heading">Point A — start</div>
            <div className="known-pair">
              <PlainRow symbol="x_A" name="xA" unit={Lu} value={xA} onChange={setXA} />
              <PlainRow symbol="y_A" name="yA" unit={Lu} value={yA} onChange={setYA} />
            </div>
            <KnownRow symbol="\\theta" name="launch angle" unit="deg" known={useTh} value={theta} onKnown={setUseTh} onValue={setTheta} />
            <KnownRow symbol="v_0" name="launch speed" unit={Vu} known={useV0} value={v0} onKnown={setUseV0} onValue={setV0} />
          </div>

          <div className="known-block">
            <div className="known-heading">Point B — where it must go</div>
            <div className="known-pair">
              <KnownRow symbol="x_B" name="xB" unit={Lu} known={useXB} value={xB} onKnown={setUseXB} onValue={setXB} />
              <KnownRow symbol="y_B" name="yB" unit={Lu} known={useYB} value={yB} onKnown={setUseYB} onValue={setYB} />
            </div>
            <KnownRow symbol="t" name="time A to B" unit="s" known={useT} value={t} onKnown={setUseT} onValue={setT} />
            <p className="hint">
              Δx = {len((xB ?? 0) - (xA ?? 0))}, Δy = {len((yB ?? 0) - (yA ?? 0))}
              {useXB && useYB ? ' · position known' : ''}
            </p>
          </div>
        </div>
      </div>

      <div className="kin-results">
        <div className="panel">
          <div className="panel-header">
            <h2>Governing relations</h2>
          </div>
          <div className="panel-body">
            <EqList
              items={
                result.ok
                  ? result.method
                  : [
                      '\\Delta\\mathbf{r} = \\mathbf{v}_0 t + \\tfrac12 \\mathbf{a} t^2',
                      '\\mathbf{v} = \\mathbf{v}_0 + \\mathbf{a} t',
                      'v_{0x} = v_0\\cos\\theta,\\quad v_{0y} = v_0\\sin\\theta',
                    ]
              }
            />
          </div>
        </div>

        {result.ok && shown ? (
          <>
            <div className="summary-grid">
              <ResultCard label="Launch speed v₀" value={shown.v0} unit={Vu} computed={result.computed.v0} />
              <ResultCard label="Launch angle θ" value={shown.thetaDeg} unit="°" computed={result.computed.theta} />
              <ResultCard label="Time A → B" value={shown.t} unit="s" computed={result.computed.t} />
              <ResultCard label="Speed at B" value={Math.hypot(shown.vxB, shown.vyB)} unit={Vu} computed />
            </div>
            <div className="summary-grid">
              <ResultCard label="x_B" value={shown.xB} unit={Lu} computed={result.computed.xB} />
              <ResultCard label="y_B" value={shown.yB} unit={Lu} computed={result.computed.yB} />
              <ResultCard label="v_x at A" value={shown.vxA} unit={Vu} computed={result.computed.v0 || result.computed.theta} />
              <ResultCard label="v_y at A" value={shown.vyA} unit={Vu} computed={result.computed.v0 || result.computed.theta} />
            </div>
            {result.notes.map((n) => (
              <p key={n} className="hint">
                {n}
              </p>
            ))}
            {points.length > 1 && (
              <div className="alt-roots">
                {points.map((p, i) => (
                  <button
                    key={`${p.t}-${p.thetaDeg}`}
                    type="button"
                    className={i === altIndex ? 'active' : ''}
                    onClick={() => setAltIndex(i)}
                  >
                    root {i + 1}: θ = {formatNumber(p.thetaDeg, 2)}°, t = {formatNumber(p.t, 3)} s, v₀ = {formatNumber(p.v0, 3)} {Vu}
                  </button>
                ))}
              </div>
            )}
          </>
        ) : (
          <div className="panel">
            <div className="panel-body">
              <p className="solver-error">{result.ok ? '' : result.error}</p>
            </div>
          </div>
        )}

        {result.ok && shown && (
          <>
            <div className="panel">
              <div className="panel-header">
                <h2>Trajectory</h2>
              </div>
              <div className="panel-body">
                <Scene2D
                  height={280}
                  aspectEqual
                  xLabel={`x (${Lu})`}
                  yLabel={`y (${Lu})`}
                  markers={[
                    { x: shown.xA, y: shown.yA, label: 'A', color: '#59d67f' },
                    { x: shown.xB, y: shown.yB, label: 'B', color: '#f5a524' },
                  ]}
                  bodies={[
                    {
                      label: 'path',
                      color: '#29d3f5',
                      path: history.x.map((x, i) => ({ x, y: history.y[i] })),
                      currentIndex: idx,
                      velocityVector: { vx: vxNow, vy: vyNow },
                    },
                  ]}
                />
                <p className="hint" style={{ marginTop: 8 }}>
                  Now: ({formatNumber(xNow, 3)}, {formatNumber(yNow, 3)}) {Lu} · speed {formatNumber(Math.hypot(vxNow, vyNow), 3)} {Vu}
                </p>
              </div>
            </div>
            <PlaybackBar playback={playback} disabled={duration < 1e-9} />
            <div className="panel">
              <div className="panel-header">
                <h2>x, y, speed vs t</h2>
              </div>
              <div className="panel-body" style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                <TimeSeriesChart
                  t={history.t}
                  series={[
                    { label: `x (${Lu})`, color: '#29d3f5', data: history.x },
                    { label: `y (${Lu})`, color: '#f5a524', data: history.y },
                  ]}
                  currentTime={clock}
                  height={140}
                />
                <TimeSeriesChart
                  t={history.t}
                  series={[{ label: `speed (${Vu})`, color: '#59d67f', data: history.vx.map((vx, i) => Math.hypot(vx, history.vy[i])) }]}
                  currentTime={clock}
                  height={120}
                />
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  )
}

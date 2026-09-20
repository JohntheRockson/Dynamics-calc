import { useEffect, useMemo, useState } from 'react'
import { Segmented, SelectField } from '../components/fields'
import { PlaybackBar } from '../components/PlaybackBar'
import { TimeSeriesChart } from '../components/Charts/TimeSeriesChart'
import { formatNumber } from '../format'
import { usePlayback } from '../hooks/usePlayback'
import { Eq, EqList } from './Eq'
import { parseExpr } from './expr'
import { sampleAt } from './physics'
import { ProjectileSolverPanel } from './ProjectileSolverPanel'
import { constAHistory, solveConstA, solveVariable, type RelationKind, type SolverResult, type TargetKind } from './solver'
import { KnownRow, PlainRow, ResultCard } from './solverUi'

type Mode = RelationKind

interface Preset {
  id: string
  label: string
  mode: Mode
  expr?: string
  s0: number
  v0: number
  t0?: number
  s: number | null
  v: number | null
  a: number | null
  t: number | null
  use: { v0: boolean; s: boolean; v: boolean; a: boolean; t: boolean }
  target?: TargetKind
  tHorizon?: number
}

const PRESETS: Preset[] = [
  {
    id: 'brake',
    label: 'Braking car — know v₀, v, a',
    mode: 'const-a',
    s0: 0,
    v0: 25,
    s: null,
    v: 0,
    a: -4,
    t: null,
    use: { v0: true, s: false, v: true, a: true, t: false },
  },
  {
    id: 'drop',
    label: 'Free fall — know v₀, a, t',
    mode: 'const-a',
    s0: 0,
    v0: 0,
    s: null,
    v: null,
    a: -9.81,
    t: 2,
    use: { v0: true, s: false, v: false, a: true, t: true },
  },
  {
    id: 'how-long',
    label: 'How long to go 40 m from rest — know v₀, a, s',
    mode: 'const-a',
    s0: 0,
    v0: 0,
    s: 40,
    v: null,
    a: 2,
    t: null,
    use: { v0: true, s: true, v: false, a: true, t: false },
  },
  {
    id: 'at',
    label: 'a = 4t − 3, find v and s at t = 4',
    mode: 'a-expr',
    expr: '4*t - 3',
    s0: 0,
    v0: 2,
    t0: 0,
    s: null,
    v: null,
    a: null,
    t: 4,
    use: { v0: true, s: false, v: false, a: false, t: true },
    target: 't',
  },
  {
    id: 'as',
    label: 'a = 2s + 4, v = 5 at s = 0, find v at s = 10',
    mode: 'a-expr',
    expr: '2*s + 4',
    s0: 0,
    v0: 5,
    t0: 0,
    s: 10,
    v: null,
    a: null,
    t: null,
    use: { v0: true, s: true, v: false, a: false, t: false },
    target: 's',
  },
  {
    id: 'av',
    label: 'a = −0.05 v² (drag), how far until v = 8',
    mode: 'a-expr',
    expr: '-0.05*v^2',
    s0: 0,
    v0: 20,
    t0: 0,
    s: null,
    v: 8,
    a: null,
    t: null,
    use: { v0: true, s: false, v: true, a: false, t: false },
    target: 'v',
  },
  {
    id: 'spring',
    label: 'a = −4s (spring-like), when is s = 0',
    mode: 'a-expr',
    expr: '-4*s',
    s0: 0.5,
    v0: 0,
    t0: 0,
    s: 0,
    v: null,
    a: null,
    t: null,
    use: { v0: true, s: true, v: false, a: false, t: false },
    target: 's',
    tHorizon: 5,
  },
  {
    id: 'vt',
    label: 'v = 4t + 2, find s and a at t = 3',
    mode: 'v-of-t',
    expr: '4*t + 2',
    s0: 0,
    v0: 2,
    t0: 0,
    s: null,
    v: null,
    a: null,
    t: 3,
    use: { v0: true, s: false, v: false, a: false, t: true },
    target: 't',
  },
]

function Solver1DPanel() {
  const [presetId, setPresetId] = useState('brake')
  const [mode, setMode] = useState<Mode>('const-a')
  const [exprText, setExprText] = useState('4*t - 3')

  const [s0, setS0] = useState<number | null>(0)
  const [t0, setT0] = useState<number | null>(0)
  const [v0, setV0] = useState<number | null>(25)
  const [s, setS] = useState<number | null>(null)
  const [v, setV] = useState<number | null>(0)
  const [a, setA] = useState<number | null>(-4)
  const [t, setT] = useState<number | null>(null)

  const [useV0, setUseV0] = useState(true)
  const [useS, setUseS] = useState(false)
  const [useV, setUseV] = useState(true)
  const [useA, setUseA] = useState(true)
  const [useT, setUseT] = useState(false)

  const [target, setTarget] = useState<TargetKind>('t')
  const [tHorizon, setTHorizon] = useState(30)
  const [altIndex, setAltIndex] = useState(0)

  const parsed = useMemo(() => (mode === 'const-a' ? null : parseExpr(exprText)), [mode, exprText])

  const applyPreset = (id: string) => {
    const p = PRESETS.find((x) => x.id === id)
    if (!p) return
    setPresetId(id)
    setMode(p.mode)
    if (p.expr) setExprText(p.expr)
    setS0(p.s0)
    setT0(p.t0 ?? 0)
    setV0(p.v0)
    setS(p.s)
    setV(p.v)
    setA(p.a)
    setT(p.t)
    setUseV0(p.use.v0)
    setUseS(p.use.s)
    setUseV(p.use.v)
    setUseA(p.use.a)
    setUseT(p.use.t)
    if (p.target) setTarget(p.target)
    if (p.tHorizon) setTHorizon(p.tHorizon)
    setAltIndex(0)
  }

  const result: SolverResult = useMemo(() => {
    const origin = s0 ?? 0
    if (mode === 'const-a') {
      return solveConstA({
        s0: origin,
        v0: useV0 ? v0 : null,
        s: useS ? s : null,
        v: useV ? v : null,
        a: useA ? a : null,
        t: useT ? t : null,
      })
    }
    if (!parsed || !parsed.ok) return { ok: false, error: parsed?.error ?? 'enter an expression' }
    const tStart = t0 ?? 0
    if (v0 === null) return { ok: false, error: 'starting velocity v₀ is required' }
    const targetValue = target === 't' ? t : target === 's' ? s : v
    if (targetValue === null) return { ok: false, error: `enter the known ${target} you are solving at` }
    return solveVariable({
      kind: mode,
      expr: parsed.expr,
      t0: tStart,
      s0: origin,
      v0,
      target: { kind: target, value: targetValue },
      tHorizon,
    })
  }, [mode, s0, t0, v0, s, v, a, t, useV0, useS, useV, useA, useT, parsed, target, tHorizon])

  const points = result.ok ? [result.primary, ...result.alternatives] : []
  const shown = points[Math.min(altIndex, Math.max(0, points.length - 1))]
  const history = useMemo(() => {
    if (!result.ok || !shown) return { t: [0], s: [0], v: [0], a: [0] }
    if (mode === 'const-a') {
      const u = shown.v - shown.a * shown.t
      return constAHistory(s0 ?? 0, u, shown.a, shown.t)
    }
    return result.history
  }, [result, shown, mode, s0])
  const tFinal = history.t.length ? history.t[history.t.length - 1] : 0
  const tStartHist = history.t.length ? history.t[0] : 0
  const duration = Math.max(0, tFinal - tStartHist)

  const playback = usePlayback(Math.max(duration, 1e-6))
  useEffect(() => {
    playback.reset(Math.max(duration, 1e-6))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [duration, result.ok ? result.primary.t : 0, result.ok ? result.primary.s : 0, result.ok ? result.primary.v : 0])
  useEffect(() => {
    setAltIndex(0)
  }, [mode, s0, v0, s, v, a, t, useV0, useS, useV, useA, useT, exprText, target, tHorizon])

  const clock = tStartHist + playback.time
  const sNow = sampleAt(history.t, history.s, clock)
  const vNow = sampleAt(history.t, history.v, clock)
  const aNow = sampleAt(history.t, history.a, clock)

  const suvatCount = (useV0 ? 1 : 0) + (useS ? 1 : 0) + (useV ? 1 : 0) + (useA ? 1 : 0) + (useT ? 1 : 0)

  return (
    <div className="kin-grid kin-grid-solver">
      <div className="kin-controls panel">
        <div className="panel-header">
          <h2>Put in what you know</h2>
        </div>
        <div className="panel-body" style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <p className="hint">
            Check what you know. Constant <Eq tex="a" /> needs any 3 of <Eq tex="\Delta s, v_0, v, a, t" />. If{' '}
            <Eq tex="a" /> or <Eq tex="v" /> is a function of <Eq tex="t,s,v" />, give the function, a start, and one
            target.
          </p>

          <SelectField
            label="Load example"
            value={presetId}
            onChange={applyPreset}
            options={[{ value: '_custom', label: 'Custom' }, ...PRESETS.map((p) => ({ value: p.id, label: p.label }))]}
          />

          <Segmented
            value={mode}
            onChange={(m: Mode) => {
              setMode(m)
              setPresetId('_custom')
              if (m === 'a-expr' && (exprText === '4*t + 2' || !exprText)) setExprText('4*t - 3')
              if (m === 'v-of-t') setExprText('4*t + 2')
              if (m === 'v-of-s') setExprText('2*s + 1')
              if (m !== 'const-a') {
                setTarget(m === 'v-of-s' ? 's' : 't')
              }
            }}
            options={[
              { value: 'const-a', label: 'Constant a' },
              { value: 'a-expr', label: 'a = f(·)' },
              { value: 'v-of-t', label: 'v(t)' },
              { value: 'v-of-s', label: 'v(s)' },
            ]}
          />

          {mode !== 'const-a' && (
            <div className="field-row">
              <label className="field-label" htmlFor="kin-expr">
                <span>{mode === 'a-expr' ? 'Acceleration a =' : 'Velocity v ='}</span>
              </label>
              <input
                id="kin-expr"
                className="num-input expr-input"
                value={exprText}
                spellCheck={false}
                placeholder={mode === 'a-expr' ? 'e.g. 4*t - 3  or  2*s + 4  or  -0.05*v^2' : 'e.g. 4*t + 2'}
                onChange={(e) => {
                  setExprText(e.target.value)
                  setPresetId('_custom')
                }}
              />
              {parsed && parsed.ok ? (
                <p className="hint" style={{ marginTop: 6 }}>
                  <Eq tex={(mode === 'a-expr' ? 'a = ' : 'v = ') + parsed.expr.latex} />
                  {parsed.expr.vars.size > 0 && (
                    <span>
                      {' '}
                      · uses {Array.from(parsed.expr.vars).join(', ')}
                    </span>
                  )}
                </p>
              ) : (
                <p className="hint solver-error">{parsed && !parsed.ok ? parsed.error : ''}</p>
              )}
              <p className="hint">
                Variables: t, s, v. Functions: sin, cos, sqrt, exp, ln, abs. Constants: pi, e, g (=9.81). Implied
                multiplication is fine (2t, 2(s+1)).
              </p>
            </div>
          )}

          <div className="known-block">
            <div className="known-heading">{mode === 'const-a' ? 'Knowns' : 'Starting state'}</div>
            {mode !== 'const-a' && <PlainRow symbol="t_0" name="start time" unit="s" value={t0} onChange={setT0} />}
            <PlainRow symbol="s_0" name="start position" unit="m" value={s0} onChange={setS0} />
            {mode === 'const-a' ? (
              <KnownRow symbol="v_0" name="initial velocity" unit="m/s" known={useV0} value={v0} onKnown={setUseV0} onValue={setV0} />
            ) : (
              <PlainRow symbol="v_0" name="start velocity" unit="m/s" value={v0} onChange={setV0} />
            )}
          </div>

          {mode === 'const-a' ? (
            <div className="known-block">
              <div className="known-heading">
                At time t · {suvatCount} of 5 filled {suvatCount >= 3 ? '✓' : `(need ${3 - suvatCount} more)`}
              </div>
              <KnownRow symbol="t" name="time" unit="s" known={useT} value={t} onKnown={setUseT} onValue={setT} />
              <KnownRow symbol="s" name="position" unit="m" known={useS} value={s} onKnown={setUseS} onValue={setS} />
              <KnownRow symbol="v" name="velocity" unit="m/s" known={useV} value={v} onKnown={setUseV} onValue={setV} />
              <KnownRow symbol="a" name="acceleration" unit="m/s^2" known={useA} value={a} onKnown={setUseA} onValue={setA} />
              <p className="hint">
                Check a box to mark that value known. Displacement is <Eq tex="\Delta s = s - s_0" />.
                {suvatCount > 3 ? ' Extra knowns are checked for consistency.' : ''}
              </p>
            </div>
          ) : (
            <div className="known-block">
              <div className="known-heading">Also know (the target)</div>
              <div className="target-picks">
                {(['t', 's', 'v'] as TargetKind[]).map((k) => (
                  <button key={k} type="button" className={target === k ? 'active' : ''} onClick={() => setTarget(k)}>
                    {k}
                  </button>
                ))}
              </div>
              {target === 't' && <PlainRow symbol="t" name="time" unit="s" value={t} onChange={setT} />}
              {target === 's' && <PlainRow symbol="s" name="position" unit="m" value={s} onChange={setS} />}
              {target === 'v' && <PlainRow symbol="v" name="velocity" unit="m/s" value={v} onChange={setV} />}
              {target !== 't' && (
                <PlainRow symbol="t_{\\max}" name="search horizon" unit="s" value={tHorizon} onChange={(n) => n !== null && setTHorizon(Math.max(0.1, n))} />
              )}
              <p className="hint">
                The solver integrates until it hits this value (and will search backward from t₀ if going forward never
                gets there).
              </p>
            </div>
          )}
        </div>
      </div>

      <div className="kin-results">
        <div className="panel">
          <div className="panel-header">
            <h2>Governing relations</h2>
          </div>
          <div className="panel-body">
            {mode === 'const-a' ? (
              <EqList
                items={
                  result.ok
                    ? result.method
                    : ['v = v_0 + a t', 's = s_0 + v_0 t + \\tfrac12 a t^2', 'v^2 = v_0^2 + 2 a (s - s_0)']
                }
              />
            ) : result.ok ? (
              <EqList items={result.method} />
            ) : (
              <EqList
                items={
                  mode === 'a-expr'
                    ? ['a = \\dfrac{dv}{dt}', 'a\\,ds = v\\,dv', 'v = \\dfrac{ds}{dt}']
                    : mode === 'v-of-t'
                      ? ['a = \\dfrac{dv}{dt}', 's = s_0 + \\int v\\,dt']
                      : ['a = v\\,\\dfrac{dv}{ds}', 'dt = \\dfrac{ds}{v}']
                }
              />
            )}
          </div>
        </div>

        {result.ok && shown ? (
          <>
            <div className="summary-grid">
              <ResultCard label="Time t" value={shown.t} unit="s" computed={result.computed.t} />
              <ResultCard label="Position s" value={shown.s} unit="m" computed={result.computed.s} />
              <ResultCard label="Velocity v" value={shown.v} unit="m/s" computed={result.computed.v} />
              <ResultCard label="Acceleration a" value={shown.a} unit="m/s²" computed={result.computed.a} />
            </div>
            {result.notes.map((n) => (
              <p key={n} className="hint">
                {n}
              </p>
            ))}
            {points.length > 1 && (
              <div className="alt-roots">
                {points.map((p, i) => (
                  <button key={`${p.t}-${p.v}`} type="button" className={i === altIndex ? 'active' : ''} onClick={() => setAltIndex(i)}>
                    root {i + 1}: t = {formatNumber(p.t, 4)} s, v = {formatNumber(p.v, 4)} m/s
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

        {result.ok && (
          <>
            <div className="summary-grid">
              <div className="stat-card">
                <div className="label">Position now</div>
                <div className="val">
                  {formatNumber(sNow, 3)}
                  <span className="unit">m</span>
                </div>
              </div>
              <div className="stat-card">
                <div className="label">Velocity now</div>
                <div className="val">
                  {formatNumber(vNow, 3)}
                  <span className="unit">m/s</span>
                </div>
              </div>
              <div className="stat-card">
                <div className="label">Acceleration now</div>
                <div className="val">
                  {formatNumber(aNow, 3)}
                  <span className="unit">m/s²</span>
                </div>
              </div>
            </div>
            <PlaybackBar playback={playback} disabled={duration < 1e-9} />
            <div className="panel">
              <div className="panel-header">
                <h2>s-t, v-t, a-t</h2>
              </div>
              <div className="panel-body" style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                <TimeSeriesChart t={history.t} series={[{ label: 's (m)', color: '#29d3f5', data: history.s }]} currentTime={clock} height={130} />
                <TimeSeriesChart t={history.t} series={[{ label: 'v (m/s)', color: '#59d67f', data: history.v }]} currentTime={clock} height={130} />
                <TimeSeriesChart t={history.t} series={[{ label: 'a (m/s²)', color: '#f5a524', data: history.a }]} currentTime={clock} height={130} />
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  )
}

export function SolverPanel() {
  const [kind, setKind] = useState<'2d' | '1d'>('2d')
  return (
    <div className="solver-kind-wrap">
      <div className="solver-kind-bar">
        <Segmented
          value={kind}
          onChange={setKind}
          options={[
            { value: '2d', label: '2D projectile' },
            { value: '1d', label: '1D particle' },
          ]}
        />
        <span className="hint">{kind === '2d' ? 'Pass through a point: Δr, g, and θ → v₀' : 'Rectilinear: any 3 of Δs, v₀, v, a, t'}</span>
      </div>
      {kind === '2d' ? <ProjectileSolverPanel /> : <Solver1DPanel />}
    </div>
  )
}

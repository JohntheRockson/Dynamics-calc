import { useMemo, useState } from 'react'
import { NumberField, Segmented, SelectField } from '../components/fields'
import { formatNumber } from '../format'
import { usePlayback } from '../hooks/usePlayback'
import { PlaybackBar } from '../components/PlaybackBar'
import { TimeSeriesChart } from '../components/Charts/TimeSeriesChart'
import { airDensity, GRAVITY_PRESETS, SHAPE_PRESETS } from './constants'
import { Eq, EqList } from './Eq'
import { rectilinearConstantAccel, rectilinearWithDrag, terminalSpeed, type DragParams } from './physics'
import { Scene2D } from './Scene2D'

type Mode = 'const-accel' | 'drag'

export function RectilinearPanel() {
  const [mode, setMode] = useState<Mode>('const-accel')
  const [s0, setS0] = useState(0)
  const [v0, setV0] = useState(0)
  const [a, setA] = useState(2)
  const [tFinal, setTFinal] = useState(10)

  const [gravityKey, setGravityKey] = useState('earth')
  const [gCustom, setGCustom] = useState(9.81)
  const g = gravityKey === 'custom' ? gCustom : GRAVITY_PRESETS.find((p) => p.key === gravityKey)!.g

  const [shapeKey, setShapeKey] = useState('sphere')
  const [cdCustom, setCdCustom] = useState(0.47)
  const cd = shapeKey === 'custom' ? cdCustom : SHAPE_PRESETS.find((p) => p.key === shapeKey)!.cd
  const [mass, setMass] = useState(1)
  const [area, setArea] = useState(0.05)
  const [altitude, setAltitude] = useState(0)
  const [tempC, setTempC] = useState(15)
  const rho = airDensity(altitude, tempC)

  const drag: DragParams = { mass, cd, area, airDensity: rho }
  const vTerm = terminalSpeed(g, drag)

  const result = useMemo(() => {
    if (mode === 'const-accel') return rectilinearConstantAccel(s0, v0, a, tFinal)
    return rectilinearWithDrag(s0, v0, g, drag, tFinal)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode, s0, v0, a, tFinal, g, mass, cd, area, rho])

  const playback = usePlayback(tFinal)
  const idx = useMemo(() => {
    const dt = tFinal / (result.t.length - 1 || 1)
    return Math.min(result.t.length - 1, Math.max(0, Math.round(playback.time / dt)))
  }, [playback.time, tFinal, result.t.length])

  const sNow = result.s[idx]
  const vNow = result.v[idx]
  const aNow = result.a[idx]

  return (
    <div className="kin-grid">
      <div className="kin-controls panel">
        <div className="panel-header">
          <h2>Rectilinear motion (1D)</h2>
        </div>
        <div className="panel-body" style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <Segmented
            value={mode}
            onChange={(v: Mode) => setMode(v)}
            options={[
              { value: 'const-accel', label: 'Constant acceleration' },
              { value: 'drag', label: 'Gravity + air drag' },
            ]}
          />

          <NumberField label="Initial position s₀" value={s0} onChange={setS0} min={-50} max={50} step={0.5} unit="m" precision={1} />
          <NumberField label="Initial velocity v₀" value={v0} onChange={setV0} min={-50} max={50} step={0.5} unit="m/s" precision={1} />
          <NumberField label="Duration" value={tFinal} onChange={setTFinal} min={1} max={60} step={1} unit="s" precision={0} />

          {mode === 'const-accel' ? (
            <NumberField label="Acceleration a" value={a} onChange={setA} min={-20} max={20} step={0.1} unit="m/s²" precision={2} />
          ) : (
            <>
              <div className="field-row">
                <SelectField label="Gravity" value={gravityKey} onChange={setGravityKey} options={GRAVITY_PRESETS.map((p) => ({ value: p.key, label: p.label }))} />
              </div>
              {gravityKey === 'custom' && <NumberField label="g" value={gCustom} onChange={setGCustom} min={0} max={30} step={0.01} unit="m/s²" precision={2} />}

              <div className="field-row">
                <SelectField label="Shape (sets Cd)" value={shapeKey} onChange={setShapeKey} options={SHAPE_PRESETS.map((p) => ({ value: p.key, label: p.label }))} />
              </div>
              {shapeKey === 'custom' && <NumberField label="Drag coefficient Cd" value={cdCustom} onChange={setCdCustom} min={0} max={2} step={0.01} precision={2} />}

              <NumberField label="Mass" value={mass} onChange={setMass} min={0.01} max={200} unit="kg" precision={2} logScale />
              <NumberField label="Cross-sectional area A" value={area} onChange={setArea} min={0.001} max={5} unit="m²" precision={3} logScale />
              <NumberField label="Altitude" value={altitude} onChange={setAltitude} min={0} max={9000} step={100} unit="m" precision={0} />
              <NumberField label="Ground temperature" value={tempC} onChange={setTempC} min={-40} max={45} step={1} unit="°C" precision={0} />
              <p className="hint">
                Air density ρ = {formatNumber(rho, 4)} kg/m³ (standard-atmosphere model). Terminal speed = {Number.isFinite(vTerm) ? formatNumber(vTerm, 2) : '∞'} m/s.
              </p>
            </>
          )}
        </div>
      </div>

      <div className="kin-results">
        <div className="panel">
          <div className="panel-header">
            <h2>Governing equations</h2>
          </div>
          <div className="panel-body">
            {mode === 'const-accel' ? (
              <EqList
                items={[
                  `s(t) = s_0 + v_0 t + \\tfrac{1}{2} a t^2 = ${formatNumber(s0, 2)} + ${formatNumber(v0, 2)}t + \\tfrac{1}{2}(${formatNumber(a, 2)})t^2`,
                  `v(t) = v_0 + a t = ${formatNumber(v0, 2)} + ${formatNumber(a, 2)}t`,
                  `v^2 = v_0^2 + 2a(s-s_0)`,
                ]}
              />
            ) : (
              <>
                <EqList items={[`m\\dot v = -mg - \\operatorname{sign}(v)\\,\\tfrac12 \\rho C_d A\\, v^2`, `v_{\\text{term}} = \\sqrt{\\dfrac{2mg}{\\rho C_d A}} = ${Number.isFinite(vTerm) ? formatNumber(vTerm, 3) : '\\infty'}\\ \\text{m/s}`]} />
                <p className="hint">
                  No elementary closed form exists for general v₀, so this is integrated numerically (RK4). For a drop from rest (v₀=0) the
                  exact solution is <Eq tex="v(t) = -v_{\text{term}}\tanh(gt/v_{\text{term}})" />.
                </p>
              </>
            )}
          </div>
        </div>

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

        <div className="panel">
          <div className="panel-header">
            <h2>Position track</h2>
          </div>
          <div className="panel-body">
            <Scene2D
              height={140}
              yLabel=""
              xLabel="s (m)"
              bodies={[{ label: 'particle', color: '#29d3f5', path: result.s.map((s) => ({ x: s, y: 0 })), currentIndex: idx }]}
            />
          </div>
        </div>
        <PlaybackBar playback={playback} disabled={false} />

        <div className="panel">
          <div className="panel-header">
            <h2>s-t, v-t, a-t</h2>
          </div>
          <div className="panel-body" style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            <TimeSeriesChart t={result.t} series={[{ label: 's (m)', color: '#29d3f5', data: result.s }]} currentTime={playback.time} height={130} />
            <TimeSeriesChart t={result.t} series={[{ label: 'v (m/s)', color: '#59d67f', data: result.v }]} currentTime={playback.time} height={130} />
            <TimeSeriesChart t={result.t} series={[{ label: 'a (m/s²)', color: '#f5a524', data: result.a }]} currentTime={playback.time} height={130} />
          </div>
        </div>
      </div>
    </div>
  )
}

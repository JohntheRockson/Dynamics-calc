import { useMemo, useState } from 'react'
import { NumberField, Segmented, SelectField, ToggleField } from '../components/fields'
import { PlaybackBar } from '../components/PlaybackBar'
import { TimeSeriesChart } from '../components/Charts/TimeSeriesChart'
import { formatNumber } from '../format'
import { usePlayback } from '../hooks/usePlayback'
import { airDensity, GRAVITY_PRESETS, SHAPE_PRESETS } from './constants'
import { CurvilinearSolverPanel } from './CurvilinearSolverPanel'
import { Eq, EqList } from './Eq'
import { noDragTimeOfFlight, normalTangentialAt, simulateProjectile, type DragParams } from './physics'
import { Scene2D } from './Scene2D'

export function CurvilinearPanel() {
  const [mode, setMode] = useState<'solver' | 'projectile'>('solver')
  return (
    <div>
      <div className="kin-mode-bar">
        <Segmented
          value={mode}
          onChange={setMode}
          options={[
            { value: 'solver', label: 'Kinematics solver' },
            { value: 'projectile', label: 'Projectile trajectory' },
          ]}
        />
      </div>
      {mode === 'solver' ? <CurvilinearSolverPanel /> : <ProjectilePanel />}
    </div>
  )
}

function ProjectilePanel() {
  const [x0, setX0] = useState(0)
  const [y0, setY0] = useState(1)
  const [speed0, setSpeed0] = useState(25)
  const [angleDeg, setAngleDeg] = useState(40)
  const [groundY, setGroundY] = useState(0)

  const [gravityKey, setGravityKey] = useState('earth')
  const [gCustom, setGCustom] = useState(9.81)
  const g = gravityKey === 'custom' ? gCustom : GRAVITY_PRESETS.find((p) => p.key === gravityKey)!.g

  const [dragOn, setDragOn] = useState(false)
  const [shapeKey, setShapeKey] = useState('sphere')
  const [cdCustom, setCdCustom] = useState(0.47)
  const cd = shapeKey === 'custom' ? cdCustom : SHAPE_PRESETS.find((p) => p.key === shapeKey)!.cd
  const [mass, setMass] = useState(0.145) // baseball-ish default
  const [area, setArea] = useState(0.0042)
  const [altitude, setAltitude] = useState(0)
  const [tempC, setTempC] = useState(15)
  const rho = airDensity(altitude, tempC)

  const drag: DragParams | null = dragOn ? { mass, cd, area, airDensity: rho } : null

  const noDragTFlight = useMemo(
    () => noDragTimeOfFlight({ x0, y0, speed0, angleDeg, g, groundY, drag: null }),
    [x0, y0, speed0, angleDeg, g, groundY],
  )
  const tFinal = useMemo(() => {
    const base = noDragTFlight ?? Math.max(2, (2 * speed0) / g)
    return Math.max(0.5, base * (dragOn ? 1.6 : 1.05))
  }, [noDragTFlight, speed0, g, dragOn])

  const result = useMemo(
    () => simulateProjectile({ x0, y0, speed0, angleDeg, g, groundY, drag }, tFinal),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [x0, y0, speed0, angleDeg, g, groundY, tFinal, mass, cd, area, rho, dragOn],
  )

  const playback = usePlayback(tFinal)
  const idx = useMemo(() => {
    const dt = tFinal / (result.t.length - 1 || 1)
    return Math.min(result.landingIndex, Math.max(0, Math.round(playback.time / dt)))
  }, [playback.time, tFinal, result])

  const nt = normalTangentialAt(result.vx[idx], result.vy[idx], result.ax[idx], result.ay[idx])
  const theta = (angleDeg * Math.PI) / 180
  const maxHeightNoDrag = y0 + (speed0 * Math.sin(theta)) ** 2 / (2 * g)
  const rangeNoDrag = noDragTFlight !== null ? x0 + speed0 * Math.cos(theta) * noDragTFlight : null

  const landedT = result.t[result.landingIndex]
  const landedX = result.x[result.landingIndex]
  const landedSpeed = Math.hypot(result.vx[result.landingIndex], result.vy[result.landingIndex])

  return (
    <div className="kin-grid">
      <div className="kin-controls panel">
        <div className="panel-header">
          <h2>Curvilinear motion: projectile</h2>
        </div>
        <div className="panel-body" style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <NumberField label="Launch speed v₀" value={speed0} onChange={setSpeed0} min={1} max={100} step={0.5} unit="m/s" precision={1} />
          <NumberField label="Launch angle θ" value={angleDeg} onChange={setAngleDeg} min={-90} max={90} step={1} unit="deg" precision={0} />
          <NumberField label="Initial height y₀" value={y0} onChange={setY0} min={0} max={200} step={0.5} unit="m" precision={1} />
          <NumberField label="Ground level" value={groundY} onChange={setGroundY} min={-50} max={y0} step={0.5} unit="m" precision={1} />
          <NumberField label="Initial x₀" value={x0} onChange={setX0} min={-50} max={50} step={1} unit="m" precision={0} />

          <div className="field-row">
            <SelectField label="Gravity" value={gravityKey} onChange={setGravityKey} options={GRAVITY_PRESETS.map((p) => ({ value: p.key, label: p.label }))} />
          </div>
          {gravityKey === 'custom' && <NumberField label="g" value={gCustom} onChange={setGCustom} min={0} max={30} step={0.01} unit="m/s²" precision={2} />}

          <ToggleField label="Include air resistance" checked={dragOn} onChange={setDragOn} />
          {dragOn && (
            <>
              <div className="field-row">
                <SelectField label="Shape (sets Cd)" value={shapeKey} onChange={setShapeKey} options={SHAPE_PRESETS.map((p) => ({ value: p.key, label: p.label }))} />
              </div>
              {shapeKey === 'custom' && <NumberField label="Drag coefficient Cd" value={cdCustom} onChange={setCdCustom} min={0} max={2} step={0.01} precision={2} />}
              <NumberField label="Mass" value={mass} onChange={setMass} min={0.001} max={200} unit="kg" precision={3} logScale />
              <NumberField label="Cross-sectional area A" value={area} onChange={setArea} min={0.0001} max={5} unit="m²" precision={4} logScale />
              <NumberField label="Altitude" value={altitude} onChange={setAltitude} min={0} max={9000} step={100} unit="m" precision={0} />
              <NumberField label="Ground temperature" value={tempC} onChange={setTempC} min={-40} max={45} step={1} unit="°C" precision={0} />
              <p className="hint">Air density ρ = {formatNumber(rho, 4)} kg/m³.</p>
            </>
          )}
        </div>
      </div>

      <div className="kin-results">
        <div className="panel">
          <div className="panel-header">
            <h2>Parametric equations</h2>
          </div>
          <div className="panel-body">
            {!dragOn ? (
              <EqList
                items={[
                  `x(t) = x_0 + v_0\\cos\\theta\\, t = ${formatNumber(x0, 1)} + ${formatNumber(speed0 * Math.cos(theta), 2)}\\,t`,
                  `y(t) = y_0 + v_0\\sin\\theta\\, t - \\tfrac12 g t^2 = ${formatNumber(y0, 1)} + ${formatNumber(speed0 * Math.sin(theta), 2)}\\,t - \\tfrac12(${formatNumber(g, 2)})t^2`,
                ]}
              />
            ) : (
              <>
                <EqList items={[`m\\dot v_x = -\\tfrac12 \\rho C_d A\\,|\\vec v|\\,v_x`, `m\\dot v_y = -mg - \\tfrac12 \\rho C_d A\\,|\\vec v|\\,v_y`]} />
                <p className="hint">Quadratic drag opposing the velocity vector has no elementary closed form here, so this is integrated numerically (RK4).</p>
              </>
            )}
          </div>
        </div>

        <div className="summary-grid">
          <div className="stat-card">
            <div className="label">Time of flight</div>
            <div className="val">
              {formatNumber(landedT, 2)}
              <span className="unit">s</span>
            </div>
          </div>
          <div className="stat-card">
            <div className="label">Range</div>
            <div className="val">
              {formatNumber(landedX - x0, 2)}
              <span className="unit">m</span>
            </div>
          </div>
          <div className="stat-card">
            <div className="label">Impact speed</div>
            <div className="val">
              {formatNumber(landedSpeed, 2)}
              <span className="unit">m/s</span>
            </div>
          </div>
          {!dragOn && (
            <>
              <div className="stat-card">
                <div className="label">Max height (vacuum)</div>
                <div className="val">
                  {formatNumber(maxHeightNoDrag, 2)}
                  <span className="unit">m</span>
                </div>
              </div>
              <div className="stat-card">
                <div className="label">Range (vacuum, exact)</div>
                <div className="val">{rangeNoDrag !== null ? `${formatNumber(rangeNoDrag - x0, 2)}` : '-'}</div>
              </div>
            </>
          )}
        </div>

        <div className="panel">
          <div className="panel-header">
            <h2>Trajectory</h2>
          </div>
          <div className="panel-body">
            <Scene2D
              height={280}
              aspectEqual
              groundY={groundY}
              bodies={[
                {
                  label: 'projectile',
                  color: '#29d3f5',
                  path: result.x.map((x, i) => ({ x, y: result.y[i] })),
                  currentIndex: idx,
                  velocityVector: { vx: result.vx[idx], vy: result.vy[idx] },
                },
              ]}
            />
          </div>
        </div>
        <PlaybackBar playback={playback} disabled={false} />

        <div className="panel">
          <div className="panel-header">
            <h2>Normal-tangential components (at scrub time)</h2>
          </div>
          <div className="panel-body">
            <div className="summary-grid">
              <div className="stat-card">
                <div className="label">Speed</div>
                <div className="val">
                  {formatNumber(nt.speed, 2)}
                  <span className="unit">m/s</span>
                </div>
              </div>
              <div className="stat-card">
                <div className="label">Tangential accel aₜ</div>
                <div className="val">
                  {formatNumber(nt.aTangential, 2)}
                  <span className="unit">m/s²</span>
                </div>
              </div>
              <div className="stat-card">
                <div className="label">Normal accel aₙ</div>
                <div className="val">
                  {formatNumber(nt.aNormal, 2)}
                  <span className="unit">m/s²</span>
                </div>
              </div>
              <div className="stat-card">
                <div className="label">Radius of curvature ρ</div>
                <div className="val">{Number.isFinite(nt.radiusOfCurvature) ? `${formatNumber(nt.radiusOfCurvature, 1)}` : '∞'}</div>
              </div>
            </div>
            <p className="hint" style={{ marginTop: 10 }}>
              <Eq tex="a_t = \dfrac{\vec a \cdot \vec v}{|\vec v|}" />
              {'  '}
              <Eq tex="a_n = \sqrt{|\vec a|^2 - a_t^2}" />
              {'  '}
              <Eq tex="\rho = \dfrac{v^2}{a_n}" />
            </p>
          </div>
        </div>

        <div className="panel">
          <div className="panel-header">
            <h2>x, y, vx, vy vs t</h2>
          </div>
          <div className="panel-body" style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            <TimeSeriesChart
              t={result.t}
              series={[
                { label: 'x (m)', color: '#fb6a6a', data: result.x },
                { label: 'y (m)', color: '#5aa8ff', data: result.y },
              ]}
              currentTime={playback.time}
              height={140}
            />
            <TimeSeriesChart
              t={result.t}
              series={[
                { label: 'vx (m/s)', color: '#fb6a6a', data: result.vx },
                { label: 'vy (m/s)', color: '#5aa8ff', data: result.vy },
              ]}
              currentTime={playback.time}
              height={140}
            />
          </div>
        </div>
      </div>
    </div>
  )
}

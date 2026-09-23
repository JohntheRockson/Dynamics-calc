import { useMemo, useState } from 'react'
import { NumberField, ToggleField, Vec2Field } from '../components/fields'
import { PlaybackBar } from '../components/PlaybackBar'
import { TimeSeriesChart } from '../components/Charts/TimeSeriesChart'
import { formatNumber } from '../format'
import { usePlayback } from '../hooks/usePlayback'
import { Eq, EqList } from './Eq'
import { sampleAt, simulateRelativeMotion, type ParticleIC } from './physics'
import { Scene2D } from './Scene2D'

function ParticleInputs({
  label,
  color,
  pos,
  setPos,
  vel,
  setVel,
  accel,
  setAccel,
  showAccel,
}: {
  label: string
  color: string
  pos: [number, number]
  setPos: (v: [number, number]) => void
  vel: [number, number]
  setVel: (v: [number, number]) => void
  accel: [number, number]
  setAccel: (v: [number, number]) => void
  showAccel: boolean
}) {
  return (
    <div style={{ borderLeft: `3px solid ${color}`, paddingLeft: 10, display: 'flex', flexDirection: 'column', gap: 8 }}>
      <strong style={{ fontSize: 12.5 }}>{label}</strong>
      <Vec2Field label="Position (x, y)" value={pos} onChange={setPos} min={-100} max={100} step={0.5} unit="m" />
      <Vec2Field label="Velocity (vx, vy)" value={vel} onChange={setVel} min={-50} max={50} step={0.5} unit="m/s" />
      {showAccel && <Vec2Field label="Acceleration (ax, ay)" value={accel} onChange={setAccel} min={-20} max={20} step={0.1} unit="m/s²" />}
    </div>
  )
}

export function RelativeMotionPanel() {
  const [aPos, setAPos] = useState<[number, number]>([0, 0])
  const [aVel, setAVel] = useState<[number, number]>([8, 2])
  const [aAcc, setAAcc] = useState<[number, number]>([0, 0])

  const [bPos, setBPos] = useState<[number, number]>([40, 30])
  const [bVel, setBVel] = useState<[number, number]>([-3, -4])
  const [bAcc, setBAcc] = useState<[number, number]>([0, 0])

  const [showAccel, setShowAccel] = useState(false)
  const [tFinal, setTFinal] = useState(12)

  const a: ParticleIC = { x0: aPos[0], y0: aPos[1], vx0: aVel[0], vy0: aVel[1], ax: aAcc[0], ay: aAcc[1] }
  const b: ParticleIC = { x0: bPos[0], y0: bPos[1], vx0: bVel[0], vy0: bVel[1], ax: bAcc[0], ay: bAcc[1] }

  const result = useMemo(
    () => simulateRelativeMotion(a, b, tFinal),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [aPos, aVel, aAcc, bPos, bVel, bAcc, tFinal],
  )

  const playback = usePlayback(tFinal)
  const idx = useMemo(() => {
    const dt = tFinal / (result.t.length - 1 || 1)
    return Math.min(result.t.length - 1, Math.max(0, Math.round(playback.time / dt)))
  }, [playback.time, tFinal, result.t.length])

  const relX = result.relPos.x[idx]
  const relY = result.relPos.y[idx]
  const relDist = Math.hypot(relX, relY)
  const aNow2 = { x: result.aPos.x[idx], y: result.aPos.y[idx] }
  const bNow2 = { x: result.bPos.x[idx], y: result.bPos.y[idx] }
  const relSpeedNow = sampleAt(result.t, result.relSpeed, playback.time)
  const rel0 = Math.hypot(a.x0 - b.x0, a.y0 - b.y0)
  const noAccel = a.ax === 0 && a.ay === 0 && b.ax === 0 && b.ay === 0

  return (
    <div className="kin-grid">
      <div className="kin-controls panel">
        <div className="panel-header">
          <h2>Relative motion (translating axes)</h2>
        </div>
        <div className="panel-body" style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          <ToggleField label="Include constant acceleration per particle" checked={showAccel} onChange={setShowAccel} />
          <ParticleInputs label="Particle A" color="#29d3f5" pos={aPos} setPos={setAPos} vel={aVel} setVel={setAVel} accel={aAcc} setAccel={setAAcc} showAccel={showAccel} />
          <ParticleInputs label="Particle B" color="#f5a524" pos={bPos} setPos={setBPos} vel={bVel} setVel={setBVel} accel={bAcc} setAccel={setBAcc} showAccel={showAccel} />
          <NumberField label="Duration" value={tFinal} onChange={setTFinal} min={1} max={60} step={1} unit="s" precision={0} />
        </div>
      </div>

      <div className="kin-results">
        <div className="panel">
          <div className="panel-header">
            <h2>Vector equation</h2>
          </div>
          <div className="panel-body">
            <EqList items={[`\\vec r_{A/B} = \\vec r_A - \\vec r_B`, `\\vec v_A = \\vec v_B + \\vec v_{A/B}`, `\\vec a_A = \\vec a_B + \\vec a_{A/B}`]} />
            {noAccel && (
              <p className="hint">
                Constant velocities: <Eq tex="\vec r_{A/B}(t) = \vec r_{A/B}(0) + \vec v_{A/B}\,t" />, a straight line in the relative frame.
              </p>
            )}
          </div>
        </div>

        <div className="summary-grid">
          <div className="stat-card">
            <div className="label">|r_A/B| now</div>
            <div className="val">
              {formatNumber(relDist, 2)}
              <span className="unit">m</span>
            </div>
          </div>
          <div className="stat-card">
            <div className="label">|r_A/B| at t=0</div>
            <div className="val">
              {formatNumber(rel0, 2)}
              <span className="unit">m</span>
            </div>
          </div>
          <div className="stat-card">
            <div className="label">Relative speed now</div>
            <div className="val">
              {formatNumber(relSpeedNow, 2)}
              <span className="unit">m/s</span>
            </div>
          </div>
          <div className="stat-card">
            <div className="label">Closest approach</div>
            <div className="val" style={{ fontSize: 15 }}>
              {formatNumber(result.closestApproach.distance, 2)} m @ t={formatNumber(result.closestApproach.t, 2)}s
            </div>
          </div>
        </div>

        <div className="panel">
          <div className="panel-header">
            <h2>Trajectories &amp; relative-position vector</h2>
          </div>
          <div className="panel-body">
            <Scene2D
              height={280}
              aspectEqual
              bodies={[
                { label: 'A', color: '#29d3f5', path: result.aPos.x.map((x, i) => ({ x, y: result.aPos.y[i] })), currentIndex: idx },
                { label: 'B', color: '#f5a524', path: result.bPos.x.map((x, i) => ({ x, y: result.bPos.y[i] })), currentIndex: idx },
                {
                  label: 'r_A/B (B → A, now)',
                  color: '#a78bfa',
                  dashed: true,
                  path: [bNow2, aNow2],
                  currentIndex: 1,
                },
              ]}
            />
          </div>
        </div>
        <PlaybackBar playback={playback} disabled={false} />

        <div className="panel">
          <div className="panel-header">
            <h2>|r_A/B| and relative speed vs t</h2>
          </div>
          <div className="panel-body" style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            <TimeSeriesChart
              t={result.t}
              series={[{ label: '|r_A/B| (m)', color: '#a78bfa', data: result.relPos.x.map((x, i) => Math.hypot(x, result.relPos.y[i])) }]}
              currentTime={playback.time}
              height={140}
            />
            <TimeSeriesChart t={result.t} series={[{ label: 'relative speed (m/s)', color: '#59d67f', data: result.relSpeed }]} currentTime={playback.time} height={140} />
          </div>
        </div>
      </div>
    </div>
  )
}

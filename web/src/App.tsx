import { lazy, Suspense, useEffect, useRef, useState } from 'react'
import { ApiError, fetchScenarios, runSimulation } from './api'
import { ErrorBanner, WarningBanners } from './components/Banners'
import { ChartsPanel } from './components/Charts/ChartsPanel'
import { ControlPanel } from './components/ControlPanel'
import { PlaybackBar } from './components/PlaybackBar'
import { SummaryCards } from './components/SummaryCards'
import { defaultSimRequest } from './defaultRequest'
import { usePlayback } from './hooks/usePlayback'
import type { ScenarioInfo, SimLog, SimRequest } from './types'

// three.js is the bulk of the bundle; split it into its own chunk so the
// control panel and charts are interactive before it finishes downloading.
const ViewerCard = lazy(() => import('./components/Viewer3D/ViewerCard').then((m) => ({ default: m.ViewerCard })))

export default function App() {
  const [req, setReqState] = useState<SimRequest>(defaultSimRequest)
  const [scenarios, setScenarios] = useState<ScenarioInfo[]>([])
  const [log, setLog] = useState<SimLog | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const playback = usePlayback(1)
  const abortRef = useRef<AbortController | null>(null)

  const setReq = (updater: (prev: SimRequest) => SimRequest) => setReqState(updater)

  useEffect(() => {
    fetchScenarios()
      .then(setScenarios)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
  }, [])

  const run = async () => {
    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller
    setLoading(true)
    setError(null)
    try {
      const result = await runSimulation(req, controller.signal)
      setLog(result)
      playback.reset(result.resolved.t_final)
      playback.play()
    } catch (e) {
      if (e instanceof ApiError) setError(e.message)
      else if (e instanceof DOMException && e.name === 'AbortError') {
        /* superseded by a newer run */
      } else setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  const reset = () => {
    setReqState(defaultSimRequest())
    setLog(null)
    setError(null)
    playback.reset(1)
  }

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="app-title">
          <span className="logo">&#128752;</span>
          <h1>SimLab</h1>
          <span className="subtitle">Attitude Control Console &middot; Rust engine</span>
        </div>
        <a href="https://github.com/JohntheRockson/Dynamics-calc" target="_blank" rel="noreferrer" className="hint" style={{ textDecoration: 'none' }}>
          6DOF rigid-body dynamics &middot; PID / LQR &middot; MEKF / Mahony
        </a>
      </header>

      <div className="app-body">
        <ControlPanel req={req} setReq={setReq} scenarios={scenarios} onRun={run} onReset={reset} loading={loading} />

        <div className="main-col">
          {error && <ErrorBanner message={error} />}
          {log && <WarningBanners warnings={log.warnings} />}

          <Suspense fallback={<div className="viewer-card" />}>
            <ViewerCard log={log} time={playback.time} />
          </Suspense>
          <PlaybackBar playback={playback} disabled={!log} />
          {log && <SummaryCards log={log} />}
          <ChartsPanel log={log} time={playback.time} />
        </div>
      </div>

      <p className="footer-note">
        Rust engine ({'`'}engine/{'`'}) cross-validated against the original Python reference ({'`'}legacy-python/{'`'}) &mdash; see the README for scope and
        parity notes.
      </p>
    </div>
  )
}

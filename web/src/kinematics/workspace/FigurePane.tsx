import { useCallback, useEffect, useRef, useState } from 'react'
import { PlaybackBar } from '../../components/PlaybackBar'
import type { Playback } from '../../hooks/usePlayback'
import { Scene2D } from '../Scene2D'
import type { FigureBody, WorkspaceView } from './evaluate'
import { DEFAULT_WINDOW, type PlotWindow } from './math/eval'
import { Scene3D } from './Scene3D'

export function FigurePane({ view, playback }: { view: WorkspaceView; playback: Playback }) {
  const stageRef = useRef<HTMLDivElement>(null)
  const [height, setHeight] = useState(480)
  const [box, setBox] = useState<PlotWindow>(DEFAULT_WINDOW)
  const [lift, setLift] = useState(false)

  useEffect(() => {
    const stage = stageRef.current
    if (!stage) return
    const observer = new ResizeObserver(() => {
      const next = Math.floor(stage.clientHeight) - 28
      if (next > 120) setHeight(next)
    })
    observer.observe(stage)
    return () => observer.disconnect()
  }, [])

  const resetView = useCallback(() => setBox(DEFAULT_WINDOW), [])
  const motion = view.bodies.some((body) => body.role !== 'plot')
  const hasCurve = view.bodies.some((body) => body.role === 'plot')
  const forced3d = view.dimension === 3
  const show3d = forced3d || (lift && hasCurve)
  const interactive = !motion && hasCurve && !show3d
  const windowFor = show3d ? DEFAULT_WINDOW : box

  const drawnBodies: FigureBody[] = view.bodies.map((body) => {
    if (motion || body.role !== 'plot' || !body.sample) return body
    return { ...body, path: body.sample(windowFor) }
  })

  const bodies2d = drawnBodies.map((body) => ({
    label: body.label,
    color: body.color,
    path: body.path.map((point) => ({ x: point.x, y: point.y })),
    currentIndex: body.index,
    dashed: body.dashed,
    hideMarker: body.hideMarker,
    velocityVector: body.velocity ? { vx: body.velocity.x, vy: body.velocity.y } : undefined,
  }))
  const hasFigure = view.bodies.length > 0 || view.surfaces.length > 0

  return (
    <section className="workspace-figure panel">
      <div className="panel-header">
        <h2>Figure</h2>
        <div className="workspace-tools">
          {hasCurve && !forced3d && (
            <button type="button" className="btn btn-ghost workspace-clear" aria-pressed={lift} onClick={() => setLift((current) => !current)}>
              {lift ? '2D view' : 'Curves in 3D'}
            </button>
          )}
          {interactive && (
            <button type="button" className="btn btn-ghost workspace-clear" onClick={resetView}>
              Reset view
            </button>
          )}
          <span className="badge">{show3d ? '3D' : '2D'}</span>
        </div>
      </div>
      <div className="workspace-figure-stage" ref={stageRef}>
        {!hasFigure ? (
          <p className="workspace-figure-empty">Graphs and motion show up here. A calculation with no graph stays in the list.</p>
        ) : show3d ? (
          <Scene3D bodies={drawnBodies} surfaces={view.surfaces} fitKey={view.fitKey} />
        ) : (
          <Scene2D
            bodies={bodies2d}
            height={height}
            aspectEqual={motion}
            includeOrigin={motion}
            viewBox={interactive ? box : undefined}
            onViewBox={interactive ? setBox : undefined}
            onReset={interactive ? resetView : undefined}
            xLabel={motion ? 'x (m)' : 'x'}
            yLabel={motion ? 'y (m)' : 'y'}
          />
        )}
        {interactive && <p className="figure-hint">Drag to move. Scroll to zoom.</p>}
      </div>
      {view.duration > 0 && <PlaybackBar playback={playback} disabled={false} />}
    </section>
  )
}

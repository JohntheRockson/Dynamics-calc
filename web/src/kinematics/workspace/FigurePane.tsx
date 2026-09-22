import { useEffect, useRef, useState } from 'react'
import { PlaybackBar } from '../../components/PlaybackBar'
import type { Playback } from '../../hooks/usePlayback'
import { Scene2D } from '../Scene2D'
import type { WorkspaceView } from './evaluate'
import { Scene3D } from './Scene3D'

export function FigurePane({ view, playback }: { view: WorkspaceView; playback: Playback }) {
  const stageRef = useRef<HTMLDivElement>(null)
  const [height, setHeight] = useState(480)

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

  const bodies2d = view.bodies.map((body) => ({
    label: body.label,
    color: body.color,
    path: body.path.map((point) => ({ x: point.x, y: point.y })),
    currentIndex: body.index,
    dashed: body.dashed,
    hideMarker: body.hideMarker,
    velocityVector: body.velocity ? { vx: body.velocity.x, vy: body.velocity.y } : undefined,
  }))
  const hasFigure = view.bodies.length > 0 || view.surfaces.length > 0
  const motion = view.bodies.some((body) => body.role !== 'plot')

  return (
    <section className="workspace-figure panel">
      <div className="panel-header">
        <h2>Figure</h2>
        <span className="badge">{view.dimension === 3 ? '3D' : '2D'}</span>
      </div>
      <div className="workspace-figure-stage" ref={stageRef}>
        {!hasFigure ? (
          <p className="workspace-figure-empty">Graphs and motion show up here. A calculation with no graph stays in the list.</p>
        ) : view.dimension === 3 ? (
          <Scene3D bodies={view.bodies} surfaces={view.surfaces} fitKey={view.fitKey} />
        ) : (
          <Scene2D bodies={bodies2d} height={height} aspectEqual includeOrigin xLabel={motion ? 'x (m)' : 'x'} yLabel={motion ? 'y (m)' : 'y'} />
        )}
      </div>
      {view.duration > 0 && <PlaybackBar playback={playback} disabled={false} />}
    </section>
  )
}

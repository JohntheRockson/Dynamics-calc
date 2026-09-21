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

  return (
    <section className="workspace-figure panel">
      <div className="panel-header">
        <h2>Figure</h2>
        <span className="badge">{view.dimension === 3 ? '3D' : '2D'}</span>
      </div>
      <div className="workspace-figure-stage" ref={stageRef}>
        {view.bodies.length === 0 ? (
          <p className="workspace-figure-empty">Add a point and the figure will draw it here.</p>
        ) : view.dimension === 3 ? (
          <Scene3D bodies={view.bodies} fitKey={view.fitKey} />
        ) : (
          <Scene2D bodies={bodies2d} height={height} aspectEqual includeOrigin />
        )}
      </div>
      {view.duration > 0 && <PlaybackBar playback={playback} disabled={false} />}
    </section>
  )
}

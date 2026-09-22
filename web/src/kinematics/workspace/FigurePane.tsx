import { useCallback, useEffect, useRef, useState } from 'react'
import { PlaybackBar } from '../../components/PlaybackBar'
import type { Playback } from '../../hooks/usePlayback'
import { Scene2D } from '../Scene2D'
import type { FigureBody, FigureSurface, WorkspaceView } from './evaluate'
import { DEFAULT_WINDOW, type PlotWindow } from './math/eval'
import { clipToDomain, sheetsFromCurve } from './math/extrude'
import { Scene3D } from './Scene3D'

export function FigurePane({ view, playback, planes }: { view: WorkspaceView; playback: Playback; planes: Record<string, boolean> }) {
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
  const planeOn = (body: FigureBody) => Boolean(body.statementId && planes[body.statementId] && body.role === 'plot')
  const anyPlane = view.bodies.some(planeOn)
  const forced3d = view.dimension === 3
  const show3d = forced3d || anyPlane || (lift && hasCurve)
  const interactive = !motion && hasCurve && !show3d
  const windowFor = show3d ? DEFAULT_WINDOW : box

  const domain = surfaceDomain(view.surfaces)
  const drawnBodies: FigureBody[] = view.bodies.map((body) => {
    if (motion || body.role !== 'plot' || !body.sample) return body
    const path = body.sample(windowFor)
    return { ...body, path: show3d && domain ? clipToDomain(path, domain) : path }
  })
  const zSpan = surfaceZ(view.surfaces)
  const planeSurfaces: FigureSurface[] = []
  const lineBodies = drawnBodies.filter((body) => {
    if (!planeOn(body)) return true
    const reach = curveReach(body.path)
    const sheets = sheetsFromCurve(body.path, zSpan?.min ?? -reach, zSpan?.max ?? reach)
    if (sheets.length === 0) return true
    planeSurfaces.push({ label: body.label, color: body.color, grid: sheets[0], sheets, flat: true })
    return false
  })

  const bodies2d = lineBodies.map((body) => ({
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
          {hasCurve && !forced3d && !anyPlane && (
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
          <Scene3D bodies={lineBodies} surfaces={[...view.surfaces, ...planeSurfaces]} fitKey={`${view.fitKey}:${planeSurfaces.map((surface) => surface.label).join(',')}`} />
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

function surfaceZ(surfaces: FigureSurface[]): { min: number; max: number } | null {
  let min = Infinity
  let max = -Infinity
  for (const surface of surfaces) {
    const sheets = surface.sheets && surface.sheets.length > 0 ? surface.sheets : [surface.grid]
    for (const grid of sheets) {
      for (const row of grid) {
        for (const point of row) {
          if (!Number.isFinite(point.z)) continue
          min = Math.min(min, point.z)
          max = Math.max(max, point.z)
        }
      }
    }
  }
  if (!Number.isFinite(min) || !Number.isFinite(max)) return null
  if (max - min < 1e-6) return { min: min - 1, max: max + 1 }
  return { min, max }
}

function surfaceDomain(surfaces: FigureSurface[]): { xMin: number; xMax: number; yMin: number; yMax: number } | null {
  let xMin = Infinity
  let xMax = -Infinity
  let yMin = Infinity
  let yMax = -Infinity
  for (const surface of surfaces) {
    const sheets = surface.sheets && surface.sheets.length > 0 ? surface.sheets : [surface.grid]
    for (const grid of sheets) {
      for (const row of grid) {
        for (const point of row) {
          if (!Number.isFinite(point.x) || !Number.isFinite(point.y)) continue
          xMin = Math.min(xMin, point.x)
          xMax = Math.max(xMax, point.x)
          yMin = Math.min(yMin, point.y)
          yMax = Math.max(yMax, point.y)
        }
      }
    }
  }
  if (!Number.isFinite(xMin)) return null
  return { xMin, xMax, yMin, yMax }
}

function curveReach(path: { x: number; y: number }[]): number {
  let reach = 10
  for (const point of path) {
    if (!Number.isFinite(point.x) || !Number.isFinite(point.y)) continue
    reach = Math.max(reach, Math.abs(point.x), Math.abs(point.y))
  }
  return reach
}

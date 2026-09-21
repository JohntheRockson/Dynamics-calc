import { useEffect, useRef, useState } from 'react'

export interface Scene2DVector {
  vx: number
  vy: number
  color: string
  dashed?: boolean
  /** Draw at a fixed pixel length (unit vectors) instead of the velocity scale. */
  unit?: boolean
}

export interface Scene2DBody {
  label: string
  color: string
  path: { x: number; y: number }[]
  currentIndex: number
  velocityVector?: { vx: number; vy: number }
  extraVectors?: Scene2DVector[]
  dashed?: boolean
  hideMarker?: boolean
}

export interface Scene2DMarker {
  x: number
  y: number
  label: string
  color?: string
}

interface Scene2DProps {
  bodies: Scene2DBody[]
  markers?: Scene2DMarker[]
  height?: number
  xLabel?: string
  yLabel?: string
  aspectEqual?: boolean
  groundY?: number
  includeOrigin?: boolean
}

function niceTick(v: number): string {
  if (v === 0) return '0'
  const abs = Math.abs(v)
  if (abs >= 1000 || abs < 0.01) return v.toExponential(1)
  return v.toFixed(abs < 1 ? 2 : abs < 10 ? 1 : 0)
}

export function Scene2D({ bodies, markers = [], height = 260, xLabel = 'x (m)', yLabel = 'y (m)', aspectEqual = false, groundY, includeOrigin = false }: Scene2DProps) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const [width, setWidth] = useState(600)

  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const ro = new ResizeObserver((entries) => {
      const w = entries[0]?.contentRect.width
      if (w && Math.abs(w - width) > 1) setWidth(w)
    })
    ro.observe(el)
    setWidth(el.clientWidth || 600)
    return () => ro.disconnect()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || width <= 0) return
    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    canvas.width = Math.round(width * dpr)
    canvas.height = Math.round(height * dpr)
    canvas.style.width = `${width}px`
    canvas.style.height = `${height}px`
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, width, height)

    const margin = { l: 48, r: 14, t: 14, b: 26 }
    const plotW = Math.max(1, width - margin.l - margin.r)
    const plotH = Math.max(1, height - margin.t - margin.b)

    let xmin = Infinity
    let xmax = -Infinity
    let ymin = Infinity
    let ymax = -Infinity
    for (const b of bodies) {
      for (const pt of b.path) {
        if (pt.x < xmin) xmin = pt.x
        if (pt.x > xmax) xmax = pt.x
        if (pt.y < ymin) ymin = pt.y
        if (pt.y > ymax) ymax = pt.y
      }
    }
    for (const m of markers) {
      if (m.x < xmin) xmin = m.x
      if (m.x > xmax) xmax = m.x
      if (m.y < ymin) ymin = m.y
      if (m.y > ymax) ymax = m.y
    }
    if (groundY !== undefined) {
      ymin = Math.min(ymin, groundY)
    }
    if (includeOrigin) {
      xmin = Math.min(xmin, 0)
      xmax = Math.max(xmax, 0)
      ymin = Math.min(ymin, 0)
      ymax = Math.max(ymax, 0)
    }
    if (!Number.isFinite(xmin) || !Number.isFinite(xmax)) {
      xmin = -1
      xmax = 1
    }
    if (!Number.isFinite(ymin) || !Number.isFinite(ymax)) {
      ymin = -1
      ymax = 1
    }
    if (xmax - xmin < 1e-6) {
      xmax += 1
      xmin -= 1
    }
    if (ymax - ymin < 1e-6) {
      ymax += 1
      ymin -= 1
    }
    const padX = (xmax - xmin) * 0.08
    const padY = (ymax - ymin) * 0.12
    xmin -= padX
    xmax += padX
    ymin -= padY
    ymax += padY

    if (aspectEqual) {
      const dataAspect = (xmax - xmin) / (ymax - ymin)
      const plotAspect = plotW / plotH
      if (dataAspect > plotAspect) {
        const targetYSpan = (xmax - xmin) / plotAspect
        const cy = (ymin + ymax) / 2
        ymin = cy - targetYSpan / 2
        ymax = cy + targetYSpan / 2
      } else {
        const targetXSpan = (ymax - ymin) * plotAspect
        const cx = (xmin + xmax) / 2
        xmin = cx - targetXSpan / 2
        xmax = cx + targetXSpan / 2
      }
    }

    const sx = (x: number) => margin.l + ((x - xmin) / (xmax - xmin)) * plotW
    const sy = (y: number) => margin.t + (1 - (y - ymin) / (ymax - ymin)) * plotH

    // Gridlines + ticks.
    ctx.font = '10px var(--font-mono), monospace'
    ctx.fillStyle = '#5b6a7e'
    const xTicks = Math.max(2, Math.min(8, Math.round(plotW / 90)))
    for (let i = 0; i <= xTicks; i++) {
      const xv = xmin + ((xmax - xmin) * i) / xTicks
      const px = sx(xv)
      ctx.strokeStyle = 'rgba(255,255,255,0.06)'
      ctx.beginPath()
      ctx.moveTo(px, margin.t)
      ctx.lineTo(px, height - margin.b)
      ctx.stroke()
      ctx.fillText(niceTick(xv), px - 10, height - 8)
    }
    const yTicks = Math.max(2, Math.min(6, Math.round(plotH / 50)))
    for (let i = 0; i <= yTicks; i++) {
      const yv = ymin + ((ymax - ymin) * i) / yTicks
      const py = sy(yv)
      ctx.strokeStyle = 'rgba(255,255,255,0.06)'
      ctx.beginPath()
      ctx.moveTo(margin.l, py)
      ctx.lineTo(width - margin.r, py)
      ctx.stroke()
      ctx.fillText(niceTick(yv), 4, py + 3)
    }

    // Zero axes, a bit brighter.
    ctx.strokeStyle = 'rgba(255,255,255,0.16)'
    if (xmin < 0 && xmax > 0) {
      const px = sx(0)
      ctx.beginPath()
      ctx.moveTo(px, margin.t)
      ctx.lineTo(px, height - margin.b)
      ctx.stroke()
    }
    if (ymin < 0 && ymax > 0) {
      const py = sy(0)
      ctx.beginPath()
      ctx.moveTo(margin.l, py)
      ctx.lineTo(width - margin.r, py)
      ctx.stroke()
    }

    if (groundY !== undefined) {
      ctx.strokeStyle = '#8b5a2b'
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.moveTo(margin.l, sy(groundY))
      ctx.lineTo(width - margin.r, sy(groundY))
      ctx.stroke()
    }

    // Axis labels.
    ctx.fillStyle = '#93a2b6'
    ctx.font = '11px var(--font-sans), sans-serif'
    ctx.fillText(xLabel, width - margin.r - ctx.measureText(xLabel).width, height - margin.b + 20 > height - 2 ? height - 2 : height - 2)
    ctx.save()
    ctx.translate(12, margin.t + 4)
    ctx.fillText(yLabel, 0, 0)
    ctx.restore()

    // Bodies: path + current marker + optional velocity vector.
    for (const b of bodies) {
      if (b.path.length === 0) continue
      ctx.strokeStyle = b.color
      ctx.lineWidth = 2
      ctx.setLineDash(b.dashed ? [5, 4] : [])
      ctx.beginPath()
      b.path.forEach((pt, i) => {
        const px = sx(pt.x)
        const py = sy(pt.y)
        if (i === 0) ctx.moveTo(px, py)
        else ctx.lineTo(px, py)
      })
      ctx.stroke()
      ctx.setLineDash([])

      const idx = Math.min(b.currentIndex, b.path.length - 1)
      const cur = b.path[idx]
      const cpx = sx(cur.x)
      const cpy = sy(cur.y)
      if (!b.hideMarker) {
        ctx.fillStyle = b.color
        ctx.beginPath()
        ctx.arc(cpx, cpy, 5, 0, Math.PI * 2)
        ctx.fill()
        ctx.strokeStyle = '#05080c'
        ctx.lineWidth = 1.5
        ctx.stroke()
      }

      const arrows: Scene2DVector[] = []
      if (b.velocityVector) arrows.push({ ...b.velocityVector, color: b.color })
      if (b.extraVectors) arrows.push(...b.extraVectors)
      const scale = Math.min(plotW, plotH) * 0.16
      const unitLen = Math.min(plotW, plotH) * 0.12
      for (const vec of arrows) {
        const vmag = Math.hypot(vec.vx, vec.vy)
        if (vmag <= 1e-9) continue
        const len = vec.unit ? unitLen : scale
        const dx = (vec.vx / vmag) * len
        const dy = (vec.vy / vmag) * len
        const ex = cpx + dx
        const ey = cpy - dy
        ctx.strokeStyle = vec.color
        ctx.fillStyle = vec.color
        ctx.lineWidth = vec.unit ? 1.6 : 2
        ctx.setLineDash(vec.dashed ? [4, 3] : [])
        ctx.beginPath()
        ctx.moveTo(cpx, cpy)
        ctx.lineTo(ex, ey)
        ctx.stroke()
        ctx.setLineDash([])
        const angle = Math.atan2(ey - cpy, ex - cpx)
        ctx.beginPath()
        ctx.moveTo(ex, ey)
        ctx.lineTo(ex - 8 * Math.cos(angle - 0.4), ey - 8 * Math.sin(angle - 0.4))
        ctx.lineTo(ex - 8 * Math.cos(angle + 0.4), ey - 8 * Math.sin(angle + 0.4))
        ctx.closePath()
        ctx.fill()
      }
    }
    for (const m of markers) {
      const px = sx(m.x)
      const py = sy(m.y)
      const color = m.color ?? '#e7edf5'
      ctx.fillStyle = color
      ctx.beginPath()
      ctx.arc(px, py, 4, 0, Math.PI * 2)
      ctx.fill()
      ctx.strokeStyle = '#05080c'
      ctx.lineWidth = 1.2
      ctx.stroke()
      ctx.font = '700 13px var(--font-sans), sans-serif'
      ctx.fillStyle = color
      ctx.fillText(m.label, px + 7, py - 7)
    }
  }, [bodies, markers, width, height, xLabel, yLabel, aspectEqual, groundY, includeOrigin])

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <canvas ref={canvasRef} className="chart-canvas" />
      <div className="chart-legend">
        {bodies.map((b) => (
          <span className="item" key={b.label}>
            <span className="swatch" style={{ background: b.color, opacity: b.dashed ? 0.7 : 1 }} />
            {b.label}
          </span>
        ))}
      </div>
    </div>
  )
}

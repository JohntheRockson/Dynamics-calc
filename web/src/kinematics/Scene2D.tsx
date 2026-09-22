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

export interface Scene2DWindow {
  xMin: number
  xMax: number
  yMin: number
  yMax: number
}

interface Scene2DProps {
  bodies: Scene2DBody[]
  height?: number
  xLabel?: string
  yLabel?: string
  aspectEqual?: boolean
  groundY?: number
  includeOrigin?: boolean
  /** When set, draw this window exactly and let the pointer pan and zoom it. */
  viewBox?: Scene2DWindow
  onViewBox?: (box: Scene2DWindow) => void
  onReset?: () => void
}

function niceTick(v: number): string {
  if (v === 0) return '0'
  const abs = Math.abs(v)
  if (abs >= 1000 || abs < 0.01) return v.toExponential(1)
  return v.toFixed(abs < 1 ? 2 : abs < 10 ? 1 : 0)
}

export function Scene2D({ bodies, height = 260, xLabel = 'x (m)', yLabel = 'y (m)', aspectEqual = false, groundY, includeOrigin = false, viewBox, onViewBox, onReset }: Scene2DProps) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const frameRef = useRef<{ marginL: number; marginT: number; plotW: number; plotH: number } | null>(null)
  const liveRef = useRef(viewBox)
  liveRef.current = viewBox
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
        if (!Number.isFinite(pt.x) || !Number.isFinite(pt.y)) continue
        if (pt.x < xmin) xmin = pt.x
        if (pt.x > xmax) xmax = pt.x
        if (pt.y < ymin) ymin = pt.y
        if (pt.y > ymax) ymax = pt.y
      }
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

    if (viewBox && viewBox.xMax > viewBox.xMin && viewBox.yMax > viewBox.yMin) {
      xmin = viewBox.xMin
      xmax = viewBox.xMax
      ymin = viewBox.yMin
      ymax = viewBox.yMax
    } else if (aspectEqual) {
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
    frameRef.current = { marginL: margin.l, marginT: margin.t, plotW, plotH }

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
    ctx.save()
    ctx.beginPath()
    ctx.rect(margin.l, margin.t, plotW, plotH)
    ctx.clip()
    for (const b of bodies) {
      if (b.path.length === 0) continue
      ctx.strokeStyle = b.color
      ctx.lineWidth = 2
      ctx.setLineDash(b.dashed ? [5, 4] : [])
      ctx.beginPath()
      let drawing = false
      for (const pt of b.path) {
        if (!Number.isFinite(pt.x) || !Number.isFinite(pt.y)) {
          drawing = false
          continue
        }
        const px = sx(pt.x)
        const py = sy(pt.y)
        if (!drawing) {
          ctx.moveTo(px, py)
          drawing = true
        } else ctx.lineTo(px, py)
      }
      ctx.stroke()
      ctx.setLineDash([])

      const idx = Math.min(b.currentIndex, b.path.length - 1)
      const cur = b.path[idx]
      if (!cur || !Number.isFinite(cur.x) || !Number.isFinite(cur.y)) continue
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
    ctx.restore()
  }, [bodies, width, height, xLabel, yLabel, aspectEqual, groundY, includeOrigin, viewBox])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !onViewBox) return
    let dragging = false
    const endDrag = () => {
      dragging = false
      canvas.classList.remove('is-grabbing')
    }
    const onDown = (event: PointerEvent) => {
      if (event.button !== 0) return
      dragging = true
      canvas.classList.add('is-grabbing')
      canvas.setPointerCapture(event.pointerId)
    }
    const onMove = (event: PointerEvent) => {
      if (!dragging) return
      const frame = frameRef.current
      const current = liveRef.current
      if (!frame || !current) return
      const xSpan = current.xMax - current.xMin
      const ySpan = current.yMax - current.yMin
      const dx = (event.movementX / frame.plotW) * xSpan
      const dy = (event.movementY / frame.plotH) * ySpan
      const next = { xMin: current.xMin - dx, xMax: current.xMax - dx, yMin: current.yMin + dy, yMax: current.yMax + dy }
      liveRef.current = next
      onViewBox(next)
    }
    const onWheel = (event: WheelEvent) => {
      event.preventDefault()
      const frame = frameRef.current
      const current = liveRef.current
      if (!frame || !current) return
      const rect = canvas.getBoundingClientRect()
      const px = event.clientX - rect.left
      const py = event.clientY - rect.top
      const xSpan = current.xMax - current.xMin
      const ySpan = current.yMax - current.yMin
      const mx = current.xMin + ((px - frame.marginL) / frame.plotW) * xSpan
      const my = current.yMin + (1 - (py - frame.marginT) / frame.plotH) * ySpan
      const factor = event.deltaY > 0 ? 1.12 : 1 / 1.12
      const xMin = mx - (mx - current.xMin) * factor
      const xMax = mx + (current.xMax - mx) * factor
      const yMin = my - (my - current.yMin) * factor
      const yMax = my + (current.yMax - my) * factor
      if (xMax - xMin < 1e-4 || yMax - yMin < 1e-4 || xMax - xMin > 1e6 || yMax - yMin > 1e6) return
      const next = { xMin, xMax, yMin, yMax }
      liveRef.current = next
      onViewBox(next)
    }
    const onDouble = (event: MouseEvent) => {
      event.preventDefault()
      onReset?.()
    }
    canvas.addEventListener('pointerdown', onDown)
    canvas.addEventListener('pointermove', onMove)
    canvas.addEventListener('pointerup', endDrag)
    canvas.addEventListener('pointercancel', endDrag)
    canvas.addEventListener('wheel', onWheel, { passive: false })
    canvas.addEventListener('dblclick', onDouble)
    return () => {
      canvas.removeEventListener('pointerdown', onDown)
      canvas.removeEventListener('pointermove', onMove)
      canvas.removeEventListener('pointerup', endDrag)
      canvas.removeEventListener('pointercancel', endDrag)
      canvas.removeEventListener('wheel', onWheel)
      canvas.removeEventListener('dblclick', onDouble)
    }
  }, [onViewBox, onReset])

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <canvas ref={canvasRef} className={onViewBox ? 'chart-canvas is-pannable' : 'chart-canvas'} />
      <div className="chart-legend">
        {bodies.map((b, index) => (
          <span className="item" key={`${b.label}-${index}`}>
            <span className="swatch" style={{ background: b.color, opacity: b.dashed ? 0.7 : 1 }} />
            {b.label}
          </span>
        ))}
      </div>
    </div>
  )
}

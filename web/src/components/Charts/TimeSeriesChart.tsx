import { useEffect, useRef, useState } from 'react'
import { formatNumber } from '../../format'

export interface ChartSeries {
  label: string
  color: string
  data: number[]
  dashed?: boolean
  dotted?: boolean
}

interface Props {
  t: number[]
  series: ChartSeries[]
  currentTime: number
  height?: number
  referenceLines?: { value: number; color: string; label: string }[]
}

function niceTick(v: number): string {
  return formatNumber(v, Math.abs(v) < 1 ? 3 : 2)
}

function bisectIndex(t: number[], time: number): number {
  let lo = 0
  let hi = t.length - 1
  if (hi < 0) return 0
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if (t[mid] < time) lo = mid + 1
    else hi = mid
  }
  return lo
}

export function TimeSeriesChart({ t, series, currentTime, height = 190, referenceLines }: Props) {
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

    const marginL = 46
    const marginR = 10
    const marginT = 10
    const marginB = 20
    const plotW = Math.max(1, width - marginL - marginR)
    const plotH = Math.max(1, height - marginT - marginB)

    let ymin = Infinity
    let ymax = -Infinity
    for (const s of series) {
      for (const v of s.data) {
        if (Number.isFinite(v)) {
          if (v < ymin) ymin = v
          if (v > ymax) ymax = v
        }
      }
    }
    for (const r of referenceLines ?? []) {
      if (r.value < ymin) ymin = r.value
      if (r.value > ymax) ymax = r.value
    }
    if (!Number.isFinite(ymin) || !Number.isFinite(ymax)) {
      ymin = 0
      ymax = 1
    }
    if (ymax - ymin < 1e-9) {
      ymin -= 1
      ymax += 1
    }
    const pad = (ymax - ymin) * 0.1
    ymin -= pad
    ymax += pad

    const xmin = t.length ? t[0] : 0
    const xmax = t.length ? t[t.length - 1] : 1
    const xSpan = xmax - xmin || 1
    const xScale = (x: number) => marginL + ((x - xmin) / xSpan) * plotW
    const yScale = (y: number) => marginT + (1 - (y - ymin) / (ymax - ymin)) * plotH

    // Gridlines + y ticks.
    ctx.font = '10px var(--font-mono), monospace'
    ctx.textBaseline = 'middle'
    const yTicks = 4
    for (let i = 0; i <= yTicks; i++) {
      const yv = ymin + ((ymax - ymin) * i) / yTicks
      const yy = yScale(yv)
      ctx.strokeStyle = 'rgba(255,255,255,0.06)'
      ctx.beginPath()
      ctx.moveTo(marginL, yy)
      ctx.lineTo(width - marginR, yy)
      ctx.stroke()
      ctx.fillStyle = '#5b6a7e'
      ctx.fillText(niceTick(yv), 2, yy)
    }
    const xTicks = Math.min(6, Math.max(2, Math.floor(plotW / 90)))
    ctx.textBaseline = 'alphabetic'
    for (let i = 0; i <= xTicks; i++) {
      const xv = xmin + (xSpan * i) / xTicks
      const xx = xScale(xv)
      ctx.fillStyle = '#5b6a7e'
      ctx.fillText(`${xv.toFixed(1)}s`, Math.min(width - marginR - 24, Math.max(marginL, xx - 12)), height - 5)
    }

    // Reference lines (e.g. expected chi-square NIS mean).
    for (const r of referenceLines ?? []) {
      const yy = yScale(r.value)
      ctx.strokeStyle = r.color
      ctx.globalAlpha = 0.6
      ctx.setLineDash([2, 3])
      ctx.beginPath()
      ctx.moveTo(marginL, yy)
      ctx.lineTo(width - marginR, yy)
      ctx.stroke()
      ctx.setLineDash([])
      ctx.globalAlpha = 1
    }

    // Series.
    for (const s of series) {
      ctx.strokeStyle = s.color
      ctx.fillStyle = s.color
      ctx.lineWidth = 1.6
      ctx.setLineDash(s.dashed ? [5, 4] : [])
      if (s.dotted) {
        for (let i = 0; i < t.length; i++) {
          const v = s.data[i]
          if (!Number.isFinite(v)) continue
          ctx.beginPath()
          ctx.arc(xScale(t[i]), yScale(v), 1.6, 0, Math.PI * 2)
          ctx.fill()
        }
      } else {
        ctx.beginPath()
        let started = false
        for (let i = 0; i < t.length; i++) {
          const v = s.data[i]
          if (!Number.isFinite(v)) {
            started = false
            continue
          }
          const x = xScale(t[i])
          const y = yScale(v)
          if (!started) {
            ctx.moveTo(x, y)
            started = true
          } else {
            ctx.lineTo(x, y)
          }
        }
        ctx.stroke()
      }
    }
    ctx.setLineDash([])

    // Playhead.
    if (t.length) {
      const clampedTime = Math.min(xmax, Math.max(xmin, currentTime))
      const px = xScale(clampedTime)
      ctx.strokeStyle = 'rgba(41,211,245,0.9)'
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.moveTo(px, marginT)
      ctx.lineTo(px, height - marginB)
      ctx.stroke()
    }
  }, [t, series, currentTime, width, height, referenceLines])

  const idx = bisectIndex(t, currentTime)

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <canvas ref={canvasRef} className="chart-canvas" />
      <div className="chart-legend">
        {series.map((s) => (
          <span className="item" key={s.label}>
            <span className="swatch" style={{ background: s.color, opacity: s.dashed || s.dotted ? 0.7 : 1 }} />
            {s.label}
            {t.length > 0 && Number.isFinite(s.data[idx]) ? `: ${formatNumber(s.data[idx], 3)}` : ''}
          </span>
        ))}
      </div>
    </div>
  )
}

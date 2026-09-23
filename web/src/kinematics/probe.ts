// Snap a click on a 2D curve to a nearby intersection, extremum, or intercept.

export interface ProbePoint {
  x: number
  y: number
  kind: 'point' | 'intersection' | 'maximum' | 'minimum' | 'extreme' | 'intercept'
  text: string
}

const HIT_PX = 14
const SNAP_PX = 22

const RANK: Record<ProbePoint['kind'], number> = {
  intersection: 0,
  maximum: 1,
  minimum: 1,
  extreme: 2,
  intercept: 3,
  point: 4,
}

export function chooseProbe(
  paths: { x: number; y: number }[][],
  click: { x: number; y: number },
  toScreen: (x: number, y: number) => { x: number; y: number },
): ProbePoint | null {
  let best: { x: number; y: number; dist: number } | null = null
  for (const path of paths) {
    const hit = closestOnPath(path, click, toScreen)
    if (!hit || hit.dist > HIT_PX) continue
    if (!best || hit.dist < best.dist) best = hit
  }
  if (!best) return null
  const features = featurePoints(paths)
  const clickPx = toScreen(click.x, click.y)
  let snapped: ProbePoint | null = null
  let snappedDist = SNAP_PX
  for (const feature of features) {
    const at = toScreen(feature.x, feature.y)
    const dist = Math.hypot(at.x - clickPx.x, at.y - clickPx.y)
    if (dist > SNAP_PX) continue
    if (!snapped || dist < snappedDist - 4 || (Math.abs(dist - snappedDist) <= 4 && RANK[feature.kind] < RANK[snapped.kind])) {
      snapped = feature
      snappedDist = dist
    }
  }
  if (snapped) return snapped
  return { x: best.x, y: best.y, kind: 'point', text: `(${formatCoord(best.x)}, ${formatCoord(best.y)})` }
}

export function featurePoints(paths: { x: number; y: number }[][]): ProbePoint[] {
  const found: ProbePoint[] = []
  const branches = paths.map(splitBranches)
  branches.forEach((pathBranches, index) => {
    for (const branch of pathBranches) {
      found.push(...extrema(branch))
      found.push(...intercepts(branch))
      if (branch.length === 2) found.push(named(branch[1], 'point'))
    }
    for (let other = index + 1; other < branches.length; other += 1) {
      for (const left of pathBranches) {
        for (const right of branches[other]) found.push(...intersections(left, right))
      }
    }
  })
  const kept: ProbePoint[] = []
  for (const point of found) {
    const match = kept.findIndex((item) => Math.hypot(item.x - point.x, item.y - point.y) <= 1e-4 * Math.max(1, Math.hypot(point.x, point.y)))
    if (match < 0) kept.push(point)
    else if (RANK[point.kind] < RANK[kept[match].kind]) kept[match] = point
  }
  return kept
}

function closestOnPath(
  path: { x: number; y: number }[],
  click: { x: number; y: number },
  toScreen: (x: number, y: number) => { x: number; y: number },
): { x: number; y: number; dist: number } | null {
  const clickPx = toScreen(click.x, click.y)
  let best: { x: number; y: number; dist: number } | null = null
  let previous: { x: number; y: number } | null = null
  for (const point of path) {
    if (!Number.isFinite(point.x) || !Number.isFinite(point.y)) {
      previous = null
      continue
    }
    if (previous) {
      const hit = closestOnSegment(previous, point, clickPx, toScreen)
      if (!best || hit.dist < best.dist) best = hit
    }
    previous = point
  }
  return best
}

function closestOnSegment(
  a: { x: number; y: number },
  b: { x: number; y: number },
  clickPx: { x: number; y: number },
  toScreen: (x: number, y: number) => { x: number; y: number },
): { x: number; y: number; dist: number } {
  const ax = toScreen(a.x, a.y)
  const bx = toScreen(b.x, b.y)
  const dx = bx.x - ax.x
  const dy = bx.y - ax.y
  const len2 = dx * dx + dy * dy
  const t = len2 <= 1e-9 ? 0 : Math.min(1, Math.max(0, ((clickPx.x - ax.x) * dx + (clickPx.y - ax.y) * dy) / len2))
  const x = a.x + (b.x - a.x) * t
  const y = a.y + (b.y - a.y) * t
  const screen = toScreen(x, y)
  return { x, y, dist: Math.hypot(screen.x - clickPx.x, screen.y - clickPx.y) }
}

function splitBranches(path: { x: number; y: number }[]): { x: number; y: number }[][] {
  const branches: { x: number; y: number }[][] = []
  let current: { x: number; y: number }[] = []
  for (const point of path) {
    if (!Number.isFinite(point.x) || !Number.isFinite(point.y)) {
      if (current.length > 1) branches.push(current)
      current = []
      continue
    }
    current.push(point)
  }
  if (current.length > 1) branches.push(current)
  return branches
}

function extrema(branch: { x: number; y: number }[]): ProbePoint[] {
  const points: ProbePoint[] = []
  const ySpan = span(branch.map((point) => point.y))
  const xSpan = span(branch.map((point) => point.x))
  for (let i = 1; i < branch.length - 1; i += 1) {
    const yTurn = turn(branch[i - 1].y, branch[i].y, branch[i + 1].y, ySpan)
    const xTurn = turn(branch[i - 1].x, branch[i].x, branch[i + 1].x, xSpan)
    if (yTurn > 0) points.push(named(branch[i], 'maximum'))
    else if (yTurn < 0) points.push(named(branch[i], 'minimum'))
    else if (xTurn !== 0) points.push(named(branch[i], 'extreme'))
  }
  return points
}

function turn(previous: number, current: number, next: number, range: number): number {
  const eps = Math.max(1e-9, range * 1e-8)
  const before = current - previous
  const after = next - current
  if (before > eps && after < -eps) return 1
  if (before < -eps && after > eps) return -1
  return 0
}

function intercepts(branch: { x: number; y: number }[]): ProbePoint[] {
  const points: ProbePoint[] = []
  for (let i = 1; i < branch.length; i += 1) {
    const crossing = (axis: 'x' | 'y') => {
      const a = branch[i - 1]
      const b = branch[i]
      const aValue = axis === 'y' ? a.y : a.x
      const bValue = axis === 'y' ? b.y : b.x
      if (Math.abs(aValue) <= 1e-8) return a
      if (aValue * bValue > 0) return null
      const t = aValue / (aValue - bValue)
      if (t < 0 || t > 1) return null
      return { x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t }
    }
    const onX = crossing('y')
    const onY = crossing('x')
    if (onX) points.push(named(onX, 'intercept'))
    if (onY && Math.hypot(onY.x - (onX?.x ?? Infinity), onY.y - (onX?.y ?? Infinity)) > 1e-6) points.push(named(onY, 'intercept'))
  }
  return points
}

function intersections(left: { x: number; y: number }[], right: { x: number; y: number }[]): ProbePoint[] {
  const points: ProbePoint[] = []
  for (let i = 1; i < left.length; i += 1) {
    for (let j = 1; j < right.length; j += 1) {
      const hit = segmentIntersection(left[i - 1], left[i], right[j - 1], right[j])
      if (hit) points.push(named(hit, 'intersection'))
    }
  }
  return points
}

function segmentIntersection(
  a: { x: number; y: number },
  b: { x: number; y: number },
  c: { x: number; y: number },
  d: { x: number; y: number },
): { x: number; y: number } | null {
  const abx = b.x - a.x
  const aby = b.y - a.y
  const cdx = d.x - c.x
  const cdy = d.y - c.y
  const den = abx * cdy - aby * cdx
  if (Math.abs(den) < 1e-12) return null
  const t = ((c.x - a.x) * cdy - (c.y - a.y) * cdx) / den
  const u = ((c.x - a.x) * aby - (c.y - a.y) * abx) / den
  if (t < -1e-6 || t > 1 + 1e-6 || u < -1e-6 || u > 1 + 1e-6) return null
  return { x: a.x + t * abx, y: a.y + t * aby }
}

function span(values: number[]): number {
  let min = Infinity
  let max = -Infinity
  for (const value of values) {
    min = Math.min(min, value)
    max = Math.max(max, value)
  }
  return max - min
}

function named(point: { x: number; y: number }, kind: ProbePoint['kind']): ProbePoint {
  const coords = `(${formatCoord(point.x)}, ${formatCoord(point.y)})`
  const text = kind === 'point' ? coords : `${kind} ${coords}`
  return { x: point.x, y: point.y, kind, text }
}

function formatCoord(value: number): string {
  if (!Number.isFinite(value) || Math.abs(value) < 1e-8) return '0'
  const rounded = Math.round(value * 1e6) / 1e6
  const abs = Math.abs(rounded)
  const text = abs >= 1e5 ? rounded.toExponential(3) : String(rounded)
  return text.replace(/(\.\d*?)0+$/, '$1').replace(/\.$/, '')
}

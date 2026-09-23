// Turn a 2D curve into a sheet by sweeping it through z. A straight curve becomes a plane.

export interface CurvePoint {
  x: number
  y: number
  z: number
}

export function sheetsFromCurve(path: CurvePoint[], zMin: number, zMax: number, rows = 2): CurvePoint[][][] {
  const branches = branchesOf(path)
  if (branches.length === 0 || !(zMax > zMin)) return []
  return branches.map((points) => {
    const grid: CurvePoint[][] = []
    for (let row = 0; row < rows; row += 1) {
      const z = rows === 1 ? (zMin + zMax) / 2 : zMin + ((zMax - zMin) * row) / (rows - 1)
      grid.push(points.map((point) => ({ x: point.x, y: point.y, z })))
    }
    return grid
  })
}

function branchesOf(path: CurvePoint[]): CurvePoint[][] {
  const branches: CurvePoint[][] = []
  let branch: CurvePoint[] = []
  const push = () => {
    if (branch.length > 1) branches.push(thin(branch, 80))
    branch = []
  }
  for (const point of path) {
    if (!Number.isFinite(point.x) || !Number.isFinite(point.y)) push()
    else branch.push(point)
  }
  push()
  return branches
}

export function clipToDomain(path: CurvePoint[], domain: { xMin: number; xMax: number; yMin: number; yMax: number }): CurvePoint[] {
  return path.map((point) => {
    if (!Number.isFinite(point.x) || !Number.isFinite(point.y)) return point
    if (point.x < domain.xMin || point.x > domain.xMax || point.y < domain.yMin || point.y > domain.yMax) return { x: Number.NaN, y: Number.NaN, z: Number.NaN }
    return point
  })
}

function thin(points: CurvePoint[], max: number): CurvePoint[] {
  if (points.length <= max) return points
  const out: CurvePoint[] = []
  const step = (points.length - 1) / (max - 1)
  for (let i = 0; i < max; i += 1) out.push(points[Math.round(i * step)])
  return out
}

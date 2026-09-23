// Math graphs are scaled so each axis fills the same cube. A plot of z = x^2 + y^2
// then reads as a bowl instead of a needle. Motion keeps meters equal on every axis.

export interface PlotBox {
  xMin: number
  xMax: number
  yMin: number
  yMax: number
  zMin: number
  zMax: number
}

export interface PlotFrame {
  box: PlotBox
  /** World units match math units. Used for kinematics, where a meter is a meter. */
  equal: boolean
}

export const CUBE = 10

export function emptyBox(): PlotBox {
  return { xMin: -10, xMax: 10, yMin: -10, yMax: 10, zMin: -10, zMax: 10 }
}

/** The point the 3D axes cross: the middle of the cube. */
export function axisThrough(box: PlotBox): { x: number; y: number; z: number } {
  return {
    x: (box.xMin + box.xMax) / 2,
    y: (box.yMin + box.yMax) / 2,
    z: (box.zMin + box.zMax) / 2,
  }
}

/** A box centered on the origin. `span` is the x and y half-width, `zAbs` the z half-height. */
export function originBox(span: number, zAbs: number): PlotBox {
  const s = Math.max(Math.abs(span), 1e-6)
  const z = Math.max(Math.abs(zAbs), 1e-6)
  return { xMin: -s, xMax: s, yMin: -s, yMax: s, zMin: -z, zMax: z }
}

export function expandPlotBox(box: PlotBox): PlotBox {
  const next = { ...box }
  const xy = Math.max(next.xMax - next.xMin, next.yMax - next.yMin, 1)
  const widen = (min: number, max: number, fraction: number): [number, number] => {
    const mid = (min + max) / 2
    return [mid - xy * fraction, mid + xy * fraction]
  }
  if (next.zMax - next.zMin < xy * 0.04) {
    const [zMin, zMax] = widen(next.zMin, next.zMax, 0.2)
    next.zMin = zMin
    next.zMax = zMax
  }
  if (next.xMax - next.xMin < xy * 0.02) {
    const [xMin, xMax] = widen(next.xMin, next.xMax, 0.15)
    next.xMin = xMin
    next.xMax = xMax
  }
  if (next.yMax - next.yMin < xy * 0.02) {
    const [yMin, yMax] = widen(next.yMin, next.yMax, 0.15)
    next.yMin = yMin
    next.yMax = yMax
  }
  return next
}

function spanOf(min: number, max: number): number {
  const span = max - min
  return span > 1e-9 ? span : 1
}

/** Y is up and is math z. X is math x. Z is math y, matching the 2D ground plane. */
export function toWorld(frame: PlotFrame, x: number, y: number, z: number): { x: number; y: number; z: number } {
  if (frame.equal) return { x, y: z, z: y }
  const { box } = frame
  return {
    x: ((x - (box.xMin + box.xMax) / 2) / spanOf(box.xMin, box.xMax)) * CUBE,
    y: ((z - (box.zMin + box.zMax) / 2) / spanOf(box.zMin, box.zMax)) * CUBE,
    z: ((y - (box.yMin + box.yMax) / 2) / spanOf(box.yMin, box.yMax)) * CUBE,
  }
}

export function fromWorld(frame: PlotFrame, x: number, y: number, z: number): { x: number; y: number; z: number } {
  if (frame.equal) return { x, y: z, z: y }
  const { box } = frame
  return {
    x: (x / CUBE) * spanOf(box.xMin, box.xMax) + (box.xMin + box.xMax) / 2,
    y: (z / CUBE) * spanOf(box.yMin, box.yMax) + (box.yMin + box.yMax) / 2,
    z: (y / CUBE) * spanOf(box.zMin, box.zMax) + (box.zMin + box.zMax) / 2,
  }
}

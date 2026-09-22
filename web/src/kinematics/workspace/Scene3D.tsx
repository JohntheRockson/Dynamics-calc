import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import type { FigureBody, FigureSurface } from './evaluate'
import { CUBE, emptyBox, expandPlotBox, fromWorld, toWorld, type PlotBox, type PlotFrame } from './math/plotFrame'

function colorHex(color: string): number {
  const hex = Number.parseInt(color.replace('#', ''), 16)
  return Number.isFinite(hex) ? hex : 0x29d3f5
}

function surfaceGrids(surface: FigureSurface): { x: number; y: number; z: number }[][][] {
  return surface.sheets && surface.sheets.length > 0 ? surface.sheets : [surface.grid]
}

function place(frame: PlotFrame, x: number, y: number, z: number): THREE.Vector3 {
  const point = toWorld(frame, x, y, z)
  return new THREE.Vector3(point.x, point.y, point.z)
}

function addSurfaceGrid(content: THREE.Group, frame: PlotFrame, grid: { x: number; y: number; z: number }[][], colorName: string, flat: boolean): void {
  const rows = grid.length
  const cols = grid[0]?.length ?? 0
  if (rows < 2 || cols < 2) return
  const positions: number[] = []
  const heights: number[] = []
  const indexOf: number[][] = []
  let zMin = Infinity
  let zMax = -Infinity
  for (let row = 0; row < rows; row += 1) {
    indexOf[row] = []
    for (let col = 0; col < cols; col += 1) {
      const point = grid[row][col]
      if (!point || !Number.isFinite(point.x) || !Number.isFinite(point.y) || !Number.isFinite(point.z)) {
        indexOf[row][col] = -1
        continue
      }
      const placed = place(frame, point.x, point.y, point.z)
      indexOf[row][col] = positions.length / 3
      positions.push(placed.x, placed.y, placed.z)
      heights.push(point.z)
      zMin = Math.min(zMin, point.z)
      zMax = Math.max(zMax, point.z)
    }
  }
  const indices: number[] = []
  const wires: number[] = []
  const pushWire = (a: number, b: number) => {
    if (a < 0 || b < 0) return
    wires.push(positions[a * 3], positions[a * 3 + 1], positions[a * 3 + 2], positions[b * 3], positions[b * 3 + 1], positions[b * 3 + 2])
  }
  const stride = Math.max(1, Math.floor((Math.max(rows, cols) - 1) / 12))
  for (let row = 0; row < rows - 1; row += 1) {
    for (let col = 0; col < cols - 1; col += 1) {
      const a = indexOf[row][col]
      const b = indexOf[row][col + 1]
      const d = indexOf[row + 1][col]
      const e = indexOf[row + 1][col + 1]
      if (a < 0 || b < 0 || d < 0 || e < 0) continue
      indices.push(a, d, b, b, d, e)
      if (row % stride === 0) pushWire(a, b)
      if (col % stride === 0) pushWire(a, d)
    }
  }
  for (let col = 0; col < cols - 1; col += 1) pushWire(indexOf[rows - 1][col], indexOf[rows - 1][col + 1])
  for (let row = 0; row < rows - 1; row += 1) pushWire(indexOf[row][cols - 1], indexOf[row + 1][cols - 1])
  if (indices.length === 0) return
  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
  geometry.setIndex(indices)
  geometry.computeVertexNormals()
  const base = new THREE.Color(colorHex(colorName))
  const shaded = !flat && zMax > zMin
  if (shaded) {
    const low = new THREE.Color('#3b82f6')
    const high = new THREE.Color('#fbbf24')
    const colors = new Float32Array(heights.length * 3)
    heights.forEach((height, index) => {
      const t = (height - zMin) / (zMax - zMin)
      const tone = t < 0.55 ? low.clone().lerp(base, t / 0.55) : base.clone().lerp(high, (t - 0.55) / 0.45)
      colors[index * 3] = tone.r
      colors[index * 3 + 1] = tone.g
      colors[index * 3 + 2] = tone.b
    })
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3))
  }
  content.add(new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({
    color: shaded ? '#ffffff' : base,
    vertexColors: shaded,
    roughness: 0.55,
    metalness: 0.04,
    side: THREE.DoubleSide,
    polygonOffset: true,
    polygonOffsetFactor: 1,
    polygonOffsetUnits: 1,
  })))
  if (wires.length > 0) {
    const lines = new THREE.BufferGeometry()
    lines.setAttribute('position', new THREE.Float32BufferAttribute(wires, 3))
    content.add(new THREE.LineSegments(lines, new THREE.LineBasicMaterial({ color: flat ? base : 0xe7eef6, transparent: true, opacity: flat ? 0.35 : 0.22 })))
  }
}

function niceStep(span: number): number {
  if (!Number.isFinite(span) || span <= 0) return 1
  const raw = span / 5
  const pow = 10 ** Math.floor(Math.log10(Math.max(raw, 1e-9)))
  const n = raw / pow
  const nice = n < 1.5 ? 1 : n < 3.5 ? 2 : n < 7.5 ? 5 : 10
  return nice * pow
}

function formatTick(value: number): string {
  if (!Number.isFinite(value) || Math.abs(value) < 1e-9) return '0'
  const abs = Math.abs(value)
  if (abs >= 1000 || abs < 0.01) return value.toExponential(1)
  const digits = abs >= 100 ? 0 : abs >= 10 ? 1 : 2
  return String(Number(value.toFixed(digits)))
}

function tickValues(min: number, max: number): number[] {
  const step = niceStep(max - min)
  if (!Number.isFinite(step) || step <= 0) return []
  const start = Math.ceil(min / step - 1e-6) * step
  const values: number[] = []
  for (let i = 0; i < 8; i += 1) {
    const value = start + i * step
    if (value < min - step * 1e-4) continue
    if (value > max + step * 1e-4) break
    values.push(Math.abs(value) <= step * 1e-6 ? 0 : value)
  }
  return values
}

function nearer(cameraValue: number, low: number, high: number): number {
  return Math.abs(cameraValue - low) <= Math.abs(cameraValue - high) ? low : high
}

function addBox(content: THREE.Group, frame: PlotFrame): void {
  const { xMin, xMax, yMin, yMax, zMin, zMax } = frame.box
  const corner = (x: number, y: number, z: number) => place(frame, x, y, z)
  const pairs: [THREE.Vector3, THREE.Vector3][] = [
    [corner(xMin, yMin, zMin), corner(xMax, yMin, zMin)],
    [corner(xMax, yMin, zMin), corner(xMax, yMax, zMin)],
    [corner(xMax, yMax, zMin), corner(xMin, yMax, zMin)],
    [corner(xMin, yMax, zMin), corner(xMin, yMin, zMin)],
    [corner(xMin, yMin, zMax), corner(xMax, yMin, zMax)],
    [corner(xMax, yMin, zMax), corner(xMax, yMax, zMax)],
    [corner(xMax, yMax, zMax), corner(xMin, yMax, zMax)],
    [corner(xMin, yMax, zMax), corner(xMin, yMin, zMax)],
    [corner(xMin, yMin, zMin), corner(xMin, yMin, zMax)],
    [corner(xMax, yMin, zMin), corner(xMax, yMin, zMax)],
    [corner(xMax, yMax, zMin), corner(xMax, yMax, zMax)],
    [corner(xMin, yMax, zMin), corner(xMin, yMax, zMax)],
  ]
  content.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(pairs.flat()), new THREE.LineBasicMaterial({ color: 0xb7c6d4, transparent: true, opacity: 0.9 })))
}

function addFloor(content: THREE.Group, frame: PlotFrame): void {
  const { box } = frame
  const origin = place(frame, box.xMin, box.yMin, box.zMin)
  const across = place(frame, box.xMax, box.yMin, box.zMin)
  const depth = place(frame, box.xMin, box.yMax, box.zMin)
  const divisions = 10
  const points: THREE.Vector3[] = []
  for (let i = 0; i <= divisions; i += 1) {
    const t = i / divisions
    const x = origin.x + (across.x - origin.x) * t
    const z = origin.z + (depth.z - origin.z) * t
    points.push(new THREE.Vector3(x, origin.y, origin.z), new THREE.Vector3(x, origin.y, depth.z))
    points.push(new THREE.Vector3(origin.x, origin.y, z), new THREE.Vector3(across.x, origin.y, z))
  }
  content.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(points), new THREE.LineBasicMaterial({ color: 0x4c6278, transparent: true, opacity: 0.9 })))
  const width = Math.max(Math.abs(across.x - origin.x), 1e-4)
  const depthSpan = Math.max(Math.abs(depth.z - origin.z), 1e-4)
  const pad = new THREE.Mesh(
    new THREE.PlaneGeometry(width, depthSpan),
    new THREE.MeshStandardMaterial({ color: 0x101820, roughness: 1, metalness: 0, side: THREE.DoubleSide }),
  )
  pad.rotation.x = -Math.PI / 2
  pad.position.set((origin.x + across.x) / 2, origin.y - Math.max(width, depthSpan) * 0.002, (origin.z + depth.z) / 2)
  content.add(pad)
}

interface ScreenLabel {
  text: string
  axis: 'x' | 'y' | 'z'
  kind: 'name' | 'tick'
  x: number
  y: number
  z: number
}

function labelsForBox(box: PlotBox, camera: { x: number; y: number; z: number }): ScreenLabel[] {
  const yOnX = nearer(camera.y, box.yMin, box.yMax)
  const zOnX = nearer(camera.z, box.zMin, box.zMax)
  const xOnY = nearer(camera.x, box.xMin, box.xMax)
  const zOnY = nearer(camera.z, box.zMin, box.zMax)
  const xOnZ = nearer(camera.x, box.xMin, box.xMax)
  const yOnZ = nearer(camera.y, box.yMin, box.yMax)
  const labels: ScreenLabel[] = []
  for (const value of tickValues(box.xMin, box.xMax)) labels.push({ text: formatTick(value), axis: 'x', kind: 'tick', x: value, y: yOnX, z: zOnX })
  for (const value of tickValues(box.yMin, box.yMax)) labels.push({ text: formatTick(value), axis: 'y', kind: 'tick', x: xOnY, y: value, z: zOnY })
  for (const value of tickValues(box.zMin, box.zMax)) labels.push({ text: formatTick(value), axis: 'z', kind: 'tick', x: xOnZ, y: yOnZ, z: value })
  labels.push({ text: 'x', axis: 'x', kind: 'name', x: box.xMax, y: yOnX, z: zOnX })
  labels.push({ text: 'y', axis: 'y', kind: 'name', x: xOnY, y: box.yMax, z: zOnY })
  labels.push({ text: 'z', axis: 'z', kind: 'name', x: xOnZ, y: yOnZ, z: box.zMax })
  return labels
}

function syncLabels(container: HTMLDivElement, camera: THREE.PerspectiveCamera, frame: PlotFrame): void {
  const width = container.clientWidth
  const height = container.clientHeight
  if (width === 0 || height === 0) return
  const { box } = frame
  const cam = fromWorld(frame, camera.position.x, camera.position.y, camera.position.z)
  const center = place(frame, (box.xMin + box.xMax) / 2, (box.yMin + box.yMax) / 2, (box.zMin + box.zMax) / 2).project(camera)
  const cx = (center.x * 0.5 + 0.5) * width
  const cy = (-center.y * 0.5 + 0.5) * height
  const placed: { x: number; y: number; text: string; axis: string; kind: string }[] = []
  const names: { x: number; y: number; text: string; axis: string; kind: string }[] = []
  const ticks: { x: number; y: number; text: string; axis: string; kind: string }[] = []
  const projected = new THREE.Vector3()
  for (const label of labelsForBox(box, cam)) {
    const world = toWorld(frame, label.x, label.y, label.z)
    projected.set(world.x, world.y, world.z).project(camera)
    if (projected.z < -1 || projected.z > 1) continue
    const onScreen = projected.x >= -1 && projected.x <= 1 && projected.y >= -1 && projected.y <= 1
    const nearScreen = Math.abs(projected.x) < 1.2 && Math.abs(projected.y) < 1.2
    if (label.kind === 'tick' && !onScreen) continue
    if (label.kind === 'name' && !nearScreen) continue
    let x = (projected.x * 0.5 + 0.5) * width
    let y = (-projected.y * 0.5 + 0.5) * height
    const dx = x - cx
    const dy = y - cy
    const len = Math.hypot(dx, dy) || 1
    x += (dx / len) * (label.kind === 'name' ? 22 : 12)
    y += (dy / len) * (label.kind === 'name' ? 18 : 10)
    if (label.kind === 'name') {
      x = Math.min(width - 18, Math.max(18, x))
      y = Math.min(height - 16, Math.max(16, y))
      names.push({ x, y, text: label.text, axis: label.axis, kind: label.kind })
    } else if (x > 10 && y > 10 && x < width - 10 && y < height - 10) {
      ticks.push({ x, y, text: label.text, axis: label.axis, kind: label.kind })
    }
  }
  for (const name of names) {
    let guard = 0
    while (placed.some((item) => Math.hypot(item.x - name.x, item.y - name.y) < 22) && guard < 6) {
      name.x = Math.min(width - 18, name.x + 18)
      name.y = Math.min(height - 16, name.y + 8)
      guard += 1
    }
    placed.push(name)
  }
  for (const tick of ticks) {
    if (placed.some((item) => Math.hypot(item.x - tick.x, item.y - tick.y) < 22)) continue
    placed.push(tick)
  }
  while (container.childElementCount < placed.length) {
    const node = document.createElement('div')
    node.className = 'scene3d-label'
    container.appendChild(node)
  }
  while (container.childElementCount > placed.length) container.lastElementChild?.remove()
  placed.forEach((item, index) => {
    const node = container.children[index] as HTMLDivElement
    node.textContent = item.text
    node.dataset.axis = item.axis
    node.dataset.kind = item.kind
    node.style.transform = `translate(${item.x}px, ${item.y}px) translate(-50%, -50%)`
  })
}

function formatReadout(value: number): string {
  if (!Number.isFinite(value)) return '—'
  const abs = Math.abs(value)
  if (abs >= 1000 || (abs > 0 && abs < 0.01)) return value.toExponential(2)
  return value.toFixed(abs < 10 ? 3 : 2)
}

function disposeObject(obj: THREE.Object3D): void {
  const mesh = obj as THREE.Mesh
  mesh.geometry?.dispose()
  const materials = Array.isArray(mesh.material) ? mesh.material : mesh.material ? [mesh.material] : []
  for (const material of materials) {
    const mapped = material as THREE.Material & { map?: THREE.Texture | null }
    mapped.map?.dispose()
    material.dispose()
  }
}

function boundsOf(bodies: FigureBody[], surfaces: FigureSurface[]): PlotBox | null {
  const box = { xMin: Infinity, xMax: -Infinity, yMin: Infinity, yMax: -Infinity, zMin: Infinity, zMax: -Infinity }
  const include = (x: number, y: number, z: number) => {
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) return
    box.xMin = Math.min(box.xMin, x)
    box.xMax = Math.max(box.xMax, x)
    box.yMin = Math.min(box.yMin, y)
    box.yMax = Math.max(box.yMax, y)
    box.zMin = Math.min(box.zMin, z)
    box.zMax = Math.max(box.zMax, z)
  }
  for (const body of bodies) {
    for (const point of body.path) include(point.x, point.y, point.z)
  }
  for (const surface of surfaces) {
    for (const grid of surfaceGrids(surface)) {
      for (const row of grid) {
        for (const point of row) include(point.x, point.y, point.z)
      }
    }
  }
  return Number.isFinite(box.xMin) ? box : null
}

export function Scene3D({ bodies, surfaces = [], fitKey }: { bodies: FigureBody[]; surfaces?: FigureSurface[]; fitKey: string }) {
  const mountRef = useRef<HTMLDivElement>(null)
  const contentRef = useRef<THREE.Group | null>(null)
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null)
  const controlsRef = useRef<OrbitControls | null>(null)
  const fittedRef = useRef('')
  const labelsRef = useRef<HTMLDivElement>(null)
  const frameRef = useRef<PlotFrame>({ equal: true, box: emptyBox() })
  const spanRef = useRef(10)
  const readoutRef = useRef('')
  const [readout, setReadout] = useState('x —    y —    z —')

  useEffect(() => {
    const mount = mountRef.current
    if (!mount) return

    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(42, 1, 0.05, 5000)
    camera.position.set(8, 6, 8)
    cameraRef.current = camera

    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))
    renderer.setClearColor(0x111926, 1)
    mount.appendChild(renderer.domElement)

    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.08
    controlsRef.current = controls

    scene.add(new THREE.HemisphereLight(0xf4f7fb, 0x243040, 0.72))
    scene.add(new THREE.AmbientLight(0xd5e0ec, 0.38))
    const key = new THREE.DirectionalLight(0xffffff, 1.15)
    key.position.set(8, 14, 6)
    scene.add(key)
    const fill = new THREE.DirectionalLight(0xd5e4f8, 0.55)
    fill.position.set(-10, 6, -8)
    scene.add(fill)

    const content = new THREE.Group()
    scene.add(content)
    contentRef.current = content

    const raycaster = new THREE.Raycaster()
    const pointer = new THREE.Vector2()
    const ground = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0)
    const onMove = (event: MouseEvent) => {
      const rect = mount.getBoundingClientRect()
      if (rect.width === 0 || rect.height === 0) return
      pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1
      pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1
      raycaster.setFromCamera(pointer, camera)
      const frame = frameRef.current
      const groundY = toWorld(frame, 0, 0, 0).y
      ground.constant = -groundY
      const root = contentRef.current
      let point: THREE.Vector3 | null = null
      if (root) {
        const hit = raycaster.intersectObjects(root.children, true).find((item) => item.object instanceof THREE.Mesh)
        if (hit) point = hit.point
      }
      if (!point) {
        const target = new THREE.Vector3()
        if (raycaster.ray.intersectPlane(ground, target) && target.clone().sub(raycaster.ray.origin).dot(raycaster.ray.direction) > 0) {
          if (target.distanceTo(camera.position) < spanRef.current * 40) point = target
        }
      }
      if (!point) return
      const math = fromWorld(frame, point.x, point.y, point.z)
      const text = `x ${formatReadout(math.x)}    y ${formatReadout(math.y)}    z ${formatReadout(math.z)}`
      if (text === readoutRef.current) return
      readoutRef.current = text
      setReadout(text)
    }
    mount.addEventListener('pointermove', onMove)

    const resize = () => {
      const width = mount.clientWidth
      const height = mount.clientHeight
      if (width === 0 || height === 0) return
      camera.aspect = width / height
      camera.updateProjectionMatrix()
      renderer.setSize(width, height)
    }
    resize()
    const observer = new ResizeObserver(resize)
    observer.observe(mount)

    let frameId = 0
    const tick = () => {
      controls.update()
      renderer.render(scene, camera)
      const labels = labelsRef.current
      if (labels) syncLabels(labels, camera, frameRef.current)
      frameId = requestAnimationFrame(tick)
    }
    tick()

    return () => {
      cancelAnimationFrame(frameId)
      observer.disconnect()
      mount.removeEventListener('pointermove', onMove)
      controls.dispose()
      renderer.dispose()
      mount.removeChild(renderer.domElement)
      contentRef.current = null
      cameraRef.current = null
      controlsRef.current = null
      fittedRef.current = ''
    }
  }, [])

  useEffect(() => {
    const content = contentRef.current
    const camera = cameraRef.current
    const controls = controlsRef.current
    if (!content || !camera || !controls) return

    for (const child of [...content.children]) {
      content.remove(child)
      child.traverse(disposeObject)
    }

    const measured = boundsOf(bodies, surfaces)
    const equal = bodies.some((body) => body.role !== 'plot')
    const frame: PlotFrame = { equal, box: measured ? expandPlotBox(measured) : emptyBox() }
    frameRef.current = frame
    const { box } = frame
    const span = frame.equal ? Math.max(box.xMax - box.xMin, box.yMax - box.yMin, box.zMax - box.zMin, 1) : CUBE
    spanRef.current = span
    addFloor(content, frame)
    addBox(content, frame)

    bodies.forEach((body) => {
      const color = colorHex(body.color)
      let segment: THREE.Vector3[] = []
      const flush = () => {
        if (segment.length > 1) {
          const geometry = new THREE.BufferGeometry().setFromPoints(segment)
          const material = body.dashed
            ? new THREE.LineDashedMaterial({ color, dashSize: span * 0.03, gapSize: span * 0.02 })
            : new THREE.LineBasicMaterial({ color })
          const line = new THREE.Line(geometry, material)
          if (body.dashed) line.computeLineDistances()
          content.add(line)
        }
        segment = []
      }
      for (const point of body.path) {
        if (!Number.isFinite(point.x) || !Number.isFinite(point.y) || !Number.isFinite(point.z)) flush()
        else segment.push(place(frame, point.x, point.y, point.z))
      }
      flush()
      if (!body.hideMarker) {
        const sample = body.path[Math.min(body.index, body.path.length - 1)]
        if (!sample || !Number.isFinite(sample.x) || !Number.isFinite(sample.y) || !Number.isFinite(sample.z)) return
        const at = place(frame, sample.x, sample.y, sample.z)
        const marker = new THREE.Mesh(new THREE.SphereGeometry(span * 0.018, 16, 12), new THREE.MeshStandardMaterial({ color, roughness: 0.4, metalness: 0.15 }))
        marker.position.copy(at)
        content.add(marker)
        if (body.velocity) {
          const direction = place(frame, sample.x + body.velocity.x, sample.y + body.velocity.y, sample.z + body.velocity.z).sub(at)
          if (direction.length() > 1e-6) {
            direction.normalize()
            content.add(new THREE.ArrowHelper(direction, at, span * 0.16, color, span * 0.05, span * 0.03))
          }
        }
      }
    })

    for (const surface of surfaces) {
      for (const grid of surfaceGrids(surface)) addSurfaceGrid(content, frame, grid, surface.color, Boolean(surface.flat))
    }

    controls.minDistance = span * 0.2
    controls.maxDistance = span * 12
    if (fittedRef.current !== fitKey) {
      fittedRef.current = fitKey
      const mid = toWorld(frame, (box.xMin + box.xMax) / 2, (box.yMin + box.yMax) / 2, (box.zMin + box.zMax) / 2)
      const center = new THREE.Vector3(mid.x, mid.y, mid.z)
      controls.target.copy(center)
      camera.position.copy(center).add(new THREE.Vector3(span * 1.35, span * 0.92, span * 0.95))
      camera.near = Math.max(span / 800, 0.01)
      camera.far = span * 40
      camera.updateProjectionMatrix()
      controls.update()
    }
  }, [bodies, surfaces, fitKey])

  return (
    <div className="workspace-scene3d">
      <div ref={mountRef} className="workspace-scene3d-canvas" />
      <div ref={labelsRef} className="scene3d-labels" />
      <div className="scene3d-readout">{readout}</div>
    </div>
  )
}

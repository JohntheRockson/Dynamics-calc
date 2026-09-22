import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import type { FigureBody, FigureSurface } from './evaluate'

function colorHex(color: string): number {
  const hex = Number.parseInt(color.replace('#', ''), 16)
  return Number.isFinite(hex) ? hex : 0x29d3f5
}

/** Math (x, y, z) with z up, so the x-y plane matches the 2D figure. */
function toThree(x: number, y: number, z: number): THREE.Vector3 {
  return new THREE.Vector3(x, z, y)
}

function surfaceGrids(surface: FigureSurface): { x: number; y: number; z: number }[][][] {
  return surface.sheets && surface.sheets.length > 0 ? surface.sheets : [surface.grid]
}

function addSurfaceGrid(content: THREE.Group, grid: { x: number; y: number; z: number }[][], colorName: string): void {
  const rows = grid.length
  const cols = grid[0]?.length ?? 0
  if (rows < 2 || cols < 2) return
  const positions: number[] = []
  const indexOf: number[][] = []
  for (let row = 0; row < rows; row += 1) {
    indexOf[row] = []
    for (let col = 0; col < cols; col += 1) {
      const point = grid[row][col]
      if (!point || !Number.isFinite(point.x) || !Number.isFinite(point.y) || !Number.isFinite(point.z)) {
        indexOf[row][col] = -1
        continue
      }
      const placed = toThree(point.x, point.y, point.z)
      indexOf[row][col] = positions.length / 3
      positions.push(placed.x, placed.y, placed.z)
    }
  }
  const indices: number[] = []
  for (let row = 0; row < rows - 1; row += 1) {
    for (let col = 0; col < cols - 1; col += 1) {
      const a = indexOf[row][col]
      const b = indexOf[row][col + 1]
      const d = indexOf[row + 1][col]
      const e = indexOf[row + 1][col + 1]
      if (a < 0 || b < 0 || d < 0 || e < 0) continue
      indices.push(a, d, b, b, d, e)
    }
  }
  if (indices.length === 0) return
  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
  geometry.setIndex(indices)
  geometry.computeVertexNormals()
  const color = colorHex(colorName)
  content.add(new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({ color, transparent: true, opacity: 0.72, side: THREE.DoubleSide, roughness: 0.55, metalness: 0.05 })))
  content.add(new THREE.Mesh(geometry.clone(), new THREE.MeshBasicMaterial({ color, wireframe: true, transparent: true, opacity: 0.4 })))
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

function textSprite(text: string, color: string): THREE.Sprite {
  const canvas = document.createElement('canvas')
  canvas.width = 256
  canvas.height = 128
  const ctx = canvas.getContext('2d')
  if (ctx) {
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    ctx.fillStyle = color
    ctx.font = '700 80px sans-serif'
    ctx.textAlign = 'center'
    ctx.textBaseline = 'middle'
    ctx.fillText(text, canvas.width / 2, canvas.height / 2)
  }
  const texture = new THREE.CanvasTexture(canvas)
  texture.needsUpdate = true
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false }))
  sprite.renderOrder = 2
  return sprite
}

function expandRange(min: number, max: number, span: number): [number, number] {
  if (!Number.isFinite(min) || !Number.isFinite(max) || max - min < Math.max(span, 1) * 0.02) {
    const mid = Number.isFinite(min) && Number.isFinite(max) ? (min + max) / 2 : 0
    const half = Math.max(span * 0.45, 1)
    return [mid - half, mid + half]
  }
  const pad = (max - min) * 0.06
  return [min - pad, max + pad]
}

function tickValues(min: number, max: number): number[] {
  const step = niceStep(max - min)
  if (!Number.isFinite(step) || step <= 0) return []
  const start = Math.ceil((min + step * 1e-6) / step) * step
  const values: number[] = []
  for (let i = 0; i < 8; i += 1) {
    const value = start + i * step
    if (value > max + step * 1e-4) break
    if (Math.abs(value) <= step * 0.15) continue
    values.push(value)
  }
  return values
}

function addAxes(content: THREE.Group, min: THREE.Vector3, max: THREE.Vector3, span: number, empty: boolean): void {
  const source = empty
    ? { xMin: -10, xMax: 10, yMin: -10, yMax: 10, zMin: -10, zMax: 10 }
    : { xMin: min.x, xMax: max.x, yMin: min.z, yMax: max.z, zMin: min.y, zMax: max.y }
  const [xMin, xMax] = expandRange(source.xMin, source.xMax, span)
  const [yMin, yMax] = expandRange(source.yMin, source.yMax, span)
  const [zMin, zMax] = expandRange(source.zMin, source.zMax, span)
  const x0 = 0 >= xMin && 0 <= xMax ? 0 : xMin
  const y0 = 0 >= yMin && 0 <= yMax ? 0 : yMin
  const z0 = 0 >= zMin && 0 <= zMax ? 0 : zMin
  const label = Math.max(span * 0.16, 0.7)
  const tick = label * 0.55
  const gap = Math.max(span * 0.07, 0.3)

  const line = (from: { x: number; y: number; z: number }, to: { x: number; y: number; z: number }, color: number) => {
    const start = toThree(from.x, from.y, from.z)
    const end = toThree(to.x, to.y, to.z)
    const direction = end.clone().sub(start)
    const length = direction.length()
    if (length < 1e-6) return
    const radius = Math.max(span * 0.008, 0.025)
    const mesh = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, length, 8), new THREE.MeshBasicMaterial({ color }))
    mesh.position.copy(start).add(end).multiplyScalar(0.5)
    mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction.normalize())
    content.add(mesh)
  }
  const place = (text: string, color: string, at: { x: number; y: number; z: number }, scale: number) => {
    const sprite = textSprite(text, color)
    sprite.position.copy(toThree(at.x, at.y, at.z))
    sprite.scale.set(scale * 2.4, scale, 1)
    content.add(sprite)
  }

  line({ x: xMin, y: y0, z: z0 }, { x: xMax, y: y0, z: z0 }, 0xff7b7b)
  line({ x: x0, y: yMin, z: z0 }, { x: x0, y: yMax, z: z0 }, 0x59d67f)
  line({ x: x0, y: y0, z: zMin }, { x: x0, y: y0, z: zMax }, 0x7eb6ff)
  place('x', '#ffb4b4', { x: xMax + gap, y: y0, z: z0 }, label)
  place('y', '#9ee7b4', { x: x0, y: yMax + gap, z: z0 }, label)
  place('z', '#b7d4ff', { x: x0 + gap, y: y0, z: zMax + gap * 0.2 }, label)
  for (const value of tickValues(xMin, xMax)) place(formatTick(value), '#d5deea', { x: value, y: y0 - gap, z: z0 }, tick)
  for (const value of tickValues(yMin, yMax)) place(formatTick(value), '#d5deea', { x: x0 - gap, y: value, z: z0 }, tick)
  for (const value of tickValues(zMin, zMax)) place(formatTick(value), '#d5deea', { x: x0 + gap, y: y0, z: value }, tick)
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

export function Scene3D({ bodies, surfaces = [], fitKey }: { bodies: FigureBody[]; surfaces?: FigureSurface[]; fitKey: string }) {
  const mountRef = useRef<HTMLDivElement>(null)
  const contentRef = useRef<THREE.Group | null>(null)
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null)
  const controlsRef = useRef<OrbitControls | null>(null)
  const fittedRef = useRef('')
  const spanRef = useRef(10)
  const readoutRef = useRef('')
  const [readout, setReadout] = useState('x —    y —    z —')

  useEffect(() => {
    const mount = mountRef.current
    if (!mount) return

    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(45, 1, 0.05, 5000)
    camera.position.set(8, 6, 8)
    cameraRef.current = camera

    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))
    renderer.setClearColor(0x080b10, 1)
    mount.appendChild(renderer.domElement)

    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.08
    controlsRef.current = controls

    scene.add(new THREE.AmbientLight(0x9aabc0, 0.75))
    const key = new THREE.DirectionalLight(0xffffff, 1.15)
    key.position.set(6, 10, 4)
    scene.add(key)

    const content = new THREE.Group()
    scene.add(content)
    contentRef.current = content

    const raycaster = new THREE.Raycaster()
    const pointer = new THREE.Vector2()
    const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0)
    const onMove = (event: MouseEvent) => {
      const rect = mount.getBoundingClientRect()
      if (rect.width === 0 || rect.height === 0) return
      pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1
      pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1
      raycaster.setFromCamera(pointer, camera)
      const root = contentRef.current
      let point: THREE.Vector3 | null = null
      if (root) {
        const hit = raycaster.intersectObjects(root.children, true).find((item) => item.object instanceof THREE.Mesh)
        if (hit) point = hit.point
      }
      if (!point) {
        const target = new THREE.Vector3()
        if (raycaster.ray.intersectPlane(plane, target) && target.clone().sub(raycaster.ray.origin).dot(raycaster.ray.direction) > 0) {
          if (target.distanceTo(camera.position) < spanRef.current * 40) point = target
        }
      }
      if (!point) return
      const text = `x ${formatReadout(point.x)}    y ${formatReadout(point.z)}    z ${formatReadout(point.y)}`
      if (text === readoutRef.current) return
      readoutRef.current = text
      setReadout(text)
    }
    mount.addEventListener('pointermove', onMove)
    mount.addEventListener('mousemove', onMove)

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

    let frame = 0
    const tick = () => {
      controls.update()
      renderer.render(scene, camera)
      frame = requestAnimationFrame(tick)
    }
    tick()

    return () => {
      cancelAnimationFrame(frame)
      observer.disconnect()
      mount.removeEventListener('pointermove', onMove)
      mount.removeEventListener('mousemove', onMove)
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

    const min = new THREE.Vector3(Infinity, Infinity, Infinity)
    const max = new THREE.Vector3(-Infinity, -Infinity, -Infinity)
    const mapped = bodies.map((body) => body.path.map((point) => (Number.isFinite(point.x) && Number.isFinite(point.y) && Number.isFinite(point.z) ? toThree(point.x, point.y, point.z) : null)))
    for (const path of mapped) {
      for (const point of path) {
        if (!point) continue
        min.min(point)
        max.max(point)
      }
    }
    for (const surface of surfaces) {
      for (const grid of surfaceGrids(surface)) {
        for (const row of grid) {
          for (const point of row) {
            if (!Number.isFinite(point.x) || !Number.isFinite(point.y) || !Number.isFinite(point.z)) continue
            const placed = toThree(point.x, point.y, point.z)
            min.min(placed)
            max.max(placed)
          }
        }
      }
    }
    const empty = !Number.isFinite(min.x)
    const span = empty ? 10 : Math.max(max.x - min.x, max.y - min.y, max.z - min.z, 1)
    spanRef.current = span
    const grid = new THREE.GridHelper(span * 1.4, 12, 0x24303f, 0x1a2330)
    content.add(grid)
    addAxes(content, min, max, span, empty)

    mapped.forEach((path, index) => {
      const body = bodies[index]
      if (!body || path.length === 0) return
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
      for (const point of path) {
        if (!point) flush()
        else segment.push(point)
      }
      flush()
      if (!body.hideMarker) {
        const at = path[Math.min(body.index, path.length - 1)]
        if (!at) return
        const marker = new THREE.Mesh(new THREE.SphereGeometry(span * 0.018, 16, 12), new THREE.MeshStandardMaterial({ color, roughness: 0.4, metalness: 0.15 }))
        marker.position.copy(at)
        content.add(marker)
        if (body.velocity) {
          const direction = toThree(body.velocity.x, body.velocity.y, body.velocity.z)
          if (direction.length() > 1e-6) {
            direction.normalize()
            content.add(new THREE.ArrowHelper(direction, at, span * 0.16, color, span * 0.05, span * 0.03))
          }
        }
      }
    })

    for (const surface of surfaces) {
      for (const grid of surfaceGrids(surface)) addSurfaceGrid(content, grid, surface.color)
    }

    if (!empty) {
      const pad = span * 0.2
      min.x -= pad
      min.y -= pad
      min.z -= pad
      max.x += pad
      max.y += pad
      max.z += pad
    }
    if (fittedRef.current !== fitKey) {
      fittedRef.current = fitKey
      const center = empty ? new THREE.Vector3() : min.clone().add(max).multiplyScalar(0.5)
      controls.target.copy(center)
      camera.position.copy(center).add(new THREE.Vector3(span * 0.9, span * 0.7, span * 0.95))
      camera.near = Math.max(span / 200, 0.01)
      camera.far = span * 40
      camera.updateProjectionMatrix()
      controls.update()
    }
  }, [bodies, surfaces, fitKey])

  return (
    <div className="workspace-scene3d">
      <div ref={mountRef} className="workspace-scene3d-canvas" />
      <div className="scene3d-readout">{readout}</div>
    </div>
  )
}

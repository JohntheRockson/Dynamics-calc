import { useEffect, useRef } from 'react'
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

function addSurface(content: THREE.Group, surface: FigureSurface): void {
  const rows = surface.grid.length
  const cols = surface.grid[0]?.length ?? 0
  if (rows < 2 || cols < 2) return
  const positions: number[] = []
  const indexOf: number[][] = []
  for (let row = 0; row < rows; row += 1) {
    indexOf[row] = []
    for (let col = 0; col < cols; col += 1) {
      const point = surface.grid[row][col]
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
  const color = colorHex(surface.color)
  content.add(new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({ color, transparent: true, opacity: 0.72, side: THREE.DoubleSide, roughness: 0.55, metalness: 0.05 })))
  content.add(new THREE.Mesh(geometry.clone(), new THREE.MeshBasicMaterial({ color, wireframe: true, transparent: true, opacity: 0.4 })))
}

export function Scene3D({ bodies, surfaces = [], fitKey }: { bodies: FigureBody[]; surfaces?: FigureSurface[]; fitKey: string }) {
  const mountRef = useRef<HTMLDivElement>(null)
  const contentRef = useRef<THREE.Group | null>(null)
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null)
  const controlsRef = useRef<OrbitControls | null>(null)
  const fittedRef = useRef('')

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
      child.traverse((obj) => {
        const mesh = obj as THREE.Mesh
        mesh.geometry?.dispose()
        if (Array.isArray(mesh.material)) mesh.material.forEach((material) => material.dispose())
        else mesh.material?.dispose()
      })
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
      for (const row of surface.grid) {
        for (const point of row) {
          if (!Number.isFinite(point.x) || !Number.isFinite(point.y) || !Number.isFinite(point.z)) continue
          const placed = toThree(point.x, point.y, point.z)
          min.min(placed)
          max.max(placed)
        }
      }
    }
    const empty = !Number.isFinite(min.x)
    const span = empty ? 10 : Math.max(max.x - min.x, max.y - min.y, max.z - min.z, 1)
    const grid = new THREE.GridHelper(span * 1.4, 12, 0x24303f, 0x1a2330)
    content.add(grid)
    content.add(new THREE.AxesHelper(span * 0.22))

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

    for (const surface of surfaces) addSurface(content, surface)

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

  return <div ref={mountRef} className="workspace-scene3d" />
}

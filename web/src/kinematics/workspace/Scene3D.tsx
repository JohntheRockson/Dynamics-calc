import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import type { FigureBody } from './evaluate'

function colorHex(color: string): number {
  const hex = Number.parseInt(color.replace('#', ''), 16)
  return Number.isFinite(hex) ? hex : 0x29d3f5
}

/** Math (x, y, z) with z up, so the x-y plane matches the 2D figure. */
function toThree(x: number, y: number, z: number): THREE.Vector3 {
  return new THREE.Vector3(x, z, y)
}

export function Scene3D({ bodies, fitKey }: { bodies: FigureBody[]; fitKey: string }) {
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
    const mapped = bodies.map((body) => body.path.map((point) => toThree(point.x, point.y, point.z)))
    for (const path of mapped) {
      for (const point of path) {
        min.min(point)
        max.max(point)
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
      if (path.length > 1) {
        const geometry = new THREE.BufferGeometry().setFromPoints(path)
        const material = body.dashed
          ? new THREE.LineDashedMaterial({ color, dashSize: span * 0.03, gapSize: span * 0.02 })
          : new THREE.LineBasicMaterial({ color })
        const line = new THREE.Line(geometry, material)
        if (body.dashed) line.computeLineDistances()
        content.add(line)
      }
      if (!body.hideMarker) {
        const at = path[Math.min(body.index, path.length - 1)]
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
  }, [bodies, fitKey])

  return <div ref={mountRef} className="workspace-scene3d" />
}

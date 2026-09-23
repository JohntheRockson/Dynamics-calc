import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import type { SimLog } from '../../types'

interface Props {
  log: SimLog | null
  time: number
  showEstimate: boolean
}

interface SceneRefs {
  renderer: THREE.WebGLRenderer
  scene: THREE.Scene
  camera: THREE.PerspectiveCamera
  controls: OrbitControls
  body: THREE.Group
  estBody: THREE.Group
  ghost: THREE.Group
  raf: number
}

function buildSatellite(bodyColor: number, opacity: number, wireframe: boolean): THREE.Group {
  const group = new THREE.Group()
  const mat = new THREE.MeshStandardMaterial({
    color: bodyColor,
    transparent: opacity < 1,
    opacity,
    wireframe,
    roughness: 0.55,
    metalness: 0.35,
  })

  const hub = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.22, 0.22), mat)
  group.add(hub)

  const panelMat = new THREE.MeshStandardMaterial({
    color: 0x14314a,
    transparent: opacity < 1,
    opacity,
    wireframe,
    roughness: 0.35,
    metalness: 0.6,
    emissive: 0x0a2a44,
    emissiveIntensity: 0.25,
  })
  const panelGeo = new THREE.BoxGeometry(0.6, 0.02, 0.24)
  const panelL = new THREE.Mesh(panelGeo, panelMat)
  panelL.position.set(0, 0.32, 0)
  const panelR = new THREE.Mesh(panelGeo, panelMat)
  panelR.position.set(0, -0.32, 0)
  group.add(panelL, panelR)

  const antennaMat = new THREE.MeshStandardMaterial({ color: 0xd8dee9, transparent: opacity < 1, opacity, wireframe })
  const antenna = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.012, 0.22, 8), antennaMat)
  antenna.rotation.x = Math.PI / 2
  antenna.position.set(0, 0, 0.22)
  group.add(antenna)

  if (!wireframe) {
    const axisLen = 0.55
    const arrows: [THREE.Vector3, number][] = [
      [new THREE.Vector3(1, 0, 0), 0xfb6a6a],
      [new THREE.Vector3(0, 1, 0), 0x59d67f],
      [new THREE.Vector3(0, 0, 1), 0x5aa8ff],
    ]
    for (const [dir, color] of arrows) {
      group.add(new THREE.ArrowHelper(dir, new THREE.Vector3(0, 0, 0), axisLen, color, 0.09, 0.05))
    }
  }

  return group
}

function buildStarfield(): THREE.Points {
  const count = 1200
  const positions = new Float32Array(count * 3)
  for (let i = 0; i < count; i++) {
    const r = 18 + Math.random() * 14
    const theta = Math.random() * Math.PI * 2
    const phi = Math.acos(2 * Math.random() - 1)
    positions[i * 3] = r * Math.sin(phi) * Math.cos(theta)
    positions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta)
    positions[i * 3 + 2] = r * Math.cos(phi)
  }
  const geo = new THREE.BufferGeometry()
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3))
  const mat = new THREE.PointsMaterial({ color: 0x93a2b6, size: 0.045, sizeAttenuation: true })
  return new THREE.Points(geo, mat)
}

function buildInertialAxes(): THREE.Group {
  const group = new THREE.Group()
  const len = 1.1
  const mat = new THREE.LineBasicMaterial({ color: 0x33455c, transparent: true, opacity: 0.7 })
  const pairs: THREE.Vector3[][] = [
    [new THREE.Vector3(-len, 0, 0), new THREE.Vector3(len, 0, 0)],
    [new THREE.Vector3(0, -len, 0), new THREE.Vector3(0, len, 0)],
    [new THREE.Vector3(0, 0, -len), new THREE.Vector3(0, 0, len)],
  ]
  for (const pts of pairs) {
    const geo = new THREE.BufferGeometry().setFromPoints(pts)
    group.add(new THREE.Line(geo, mat))
  }
  return group
}

export function SatelliteViewer({ log, time, showEstimate }: Props) {
  const mountRef = useRef<HTMLDivElement | null>(null)
  const logRef = useRef<SimLog | null>(log)
  const timeRef = useRef<number>(time)
  const showEstRef = useRef<boolean>(showEstimate)

  useEffect(() => {
    logRef.current = log
  }, [log])
  useEffect(() => {
    timeRef.current = time
  }, [time])
  useEffect(() => {
    showEstRef.current = showEstimate
  }, [showEstimate])

  useEffect(() => {
    const mount = mountRef.current
    if (!mount) return

    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(45, 1, 0.05, 100)
    camera.position.set(1.7, 1.15, 1.9)

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setClearColor(0x05080c, 1)
    mount.appendChild(renderer.domElement)

    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.08
    controls.minDistance = 1.0
    controls.maxDistance = 12

    scene.add(new THREE.AmbientLight(0x8899aa, 0.55))
    const key = new THREE.DirectionalLight(0xffffff, 1.4)
    key.position.set(3, 4, 2)
    scene.add(key)
    const rim = new THREE.DirectionalLight(0x4fd1ff, 0.4)
    rim.position.set(-3, -2, -3)
    scene.add(rim)

    scene.add(buildStarfield())
    scene.add(buildInertialAxes())

    const body = buildSatellite(0x8b96a5, 1, false)
    scene.add(body)

    const estBody = buildSatellite(0xf5a524, 0.55, false)
    scene.add(estBody)

    const ghost = buildSatellite(0x29d3f5, 0.16, true)
    scene.add(ghost)

    const resize = () => {
      const w = mount.clientWidth
      const h = mount.clientHeight
      if (w === 0 || h === 0) return
      camera.aspect = w / h
      camera.updateProjectionMatrix()
      renderer.setSize(w, h)
    }
    resize()
    const ro = new ResizeObserver(resize)
    ro.observe(mount)

    const sceneRefs: SceneRefs = { renderer, scene, camera, controls, body, estBody, ghost, raf: 0 }

    const sampleIndex = (l: SimLog, t: number) => {
      const dt = l.resolved.dt
      const n = l.t.length
      const idx = Math.round(t / dt)
      return Math.min(n - 1, Math.max(0, idx))
    }

    const animate = () => {
      const l = logRef.current
      if (l && l.t.length > 0) {
        const idx = sampleIndex(l, timeRef.current)
        const q = l.q[idx]
        body.quaternion.set(q[1], q[2], q[3], q[0])
        body.visible = true

        const qDes = l.q_des
        ghost.quaternion.set(qDes[1], qDes[2], qDes[3], qDes[0])
        ghost.visible = true

        if (showEstRef.current && l.q_hat && l.resolved.estimator !== 'truth') {
          const qh = l.q_hat[idx]
          estBody.quaternion.set(qh[1], qh[2], qh[3], qh[0])
          estBody.visible = true
        } else {
          estBody.visible = false
        }
      } else {
        body.visible = false
        ghost.visible = false
        estBody.visible = false
      }
      controls.update()
      renderer.render(scene, camera)
      sceneRefs.raf = requestAnimationFrame(animate)
    }
    sceneRefs.raf = requestAnimationFrame(animate)

    return () => {
      cancelAnimationFrame(sceneRefs.raf)
      ro.disconnect()
      controls.dispose()
      renderer.dispose()
      mount.removeChild(renderer.domElement)
    }
  }, [])

  return <div ref={mountRef} className="viewer-canvas-wrap" />
}

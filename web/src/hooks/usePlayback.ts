import { useEffect, useRef, useState } from 'react'

export interface Playback {
  time: number
  playing: boolean
  speed: number
  duration: number
  play: () => void
  pause: () => void
  toggle: () => void
  seek: (t: number) => void
  reset: (newDuration?: number) => void
  setSpeed: (s: number) => void
}

/** Wall-clock-driven playback clock over `[0, duration]` seconds. */
export function usePlayback(initialDuration: number): Playback {
  const [duration, setDuration] = useState(initialDuration)
  const [time, setTime] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(1)
  const lastRef = useRef<number | null>(null)
  const rafRef = useRef<number | null>(null)

  useEffect(() => {
    if (!playing) {
      lastRef.current = null
      return
    }
    const tick = (now: number) => {
      if (lastRef.current !== null) {
        const dtSec = (now - lastRef.current) / 1000
        setTime((t) => {
          const next = t + dtSec * speed
          if (next >= duration) {
            setPlaying(false)
            return duration
          }
          return next
        })
      }
      lastRef.current = now
      rafRef.current = requestAnimationFrame(tick)
    }
    rafRef.current = requestAnimationFrame(tick)
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current)
    }
  }, [playing, speed, duration])

  return {
    time,
    playing,
    speed,
    duration,
    play: () => {
      setTime((t) => (t >= duration ? 0 : t))
      setPlaying(true)
    },
    pause: () => setPlaying(false),
    toggle: () =>
      setPlaying((p) => {
        if (!p) setTime((t) => (t >= duration ? 0 : t))
        return !p
      }),
    seek: (t: number) => setTime(Math.min(duration, Math.max(0, t))),
    reset: (newDuration?: number) => {
      setPlaying(false)
      setTime(0)
      if (newDuration !== undefined) setDuration(newDuration)
    },
    setSpeed,
  }
}

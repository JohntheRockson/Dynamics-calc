import type { Playback } from '../hooks/usePlayback'

const SPEEDS = [0.25, 0.5, 1, 2, 4, 8]

function fmtTime(t: number): string {
  return `${t.toFixed(2)}s`
}

interface Props {
  playback: Playback
  disabled: boolean
}

export function PlaybackBar({ playback, disabled }: Props) {
  const { time, playing, duration, speed, toggle, seek, reset, setSpeed } = playback
  return (
    <div className="panel">
      <div className="playback-bar">
        <button type="button" className="btn btn-icon" onClick={toggle} disabled={disabled} title={playing ? 'Pause' : 'Play'}>
          {playing ? '⏸' : '▶'}
        </button>
        <button type="button" className="btn btn-icon btn-ghost" onClick={() => reset()} disabled={disabled} title="Restart">
          ⏮
        </button>
        <span className="playback-time">
          {fmtTime(time)} / {fmtTime(duration)}
        </span>
        <input
          className="playback-scrub"
          id="playback-scrub"
          aria-label="Playback position"
          type="range"
          min={0}
          max={duration}
          step={duration / 1000 || 0.01}
          value={Math.min(time, duration)}
          disabled={disabled}
          onChange={(e) => seek(parseFloat(e.target.value))}
        />
        <select
          className="num-input speed-select"
          id="playback-speed"
          aria-label="Playback speed"
          value={speed}
          disabled={disabled}
          onChange={(e) => setSpeed(parseFloat(e.target.value))}
        >
          {SPEEDS.map((s) => (
            <option key={s} value={s}>
              {s}&times;
            </option>
          ))}
        </select>
      </div>
    </div>
  )
}

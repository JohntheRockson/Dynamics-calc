import { useState } from 'react'
import { CurvilinearPanel } from './CurvilinearPanel'
import { RectilinearPanel } from './RectilinearPanel'
import { RelativeMotionPanel } from './RelativeMotionPanel'
import { SolverPanel } from './SolverPanel'

type Tab = 'solver' | 'rectilinear' | 'curvilinear' | 'relative'

const TABS: { id: Tab; label: string; icon: string }[] = [
  { id: 'solver', label: 'Solver', icon: '🧮' },
  { id: 'rectilinear', label: 'Rectilinear motion', icon: '📏' },
  { id: 'curvilinear', label: 'Curvilinear motion', icon: '🎯' },
  { id: 'relative', label: 'Relative motion', icon: '🔀' },
]

export function KinematicsLab() {
  const [tab, setTab] = useState<Tab>('solver')

  return (
    <div className="kin-lab">
      <div className="kin-tabs">
        {TABS.map((t) => (
          <button key={t.id} type="button" className={`chart-tab ${tab === t.id ? 'active' : ''}`} onClick={() => setTab(t.id)} style={{ fontSize: 12.5, padding: '7px 14px' }}>
            {t.icon} {t.label}
          </button>
        ))}
      </div>
      {tab === 'solver' && <SolverPanel />}
      {tab === 'rectilinear' && <RectilinearPanel />}
      {tab === 'curvilinear' && <CurvilinearPanel />}
      {tab === 'relative' && <RelativeMotionPanel />}
    </div>
  )
}


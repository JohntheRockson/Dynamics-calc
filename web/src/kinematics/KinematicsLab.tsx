import { useState } from 'react'
import { SolverPanel } from './SolverPanel'
import { Workspace } from './workspace/Workspace'

type Tab = 'workspace' | 'solver'

const TABS: { id: Tab; label: string; icon: string }[] = [
  { id: 'workspace', label: 'Workspace', icon: '📝' },
  { id: 'solver', label: 'Solver', icon: '🧮' },
]

export function KinematicsLab() {
  const [tab, setTab] = useState<Tab>('workspace')

  return (
    <div className="kin-lab">
      <div className="kin-tabs">
        {TABS.map((t) => (
          <button key={t.id} type="button" className={`chart-tab ${tab === t.id ? 'active' : ''}`} onClick={() => setTab(t.id)} style={{ fontSize: 12.5, padding: '7px 14px' }}>
            {t.icon} {t.label}
          </button>
        ))}
      </div>
      {tab === 'workspace' ? (
        <Workspace />
      ) : (
        <div className="kin-scroll">
          <SolverPanel />
        </div>
      )}
    </div>
  )
}

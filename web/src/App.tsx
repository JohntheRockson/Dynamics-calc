import { lazy, Suspense, useState } from 'react'
import { AttitudeConsole } from './AttitudeConsole'

// KaTeX (equation rendering) pulls in a lot of font assets; split the whole
// Kinematics Lab into its own chunk so switching sections is the only thing
// that pays for it.
const KinematicsLab = lazy(() => import('./kinematics/KinematicsLab').then((m) => ({ default: m.KinematicsLab })))

type Section = 'attitude' | 'kinematics'

export default function App() {
  const [section, setSection] = useState<Section>('kinematics')

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="app-title">
          <span className="logo">{section === 'attitude' ? '\u{1F6F0}\u{FE0F}' : '\u{1F4D0}'}</span>
          <h1>{section === 'attitude' ? 'SimLab' : 'Dynamics Lab'}</h1>
          <span className="subtitle">{section === 'attitude' ? 'Attitude Control Console \u00b7 Rust engine' : 'Statements and figure'}</span>
        </div>
        <nav className="section-nav">
          <button type="button" className={section === 'kinematics' ? 'active' : ''} onClick={() => setSection('kinematics')}>
            Kinematics Lab
          </button>
          <button type="button" className={section === 'attitude' ? 'active' : ''} onClick={() => setSection('attitude')}>
            Attitude Console
          </button>
        </nav>
      </header>

      {section === 'attitude' ? (
        <AttitudeConsole />
      ) : (
        <div className="app-body-single">
          <Suspense fallback={<div className="hint" style={{ padding: 20 }}>Loading&hellip;</div>}>
            <KinematicsLab />
          </Suspense>
        </div>
      )}

      <p className="footer-note">
        {section === 'attitude'
          ? "Rust engine (`engine/`) cross-validated against the original Python reference (`legacy-python/`) — see the README for scope and parity notes."
          : 'Type math or a statement on the left. Results stay in the list, and graphs share the figure. A 3D graph switches the figure to 3D.'}
      </p>
    </div>
  )
}

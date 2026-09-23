import { useState, type ReactNode } from 'react'

interface SectionProps {
  title: string
  icon?: string
  defaultOpen?: boolean
  children: ReactNode
  rightSlot?: ReactNode
}

export function Section({ title, icon, defaultOpen = true, children, rightSlot }: SectionProps) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="section">
      <div className="section-header" onClick={() => setOpen((o) => !o)}>
        <span className="section-title">
          {icon && <span className="icon">{icon}</span>}
          {title}
        </span>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          {rightSlot && (
            <span onClick={(e) => e.stopPropagation()} style={{ display: 'flex' }}>
              {rightSlot}
            </span>
          )}
          <span className={`section-chevron ${open ? 'open' : ''}`}>&#9656;</span>
        </div>
      </div>
      {open && <div className="section-body">{children}</div>}
    </div>
  )
}

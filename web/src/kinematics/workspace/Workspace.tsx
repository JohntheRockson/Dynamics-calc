import { useEffect, useMemo, useState } from 'react'
import { Eq } from '../Eq'
import { usePlayback } from '../../hooks/usePlayback'
import { Composer } from './Composer'
import { commitCommand, exampleDocument, removeStatement, type ExampleId, type WorkspaceDocument } from './document'
import { compileDocument, viewAt, type RowModel } from './evaluate'
import { FigurePane } from './FigurePane'

const EXAMPLES: { id: ExampleId; label: string }[] = [
  { id: 'circular', label: 'Circular track' },
  { id: 'two-points', label: 'Two points' },
  { id: 'projectile', label: 'Projectile' },
  { id: 'helix', label: 'Helix' },
]

function RowView({ row, open, onToggle, onRemove }: { row: RowModel; open: boolean; onToggle: () => void; onRemove: () => void }) {
  const body = (
    <>
      <span className="statement-label">{row.label}</span>
      <span className="statement-value">
        {row.tex && <Eq tex={row.tex} />}
        {row.text && <span>{row.text}</span>}
      </span>
    </>
  )
  return (
    <li className={`statement-row is-${row.source}`}>
      {row.formula ? (
        <button type="button" className="statement-main" onClick={onToggle} aria-expanded={open}>
          {body}
        </button>
      ) : (
        <div className="statement-main">{body}</div>
      )}
      {row.statementId && (
        <button type="button" className="btn btn-ghost statement-remove" onClick={onRemove} aria-label={`Remove ${row.label}`}>
          Remove
        </button>
      )}
      {open && row.formula && (
        <div className="statement-formula">
          <Eq tex={row.formula} />
        </div>
      )}
    </li>
  )
}

export function Workspace() {
  const [doc, setDoc] = useState<WorkspaceDocument>(() => exampleDocument('circular'))
  const [openRow, setOpenRow] = useState<string | null>(null)
  const compiled = useMemo(() => compileDocument(doc), [doc])
  const playback = usePlayback(Math.max(compiled.duration, 0.001))

  useEffect(() => {
    playback.reset(Math.max(compiled.duration, 0.001))
    // Reset only when the problem's duration changes. `reset` is a fresh function each render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [compiled.duration])

  const time = compiled.duration > 0 ? Math.min(playback.time, compiled.duration) : 0
  const view = useMemo(() => viewAt(compiled, time), [compiled, time])

  return (
    <div className="workspace">
      <section className="workspace-console panel">
        <div className="panel-header">
          <h2>Statements</h2>
          <label className="workspace-example">
            <span>Example</span>
            <select
              className="num-input"
              defaultValue=""
              aria-label="Load example"
              onChange={(event) => {
                const id = event.target.value as ExampleId
                if (id) setDoc(exampleDocument(id))
                event.target.value = ''
              }}
            >
              <option value="">Load…</option>
              {EXAMPLES.map((example) => (
                <option key={example.id} value={example.id}>
                  {example.label}
                </option>
              ))}
            </select>
          </label>
        </div>
        <div className="workspace-statements">
          {view.blocks.length === 0 && <p className="hint">Type “point” below to start a problem.</p>}
          {view.blocks.map((block) => (
            <article key={block.id} className="statement-block" style={{ borderLeftColor: block.color }}>
              <h3>{block.title}</h3>
              {block.note && <p className="statement-note">{block.note}</p>}
              <ul>
                {block.rows.map((row) => (
                  <RowView
                    key={row.id}
                    row={row}
                    open={openRow === row.id}
                    onToggle={() => setOpenRow((current) => (current === row.id ? null : row.id))}
                    onRemove={() => {
                      if (row.statementId) setDoc((current) => removeStatement(current, row.statementId!))
                    }}
                  />
                ))}
              </ul>
            </article>
          ))}
          {compiled.duration > 0 && <p className="hint">Solved values follow the playback time.</p>}
        </div>
        <Composer
          doc={doc}
          onCommit={(commandId, args) => {
            const result = commitCommand(doc, commandId, args)
            if (result.error) return result.error
            setDoc(result.doc)
            return null
          }}
        />
      </section>
      <FigurePane view={view} playback={playback} />
    </div>
  )
}

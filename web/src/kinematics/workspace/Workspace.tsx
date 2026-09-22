import { useEffect, useMemo, useState } from 'react'
import { Eq } from '../Eq'
import { usePlayback } from '../../hooks/usePlayback'
import { Composer } from './Composer'
import { appendMath, commitCommand, emptyDocument, exampleDocument, removeStatement, setMathVisible, type ExampleId, type WorkspaceDocument } from './document'
import { validateMath } from './math/expr'
import { compileDocument, viewAt, type RowModel } from './evaluate'
import { FigurePane } from './FigurePane'

const EXAMPLES: { id: ExampleId; label: string }[] = [
  { id: 'circular', label: 'Circular track' },
  { id: 'two-points', label: 'Two points' },
  { id: 'projectile', label: 'Projectile' },
  { id: 'helix', label: 'Helix' },
]

function RowView({ row, open, decimal, onToggle, onRemove, onToggleVisible, onToggleDecimal, showRemove }: { row: RowModel; open: boolean; decimal: boolean; onToggle: () => void; onRemove: () => void; onToggleVisible: () => void; onToggleDecimal: () => void; showRemove: boolean }) {
  const typed = row.input
  const showTypedLabel = Boolean(typed && !row.plotKind)
  const canApproximate = Boolean(row.exactTex && row.approxTex && row.exactTex !== row.approxTex)
  const shownTex = decimal && row.approxTex ? row.approxTex : (row.exactTex ?? row.tex)
  const expandable = Boolean(row.formula || typed)
  const body = (
    <>
      <span className={showTypedLabel ? 'statement-label is-typed' : 'statement-label'} title={showTypedLabel ? typed : undefined}>
        {showTypedLabel ? typed : row.label}
      </span>
      <span className="statement-value">
        {shownTex && <Eq tex={shownTex} />}
        {row.text && row.showText !== false && <span>{row.text}</span>}
      </span>
    </>
  )
  return (
    <li className={`statement-row is-${row.source}`}>
      {expandable ? (
        <button type="button" className="statement-main" onClick={onToggle} aria-expanded={open}>
          {body}
        </button>
      ) : (
        <div className="statement-main">{body}</div>
      )}
      <div className="statement-actions">
        {canApproximate && (
          <button type="button" className="btn btn-ghost statement-eye" onClick={onToggleDecimal} aria-label={decimal ? 'Show the exact value' : 'Show a decimal approximation'}>
            {decimal ? 'Exact' : 'Decimal'}
          </button>
        )}
        {row.plotKind && (
          <button type="button" className="btn btn-ghost statement-eye" aria-pressed={row.visible !== false} onClick={onToggleVisible} aria-label={row.visible === false ? `Show ${row.label} on the figure` : `Hide ${row.label} on the figure`}>
            {row.visible === false ? 'Show' : 'Hide'}
          </button>
        )}
        {showRemove && (
          <button type="button" className="btn btn-ghost statement-remove" onClick={onRemove} aria-label={`Remove ${row.label}`}>
            Remove
          </button>
        )}
      </div>
      {open && expandable && (
        <div className="statement-formula">
          {typed && (
            <p className="statement-typed">
              <span>Typed</span> <code>{typed}</code>
            </p>
          )}
          {row.formula && <Eq tex={row.formula} />}
        </div>
      )}
    </li>
  )
}

export function Workspace() {
  const [doc, setDoc] = useState<WorkspaceDocument>(() => exampleDocument('circular'))
  const [openRow, setOpenRow] = useState<string | null>(null)
  const [decimal, setDecimal] = useState<Record<string, boolean>>({})
  const compiled = useMemo(() => compileDocument(doc), [doc])
  const playback = usePlayback(Math.max(compiled.duration, 0.001))

  useEffect(() => {
    playback.reset(Math.max(compiled.duration, 0.001))
    // Restart whenever the statements change, including a new example with the same duration.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [doc])

  const time = compiled.duration > 0 ? Math.min(playback.time, compiled.duration) : 0
  const view = useMemo(() => viewAt(compiled, time), [compiled, time])

  return (
    <div className="workspace">
      <section className="workspace-console panel">
        <div className="panel-header">
          <h2>Statements</h2>
          <div className="workspace-tools">
          <button type="button" className="btn btn-ghost workspace-clear" onClick={() => setDoc(emptyDocument())}>
            Clear
          </button>
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
        </div>
        <div className="workspace-statements">
          {view.blocks.length === 0 && <p className="hint">Type a calculation, or “point” to start a problem. Clear empties the list.</p>}
          {view.blocks.map((block) => (
            <article key={block.id} className="statement-block" style={{ borderLeftColor: block.color }}>
              <div className="statement-block-head">
                <h3>{block.title}</h3>
                {block.removeId && (
                  <button type="button" className="btn btn-ghost statement-remove" onClick={() => setDoc((current) => removeStatement(current, block.removeId))} aria-label={`Remove ${block.title}`}>
                    Remove
                  </button>
                )}
              </div>
              {block.note && <p className="statement-note">{block.note}</p>}
              <ul>
                {block.rows.map((row) => (
                  <RowView
                    key={row.id}
                    row={row}
                    open={openRow === row.id}
                    decimal={decimal[row.id] ?? Boolean(row.preferDecimal)}
                    onToggle={() => setOpenRow((current) => (current === row.id ? null : row.id))}
                    onRemove={() => {
                      if (row.statementId) setDoc((current) => removeStatement(current, row.statementId!))
                    }}
                    onToggleVisible={() => {
                      if (!row.statementId || !row.plotKind) return
                      setDoc((current) => setMathVisible(current, row.statementId!, row.visible === false))
                    }}
                    onToggleDecimal={() => {
                      const showing = decimal[row.id] ?? Boolean(row.preferDecimal)
                      setDecimal((current) => ({ ...current, [row.id]: !showing }))
                    }}
                    showRemove={row.statementId !== null && row.statementId !== block.removeId}
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
          onMath={(input) => {
            const message = validateMath(input)
            if (message) return message
            setDoc((current) => appendMath(current, input))
            return null
          }}
        />
      </section>
      <FigurePane view={view} playback={playback} />
    </div>
  )
}

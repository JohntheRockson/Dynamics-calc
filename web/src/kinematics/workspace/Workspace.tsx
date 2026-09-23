import { useEffect, useMemo, useRef, useState } from 'react'
import { Eq } from '../Eq'
import { usePlayback } from '../../hooks/usePlayback'
import { Composer } from './Composer'
import { appendMath, commitCommand, convertDocumentAngles, emptyDocument, exampleDocument, removeStatement, replaceMath, setMathVisible, type ExampleId, type WorkspaceDocument } from './document'
import { MathField } from './MathField'
import type { AngleMode } from './math/expr'
import { previewTex, validateMath } from './math/syntax'
import { compileDocument, viewAt, type RowModel } from './evaluate'
import { FigurePane } from './FigurePane'

const EXAMPLES: { id: ExampleId; label: string }[] = [
  { id: 'circular', label: 'Circular track' },
  { id: 'two-points', label: 'Two points' },
  { id: 'projectile', label: 'Projectile' },
  { id: 'helix', label: 'Helix' },
]

function unitTex(unit: string): string {
  if (unit.endsWith('²')) return `\\mathrm{${unit.slice(0, -1)}}^{2}`
  if (unit.endsWith('³')) return `\\mathrm{${unit.slice(0, -1)}}^{3}`
  return `\\mathrm{${unit}}`
}

/** A kinematics quantity such as `25 m/s` or `(0, 0) m`, drawn as math instead of a sentence. */
function quantityTex(text: string): string | null {
  const trimmed = text.trim()
  const tuple = /^\(([^)]+)\)\s+(\S+)$/.exec(trimmed)
  if (tuple) {
    const parts = tuple[1].split(',').map((part) => part.trim())
    if (parts.every((part) => /^-?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?$/.test(part))) return `\\left(${parts.join(', ')}\\right)\\,${unitTex(tuple[2])}`
  }
  const match = /^(-?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?|∞)\s+(\S+)$/.exec(trimmed)
  if (!match) return null
  const n = match[1] === '∞' ? '\\infty' : match[1]
  return `${n}\\,${unitTex(match[2])}`
}

function trailingNote(text: string): string | null {
  const match = /\s\(([^()]*)\)\s*$/.exec(text.trim())
  return match?.[1] ?? null
}

function copyLatex(value: string): void {
  const write = navigator.clipboard?.writeText(value)
  if (write) {
    void write.catch(() => copyLatexFallback(value))
    return
  }
  copyLatexFallback(value)
}

function copyLatexFallback(value: string): void {
  const area = document.createElement('textarea')
  area.value = value
  area.setAttribute('readonly', '')
  area.style.position = 'fixed'
  area.style.left = '-9999px'
  document.body.appendChild(area)
  area.select()
  document.execCommand('copy')
  area.remove()
}

function CopyIcon({ copied }: { copied: boolean }) {
  if (copied) {
    return (
      <svg viewBox="0 0 16 16" width="15" height="15" aria-hidden="true">
        <path d="M3.2 8.2 6.4 11.6 12.8 4.2" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    )
  }
  return (
    <svg viewBox="0 0 16 16" width="15" height="15" aria-hidden="true">
      <rect x="5.2" y="1.6" width="8.2" height="9.4" rx="1.2" fill="none" stroke="currentColor" strokeWidth="1.3" />
      <rect x="2.4" y="4.6" width="8.2" height="9.4" rx="1.2" fill="var(--panel)" stroke="currentColor" strokeWidth="1.3" />
    </svg>
  )
}

function MathRowEditor({ input, onCommit }: { input: string; onCommit: (value: string) => string | null }) {
  const [draft, setDraft] = useState(input)
  const [error, setError] = useState<string | null>(null)
  return (
    <>
      <MathField
        value={draft}
        label="Edit statement"
        math
        onValue={setDraft}
        onSubmit={(value) => {
          if (value.trim() === input.trim()) return
          setError(onCommit(value))
        }}
      />
      {error && <p className="statement-note">{error}</p>}
    </>
  )
}

function RowView({ row, open, decimal, plane, onToggle, onRemove, onToggleVisible, onToggleDecimal, onTogglePlane, onCommitMath, showRemove }: { row: RowModel; open: boolean; decimal: boolean; plane: boolean; onToggle: () => void; onRemove: () => void; onToggleVisible: () => void; onToggleDecimal: () => void; onTogglePlane: () => void; onCommitMath: (value: string) => string | null; showRemove: boolean }) {
  const [copied, setCopied] = useState<'input' | 'output' | null>(null)
  const typed = row.input
  const inputTex = typed ? previewTex(typed) : null
  const canApproximate = Boolean(row.exactTex && row.approxTex && row.exactTex !== row.approxTex)
  const shownTex = decimal && row.approxTex ? row.approxTex : (row.exactTex ?? row.tex)
  const quantity = quantityTex(row.text)
  const note = trailingNote(row.text)
  const expandable = Boolean(row.formula)
  const copy = (value: string, which: 'input' | 'output') => {
    copyLatex(value)
    setCopied(which)
    window.setTimeout(() => setCopied((current) => (current === which ? null : current)), 1200)
  }
  const actions = (
    <div className="statement-actions">
      {canApproximate && (
        <button type="button" className="btn btn-ghost statement-eye" onClick={onToggleDecimal} aria-label={decimal ? 'Show the exact value' : 'Show a decimal approximation'}>
          {decimal ? 'Exact' : 'Decimal'}
        </button>
      )}
      {row.plotKind === 'curve' && (
        <button type="button" className="btn btn-ghost statement-eye" aria-pressed={plane} onClick={onTogglePlane} aria-label={plane ? `Draw ${row.label} as a curve` : `Extend ${row.label} into a plane`}>
          {plane ? 'Curve' : 'Plane'}
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
  )
  if (row.input !== undefined) {
    const same = Boolean(shownTex && inputTex && shownTex === inputTex)
    return (
      <li className={`statement-row is-${row.source} is-math`}>
        <div className="statement-stack">
          <div className="statement-math-line">
            <MathRowEditor key={row.input} input={row.input} onCommit={onCommitMath} />
            <span className="statement-copies">
              <button type="button" className="btn btn-ghost statement-copy" aria-label={copied === 'input' ? 'Copied' : 'Copy input'} onClick={() => copy(inputTex ?? row.input ?? '', 'input')}>
                <CopyIcon copied={copied === 'input'} />
              </button>
            </span>
          </div>
          {shownTex && !same && (
            <div className="statement-math-line is-output">
              <Eq tex={`=\\;${shownTex}`} />
              <button type="button" className="btn btn-ghost statement-copy" aria-label={copied === 'output' ? 'Copied' : 'Copy output'} onClick={() => copy(shownTex, 'output')}>
                <CopyIcon copied={copied === 'output'} />
              </button>
            </div>
          )}
          {row.source === 'error' && <p className="statement-note">{row.text}</p>}
          {row.source !== 'error' && note && <p className="statement-note">{note}</p>}
        </div>
        {actions}
        {open && expandable && row.formula && (
          <div className="statement-formula">
            <Eq tex={row.formula} />
          </div>
        )}
      </li>
    )
  }
  const body = (
    <>
      <span className="statement-label">{row.label}</span>
      <span className="statement-value">
        {quantity && <Eq tex={quantity} />}
        {shownTex && <Eq tex={shownTex} />}
        {!quantity && !shownTex && row.text && <span>{row.text}</span>}
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
      {actions}
      {open && expandable && row.formula && (
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
  const [decimal, setDecimal] = useState<Record<string, boolean>>({})
  const [planes, setPlanes] = useState<Record<string, boolean>>({})
  const [angles, setAngles] = useState<AngleMode>('rad')
  const anglesRef = useRef(angles)
  anglesRef.current = angles
  const compiled = useMemo(() => compileDocument(doc, angles), [doc, angles])
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
          <button
            type="button"
            className="btn btn-ghost workspace-clear"
            aria-pressed={angles === 'deg'}
            aria-label={angles === 'deg' ? 'Angles are in degrees. Switch to radians.' : 'Angles are in radians. Switch to degrees.'}
            onClick={() => {
              const from = anglesRef.current
              const to: AngleMode = from === 'rad' ? 'deg' : 'rad'
              anglesRef.current = to
              setAngles(to)
              setDoc((current) => convertDocumentAngles(current, from, to))
            }}
          >
            {angles === 'deg' ? 'Degrees' : 'Radians'}
          </button>
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
                    plane={Boolean(row.statementId && planes[row.statementId])}
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
                    onTogglePlane={() => {
                      if (!row.statementId) return
                      const id = row.statementId
                      setPlanes((current) => ({ ...current, [id]: !current[id] }))
                    }}
                    onCommitMath={(value) => {
                      if (!row.statementId) return 'That line cannot be edited.'
                      const text = value.trim()
                      if (!text) {
                        setDoc((current) => removeStatement(current, row.statementId!))
                        return null
                      }
                      const message = validateMath(text)
                      if (message) return message
                      setDoc((current) => replaceMath(current, row.statementId!, text))
                      return null
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
      <FigurePane view={view} playback={playback} planes={planes} />
    </div>
  )
}

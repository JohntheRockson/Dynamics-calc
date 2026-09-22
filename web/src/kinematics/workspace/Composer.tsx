import { useId, useRef, useState } from 'react'
import { Eq } from '../Eq'
import { filterCommands, type CommandDef } from './commands'
import { nextPointName, pointNames, type WorkspaceDocument } from './document'
import { previewTex } from './math/expr'
import { emptyFunctionShortcut, expandMathShortcut, looksLikeMath } from './math/shortcuts'

function defaultArgs(command: CommandDef, doc: WorkspaceDocument): Record<string, string> {
  const names = pointNames(doc)
  const values: Record<string, string> = {}
  for (const arg of command.args) {
    if (arg.kind === 'new-name') values[arg.id] = nextPointName(doc)
    else if (arg.kind === 'point') {
      if (command.id === 'relative' && arg.id === 'from') values[arg.id] = names[0] ?? ''
      else if (command.id === 'relative' && arg.id === 'to') values[arg.id] = names[1] ?? names[0] ?? ''
      else values[arg.id] = names[names.length - 1] ?? ''
    } else values[arg.id] = ''
  }
  return values
}

export function Composer({ doc, onCommit, onMath }: { doc: WorkspaceDocument; onCommit: (commandId: string, args: Record<string, string>) => string | null; onMath: (input: string) => string | null }) {
  const listId = useId()
  const inputRef = useRef<HTMLInputElement>(null)
  const [query, setQuery] = useState('')
  const [open, setOpen] = useState(false)
  const [highlight, setHighlight] = useState(0)
  const [command, setCommand] = useState<CommandDef | null>(null)
  const [args, setArgs] = useState<Record<string, string>>({})
  const [error, setError] = useState<string | null>(null)
  const mathMode = !command && looksLikeMath(query)
  const matches = mathMode ? [] : filterCommands(query)
  const names = pointNames(doc)
  const activeIndex = Math.min(highlight, Math.max(matches.length - 1, 0))
  const preview = mathMode ? previewTex(query) : null

  const choose = (next: CommandDef) => {
    setCommand(next)
    setArgs(defaultArgs(next, doc))
    setQuery('')
    setOpen(false)
    if (next.id !== 'point' && names.length === 0) setError('Add a point first.')
    else if (next.id === 'relative' && names.length < 2) setError('Add a second point, then use Relative.')
    else setError(null)
  }

  const clearCommand = () => {
    setCommand(null)
    setArgs({})
    setError(null)
    inputRef.current?.focus()
  }

  const commitMath = () => {
    const message = onMath(query)
    if (message) {
      setError(message)
      return
    }
    setQuery('')
    setError(null)
    setOpen(false)
    inputRef.current?.focus()
  }

  const submit = () => {
    if (!command) return
    const message = onCommit(command.id, args)
    if (message) {
      setError(message)
      return
    }
    setCommand(null)
    setArgs({})
    setError(null)
    inputRef.current?.focus()
  }

  return (
    <form
      className="composer"
      onSubmit={(event) => {
        event.preventDefault()
        if (command) submit()
        else if (mathMode || (query.trim() && matches.length === 0)) commitMath()
        else if (matches[activeIndex]) choose(matches[activeIndex])
      }}
    >
      {command ? (
        <div className="composer-slots">
          <button type="button" className="composer-chip" onClick={clearCommand}>
            {command.title}
          </button>
          {command.args.map((arg, index) => (
            <label key={arg.id} className="composer-slot">
              <span>{arg.label}</span>
              {arg.kind === 'point' ? (
                <select
                  className="num-input"
                  aria-label={arg.label}
                  value={args[arg.id] ?? ''}
                  onChange={(event) => setArgs((current) => ({ ...current, [arg.id]: event.target.value }))}
                >
                  {names.length === 0 && <option value="">—</option>}
                  {names.map((name) => (
                    <option key={name} value={name}>
                      {name}
                    </option>
                  ))}
                </select>
              ) : (
                <input
                  className="num-input"
                  aria-label={arg.label}
                  value={args[arg.id] ?? ''}
                  placeholder={arg.placeholder}
                  autoFocus={index === 0}
                  inputMode={arg.kind === 'new-name' ? 'text' : 'decimal'}
                  onChange={(event) => setArgs((current) => ({ ...current, [arg.id]: event.target.value }))}
                />
              )}
              {arg.unit && <span className="composer-unit">{arg.unit}</span>}
            </label>
          ))}
          <button type="submit" className="btn btn-primary">
            Add
          </button>
        </div>
      ) : (
        <div className="composer-search">
          <div className={preview ? 'composer-field is-math' : 'composer-field'}>
            {preview && (
              <div className="composer-preview" aria-hidden="true">
                <Eq tex={preview} />
              </div>
            )}
          <input
            ref={inputRef}
            className="num-input"
            role="combobox"
            aria-expanded={open}
            aria-controls={listId}
            aria-autocomplete="list"
            aria-label="Type a statement"
            placeholder="Type a statement"
            value={query}
            onFocus={() => setOpen(true)}
            onBlur={() => setOpen(false)}
            onChange={(event) => {
              setQuery(event.target.value)
              setHighlight(0)
              setOpen(true)
            }}
            onKeyDown={(event) => {
              const input = event.currentTarget
              const typed = input.value
              const cursor = input.selectionStart ?? typed.length
              const expanded = emptyFunctionShortcut(typed, cursor, event.key) ?? expandMathShortcut(typed, cursor, event.key)
              if (expanded) {
                event.preventDefault()
                const input = event.currentTarget
                setQuery(expanded.value)
                setHighlight(0)
                setError(null)
                requestAnimationFrame(() => {
                  input.setSelectionRange(expanded.cursor, expanded.cursor)
                })
                return
              }
              if (mathMode) {
                if (event.key === 'Escape') setOpen(false)
                return
              }
              if (event.key === 'ArrowDown') {
                event.preventDefault()
                setOpen(true)
                setHighlight((index) => Math.min(matches.length - 1, index + 1))
              } else if (event.key === 'ArrowUp') {
                event.preventDefault()
                setHighlight((index) => Math.max(0, index - 1))
              } else if (event.key === 'Escape') {
                setOpen(false)
              }
            }}
          />
          </div>
          {open && !mathMode && (
            <ul className="composer-list" id={listId} role="listbox">
              {matches.length === 0 ? (
                <li className="composer-empty">No matching statement. Try point, speed, circle, or simulate.</li>
              ) : (
                matches.map((item, index) => (
                  <li key={item.id} role="option" aria-selected={index === activeIndex}>
                    <button
                      type="button"
                      className={index === activeIndex ? 'active' : ''}
                      onMouseDown={(event) => event.preventDefault()}
                      onClick={() => choose(item)}
                    >
                      <span>{item.title}</span>
                      <span>{item.blurb}</span>
                    </button>
                  </li>
                ))
              )}
            </ul>
          )}
        </div>
      )}
      <p className="hint composer-hint">{error ?? command?.blurb ?? (mathMode ? 'Enter adds this calculation. / starts an empty function. Space after sqrt, ln, log, or sin opens it.' : 'Type math, or a statement such as point, speed, or circle. / starts an empty function.')}</p>
    </form>
  )
}

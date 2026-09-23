import { useId, useRef, useState } from 'react'
import { Eq } from '../Eq'
import { filterCommands, type CommandDef } from './commands'
import { nextPointName, pointNames, type WorkspaceDocument } from './document'
import { MATH_SECTIONS } from './math/catalog'
import { previewTex } from './math/expr'
import { latexToSource, moveMathCursor } from './math/inputView'
import { emptyFunctionShortcut, expandMathShortcut, looksLikeMath } from './math/shortcuts'

function MathLine({ tex }: { tex: string }) {
  return (
    <div className="composer-math" aria-hidden="true">
      <Eq tex={tex} />
    </div>
  )
}

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
  const [menu, setMenu] = useState(false)
  const [section, setSection] = useState(MATH_SECTIONS[0].id)
  const [cursor, setCursor] = useState(0)
  const mathMode = !command && looksLikeMath(query)
  const matches = mathMode ? [] : filterCommands(query)
  const names = pointNames(doc)
  const activeIndex = Math.min(highlight, Math.max(matches.length - 1, 0))
  const preview = mathMode ? previewTex(query, cursor) : null

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
    const live = latexToSource(inputRef.current?.value ?? query)
    const message = onMath(live)
    if (message) {
      setError(message)
      return
    }
    setQuery('')
    setCursor(0)
    setError(null)
    setOpen(false)
    inputRef.current?.focus()
  }

  const insertTemplate = (template: string) => {
    setQuery(template)
    setMenu(false)
    setOpen(false)
    setError(null)
    const caret = template.indexOf('(')
    const next = caret >= 0 ? caret + 1 : template.length
    setCursor(next)
    requestAnimationFrame(() => {
      const input = inputRef.current
      if (!input) return
      input.focus()
      input.setSelectionRange(next, next)
    })
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
        const live = latexToSource(inputRef.current?.value ?? query)
        if (command) submit()
        else if (looksLikeMath(live) || (live.trim() && filterCommands(live).length === 0)) commitMath()
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
            {preview && <MathLine tex={preview} />}
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
            spellCheck={false}
            autoCapitalize="off"
            onFocus={() => setOpen(true)}
            onBlur={() => setOpen(false)}
            onChange={(event) => {
              const raw = event.target.value
              const ascii = latexToSource(raw)
              const nextCursor = ascii === raw ? (event.target.selectionStart ?? ascii.length) : ascii.length
              setQuery(ascii)
              setCursor(nextCursor)
              setHighlight(0)
              setOpen(true)
              if (ascii !== raw) {
                const placed = ascii
                requestAnimationFrame(() => {
                  if (event.target.value === placed) event.target.setSelectionRange(placed.length, placed.length)
                })
              }
            }}
            onSelect={(event) => setCursor(event.currentTarget.selectionStart ?? event.currentTarget.value.length)}
            onKeyUp={(event) => setCursor(event.currentTarget.selectionStart ?? event.currentTarget.value.length)}
            onClick={(event) => setCursor(event.currentTarget.selectionStart ?? event.currentTarget.value.length)}
            onKeyDown={(event) => {
              const input = event.currentTarget
              const typed = input.value
              const caret = input.selectionStart ?? typed.length
              const end = input.selectionEnd ?? caret
              const liveMath = looksLikeMath(typed)
              const expanded = emptyFunctionShortcut(typed, caret, event.key) ?? expandMathShortcut(typed, caret, event.key)
              if (expanded) {
                event.preventDefault()
                setQuery(expanded.value)
                setCursor(expanded.cursor)
                setHighlight(0)
                setError(null)
                const placed = expanded.value
                const placedCursor = expanded.cursor
                requestAnimationFrame(() => {
                  if (input.value === placed) input.setSelectionRange(placedCursor, placedCursor)
                })
                return
              }
              if (event.key === '/' && typed.slice(0, caret).trim() !== '' && looksLikeMath(typed.slice(0, caret))) {
                event.preventDefault()
                const left = typed.slice(0, caret)
                const right = typed.slice(end)
                const value = `${left}/()${right}`
                const next = left.length + 2
                setQuery(value)
                setCursor(next)
                setHighlight(0)
                setError(null)
                const placed = value
                requestAnimationFrame(() => {
                  if (input.value === placed) input.setSelectionRange(next, next)
                })
                return
              }
              if (liveMath && (event.key === 'ArrowLeft' || event.key === 'ArrowRight' || event.key === 'ArrowUp' || event.key === 'ArrowDown')) {
                event.preventDefault()
                const dir = event.key === 'ArrowLeft' ? 'left' : event.key === 'ArrowRight' ? 'right' : event.key === 'ArrowUp' ? 'up' : 'down'
                const next = moveMathCursor(typed, caret, dir)
                setCursor(next)
                input.setSelectionRange(next, next)
                return
              }
              if (liveMath) {
                if (event.key === 'Escape') {
                  setOpen(false)
                  setMenu(false)
                }
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
          {menu && (
            <div className="math-menu" role="dialog" aria-label="Functions">
              <div className="math-menu-tabs">
                {MATH_SECTIONS.map((item) => (
                  <button key={item.id} type="button" className={item.id === section ? 'is-on' : ''} onMouseDown={(event) => event.preventDefault()} onClick={() => setSection(item.id)}>
                    {item.title}
                  </button>
                ))}
              </div>
              <ul>
                {(MATH_SECTIONS.find((item) => item.id === section) ?? MATH_SECTIONS[0]).items.map((item) => (
                  <li key={item.template}>
                    <button type="button" onMouseDown={(event) => event.preventDefault()} onClick={() => insertTemplate(item.template)}>
                      <span>{item.name}</span>
                      <span>{item.blurb}</span>
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          )}
          {open && !mathMode && !menu && (
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
      <div className="composer-hint-row">
        <button type="button" className="btn btn-ghost workspace-clear" aria-expanded={menu} onClick={() => setMenu((current) => !current)}>
          Functions
        </button>
        <p className="hint composer-hint">{error ?? command?.blurb ?? (mathMode ? 'Enter adds this calculation. / starts f(x), and // starts f(x, y). Up and down move between the top and bottom of a fraction, and left and right move across. A prime dots the previous symbol, and _ writes a subscript.' : 'Type math, or a statement such as point or circle. / starts a function.')}</p>
      </div>
    </form>
  )
}

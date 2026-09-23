import { useId, useRef, useState } from 'react'
import { filterCommands, type CommandDef } from './commands'
import { nextPointName, pointNames, type WorkspaceDocument } from './document'
import { MathField, type MathFieldHandle } from './MathField'
import { MATH_SECTIONS } from './math/catalog'
import { latexToSource } from './math/inputView'
import { looksLikeMath } from './math/shortcuts'

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
  const fieldRef = useRef<MathFieldHandle>(null)
  const [query, setQuery] = useState('')
  const [open, setOpen] = useState(false)
  const [highlight, setHighlight] = useState(0)
  const [command, setCommand] = useState<CommandDef | null>(null)
  const [args, setArgs] = useState<Record<string, string>>({})
  const [error, setError] = useState<string | null>(null)
  const [menu, setMenu] = useState(false)
  const [section, setSection] = useState(MATH_SECTIONS[0].id)
  const mathMode = !command && looksLikeMath(query)
  const matches = mathMode ? [] : filterCommands(query)
  const names = pointNames(doc)
  const activeIndex = Math.min(highlight, Math.max(matches.length - 1, 0))

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
    fieldRef.current?.focus()
  }

  const commitMath = () => {
    const live = latexToSource(fieldRef.current?.read() ?? query)
    const message = onMath(live)
    if (message) {
      setError(message)
      return
    }
    setQuery('')
    setError(null)
    setOpen(false)
    fieldRef.current?.focus()
  }

  const insertTemplate = (template: string) => {
    setMenu(false)
    setOpen(false)
    setError(null)
    const caret = template.indexOf('(')
    const next = caret >= 0 ? caret + 1 : template.length
    fieldRef.current?.place(template, next)
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
    fieldRef.current?.focus()
  }

  return (
    <form
      className="composer"
      onSubmit={(event) => {
        event.preventDefault()
        const live = latexToSource(fieldRef.current?.read() ?? query)
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
          <MathField
            ref={fieldRef}
            value={query}
            label="Type a statement"
            placeholder="Type a statement"
            onFocus={() => setOpen(true)}
            onBlur={() => setOpen(false)}
            onValue={(next) => {
              setHighlight(0)
              setOpen(true)
              setError(null)
              setQuery(next)
            }}
            onSubmit={(live) => {
              const text = latexToSource(live)
              if (looksLikeMath(text) || (text.trim() && filterCommands(text).length === 0)) commitMath()
              else if (matches[activeIndex]) choose(matches[activeIndex])
            }}
            onEscape={() => {
              setOpen(false)
              setMenu(false)
            }}
            onCommandKey={(key) => {
              if (key === 'ArrowDown') {
                setOpen(true)
                setHighlight((index) => Math.min(matches.length - 1, index + 1))
              } else if (key === 'ArrowUp') setHighlight((index) => Math.max(0, index - 1))
              else setOpen(false)
            }}
          />
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
        <p className="hint composer-hint">{error ?? command?.blurb ?? (mathMode ? 'Enter runs this line. Shift+Enter starts a new line. Click a previous line to edit it, then Enter runs it again. / starts f(x), and // starts f(x, y). A parametric curve takes a domain such as t = 0..2*pi.' : 'Type math, or a statement such as point or circle. Enter runs it. Shift+Enter starts a new line.')}</p>
      </div>
    </form>
  )
}

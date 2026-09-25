import { useId, useRef, useState } from 'react'
import { filterCommands, type CommandDef } from './commands'
import { nextPointName, pointNames, type WorkspaceDocument } from './document'
import { MathField, type MathFieldHandle } from './MathField'
import { FUNCTIONS, MATH_FORMS, MATH_SECTIONS, signatureText, type FunctionSpec, type SectionId } from './math/functions'
import { latexToSource } from './math/inputView'
import { looksLikeMath } from './math/shortcuts'

function settingsText(spec: FunctionSpec): string | null {
  const names = spec.options.map((option) => option.name)
  if (spec.draws && !names.includes('Color')) names.push('Color')
  if (spec.draws) names.push('…')
  return names.length > 0 ? `{${names.join(', ')}}` : null
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
  const fieldRef = useRef<MathFieldHandle>(null)
  const [query, setQuery] = useState('')
  const [open, setOpen] = useState(false)
  const [highlight, setHighlight] = useState(0)
  const [command, setCommand] = useState<CommandDef | null>(null)
  const [args, setArgs] = useState<Record<string, string>>({})
  const [error, setError] = useState<string | null>(null)
  const [menu, setMenu] = useState(false)
  const [section, setSection] = useState<SectionId>('algebra')
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

  const insertCall = (spec: FunctionSpec) => {
    setMenu(false)
    setOpen(false)
    setError(null)
    fieldRef.current?.insert(`${spec.name}()`, spec.name.length + 1)
  }

  const insertExample = (example: string) => {
    setMenu(false)
    setOpen(false)
    setError(null)
    fieldRef.current?.place(example, example.length)
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
            assist={!menu}
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
              <p className="math-menu-rule">
                Required inputs go in <code>( )</code>, in order. Optional settings go in <code>{'{ }'}</code> right after the closing parenthesis, in any order: <code>{'Solve(x^2 = 4, x){Domain: 0..5}'}</code>
              </p>
              <div className="math-menu-tabs" role="tablist">
                {MATH_SECTIONS.map((item) => (
                  <button key={item.id} type="button" role="tab" aria-selected={item.id === section} className={item.id === section ? 'is-on' : ''} onMouseDown={(event) => event.preventDefault()} onClick={() => setSection(item.id)}>
                    {item.title}
                  </button>
                ))}
              </div>
              <ul>
                {FUNCTIONS.filter((spec) => spec.section === section).map((spec) => {
                  const settings = settingsText(spec)
                  return (
                    <li key={spec.name}>
                      <button type="button" className="math-menu-item" onMouseDown={(event) => event.preventDefault()} onClick={() => insertCall(spec)}>
                        <span className="math-menu-signature">
                          {signatureText(spec)}
                          {settings && <span className="math-menu-settings">{settings}</span>}
                        </span>
                        <span className="math-menu-summary">{spec.summary}</span>
                      </button>
                      <button type="button" className="math-menu-example" title="Use this example" onMouseDown={(event) => event.preventDefault()} onClick={() => insertExample(spec.examples[0] ?? `${spec.name}()`)}>
                        {spec.examples[0]}
                      </button>
                    </li>
                  )
                })}
                {MATH_FORMS.filter((form) => form.section === section).map((form) => (
                  <li key={form.title}>
                    <button type="button" className="math-menu-item" onMouseDown={(event) => event.preventDefault()} onClick={() => insertExample(form.example)}>
                      <span className="math-menu-signature">{form.title}</span>
                      <span className="math-menu-summary">{form.summary}</span>
                    </button>
                    <button type="button" className="math-menu-example" title="Use this example" onMouseDown={(event) => event.preventDefault()} onClick={() => insertExample(form.example)}>
                      {form.example}
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
        <p className="hint composer-hint">{error ?? command?.blurb ?? (mathMode ? 'Enter runs this line, Shift+Enter starts a new one. Inputs go in ( ), settings in { } after the ): Solve(x^2 = 4, x){Domain: 0..5}. / starts f(x), // starts f(x, y).' : 'Type math, or a statement such as point or circle. Enter runs it. Functions lists every math function.')}</p>
      </div>
    </form>
  )
}

import { Fragment, useLayoutEffect, useRef, useState, type ReactNode, type RefObject } from 'react'
import { createPortal } from 'react-dom'
import { findFunction, suggestName, type FunctionSpec, type OptionSpec } from './math/functions'
import type { Assist, AssistItem } from './math/signature'

function settingsList(spec: FunctionSpec, graph: boolean): string | null {
  const names = spec.options.map((option) => option.name)
  if (spec.draws && graph) names.push(names.includes('Color') ? '…' : 'Color, …')
  return names.length > 0 ? names.join(', ') : null
}

function Signature({ spec, active, graph }: { spec: FunctionSpec; active: number | null; graph: boolean }) {
  const settings = settingsList(spec, graph)
  return (
    <div className="math-assist-signature">
      <span className="math-assist-name">{spec.name}</span>(
      {spec.params.map((param, index) => (
        <Fragment key={param.name}>
          {index > 0 && ', '}
          <span className={index === active ? 'math-assist-param is-active' : 'math-assist-param'}>
            {param.name}
            {param.repeats ? ', …' : ''}
          </span>
        </Fragment>
      ))}
      )
      {settings && <span className="math-assist-braces">{`{${settings}}`}</span>}
    </div>
  )
}

function exampleWithSettings(spec: FunctionSpec): string | null {
  return spec.examples.find((example) => example.includes('{')) ?? null
}

function valueRule(option: OptionSpec): string {
  const fallback = option.default === null || option.default === 'auto' ? '' : `, default ${String(option.default)}`
  switch (option.kind) {
    case 'range':
      return `A range such as ${option.example}`
    case 'whole':
      return option.max === undefined ? `A whole number of at least ${option.min ?? 0}${fallback}` : `A whole number from ${option.min ?? 0} to ${option.max}${fallback}`
    case 'boolean':
      return `true or false${fallback}`
    case 'color':
      return 'A color name, or a hex code such as #ff8800'
    default:
      return `A number or expression, such as ${option.example}${fallback}`
  }
}

function ItemList({ items, active, onPick, onHover }: { items: AssistItem[]; active: number; onPick: (item: AssistItem) => void; onHover: (index: number) => void }) {
  return (
    <ul className="math-assist-list" role="listbox">
      {items.map((item, index) => (
        <li key={item.id} role="option" aria-selected={index === active}>
          <button
            type="button"
            tabIndex={-1}
            className={index === active ? 'is-active' : ''}
            onMouseDown={(event) => event.preventDefault()}
            onMouseEnter={() => onHover(index)}
            onClick={() => onPick(item)}
          >
            <span className="math-assist-label">{item.label}</span>
            {item.detail && <span className="math-assist-detail">{item.detail}</span>}
            {item.note && <span className="math-assist-note">{item.note}</span>}
          </button>
        </li>
      ))}
    </ul>
  )
}

function Chips({ items, active, onPick, onHover }: { items: AssistItem[]; active: number; onPick: (item: AssistItem) => void; onHover: (index: number) => void }) {
  return (
    <div className="math-assist-chips" role="listbox">
      {items.map((item, index) => (
        <button
          key={item.id}
          type="button"
          role="option"
          tabIndex={-1}
          aria-selected={index === active}
          className={index === active ? 'is-active' : ''}
          onMouseDown={(event) => event.preventDefault()}
          onMouseEnter={() => onHover(index)}
          onClick={() => onPick(item)}
        >
          {item.swatch && <span className="math-assist-swatch" style={{ background: item.swatch }} />}
          {item.label}
        </button>
      ))}
    </div>
  )
}

function Body({ assist, active, onPick, onHover }: { assist: Assist; active: number; onPick: (item: AssistItem) => void; onHover: (index: number) => void }): ReactNode {
  switch (assist.kind) {
    case 'complete':
      return <ItemList items={assist.items} active={active} onPick={onPick} onHover={onHover} />
    case 'params': {
      const param = assist.spec.params[assist.active]
      const example = exampleWithSettings(assist.spec)
      return (
        <>
          <Signature spec={assist.spec} active={assist.active} graph={assist.graph} />
          {param ? (
            <>
              <p className="math-assist-text">{param.description}</p>
              <p className="math-assist-example">e.g. {param.example}</p>
            </>
          ) : (
            <p className="math-assist-text is-warning">
              {assist.spec.name} takes {assist.spec.params.length === 1 ? 'one input' : `${assist.spec.params.length} inputs`}.
              {example ? ` Settings go in { } after the closing parenthesis, as in ${example}` : ` The line ends with the closing parenthesis.`}
            </p>
          )}
        </>
      )
    }
    case 'after': {
      const own = assist.spec.options
      const example = exampleWithSettings(assist.spec)
      return (
        <>
          <Signature spec={assist.spec} active={null} graph={assist.graph} />
          <p className="math-assist-text">Optional settings go in {'{ }'} right after the closing parenthesis, in any order.</p>
          {own.length > 0 && (
            <ul className="math-assist-options">
              {own.map((option) => (
                <li key={option.name}>
                  <span className="math-assist-label">{option.name}</span>
                  <span className="math-assist-detail">{option.description}</span>
                </li>
              ))}
            </ul>
          )}
          {assist.graph && <p className="math-assist-text is-dim">Graph: Color, PlotPoints, MaxRecursion, Exclusions, Dashed{own.some((option) => option.name === 'Domain') ? '' : ', Domain'}</p>}
          {example && <p className="math-assist-example">e.g. {example}</p>}
        </>
      )
    }
    case 'options': {
      const title = assist.owner ? `${assist.owner} settings` : 'Settings'
      let message: string | null = null
      if (assist.items.length === 0) {
        if (assist.owner === null) message = "Settings go right after a function's closing parenthesis, as in Plot(sin(x), x){Color: red}."
        else if (assist.options.length === 0) {
          const spec = findFunction(assist.owner)
          if (spec?.draws) message = 'Graph settings such as Color go at the end of the line, on the function the line draws.'
          else if (spec?.section === 'basic') message = `${assist.owner} has no settings. To draw it, use Plot(${assist.owner}(x), x){Color: red}.`
          else message = `${assist.owner} has no settings. Put each setting right after the function it belongs to, as in Expand(Derivative(x^3, x){Order: 2}).`
        } else if (assist.prefix && !/^[A-Za-z]/.test(assist.prefix)) message = `Start each setting with its name, as in {${assist.options[0]?.name ?? 'Domain'}: ${assist.options[0]?.example ?? '0..5'}}.`
        else if (assist.prefix) {
          const guess = suggestName(assist.prefix, assist.options.map((option) => option.name))
          message = guess ? `No setting starts with ${assist.prefix}. Did you mean ${guess}?` : `No setting starts with ${assist.prefix}. It takes ${assist.options.map((option) => option.name).join(', ')}.`
        } else message = 'Every setting is already here.'
      }
      return (
        <>
          <div className="math-assist-title">
            <span>{title}</span>
            <span className="math-assist-dim">Name: value, any order</span>
          </div>
          {message ? <p className="math-assist-text">{message}</p> : <ItemList items={assist.items} active={active} onPick={onPick} onHover={onHover} />}
        </>
      )
    }
    case 'value': {
      const option = assist.option
      const guess = option ? null : suggestName(assist.key, assist.options.map((known) => known.name))
      return (
        <>
          <div className="math-assist-title">
            <span className="math-assist-name">{option?.name ?? assist.key}</span>
            {assist.owner && <span className="math-assist-dim">{assist.owner}</span>}
          </div>
          {option ? (
            <>
              <p className="math-assist-text">{option.description}</p>
              <p className="math-assist-example">{valueRule(option)}</p>
              {assist.items.length > 0 && <Chips items={assist.items} active={active} onPick={onPick} onHover={onHover} />}
            </>
          ) : (
            <p className="math-assist-text is-warning">{assist.owner ? `${assist.owner} has no setting ${assist.key}.` : `${assist.key} is not a setting here.`}{guess ? ` Did you mean ${guess}?` : ''}</p>
          )}
        </>
      )
    }
  }
}

/** The hint card, drawn above the field (or below when there is no room) so it never pushes the layout. */
export function MathAssist({ assist, anchor, active, onPick, onHover }: { assist: Assist; anchor: RefObject<HTMLElement | null>; active: number; onPick: (item: AssistItem) => void; onHover: (index: number) => void }) {
  const cardRef = useRef<HTMLDivElement>(null)
  const [place, setPlace] = useState<{ left: number; top: number; maxWidth: number } | null>(null)

  useLayoutEffect(() => {
    const card = cardRef.current
    const field = anchor.current
    if (!card || !field) return
    const update = () => {
      const box = field.getBoundingClientRect()
      const maxWidth = Math.max(260, Math.min(520, window.innerWidth - 16))
      const width = Math.min(card.offsetWidth, maxWidth)
      const height = card.offsetHeight
      const left = Math.round(Math.max(8, Math.min(box.left, window.innerWidth - width - 8)))
      const above = box.top - height - 6
      const top = Math.round(above >= 8 ? above : Math.min(box.bottom + 6, window.innerHeight - height - 8))
      setPlace((current) => (current && current.left === left && current.top === top && current.maxWidth === maxWidth ? current : { left, top, maxWidth }))
    }
    update()
    window.addEventListener('resize', update)
    window.addEventListener('scroll', update, true)
    return () => {
      window.removeEventListener('resize', update)
      window.removeEventListener('scroll', update, true)
    }
  })

  const pickable = 'items' in assist && assist.items.length > 0
  return createPortal(
    <div
      ref={cardRef}
      className="math-assist"
      role="dialog"
      aria-label="Input help"
      style={place ? { left: place.left, top: place.top, maxWidth: place.maxWidth } : { left: 0, top: 0, visibility: 'hidden' }}
    >
      <Body assist={assist} active={active} onPick={onPick} onHover={onHover} />
      {pickable && <div className="math-assist-keys">Tab inserts · ↑↓ choose · Esc hides</div>}
    </div>,
    document.body,
  )
}

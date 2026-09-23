import katex from 'katex'
import { useMemo } from 'react'

interface EqProps {
  tex: string
  block?: boolean
}

/** Renders a LaTeX string with KaTeX. `block` centers it on its own line. */
export function Eq({ tex, block = false }: EqProps) {
  const html = useMemo(() => katex.renderToString(tex, { throwOnError: false, displayMode: block }), [tex, block])
  return <span className={block ? 'eq-block' : 'eq-inline'} dangerouslySetInnerHTML={{ __html: html }} />
}

interface EqListProps {
  items: string[]
}

/** A stack of display-mode equations, e.g. the governing equations for a mode. */
export function EqList({ items }: EqListProps) {
  return (
    <div className="eq-list">
      {items.map((tex) => (
        <Eq key={tex} tex={tex} block />
      ))}
    </div>
  )
}

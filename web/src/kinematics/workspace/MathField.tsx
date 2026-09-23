import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from 'react'
import { Eq } from '../Eq'
import { previewTex } from './math/expr'
import { closeOpenGroups, exitSlotsForComma, latexToSource, moveMathCursor } from './math/inputView'
import { emptyFunctionShortcut, expandMathShortcut, insertMathSlot, looksLikeMath } from './math/shortcuts'

export interface MathFieldHandle {
  place: (value: string, cursor: number) => void
  focus: () => void
  read: () => string
}

/**
 * Write the new value and caret straight into the real field, then tell React about it.
 * A controlled field only gets its DOM value back on the next render, so placing the caret
 * in a later frame races a fast second keystroke.
 */
function placeMathInput(
  input: HTMLTextAreaElement | HTMLInputElement,
  value: string,
  cursor: number,
  onValue: (value: string) => void,
  setCursor: (cursor: number) => void,
): void {
  input.value = value
  input.setSelectionRange(cursor, cursor)
  onValue(value)
  setCursor(cursor)
}

function moveLine(source: string, cursor: number, dir: 'up' | 'down'): number {
  const lines = source.split('\n')
  let offset = 0
  let index = 0
  let column = 0
  for (let i = 0; i < lines.length; i += 1) {
    const end = offset + (lines[i]?.length ?? 0)
    if (cursor <= end || i === lines.length - 1) {
      index = i
      column = cursor - offset
      break
    }
    offset = end + 1
  }
  const target = index + (dir === 'up' ? -1 : 1)
  if (target < 0 || target >= lines.length) return cursor
  let start = 0
  for (let i = 0; i < target; i += 1) start += (lines[i]?.length ?? 0) + 1
  return start + Math.min(Math.max(column, 0), lines[target]?.length ?? 0)
}

function lineViews(value: string, cursor: number): { tex: string | null; source: string; active: boolean }[] {
  const parts = value.split('\n')
  let offset = 0
  return parts.map((source, index) => {
    const start = offset
    const end = start + source.length
    const active = cursor >= start && (cursor <= end || index === parts.length - 1)
    offset = end + 1
    const local = Math.max(0, Math.min(source.length, cursor - start))
    const tex = source.trim() ? previewTex(source, active ? local : undefined) : null
    return { tex, source, active: active && cursor <= end }
  })
}

export const MathField = forwardRef<MathFieldHandle, {
  value: string
  label: string
  placeholder?: string
  onValue: (value: string) => void
  onSubmit: (value: string) => void
  onFocus?: () => void
  onBlur?: () => void
  onCommandKey?: (key: 'ArrowUp' | 'ArrowDown' | 'Escape') => void
  onEscape?: () => void
}>(function MathField({ value, label, placeholder, onValue, onSubmit, onFocus, onBlur, onCommandKey, onEscape }, handle) {
  const fieldRef = useRef<HTMLTextAreaElement>(null)
  const [cursor, setCursor] = useState(0)
  const lines = lineViews(value, cursor)
  const overlay = lines.some((line) => line.tex)

  useEffect(() => {
    const field = fieldRef.current
    if (!field || overlay) return
    field.style.height = 'auto'
    field.style.height = `${field.scrollHeight}px`
  }, [value, overlay])

  useImperativeHandle(handle, () => ({
    place(next, at) {
      const field = fieldRef.current
      if (!field) {
        onValue(next)
        setCursor(at)
        return
      }
      field.focus()
      placeMathInput(field, next, at, onValue, setCursor)
    },
    focus() {
      fieldRef.current?.focus()
    },
    read() {
      return fieldRef.current?.value ?? value
    },
  }))

  return (
    <div className={overlay ? 'composer-field is-math' : 'composer-field'}>
      {overlay && (
        <div className="composer-math" aria-hidden="true">
          <div className="math-lines">
            {lines.map((line, index) =>
              line.tex ? (
                <Eq key={index} tex={line.tex} />
              ) : (
                <span key={index} className="math-line-source">
                  {line.source || '\u00a0'}
                  {line.active && <span className="math-caret" />}
                </span>
              ),
            )}
          </div>
        </div>
      )}
      <textarea
        ref={fieldRef}
        className="num-input"
        aria-label={label}
        placeholder={overlay ? undefined : placeholder}
        value={value}
        rows={1}
        spellCheck={false}
        autoCapitalize="off"
        onFocus={onFocus}
        onBlur={onBlur}
        onChange={(event) => {
          const raw = event.target.value
          const ascii = latexToSource(raw)
          if (ascii !== raw) placeMathInput(event.target, ascii, ascii.length, onValue, setCursor)
          else {
            onValue(ascii)
            setCursor(event.target.selectionStart ?? ascii.length)
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
          if (event.key === 'Enter' && event.shiftKey) {
            event.preventDefault()
            const left = typed.slice(0, caret)
            const right = typed.slice(end)
            placeMathInput(input, `${left}\n${right}`, left.length + 1, onValue, setCursor)
            return
          }
          if (event.key === 'Enter') {
            event.preventDefault()
            onSubmit(typed)
            return
          }
          const liveMath = looksLikeMath(typed)
          const expanded = emptyFunctionShortcut(typed, caret, event.key) ?? expandMathShortcut(typed, caret, event.key)
          if (expanded) {
            event.preventDefault()
            placeMathInput(input, expanded.value, expanded.cursor, onValue, setCursor)
            return
          }
          if (event.key === '/' && typed.slice(0, caret).trim() !== '' && looksLikeMath(typed.slice(0, caret))) {
            event.preventDefault()
            const left = typed.slice(0, caret)
            const right = typed.slice(end)
            const next = `${left}/()${right}`
            placeMathInput(input, next, left.length + 2, onValue, setCursor)
            return
          }
          const slotted = insertMathSlot(typed, caret, end, event.key)
          if (slotted) {
            event.preventDefault()
            placeMathInput(input, slotted.value, slotted.cursor, onValue, setCursor)
            return
          }
          if (event.key === ',' && liveMath) {
            const exit = exitSlotsForComma(typed, caret)
            if (exit !== caret) {
              event.preventDefault()
              const left = typed.slice(0, exit)
              const right = typed.slice(Math.max(exit, end))
              placeMathInput(input, `${left},${right}`, left.length + 1, onValue, setCursor)
              return
            }
          }
          if (liveMath && (event.key === 'ArrowLeft' || event.key === 'ArrowRight' || event.key === 'ArrowUp' || event.key === 'ArrowDown')) {
            if (event.key === 'ArrowRight' && caret === end && caret === typed.length) {
              const closed = closeOpenGroups(typed)
              if (closed !== typed) {
                event.preventDefault()
                placeMathInput(input, closed, closed.length, onValue, setCursor)
                return
              }
            }
            event.preventDefault()
            const dir = event.key === 'ArrowLeft' ? 'left' : event.key === 'ArrowRight' ? 'right' : event.key === 'ArrowUp' ? 'up' : 'down'
            let next = moveMathCursor(typed, caret, dir)
            if ((dir === 'up' || dir === 'down') && next === caret && typed.includes('\n')) next = moveLine(typed, caret, dir)
            setCursor(next)
            input.setSelectionRange(next, next)
            return
          }
          if (liveMath) {
            if (event.key === 'Escape') onEscape?.()
            return
          }
          if (event.key === 'ArrowDown' || event.key === 'ArrowUp' || event.key === 'Escape') {
            if (event.key !== 'Escape') event.preventDefault()
            onCommandKey?.(event.key)
          }
        }}
      />
    </div>
  )
})

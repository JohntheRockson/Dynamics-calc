import { appendMath, emptyDocument, exampleDocument, setMathVisible } from '../document'
import { evaluateDocument } from '../evaluate'
import { expandMathShortcut, looksLikeMath } from './shortcuts'

export function runMathChecks(): string[] {
  const errors: string[] = []
  const expect = (cond: boolean, message: string) => {
    if (!cond) errors.push(message)
  }

  let doc = emptyDocument()
  for (const input of ['2+2', 'sqrt(4)', '1/2', 'ln(e)', 'log(100)', 'sin(pi/2)', 'f(x) = x^2', 'f(3)', 'solve(x^2 - 4 = 0)', 'solve(2*x + 1 = 5)', 'solve(x + y = 3, x)', 'solve(x^2 - 2 = 0)']) {
    doc = appendMath(doc, input)
  }
  const view = evaluateDocument(doc)
  const texts = view.blocks.flatMap((block) => block.rows.map((row) => row.text))
  const find = (part: string) => texts.find((text) => text.includes(part)) ?? ''

  expect(find('2 + 2').includes('4'), `2+2: ${texts.join(' | ')}`)
  expect(texts.some((text) => text.includes('sqrt(4)') && text.includes('2')), `sqrt(4): ${texts.join(' | ')}`)
  expect(texts.some((text) => text.includes('1/2')), `1/2: ${texts.join(' | ')}`)
  expect(texts.some((text) => text.includes('ln(e)') && text.includes('1')), `ln(e): ${texts.join(' | ')}`)
  expect(texts.some((text) => text.includes('log(100)') && text.includes('2')), `log: ${texts.join(' | ')}`)
  expect(texts.some((text) => text.includes('sin(pi/2)') && text.includes('1')), `sin: ${texts.join(' | ')}`)
  expect(texts.some((text) => text.includes('f(3)') && text.includes('9')), `f(3): ${texts.join(' | ')}`)
  expect(texts.some((text) => text.includes('x = 2') && text.includes('x = -2')), `quadratic: ${texts.join(' | ')}`)
  expect(texts.some((text) => text === 'x = 2'), `linear: ${texts.join(' | ')}`)
  expect(texts.some((text) => text.includes('3 - y') || text.includes('3-y') || text.includes('-y + 3')), `symbolic: ${texts.join(' | ')}`)
  expect(texts.some((text) => text.includes('sqrt(2)') && text.includes('-sqrt(2)')), `radical: ${texts.join(' | ')}`)

  let more = emptyDocument()
  for (const input of ['1/2+1/3', 'pi/2', 'a = 1/2', 'a+a', 'sqrt 4', 'log(8, 2)', '2sin(pi/2)']) more = appendMath(more, input)
  const moreText = evaluateDocument(more).blocks.flatMap((block) => block.rows.map((row) => row.text))
  expect(moreText.some((text) => text.includes('5/6')), `sum of fractions: ${moreText.join(' | ')}`)
  expect(moreText.some((text) => text === 'pi/2' || text.includes('pi/2') && !text.includes('1/2pi')), `pi/2: ${moreText.join(' | ')}`)
  expect(moreText.some((text) => text === 'a = 1/2'), `assignment: ${moreText.join(' | ')}`)
  expect(moreText.some((text) => text.includes('a + a') && text.includes('1')), `a+a: ${moreText.join(' | ')}`)
  expect(moreText.some((text) => text.includes('sqrt(4)') && text.includes('2')), `sqrt space: ${moreText.join(' | ')}`)
  expect(moreText.some((text) => text.includes('log(8, 2)') && text.includes('3')), `log base: ${moreText.join(' | ')}`)
  expect(moreText.some((text) => text.includes('2') && text.endsWith('2') && text.includes('sin')), `2sin: ${moreText.join(' | ')}`)

  const curve = view.bodies.find((body) => body.label === 'f(x)')
  expect(Boolean(curve && curve.path.some((point) => Math.abs(point.x) < 1e-9 && Math.abs(point.y) < 1e-6)), 'parabola sample at 0')
  expect(view.dimension === 2, `curve dimension ${view.dimension}`)
  expect(view.surfaces.length === 0, 'curve has no surface')

  let graphs = appendMath(emptyDocument(), 'f(x) = x')
  graphs = appendMath(graphs, 'g(x, y) = x + y')
  const both = evaluateDocument(graphs)
  expect(both.dimension === 3, `surface forces 3D: ${both.dimension}`)
  expect(both.bodies.some((body) => body.label === 'f(x)'), '2D curve stays in the 3D figure')
  expect(both.surfaces.length === 1, 'surface is drawn')
  const surfaceId = graphs.statements.find((statement) => statement.type === 'math' && statement.input.startsWith('g('))?.id
  expect(Boolean(surfaceId), 'surface id')
  if (surfaceId) {
    const hidden = evaluateDocument(setMathVisible(graphs, surfaceId, false))
    expect(hidden.dimension === 2, `hidden surface returns to 2D: ${hidden.dimension}`)
    expect(hidden.surfaces.length === 0, 'hidden surface is not drawn')
    expect(hidden.bodies.some((body) => body.label === 'f(x)'), 'curve remains after hiding the surface')
  }

  const onlySurface = evaluateDocument(appendMath(emptyDocument(), 'h(x, y) = x*y'))
  expect(onlySurface.dimension === 3, 'a surface alone is 3D')
  expect(onlySurface.bodies.length === 0, 'a surface is not drawn as a 2D curve')

  const withHelix = evaluateDocument(appendMath(exampleDocument('helix'), 'p(x) = x'))
  expect(withHelix.dimension === 3, 'helix stays 3D')
  expect(withHelix.bodies.some((body) => body.label === 'p(x)'), 'curve joins the 3D figure')

  const shortcut = expandMathShortcut('sqrt', 4, ' ')
  expect(shortcut?.value === 'sqrt(' && shortcut.cursor === 5, `sqrt shortcut: ${shortcut?.value}`)
  const digit = expandMathShortcut('sqrt', 4, '4')
  expect(digit?.value === 'sqrt(4' && digit.cursor === 6, `sqrt digit: ${digit?.value}`)
  expect(looksLikeMath('sqrt(4)') && looksLikeMath('f(x) = x') && looksLikeMath('e') && !looksLikeMath('point'), 'math detection')
  expect(!looksLikeMath('speed'), 'speed stays a statement')
  expect(find('1/2').length > 0, 'fraction row')

  return errors
}

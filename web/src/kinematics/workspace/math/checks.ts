import { chooseProbe } from '../../probe'
import { appendMath, convertDocumentAngles, emptyDocument, exampleDocument, setMathVisible } from '../document'
import { evaluateDocument } from '../evaluate'
import { convertAngleInput, previewTex, validateMath } from './expr'
import { clipToDomain, sheetsFromCurve } from './extrude'
import { formatTick, tickMarks } from '../../ticks'
import { closeOpenGroups, exitSlotsForComma, moveMathCursor } from './inputView'
import { axisThrough, expandPlotBox, fromWorld, originBox, toWorld, type PlotFrame } from './plotFrame'
import { emptyFunctionShortcut, expandMathShortcut, insertMathSlot, looksLikeMath } from './shortcuts'

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
  expect(shortcut?.value === 'sqrt()' && shortcut.cursor === 5, `sqrt shortcut: ${shortcut?.value}`)
  const digit = expandMathShortcut('sqrt', 4, '4')
  expect(digit?.value === 'sqrt(4)' && digit.cursor === 6, `sqrt digit: ${digit?.value}`)
  expect(looksLikeMath('sqrt(4)') && looksLikeMath('f(x) = x') && looksLikeMath('e') && !looksLikeMath('point'), 'math detection')
  expect(!looksLikeMath('speed'), 'speed stays a statement')
  expect(find('1/2').length > 0, 'fraction row')

  const radical = evaluateDocument(appendMath(emptyDocument(), 'sqrt(23)'))
  const radicalRow = radical.blocks.flatMap((block) => block.rows)[0]
  expect(Boolean(radicalRow?.preferDecimal && radicalRow.text.includes('4.795')), `sqrt(23) decimal: ${radicalRow?.text}`)
  expect(radicalRow?.input === 'sqrt(23)', `sqrt(23) input: ${radicalRow?.input}`)
  expect(Boolean(radicalRow?.exactText?.includes('sqrt(23)') && radicalRow.approxText?.includes('4.795')), 'sqrt(23) keeps both forms')

  const simplified = evaluateDocument(appendMath(emptyDocument(), 'sqrt(32)'))
  const simplifiedRow = simplified.blocks.flatMap((block) => block.rows)[0]
  expect(simplifiedRow?.preferDecimal === false && Boolean(simplifiedRow?.text.includes('sqrt(2)')), `sqrt(32) stays exact: ${simplifiedRow?.text}`)
  expect(Boolean(simplifiedRow?.approxText && simplifiedRow.approxText !== simplifiedRow.exactText), 'sqrt(32) offers a decimal')

  const parabola = evaluateDocument(appendMath(emptyDocument(), 'y = x^2'))
  const yCurve = parabola.bodies.find((body) => body.label === 'y')
  expect(parabola.dimension === 2 && Boolean(yCurve?.sample), 'y = x^2 is a 2D curve')
  const shifted = yCurve?.sample?.({ xMin: 2, xMax: 4, yMin: 0, yMax: 20 }) ?? []
  const xs = shifted.filter((point) => Number.isFinite(point.x)).map((point) => point.x)
  expect(xs.length > 0 && Math.min(...xs) < 2.05 && Math.max(...xs) > 3.95, `y = x^2 fills the window: ${Math.min(...xs)} to ${Math.max(...xs)}`)

  const sideways = evaluateDocument(appendMath(emptyDocument(), 'y^2 = x'))
  const branch = sideways.bodies.find((body) => body.label === 'y^2')
  const around4 = (branch?.path ?? []).filter((point) => Number.isFinite(point.y) && Math.abs(point.x - 4) < 0.2)
  expect(around4.some((point) => point.y > 1) && around4.some((point) => point.y < -1), 'y^2 = x draws both branches')

  const vertical = evaluateDocument(appendMath(emptyDocument(), 'x = 2'))
  const line = vertical.bodies.find((body) => body.label === 'x')
  const finiteLine = (line?.path ?? []).filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y))
  expect(finiteLine.length > 10 && finiteLine.every((point) => Math.abs(point.x - 2) < 1e-6), 'x = 2 is a vertical line')
  expect(Math.min(...finiteLine.map((point) => point.y)) < -9 && Math.max(...finiteLine.map((point) => point.y)) > 9, 'x = 2 spans the default window')

  const saddle = evaluateDocument(appendMath(emptyDocument(), 'z = x^2 - y^2'))
  expect(saddle.dimension === 3 && saddle.surfaces.length === 1, `z = is a surface: ${saddle.dimension}, ${saddle.surfaces.length}`)
  const cone = evaluateDocument(appendMath(emptyDocument(), 'z^2 = x^2 + y^2'))
  expect(cone.dimension === 3 && (cone.surfaces[0]?.sheets?.length ?? 0) === 2, `z^2 draws two sheets: ${cone.surfaces[0]?.sheets?.length}`)

  const solved = evaluateDocument(appendMath(emptyDocument(), 'solve(x + 1 = 4)'))
  const solvedRow = solved.blocks.flatMap((block) => block.rows)[0]
  expect(solvedRow?.input === 'solve(x + 1 = 4)' && Boolean(solvedRow.text.includes('x = 3')), `solve keeps the input: ${solvedRow?.input} -> ${solvedRow?.text}`)

  const sqrtPreview = previewTex('sqrt(4)')
  expect(Boolean(sqrtPreview?.includes('\\sqrt') && !sqrtPreview.includes('= 2')), `preview keeps sqrt(4): ${sqrtPreview}`)
  expect(Boolean(previewTex('ln(2)/3')?.includes('\\frac')), `fraction preview: ${previewTex('ln(2)/3')}`)
  expect(Boolean(previewTex('x^2')?.includes('^')), `power preview: ${previewTex('x^2')}`)
  expect(Boolean(previewTex('sqrt(')?.includes('\\square')), `open sqrt preview: ${previewTex('sqrt(')}`)
  expect(Boolean(previewTex('2+')?.includes('\\square')), `trailing operator preview: ${previewTex('2+')}`)
  const blankFn = emptyFunctionShortcut('', 0, '/')
  expect(blankFn?.value === 'f(x) = ' && blankFn.cursor === 7, `slash starts a function: ${blankFn?.value}`)
  const surfaceFn = emptyFunctionShortcut('', 0, '/')
  expect(surfaceFn?.value === 'f(x, y) = ', `a second slash takes x and y: ${surfaceFn?.value}`)
  const upgraded = emptyFunctionShortcut('f(x) = ', 7, '/')
  expect(upgraded?.value === 'f(x, y) = ' && upgraded.cursor === 10, `slash upgrades the one-input template: ${upgraded?.value} @ ${upgraded?.cursor}`)
  expect(emptyFunctionShortcut('1', 1, '/') === null, 'slash after a value stays division')
  expect(Boolean(previewTex('f(x) =')?.includes('\\square')), `empty function preview: ${previewTex('f(x) =')}`)

  const deg = evaluateDocument(appendMath(emptyDocument(), 'sin(30)'), 0, 'deg')
  const degText = deg.blocks.flatMap((block) => block.rows)[0]?.text ?? ''
  expect(degText.includes('1/2'), `sin(30) degrees: ${degText}`)
  const cos45 = evaluateDocument(appendMath(emptyDocument(), 'cos(45)'), 0, 'deg').blocks.flatMap((block) => block.rows)[0]?.text ?? ''
  expect(cos45.includes('sqrt(2)'), `cos(45) degrees: ${cos45}`)
  const right = evaluateDocument(appendMath(emptyDocument(), 'asin(1)'), 0, 'deg').blocks.flatMap((block) => block.rows)[0]?.text ?? ''
  expect(right.includes('90'), `asin(1) degrees: ${right}`)
  const wave = evaluateDocument(appendMath(emptyDocument(), 'y = sin(x)'), 0, 'deg')
  const at90 = wave.bodies[0]?.sample?.({ xMin: 90, xMax: 90, yMin: -2, yMax: 2 }) ?? []
  expect(at90.some((point) => Math.abs(point.y - 1) < 1e-6), `degree graph at 90: ${at90[0]?.y}`)
  const stillRad = evaluateDocument(appendMath(emptyDocument(), 'sin(pi/2)')).blocks.flatMap((block) => block.rows)[0]?.text ?? ''
  expect(stillRad.includes('1') && !stillRad.includes('90'), `radians stay default: ${stillRad}`)

  const sheets = sheetsFromCurve(
    [
      { x: -2, y: -1, z: 0 },
      { x: 0, y: 5, z: 0 },
      { x: 2, y: 11, z: 0 },
    ],
    0,
    10,
    3,
  )
  expect(sheets.length === 1 && sheets[0].length === 3, 'a curve sweeps into one sheet')
  expect(sheets[0][0][0].z === 0 && sheets[0][2][0].z === 10 && sheets[0][1][1].y === 5, 'the sheet keeps the curve and spans z')
  const split = sheetsFromCurve(
    [
      { x: 0, y: 1, z: 0 },
      { x: 1, y: 2, z: 0 },
      { x: Number.NaN, y: Number.NaN, z: Number.NaN },
      { x: 0, y: -1, z: 0 },
      { x: 1, y: -2, z: 0 },
    ],
    -4,
    4,
    2,
  )
  expect(split.length === 2, `a gap stays two sheets: ${split.length}`)

  const tall: PlotFrame = { equal: false, box: expandPlotBox({ xMin: -10, xMax: 10, yMin: -10, yMax: 10, zMin: 0, zMax: 200 }) }
  const tip = toWorld(tall, 0, 0, 0)
  const rim = toWorld(tall, 10, 10, 200)
  expect(Math.abs(tip.y + 5) < 1e-6 && Math.abs(tip.x) < 1e-6, `paraboloid tip sits on the cube floor: ${tip.x}, ${tip.y}`)
  expect(Math.abs(rim.x - 5) < 1e-6 && Math.abs(rim.y - 5) < 1e-6 && Math.abs(rim.z - 5) < 1e-6, `tall surface fills the cube: ${rim.x}, ${rim.y}, ${rim.z}`)
  const back = fromWorld(tall, rim.x, rim.y, rim.z)
  expect(Math.abs(back.x - 10) < 1e-6 && Math.abs(back.y - 10) < 1e-6 && Math.abs(back.z - 200) < 1e-6, 'cube positions convert back to math')
  const clipped = clipToDomain(
    [
      { x: -10, y: -25, z: 0 },
      { x: -5, y: -10, z: 0 },
      { x: 0, y: 5, z: 0 },
      { x: 2, y: 11, z: 0 },
    ],
    { xMin: -10, xMax: 10, yMin: -10, yMax: 10 },
  )
  expect(Number.isFinite(clipped[2].y) && !Number.isFinite(clipped[0].y) && !Number.isFinite(clipped[3].y), 'a plane stays inside the surface domain')

  const flat = expandPlotBox({ xMin: -10, xMax: 10, yMin: -10, yMax: 10, zMin: 0, zMax: 0 })
  expect(flat.zMax - flat.zMin > 1, 'a flat curve gets a z range')
  const motion = toWorld({ equal: true, box: flat }, 1, 2, 3)
  expect(motion.x === 1 && motion.y === 3 && motion.z === 2, 'motion keeps equal meters')
  const centered = originBox(10, 200)
  const origin = toWorld({ equal: false, box: centered }, 0, 0, 0)
  const corner = toWorld({ equal: false, box: centered }, 10, 10, 200)
  expect(Math.abs(origin.x) < 1e-6 && Math.abs(origin.y) < 1e-6 && Math.abs(origin.z) < 1e-6, 'the 3D box is centered on the origin')
  expect(Math.abs(corner.x - 5) < 1e-6 && Math.abs(corner.y - 5) < 1e-6 && Math.abs(corner.z - 5) < 1e-6, 'a centered box still fills the cube')
  const parked = tickMarks(-10, 10, 2)
  const movedTicks = tickMarks(-9, 11, 2)
  expect(parked[0] === -10 && movedTicks[0] === -8 && parked.includes(0) && movedTicks.includes(0), `grid lines stay on world numbers: ${parked.join(',')} -> ${movedTicks.join(',')}`)

  let algebra = emptyDocument()
  for (const input of [
    'decimal(1/2)',
    'fraction(0.333)',
    'factor(12)',
    'gcd(12, 18)',
    'lcm(4, 6)',
    'mod(7, 3)',
    'expand((x+1)*(x+2))',
    'zeros(x^2 - 1)',
    'solve(x + y = 3, x - y = 1)',
    'diff(x^2, x)',
    'diff(x^3, x, 2)',
    'diff(x^2, x, 1, 3)',
    'integrate(x^2, x)',
    'integrate(x, x, 0, 1)',
    'limit(sin(x)/x, x, 0)',
    'sum(i, i, 1, 5)',
    'prod(i, i, 1, 4)',
    'fmin(x^2, x, -2, 2)',
    'fmax(-x^2, x, -2, 2)',
    'tangent(x^2, x, 1)',
    'normal(x^2, x, 1)',
    'series(exp(x), x, 0, 3)',
    'dsolve(diff(y, x) = y, y, x)',
    'idiff(x^2 + y^2 = 1, y, x)',
  ]) algebra = appendMath(algebra, input)
  const algebraText = evaluateDocument(algebra).blocks.flatMap((block) => block.rows.map((row) => row.text))
  const algebraPlots = evaluateDocument(algebra).bodies.filter((body) => body.role === 'plot')
  const has = (part: string) => algebraText.some((text) => text.includes(part))
  expect(has('0.5'), `decimal: ${algebraText.join(' | ')}`)
  expect(has('1/3'), `fraction: ${algebraText.join(' | ')}`)
  expect(has('2^2') && has('3'), `factor: ${algebraText.join(' | ')}`)
  expect(has('6'), `gcd: ${algebraText.join(' | ')}`)
  expect(has('12'), `lcm: ${algebraText.join(' | ')}`)
  expect(has('1'), `mod: ${algebraText.join(' | ')}`)
  expect(algebraText.some((text) => text.includes('x^2') && text.includes('3')), `expand: ${algebraText.join(' | ')}`)
  expect(algebraText.some((text) => text.includes('x = 1') && text.includes('x = -1')), `zeros: ${algebraText.join(' | ')}`)
  expect(algebraText.some((text) => text.includes('x = 2') && text.includes('y = 1')), `system: ${algebraText.join(' | ')}`)
  expect(algebraText.some((text) => text.includes('2x') && text.includes('diff(x^2, x)')), `derivative: ${algebraText.join(' | ')}`)
  expect(has('6x'), `second derivative: ${algebraText.join(' | ')}`)
  expect(algebraText.some((text) => text.includes('diff(x^2, x, 1, 3)') && text.includes('6')), `derivative at a point: ${algebraText.join(' | ')}`)
  expect(has('x^3/3') && has('C'), `integral: ${algebraText.join(' | ')}`)
  expect(algebraText.some((text) => text.includes('integrate(x, x, 0, 1)') && text.includes('1/2')), `definite integral: ${algebraText.join(' | ')}`)
  expect(algebraText.some((text) => text.includes('limit') && text.includes('1')), `limit: ${algebraText.join(' | ')}`)
  expect(algebraText.some((text) => text.includes('sum') && text.includes('15')), `sum: ${algebraText.join(' | ')}`)
  expect(algebraText.some((text) => text.includes('prod') && text.includes('24')), `product: ${algebraText.join(' | ')}`)
  expect(algebraText.some((text) => text.includes('minimum') && text.includes('0')), `minimum: ${algebraText.join(' | ')}`)
  expect(algebraText.some((text) => text.includes('maximum') && text.includes('0')), `maximum: ${algebraText.join(' | ')}`)
  expect(algebraText.some((text) => text.includes('tangent') && text.includes('2x')), `tangent: ${algebraText.join(' | ')}`)
  expect(has('normal'), `normal: ${algebraText.join(' | ')}`)
  expect(algebraText.some((text) => text.includes('series') && text.includes('x^2')), `series: ${algebraText.join(' | ')}`)
  expect(algebraText.some((text) => text.includes('exp') && text.includes('C')), `differential equation: ${algebraText.join(' | ')}`)
  expect(algebraText.some((text) => text.includes('idiff') && text.includes('x') && text.includes('y')), `implicit: ${algebraText.join(' | ')}`)
  expect(algebraPlots.some((body) => body.label === 'diff' || body.label === 'Derivative' || body.path.length > 2), `derivative is drawn: ${algebraPlots.map((body) => body.label).join(',')}`)

  const degreeDerivative = evaluateDocument(appendMath(emptyDocument(), 'diff(sin(x), x)'), 0, 'deg').blocks.flatMap((block) => block.rows.map((row) => row.text))
  expect(degreeDerivative.some((text) => text.includes('pi') && text.includes('180')), `degree derivative follows the angle mode: ${degreeDerivative.join(' | ')}`)

  const constantDerivative = evaluateDocument(appendMath(emptyDocument(), 'diff(x^3, x, 3)'))
  const constantText = constantDerivative.blocks.flatMap((block) => block.rows.map((row) => row.text))
  const constantPath = constantDerivative.bodies.filter((body) => body.role === 'plot').flatMap((body) => body.path)
  const constantYs = constantPath.filter((point) => Number.isFinite(point.y)).map((point) => point.y)
  expect(constantText.some((text) => text.includes('6')), `third derivative: ${constantText.join(' | ')}`)
  expect(constantYs.length > 2 && constantYs.every((y) => Math.abs(y - 6) < 1e-6), 'a constant derivative is a horizontal line')
  const atPoint = evaluateDocument(appendMath(emptyDocument(), 'diff(x^3, x, 3, 1)'))
  expect(atPoint.bodies.filter((body) => body.role === 'plot').length === 0, 'a derivative at a point stays a number')

  const integral = evaluateDocument(appendMath(emptyDocument(), 'integrate(x, x, 0, 1)'))
  const integralBody = integral.bodies.find((body) => body.role === 'plot')
  expect(Boolean(integralBody?.shade && integralBody.shade.from === 0 && integralBody.shade.to === 1 && !integralBody.hideStroke && integralBody.path.length > 2), 'a definite integral shades the integrand')
  let shaded = emptyDocument()
  shaded = appendMath(shaded, 'y = x')
  shaded = appendMath(shaded, 'integrate(x, x, 0, 1)')
  const shadedBodies = evaluateDocument(shaded).bodies.filter((body) => body.role === 'plot')
  const shadedIntegral = shadedBodies.find((body) => body.shade)
  expect(Boolean(shadedIntegral?.hideStroke) && shadedBodies.filter((body) => !body.hideStroke).length === 1, 'the integrand is not drawn twice')

  let linear = emptyDocument()
  for (const input of [
    'dot([1, 2], [3, 4])',
    'cross([1, 0, 0], [0, 1, 0])',
    'unit([3, 4])',
    'norm([3, 4])',
    'det([[1, 2], [3, 4]])',
    '[[1, 2], [3, 4]]*[1, 0]',
    'r(t) = [cos(t), sin(t)]',
    's(t) = [cos(t), sin(t), t]',
  ]) linear = appendMath(linear, input)
  const linearView = evaluateDocument(linear)
  const linearText = linearView.blocks.flatMap((block) => block.rows.map((row) => row.text))
  expect(linearText.some((text) => text.includes('dot') && text.includes('11')), `dot product: ${linearText.join(' | ')}`)
  expect(linearText.some((text) => text.includes('[0, 0, 1]')), `cross product: ${linearText.join(' | ')}`)
  expect(linearText.some((text) => text.includes('3/5') && text.includes('4/5')), `unit vector: ${linearText.join(' | ')}`)
  expect(linearText.some((text) => text.includes('norm') && text.includes('= 5')), `norm: ${linearText.join(' | ')}`)
  expect(linearText.some((text) => text.includes('-2')), `determinant: ${linearText.join(' | ')}`)
  expect(linearText.some((text) => text.includes('[1, 3]')), `matrix times vector: ${linearText.join(' | ')}`)
  const circle = linearView.bodies.find((body) => body.label.includes('r(t)'))
  expect(Boolean(circle && circle.path.filter((point) => Number.isFinite(point.x)).length > 20), 'a parametric curve is drawn')
  expect(linearView.dimension === 3, 'a three-component vector function is 3D')
  const arrow = evaluateDocument(appendMath(emptyDocument(), '[3, 4]')).bodies.find((body) => body.arrow)
  expect(Boolean(arrow && arrow.path.some((point) => Math.abs(point.x - 3) < 1e-6 && Math.abs(point.y - 4) < 1e-6)), 'a constant vector is an arrow')

  const scale = (x: number, y: number) => ({ x: x * 40, y: -y * 40 })
  const crossing = chooseProbe(
    [
      [{ x: 0, y: 0 }, { x: 2, y: 2 }],
      [{ x: 0, y: 2 }, { x: 2, y: 0 }],
    ],
    { x: 1.05, y: 0.95 },
    scale,
  )
  const nearby = chooseProbe(
    [
      [{ x: 0, y: 0 }, { x: 2, y: 2 }],
      [{ x: 0, y: 2 }, { x: 2, y: 0 }],
    ],
    { x: 0.5, y: 0.5 },
    scale,
  )
  expect(crossing?.kind === 'intersection' && Math.abs((crossing?.x ?? 0) - 1) < 1e-6, `snap to an intersection: ${crossing?.text ?? 'none'}`)
  expect(nearby?.kind === 'point' && Math.abs((nearby?.x ?? 0) - 1) > 0.4, `a click away from the crossing stays on the line: ${nearby?.text ?? 'none'}`)
  const vertex = chooseProbe(yCurve ? [yCurve.path] : [], { x: 0.05, y: 0.2 }, scale)
  expect(vertex?.kind === 'minimum' && Math.abs(vertex.x) < 1e-6 && Math.abs(vertex.y) < 1e-6, `snap to a minimum: ${vertex?.text ?? 'none'}`)

  expect(looksLikeMath('theta') && looksLikeMath('alpha') && looksLikeMath('e_r') && looksLikeMath("theta'"), 'greek names and subscripts are math')
  expect(!looksLikeMath('point') && !looksLikeMath('speed'), 'statement names stay statements')
  const thetaPreview = previewTex('theta')
  const alphaPreview = previewTex('alpha')
  const subscriptPreview = previewTex('e_theta')
  const openSubscript = previewTex('e_')
  const dottedPower = previewTex("theta'^2")
  expect(Boolean(thetaPreview?.includes('\\theta')), `theta previews as a symbol: ${thetaPreview}`)
  expect(Boolean(alphaPreview?.includes('\\alpha')), `alpha previews as a symbol: ${alphaPreview}`)
  expect(Boolean(subscriptPreview?.includes('\\mathbf{e}') && subscriptPreview.includes('\\theta')), `e_theta is a subscript: ${subscriptPreview}`)
  expect(Boolean(openSubscript?.includes('\\square')), `a trailing _ keeps a subscript hole: ${openSubscript}`)
  expect(Boolean(dottedPower?.includes('\\dot') && dottedPower.includes('^')), `a prime binds before a power: ${dottedPower}`)

  const timeDerivative = evaluateDocument(appendMath(emptyDocument(), "diff(r*theta'*e_theta, t)"))
  const motionRow = timeDerivative.blocks.flatMap((block) => block.rows)[0]
  expect(Boolean(motionRow?.text.includes("r'") && motionRow.text.includes("theta''") && motionRow.text.includes("e_theta'")), `time derivative: ${motionRow?.text}`)
  expect(Boolean(motionRow?.tex?.includes('\\dot') && motionRow.tex.includes('\\ddot') && motionRow.tex.includes('\\mathbf{e}') && motionRow.tex.includes('\\theta')), `time derivative tex: ${motionRow?.tex}`)
  const partial = evaluateDocument(appendMath(emptyDocument(), 'diff(x^2, x)')).blocks.flatMap((block) => block.rows)[0]
  expect(Boolean(partial?.text.includes('2x') && !partial.text.includes("x'")), `partial derivative stays partial: ${partial?.text}`)
  const chain = evaluateDocument(appendMath(emptyDocument(), 'diff(sin(theta), t)')).blocks.flatMap((block) => block.rows)[0]
  expect(Boolean(chain?.text.includes('cos') && chain.text.includes("theta'")), `theta depends on time: ${chain?.text}`)
  const undone = evaluateDocument(appendMath(emptyDocument(), "integrate(theta', t)")).blocks.flatMap((block) => block.rows)[0]
  expect(Boolean(undone?.tex?.includes('\\theta') && undone.tex.includes('C') && !undone.tex.includes('\\dot')), `a dot integrates back: ${undone?.tex}`)

  const decimalPreview = previewTex('32.2')
  expect(Boolean(decimalPreview?.includes('32.2') && !decimalPreview.includes('161')), `a typed decimal stays a decimal: ${decimalPreview}`)
  const decimalRow = evaluateDocument(appendMath(emptyDocument(), '32.2')).blocks.flatMap((block) => block.rows)[0]
  expect(Boolean(decimalRow?.text.includes('32.2') && !decimalRow.text.includes('161')), `decimal output: ${decimalRow?.text}`)
  expect(Boolean(previewTex('(1+2)')?.includes('\\left(')), `grouping parentheses stay visible: ${previewTex('(1+2)')}`)
  const flatFraction = previewTex('15*cos(30)/2*30') ?? ''
  expect(flatFraction.includes('\\frac') && flatFraction.includes('}{2}') && flatFraction.includes('\\cdot'), `a product after a fraction stays outside: ${flatFraction}`)
  const heldFraction = previewTex('15*cos(30)/(2*30)') ?? ''
  expect(/\\frac\{[^}]+\}\{[^}]*2[^}]*30[^}]*\}/.test(heldFraction), `parentheses keep the product in the denominator: ${heldFraction}`)
  const starPreview = previewTex('15*', 3) ?? ''
  expect(starPreview.includes('\\cdot') && starPreview.includes('\\rule') && !starPreview.includes('\\square'), `a trailing star keeps a caret: ${starPreview}`)
  const insideCall = previewTex('cos(30)', 4) ?? ''
  const cosAt = insideCall.indexOf('\\cos')
  const parenAt = insideCall.indexOf('\\left(')
  const ruleAt = insideCall.indexOf('\\rule')
  const thirtyAt = insideCall.indexOf('30')
  expect(cosAt >= 0 && parenAt > cosAt && ruleAt > parenAt && thirtyAt > ruleAt, `the caret sits inside the call: ${insideCall}`)
  expect(validateMath('\\frac{1}{2}') === null, 'a latex fraction parses')
  const pastedHalf = evaluateDocument(appendMath(emptyDocument(), '\\frac{1}{2}')).blocks.flatMap((block) => block.rows)[0]
  expect(pastedHalf?.source !== 'error' && Boolean(pastedHalf?.text.includes('1/2')), `latex fraction evaluates: ${pastedHalf?.text}`)
  const pastedCos = evaluateDocument(appendMath(emptyDocument(), '\\frac{15\\cos\\left(30\\right)}{2 \\cdot 30 + 15\\sin\\left(30\\right)}')).blocks.flatMap((block) => block.rows)[0]
  expect(pastedCos?.source !== 'error', `latex cosine fraction evaluates: ${pastedCos?.text}`)
  const slot = '15*cos(30)/(2)'
  const closing = slot.lastIndexOf(')')
  expect(moveMathCursor(slot, closing, 'right') === slot.length, 'right leaves the denominator')
  expect(moveMathCursor(slot, closing, 'up') === slot.indexOf('/'), 'up returns to the numerator')
  const groupedProduct = evaluateDocument(appendMath(emptyDocument(), '(1+2)*3')).blocks.flatMap((block) => block.rows)[0]
  expect(Boolean(groupedProduct?.text.includes('9')), `grouping still multiplies: ${groupedProduct?.text}`)

  expect(formatTick(10, 2) === '10' && formatTick(20, 2) === '20' && formatTick(-10, 2) === '-10', `tick labels keep their zeros: ${formatTick(10, 2)}, ${formatTick(20, 2)}`)
  expect(formatTick(2, 0.5) === '2' && formatTick(1.5, 0.5) === '1.5', `fractional ticks still trim: ${formatTick(2, 0.5)}`)

  const greekPreview = previewTex('f(sigma) = sigma') ?? ''
  const greekRow = evaluateDocument(appendMath(emptyDocument(), 'f(theta) = theta^2')).blocks.flatMap((block) => block.rows)[0]
  expect(greekPreview.startsWith('f\\left(\\sigma\\right)'), `a parameter keeps the greek symbol: ${greekPreview}`)
  expect(Boolean(greekRow?.tex?.startsWith('f\\left(\\theta\\right)')), `a saved parameter keeps the greek symbol: ${greekRow?.tex}`)

  const singular = evaluateDocument(appendMath(emptyDocument(), 'f(x) = 1/cos(x)^2'))
  const singularPath = singular.bodies.find((body) => body.role === 'plot')?.path ?? []
  const pole = Math.PI / 2
  let beforePole = false
  let brokeAtPole = false
  let connectedAcross = false
  for (const point of singularPath) {
    const finite = Number.isFinite(point.x) && Number.isFinite(point.y)
    if (!finite) {
      if (beforePole) brokeAtPole = true
      continue
    }
    if (point.x < pole) {
      beforePole = true
      brokeAtPole = false
    } else if (beforePole) {
      connectedAcross = !brokeAtPole
      break
    }
  }
  expect(beforePole && !connectedAcross, `singular curves break at a pole: before ${beforePole}, connected ${connectedAcross}`)

  const coarse = evaluateDocument(appendMath(emptyDocument(), 'g(x) = sin(x), plotpoints = 20, maxrecursion = 0, exclusions = false'))
  const coarsePath = coarse.bodies.find((body) => body.role === 'plot')?.path.filter((point) => Number.isFinite(point.y)) ?? []
  expect(coarsePath.length === 20, `plotpoints sets the sample count: ${coarsePath.length}`)

  const nonlinear = evaluateDocument(appendMath(emptyDocument(), 'solve(10 = sigma*(cos(theta))^2, 5 = sigma*(sin(theta))^2, (0, 2*pi))'))
  const nonlinearRow = nonlinear.blocks.flatMap((block) => block.rows)[0]
  expect(Boolean(nonlinearRow?.text.includes('15') && nonlinearRow.text.includes('theta') && !nonlinearRow.text.includes('not linear')), `a domain searches a nonlinear system: ${nonlinearRow?.text}`)
  const missingDomain = evaluateDocument(appendMath(emptyDocument(), 'solve(10 = sigma*(cos(theta))^2, 5 = sigma*(sin(theta))^2)'))
  const missingRow = missingDomain.blocks.flatMap((block) => block.rows)[0]
  expect(Boolean(missingRow?.text.includes('domain')), `a nonlinear system asks for a domain: ${missingRow?.text}`)

  expect(moveMathCursor('sigma', 3, 'right') === 5 && moveMathCursor('sigma', 3, 'left') === 0, 'arrows jump a whole name')
  expect(moveMathCursor('sigma', 0, 'right') === 5 && moveMathCursor('sigma', 5, 'left') === 0, 'arrows cross a name in one step')
  const splitName = previewTex('sigma', 3) ?? ''
  expect(splitName.includes('\\sigma') && splitName.includes('\\rule') && !splitName.includes('\\cdot'), `a caret inside a name stays one symbol: ${splitName}`)
  const named = '15*cos(30)/(2)'
  expect(moveMathCursor(named, named.indexOf('c'), 'right') === named.indexOf('('), 'a name jump stops before a parenthesis')
  const braced = 'e_{' + 'theta}'
  expect(moveMathCursor(braced, braced.indexOf('h'), 'right') === braced.length, 'right leaves a subscript')
  const emptySub = 'e_{}'
  expect(moveMathCursor(emptySub, emptySub.indexOf('}'), 'left') === emptySub.indexOf('_'), 'left leaves a subscript')
  expect(moveMathCursor(emptySub, emptySub.indexOf('}'), 'right') === emptySub.length, 'right at a brace leaves the subscript')
  const exponent = 'x^()'
  expect(moveMathCursor(exponent, exponent.indexOf('^'), 'right') === exponent.length - 1, 'right enters an exponent')
  expect(moveMathCursor(exponent, exponent.length - 1, 'right') === exponent.length, 'right leaves an exponent')
  expect(moveMathCursor(exponent, exponent.length - 1, 'left') === exponent.indexOf('^'), 'left leaves an exponent')
  const nested = '15/x^(2)'
  expect(moveMathCursor(nested, nested.indexOf('2'), 'up') === nested.indexOf('/'), 'up still leaves a fraction')
  expect(expandMathShortcut('sqrt', 4, '^') === null, 'a caret does not turn sqrt into a call')
  const opened = insertMathSlot('x', 1, 1, '^')
  expect(opened?.value === 'x^()' && opened.cursor === 3, `caret opens an exponent: ${opened?.value}`)
  const subscripted = insertMathSlot('e', 1, 1, '_')
  expect(subscripted?.value === 'e_{}' && subscripted.cursor === 3, `underscore opens a subscript: ${subscripted?.value}`)
  const wrapped = insertMathSlot('', 0, 0, '(')
  expect(wrapped?.value === '()' && wrapped.cursor === 1, `an opening parenthesis inserts a pair: ${wrapped?.value}`)
  expect(insertMathSlot('point', 5, 5, '(')?.value === 'point' && !looksLikeMath('point('), 'point stays a statement')
  expect(insertMathSlot('cos(30)', 6, 6, ')')?.cursor === 7, 'a typed parenthesis steps past the closer')
  expect(closeOpenGroups('cos(30') === 'cos(30)' && closeOpenGroups('decimal(') === 'decimal()', 'right closes an open group')
  const emptyParens = previewTex('()', 1) ?? ''
  expect(emptyParens.includes('\\rule') && emptyParens.includes('\\vphantom') && !emptyParens.includes('\\left('), `empty parentheses stay one size: ${emptyParens}`)
  const powerTex = previewTex('x^(2*3)') ?? ''
  const exponentTex = powerTex.split('^')[1] ?? ''
  expect(powerTex.includes('\\cdot') && !exponentTex.includes('\\left('), `an exponent drops one set of parentheses: ${powerTex}`)
  const subCaret = previewTex('e_{}', 3) ?? ''
  expect(subCaret.includes('_{') && subCaret.includes('\\rule') && subCaret.includes('\\mathbf{e}'), `the subscript caret sits in the subscript: ${subCaret}`)

  const bigScientific = previewTex('4.7947e23') ?? ''
  expect(bigScientific.includes('\\times 10^{23}') && !/\de[+-]?\d/i.test(bigScientific), `a typed exponential number is prettied: ${bigScientific}`)
  const smallDecimalRow = evaluateDocument(appendMath(emptyDocument(), 'sqrt(23)/10000000')).blocks.flatMap((block) => block.rows)[0]
  expect(Boolean(smallDecimalRow?.tex?.includes('\\times 10^{-')), `a computed tiny decimal is prettied: ${smallDecimalRow?.tex}`)

  // The auto-pair always closes an exponent the instant `^` is typed, so live typing of
  // `...theta))^2` leaves the caret right after "2" with the closing paren still unconsumed.
  const exponentComma = 'solve(10=sigma*(cos(theta))^(2)'
  const exponentCaret = exponentComma.length - 1
  expect(exitSlotsForComma(exponentComma, exponentCaret) === exponentComma.length, `a comma steps out of an open exponent: ${exitSlotsForComma(exponentComma, exponentCaret)}`)
  const openCall = 'f(1)'
  expect(exitSlotsForComma(openCall, openCall.length - 1) === openCall.length - 1, 'a comma stays inside an explicit call')
  const nestedExponent = 'a^(2^(3))'
  expect(exitSlotsForComma(nestedExponent, nestedExponent.length - 2) === nestedExponent.length, `a comma steps out of every nested exponent: ${exitSlotsForComma(nestedExponent, nestedExponent.length - 2)}`)
  const optionsWithCaret = previewTex('f(x)=sin(x),plotpoints=160,maxrecursion=6', 20)
  expect(Boolean(optionsWithCaret?.includes('\\sin')), `a caret inside the plot options tail still previews: ${optionsWithCaret}`)
  expect(Boolean(previewTex('f(x)=sin(x),plotpoints=160,maxrecursion=6')?.includes('\\sin')), 'the plot options tail still previews without a caret')

  expect(convertAngleInput('sin(pi)', 'rad', 'deg') === 'sin(180)', `radians pi becomes 180 degrees: ${convertAngleInput('sin(pi)', 'rad', 'deg')}`)
  expect(convertAngleInput('sin(180)', 'deg', 'rad') === 'sin(pi)', `180 degrees becomes pi: ${convertAngleInput('sin(180)', 'deg', 'rad')}`)
  expect(convertAngleInput('sin(pi/2)', 'rad', 'deg') === 'sin(90)', `pi/2 becomes 90: ${convertAngleInput('sin(pi/2)', 'rad', 'deg')}`)
  expect(convertAngleInput('sin(90)', 'deg', 'rad') === 'sin(pi/2)', `90 becomes pi/2: ${convertAngleInput('sin(90)', 'deg', 'rad')}`)
  expect(convertAngleInput('sin(30)', 'deg', 'rad') === 'sin(pi/6)', `30 degrees becomes pi/6: ${convertAngleInput('sin(30)', 'deg', 'rad')}`)
  expect(convertAngleInput('cos(45)', 'deg', 'rad') === 'cos(pi/4)', `45 degrees becomes pi/4: ${convertAngleInput('cos(45)', 'deg', 'rad')}`)
  expect(convertAngleInput('tan(60)', 'deg', 'rad') === 'tan(pi/3)', `60 degrees becomes pi/3: ${convertAngleInput('tan(60)', 'deg', 'rad')}`)
  expect(convertAngleInput('sin(2)', 'rad', 'deg') === 'sin(360/pi)', `a bare radian stays exact: ${convertAngleInput('sin(2)', 'rad', 'deg')}`)
  expect(convertAngleInput('y = sin(x)', 'rad', 'deg') === 'y = sin(x)', 'a free variable stays written the way it was')
  expect(convertAngleInput('asin(1)', 'rad', 'deg') === 'asin(1)', 'inverse trig input is not an angle')
  expect(convertAngleInput('diff(sin(x), x)', 'rad', 'deg') === 'diff(sin(x), x)', 'a derivative in x stays written in x')
  expect(convertAngleInput('sin(x + pi)', 'rad', 'deg') === 'sin(x + 180)', `a constant offset converts: ${convertAngleInput('sin(x + pi)', 'rad', 'deg')}`)
  expect(convertAngleInput('f(x) = sin(x), plotpoints = 160', 'rad', 'deg') === 'f(x) = sin(x), plotpoints = 160', 'an unchanged curve keeps its plot options')
  const withPoints = convertAngleInput('f(x) = sin(pi), plotpoints = 160', 'rad', 'deg')
  expect(withPoints === 'f(x) = sin(180), plotpoints = 160', `a converted curve keeps its plot options: ${withPoints}`)
  let angleDoc = appendMath(emptyDocument(), 'sin(pi)')
  const radAngle = evaluateDocument(angleDoc).blocks.flatMap((block) => block.rows)[0]?.text ?? ''
  expect(radAngle.includes('sin(pi)') && radAngle.includes('0') && !radAngle.includes('0.05'), `sin(pi) radians: ${radAngle}`)
  angleDoc = convertDocumentAngles(angleDoc, 'rad', 'deg')
  const degStatement = angleDoc.statements.find((statement) => statement.type === 'math')
  expect(degStatement?.type === 'math' && degStatement.input === 'sin(180)', `stored input becomes sin(180): ${degStatement?.type === 'math' ? degStatement.input : ''}`)
  const degAngle = evaluateDocument(angleDoc, 0, 'deg').blocks.flatMap((block) => block.rows)[0]?.text ?? ''
  expect(degAngle.includes('180') && degAngle.includes('0') && !degAngle.includes('0.05'), `sin(180) degrees stays 0: ${degAngle}`)
  const restored = convertDocumentAngles(angleDoc, 'deg', 'rad')
  const restoredStatement = restored.statements.find((statement) => statement.type === 'math')
  expect(restoredStatement?.type === 'math' && restoredStatement.input === 'sin(pi)', `switching back restores sin(pi): ${restoredStatement?.type === 'math' ? restoredStatement.input : ''}`)

  const axisMiddle = axisThrough(originBox(10, 10))
  expect(axisMiddle.x === 0 && axisMiddle.y === 0 && axisMiddle.z === 0, `axes cross the middle of an origin cube: ${axisMiddle.x},${axisMiddle.y},${axisMiddle.z}`)
  const axisShifted = axisThrough({ xMin: 2, xMax: 8, yMin: 2, yMax: 8, zMin: 4, zMax: 10 })
  expect(axisShifted.x === 5 && axisShifted.y === 5 && axisShifted.z === 7, `axes cross the middle of a shifted cube: ${axisShifted.x},${axisShifted.y},${axisShifted.z}`)

  const ranged = evaluateDocument(appendMath(emptyDocument(), 'r(t) = [t, 0, 0], t = 0..2'))
  const rangedXs = ranged.bodies.find((body) => body.role === 'plot')?.path.filter((point) => Number.isFinite(point.x)).map((point) => point.x) ?? []
  expect(rangedXs.length > 5 && Math.min(...rangedXs) >= -1e-6 && Math.max(...rangedXs) <= 2 + 1e-6 && Math.max(...rangedXs) > 1, `t = 0..2 stays inside that domain: ${Math.min(...rangedXs)}..${Math.max(...rangedXs)}`)
  const open = evaluateDocument(appendMath(emptyDocument(), 'r(t) = [t, 0, 0]'))
  const openXs = open.bodies.find((body) => body.role === 'plot')?.path.filter((point) => Number.isFinite(point.x)).map((point) => point.x) ?? []
  expect(openXs.length > 5 && Math.min(...openXs) < -5 && Math.max(...openXs) > 5, `a parametric curve still defaults to t from -10 to 10: ${Math.min(...openXs)}..${Math.max(...openXs)}`)
  const parenDomain = evaluateDocument(appendMath(emptyDocument(), 'r(t) = [t, t^2, 0], t = (0, 1)'))
  const parenXs = parenDomain.bodies.find((body) => body.role === 'plot')?.path.filter((point) => Number.isFinite(point.x)).map((point) => point.x) ?? []
  expect(parenXs.length > 5 && Math.min(...parenXs) >= -1e-6 && Math.max(...parenXs) <= 1 + 1e-6, `t = (0, 1) is a domain: ${Math.min(...parenXs)}..${Math.max(...parenXs)}`)
  const broken = 'r(t) = [t, 0, 0]\nt = 0..2*pi'
  const brokenView = evaluateDocument(appendMath(emptyDocument(), broken))
  const brokenXs = brokenView.bodies.find((body) => body.role === 'plot')?.path.filter((point) => Number.isFinite(point.x)).map((point) => point.x) ?? []
  expect(brokenXs.length > 5 && Math.min(...brokenXs) >= -1e-6 && Math.max(...brokenXs) <= 2 * Math.PI + 1e-6, `a domain on the next line is kept: ${Math.min(...brokenXs)}..${Math.max(...brokenXs)}`)
  const domainPreview = previewTex('r(t) = [cos(t), sin(t)], t = 0..2*pi') ?? ''
  expect(domainPreview.includes('\\cos') && domainPreview.includes('\\ldots') && domainPreview.includes('\\pi'), `the domain shows with the curve: ${domainPreview}`)
  const domainLine = previewTex('t = 0..2') ?? ''
  expect(domainLine.includes('0') && domainLine.includes('\\ldots') && domainLine.includes('2'), `a domain line previews: ${domainLine}`)
  const convertedDomain = convertAngleInput('r(t) = [cos(pi), sin(t)], t = 0..2', 'rad', 'deg')
  expect(convertedDomain.includes('180') && convertedDomain.includes('t = 0..2'), `a domain survives an angle switch: ${convertedDomain}`)

  return errors
}

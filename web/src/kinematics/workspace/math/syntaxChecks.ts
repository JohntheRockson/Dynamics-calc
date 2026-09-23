// Checks for the Name(inputs){Setting: value} console syntax: the function table, the reader and
// binder, the hint card's caret states, and the paste and typing helpers around settings blocks.

import { appendMath, emptyDocument } from '../document'
import { evaluateDocument } from '../evaluate'
import { FUNCTIONS, MATH_FORMS, PLOT_PALETTE, describeFunctions, findFunction } from './functions'
import { latexToSource } from './inputView'
import { expandMathShortcut, insertMathSlot, looksLikeMath } from './shortcuts'
import { assistAt, signatureAt } from './signature'
import { convertAngleInput, previewTex, validateMath } from './syntax'

export function runSyntaxChecks(): string[] {
  const errors: string[] = []
  const expect = (cond: boolean, message: string) => {
    if (!cond) errors.push(message)
  }
  const run = (input: string) => evaluateDocument(appendMath(emptyDocument(), input))
  const rowOf = (input: string) => run(input).blocks.flatMap((block) => block.rows)[0]
  const textOf = (input: string) => rowOf(input)?.text ?? ''
  const gives = (input: string, want: string) => {
    const row = rowOf(input)
    expect(row?.source !== 'error' && row?.text === want, `${input} should give ${want}: ${row?.text}`)
  }
  const fails = (input: string, want: string) => {
    const row = rowOf(input)
    expect(row?.source === 'error' && row.text === want, `${input} should fail with ${want}: ${row?.text}`)
  }
  const plotsOf = (input: string) => run(input).bodies.filter((body) => body.role === 'plot')
  const finite = (path: { x: number; y: number }[]) => path.filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y))
  const within = (values: number[], min: number, max: number) => values.length > 5 && Math.min(...values) >= min - 1e-6 && Math.max(...values) <= max + 1e-6
  const span = (values: number[]) => `${Math.min(...values)}..${Math.max(...values)}`

  gives('Derivative(x^3, x){At: 1, Order: 2}', 'Derivative(x^3, x){Order: 2, At: 1} = 6')
  gives('Derivative(x^3, x){Order: 2, At: 1}', 'Derivative(x^3, x){Order: 2, At: 1} = 6')
  for (const input of ['Derivative(x^3, x){Order -> 2}', 'Derivative(x^3, x){ORDER = 2}', 'derivative(x^3, x){ order : 2 }']) gives(input, 'Derivative(x^3, x){Order: 2} = 6x')
  expect(Boolean(plotsOf('Plot(sin(x), x){Dashed}')[0]?.dashed), 'a bare flag turns its setting on')
  expect(Boolean(plotsOf('Plot(sin(x), x){Dashed: true}')[0]?.dashed) && !plotsOf('Plot(sin(x), x){Dashed: false}')[0]?.dashed, 'a flag also takes true or false')
  const sparse = finite(plotsOf('Plot(sin(x), x){plot points = 20, max_recursion -> 0}')[0]?.path ?? [])
  expect(sparse.length === 20, `setting names ignore case, spaces, and underscores: ${sparse.length} points`)
  gives('Expand(Derivative(x^3, x){Order: 2})', 'Expand(Derivative(x^3, x){Order: 2}) = 6x')
  fails('Expand(Derivative(x^3, x)){Order: 2}', 'Expand has no settings. Order belongs to Derivative, so put it right after Derivative(…), as in Derivative(x^3, x){Order: 2}.')
  fails('Integrate(Derivative(x^3, x), x){Order: 2}', 'Integrate has no setting Order. Order belongs to Derivative, so put it right after Derivative(…), as in Derivative(x^3, x){Order: 2}.')
  fails('Expand(Derivative(x^3, x){Color: red})', 'Graph settings such as Color go at the end of the line, on the function the line draws.')
  gives('e_{theta} + 1', 'e_theta + 1')
  expect(validateMath('x^{2}') === null && !(previewTex('e_{theta}') ?? '').includes('\\{'), 'braces after _ and ^ stay a subscript and an exponent')
  const settingsTex = previewTex('Solve(x^2 = 4, x){Domain: 0..5}') ?? ''
  expect(settingsTex.includes('\\textcolor{') && settingsTex.includes('\\{\\mathrm{Domain}\\colon 0 \\ldots 5\\}'), `the settings block previews muted: ${settingsTex}`)
  const typingTex = previewTex('Plot(sin(x), x){Color: re', 25) ?? ''
  expect(typingTex.includes('\\mathrm{Color}') && typingTex.includes('\\rule'), `a half-typed setting still previews with the caret: ${typingTex}`)

  fails('Plot(sin(x), x){Colr: red}', 'Plot has no setting Colr. Did you mean Color?')
  fails('Solve(x^2 = 4, x){Domian: 0..5}', 'Solve has no setting Domian. Did you mean Domain?')
  fails('Plot(sin(x), x){PlotPoints: 5}', 'PlotPoints needs a whole number from 12 to 800.')
  fails('Derivative(x^3, x){Order: 1.5}', 'Order needs a whole number from 1 to 6.')
  fails('Derivative(x^3, x){Order: 2, Order: 3}', 'Order is set twice.')
  fails('Solve(x^2 = 4, x){Domain: 5}', 'Domain needs a range such as 0..2*pi.')
  fails('Plot(sin(x), x){Color: blurple}', 'Color needs a name such as red, blue, or green, or a hex code such as #ff8800.')
  fails('Derivative(x^2)', 'Derivative needs an expression and a variable: Derivative(x^3, x)')
  fails('Solve(x^2 = 4, x, 0, 5)', 'Solve takes an equation and a variable. Other settings go in { } after the ), as in Solve(sin(x) = 1/2, x){Domain: 0..2*pi}')
  fails('sin(x){Color: red}', 'sin has no settings. To draw it, use Plot(sin(x), x){Color: …}.')
  fails('Solv(x^2 = 4, x)', 'Solv is not defined. Did you mean Solve?')
  expect(textOf('Plot(sin(x), x){t: 0..1}') === "This graph's input is x, so write {x: …} or {Domain: …}.", `a range for the wrong input warns: ${textOf('Plot(sin(x), x){t: 0..1}')}`)
  expect(textOf('Plot3D(x*y, x, y){z: 0..1}') === "This surface's inputs are x and y, so name one of those: {x: -2..2}.", `a surface range for the wrong input warns: ${textOf('Plot3D(x*y, x, y){z: 0..1}')}`)

  const aliases: [string, string][] = [
    ['deriv(x^2, x)', 'Derivative(x^2, x) = 2x'],
    ['DERIVATIVE(x^2, x)', 'Derivative(x^2, x) = 2x'],
    ['int(2x, x)', 'Integrate(2x, x) = x^2 + C (Graph uses C = 0.)'],
    ['lim(sin(x)/x, x, 0)', 'Limit(sin(x)/x, x, 0) = 1'],
    ['roots(x^2 - 1, x)', 'x = 1 or x = -1'],
    ['taylor(exp(x), x)', 'Series(exp(x), x) = x^3/6 + x^2/2 + x + 1'],
    ['minimum(x^2, x)', 'minimum 0 at x = 0'],
    ['maximum(-x^2, x)', 'maximum 0 at x = 0'],
    ['idiff(x^2 + y^2 = 1, y, x)', 'ImplicitDerivative(x^2 + y^2 = 1, y, x) = -x/y'],
    ['normalize([3, 4])', 'Unit([3, 4]) = [3/5, 4/5]'],
    ['magnitude([3, 4])', 'Norm([3, 4]) = 5'],
    ['determinant([[1, 2], [3, 4]])', 'Det([[1, 2], [3, 4]]) = -2'],
    ['inv([[1, 0], [0, 2]])', 'Inverse([[1, 0], [0, 2]]) = [[1, 0], [0, 1/2]]'],
    ['rem(7, 3)', 'Mod(7, 3) = 1'],
    ['SOLVE(x + 1 = 4, x)', 'x = 3'],
  ]
  for (const [input, want] of aliases) gives(input, want)
  expect(findFunction('D') === null && findFunction('N') === null, 'D and N stay free for physics variables')

  const owners = new Map<string, string>()
  for (const spec of FUNCTIONS) {
    for (const name of [spec.name, ...spec.aliases]) {
      const owner = owners.get(name.toLowerCase())
      expect(owner === undefined || owner === spec.name, `${name} names both ${owner} and ${spec.name}`)
      owners.set(name.toLowerCase(), spec.name)
      expect(findFunction(name)?.name === spec.name, `${name} should find ${spec.name}`)
    }
  }
  for (const example of [...FUNCTIONS.flatMap((spec) => spec.examples), ...MATH_FORMS.map((form) => form.example)]) {
    const row = rowOf(example)
    expect(validateMath(example) === null && row?.source !== 'error', `the example ${example} runs: ${row?.text}`)
  }
  const described = JSON.parse(JSON.stringify(describeFunctions())) as { functions: { name: string }[] }
  expect(described.functions.length === FUNCTIONS.length && described.functions.every((spec, index) => spec.name === FUNCTIONS[index]?.name), 'the function table survives JSON')

  gives('Series(exp(x), x){Point: 1, Order: 1}', 'Series(exp(x), x){Point: 1, Order: 1} = exp(1)*x')
  gives('Series(sin(x), x){Order: 3, Point: 0}', 'Series(sin(x), x) = -x^3/6 + x')
  gives('Zeros(sin(x), x){Domain: 1..7}', 'x = pi or x = 2pi')
  gives('Minimize(x^2 - 2x, x)', 'minimum -1 at x = 1')
  gives('Maximize(-x^2 + 4x, x)', 'maximum 4 at x = 2')
  gives('Minimize(x^2, x){Domain: 1..3}', 'minimum 1 at x = 1')
  gives('Solve(x^2 = 4, x){Domain: 0..5}', 'x = 2')
  expect(textOf('Solve(sin(x) = 1/2, x){Domain: 0..2*pi}').startsWith('x = pi/6 or x = 5pi/6'), `a Domain searches for numeric roots: ${textOf('Solve(sin(x) = 1/2, x){Domain: 0..2*pi}')}`)
  gives('Limit((1 + 1/n)^n, n, 1000000)', 'Limit((1 + 1/n)^n, n, 1000000) = 2.7182805')
  const dsolveCases: [string, string][] = [
    ['DSolve(Derivative(y, x){Order: 2} = -y, y, x)', '= cos(x)*C1 + sin(x)*C2'],
    ['DSolve(Derivative(y, x){Order: 2} = -4y, y, x)', '= cos(2x)*C1 + sin(2x)*C2'],
    ['DSolve(Derivative(y, x){Order: 2} + 2Derivative(y, x) + 5y = 0, y, x)', '= cos(2x)*exp(-x)*C1 + exp(-x)*sin(2x)*C2'],
  ]
  for (const [input, want] of dsolveCases) expect(textOf(input).includes(want), `${input} should include ${want}: ${textOf(input)}`)

  const orange = plotsOf('Plot(sin(x), x){Color: orange}')
  expect(orange.length === 1 && orange[0]?.color === PLOT_PALETTE.orange, `Color names a palette color: ${orange[0]?.color}`)
  const styled = plotsOf('Plot(x^2, x){Color: #ff8800, Dashed}')[0]
  expect(styled?.color === '#ff8800' && Boolean(styled.dashed), `a hex color and a flag together: ${styled?.color}`)
  const clippedXs = finite(plotsOf('Plot(sin(x), x){Domain: 0..1}')[0]?.path ?? []).map((point) => point.x)
  expect(within(clippedXs, 0, 1) && Math.max(...clippedXs) > 0.99, `Domain clips a plot: ${span(clippedXs)}`)
  const halfCircleYs = finite(plotsOf('Plot([cos(t), sin(t)], t){Domain: 0..pi}')[0]?.path ?? []).map((point) => point.y)
  expect(within(halfCircleYs, 0, 1), `Domain clips a parametric plot: ${span(halfCircleYs)}`)
  const redCurve = plotsOf('f(x) = sin(x){Color: red}')[0]
  expect(redCurve?.color === PLOT_PALETTE.red && textOf('f(x) = sin(x){Color: red}') === 'f(x) = sin(x)', `a trailing block styles a definition: ${redCurve?.color}`)
  const styledCurve = plotsOf('f(x) = sin(x){Domain: 0..1, Color: blue, Dashed}')[0]
  const styledXs = finite(styledCurve?.path ?? []).map((point) => point.x)
  expect(styledCurve?.color === PLOT_PALETTE.blue && Boolean(styledCurve.dashed) && within(styledXs, 0, 1), `a definition takes Domain, Color, and Dashed together: ${span(styledXs)}`)
  const arcYs = finite(plotsOf('r(t) = [cos(t), sin(t)]{t: 0..pi}')[0]?.path ?? []).map((point) => point.y)
  expect(within(arcYs, 0, 1), `a named range trims a vector function: ${span(arcYs)}`)
  const coarse = finite(plotsOf('y = x^2{PlotPoints: 20, MaxRecursion: 0}')[0]?.path ?? [])
  expect(coarse.length === 20, `a trailing block samples an equation curve: ${coarse.length}`)
  const patch = run('Plot3D(x*y, x, y){x: 0..1, y: 0..2}')
  const patchSurface = patch.surfaces[0]
  const patchPoints = (patchSurface?.sheets?.length ? patchSurface.sheets : [patchSurface?.grid ?? []]).flat(2).filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y))
  const patchText = patch.blocks.flatMap((block) => block.rows)[0]?.text
  expect(patchText === 'Plot3D(x*y, x, y)' && within(patchPoints.map((point) => point.x), 0, 1) && within(patchPoints.map((point) => point.y), 0, 2), `Plot3D takes a range for each input without a warning: ${patchText}`)

  const marked = (text: string) => ({ source: text.replace('|', ''), cursor: text.indexOf('|') })
  const stateAt = (text: string): string => {
    const { source, cursor } = marked(text)
    const help = signatureAt(source, cursor)
    if (!help) return 'none'
    switch (help.kind) {
      case 'params':
        return `params ${help.spec.name} ${help.active}${help.graph ? ' graph' : ''}`
      case 'after':
        return `after ${help.spec.name}${help.graph ? ' graph' : ''}`
      case 'options':
        return `options ${help.owner ?? '-'} [${help.options.map((option) => option.name).join(', ')}]`
      case 'value':
        return `value ${help.owner ?? '-'} ${help.option?.name ?? `?${help.key}`}`
    }
  }
  const graphSettings = 'Color, PlotPoints, MaxRecursion, Exclusions, Dashed, Domain'
  const states: [string, string][] = [
    ['Solve(|', 'params Solve 0'],
    ['Solve(x^2 = 4, |', 'params Solve 1'],
    ['Solve(x^2 = 4, x, |', 'params Solve 2'],
    ['Solve([x + y = 3, |', 'params Solve 0'],
    ['GCD(12, 18, 30, |', 'params GCD 1'],
    ['Derivative(|', 'params Derivative 0 graph'],
    ['Expand(Derivative(|', 'params Derivative 0'],
    ['Solve(x = 1, x){Domain: 0..5}\nDerivative(|', 'params Derivative 0 graph'],
    ['Solve(x^2 = 4, x)|', 'after Solve'],
    ['Solve(x^2 = 4, x) |', 'after Solve'],
    ['Derivative(x^3, x)|', 'after Derivative graph'],
    ['Expand(Derivative(x^3, x)|', 'after Derivative'],
    ['Expand(Derivative(x^3, x){Order: 2}|', 'params Expand 0'],
    ['Expand(Derivative(x^3, x){Order: 2})|', 'none'],
    ['Solve(x^2 = 4, x){|}', 'options Solve [Domain]'],
    ['Solve([a = 1, b = 2], [sigma, theta]){|}', 'options Solve [Domain, sigma, theta]'],
    ['Plot3D(x*y, x, y){|}', `options Plot3D [x, y, ${graphSettings}]`],
    ['f(x) = sin(x){|}', `options Graph [x, ${graphSettings}]`],
    ['r(t) = [cos(t), sin(t)]{|}', `options Graph [t, ${graphSettings}]`],
    ['Derivative(x^3, x){|}', `options Derivative [Order, At, ${graphSettings}]`],
    ['Expand(Derivative(x^3, x){|})', 'options Derivative [Order, At]'],
    ['2*Derivative(x^3, x){|}', 'options Derivative [Order, At]'],
    ['Expand(Derivative(x^3, x)){|}', 'options Expand []'],
    ['2 + 2{|}', 'options - []'],
    ['Solve(x^2 = 4, x){Domain: |}', 'value Solve Domain'],
    ['Plot(sin(x), x){Color: re|}', 'value Plot Color'],
    ['Expand(x){Order: |}', 'value Expand ?Order'],
    ['e_{th|}', 'none'],
  ]
  for (const [text, want] of states) expect(stateAt(text) === want, `hint at ${JSON.stringify(text)} should be ${want}: ${stateAt(text)}`)

  const picks = (text: string) => {
    const { source, cursor } = marked(text)
    const assist = assistAt(source, cursor)
    return assist && 'items' in assist ? assist.items : []
  }
  const labels = (text: string) => picks(text).map((item) => item.label).join(', ')
  const choose = (text: string, id: string) => {
    const item = picks(text).find((candidate) => candidate.id === id)
    if (!item) return null
    const { source } = marked(text)
    const { start, end, text: inserted, caret } = item.edit
    const value = source.slice(0, start) + inserted + source.slice(end)
    return `${value.slice(0, start + caret)}|${value.slice(start + caret)}`
  }
  expect(labels('Plot(sin(x), x){Color: red, |}') === 'Domain, PlotPoints, MaxRecursion, Exclusions, Dashed', `used settings drop out of the list: ${labels('Plot(sin(x), x){Color: red, |}')}`)
  expect(labels('Plot(sin(x), x){Dashed, Pl|}') === 'PlotPoints', `typed letters filter the settings: ${labels('Plot(sin(x), x){Dashed, Pl|}')}`)
  expect(labels('Plot(sin(x), x){Color: re|}') === 'red', `typed letters filter the colors: ${labels('Plot(sin(x), x){Color: re|}')}`)
  expect(labels('2si|').startsWith('sin(') && labels('Solve(sig|') === '' && labels('so|lve') === '', 'names complete after two letters, and not in the middle of a word')
  expect(choose('so|', 'Solve') === 'Solve(|)', `picking a name opens its parentheses: ${choose('so|', 'Solve')}`)
  expect(choose('Solve(x^2 = 4, x){Do|}', 'Domain') === 'Solve(x^2 = 4, x){Domain: |}', `picking a setting adds its separator: ${choose('Solve(x^2 = 4, x){Do|}', 'Domain')}`)
  expect(choose('Plot(sin(x), x){Da|}', 'Dashed') === 'Plot(sin(x), x){Dashed|}', `picking an off-by-default flag writes it bare: ${choose('Plot(sin(x), x){Da|}', 'Dashed')}`)
  expect(choose('Plot(sin(x), x){Color: re|}', 'red') === 'Plot(sin(x), x){Color: red|}', `picking a color finishes the value: ${choose('Plot(sin(x), x){Color: re|}', 'red')}`)

  for (const line of ['Solve(x^2 = 4, x){Domain: 0..5}', 'Plot(sin(x), x){Color: red, Dashed}', 'Derivative(x^3, x){Order: 2}']) {
    const back = latexToSource(previewTex(line) ?? '')
    expect(back === line, `the preview of ${line} pastes back as ${back}`)
  }
  const pastedRange = latexToSource(previewTex('r(t) = [cos(t), sin(t)]{t: 0..2*pi}') ?? '')
  expect(pastedRange === 'r(t) = [cos(t), sin(t)]{t: 0..2pi}', `a pasted range keeps its block: ${pastedRange}`)
  expect(latexToSource('Solve(sin(x) = 1/2, x){Domain: 0..2\\pi}') === 'Solve(sin(x) = 1/2, x){Domain: 0..2pi}', 'latex inside a pasted block converts')
  expect(latexToSource('\\frac{1}{2}{x}') === '(1)/(2)(x)', `a brace group that is not settings stays a product: ${latexToSource('\\frac{1}{2}{x}')}`)
  expect(latexToSource('e_{\\theta}') === 'e_theta', `a pasted subscript is not settings: ${latexToSource('e_{\\theta}')}`)
  expect(convertAngleInput('Plot(sin(x), x){Domain: 0..pi, Color: red}', 'rad', 'deg') === 'Plot(sin(x), x){Domain: 0..pi, Color: red}', 'an angle switch leaves a settings block alone')
  expect(convertAngleInput('Derivative(sin(pi), x){Order: 2}', 'rad', 'deg') === 'Derivative(sin(180), x){Order: 2}', `an angle switch keeps the block after a converted call: ${convertAngleInput('Derivative(sin(pi), x){Order: 2}', 'rad', 'deg')}`)

  const typedSolve = expandMathShortcut('solve', 5, ' ')
  expect(typedSolve?.value === 'Solve()' && typedSolve.cursor === 6, `a typed name becomes its canonical call: ${typedSolve?.value}`)
  const typedAlias = expandMathShortcut('diff', 4, 'Tab')
  expect(typedAlias?.value === 'Derivative()' && typedAlias.cursor === 11, `an older name expands to the canonical one: ${typedAlias?.value}`)
  expect(expandMathShortcut('Plot', 4, '3') === null, 'a digit after Plot keeps typing toward Plot3D')
  expect(expandMathShortcut('dt', 2, ' ') === null, 'letters that name no function stay as typed')
  const openBlock = insertMathSlot('Solve(x^2 = 4, x)', 17, 17, '{')
  expect(openBlock?.value === 'Solve(x^2 = 4, x){}' && openBlock.cursor === 18, `a brace after a call opens a block: ${openBlock?.value}`)
  const closeBlock = insertMathSlot('a{}', 2, 2, '}')
  expect(closeBlock?.value === 'a{}' && closeBlock.cursor === 3, 'a typed closing brace steps over the pair')
  expect(insertMathSlot('point', 5, 5, '{') === null, 'a statement keeps its braces as typed')
  expect(looksLikeMath('so') && looksLikeMath('dif') && looksLikeMath('Deri') && !looksLikeMath('po') && !looksLikeMath('sp') && !looksLikeMath('pl'), 'two letters of a function name start math, unless a statement starts the same way')

  return errors
}

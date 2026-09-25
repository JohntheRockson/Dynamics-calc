// Every console function in one table: names, required inputs, settings, and examples.
// The table is plain data, so the editor hints, the Functions menu, the parser, and a
// later API all read the same description.

export type SectionId = 'algebra' | 'calculus' | 'graphs' | 'linear' | 'number' | 'basic'

export type ParamKind = 'expression' | 'equation' | 'variable' | 'value' | 'whole' | 'vector' | 'matrix'

export type OptionKind = 'range' | 'whole' | 'value' | 'boolean' | 'color'

export interface ParamSpec {
  name: string
  kind: ParamKind
  /** How an error message names this input: "an equation". */
  phrase: string
  description: string
  example: string
  /** The last input can repeat, as in GCD(12, 18, 30). */
  repeats?: boolean
}

export interface OptionSpec {
  name: string
  kind: OptionKind
  description: string
  default: string | number | boolean | null
  example: string
  min?: number
  max?: number
  choices?: string[]
  /** Other names typed for the same setting, as Fill for Shade. */
  aliases?: string[]
}

export interface FunctionSpec {
  name: string
  /** The name the math kernel uses internally. */
  kernel: string
  aliases: string[]
  section: SectionId
  summary: string
  params: ParamSpec[]
  options: OptionSpec[]
  /** Draws on the figure, so the graph settings (Color, PlotPoints, …) apply too. */
  draws: boolean
  /** A setting can name one of the variables directly, as in Solve(…){x: 0..5}. */
  variableRanges?: boolean
  /** Shown when a graph setting is given to a function that only computes. */
  graphHint?: string
  examples: string[]
}

export interface SectionSpec {
  id: SectionId
  title: string
}

export interface FormSpec {
  section: SectionId
  title: string
  summary: string
  example: string
}

export const MATH_SECTIONS: SectionSpec[] = [
  { id: 'algebra', title: 'Algebra' },
  { id: 'calculus', title: 'Calculus' },
  { id: 'graphs', title: 'Graphs' },
  { id: 'linear', title: 'Vectors' },
  { id: 'number', title: 'Numbers' },
  { id: 'basic', title: 'Basic' },
]

export const PLOT_PALETTE: Record<string, string> = {
  red: '#fb6a6a',
  orange: '#f5a524',
  yellow: '#facc15',
  green: '#59d67f',
  teal: '#2dd4bf',
  cyan: '#29d3f5',
  blue: '#5aa8ff',
  purple: '#a78bfa',
  pink: '#e879f9',
  white: '#e7edf5',
  gray: '#93a2b6',
}

const COLOR_ALIASES: Record<string, string> = { grey: 'gray', violet: 'purple', magenta: 'pink' }

export const GRAPH_OPTIONS: OptionSpec[] = [
  { name: 'Color', kind: 'color', description: 'Line color: a name such as red or blue, or a hex code', default: 'auto', example: 'red', choices: Object.keys(PLOT_PALETTE) },
  { name: 'PlotPoints', kind: 'whole', description: 'How many points to sample before refining', default: 128, example: '300', min: 12, max: 800 },
  { name: 'MaxRecursion', kind: 'whole', description: 'How many times to refine sharp bends', default: 5, example: '3', min: 0, max: 8 },
  { name: 'Exclusions', kind: 'boolean', description: 'Break the line at jumps and poles', default: true, example: 'false' },
  { name: 'Dashed', kind: 'boolean', description: 'Draw a dashed line', default: false, example: 'true' },
  { name: 'Domain', kind: 'range', description: 'Only draw this interval of the input', default: null, example: '0..2*pi' },
  { name: 'Shade', kind: 'range', description: 'Shade between the curve and the x-axis over this interval. Shade: true shades the whole curve', default: null, example: '0..2', aliases: ['Fill', 'Filling'] },
]

const expression = (description = 'The expression to work on', example = 'x^2'): ParamSpec => ({ name: 'expression', kind: 'expression', phrase: 'an expression', description, example })
const variable = (description = 'The variable, such as x or t', example = 'x'): ParamSpec => ({ name: 'variable', kind: 'variable', phrase: 'a variable', description, example })
const point = (description: string): ParamSpec => ({ name: 'point', kind: 'value', phrase: 'a point', description, example: '0' })
const one = (name: string, kind: ParamKind, phrase: string, description: string, example: string): ParamSpec => ({ name, kind, phrase, description, example })

const domainSearch = (description: string, fallback: string | null): OptionSpec => ({ name: 'Domain', kind: 'range', description, default: fallback, example: '0..2*pi' })

export const FUNCTIONS: FunctionSpec[] = [
  {
    name: 'Solve',
    kernel: 'solve',
    aliases: [],
    section: 'algebra',
    summary: 'Solve an equation, or a list of equations, for the variables you name',
    params: [
      one('equation', 'equation', 'an equation', 'An equation, or a list [eq1, eq2] for a system', 'x^2 = 4'),
      one('variable', 'variable', 'a variable', 'The variable to solve for, or a list [x, y]', 'x'),
    ],
    options: [domainSearch('Only keep solutions in this interval, and search it numerically when there is no exact answer', null)],
    draws: false,
    variableRanges: true,
    examples: ['Solve(x^2 = 4, x)', 'Solve(sin(x) = 1/2, x){Domain: 0..2*pi}', 'Solve([x + y = 3, x - y = 1], [x, y])'],
  },
  {
    name: 'Zeros',
    kernel: 'zeros',
    aliases: ['roots'],
    section: 'algebra',
    summary: 'Where an expression equals zero, marked on its graph',
    params: [expression('The expression to find roots of', 'x^2 - 1'), variable()],
    options: [domainSearch('Where to look for roots', '-10..10')],
    draws: true,
    examples: ['Zeros(x^2 - 1, x)', 'Zeros(sin(x), x){Domain: 0..10}'],
  },
  {
    name: 'Expand',
    kernel: 'expand',
    aliases: [],
    section: 'algebra',
    summary: 'Multiply out products and powers, including FOIL',
    params: [expression('The product to multiply out', '(x + 1)(x + 2)')],
    options: [],
    draws: false,
    examples: ['Expand((x + 1)(x + 2))'],
  },
  {
    name: 'Factor',
    kernel: 'factor',
    aliases: [],
    section: 'algebra',
    summary: 'Factor a whole number, or a polynomial in one variable',
    params: [expression('A whole number or a polynomial', 'x^2 - 1')],
    options: [],
    draws: false,
    examples: ['Factor(x^2 - 1)', 'Factor(360)'],
  },
  {
    name: 'Derivative',
    kernel: 'diff',
    aliases: ['diff', 'deriv'],
    section: 'calculus',
    summary: 'Differentiate with respect to a variable. With t, every symbol depends on time',
    params: [expression('The expression to differentiate', 'x^3'), variable('The variable to differentiate by', 'x')],
    options: [
      { name: 'Order', kind: 'whole', description: 'How many times to differentiate', default: 1, example: '2', min: 1, max: 6 },
      { name: 'At', kind: 'value', description: 'Evaluate the derivative at this value', default: null, example: '3' },
    ],
    draws: true,
    examples: ['Derivative(x^3, x)', 'Derivative(x^3, x){Order: 2}', 'Derivative(x^2, x){At: 3}', "Derivative(r*theta'*e_theta, t)"],
  },
  {
    name: 'Integrate',
    kernel: 'integrate',
    aliases: ['integral', 'int'],
    section: 'calculus',
    summary: 'Antiderivative, or a definite integral when you give Bounds. It computes and does not draw',
    params: [expression('The expression to integrate', 'x^2'), variable('The variable to integrate by', 'x')],
    options: [{ name: 'Bounds', kind: 'range', description: 'Integrate from the left end to the right end', default: null, example: '0..1' }],
    draws: false,
    graphHint: 'Integrate only computes. To picture an area, shade the curve instead: Plot(x^2, x){Shade: 0..2}.',
    examples: ['Integrate(x^2, x)', 'Integrate(x, x){Bounds: 0..1}'],
  },
  {
    name: 'Limit',
    kernel: 'limit',
    aliases: ['lim'],
    section: 'calculus',
    summary: 'The value an expression approaches. The last two inputs can be written x = 0',
    params: [expression('The expression', 'sin(x)/x'), variable('The variable that moves', 'x'), point('The value the variable approaches')],
    options: [],
    draws: false,
    examples: ['Limit(sin(x)/x, x, 0)', 'Limit((1 + 1/n)^n, n = 1000000)'],
  },
  {
    name: 'Series',
    kernel: 'series',
    aliases: ['taylor'],
    section: 'calculus',
    summary: 'Taylor polynomial, drawn with the original curve',
    params: [expression('The expression to approximate', 'exp(x)'), variable()],
    options: [
      { name: 'Point', kind: 'value', description: 'The point to expand around', default: 0, example: '1' },
      { name: 'Order', kind: 'whole', description: 'The highest power to keep', default: 3, example: '5', min: 0, max: 8 },
    ],
    draws: true,
    examples: ['Series(exp(x), x)', 'Series(cos(x), x){Order: 6}', 'Series(ln(x), x){Point: 1, Order: 4}'],
  },
  {
    name: 'Tangent',
    kernel: 'tangent',
    aliases: [],
    section: 'calculus',
    summary: 'Tangent line at a point. The last two inputs can be written x = 1',
    params: [expression('The curve', 'x^2'), variable(), point('Where the line touches')],
    options: [],
    draws: true,
    examples: ['Tangent(x^2, x, 1)', 'Tangent(x^3, x = 1)'],
  },
  {
    name: 'Normal',
    kernel: 'normal',
    aliases: [],
    section: 'calculus',
    summary: 'Normal line at a point, perpendicular to the tangent',
    params: [expression('The curve', 'x^2'), variable(), point('Where the line crosses')],
    options: [],
    draws: true,
    examples: ['Normal(x^2, x, 1)'],
  },
  {
    name: 'Minimize',
    kernel: 'fmin',
    aliases: ['fmin', 'minimum'],
    section: 'calculus',
    summary: 'Lowest value on an interval, and where it happens',
    params: [expression('The expression to minimize', 'x^2 - 2x'), variable()],
    options: [domainSearch('The interval to search', '-10..10')],
    draws: true,
    examples: ['Minimize(x^2 - 2x, x)', 'Minimize(sin(x), x){Domain: 0..2*pi}'],
  },
  {
    name: 'Maximize',
    kernel: 'fmax',
    aliases: ['fmax', 'maximum'],
    section: 'calculus',
    summary: 'Highest value on an interval, and where it happens',
    params: [expression('The expression to maximize', '4 - x^2'), variable()],
    options: [domainSearch('The interval to search', '-10..10')],
    draws: true,
    examples: ['Maximize(4 - x^2, x)', 'Maximize(x*exp(-x), x){Domain: 0..5}'],
  },
  {
    name: 'Sum',
    kernel: 'sum',
    aliases: [],
    section: 'calculus',
    summary: 'Add up a term for each whole number from a start to an end',
    params: [
      expression('The term to add up', 'i^2'),
      one('index', 'variable', 'an index variable', 'The counting variable', 'i'),
      one('from', 'whole', 'a start value', 'The first value of the index', '1'),
      one('to', 'whole', 'an end value', 'The last value of the index', '5'),
    ],
    options: [],
    draws: false,
    examples: ['Sum(i^2, i, 1, 5)'],
  },
  {
    name: 'Product',
    kernel: 'prod',
    aliases: ['prod'],
    section: 'calculus',
    summary: 'Multiply a term for each whole number from a start to an end',
    params: [
      expression('The term to multiply', 'i'),
      one('index', 'variable', 'an index variable', 'The counting variable', 'i'),
      one('from', 'whole', 'a start value', 'The first value of the index', '1'),
      one('to', 'whole', 'an end value', 'The last value of the index', '4'),
    ],
    options: [],
    draws: false,
    examples: ['Product(i, i, 1, 4)'],
  },
  {
    name: 'DSolve',
    kernel: 'dsolve',
    aliases: [],
    section: 'calculus',
    summary: "Solve y' = f(x), y' = ky, or a constant-coefficient second-order equation",
    params: [
      one('equation', 'equation', 'a differential equation', 'The differential equation', 'Derivative(y, x) = y'),
      one('function', 'variable', 'the unknown function', 'The unknown function, such as y', 'y'),
      variable('The independent variable', 'x'),
    ],
    options: [],
    draws: true,
    examples: ['DSolve(Derivative(y, x) = y, y, x)', 'DSolve(Derivative(y, x){Order: 2} = -y, y, x)'],
  },
  {
    name: 'ImplicitDerivative',
    kernel: 'idiff',
    aliases: ['idiff'],
    section: 'calculus',
    summary: 'dy/dx from an equation that mixes x and y',
    params: [
      one('equation', 'equation', 'an equation', 'An equation in both variables', 'x^2 + y^2 = 1'),
      one('dependent', 'variable', 'a dependent variable', 'The variable that depends on the other, such as y', 'y'),
      one('independent', 'variable', 'an independent variable', 'The variable to differentiate by, such as x', 'x'),
    ],
    options: [],
    draws: false,
    examples: ['ImplicitDerivative(x^2 + y^2 = 1, y, x)'],
  },
  {
    name: 'Plot',
    kernel: 'plot',
    aliases: [],
    section: 'graphs',
    summary: 'Draw a curve. A list of two or three expressions draws a parametric curve',
    params: [expression('What to draw: an expression, or a list [x(t), y(t)]', 'sin(x)'), variable('The input to plot against', 'x')],
    options: [domainSearch('The interval of the input to draw. A parametric curve uses -10..10 without one', null)],
    draws: true,
    variableRanges: true,
    examples: ['Plot(sin(x), x){Color: orange}', 'Plot(x^2, x){Shade: 0..2}', 'Plot(tan(x), x){Domain: -pi..pi, Exclusions: true}', 'Plot([cos(t), sin(t)], t){Domain: 0..2*pi, Dashed}'],
  },
  {
    name: 'Plot3D',
    kernel: 'plot3d',
    aliases: [],
    section: 'graphs',
    summary: 'Draw a surface z = f(x, y). Name a range for either input to trim it',
    params: [
      expression('The height of the surface', 'x^2 - y^2'),
      one('x', 'variable', 'a first variable', 'The first input', 'x'),
      one('y', 'variable', 'a second variable', 'The second input', 'y'),
    ],
    options: [],
    draws: true,
    variableRanges: true,
    examples: ['Plot3D(x^2 - y^2, x, y)', 'Plot3D(sin(x)*cos(y), x, y){x: -pi..pi, y: -pi..pi, Color: teal}'],
  },
  {
    name: 'Dot',
    kernel: 'dot',
    aliases: [],
    section: 'linear',
    summary: 'Dot product u · v',
    params: [one('u', 'vector', 'a first vector', 'The first vector', '[1, 2]'), one('v', 'vector', 'a second vector', 'The second vector', '[3, 4]')],
    options: [],
    draws: false,
    examples: ['Dot([1, 2], [3, 4])'],
  },
  {
    name: 'Cross',
    kernel: 'cross',
    aliases: [],
    section: 'linear',
    summary: 'Cross product: a scalar in the plane, a vector in space',
    params: [one('u', 'vector', 'a first vector', 'The first vector', '[1, 0, 0]'), one('v', 'vector', 'a second vector', 'The second vector', '[0, 1, 0]')],
    options: [],
    draws: false,
    examples: ['Cross([1, 0, 0], [0, 1, 0])'],
  },
  {
    name: 'Unit',
    kernel: 'unit',
    aliases: ['normalize'],
    section: 'linear',
    summary: 'Same direction, length 1',
    params: [one('vector', 'vector', 'a vector', 'The vector to scale', '[3, 4]')],
    options: [],
    draws: false,
    examples: ['Unit([3, 4])'],
  },
  {
    name: 'Norm',
    kernel: 'norm',
    aliases: ['mag', 'magnitude'],
    section: 'linear',
    summary: 'Length of a vector',
    params: [one('vector', 'vector', 'a vector', 'The vector to measure', '[3, 4]')],
    options: [],
    draws: false,
    examples: ['Norm([3, 4])'],
  },
  {
    name: 'Det',
    kernel: 'det',
    aliases: ['determinant'],
    section: 'linear',
    summary: 'Determinant of a 2 by 2 or 3 by 3 matrix',
    params: [one('matrix', 'matrix', 'a matrix', 'A square matrix', '[[1, 2], [3, 4]]')],
    options: [],
    draws: false,
    examples: ['Det([[1, 2], [3, 4]])'],
  },
  {
    name: 'Inverse',
    kernel: 'inv',
    aliases: ['inv'],
    section: 'linear',
    summary: 'Inverse of a 2 by 2 or 3 by 3 matrix',
    params: [one('matrix', 'matrix', 'a matrix', 'A square matrix', '[[1, 2], [3, 4]]')],
    options: [],
    draws: false,
    examples: ['Inverse([[1, 2], [3, 4]])'],
  },
  {
    name: 'Transpose',
    kernel: 'transpose',
    aliases: [],
    section: 'linear',
    summary: 'Swap rows and columns',
    params: [one('matrix', 'matrix', 'a matrix', 'Any matrix', '[[1, 2], [3, 4]]')],
    options: [],
    draws: false,
    examples: ['Transpose([[1, 2], [3, 4]])'],
  },
  {
    name: 'Trace',
    kernel: 'trace',
    aliases: [],
    section: 'linear',
    summary: 'Sum of the diagonal of a square matrix',
    params: [one('matrix', 'matrix', 'a matrix', 'A square matrix', '[[1, 2], [3, 4]]')],
    options: [],
    draws: false,
    examples: ['Trace([[1, 2], [3, 4]])'],
  },
  {
    name: 'Decimal',
    kernel: 'decimal',
    aliases: [],
    section: 'number',
    summary: 'Write a value as a decimal',
    params: [one('value', 'value', 'a value', 'Any number or numeric expression', 'sqrt(2)')],
    options: [{ name: 'Digits', kind: 'whole', description: 'Significant digits to show', default: 8, example: '4', min: 1, max: 15 }],
    draws: false,
    examples: ['Decimal(1/3)', 'Decimal(pi){Digits: 4}'],
  },
  {
    name: 'Fraction',
    kernel: 'fraction',
    aliases: [],
    section: 'number',
    summary: 'The simplest fraction close to a decimal',
    params: [one('value', 'value', 'a value', 'A decimal to approximate', '0.375')],
    options: [],
    draws: false,
    examples: ['Fraction(0.375)'],
  },
  {
    name: 'GCD',
    kernel: 'gcd',
    aliases: [],
    section: 'number',
    summary: 'Greatest common divisor of whole numbers',
    params: [one('a', 'whole', 'a whole number', 'A whole number', '12'), { ...one('b', 'whole', 'another whole number', 'Another whole number. Add more after a comma', '18'), repeats: true }],
    options: [],
    draws: false,
    examples: ['GCD(12, 18)', 'GCD(12, 18, 30)'],
  },
  {
    name: 'LCM',
    kernel: 'lcm',
    aliases: [],
    section: 'number',
    summary: 'Least common multiple of whole numbers',
    params: [one('a', 'whole', 'a whole number', 'A whole number', '4'), { ...one('b', 'whole', 'another whole number', 'Another whole number. Add more after a comma', '6'), repeats: true }],
    options: [],
    draws: false,
    examples: ['LCM(4, 6)'],
  },
  {
    name: 'Mod',
    kernel: 'mod',
    aliases: ['rem'],
    section: 'number',
    summary: 'Remainder after dividing a by b',
    params: [one('a', 'whole', 'a whole number', 'The number to divide', '7'), one('b', 'whole', 'a divisor', 'The number to divide by', '3')],
    options: [],
    draws: false,
    examples: ['Mod(7, 3)'],
  },
  ...(['sin', 'cos', 'tan'] as const).map((name): FunctionSpec => ({
    name,
    kernel: name,
    aliases: [],
    section: 'basic',
    summary: `${name === 'sin' ? 'Sine' : name === 'cos' ? 'Cosine' : 'Tangent'} of an angle in the current unit`,
    params: [one('angle', 'value', 'an angle', 'An angle in radians or degrees, whichever is selected', 'pi/6')],
    options: [],
    draws: false,
    examples: [`${name}(pi/6)`],
  })),
  ...(['asin', 'acos', 'atan'] as const).map((name): FunctionSpec => ({
    name,
    kernel: name,
    aliases: [`arc${name.slice(1)}`],
    section: 'basic',
    summary: `Inverse ${name === 'asin' ? 'sine' : name === 'acos' ? 'cosine' : 'tangent'}, as an angle in the current unit`,
    params: [one('value', 'value', 'a value', 'A number', '1/2')],
    options: [],
    draws: false,
    examples: [`${name}(1/2)`],
  })),
  {
    name: 'sqrt',
    kernel: 'sqrt',
    aliases: [],
    section: 'basic',
    summary: 'Square root',
    params: [one('value', 'value', 'a value', 'The number under the root', '2')],
    options: [],
    draws: false,
    examples: ['sqrt(8)'],
  },
  {
    name: 'abs',
    kernel: 'abs',
    aliases: [],
    section: 'basic',
    summary: 'Absolute value, or the length of a vector',
    params: [one('value', 'value', 'a value', 'A number or a vector', '-3')],
    options: [],
    draws: false,
    examples: ['abs(-3)'],
  },
  {
    name: 'exp',
    kernel: 'exp',
    aliases: [],
    section: 'basic',
    summary: 'e raised to a power',
    params: [one('power', 'value', 'a power', 'The power of e', '1')],
    options: [],
    draws: false,
    examples: ['exp(1)'],
  },
  {
    name: 'ln',
    kernel: 'ln',
    aliases: [],
    section: 'basic',
    summary: 'Natural logarithm',
    params: [one('value', 'value', 'a value', 'A positive number', 'e')],
    options: [],
    draws: false,
    examples: ['ln(e)'],
  },
  {
    name: 'log',
    kernel: 'log',
    aliases: [],
    section: 'basic',
    summary: 'Logarithm, base 10 unless you give a Base',
    params: [one('value', 'value', 'a value', 'A positive number', '100')],
    options: [{ name: 'Base', kind: 'value', description: 'The base of the logarithm', default: 10, example: '2' }],
    draws: false,
    examples: ['log(100)', 'log(8){Base: 2}'],
  },
]

export const MATH_FORMS: FormSpec[] = [
  { section: 'graphs', title: 'Curve', summary: 'A one-input function draws its graph', example: 'f(x) = x^2{Color: blue}' },
  { section: 'graphs', title: 'Equation curve', summary: 'y = , x = , or y^2 = draws a curve', example: 'y = sin(x){PlotPoints: 300}' },
  { section: 'graphs', title: 'Shaded area', summary: 'Shade between a curve and the x-axis, for example to show an integral as an area', example: 'f(x) = x^2{Shade: 0..2}' },
  { section: 'graphs', title: 'Parametric curve', summary: 'A vector function of one input', example: 'r(t) = [cos(t), sin(t)]{t: 0..2*pi}' },
  { section: 'graphs', title: 'Space curve', summary: 'Three components draw in 3D', example: 'r(t) = [cos(t), sin(t), t/4]{t: 0..4*pi}' },
  { section: 'graphs', title: 'Surface', summary: 'A two-input function, or z = , draws a surface', example: 'g(x, y) = x^2 - y^2' },
  { section: 'linear', title: 'Vector', summary: 'Square brackets, drawn as an arrow', example: '[1, 2, 3]' },
  { section: 'linear', title: 'Matrix', summary: 'Rows of the same length', example: '[[1, 2], [3, 4]]' },
]

export const SYNTAX_RULES: string[] = [
  'Parentheses hold the required inputs, in order: Solve(x^2 = 4, x).',
  'Optional settings go in { } right after the closing parenthesis, in any order, separated by commas: Solve(x^2 = 4, x){Domain: 0..5}.',
  'Write a setting as Name: value. = and -> work too, and names ignore case, spaces, and underscores.',
  'A yes-or-no setting can stand alone: {Dashed}.',
  'Settings at the end of a definition or an equation style its graph: f(x) = sin(x){Color: red}.',
  'Integrate only computes. To show an area, shade the curve that draws it: Plot(x^2, x){Shade: 0..2}.',
]

function nameKey(name: string): string {
  return name.toLowerCase()
}

/** Option names ignore case, spaces, underscores, and hyphens: `plot points` is PlotPoints. */
export function optionKey(name: string): string {
  return name.toLowerCase().replace(/[\s_-]+/g, '')
}

const BY_NAME = new Map<string, FunctionSpec>()
const BY_KERNEL = new Map<string, FunctionSpec>()
for (const spec of FUNCTIONS) {
  BY_KERNEL.set(spec.kernel, spec)
  for (const name of [spec.name, spec.kernel, ...spec.aliases]) BY_NAME.set(nameKey(name), spec)
}

/** Every word that names a function, lowercased: canonical names, kernel names, and aliases. */
export const FUNCTION_WORDS: string[] = [...BY_NAME.keys()]

export function findFunction(name: string): FunctionSpec | null {
  return BY_NAME.get(nameKey(name)) ?? null
}

export function functionByKernel(kernel: string): FunctionSpec | null {
  return BY_KERNEL.get(kernel) ?? null
}

/** A function's own settings, then the graph settings it has not replaced. */
export function optionsFor(spec: FunctionSpec | null): OptionSpec[] {
  if (!spec) return []
  if (!spec.draws) return spec.options
  const own = new Set(spec.options.map((option) => optionKey(option.name)))
  return [...spec.options, ...GRAPH_OPTIONS.filter((option) => !own.has(optionKey(option.name)))]
}

function optionNamed(option: OptionSpec, wanted: string): boolean {
  return optionKey(option.name) === wanted || (option.aliases ?? []).some((alias) => optionKey(alias) === wanted)
}

export function findOption(options: OptionSpec[], key: string): OptionSpec | null {
  const wanted = optionKey(key)
  return options.find((option) => optionNamed(option, wanted)) ?? null
}

export function isGraphOption(key: string): boolean {
  return findOption(GRAPH_OPTIONS, key) !== null
}

/** Every setting name any function accepts, for spelling suggestions and for display. */
export const ALL_OPTION_NAMES: string[] = [...new Set([...FUNCTIONS.flatMap((spec) => spec.options.map((option) => option.name)), ...GRAPH_OPTIONS.map((option) => option.name)])]

export function canonicalOptionName(key: string): string | null {
  const wanted = optionKey(key)
  const named = ALL_OPTION_NAMES.find((name) => optionKey(name) === wanted)
  if (named) return named
  return [...FUNCTIONS.flatMap((spec) => spec.options), ...GRAPH_OPTIONS].find((option) => optionNamed(option, wanted))?.name ?? null
}

export function parseColor(value: string): string | null {
  const text = value.trim().toLowerCase()
  if (/^#(?:[0-9a-f]{3}|[0-9a-f]{6})$/.test(text)) return text
  const name = COLOR_ALIASES[text] ?? text
  return PLOT_PALETTE[name] ?? null
}

function editDistance(a: string, b: string): number {
  const rows = a.length + 1
  const cols = b.length + 1
  const table: number[][] = Array.from({ length: rows }, (_, i) => Array.from({ length: cols }, (_, j) => (i === 0 ? j : j === 0 ? i : 0)))
  for (let i = 1; i < rows; i += 1) {
    for (let j = 1; j < cols; j += 1) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1
      let best = Math.min((table[i - 1]?.[j] ?? 0) + 1, (table[i]?.[j - 1] ?? 0) + 1, (table[i - 1]?.[j - 1] ?? 0) + cost)
      if (i > 1 && j > 1 && a[i - 1] === b[j - 2] && a[i - 2] === b[j - 1]) best = Math.min(best, (table[i - 2]?.[j - 2] ?? 0) + 1)
      const row = table[i]
      if (row) row[j] = best
    }
  }
  return table[a.length]?.[b.length] ?? Math.max(a.length, b.length)
}

/** The closest candidate to a misspelled word, if one is close enough to be what was meant. */
export function suggestName(word: string, candidates: string[]): string | null {
  const typed = optionKey(word)
  if (!typed) return null
  let best: string | null = null
  let bestScore = Infinity
  for (const candidate of candidates) {
    const key = optionKey(candidate)
    const score = key.startsWith(typed) && typed.length >= 3 ? 0.5 : editDistance(typed, key)
    if (score < bestScore) {
      best = candidate
      bestScore = score
    }
  }
  const allowed = typed.length <= 4 ? 1 : 2
  return bestScore <= allowed ? best : null
}

/** `Solve(equation, variable)`: the required inputs in order. */
export function signatureText(spec: FunctionSpec): string {
  const params = spec.params.map((param) => (param.repeats ? `${param.name}, …` : param.name))
  return `${spec.name}(${params.join(', ')})`
}

/** Functions whose name or alias starts with the typed prefix, canonical-name matches first. */
export function completeFunctionName(prefix: string): FunctionSpec[] {
  const typed = nameKey(prefix)
  if (!typed) return []
  const hits: { spec: FunctionSpec; rank: number }[] = []
  for (const spec of FUNCTIONS) {
    if (nameKey(spec.name).startsWith(typed)) hits.push({ spec, rank: 0 })
    else if (spec.aliases.some((alias) => nameKey(alias).startsWith(typed))) hits.push({ spec, rank: 1 })
  }
  return hits.sort((a, b) => a.rank - b.rank || a.spec.name.length - b.spec.name.length).map((hit) => hit.spec)
}

export interface FunctionReference {
  syntax: string[]
  sections: SectionSpec[]
  functions: FunctionSpec[]
  graphOptions: OptionSpec[]
  colors: Record<string, string>
  forms: FormSpec[]
}

/** The whole console vocabulary as plain JSON, for a help page or an API client. */
export function describeFunctions(): FunctionReference {
  return JSON.parse(JSON.stringify({ syntax: SYNTAX_RULES, sections: MATH_SECTIONS, functions: FUNCTIONS, graphOptions: GRAPH_OPTIONS, colors: PLOT_PALETTE, forms: MATH_FORMS })) as FunctionReference
}

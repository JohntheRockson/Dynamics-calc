export interface MathTool {
  name: string
  template: string
  blurb: string
}

export interface MathSection {
  id: string
  title: string
  items: MathTool[]
}

export const MATH_SECTIONS: MathSection[] = [
  {
    id: 'number',
    title: 'Number',
    items: [
      { name: 'Decimal', template: 'decimal(', blurb: 'Write a value as a decimal' },
      { name: 'Fraction', template: 'fraction(', blurb: 'Approximate a decimal with a fraction' },
      { name: 'Factor', template: 'factor(', blurb: 'Factor an integer or a polynomial' },
      { name: 'LCM', template: 'lcm(', blurb: 'Least common multiple' },
      { name: 'GCD', template: 'gcd(', blurb: 'Greatest common divisor' },
      { name: 'Remainder', template: 'mod(', blurb: 'Remainder, mod(a, b)' },
    ],
  },
  {
    id: 'algebra',
    title: 'Algebra',
    items: [
      { name: 'Expand', template: 'expand((x + 1)(x + 2))', blurb: 'Distribute, including FOIL' },
      { name: 'Factor', template: 'factor(x^2 - 1)', blurb: 'Factor an integer or a polynomial' },
      { name: 'Zeros', template: 'zeros(x^2 - 1)', blurb: 'Roots of an expression' },
      { name: 'Solve', template: 'solve(x + y = 3, x - y = 1)', blurb: 'One equation, or a linear system' },
    ],
  },
  {
    id: 'calculus',
    title: 'Calculus',
    items: [
      { name: 'Derivative', template: 'diff(x^2, x)', blurb: 'Derivative with respect to a variable' },
      { name: 'Second derivative', template: 'diff(x^3, x, 2)', blurb: 'Differentiate more than once' },
      { name: 'Derivative at a point', template: 'diff(x^2, x, 1, 3)', blurb: 'diff(expr, x, order, point)' },
      { name: 'Integral', template: 'integrate(x^2, x)', blurb: 'Indefinite integral' },
      { name: 'Definite integral', template: 'integrate(x, x, 0, 1)', blurb: 'Integral from a to b' },
      { name: 'Limit', template: 'limit(sin(x)/x, x, 0)', blurb: 'Limit as a variable approaches a value' },
      { name: 'Sum', template: 'sum(i, i, 1, 5)', blurb: 'Finite sum' },
      { name: 'Product', template: 'prod(i, i, 1, 4)', blurb: 'Finite product' },
      { name: 'Minimum', template: 'fmin(x^2, x, -2, 2)', blurb: 'Lowest value on an interval' },
      { name: 'Maximum', template: 'fmax(x^2, x, -2, 2)', blurb: 'Highest value on an interval' },
      { name: 'Tangent', template: 'tangent(x^2, x, 1)', blurb: 'Tangent line at a point' },
      { name: 'Normal', template: 'normal(x^2, x, 1)', blurb: 'Normal line at a point' },
      { name: 'Series', template: 'series(exp(x), x, 0, 3)', blurb: 'Taylor polynomial' },
      { name: 'Differential equation', template: 'dsolve(diff(y, x) = y, y, x)', blurb: "y' = f(x) or y' = ky" },
      { name: 'Implicit derivative', template: 'idiff(x^2 + y^2 = 1, y, x)', blurb: 'dy/dx from an equation' },
    ],
  },
]

/** Names the console treats as math, including aliases that are not in the menu. */
export const MATH_FUNCTION_NAMES = [
  'decimal',
  'fraction',
  'factor',
  'lcm',
  'gcd',
  'mod',
  'rem',
  'expand',
  'zeros',
  'solve',
  'diff',
  'integrate',
  'limit',
  'sum',
  'prod',
  'fmin',
  'fmax',
  'tangent',
  'normal',
  'series',
  'dsolve',
  'idiff',
]

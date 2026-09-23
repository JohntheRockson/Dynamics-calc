// Vectors and matrices on top of the scalar kernel.
// A scalar subexpression is handed back to normalize so fractions stay exact.

import { MathError, type AngleMode, type Expr } from './expr'
import { functionByKernel } from './functions'

const LINEAR_CALLS = new Set(['dot', 'cross', 'unit', 'norm', 'mag', 'det', 'transpose', 'inv', 'trace'])

type Scalar = (expr: Expr, angles: AngleMode) => Expr

export function containsAggregate(expr: Expr): boolean {
  let found = false
  const visit = (node: Expr) => {
    if (found) return
    if (node.type === 'vec' || node.type === 'mat') found = true
    else if (node.type === 'call' && (LINEAR_CALLS.has(node.name) || (node.name === 'abs' && node.args.some((arg) => arg.type === 'vec' || arg.type === 'mat')))) found = true
    else if (node.type === 'add' || node.type === 'mul' || node.type === 'call') node.args.forEach(visit)
    else if (node.type === 'div' || node.type === 'pow' || node.type === 'eq') {
      visit(node.type === 'eq' ? node.left : node.type === 'div' ? node.num : node.base)
      visit(node.type === 'eq' ? node.right : node.type === 'div' ? node.den : node.exp)
    } else if ((node.type === 'group' || node.type === 'caret') && node.body) visit(node.body)
  }
  visit(expr)
  return found
}

export function reduceAlgebra(expr: Expr, angles: AngleMode, scalar: Scalar): Expr {
  const simplify = (node: Expr): Expr => (containsAggregate(node) ? evalAlgebra(node) : scalar(node, angles))

  const evalAlgebra = (node: Expr): Expr => {
    switch (node.type) {
      case 'vec':
        return { type: 'vec', args: node.args.map(simplify) }
      case 'mat':
        return matrix(node.rows.map((row) => row.map(simplify)))
      case 'add':
        return addAll(node.args.map(simplify), angles, scalar)
      case 'mul':
        return multiplyAll(node.args.map(simplify), angles, scalar)
      case 'div':
        return divide(simplify(node.num), simplify(node.den), angles, scalar)
      case 'call':
        return callAlgebra(node.name, node.args.map(simplify), angles, scalar)
      case 'pow':
      case 'eq':
        throw new MathError('Use Dot, Cross, or matrix multiplication for vectors and matrices.')
      default:
        return scalar(node, angles)
    }
  }

  return evalAlgebra(expr)
}

function addAll(args: Expr[], angles: AngleMode, scalar: Scalar): Expr {
  if (args.every(isVec)) return { type: 'vec', args: addComponents(args.map(vectorOf), angles, scalar) }
  if (args.every(isMat)) return addMatrices(args.map(matrixOf), angles, scalar)
  if (args.some(isVec) || args.some(isMat)) throw new MathError('Add vectors or matrices of the same shape.')
  return scalar({ type: 'add', args }, angles)
}

function multiplyAll(args: Expr[], angles: AngleMode, scalar: Scalar): Expr {
  return args.reduce((acc, arg) => multiplyPair(acc, arg, angles, scalar))
}

function multiplyPair(left: Expr, right: Expr, angles: AngleMode, scalar: Scalar): Expr {
  if (isVec(left) && isVec(right)) throw new MathError('Use Dot(u, v) or Cross(u, v) to multiply vectors.')
  if (!isVec(left) && !isMat(left) && isVec(right)) return scaleVector(left, vectorOf(right), angles, scalar)
  if (isVec(left) && !isVec(right) && !isMat(right)) return scaleVector(right, vectorOf(left), angles, scalar)
  if (!isVec(left) && !isMat(left) && isMat(right)) return scaleMatrix(left, matrixOf(right), angles, scalar)
  if (isMat(left) && !isVec(right) && !isMat(right)) return scaleMatrix(right, matrixOf(left), angles, scalar)
  if (isMat(left) && isVec(right)) return multiplyMatrixVector(matrixOf(left), vectorOf(right), angles, scalar)
  if (isMat(left) && isMat(right)) return multiplyMatrices(matrixOf(left), matrixOf(right), angles, scalar)
  return scalar({ type: 'mul', args: [left, right] }, angles)
}

function divide(num: Expr, den: Expr, angles: AngleMode, scalar: Scalar): Expr {
  if (isVec(den) || isMat(den)) throw new MathError('Divide by a scalar, or use Inverse for a matrix.')
  if (isVec(num)) return scaleVector(scalar({ type: 'div', num: { type: 'rat', n: 1n, d: 1n }, den }, angles), vectorOf(num), angles, scalar)
  if (isMat(num)) return scaleMatrix(scalar({ type: 'div', num: { type: 'rat', n: 1n, d: 1n }, den }, angles), matrixOf(num), angles, scalar)
  return scalar({ type: 'div', num, den }, angles)
}

function callAlgebra(name: string, args: Expr[], angles: AngleMode, scalar: Scalar): Expr {
  if (name === 'abs' && args.length === 1 && isVec(args[0])) return magnitude(vectorOf(args[0]), angles, scalar)
  if (name === 'dot' && args.length === 2) return dot(vectorOf(args[0]), vectorOf(args[1]), angles, scalar)
  if (name === 'cross' && args.length === 2) return cross(vectorOf(args[0]), vectorOf(args[1]), angles, scalar)
  if (name === 'unit' && args.length === 1) return unit(vectorOf(args[0]), angles, scalar)
  if ((name === 'norm' || name === 'mag') && args.length === 1) return magnitude(vectorOf(args[0]), angles, scalar)
  if (name === 'det' && args.length === 1) return determinant(matrixOf(args[0]), angles, scalar)
  if (name === 'transpose' && args.length === 1) return transpose(matrixOf(args[0]))
  if (name === 'inv' && args.length === 1) return inverse(matrixOf(args[0]), angles, scalar)
  if (name === 'trace' && args.length === 1) return trace(matrixOf(args[0]), angles, scalar)
  if (args.some((arg) => isVec(arg) || isMat(arg))) throw new MathError(`Cannot use ${functionByKernel(name)?.name ?? name} on a vector or matrix yet.`)
  return scalar({ type: 'call', name, args }, angles)
}

function addComponents(columns: Expr[][], angles: AngleMode, scalar: Scalar): Expr[] {
  const width = columns[0]?.length ?? 0
  if (width === 0 || columns.some((column) => column.length !== width)) throw new MathError('Add vectors or matrices of the same shape.')
  return columns[0].map((_, index) => scalar({ type: 'add', args: columns.map((column) => column[index]) }, angles))
}

function addMatrices(matrices: Expr[][][], angles: AngleMode, scalar: Scalar): Expr {
  const rows = matrices[0].length
  const cols = matrices[0][0]?.length ?? 0
  if (matrices.some((matrix) => matrix.length !== rows || matrix.some((row) => row.length !== cols))) throw new MathError('Add vectors or matrices of the same shape.')
  const summed = matrices[0].map((row, rowIndex) => row.map((_, colIndex) => scalar({ type: 'add', args: matrices.map((matrix) => matrix[rowIndex][colIndex]) }, angles)))
  return { type: 'mat', rows: summed }
}

function scaleVector(factor: Expr, components: Expr[], angles: AngleMode, scalar: Scalar): Expr {
  return { type: 'vec', args: components.map((component) => scalar({ type: 'mul', args: [factor, component] }, angles)) }
}

function scaleMatrix(factor: Expr, rows: Expr[][], angles: AngleMode, scalar: Scalar): Expr {
  return { type: 'mat', rows: rows.map((row) => row.map((entry) => scalar({ type: 'mul', args: [factor, entry] }, angles))) }
}

function multiplyMatrixVector(rows: Expr[][], vector: Expr[], angles: AngleMode, scalar: Scalar): Expr {
  if (rows[0]?.length !== vector.length) throw new MathError('The matrix columns should match the vector length.')
  return { type: 'vec', args: rows.map((row) => dot(row, vector, angles, scalar)) }
}

function multiplyMatrices(left: Expr[][], right: Expr[][], angles: AngleMode, scalar: Scalar): Expr {
  const shared = left[0]?.length ?? 0
  if (shared !== right.length) throw new MathError('The left matrix columns should match the right matrix rows.')
  const cols = right[0]?.length ?? 0
  const rows = left.map((row) =>
    Array.from({ length: cols }, (_, col) => dot(row, right.map((rightRow) => rightRow[col]), angles, scalar)),
  )
  return { type: 'mat', rows }
}

function dot(left: Expr[], right: Expr[], angles: AngleMode, scalar: Scalar): Expr {
  if (left.length !== right.length || left.length === 0) throw new MathError('Dot needs two vectors of the same length.')
  return scalar({ type: 'add', args: left.map((component, index) => ({ type: 'mul', args: [component, right[index]] })) }, angles)
}

function cross(left: Expr[], right: Expr[], angles: AngleMode, scalar: Scalar): Expr {
  if (left.length === 2 && right.length === 2) {
    return scalar(
      {
        type: 'add',
        args: [
          { type: 'mul', args: [left[0], right[1]] },
          { type: 'mul', args: [{ type: 'rat', n: -1n, d: 1n }, left[1], right[0]] },
        ],
      },
      angles,
    )
  }
  if (left.length === 3 && right.length === 3) {
    const term = (a: Expr, b: Expr, c: Expr, d: Expr): Expr =>
      scalar(
        {
          type: 'add',
          args: [
            { type: 'mul', args: [a, b] },
            { type: 'mul', args: [{ type: 'rat', n: -1n, d: 1n }, c, d] },
          ],
        },
        angles,
      )
    return {
      type: 'vec',
      args: [term(left[1], right[2], left[2], right[1]), term(left[2], right[0], left[0], right[2]), term(left[0], right[1], left[1], right[0])],
    }
  }
  throw new MathError('Cross takes two vectors in the plane, or two vectors in space.')
}

function magnitude(components: Expr[], angles: AngleMode, scalar: Scalar): Expr {
  return scalar({ type: 'call', name: 'sqrt', args: [{ type: 'add', args: components.map((component) => ({ type: 'pow', base: component, exp: { type: 'rat', n: 2n, d: 1n } })) }] }, angles)
}

function unit(components: Expr[], angles: AngleMode, scalar: Scalar): Expr {
  const length = magnitude(components, angles, scalar)
  if (isZero(length)) throw new MathError('The zero vector has no direction.')
  return scaleVector({ type: 'div', num: { type: 'rat', n: 1n, d: 1n }, den: length }, components, angles, scalar)
}

function determinant(rows: Expr[][], angles: AngleMode, scalar: Scalar): Expr {
  if (rows.length === 2 && rows[0].length === 2 && rows[1].length === 2) {
    return scalar(
      {
        type: 'add',
        args: [
          { type: 'mul', args: [rows[0][0], rows[1][1]] },
          { type: 'mul', args: [{ type: 'rat', n: -1n, d: 1n }, rows[0][1], rows[1][0]] },
        ],
      },
      angles,
    )
  }
  if (rows.length === 3 && rows.every((row) => row.length === 3)) {
    const [a, b, c] = rows[0]
    const [d, e, f] = rows[1]
    const [g, h, i] = rows[2]
    const product = (x: Expr, y: Expr, z: Expr, sign: bigint): Expr => ({ type: 'mul', args: sign < 0n ? [{ type: 'rat', n: -1n, d: 1n }, x, y, z] : [x, y, z] })
    return scalar({ type: 'add', args: [product(a, e, i, 1n), product(b, f, g, 1n), product(c, d, h, 1n), product(c, e, g, -1n), product(b, d, i, -1n), product(a, f, h, -1n)] }, angles)
  }
  throw new MathError('Determinant is ready for a 2 by 2 or 3 by 3 matrix.')
}

function transpose(rows: Expr[][]): { type: 'mat'; rows: Expr[][] } {
  const cols = rows[0]?.length ?? 0
  return { type: 'mat', rows: Array.from({ length: cols }, (_, col) => rows.map((row) => row[col])) }
}

function inverse(rows: Expr[][], angles: AngleMode, scalar: Scalar): Expr {
  const det = determinant(rows, angles, scalar)
  if (isZero(det)) throw new MathError('That matrix has no inverse.')
  const factor: Expr = { type: 'div', num: { type: 'rat', n: 1n, d: 1n }, den: det }
  if (rows.length === 2) {
    return scaleMatrix(factor, [
      [rows[1][1], neg(rows[0][1])],
      [neg(rows[1][0]), rows[0][0]],
    ], angles, scalar)
  }
  if (rows.length === 3) {
    const cofactor = (dropRow: number, dropCol: number, sign: bigint): Expr => {
      const minor = rows.filter((_, row) => row !== dropRow).map((row) => row.filter((_, col) => col !== dropCol))
      const value = determinant(minor, angles, scalar)
      return sign < 0n ? neg(value) : value
    }
    const signs = [
      [1n, -1n, 1n],
      [-1n, 1n, -1n],
      [1n, -1n, 1n],
    ]
    const cofactors = signs.map((row, rowIndex) => row.map((sign, colIndex) => cofactor(rowIndex, colIndex, sign)))
    const adjugate = transpose(cofactors)
    return scaleMatrix(factor, adjugate.rows, angles, scalar)
  }
  throw new MathError('Inverse is ready for a 2 by 2 or 3 by 3 matrix.')
}

function trace(rows: Expr[][], angles: AngleMode, scalar: Scalar): Expr {
  if (rows.length !== rows[0]?.length) throw new MathError('Trace needs a square matrix.')
  return scalar({ type: 'add', args: rows.map((row, index) => row[index]) }, angles)
}

function matrix(rows: Expr[][]): Expr {
  const width = rows[0]?.length ?? 0
  if (rows.length === 0 || width === 0 || rows.some((row) => row.length !== width)) throw new MathError('Matrix rows should be the same length.')
  return { type: 'mat', rows }
}

function vectorOf(expr: Expr): Expr[] {
  if (expr.type !== 'vec' || expr.args.length === 0) throw new MathError('That needs a vector, written [1, 2].')
  return expr.args
}

function matrixOf(expr: Expr): Expr[][] {
  if (expr.type !== 'mat') throw new MathError('That needs a matrix, written [[1, 2], [3, 4]].')
  return expr.rows
}

function isVec(expr: Expr): boolean {
  return expr.type === 'vec'
}

function isMat(expr: Expr): boolean {
  return expr.type === 'mat'
}

function neg(expr: Expr): Expr {
  return { type: 'mul', args: [{ type: 'rat', n: -1n, d: 1n }, expr] }
}

function isZero(expr: Expr): boolean {
  return expr.type === 'rat' && expr.n === 0n
}

/** Safe arithmetic expressions for a(t,s,v) / v(t) / v(s). No eval(). */

export type Ast =
  | { type: 'num'; value: number }
  | { type: 'var'; name: 't' | 's' | 'v' }
  | { type: 'const'; name: 'pi' | 'e' | 'g' }
  | { type: 'unary'; op: '-'; arg: Ast }
  | { type: 'binary'; op: '+' | '-' | '*' | '/' | '^'; left: Ast; right: Ast }
  | { type: 'call'; name: string; arg: Ast }

export interface EvalCtx {
  t: number
  s: number
  v: number
}

export interface CompiledExpr {
  source: string
  ast: Ast
  vars: Set<'t' | 's' | 'v'>
  eval: (ctx: EvalCtx) => number
  latex: string
  deriv: (variable: 't' | 's' | 'v') => CompiledExpr
}

const CONSTS: Record<'pi' | 'e' | 'g', number> = {
  pi: Math.PI,
  e: Math.E,
  g: 9.81,
}

const FUNS: Record<string, (x: number) => number> = {
  sin: Math.sin,
  cos: Math.cos,
  tan: Math.tan,
  asin: Math.asin,
  acos: Math.acos,
  atan: Math.atan,
  sinh: Math.sinh,
  cosh: Math.cosh,
  tanh: Math.tanh,
  exp: Math.exp,
  ln: Math.log,
  log: Math.log,
  log10: Math.log10,
  sqrt: Math.sqrt,
  abs: Math.abs,
  sign: Math.sign,
}

const FUN_NAMES = new Set(Object.keys(FUNS))

type Tok =
  | { kind: 'num'; value: number }
  | { kind: 'id'; value: string }
  | { kind: 'op'; value: string }
  | { kind: 'eof' }

function normalize(src: string): string {
  return src
    .replace(/[−–—]/g, '-')
    .replace(/[×·]/g, '*')
    .replace(/÷/g, '/')
    .replace(/π/g, 'pi')
    .replace(/√/g, 'sqrt')
    .replace(/\*\*/g, '^')
}

/** Allow `a = ...`, `v(t)=...`, `a(s) = ...` on the left-hand side. */
export function stripLhs(src: string): string {
  return normalize(src).replace(/^\s*[avAV]\s*(\(\s*[tsvTSV]\s*\))?\s*=\s*/i, '')
}

function tokenize(src: string): Tok[] {
  const s = normalize(src)
  const out: Tok[] = []
  let i = 0
  while (i < s.length) {
    const c = s[i]
    if (c <= ' ') {
      i++
      continue
    }
    if ((c >= '0' && c <= '9') || (c === '.' && i + 1 < s.length && s[i + 1] >= '0' && s[i + 1] <= '9')) {
      const m = s.slice(i).match(/^\d*\.?\d+(e[+-]?\d+)?/i)
      if (!m) throw new Error(`bad number at column ${i + 1}`)
      out.push({ kind: 'num', value: parseFloat(m[0]) })
      i += m[0].length
      continue
    }
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c === '_') {
      const m = s.slice(i).match(/^[A-Za-z_]\w*/)
      if (!m) throw new Error(`bad name at column ${i + 1}`)
      out.push({ kind: 'id', value: m[0].toLowerCase() })
      i += m[0].length
      continue
    }
    if ('+-*/^(),'.includes(c)) {
      out.push({ kind: 'op', value: c })
      i++
      continue
    }
    throw new Error(`unexpected '${c}' at column ${i + 1}`)
  }
  out.push({ kind: 'eof' })
  return out
}

class Parser {
  private i = 0
  private readonly toks: Tok[]
  constructor(toks: Tok[]) {
    this.toks = toks
  }

  peek(): Tok {
    return this.toks[this.i]
  }

  take(): Tok {
    return this.toks[this.i++]
  }

  parse(): Ast {
    const ast = this.parseAdd()
    if (this.peek().kind !== 'eof') {
      const t = this.peek()
      const extra = t.kind === 'id' ? t.value : t.kind === 'op' ? t.value : t.kind === 'num' ? String(t.value) : 'end'
      throw new Error(`unexpected '${extra}'`)
    }
    return ast
  }

  private parseAdd(): Ast {
    let left = this.parseMul()
    for (;;) {
      const t = this.peek()
      if (t.kind === 'op' && (t.value === '+' || t.value === '-')) {
        this.take()
        const right = this.parseMul()
        left = { type: 'binary', op: t.value, left, right }
      } else break
    }
    return left
  }

  private parseMul(): Ast {
    let left = this.parseUnary()
    for (;;) {
      const t = this.peek()
      if (t.kind === 'op' && (t.value === '*' || t.value === '/')) {
        this.take()
        const right = this.parseUnary()
        left = { type: 'binary', op: t.value, left, right }
        continue
      }
      // Implied multiplication: 2t, 2(t+1), (s+1)(s-1), 2sin(t), s(s+1).
      if (this.startsPrimary(t)) {
        const right = this.parseUnary()
        left = { type: 'binary', op: '*', left, right }
        continue
      }
      break
    }
    return left
  }

  private startsPrimary(t: Tok): boolean {
    if (t.kind === 'num' || t.kind === 'id') return true
    return t.kind === 'op' && t.value === '('
  }

  private parseUnary(): Ast {
    const t = this.peek()
    if (t.kind === 'op' && t.value === '-') {
      this.take()
      return { type: 'unary', op: '-', arg: this.parseUnary() }
    }
    if (t.kind === 'op' && t.value === '+') {
      this.take()
      return this.parseUnary()
    }
    return this.parsePower()
  }

  private parsePower(): Ast {
    const base = this.parsePrimary()
    const t = this.peek()
    if (t.kind === 'op' && t.value === '^') {
      this.take()
      // Right-assoc: 2^3^2 = 2^(3^2). Unary binds on the exponent.
      return { type: 'binary', op: '^', left: base, right: this.parseUnary() }
    }
    return base
  }

  private parsePrimary(): Ast {
    const t = this.take()
    if (t.kind === 'num') return { type: 'num', value: t.value }
    if (t.kind === 'id') {
      if (t.value === 't' || t.value === 's' || t.value === 'v') return { type: 'var', name: t.value }
      if (t.value === 'pi' || t.value === 'e' || t.value === 'g') return { type: 'const', name: t.value }
      if (FUN_NAMES.has(t.value)) {
        const n = this.peek()
        if (!(n.kind === 'op' && n.value === '(')) throw new Error(`${t.value}(...) needs parentheses`)
        this.take()
        const arg = this.parseAdd()
        const close = this.take()
        if (!(close.kind === 'op' && close.value === ')')) throw new Error(`missing ')' after ${t.value}(...)`)
        return { type: 'call', name: t.value, arg }
      }
      throw new Error(`unknown name '${t.value}' (use t, s, v, pi, e, g, or a function like sin, sqrt)`)
    }
    if (t.kind === 'op' && t.value === '(') {
      const inner = this.parseAdd()
      const close = this.take()
      if (!(close.kind === 'op' && close.value === ')')) throw new Error("missing ')'")
      return inner
    }
    throw new Error('expected a number, t, s, v, or parentheses')
  }
}

function collectVars(ast: Ast, into: Set<'t' | 's' | 'v'>): void {
  switch (ast.type) {
    case 'var':
      into.add(ast.name)
      return
    case 'unary':
      collectVars(ast.arg, into)
      return
    case 'binary':
      collectVars(ast.left, into)
      collectVars(ast.right, into)
      return
    case 'call':
      collectVars(ast.arg, into)
      return
    default:
      return
  }
}

function evalAst(ast: Ast, ctx: EvalCtx): number {
  switch (ast.type) {
    case 'num':
      return ast.value
    case 'var':
      return ctx[ast.name]
    case 'const':
      return CONSTS[ast.name]
    case 'unary':
      return -evalAst(ast.arg, ctx)
    case 'binary': {
      const l = evalAst(ast.left, ctx)
      const r = evalAst(ast.right, ctx)
      switch (ast.op) {
        case '+':
          return l + r
        case '-':
          return l - r
        case '*':
          return l * r
        case '/':
          return l / r
        case '^':
          return l ** r
      }
    }
    case 'call': {
      const fn = FUNS[ast.name]
      return fn(evalAst(ast.arg, ctx))
    }
  }
}

function isNum(ast: Ast, value: number): boolean {
  return ast.type === 'num' && ast.value === value
}

function simplify(ast: Ast): Ast {
  switch (ast.type) {
    case 'unary': {
      const arg = simplify(ast.arg)
      if (arg.type === 'num') return { type: 'num', value: -arg.value }
      if (arg.type === 'unary' && arg.op === '-') return arg.arg
      return { type: 'unary', op: '-', arg }
    }
    case 'binary': {
      const left = simplify(ast.left)
      const right = simplify(ast.right)
      if (left.type === 'num' && right.type === 'num') {
        return { type: 'num', value: evalAst({ type: 'binary', op: ast.op, left, right }, { t: 0, s: 0, v: 0 }) }
      }
      if (ast.op === '+') {
        if (isNum(left, 0)) return right
        if (isNum(right, 0)) return left
      }
      if (ast.op === '-') {
        if (isNum(right, 0)) return left
        if (isNum(left, 0)) return simplify({ type: 'unary', op: '-', arg: right })
      }
      if (ast.op === '*') {
        if (isNum(left, 0) || isNum(right, 0)) return { type: 'num', value: 0 }
        if (isNum(left, 1)) return right
        if (isNum(right, 1)) return left
        if (isNum(left, -1)) return simplify({ type: 'unary', op: '-', arg: right })
        if (isNum(right, -1)) return simplify({ type: 'unary', op: '-', arg: left })
      }
      if (ast.op === '/') {
        if (isNum(left, 0)) return { type: 'num', value: 0 }
        if (isNum(right, 1)) return left
      }
      if (ast.op === '^') {
        if (isNum(right, 0)) return { type: 'num', value: 1 }
        if (isNum(right, 1)) return left
        if (isNum(left, 1)) return { type: 'num', value: 1 }
      }
      return { type: 'binary', op: ast.op, left, right }
    }
    case 'call':
      return { type: 'call', name: ast.name, arg: simplify(ast.arg) }
    default:
      return ast
  }
}

function mul(a: Ast, b: Ast): Ast {
  return simplify({ type: 'binary', op: '*', left: a, right: b })
}

function add(a: Ast, b: Ast): Ast {
  return simplify({ type: 'binary', op: '+', left: a, right: b })
}

function sub(a: Ast, b: Ast): Ast {
  return simplify({ type: 'binary', op: '-', left: a, right: b })
}

function div(a: Ast, b: Ast): Ast {
  return simplify({ type: 'binary', op: '/', left: a, right: b })
}

function pow(a: Ast, b: Ast): Ast {
  return simplify({ type: 'binary', op: '^', left: a, right: b })
}

function call(name: string, arg: Ast): Ast {
  return simplify({ type: 'call', name, arg })
}

function n(value: number): Ast {
  return { type: 'num', value }
}

function differentiate(ast: Ast, x: 't' | 's' | 'v'): Ast {
  switch (ast.type) {
    case 'num':
    case 'const':
      return n(0)
    case 'var':
      return n(ast.name === x ? 1 : 0)
    case 'unary':
      return simplify({ type: 'unary', op: '-', arg: differentiate(ast.arg, x) })
    case 'binary': {
      const u = ast.left
      const w = ast.right
      const du = differentiate(u, x)
      const dw = differentiate(w, x)
      switch (ast.op) {
        case '+':
          return add(du, dw)
        case '-':
          return sub(du, dw)
        case '*':
          return add(mul(du, w), mul(u, dw))
        case '/':
          return div(sub(mul(du, w), mul(u, dw)), pow(w, n(2)))
        case '^':
          // u^w * (w' ln u + w u' / u)
          return mul(pow(u, w), add(mul(dw, call('ln', u)), mul(w, div(du, u))))
      }
    }
    case 'call': {
      const u = ast.arg
      const du = differentiate(u, x)
      let dfdx: Ast
      switch (ast.name) {
        case 'sin':
          dfdx = call('cos', u)
          break
        case 'cos':
          dfdx = { type: 'unary', op: '-', arg: call('sin', u) }
          break
        case 'tan':
          dfdx = div(n(1), pow(call('cos', u), n(2)))
          break
        case 'exp':
          dfdx = call('exp', u)
          break
        case 'ln':
        case 'log':
          dfdx = div(n(1), u)
          break
        case 'log10':
          dfdx = div(n(1), mul(u, call('ln', n(10))))
          break
        case 'sqrt':
          dfdx = div(n(1), mul(n(2), call('sqrt', u)))
          break
        case 'abs':
          dfdx = call('sign', u)
          break
        case 'sign':
          dfdx = n(0)
          break
        case 'sinh':
          dfdx = call('cosh', u)
          break
        case 'cosh':
          dfdx = call('sinh', u)
          break
        case 'tanh':
          dfdx = sub(n(1), pow(call('tanh', u), n(2)))
          break
        case 'asin':
          dfdx = div(n(1), call('sqrt', sub(n(1), pow(u, n(2)))))
          break
        case 'acos':
          dfdx = { type: 'unary', op: '-', arg: div(n(1), call('sqrt', sub(n(1), pow(u, n(2))))) }
          break
        case 'atan':
          dfdx = div(n(1), add(n(1), pow(u, n(2))))
          break
        default:
          throw new Error(`cannot differentiate ${ast.name}`)
      }
      return mul(dfdx, du)
    }
  }
}

function parenLatex(ast: Ast, parentPrec: number): string {
  const p = prec(ast)
  const inner = astToLatex(ast)
  return p < parentPrec ? `\\left(${inner}\\right)` : inner
}

function prec(ast: Ast): number {
  if (ast.type === 'binary') {
    if (ast.op === '+' || ast.op === '-') return 1
    if (ast.op === '*' || ast.op === '/') return 2
    if (ast.op === '^') return 3
  }
  if (ast.type === 'unary') return 2
  return 4
}

export function astToLatex(ast: Ast): string {
  switch (ast.type) {
    case 'num': {
      const v = ast.value
      if (Number.isInteger(v)) return String(v)
      return String(v)
    }
    case 'var':
      return ast.name
    case 'const':
      if (ast.name === 'pi') return '\\pi'
      if (ast.name === 'g') return 'g'
      return 'e'
    case 'unary':
      return `-${parenLatex(ast.arg, 2)}`
    case 'binary': {
      if (ast.op === '/') return `\\dfrac{${astToLatex(ast.left)}}{${astToLatex(ast.right)}}`
      if (ast.op === '^') return `${parenLatex(ast.left, 3)}^{${astToLatex(ast.right)}}`
      if (ast.op === '*') {
        const l = parenLatex(ast.left, 2)
        const r = parenLatex(ast.right, 2)
        if (ast.left.type === 'num' && ast.right.type !== 'num') return `${l}\\,${r}`
        return `${l}\\,${r}`
      }
      return `${parenLatex(ast.left, 1)} ${ast.op} ${parenLatex(ast.right, 1)}`
    }
    case 'call': {
      const map: Record<string, string> = {
        sin: '\\sin',
        cos: '\\cos',
        tan: '\\tan',
        asin: '\\arcsin',
        acos: '\\arccos',
        atan: '\\arctan',
        sinh: '\\sinh',
        cosh: '\\cosh',
        tanh: '\\tanh',
        exp: '\\exp',
        ln: '\\ln',
        log: '\\ln',
        log10: '\\log_{10}',
        sqrt: '\\sqrt',
        abs: '',
        sign: '\\operatorname{sign}',
      }
      const name = map[ast.name] ?? `\\operatorname{${ast.name}}`
      if (ast.name === 'sqrt') return `\\sqrt{${astToLatex(ast.arg)}}`
      if (ast.name === 'abs') return `\\left|${astToLatex(ast.arg)}\\right|`
      return `${name}\\left(${astToLatex(ast.arg)}\\right)`
    }
  }
}

export function compileAst(ast: Ast, source: string): CompiledExpr {
  const simplified = simplify(ast)
  const vars = new Set<'t' | 's' | 'v'>()
  collectVars(simplified, vars)
  return {
    source,
    ast: simplified,
    vars,
    eval: (ctx) => evalAst(simplified, ctx),
    latex: astToLatex(simplified),
    deriv: (variable) => compileAst(differentiate(simplified, variable), `d/d${variable}(${source})`),
  }
}

export type ParseResult = { ok: true; expr: CompiledExpr } | { ok: false; error: string }

export function parseExpr(input: string): ParseResult {
  const trimmed = stripLhs(input).trim()
  if (!trimmed) return { ok: false, error: 'enter an expression, e.g. 4*t - 3 or 2*s + 4' }
  try {
    const ast = new Parser(tokenize(trimmed)).parse()
    return { ok: true, expr: compileAst(ast, trimmed) }
  } catch (err) {
    return { ok: false, error: err instanceof Error ? err.message : String(err) }
  }
}

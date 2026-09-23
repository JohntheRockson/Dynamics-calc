import { parseExpr } from './expr'
import { solveConstA, solveFromExpression } from './solver'
import { solve2D } from './solver2d'

function close(got: number, exp: number, tol = 1e-3): boolean {
  return Math.abs(got - exp) <= tol * (1 + Math.abs(exp))
}

function check(name: string, cond: boolean, detail = ''): string | null {
  return cond ? null : `${name}${detail ? `: ${detail}` : ''}`
}

export function runKinematicsSolverSelftest(): string[] {
  const fails: string[] = []
  const fail = (s: string | null) => {
    if (s) fails.push(s)
  }

  const e1 = parseExpr('2t + 4')
  fail(check('parse 2t+4', e1.ok && e1.expr.eval({ t: 3, s: 0, v: 0 }) === 10))
  const e2 = parseExpr('a(s) = 2*s + 4')
  fail(check('strip a(s)=', e2.ok && Math.abs(e2.expr.eval({ t: 0, s: 10, v: 0 }) - 24) < 1e-12))
  const e3 = parseExpr('(s+1)(s-1)')
  fail(check('implied mul parens', e3.ok && Math.abs(e3.expr.eval({ t: 0, s: 3, v: 0 }) - 8) < 1e-12))
  const e4 = parseExpr('sin(pi/2)')
  fail(check('sin(pi/2)', e4.ok && Math.abs(e4.expr.eval({ t: 0, s: 0, v: 0 }) - 1) < 1e-12))
  const e5 = parseExpr('v^2')
  fail(check('d/dv v^2 = 2v', e5.ok && Math.abs(e5.expr.deriv('v').eval({ t: 0, s: 0, v: 5 }) - 10) < 1e-9))
  const e6 = parseExpr('-0.05 v^2')
  fail(check('space implied mul', e6.ok && Math.abs(e6.expr.eval({ t: 0, s: 0, v: 10 }) + 5) < 1e-12))

  const brake = solveConstA({ s0: 0, v0: 25, s: null, v: 0, a: -4, t: null })
  fail(check('brake ok', brake.ok))
  if (brake.ok) {
    fail(check('brake t', close(brake.primary.t, 6.25), `got ${brake.primary.t}`))
    fail(check('brake s', close(brake.primary.s, 78.125), `got ${brake.primary.s}`))
  }

  const drop = solveConstA({ s0: 0, v0: 0, s: null, v: null, a: -9.81, t: 2 })
  fail(check('drop ok', drop.ok))
  if (drop.ok) {
    fail(check('drop s', close(drop.primary.s, -19.62), `got ${drop.primary.s}`))
    fail(check('drop v', close(drop.primary.v, -19.62), `got ${drop.primary.v}`))
  }

  const howLong = solveConstA({ s0: 0, v0: 0, s: 40, v: null, a: 2, t: null })
  fail(check('howLong ok', howLong.ok))
  if (howLong.ok) {
    fail(check('howLong t', close(howLong.primary.t, Math.sqrt(40)), `got ${howLong.primary.t}`))
    fail(check('howLong v', close(howLong.primary.v, 2 * Math.sqrt(40)), `got ${howLong.primary.v}`))
  }

  const under = solveConstA({ s0: 0, v0: 5, s: null, v: null, a: 2, t: null })
  fail(check('underdetermined', !under.ok))

  const at = solveFromExpression('a-expr', '4*t - 3', 0, 0, 2, { kind: 't', value: 4 }, 30)
  fail(check('a(t) ok', at.ok))
  if (at.ok) {
    fail(check('a(t) v', close(at.primary.v, 22, 2e-3), `got ${at.primary.v}`))
    fail(check('a(t) s', close(at.primary.s, 80 / 3, 2e-3), `got ${at.primary.s}`))
  }

  const as_ = solveFromExpression('a-expr', '2*s + 4', 0, 0, 5, { kind: 's', value: 10 }, 30)
  fail(check('a(s) ok', as_.ok))
  if (as_.ok) {
    fail(check('a(s) v', close(as_.primary.v, Math.sqrt(305), 5e-3), `got ${as_.primary.v}`))
  }

  const spring = solveFromExpression('a-expr', '-4*s', 0, 0.5, 0, { kind: 's', value: 0 }, 5)
  fail(check('spring ok', spring.ok))
  if (spring.ok) {
    fail(check('spring t', close(spring.primary.t, Math.PI / 4, 2e-2), `got ${spring.primary.t}`))
    fail(check('spring v', close(spring.primary.v, -1, 2e-2), `got ${spring.primary.v}`))
  }

  const vt = solveFromExpression('v-of-t', '4*t + 2', 0, 0, 2, { kind: 't', value: 3 }, 30)
  fail(check('v(t) ok', vt.ok))
  if (vt.ok) {
    fail(check('v(t) v', close(vt.primary.v, 14, 2e-3), `got ${vt.primary.v}`))
    fail(check('v(t) s', close(vt.primary.s, 24, 2e-3), `got ${vt.primary.s}`))
    fail(check('v(t) a', close(vt.primary.a, 4, 2e-3), `got ${vt.primary.a}`))
  }

  const drag = solveFromExpression('a-expr', '-0.05*v^2', 0, 0, 20, { kind: 'v', value: 8 }, 60)
  fail(check('drag ok', drag.ok))
  if (drag.ok) {
    // v = v0 / (1 + k v0 t) with k=0.05, v=8, v0=20 ⇒ t = (1/8 - 1/20)/0.05 = (0.125-0.05)/0.05 = 1.5
    fail(check('drag t', close(drag.primary.t, 1.5, 2e-2), `got ${drag.primary.t}`))
  }

  const toss = solveConstA({ s0: 0, v0: 10, s: 0, v: null, a: -10, t: null })
  fail(check('toss return ok', toss.ok))
  if (toss.ok) {
    fail(check('toss prefers t=2 over t=0', close(toss.primary.t, 2), `got ${toss.primary.t}`))
    fail(check('toss has t=0 alt', toss.alternatives.some((p) => close(p.t, 0)), `alts ${toss.alternatives.map((p) => p.t).join(',')}`))
  }

  const snow = solve2D({
    xA: 0,
    yA: 2,
    xB: 14,
    yB: 3.5,
    ax: 0,
    ay: -32.2,
    v0: null,
    thetaDeg: 40,
    t: null,
  })
  fail(check('snowblower ok', snow.ok, snow.ok ? '' : snow.error))
  if (snow.ok) {
    fail(check('snowblower v0', close(snow.primary.v0, 22.907, 2e-3), `got ${snow.primary.v0}`))
    fail(check('snowblower t', close(snow.primary.t, 0.7978, 2e-3), `got ${snow.primary.t}`))
    fail(check('snowblower lands B', close(snow.primary.xB, 14) && close(snow.primary.yB, 3.5)))
  }

  const twoTh = solve2D({
    xA: 0,
    yA: 0,
    xB: 30,
    yB: 8,
    ax: 0,
    ay: -9.81,
    v0: 25,
    thetaDeg: null,
    t: null,
  })
  fail(check('two θ ok', twoTh.ok, twoTh.ok ? '' : twoTh.error))
  if (twoTh.ok) {
    fail(check('two θ count', twoTh.alternatives.length >= 1, `alts ${twoTh.alternatives.length}`))
    const thetas = [twoTh.primary, ...twoTh.alternatives].map((p) => p.thetaDeg).sort((a, b) => a - b)
    fail(check('two θ low', close(thetas[0], 30.2, 5e-2), `got ${thetas[0]}`))
    fail(check('two θ high', close(thetas[thetas.length - 1], 74.7, 5e-2), `got ${thetas[thetas.length - 1]}`))
  }

  const fwd = solve2D({
    xA: 0,
    yA: 0,
    xB: null,
    yB: null,
    ax: 0,
    ay: -9.81,
    v0: 20,
    thetaDeg: 35,
    t: 2,
  })
  fail(check('forward ok', fwd.ok, fwd.ok ? '' : fwd.error))
  if (fwd.ok) {
    const th = (35 * Math.PI) / 180
    fail(check('forward x', close(fwd.primary.xB, 20 * Math.cos(th) * 2, 2e-3), `got ${fwd.primary.xB}`))
    fail(check('forward y', close(fwd.primary.yB, 20 * Math.sin(th) * 2 - 0.5 * 9.81 * 4, 2e-3), `got ${fwd.primary.yB}`))
  }

  return fails
}

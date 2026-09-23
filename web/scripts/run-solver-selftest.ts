import { runKinematicsSolverSelftest } from '../src/kinematics/solver.selftest.ts'

const fails = runKinematicsSolverSelftest()
if (fails.length) {
  console.error(`FAILED ${fails.length}`)
  for (const f of fails) console.error(' -', f)
  process.exit(1)
}
console.log('kinematics solver selftest: all passed')

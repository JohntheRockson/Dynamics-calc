import { runWorkspaceChecks } from '../src/kinematics/workspace/evaluate.ts'
import { runMathChecks } from '../src/kinematics/workspace/math/checks.ts'

const fails = [...runMathChecks(), ...runWorkspaceChecks()]
if (fails.length) {
  console.error(`FAILED ${fails.length}`)
  for (const f of fails) console.error(' -', f)
  process.exit(1)
}
console.log('math console checks: all passed')

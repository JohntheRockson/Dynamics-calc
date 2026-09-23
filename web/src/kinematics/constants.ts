// Reference data for the Kinematics Lab: gravity presets, drag-shape
// presets, and a standard-atmosphere air density model. Kept separate from
// physics.ts (the actual motion equations) so the "what values are
// reasonable" data is easy to scan/extend on its own.

export interface GravityPreset {
  key: string
  label: string
  g: number
}

export const GRAVITY_PRESETS: GravityPreset[] = [
  { key: 'earth', label: 'Earth (9.81 m/s²)', g: 9.81 },
  { key: 'moon', label: 'Moon (1.62 m/s²)', g: 1.62 },
  { key: 'mars', label: 'Mars (3.71 m/s²)', g: 3.71 },
  { key: 'jupiter', label: 'Jupiter (24.79 m/s²)', g: 24.79 },
  { key: 'custom', label: 'Custom', g: 9.81 },
]

export interface ShapePreset {
  key: string
  label: string
  cd: number
}

// Standard reference drag coefficients (dimensionless), quadratic-drag model.
export const SHAPE_PRESETS: ShapePreset[] = [
  { key: 'sphere', label: 'Sphere (smooth), Cd=0.47', cd: 0.47 },
  { key: 'streamlined', label: 'Streamlined body / airfoil, Cd=0.04', cd: 0.04 },
  { key: 'cone', label: 'Cone (point forward), Cd=0.50', cd: 0.5 },
  { key: 'cylinder', label: 'Cylinder (side-on), Cd=1.15', cd: 1.15 },
  { key: 'cube', label: 'Cube, Cd=1.05', cd: 1.05 },
  { key: 'skydiver', label: 'Skydiver (belly-down), Cd=1.00', cd: 1.0 },
  { key: 'plate', label: 'Flat plate (perpendicular), Cd=1.28', cd: 1.28 },
  { key: 'custom', label: 'Custom', cd: 0.47 },
]

export const G0_STANDARD = 9.80665 // m/s^2, standard gravity (used only inside the ISA model itself)
export const AIR_GAS_CONSTANT = 287.05 // J/(kg*K), specific gas constant for dry air
export const SEA_LEVEL_PRESSURE = 101_325 // Pa
export const TROPOPAUSE_ALTITUDE = 11_000 // m
export const LAPSE_RATE = 0.0065 // K/m, troposphere

/**
 * Air density (kg/m^3) from a barometric/ISA-style model: exponential
 * pressure decay with a linear temperature lapse rate up to the
 * tropopause (11 km), isothermal above it. `groundTempC` is the
 * temperature at the reference altitude (h=0), i.e. "local conditions",
 * not necessarily the fixed 15 degC ISA standard -- this is what makes
 * "temperature" an independent, adjustable parameter as requested.
 */
export function airDensity(altitude_m: number, groundTempC: number): number {
  const t0 = groundTempC + 273.15
  const h = Math.max(0, altitude_m)
  const exponent = G0_STANDARD / (LAPSE_RATE * AIR_GAS_CONSTANT)
  if (h <= TROPOPAUSE_ALTITUDE) {
    const t = t0 - LAPSE_RATE * h
    const p = SEA_LEVEL_PRESSURE * (t / t0) ** exponent
    return p / (AIR_GAS_CONSTANT * t)
  }
  const t11 = t0 - LAPSE_RATE * TROPOPAUSE_ALTITUDE
  const p11 = SEA_LEVEL_PRESSURE * (t11 / t0) ** exponent
  const p = p11 * Math.exp((-G0_STANDARD * (h - TROPOPAUSE_ALTITUDE)) / (AIR_GAS_CONSTANT * t11))
  return p / (AIR_GAS_CONSTANT * t11)
}

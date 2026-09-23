// Closed statement catalog for the kinematics workspace.
// Plain words, not a programming language: the composer filters this list
// as the user types, then turns the chosen statement into labeled slots.
// A later agent can call the same ids through `commitCommand`.

export type ArgKind = 'new-name' | 'point' | 'number' | 'optional-number'

export interface CommandArg {
  id: string
  label: string
  kind: ArgKind
  unit?: string
  placeholder?: string
}

export interface CommandDef {
  id: string
  title: string
  aliases: string[]
  blurb: string
  args: CommandArg[]
}

const pointArg: CommandArg = { id: 'point', label: 'Point', kind: 'point' }

export const COMMANDS: CommandDef[] = [
  {
    id: 'point',
    title: 'Point',
    aliases: ['particle'],
    blurb: 'Add a point',
    args: [{ id: 'name', label: 'Name', kind: 'new-name', placeholder: 'A' }],
  },
  {
    id: 'position',
    title: 'Position',
    aliases: ['place', 'location'],
    blurb: 'Set x, y, and optional z',
    args: [
      pointArg,
      { id: 'x', label: 'x', kind: 'number', unit: 'm', placeholder: '0' },
      { id: 'y', label: 'y', kind: 'number', unit: 'm', placeholder: '0' },
      { id: 'z', label: 'z', kind: 'optional-number', unit: 'm', placeholder: 'optional' },
    ],
  },
  {
    id: 'velocity',
    title: 'Velocity',
    aliases: ['vx', 'vy'],
    blurb: 'Rectangular velocity',
    args: [
      pointArg,
      { id: 'vx', label: 'vx', kind: 'number', unit: 'm/s', placeholder: '0' },
      { id: 'vy', label: 'vy', kind: 'number', unit: 'm/s', placeholder: '0' },
      { id: 'vz', label: 'vz', kind: 'optional-number', unit: 'm/s', placeholder: 'optional' },
    ],
  },
  {
    id: 'acceleration',
    title: 'Acceleration',
    aliases: ['ax', 'ay'],
    blurb: 'Rectangular acceleration',
    args: [
      pointArg,
      { id: 'ax', label: 'ax', kind: 'number', unit: 'm/s²', placeholder: '0' },
      { id: 'ay', label: 'ay', kind: 'number', unit: 'm/s²', placeholder: '0' },
      { id: 'az', label: 'az', kind: 'optional-number', unit: 'm/s²', placeholder: 'optional' },
    ],
  },
  {
    id: 'speed',
    title: 'Speed',
    aliases: ['tangential velocity', 'v'],
    blurb: 'Tangential speed',
    args: [pointArg, { id: 'value', label: 'Speed', kind: 'number', unit: 'm/s', placeholder: '0' }],
  },
  {
    id: 'direction',
    title: 'Direction',
    aliases: ['heading', 'psi', 'angle'],
    blurb: 'Direction of the velocity',
    args: [pointArg, { id: 'value', label: 'Direction', kind: 'number', unit: 'deg', placeholder: '0' }],
  },
  {
    id: 'tangential-acceleration',
    title: 'Tangential acceleration',
    aliases: ['at', 'tangential accel'],
    blurb: 'Speeds the point up or slows it down',
    args: [pointArg, { id: 'value', label: 'at', kind: 'number', unit: 'm/s²', placeholder: '0' }],
  },
  {
    id: 'normal-acceleration',
    title: 'Normal acceleration',
    aliases: ['an', 'normal accel', 'centripetal'],
    blurb: 'Acceleration toward the center of curvature',
    args: [pointArg, { id: 'value', label: 'an', kind: 'number', unit: 'm/s²', placeholder: '0' }],
  },
  {
    id: 'radius',
    title: 'Radius',
    aliases: ['rho', 'radius of curvature'],
    blurb: 'Radius of the path',
    args: [pointArg, { id: 'value', label: 'Radius', kind: 'number', unit: 'm', placeholder: '1' }],
  },
  {
    id: 'angular-velocity',
    title: 'Angular velocity',
    aliases: ['omega', 'angular speed'],
    blurb: 'How fast the angle is changing',
    args: [pointArg, { id: 'value', label: 'ω', kind: 'number', unit: 'rad/s', placeholder: '0' }],
  },
  {
    id: 'angular-acceleration',
    title: 'Angular acceleration',
    aliases: ['alpha'],
    blurb: 'How fast the angular velocity is changing',
    args: [pointArg, { id: 'value', label: 'α', kind: 'number', unit: 'rad/s²', placeholder: '0' }],
  },
  {
    id: 'circle',
    title: 'Circle',
    aliases: ['circular', 'curved path'],
    blurb: 'Move on a circle of constant radius',
    args: [pointArg],
  },
  {
    id: 'circle-origin',
    title: 'Circle about origin',
    aliases: ['orbit', 'circle origin'],
    blurb: 'Circle centered at the origin',
    args: [pointArg],
  },
  {
    id: 'relative',
    title: 'Relative',
    aliases: ['relative motion', 'relative to'],
    blurb: 'Position, velocity, and acceleration of one point relative to another',
    args: [
      { id: 'from', label: 'From', kind: 'point' },
      { id: 'to', label: 'To', kind: 'point' },
    ],
  },
  {
    id: 'simulate',
    title: 'Simulate',
    aliases: ['play', 'time', 'animate'],
    blurb: 'Move the point for a length of time',
    args: [
      pointArg,
      { id: 'duration', label: 'Duration', kind: 'number', unit: 's', placeholder: '4' },
    ],
  },
]

export function commandById(id: string): CommandDef | undefined {
  return COMMANDS.find((c) => c.id === id)
}

/** Commands whose title or alias matches what the user has typed so far. */
export function filterCommands(query: string): CommandDef[] {
  const q = query.trim().toLowerCase()
  if (!q) return COMMANDS
  const ranked = COMMANDS.map((command) => {
    const names = [command.title, ...command.aliases].map((name) => name.toLowerCase())
    const starts = names.some((name) => name.startsWith(q))
    const includes = names.some((name) => name.includes(q))
    return { command, starts, includes }
  }).filter((row) => row.starts || row.includes)
  ranked.sort((a, b) => Number(b.starts) - Number(a.starts) || a.command.title.localeCompare(b.command.title))
  return ranked.map((row) => row.command)
}

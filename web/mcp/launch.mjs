// Starts the math MCP server without npm in the middle.
// Cursor talks to this process on stdin and stdout. npm's script runner often
// exits or fails to pass that pipe through, which Cursor reports as
// "Connection closed". This file is plain JavaScript so Node can run it
// before tsx is loaded.

import { spawn } from 'node:child_process'
import { existsSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const web = dirname(here)
const tsx = join(web, 'node_modules', 'tsx', 'dist', 'cli.mjs')
const server = join(here, 'server.ts')

if (!existsSync(tsx)) {
  console.error('dynamics-math: dependencies are missing. In the web folder, run npm install, then reload Cursor.')
  process.exit(1)
}

const child = spawn(process.execPath, [tsx, server], { stdio: 'inherit' })
child.on('error', (error) => {
  console.error(`dynamics-math: ${error.message}`)
  process.exit(1)
})
child.on('exit', (code, signal) => {
  if (signal) console.error(`dynamics-math stopped (${signal})`)
  else if (code) console.error(`dynamics-math stopped with code ${code}`)
  process.exit(code ?? 1)
})

// MCP server for Cursor and other agents.
//
// MCP is a small protocol over stdin/stdout. The client (Cursor) starts this
// process, asks what tools it has, and later asks it to run one. Anything
// written to stdout must be protocol, so notes go to stderr.
//
// Connect it from the repo root with the project's .cursor/mcp.json, or run
// `npm run mcp` in web/ yourself. This process does not call Grok. The model
// that decides to use a tool is whoever connected: Cursor's agent, or the
// separate `npm run agent` process.

import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js'
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js'
import { z } from 'zod'
import { evaluateMath, listMathFunctions } from './mathTools.ts'

const server = new McpServer({ name: 'dynamics-math', version: '0.1.0' })

server.registerTool(
  'list_math_functions',
  {
    title: 'List math functions',
    description: 'List the calculator functions, the inputs they require, and their optional settings.',
    inputSchema: {},
  },
  async () => ({ content: [{ type: 'text', text: JSON.stringify(listMathFunctions()) }] }),
)

server.registerTool(
  'evaluate_math',
  {
    title: 'Evaluate math',
    description: 'Run one or more calculator lines. Later lines see names defined above them. Returns each line\'s text and whether it drew a curve.',
    inputSchema: {
      script: z.string().describe('Console lines separated by newlines.'),
      angles: z.enum(['rad', 'deg']).optional().describe('Angle unit. Defaults to radians.'),
    },
  },
  async ({ script, angles }) => ({ content: [{ type: 'text', text: JSON.stringify(evaluateMath(script, angles ?? 'rad')) }] }),
)

async function main(): Promise<void> {
  const transport = new StdioServerTransport()
  await server.connect(transport)
  console.error('dynamics-math MCP server is connected on stdin/stdout')
}

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.message : error)
  process.exit(1)
})

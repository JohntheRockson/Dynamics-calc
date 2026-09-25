# Math agent

Two doors into the same calculator.

**Cursor** starts `npm run mcp` from `web/` when this folder is open. That process speaks MCP on stdin/stdout and exposes `list_math_functions` and `evaluate_math`. Cursor's own model decides when to call them. The project config is `.cursor/mcp.json`. It is a project server, so it shows under this repo on the MCP page, not in the plugin marketplace.

**The page** has Ask Grok. The browser posts the question to `/api/agent`. A second process, `npm run agent`, calls Grok 4.7 at `https://api.x.ai/v1/responses`. When Grok asks for a calculation, that process runs it and sends the result back. The key stays in the environment:

```bash
export XAI_API_KEY=your-key
cd web && npm run agent
```

In another terminal, `npm run dev` proxies `/api/agent` to port 8788. If you open the built app through the Rust server instead, that server forwards the same path. Leave `MATH_AGENT_PORT` unset unless 8788 is taken.

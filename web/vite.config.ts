import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // Forward API calls to the Rust backend during development so the
      // browser never needs CORS and the frontend code can just call
      // relative `/api/...` URLs in both dev and the production build
      // (where attitude-server serves this app's own static files).
      // The math agent is a separate process (npm run agent). It must be
      // matched before the general /api proxy, which goes to the Rust server.
      '/api/agent': {
        target: process.env.VITE_AGENT_PROXY_TARGET ?? 'http://127.0.0.1:8788',
        changeOrigin: true,
      },
      '/api': {
        target: process.env.VITE_API_PROXY_TARGET ?? 'http://localhost:8080',
        changeOrigin: true,
      },
    },
  },
})

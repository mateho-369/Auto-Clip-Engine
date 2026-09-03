import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// The FastAPI studio serves the battle-tested vanilla console from
// ai_studio/static (it is what the user asked to keep 100% functional).
// This Vite project is the maintained React prototype and is built into a
// separate ignored folder so `npm run build` can never overwrite the served
// vanilla index/app.js/style.css. To switch to it, point FastAPI at the
// output (or copy the built files) deliberately.
export default defineConfig({
  plugins: [react()],
  base: '/static/react/',
  build: {
    outDir: '../static-react',
    emptyOutDir: true,
    sourcemap: true,
  },
  server: {
    host: '0.0.0.0',
    port: 5173,
    proxy: {
      '/api': 'http://127.0.0.1:8000',
      '/files': 'http://127.0.0.1:8000',
      '/static': 'http://127.0.0.1:8000',
    },
  },
});

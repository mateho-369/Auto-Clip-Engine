import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// The studio uses `npm run build` to output directly into the Python app's
// static root, so a single FastAPI process serves both the API and the UI.
// `/static/` base is required because the FastAPI app only mounts /static.
export default defineConfig({
  plugins: [react()],
  base: '/static/',
  build: {
    outDir: '../ai_studio/static',
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

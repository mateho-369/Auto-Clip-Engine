import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Build target = ../static (the FastAPI app serves that directory at /static).
// Dev server proxies /api + /files to the studio on :8000 so the preview works
// in this repo's live-preview environment too.
export default defineConfig({
  plugins: [react()],
  base: "/static/",
  build: {
    outDir: "../static",
    emptyOutDir: true,
    sourcemap: true,
    chunkSizeWarningLimit: 1200,
  },
  server: {
    host: "0.0.0.0",
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8000",
      "/files": "http://127.0.0.1:8000",
      "/static/api": "http://127.0.0.1:8000",
    },
  },
});

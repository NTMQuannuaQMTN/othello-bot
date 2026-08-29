import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The Python API server (scripts/serve.py) runs on :8000.
//  - `npm run dev`  : Vite dev server on :5173, proxies /api -> :8000
//  - `npm run build`: static bundle in web/dist, which scripts/serve.py serves
//    directly (so `python3 scripts/serve.py` alone works after one build).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: "http://127.0.0.1:8000", changeOrigin: true },
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});

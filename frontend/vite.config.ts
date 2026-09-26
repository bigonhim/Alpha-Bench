import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev: `npm run dev` proxies API/WebSocket calls to the Python server on :8765.
// Prod: `npm run build` emits dist/, served by FastAPI at http://127.0.0.1:8765.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8765",
      "/ws": { target: "ws://127.0.0.1:8765", ws: true },
    },
  },
  build: {
    outDir: "dist",
    chunkSizeWarningLimit: 4000,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes("monaco-editor")) return "monaco";
          if (id.includes("echarts") || id.includes("zrender")) return "echarts";
          if (id.includes("node_modules")) return "vendor";
        },
      },
    },
  },
  worker: { format: "es" },
});

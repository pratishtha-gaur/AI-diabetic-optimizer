import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // Proxy API calls to FastAPI during development
    // This means you can call /predict instead of http://localhost:8000/predict
    proxy: {
      "/predict": "http://localhost:8000",
      "/recommend": "http://localhost:8000",
      "/health": "http://localhost:8000",
    },
  },
});

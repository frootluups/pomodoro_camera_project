import { defineConfig } from "vite";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  root: __dirname,
  publicDir: resolve(__dirname, "public"),
  server: {
    port: 5173,
    open: false,
  },
  build: {
    target: "es2023",
    outDir: resolve(__dirname, "dist"),
    emptyOutDir: true,
    sourcemap: true,
    minify: "esbuild",
    cssMinify: true,
    chunkSizeWarningLimit: 800,
    rollupOptions: {
      input: {
        main: resolve(__dirname, "index.html"),
      },
      output: {
        manualChunks: {
          tfjs: ["@tensorflow/tfjs"],
          blazeface: ["@tensorflow-models/blazeface"],
        },
      },
    },
  },
  worker: { format: "es" },
  optimizeDeps: {
    include: ["@tensorflow/tfjs", "@tensorflow-models/blazeface"],
    exclude: ["packages/core/src/workers/motion.worker.ts"],
  },
  esbuild: {
    drop: process.env.NODE_ENV === "production" ? ["console", "debugger"] : [],
  },
});

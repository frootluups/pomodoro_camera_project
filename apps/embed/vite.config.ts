import { defineConfig } from "vite";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  root: __dirname,
  server: {
    port: 5174,
    open: false,
  },
  build: {
    target: "es2023",
    outDir: resolve(__dirname, "dist"),
    emptyOutDir: false,
    sourcemap: true,
    minify: "esbuild",
    cssMinify: true,
    chunkSizeWarningLimit: 800,
    rollupOptions: {
      input: {
        embed: resolve(__dirname, "embed.html"),
        demo: resolve(__dirname, "embed-demo.html"),
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
});

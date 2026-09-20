import { defineConfig } from "vite";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  root: __dirname,
  build: {
    target: "es2023",
    outDir: resolve(__dirname, "dist"),
    emptyOutDir: false,
    sourcemap: true,
    minify: "esbuild",
    cssMinify: true,
    lib: {
      entry: resolve(__dirname, "src/embed.ts"),
      name: "PomodoroCamera",
      fileName: "pomodoro-embed",
      formats: ["es", "umd"],
    },
  },
  esbuild: {
    // Keep console for embed debugging; host can filter
  },
});

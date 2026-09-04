import { defineConfig } from "vite";

export default defineConfig({
  build: {
    target: "es2023",
    outDir: "dist",
    sourcemap: true,
    minify: "esbuild",
    cssMinify: true,
    lib: {
      entry: "src/embed.ts",
      name: "PomodoroCamera",
      fileName: "pomodoro-embed",
      formats: ["es", "umd"],
    },
  },
  esbuild: {
    // Keep console for embed debugging; host can filter
  },
});

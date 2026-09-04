import { defineConfig } from "vite";

export default defineConfig({
  server: {
    port: 5173,
    open: false,
  },
  build: {
    target: "es2023",
    outDir: "dist",
    sourcemap: true,
    minify: "esbuild",
    cssMinify: true,
    chunkSizeWarningLimit: 800,
    rollupOptions: {
      input: {
        main: "index.html",
        embed: "embed.html",
        demo: "embed-demo.html",
      },
      output: {
        manualChunks: {
          tfjs: ["@tensorflow/tfjs"],
          blazeface: ["@tensorflow-models/blazeface"],
        },
      },
    },
  },
  optimizeDeps: {
    include: ["@tensorflow/tfjs", "@tensorflow-models/blazeface"],
  },
  esbuild: {
    drop: process.env.NODE_ENV === "production" ? ["console", "debugger"] : [],
  },
});

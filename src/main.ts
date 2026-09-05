// src/main.ts — entrypoint (browser). Mirrors main.py:main()
import "./style.css";
import { PomodoroApp } from "./app.ts";

function qs<T extends HTMLElement>(sel: string): T {
  const el = document.querySelector(sel);
  if (!el) throw new Error(`Missing element: ${sel}`);
  return el as T;
}

async function boot(): Promise<void> {
  const video = qs<HTMLVideoElement>("#cam");
  const canvas = qs<HTMLCanvasElement>("#overlay");
  const stageInner = qs<HTMLElement>("#stage-inner");
  const layoutLayer = qs<HTMLElement>("#layout-layer");
  const cameraOffEl = qs<HTMLElement>("#camera-off");
  const settingsDialog = qs<HTMLDialogElement>("#settings-dialog");
  const onboardingEl = qs<HTMLElement>("#onboarding");

  // DPI-aware canvas size — debounced resize
  function sizeCanvas(): void {
    const r = stageInner.getBoundingClientRect();
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = Math.max(320, Math.round(r.width * dpr));
    const h = Math.max(240, Math.round(r.height * dpr));
    if (canvas.width === w && canvas.height === h) return;
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    if (ctx) ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  sizeCanvas();
  let resizeTimer: number | null = null;
  window.addEventListener("resize", () => {
    if (resizeTimer !== null) cancelAnimationFrame(resizeTimer);
    resizeTimer = requestAnimationFrame(() => { resizeTimer = null; sizeCanvas(); });
  });

  const app = new PomodoroApp({ video, canvas, stageInner, layoutLayer, cameraOffEl, settingsDialog, onboardingEl });

  // PWA: register service worker
  if ("serviceWorker" in navigator) {
    navigator.serviceWorker.register("/sw.js").catch(() => {});
  }

  // ARIA live region for timer/focus
  const liveRegion = document.getElementById("live-region");
  let lastAnnounce = 0;
  window.addEventListener("pomodoro:phase", () => {
    if (!liveRegion) return;
    const now = Date.now();
    if (now - lastAnnounce < 3000) return;
    lastAnnounce = now;
    liveRegion.textContent = `${app.timer.currentPhase} ${app.timer.timerText}`;
  });

  // Global error boundary
  window.addEventListener("error", (e) => {
    console.error("Pomodoro error", e.error || e.message);
  });
  window.addEventListener("unhandledrejection", (e) => {
    console.error("Pomodoro rejection", e.reason);
  });

  await app.run();

  // expose for debugging / parity with Python console logs
  (window as unknown as { pomodoroApp: PomodoroApp }).pomodoroApp = app;
  console.info("Pomodoro Camera (web) started — press S to start, E to edit, Q to pause.");
}

boot().catch((e) => {
  console.error(e);
  const el = document.createElement("pre");
  el.textContent = String(e?.stack ?? e);
  el.style.cssText = "position:fixed;inset:12px;background:#1e1e22;color:#ff8a8a;padding:12px;border-radius:12px;z-index:9999;overflow:auto";
  document.body.appendChild(el);
});

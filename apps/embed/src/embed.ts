// apps/embed/src/embed.ts — Embeddable Pomodoro Camera widget
// Provides: <pomodoro-camera> custom element + window.PomodoroCamera API + iframe support
// Usage:
//   <script type="module" src="/apps/embed/dist/pomodoro-embed.js"></script>
//   <pomodoro-camera theme="dark" pomodoro="25" break="5"></pomodoro-camera>
// Or:
//   <div id="my-widget"></div>
//   <script> PomodoroCamera.mount('#my-widget', { theme: 'dark' }) </script>
// Or iframe:
//   <iframe src="/apps/embed/embed.html?theme=dark" allow="camera; microphone" width="420" height="600"></iframe>

import { PomodoroApp } from "../../../packages/core/src/app.ts";
import type { AppDeps } from "../../../packages/core/src/app.ts";

// Inline the widget CSS — scoped to shadow DOM, no global pollution
const WIDGET_CSS = `
:host {
  display: block;
  contain: content;
  --bg: #1c1c1e;
  --panel: #222226;
  --panel-border: #a8a8af;
  --divider: #5c5c62;
  --text: #f0f0f0;
  --subtext: #b9b9be;
  --accent: #78a0ff;
  --on: #8cdc8c;
  --off: #6e6ee6;
  --radius: 14px;
  --shadow: 0 20px 60px rgba(0,0,0,0.35);
  --shadow-lg: 0 24px 80px rgba(0,0,0,0.5);
  font-family: "Segoe UI", system-ui, -apple-system, sans-serif;
  -webkit-font-smoothing: antialiased;
  text-rendering: optimizeLegibility;
  color: var(--text);
}
:host([theme="light"]) {
  --bg: #ececf0;
  --panel: #fafafc;
  --panel-border: #46464e;
  --divider: #a5a5ac;
  --text: #222228;
  --subtext: #5f5f66;
  --accent: #1e5adc;
  --on: #288c3c;
  --off: #283cbe;
}
:host([theme="xp"]) {
  --bg: #d4d0c8;
  --panel: #ece9d8;
  --panel-border: #686868;
  --divider: #aca8a0;
  --text: #101010;
  --subtext: #5a503c;
  --accent: #c8821e;
  --on: #50aa32;
  --off: #3c3cb4;
}
* { box-sizing: border-box; }
.widget {
  display: flex;
  flex-direction: column;
  background: var(--bg);
  color: var(--text);
  border-radius: var(--radius);
  overflow: hidden;
  border: 1px solid var(--divider);
  box-shadow: var(--shadow);
  width: 100%;
  max-width: 100%;
}
.widget.compact { border-radius: 12px; }
.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding: 10px 12px;
  border-bottom: 1px solid var(--divider);
  background: color-mix(in srgb, var(--panel) 88%, transparent);
  backdrop-filter: blur(10px) saturate(1.1);
}
.brand { font-weight: 800; letter-spacing: 0.02em; display: flex; align-items: center; gap: 8px; font-size: 14px; }
.brand::before { content: ""; width: 8px; height: 8px; border-radius: 999px; background: var(--accent); box-shadow: 0 0 12px color-mix(in srgb, var(--accent) 60%, transparent); }
.topbar-actions { display: flex; gap: 6px; flex-wrap: wrap; }
.btn {
  appearance: none;
  border: 1px solid var(--panel-border);
  background: var(--panel);
  color: var(--text);
  padding: 6px 10px;
  border-radius: 999px;
  font-weight: 650;
  font-size: 12px;
  cursor: pointer;
  transition: transform 120ms ease, filter 120ms ease, box-shadow 120ms ease;
  box-shadow: 0 1px 0 rgba(255,255,255,0.06) inset, 0 4px 12px rgba(0,0,0,0.15);
}
.btn:hover { filter: brightness(1.08); }
.btn:active { transform: translateY(1px) scale(0.98); }
.btn:focus-visible { outline: none; box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 35%, transparent); }
.btn-accent { background: linear-gradient(180deg, color-mix(in srgb, var(--accent) 92%, white 8%), var(--accent)); color: white; border-color: transparent; }
.btn-ghost { background: transparent; border-color: var(--divider); color: var(--subtext); box-shadow: none; }
.btn-small { padding: 5px 8px; font-size: 11px; border-radius: 8px; }
.stage {
  position: relative;
  display: grid;
  place-items: center;
  padding: 12px;
  background:
    radial-gradient(600px 300px at 50% -8%, color-mix(in srgb, var(--accent) 14%, transparent), transparent 60%),
    var(--bg);
  min-height: 280px;
}
.stage-inner {
  position: relative;
  width: 100%;
  aspect-ratio: 16 / 10;
  max-height: 420px;
  background: #0e0e10;
  border-radius: 10px;
  overflow: hidden;
  border: 1px solid color-mix(in srgb, var(--divider) 85%, transparent);
  box-shadow: 0 10px 30px rgba(0,0,0,0.25);
}
.compact .stage-inner { aspect-ratio: 16 / 9; max-height: 320px; }
#cam, #overlay {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: cover;
}
#cam { transform: scaleX(-1); }
#overlay { pointer-events: none; }
#layout-layer { position: absolute; inset: 0; pointer-events: none; }
.camera-off {
  position: absolute;
  inset: 0;
  display: grid;
  place-items: center;
  background: color-mix(in srgb, var(--panel) 85%, transparent);
  backdrop-filter: blur(6px);
  text-align: center;
  padding: 16px;
}
.camera-off-card { font-weight: 800; font-size: 16px; }
.camera-off-hint { margin: 6px 0 0; color: var(--subtext); font-size: 12px; }
.hidden { display: none !important; }
.hintbar {
  display: flex;
  gap: 10px;
  justify-content: center;
  padding: 8px 10px;
  color: var(--subtext);
  font-size: 11px;
  border-top: 1px solid var(--divider);
  background: color-mix(in srgb, var(--panel) 55%, transparent);
}
kbd {
  display: inline-block;
  border: 1px solid var(--divider);
  border-bottom-width: 2px;
  padding: 1px 5px;
  border-radius: 5px;
  background: var(--panel);
  font-family: ui-monospace, monospace;
  font-size: 10px;
}
.settings-dialog {
  border: 1px solid color-mix(in srgb, var(--panel-border) 85%, transparent);
  border-radius: 14px;
  padding: 0;
  width: min(520px, 92%);
  max-width: 92%;
  background: linear-gradient(180deg, color-mix(in srgb, var(--panel) 96%, white 4%), var(--panel));
  color: var(--text);
  box-shadow: var(--shadow-lg);
  position: fixed;
  inset: 0;
  margin: auto;
  height: fit-content;
  max-height: 90vh;
  overflow: auto;
  z-index: 100;
}
.settings-dialog::backdrop { background: rgba(0,0,0,0.55); backdrop-filter: blur(4px); }
.settings-form { padding: 14px; }
.settings-header { display: flex; align-items: center; justify-content: space-between; gap: 10px; border-bottom: 1px solid var(--divider); padding-bottom: 10px; }
.settings-header h2 { margin: 0; font-size: 16px; }
.settings-grid { display: grid; gap: 10px; padding: 12px 0; }
.field { display: flex; align-items: center; justify-content: space-between; gap: 8px; padding: 8px 10px; border-radius: 10px; background: color-mix(in srgb, var(--panel) 85%, transparent); border: 1px solid color-mix(in srgb, var(--divider) 55%, transparent); }
.field > span:first-child { color: var(--subtext); font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.02em; }
.row { display: flex; gap: 6px; flex-wrap: wrap; }
.stepper { display: flex; align-items: center; gap: 6px; }
.stepper output { min-width: 3ch; text-align: center; font-weight: 800; font-variant-numeric: tabular-nums; font-size: 13px; }
.settings-footer { display: flex; padding-top: 10px; border-top: 1px solid var(--divider); }
.settings-footer .btn { flex: 1; }
.onboarding {
  position: absolute;
  inset: 0;
  display: grid;
  place-items: center;
  background: rgba(10,10,12,0.58);
  backdrop-filter: blur(6px);
  z-index: 50;
  padding: 12px;
}
.onboarding-card {
  width: min(420px, 92%);
  background: linear-gradient(180deg, #232326, #1c1c1e);
  border: 1px solid color-mix(in srgb, var(--accent) 35%, transparent);
  border-radius: 16px;
  padding: 16px;
  box-shadow: var(--shadow-lg);
  text-align: left;
}
.onboarding-bar { height: 3px; background: var(--accent); border-radius: 999px; margin: -16px -16px 12px; }
.onboarding-card h2 { margin: 0 0 6px; font-size: 16px; color: var(--accent); }
.onboarding-card p { margin: 0; color: var(--text); line-height: 1.5; white-space: pre-line; font-size: 13px; }
.onboarding-dots { display: flex; gap: 6px; justify-content: center; margin: 12px 0 6px; }
.onboarding-dots i { width: 7px; height: 7px; border-radius: 999px; background: #3a3a3e; display: block; transition: all 180ms; }
.onboarding-dots i.active { background: var(--accent); width: 18px; }
.onboarding-actions { display: flex; gap: 6px; margin-top: 12px; }
.onboarding-actions .btn { flex: 1; font-size: 12px; }
`;

const WIDGET_HTML = `
<div class="widget">
  <header class="topbar">
    <div class="brand">🍅 Pomodoro Camera</div>
    <div class="topbar-actions">
      <button id="btn-start" class="btn btn-accent">Start</button>
      <button id="btn-reset" class="btn">Reset</button>
      <button id="btn-settings" class="btn">Settings</button>
      <button id="btn-edit" class="btn btn-ghost">Edit</button>
    </div>
  </header>
  <main class="stage">
    <div class="stage-inner" id="stage-inner">
      <video id="cam" autoplay playsinline muted></video>
      <canvas id="overlay"></canvas>
      <div id="layout-layer"></div>
      <div id="camera-off" class="camera-off hidden">
        <div>
          <div class="camera-off-card">Camera Off</div>
          <p class="camera-off-hint">Enable camera in Settings</p>
        </div>
      </div>
      <div id="onboarding" class="onboarding hidden">
        <div class="onboarding-card">
          <div id="onboarding-bar" class="onboarding-bar"></div>
          <h2 id="onboarding-title">Welcome</h2>
          <p id="onboarding-body"></p>
          <div class="onboarding-dots" id="onboarding-dots"></div>
          <div class="onboarding-actions">
            <button id="onboarding-back" class="btn">Back</button>
            <button id="onboarding-skip" class="btn btn-ghost">Skip</button>
            <button id="onboarding-next" class="btn btn-accent">Next</button>
          </div>
        </div>
      </div>
    </div>
  </main>
  <footer class="hintbar">
    <span><kbd>S</kbd> start</span>
    <span><kbd>E</kbd> edit</span>
    <span><kbd>Esc</kbd> exit</span>
  </footer>
</div>
<dialog id="settings-dialog" class="settings-dialog">
  <form method="dialog" class="settings-form">
    <div class="settings-header">
      <h2>Settings</h2>
      <button type="submit" value="close" class="btn btn-small">Close</button>
    </div>
    <div class="settings-grid">
      <label class="field">
        <span>Pomodoro (min)</span>
        <span class="stepper">
          <button type="button" data-action="pom-minus" class="btn btn-small">−</button>
          <output id="out-pom">25</output>
          <button type="button" data-action="pom-plus" class="btn btn-small">+</button>
        </span>
      </label>
      <label class="field">
        <span>Break (min)</span>
        <span class="stepper">
          <button type="button" data-action="break-minus" class="btn btn-small">−</button>
          <output id="out-break">5</output>
          <button type="button" data-action="break-plus" class="btn btn-small">+</button>
        </span>
      </label>
      <label class="field">
        <span>UI Scale</span>
        <span class="stepper">
          <button type="button" data-action="scale-minus" class="btn btn-small">−</button>
          <output id="out-scale">82%</output>
          <button type="button" data-action="scale-plus" class="btn btn-small">+</button>
        </span>
      </label>
      <div class="field row">
        <button type="button" data-action="toggle-camera" id="btn-toggle-camera" class="btn btn-small">Camera: On</button>
        <button type="button" data-action="toggle-bar" id="btn-toggle-bar" class="btn btn-small">Bar: On</button>
        <button type="button" data-action="toggle-alerts" id="btn-toggle-alerts" class="btn btn-small">Alerts: On</button>
      </div>
      <div class="field row">
        <button type="button" data-action="toggle-theme" id="btn-toggle-theme" class="btn btn-small">Theme: Dark</button>
        <button type="button" data-action="toggle-corners" id="btn-toggle-corners" class="btn btn-small">Corners: Rounded</button>
      </div>
      <div class="field">
        <span>Display mode</span>
        <div class="row">
          <button type="button" data-action="mode-progress" class="btn btn-small">Progress</button>
          <button type="button" data-action="mode-popup" class="btn btn-small">Popup</button>
          <button type="button" data-action="mode-both" class="btn btn-small">Both</button>
        </div>
      </div>
      <div class="field">
        <span>Layout tiles</span>
        <div class="row">
          <button type="button" data-action="layout-timer" class="btn btn-small">Timer</button>
          <button type="button" data-action="layout-bar" class="btn btn-small">Bar</button>
          <button type="button" data-action="layout-focus" class="btn btn-small">Focus</button>
          <button type="button" data-action="layout-phase" class="btn btn-small">Phase</button>
        </div>
      </div>
      <div class="field row">
        <button type="button" data-action="layout-preset" class="btn btn-small">Presets</button>
        <button type="button" data-action="layout-reset" class="btn btn-small">Default</button>
        <button type="button" data-action="layout-grid" class="btn btn-small" id="btn-grid">Grid: 12×8</button>
      </div>
      <div class="field" style="flex-direction:column;align-items:stretch">
        <span>Faces · <span id="gallery-count">None yet</span></span>
        <div id="gallery-list" style="width:100%"></div>
        <div class="row" style="margin-top:8px">
          <button type="button" data-action="gallery-clear" class="btn btn-small">Forget all faces</button>
        </div>
      </div>
    </div>
    <div class="settings-footer">
      <button type="button" data-action="layout-edit" class="btn btn-accent">Layout Edit Mode</button>
    </div>
  </form>
</dialog>
`;

export interface EmbedOptions {
  theme?: "dark" | "light" | "xp";
  pomodoroMinutes?: number;
  breakMinutes?: number;
  cameraEnabled?: boolean;
  compact?: boolean;
  width?: string;
  height?: string;
  showHintbar?: boolean;
}

function parseOptionsFromElement(el: HTMLElement): EmbedOptions {
  const opts: EmbedOptions = {};
  const theme = el.getAttribute("theme") || el.getAttribute("data-theme");
  if (theme === "dark" || theme === "light" || theme === "xp") opts.theme = theme;
  const pom = el.getAttribute("pomodoro") || el.getAttribute("data-pomodoro");
  if (pom) { const n = parseInt(pom, 10); if (!isNaN(n)) opts.pomodoroMinutes = n; }
  const brk = el.getAttribute("break") || el.getAttribute("data-break");
  if (brk) { const n = parseInt(brk, 10); if (!isNaN(n)) opts.breakMinutes = n; }
  const cam = el.getAttribute("camera") || el.getAttribute("data-camera");
  if (cam !== null) opts.cameraEnabled = cam !== "false" && cam !== "0";
  if (el.hasAttribute("compact") || el.getAttribute("data-compact") === "true") opts.compact = true;
  const w = el.getAttribute("width"); if (w) opts.width = w;
  const h = el.getAttribute("height"); if (h) opts.height = h;
  return opts;
}

function applyOptionsToApp(app: PomodoroApp, opts: EmbedOptions): void {
  if (opts.theme) { app.timer.theme = opts.theme; app.timer.saveSettings(); app.applyTheme(); }
  if (opts.pomodoroMinutes !== undefined || opts.breakMinutes !== undefined) {
    const patch: { pomodoroMinutes?: number; breakMinutes?: number } = {};
    if (opts.pomodoroMinutes !== undefined) patch.pomodoroMinutes = opts.pomodoroMinutes;
    if (opts.breakMinutes !== undefined) patch.breakMinutes = opts.breakMinutes;
    app.timer.applySettings(patch);
  }
  if (opts.cameraEnabled === false) {
    app.timer.cameraEnabled = false;
    app.stopCamera();
  }
}

export class PomodoroCameraElement extends HTMLElement {
  private app: PomodoroApp | null = null;
  private shadow: ShadowRoot | null = null;
  private opts: EmbedOptions = {};

  static get observedAttributes(): string[] {
    return ["theme", "data-theme", "pomodoro", "data-pomodoro", "break", "data-break", "camera", "data-camera", "compact"];
  }

  connectedCallback(): void {
    if (this.shadow) return; // already initialized
    this.opts = parseOptionsFromElement(this);
    // Support URL params for iframe mode
    try {
      const params = new URLSearchParams(window.location.search);
      const t = params.get("theme"); if (t === "dark" || t === "light" || t === "xp") this.opts.theme = t;
      const p = params.get("pomodoro"); if (p) { const n = parseInt(p, 10); if (!isNaN(n)) this.opts.pomodoroMinutes = n; }
      const b = params.get("break"); if (b) { const n = parseInt(b, 10); if (!isNaN(n)) this.opts.breakMinutes = n; }
      if (params.get("compact") === "1" || params.get("compact") === "true") this.opts.compact = true;
    } catch {}

    this.shadow = this.attachShadow({ mode: "open" });
    const style = document.createElement("style");
    style.textContent = WIDGET_CSS;
    this.shadow.appendChild(style);

    const wrapper = document.createElement("div");
    wrapper.innerHTML = WIDGET_HTML;
    // Apply compact class
    if (this.opts.compact) wrapper.querySelector(".widget")?.classList.add("compact");
    // Apply width/height if specified
    if (this.opts.width) (this as HTMLElement).style.width = this.opts.width;
    if (this.opts.height) (this as HTMLElement).style.height = this.opts.height;
    if (this.opts.showHintbar === false) {
      const hb = wrapper.querySelector(".hintbar") as HTMLElement | null;
      if (hb) hb.style.display = "none";
    }
    this.shadow.appendChild(wrapper);

    // Set theme attribute for CSS
    if (this.opts.theme) this.setAttribute("theme", this.opts.theme);
    else this.setAttribute("theme", "dark");

    // Query elements inside shadow DOM
    const video = this.shadow.querySelector("#cam") as HTMLVideoElement;
    const canvas = this.shadow.querySelector("#overlay") as HTMLCanvasElement;
    const stageInner = this.shadow.querySelector("#stage-inner") as HTMLElement;
    const layoutLayer = this.shadow.querySelector("#layout-layer") as HTMLElement;
    const cameraOffEl = this.shadow.querySelector("#camera-off") as HTMLElement;
    const settingsDialog = this.shadow.querySelector("#settings-dialog") as HTMLDialogElement;
    const onboardingEl = this.shadow.querySelector("#onboarding") as HTMLElement;

    if (!video || !canvas || !stageInner || !layoutLayer || !cameraOffEl || !settingsDialog || !onboardingEl) {
      console.error("PomodoroCamera: missing shadow DOM elements");
      return;
    }

    // PomodoroApp now supports scoped root/host — pass shadowRoot directly
    const deps: AppDeps = {
      video, canvas, stageInner, layoutLayer, cameraOffEl, settingsDialog, onboardingEl,
      root: this.shadow as unknown as ParentNode & { getElementById?(id: string): HTMLElement | null },
      host: this as unknown as HTMLElement,
    };
    this.app = new PomodoroApp(deps);

    applyOptionsToApp(this.app, this.opts);

    // Size canvas after attach
    requestAnimationFrame(() => {
      this.app?.["handleResize"]?.();
      this.app?.run().catch(console.error);
    });

    // Dispatch ready event
    this.dispatchEvent(new CustomEvent("pomodoro-ready", { detail: { app: this.app } }));
  }

  attributeChangedCallback(name: string, _old: string | null, value: string | null): void {
    if (!this.app) return;
    if (name === "theme" || name === "data-theme") {
      if (value === "dark" || value === "light" || value === "xp") {
        this.app.timer.theme = value;
        this.app.timer.saveSettings();
        this.app.applyTheme();
      }
    } else if (name === "pomodoro" || name === "data-pomodoro") {
      const n = value ? parseInt(value, 10) : NaN;
      if (!isNaN(n)) this.app.timer.applySettings({ pomodoroMinutes: n });
    } else if (name === "break" || name === "data-break") {
      const n = value ? parseInt(value, 10) : NaN;
      if (!isNaN(n)) this.app.timer.applySettings({ breakMinutes: n });
    }
  }

  disconnectedCallback(): void {
    this.app?.dispose();
    this.app = null;
  }

  // Public API for imperative control
  getApp(): PomodoroApp | null { return this.app; }
  start(): void { this.app?.timer.startTimer(); }
  pause(): void { this.app?.timer.stopTimer(); }
  reset(): void { this.app?.timer.resetTimer(); }
  setTheme(theme: "dark" | "light" | "xp"): void {
    if (!this.app) return;
    this.app.timer.theme = theme;
    this.app.timer.saveSettings();
    this.app.applyTheme();
  }
}

let defined = false;
export function definePomodoroElement(tag = "pomodoro-camera"): void {
  if (defined) return;
  if (!customElements.get(tag)) {
    customElements.define(tag, PomodoroCameraElement);
    defined = true;
  }
}

// Global mount API — works with any container
export function mountPomodoro(target: string | HTMLElement, opts: EmbedOptions = {}): PomodoroCameraElement {
  const el = typeof target === "string" ? document.querySelector(target) as HTMLElement | null : target;
  if (!el) throw new Error(`PomodoroCamera.mount: target not found: ${target}`);
  // If target is already a pomodoro-camera element, return it
  if (el instanceof PomodoroCameraElement) return el;
  // Otherwise create and append
  const widget = document.createElement("pomodoro-camera") as PomodoroCameraElement;
  if (opts.theme) widget.setAttribute("theme", opts.theme);
  if (opts.pomodoroMinutes !== undefined) widget.setAttribute("pomodoro", String(opts.pomodoroMinutes));
  if (opts.breakMinutes !== undefined) widget.setAttribute("break", String(opts.breakMinutes));
  if (opts.cameraEnabled === false) widget.setAttribute("camera", "false");
  if (opts.compact) widget.setAttribute("compact", "");
  if (opts.width) widget.style.width = opts.width;
  if (opts.height) widget.style.height = opts.height;
  el.appendChild(widget);
  return widget;
}

// Auto-define on load
definePomodoroElement();

// Expose global for script-tag usage
declare global {
  interface Window {
    PomodoroCamera: {
      mount: typeof mountPomodoro;
      define: typeof definePomodoroElement;
      PomodoroCameraElement: typeof PomodoroCameraElement;
    };
  }
}
if (typeof window !== "undefined") {
  window.PomodoroCamera = {
    mount: mountPomodoro,
    define: definePomodoroElement,
    PomodoroCameraElement,
  };
}

// Auto-mount for embed.html (iframe mode)
if (typeof document !== "undefined" && document.documentElement.hasAttribute("data-pomodoro-embed")) {
  const mountEmbed = (): void => {
    const container = document.getElementById("pomodoro-embed-root") || document.body;
    if (container.querySelector("pomodoro-camera")) return;
    const w = document.createElement("pomodoro-camera") as PomodoroCameraElement;
    try {
      const p = new URLSearchParams(window.location.search);
      const t = p.get("theme"); if (t) w.setAttribute("theme", t);
      const pm = p.get("pomodoro"); if (pm) w.setAttribute("pomodoro", pm);
      const br = p.get("break"); if (br) w.setAttribute("break", br);
      if (p.get("compact") === "1") w.setAttribute("compact", "");
    } catch {}
    container.appendChild(w);
  };
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", mountEmbed);
  } else {
    mountEmbed();
  }
}

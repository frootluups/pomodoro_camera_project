// src/app.ts — main browser app: camera, canvas rendering, layout drag, settings, onboarding
import { CornerStyle, DisplayMode, FocusState, ThemeName } from "./types.ts";
import type { BBox } from "./types.ts";
import { getTheme } from "./theme.ts";
import { LayoutConfig, LayoutManager } from "./layout.ts";
import { PomodoroTimer } from "./pomodoro.ts";
import { VisionEngine } from "./vision.ts";
import { OnboardingManager } from "./onboarding.ts";
import { styledRect, hGradient, drawXpProgressBar, drawXpTitleBar } from "./render.ts";

// DOM refs — injected by main.ts or embed.ts (shadow DOM)
export interface AppDeps {
  video: HTMLVideoElement;
  canvas: HTMLCanvasElement;
  stageInner: HTMLElement;
  layoutLayer: HTMLElement;
  cameraOffEl: HTMLElement;
  settingsDialog: HTMLDialogElement;
  onboardingEl: HTMLElement;
  /** Optional root for scoped queries (shadowRoot or host element). Defaults to document. */
  root?: ParentNode & { getElementById?(id: string): HTMLElement | null };
  /** Optional host element for theme attribute (shadow host). Defaults to document.documentElement */
  host?: HTMLElement;
}

export class PomodoroApp {
  timer: PomodoroTimer;
  vision = new VisionEngine();
  layout: LayoutManager;
  onboarding = new OnboardingManager();
  deps: AppDeps;

  private ctx: CanvasRenderingContext2D;
  private stream: MediaStream | null = null;
  private rafId = 0;
  private dragging: string | null = null;
  private dragOff = { x: 0, y: 0 };
  private layoutEditMode = false;
  private presetIdx = 0;
  private visionEveryN = 2;
  /** Scoped query root — document or shadowRoot */
  private root: ParentNode & { getElementById?(id: string): HTMLElement | null };
  /** Host element for theme attribute */
  private hostEl: HTMLElement;

  constructor(deps: AppDeps) {
    this.deps = deps;
    const c = deps.canvas.getContext("2d");
    if (!c) throw new Error("Canvas 2D not available");
    this.ctx = c;
    this.timer = new PomodoroTimer();
    const loaded = LayoutConfig.load();
    const cfg = loaded ?? LayoutConfig.default();
    // migrate if needed (ensure minimalist defaults)
    if (loaded) {
      const d = LayoutConfig.default();
      for (const [k, v] of Object.entries(d.elements)) if (!cfg.elements[k]) cfg.elements[k] = v;
    }
    this.layout = new LayoutManager(cfg);
    // onboarding: show only on first visit
    if (localStorage.getItem("pomodoro.settings.v1")) this.onboarding.active = false;

    // Scoped root/host for embed (shadow DOM) vs standalone (document)
    this.root = (deps.root ?? document) as ParentNode & { getElementById?(id: string): HTMLElement | null };
    this.hostEl = deps.host ?? document.documentElement;

    this.bindSettingsUI();
    this.bindOnboardingUI();
    this.bindTopbar();
    this.bindKeys();
    this.bindCanvasDrag();
    this.applyTheme();
    // Debounced resize — avoid thrashing on drag-resize
    let resizeTimer: number | null = null;
    window.addEventListener("resize", () => {
      if (resizeTimer !== null) cancelAnimationFrame(resizeTimer);
      resizeTimer = requestAnimationFrame(() => { this.handleResize(); resizeTimer = null; });
    });
  }

  private qs<T extends HTMLElement>(id: string): T | null {
    // Try scoped root first, then document fallback
    const fromRoot = (this.root as unknown as { getElementById?: (id: string) => HTMLElement | null }).getElementById?.(id)
      ?? (this.root as unknown as Document).querySelector?.(`#${id}`) as T | null
      ?? null;
    if (fromRoot) return fromRoot as T;
    return document.getElementById(id) as T | null;
  }

  applyTheme(): void {
    this.hostEl.setAttribute("data-theme", this.timer.theme);
    // Also set on document for standalone mode
    if (this.hostEl !== document.documentElement) {
      document.documentElement.setAttribute("data-theme", this.timer.theme);
    }
  }

  async startCamera(): Promise<boolean> {
    if (!this.timer.cameraEnabled) { this.stopCamera(); return false; }
    try {
      const s = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" }, audio: false });
      this.stream = s;
      this.deps.video.srcObject = s;
      await this.deps.video.play().catch(() => {});
      this.vision.reset();
      return true;
    } catch (e) {
      console.warn("Camera start failed", e);
      this.timer.cameraEnabled = false;
      return false;
    }
  }

  stopCamera(): void {
    if (this.stream) { for (const t of this.stream.getTracks()) t.stop(); this.stream = null; }
    this.deps.video.srcObject = null;
    this.vision.reset();
  }

  bindTopbar(): void {
    this.qs<HTMLButtonElement>("btn-start")?.addEventListener("click", () => { this.timer.unlockAudio(); this.timer.toggleTimer(); });
    this.qs<HTMLButtonElement>("btn-reset")?.addEventListener("click", () => { this.timer.unlockAudio(); this.timer.resetTimer(); });
    this.qs<HTMLButtonElement>("btn-settings")?.addEventListener("click", () => this.openSettings());
    this.qs<HTMLButtonElement>("btn-edit")?.addEventListener("click", () => this.toggleEditMode());
  }

  bindKeys(): void {
    window.addEventListener("keydown", (e) => {
      if (this.onboarding.active) {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); this.timer.unlockAudio(); this.onboarding.next(); if (!this.onboarding.active) this.timer.saveSettings(); this.renderOnboarding(); }
        else if (e.key === "Backspace") { this.onboarding.back(); this.renderOnboarding(); }
        else if (e.key === "Escape") { this.onboarding.skip(); this.timer.saveSettings(); this.renderOnboarding(); }
        return;
      }
      const k = e.key.toLowerCase();
      if (k === "s") { this.timer.unlockAudio(); this.timer.toggleTimer(); }
      else if (k === "e") this.toggleEditMode();
      else if (k === "q") this.timer.stopTimer();
      else if (k === "escape" && this.layoutEditMode) this.toggleEditMode(false);
    });
    // Unlock audio on any first interaction (required by autoplay policy)
    const unlockOnce = (): void => { this.timer.unlockAudio(); document.removeEventListener("click", unlockOnce); document.removeEventListener("keydown", unlockOnce); };
    document.addEventListener("click", unlockOnce);
    document.addEventListener("keydown", unlockOnce);
  }

  bindCanvasDrag(): void {
    const c = this.deps.canvas;
    // Use stageInner coords; also listen on layoutLayer tiles
    const toCanvas = (clientX: number, clientY: number): [number, number] => {
      const r = c.getBoundingClientRect();
      return [(clientX - r.left) * (c.width / r.width), (clientY - r.top) * (c.height / r.height)];
    };

    c.addEventListener("pointerdown", (e) => {
      if (this.onboarding.active) return;
      if (!this.layoutEditMode) return;
      const [x, y] = toCanvas(e.clientX, e.clientY);
      // Done pill hit?
      const pill = this.editDoneRect;
      if (pill && x >= pill[0] && x <= pill[2] && y >= pill[1] && y <= pill[3]) { this.toggleEditMode(false); return; }
      for (const name of Object.keys(this.layout.config.elements)) {
        const [x1, y1, x2, y2] = this.layout.getElementRect(name, c.width, c.height, true);
        if (x >= x1 && x <= x2 && y >= y1 && y <= y2) {
          this.dragging = name; this.dragOff = { x: x - x1, y: y - y1 };
          (e.target as HTMLElement).setPointerCapture(e.pointerId);
          break;
        }
      }
    });
    c.addEventListener("pointermove", (e) => {
      if (!this.dragging) return;
      const [x, y] = toCanvas(e.clientX, e.clientY);
      const cellW = c.width / this.layout.gridCols, cellH = c.height / this.layout.gridRows;
      const el = this.layout.config.elements[this.dragging]!;
      el.x = Math.max(0, Math.min((x - this.dragOff.x) / cellW, this.layout.gridCols - el.width));
      el.y = Math.max(0, Math.min((y - this.dragOff.y) / cellH, this.layout.gridRows - el.height));
    });
    c.addEventListener("pointerup", (e) => {
      if (!this.dragging) return;
      const el = this.layout.config.elements[this.dragging]!;
      const maxQx = Math.max(0, this.layout.gridCols - el.width) * 4;
      const maxQy = Math.max(0, this.layout.gridRows - el.height) * 4;
      el.x = Math.max(0, Math.min(Math.round(el.x * 4), maxQx)) / 4;
      el.y = Math.max(0, Math.min(Math.round(el.y * 4), maxQy)) / 4;
      this.dragging = null;
      try { (e.target as HTMLElement).releasePointerCapture(e.pointerId); } catch {}
      this.layout.config.save();
      this.renderHtmlTiles();
    });
  }

  private editDoneRect: [number, number, number, number] | null = null;

  toggleEditMode(force?: boolean): void {
    this.layoutEditMode = force ?? !this.layoutEditMode;
    this.deps.stageInner.classList.toggle("editing", this.layoutEditMode);
    if (!this.layoutEditMode) this.layout.config.save();
  }

  openSettings(): void {
    this.syncSettingsUI();
    this.deps.settingsDialog.showModal();
  }

  bindSettingsUI(): void {
    const dlg = this.deps.settingsDialog;
    dlg.addEventListener("click", (e) => {
      const t = e.target as HTMLElement;
      if (t === dlg) dlg.close();
    });
    dlg.addEventListener("close", () => {});
    const bind = (action: string, fn: () => void): void => {
      const el = dlg.querySelector(`[data-action="${action}"]`);
      el?.addEventListener("click", fn);
    };
    bind("pom-minus", () => { this.timer.applySettings({ pomodoroMinutes: this.timer.sessionMinutes - 1 }); this.syncSettingsUI(); });
    bind("pom-plus", () => { this.timer.applySettings({ pomodoroMinutes: this.timer.sessionMinutes + 1 }); this.syncSettingsUI(); });
    bind("break-minus", () => { this.timer.applySettings({ breakMinutes: this.timer.breakMinutes - 1 }); this.syncSettingsUI(); });
    bind("break-plus", () => { this.timer.applySettings({ breakMinutes: this.timer.breakMinutes + 1 }); this.syncSettingsUI(); });
    bind("scale-minus", () => { this.timer.uiScale = Math.max(0.6, Math.round((this.timer.uiScale - 0.08) * 100) / 100); this.timer.saveSettings(); this.syncSettingsUI(); });
    bind("scale-plus", () => { this.timer.uiScale = Math.min(1.4, Math.round((this.timer.uiScale + 0.08) * 100) / 100); this.timer.saveSettings(); this.syncSettingsUI(); });
    bind("toggle-camera", async () => {
      this.timer.cameraEnabled = !this.timer.cameraEnabled;
      if (this.timer.cameraEnabled) await this.startCamera(); else this.stopCamera();
      this.syncSettingsUI();
    });
    bind("toggle-bar", () => { this.timer.progressBarEnabled = !this.timer.progressBarEnabled; this.syncSettingsUI(); });
    bind("toggle-alerts", () => { this.timer.alertsEnabled = !this.timer.alertsEnabled; this.timer.saveSettings(); this.syncSettingsUI(); });
    bind("toggle-theme", () => {
      const order: string[] = [ThemeName.Dark, ThemeName.Light, ThemeName.XP];
      const idx = order.indexOf(this.timer.theme); this.timer.theme = order[(idx + 1) % order.length] as typeof this.timer.theme;
      this.timer.saveSettings(); this.applyTheme(); this.syncSettingsUI();
    });
    bind("toggle-corners", () => {
      this.timer.cornerStyle = this.timer.cornerStyle === CornerStyle.Rounded ? CornerStyle.Boxy : CornerStyle.Rounded;
      this.timer.saveSettings(); this.syncSettingsUI();
    });
    bind("mode-progress", () => { this.timer.applySettings({ displayMode: DisplayMode.ProgressBar }); this.syncSettingsUI(); });
    bind("mode-popup", () => { this.timer.applySettings({ displayMode: DisplayMode.TimerPopup }); this.syncSettingsUI(); });
    bind("mode-both", () => { this.timer.applySettings({ displayMode: DisplayMode.Both }); this.syncSettingsUI(); });
    bind("layout-timer", () => { this.layout.enableElement("timer_popup", !this.layout.config.elements["timer_popup"]?.enabled); this.layout.config.save(); this.syncSettingsUI(); this.renderHtmlTiles(); });
    bind("layout-bar", () => { this.layout.enableElement("progress_bar", !this.layout.config.elements["progress_bar"]?.enabled); this.layout.config.save(); this.syncSettingsUI(); this.renderHtmlTiles(); });
    bind("layout-focus", () => { this.layout.enableElement("focus_display", !this.layout.config.elements["focus_display"]?.enabled); this.layout.config.save(); this.syncSettingsUI(); this.renderHtmlTiles(); });
    bind("layout-phase", () => { this.layout.enableElement("phase_label", !this.layout.config.elements["phase_label"]?.enabled); this.layout.config.save(); this.syncSettingsUI(); this.renderHtmlTiles(); });
    bind("layout-preset", () => { this.cyclePreset(); this.syncSettingsUI(); this.renderHtmlTiles(); });
    bind("layout-reset", () => { const d = LayoutConfig.default(); this.layout.config = d; this.layout.gridCols = d.gridCols; this.layout.gridRows = d.gridRows; this.layout.config.save(); this.syncSettingsUI(); this.renderHtmlTiles(); });
    bind("layout-grid", () => {
      let nc = this.layout.gridCols + 2, nr = this.layout.gridRows + 1;
      if (nc > 20) { nc = 12; nr = 8; }
      this.layout.setGrid(nc, nr); this.layout.config.save(); this.syncSettingsUI();
    });
    bind("layout-edit", () => { dlg.close(); this.toggleEditMode(true); });
  }

  syncSettingsUI(): void {
    const q = (id: string): HTMLElement | null => this.deps.settingsDialog.querySelector(`#${id}`) as HTMLElement | null;
    const qa = (sel: string): HTMLElement | null => this.deps.settingsDialog.querySelector(sel) as HTMLElement | null;
    const setOut = (id: string, v: string): void => { const el = q(id); if (el) el.textContent = v; };
    setOut("out-pom", String(this.timer.sessionMinutes));
    setOut("out-break", String(this.timer.breakMinutes));
    setOut("out-scale", `${Math.round(this.timer.uiScale * 100)}%`);
    const camBtn = qa('[data-action="toggle-camera"]');
    if (camBtn) camBtn.textContent = `Camera: ${this.timer.cameraEnabled ? "On" : "Off"}`;
    const barBtn = qa('[data-action="toggle-bar"]');
    if (barBtn) barBtn.textContent = `Bar: ${this.timer.progressBarEnabled ? "On" : "Off"}`;
    const alertsBtn = qa('[data-action="toggle-alerts"]');
    if (alertsBtn) alertsBtn.textContent = `Alerts: ${this.timer.alertsEnabled ? "On" : "Off"}`;
    const themeBtn = qa('[data-action="toggle-theme"]');
    if (themeBtn) themeBtn.textContent = `Theme: ${this.timer.theme[0]?.toUpperCase()}${this.timer.theme.slice(1)}`;
    const cornersBtn = qa('[data-action="toggle-corners"]');
    if (cornersBtn) cornersBtn.textContent = `Corners: ${this.timer.cornerStyle === CornerStyle.Rounded ? "Rounded" : "Boxy"}`;
    const gridBtn = qa("#btn-grid");
    if (gridBtn) gridBtn.textContent = `Grid: ${this.layout.gridCols}×${this.layout.gridRows}`;
    // active display mode highlight
    for (const a of ["progress", "popup", "both"] as const) {
      const btn = qa(`[data-action="mode-${a}"]`);
      if (!btn) continue;
      const isActive = (a === "progress" && this.timer.displayMode === DisplayMode.ProgressBar) || (a === "popup" && this.timer.displayMode === DisplayMode.TimerPopup) || (a === "both" && this.timer.displayMode === DisplayMode.Both);
      btn.classList.toggle("btn-accent", isActive);
    }
  }

  bindOnboardingUI(): void {
    const el = this.deps.onboardingEl;
    el.querySelector("#onboarding-next")?.addEventListener("click", () => { this.onboarding.next(); if (!this.onboarding.active) this.timer.saveSettings(); this.renderOnboarding(); });
    el.querySelector("#onboarding-back")?.addEventListener("click", () => { this.onboarding.back(); this.renderOnboarding(); });
    el.querySelector("#onboarding-skip")?.addEventListener("click", () => { this.onboarding.skip(); this.timer.saveSettings(); this.renderOnboarding(); });
  }

  renderOnboarding(): void {
    const el = this.deps.onboardingEl;
    if (!this.onboarding.active) { el.classList.add("hidden"); return; }
    el.classList.remove("hidden");
    const cur = this.onboarding.current;
    const title = el.querySelector("#onboarding-title") as HTMLElement;
    const body = el.querySelector("#onboarding-body") as HTMLElement;
    const bar = el.querySelector("#onboarding-bar") as HTMLElement;
    const dots = el.querySelector("#onboarding-dots") as HTMLElement;
    const nextBtn = el.querySelector("#onboarding-next") as HTMLButtonElement;
    const backBtn = el.querySelector("#onboarding-back") as HTMLButtonElement;
    if (title) title.textContent = cur.title;
    if (body) body.textContent = cur.body;
    if (bar) bar.style.background = `rgb(${cur.accent[0]} ${cur.accent[1]} ${cur.accent[2]})`;
    if (dots) {
      dots.innerHTML = "";
      for (let i = 0; i < 6; i++) { const d = document.createElement("i"); if (i === this.onboarding.step) d.className = "active"; dots.appendChild(d); }
    }
    if (nextBtn) nextBtn.textContent = this.onboarding.isLast ? "Start  >" : "Next  >";
    if (backBtn) backBtn.style.visibility = this.onboarding.isFirst ? "hidden" : "visible";
  }

  cyclePreset(): void {
    const presets: { name: string; cfg: LayoutConfig }[] = [
      { name: "Default", cfg: LayoutConfig.default() },
      { name: "Focus", cfg: new LayoutConfig(12, 8, { timer_popup: { enabled: true, x: 4.4, y: 0.38, width: 3.2, height: 0.95, anchor: "top-left", margin: 0.14, padding: 0, fontScale: 1.02, customProps: {} }, phase_label: { enabled: false, x: 4.6, y: 0.55, width: 2.9, height: 0.7, anchor: "top-left", margin: 0.12, padding: 0, fontScale: 1, customProps: {} }, progress_bar: { enabled: false, x: 0.30, y: 0.10, width: 11.4, height: 0.12, anchor: "top-left", margin: 0.06, padding: 0, fontScale: 1, customProps: {} }, focus_display: { enabled: false, x: 0.30, y: 0.38, width: 2.7, height: 0.40, anchor: "top-left", margin: 0.06, padding: 0, fontScale: 1, customProps: {} }, status_display: { enabled: false, x: 0.4, y: 1.0, width: 3.2, height: 0.42, anchor: "top-left", margin: 0.08, padding: 0, fontScale: 1, customProps: {} }, main_buttons: { enabled: true, x: 3.6, y: 7.28, width: 4.8, height: 0.60, anchor: "top-left", margin: 0.10, padding: 0, fontScale: 1, customProps: {} }, quit_hint: { enabled: false, x: 10.6, y: 7.45, width: 1.2, height: 0.4, anchor: "top-left", margin: 0.1, padding: 0, fontScale: 1, customProps: {} } }) },
      { name: "Dashboard", cfg: new LayoutConfig(16, 9, { timer_popup: { enabled: true, x: 11.8, y: 0.45, width: 3.6, height: 1.00, anchor: "top-left", margin: 0.12, padding: 0, fontScale: 0.96, customProps: {} }, phase_label: { enabled: false, x: 11.0, y: 2.6, width: 4.6, height: 0.9, anchor: "top-left", margin: 0.15, padding: 0, fontScale: 1, customProps: {} }, progress_bar: { enabled: true, x: 0.35, y: 0.14, width: 15.3, height: 0.14, anchor: "top-left", margin: 0.05, padding: 0, fontScale: 1, customProps: {} }, focus_display: { enabled: true, x: 0.35, y: 0.42, width: 3.8, height: 0.44, anchor: "top-left", margin: 0.06, padding: 0, fontScale: 1, customProps: {} }, status_display: { enabled: false, x: 0.4, y: 2.5, width: 5.0, height: 0.7, anchor: "top-left", margin: 0.1, padding: 0, fontScale: 1, customProps: {} }, main_buttons: { enabled: true, x: 4.8, y: 8.02, width: 6.4, height: 0.72, anchor: "top-left", margin: 0.10, padding: 0, fontScale: 1, customProps: {} }, quit_hint: { enabled: false, x: 14.2, y: 8.45, width: 1.6, height: 0.4, anchor: "top-left", margin: 0.1, padding: 0, fontScale: 1, customProps: {} } }) },
      { name: "Minimal", cfg: new LayoutConfig(12, 8, { timer_popup: { enabled: true, x: 8.8, y: 0.38, width: 2.6, height: 0.82, anchor: "top-left", margin: 0.14, padding: 0, fontScale: 0.88, customProps: {} }, phase_label: { enabled: false, x: 4.4, y: 0.55, width: 3.2, height: 0.8, anchor: "top-left", margin: 0.15, padding: 0, fontScale: 1, customProps: {} }, progress_bar: { enabled: false, x: 0.30, y: 0.10, width: 11.4, height: 0.12, anchor: "top-left", margin: 0.06, padding: 0, fontScale: 1, customProps: {} }, focus_display: { enabled: false, x: 0.30, y: 0.38, width: 2.7, height: 0.40, anchor: "top-left", margin: 0.06, padding: 0, fontScale: 1, customProps: {} }, status_display: { enabled: false, x: 0.4, y: 1.05, width: 3.6, height: 0.5, anchor: "top-left", margin: 0.1, padding: 0, fontScale: 1, customProps: {} }, main_buttons: { enabled: true, x: 4.0, y: 7.32, width: 4.0, height: 0.58, anchor: "top-left", margin: 0.08, padding: 0, fontScale: 1, customProps: {} }, quit_hint: { enabled: false, x: 10.6, y: 7.45, width: 1.2, height: 0.4, anchor: "top-left", margin: 0.1, padding: 0, fontScale: 1, customProps: {} } }) },
    ];
    this.presetIdx = (this.presetIdx + 1) % presets.length;
    const cfg = presets[this.presetIdx]!.cfg;
    this.layout.config = cfg; this.layout.gridCols = cfg.gridCols; this.layout.gridRows = cfg.gridRows;
    this.layout.config.save();
  }

  private handleResize(): void {
    const r = this.deps.stageInner.getBoundingClientRect();
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = Math.max(320, Math.round(r.width * dpr));
    const h = Math.max(240, Math.round(r.height * dpr));
    if (this.deps.canvas.width === w && this.deps.canvas.height === h) return;
    this.deps.canvas.width = w;
    this.deps.canvas.height = h;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.renderHtmlTiles();
  }

  // HTML tile layer (website-friendly alternative to pure canvas tiles)
  renderHtmlTiles(): void {
    const layer = this.deps.layoutLayer;
    // Only render minimal tiles as HTML for accessibility; canvas keeps the chrome
    layer.innerHTML = "";
  }

  // Core render loop — draws webcam + overlays onto canvas
  private draw(): void {
    const canvas = this.deps.canvas;
    const ctx = this.ctx;
    const w = canvas.width / (window.devicePixelRatio ?? 1);
    const h = canvas.height / (window.devicePixelRatio ?? 1);

    // clear
    const theme = getTheme(this.timer.theme);
    const isXp = this.timer.theme === ThemeName.XP;
    const rounded = this.timer.cornerStyle === CornerStyle.Rounded;
    const uiScale = Math.max(0.75, Math.min(2.5, h / 480)) * this.timer.uiScale;

    // Backdrop / video handled via <video> element beneath; we overlay effects on canvas transparent
    ctx.clearRect(0, 0, w, h);

    // Camera-off backdrop when no stream
    const hasVideo = !!this.stream && this.deps.video.readyState >= 2 && this.timer.cameraEnabled;
    this.deps.cameraOffEl.classList.toggle("hidden", hasVideo);

    // Face outlines — need to map cam coords (videoWidth/Height) → canvas coords (mirrored)
    if (hasVideo) this.drawFaceOutlines(w, h, theme, rounded);

    // Progress bar
    this.drawProgressBar(w, h, theme, rounded);
    // Timer popup
    this.drawTimerPopup(w, h, theme, rounded, uiScale);
    // Phase label
    this.drawPhaseLabel(w, h, uiScale);
    // Focus pill
    this.drawFocusDisplay(w, h, theme, uiScale);
    // Topbar label already HTML; keep canvas status minimal

    // Slacking alert
    if (this.timer.isRunning && this.timer.cameraEnabled && this.timer.focusState === FocusState.Slacking && this.timer.alertsEnabled) {
      this.drawSlackingAlert(w, h, rounded, uiScale);
      this.timer.playAlertSound();
    }

    // Layout edit overlay
    if (this.layoutEditMode) this.drawEditOverlay(w, h, theme, rounded, uiScale);

    // Onboarding is HTML overlay; keep canvas pass-through
    this.updateTopbarLabels();
  }

  private drawFaceOutlines(fw: number, fh: number, theme: ReturnType<typeof getTheme>, rounded: boolean): void {
    const tracks = this.vision.tracker.active;
    if (!tracks.length && !this.vision.lastFaces.length) return;
    const list = tracks.length ? tracks : this.vision.lastFaces.map((b, i) => ({ pid: i + 1, bbox: b, smooth: [b[0], b[1], b[0] + b[2], b[1] + b[3]] as [number, number, number, number], color: [120, 160, 255] as [number, number, number], hits: 2, misses: 0, lastUpdate: 0, gid: null, label: `Person #${i + 1}` }));
    const vw = this.deps.video.videoWidth || fw, vh = this.deps.video.videoHeight || fh;
    // video is object-fit: cover — compute letterbox mapping (simplified: assume stageInner covers canvas 1:1)
    // For cover, scale = max(fw/vw, fh/vh), offsets center crop
    const scale = Math.max(fw / vw, fh / vh);
    const fitW = vw * scale, fitH = vh * scale;
    const offX = (fw - fitW) / 2, offY = (fh - fitH) / 2;

    const isSlacking = this.timer.focusState === FocusState.Slacking && this.timer.isRunning;
    const isConcentrated = this.timer.focusState === FocusState.Concentrated;
    for (const p of list) {
      const [x1s, y1s, x2s, y2s] = p.smooth as unknown as [number, number, number, number];
      // mirror X (video is CSS scaleX(-1))
      const mx1 = vw - x2s, mx2 = vw - x1s;
      let bx1 = offX + mx1 * scale, by1 = offY + y1s * scale, bx2 = offX + mx2 * scale, by2 = offY + y2s * scale;
      const pad = 6; bx1 -= pad; by1 -= pad; bx2 += pad; by2 += pad;
      const bw = bx2 - bx1, bh = by2 - by1;
      if (bw < 10 || bh < 10) continue;
      const color: [number, number, number] = isSlacking ? [60, 60, 255] : isConcentrated ? theme.onColor as [number, number, number] : (p.color as [number, number, number]);
      const thick = isSlacking ? 3 : 2;
      const r = rounded ? Math.min(bw, bh) * 0.14 : 0;
      styledRect(this.ctx, bx1, by1, bx2, by2, { border: color, thickness: thick, radius: r });
      // ID pill
      if (bw > 60 && bh > 60) {
        this.ctx.save();
        this.ctx.font = `700 ${Math.max(9, bh * 0.095)}px 'Segoe UI', sans-serif`;
        const label = `#${p.pid}`;
        const tw = this.ctx.measureText(label).width;
        const px1 = bx1 + 4, py1 = Math.max(0, by1 + 4);
        styledRect(this.ctx, px1 - 3, py1 - 1, px1 + tw + 3, py1 + 14, { fill: [color[0] * 0.38, color[1] * 0.38, color[2] * 0.38] as [number, number, number], radius: 3 });
        this.ctx.fillStyle = "white"; this.ctx.textBaseline = "top"; this.ctx.fillText(label, px1, py1);
        this.ctx.restore();
        if (isSlacking) {
          this.ctx.save();
          const tag = "SLACKING"; this.ctx.font = `700 ${Math.max(9, bh * 0.10)}px sans-serif`;
          const ttw = this.ctx.measureText(tag).width; const tx = bx1 + (bw - ttw) / 2, ty = by1 - 16;
          styledRect(this.ctx, tx - 5, ty - 1, tx + ttw + 5, ty + 13, { fill: [40, 40, 210], radius: 3 });
          this.ctx.fillStyle = "white"; this.ctx.textBaseline = "top"; this.ctx.fillText(tag, tx, ty);
          this.ctx.restore();
        }
      }
    }
    if (list.length > 1) {
      this.ctx.save();
      const cnt = String(list.length); this.ctx.font = "700 13px sans-serif";
      const tw = this.ctx.measureText(cnt).width; const bw2 = Math.max(22, tw + 10), bh2 = 18;
      const bx = fw - bw2 - 10, by = 10;
      styledRect(this.ctx, bx, by, bx + bw2, by + bh2, { fill: [22, 22, 26], border: [70, 70, 75], thickness: 1, radius: bh2 / 2 });
      this.ctx.fillStyle = "rgb(200 200 210)"; this.ctx.textAlign = "center"; this.ctx.textBaseline = "middle"; this.ctx.fillText(cnt, bx + bw2 / 2, by + bh2 / 2);
      this.ctx.restore();
    }
  }

  private drawProgressBar(fw: number, fh: number, theme: ReturnType<typeof getTheme>, rounded: boolean): void {
    const el = this.layout.config.elements["progress_bar"];
    if (!el?.enabled || !(this.timer.showProgressBar && this.timer.progressBarEnabled)) return;
    const [x1, y1, x2, y2] = this.layout.getElementRect("progress_bar", fw, fh);
    if (x2 - x1 <= 4) return;
    const { progress, color } = this.timer.resolvePhase();
    const prog = progress;
    if (this.timer.theme === ThemeName.XP) { drawXpProgressBar(this.ctx, x1, y1, x2, y2, prog); return; }
    const r = rounded ? (y2 - y1) / 2 : 0;
    this.ctx.save(); this.ctx.globalAlpha = 0.65;
    styledRect(this.ctx, x1, y1, x2, y2, { fill: [38, 38, 42], radius: r }); this.ctx.restore();
    if (prog > 0) {
      const fx2 = x1 + (x2 - x1) * prog;
      const fill: [number, number, number] = color[0] === 231 ? [120, 160, 255] : color as [number, number, number];
      styledRect(this.ctx, x1, y1, fx2, y2, { fill, radius: r });
    }
  }

  private drawTimerPopup(fw: number, fh: number, theme: ReturnType<typeof getTheme>, rounded: boolean, uiScale: number): void {
    const el = this.layout.config.elements["timer_popup"];
    if (!el?.enabled || !this.timer.showTimerPopup) return;
    const [x1, y1, x2, y2] = this.layout.getElementRect("timer_popup", fw, fh);
    const bw = x2 - x1, bh = y2 - y1; if (bw <= 4 || bh <= 4) return;
    const txt = this.timer.timerText;
    const { text: phase } = this.timer.resolvePhase();
    this.ctx.save();
    // auto-fit font — binary search instead of linear decrement
    const maxPx = Math.min(72, bh * 0.55) * el.fontScale * uiScale;
    let px = Math.max(10, maxPx);
    // Quick check: if max fits, skip loop
    this.ctx.font = `700 ${px}px 'Segoe UI', sans-serif`;
    if (this.ctx.measureText(txt).width > bw - 12) {
      let lo = 10, hi = px, best = 10;
      while (lo <= hi) {
        const mid = (lo + hi) >> 1;
        this.ctx.font = `700 ${mid}px 'Segoe UI', sans-serif`;
        if (this.ctx.measureText(txt).width <= bw - 12) { best = mid; lo = mid + 1; }
        else hi = mid - 1;
      }
      px = best;
      this.ctx.font = `700 ${px}px 'Segoe UI', sans-serif`;
    }
    if (this.timer.theme === ThemeName.XP) {
      const tbH = Math.max(18, bh * 0.26);
      drawXpTitleBar(this.ctx, x1, y1, x2, y1 + tbH, phase);
      styledRect(this.ctx, x1, y1 + tbH, x2, y2, { fill: theme.popupFill as [number, number, number], border: [104, 104, 104], thickness: 1 });
      this.ctx.fillStyle = `rgb(${theme.text[0]} ${theme.text[1]} ${theme.text[2]})`; this.ctx.textAlign = "center"; this.ctx.textBaseline = "middle";
      this.ctx.fillText(txt, x1 + bw / 2, y1 + tbH + (bh - tbH) / 2);
    } else {
      const r = rounded ? Math.min(bw, bh) * 0.32 : 0;
      this.ctx.globalAlpha = 0.72; styledRect(this.ctx, x1, y1, x2, y2, { fill: [32, 32, 36], radius: r }); this.ctx.globalAlpha = 1;
      styledRect(this.ctx, x1, y1, x2, y2, { border: [68, 68, 75], thickness: 1, radius: r });
      this.ctx.fillStyle = `rgb(${theme.text[0]} ${theme.text[1]} ${theme.text[2]})`; this.ctx.textAlign = "center"; this.ctx.textBaseline = "middle";
      this.ctx.fillText(txt, x1 + bw / 2, y1 + bh / 2);
    }
    this.ctx.restore();
  }

  private drawPhaseLabel(fw: number, fh: number, uiScale: number): void {
    const el = this.layout.config.elements["phase_label"]; if (!el?.enabled) return;
    const [x1, y1, x2, y2] = this.layout.getElementRect("phase_label", fw, fh);
    const w = x2 - x1, h = y2 - y1; if (w <= 4 || h <= 4) return;
    const { text, color } = this.timer.resolvePhase();
    this.ctx.save(); this.ctx.font = `700 ${Math.max(10, h * 0.5 * uiScale)}px 'Segoe UI', sans-serif`;
    this.ctx.fillStyle = `rgb(${color[0]} ${color[1]} ${color[2]})`; this.ctx.textAlign = "center"; this.ctx.textBaseline = "middle";
    this.ctx.fillText(text, x1 + w / 2, y1 + h / 2); this.ctx.restore();
  }

  private drawFocusDisplay(fw: number, fh: number, theme: ReturnType<typeof getTheme>, uiScale: number): void {
    const el = this.layout.config.elements["focus_display"]; if (!el?.enabled) return;
    const [x1, y1, x2, y2] = this.layout.getElementRect("focus_display", fw, fh);
    if (x2 - x1 <= 4 || y2 - y1 <= 4) return;
    const n = this.vision.tracker.active.length;
    const txt = !this.timer.isRunning ? `Ready${n ? ` · ${n}` : ""}` : `${this.timer.focusState[0]?.toUpperCase()}${this.timer.focusState.slice(1)}${n > 1 ? ` · ${n}` : ""}`;
    this.ctx.save();
    const h = y2 - y1; let px = Math.max(10, h * 0.5 * uiScale);
    this.ctx.font = `600 ${px}px 'Segoe UI', sans-serif`;
    if (this.ctx.measureText(txt).width > (x2 - x1) - 14) {
      let lo = 16, hi = Math.round(px * 2), best = 8;
      // Binary search for fitting size (px is float, search integer*0.5 steps)
      lo = 16; hi = Math.round(px * 2);
      while (lo <= hi) {
        const mid = (lo + hi) >> 1;
        const testPx = mid * 0.5;
        this.ctx.font = `600 ${testPx}px 'Segoe UI', sans-serif`;
        if (this.ctx.measureText(txt).width <= (x2 - x1) - 14) { best = mid; lo = mid + 1; }
        else hi = mid - 1;
      }
      px = Math.max(8, best * 0.5);
      this.ctx.font = `600 ${px}px 'Segoe UI', sans-serif`;
    }
    const tw = this.ctx.measureText(txt).width, th = px * 0.9;
    const pillW = tw + 14, pillH = th + 8;
    const px1 = x1 + 4, py1 = y1 + (y2 - y1 - pillH) / 2;
    this.ctx.globalAlpha = 0.62; styledRect(this.ctx, px1, py1, px1 + pillW, py1 + pillH, { fill: [28, 28, 32], radius: pillH / 2 }); this.ctx.globalAlpha = 1;
    styledRect(this.ctx, px1, py1, px1 + pillW, py1 + pillH, { border: [60, 60, 68], thickness: 1, radius: pillH / 2 });
    this.ctx.fillStyle = `rgb(${theme.text[0]} ${theme.text[1]} ${theme.text[2]})`; this.ctx.textAlign = "center"; this.ctx.textBaseline = "middle"; this.ctx.fillText(txt, px1 + pillW / 2, py1 + pillH / 2);
    this.ctx.restore();
  }

  private drawSlackingAlert(fw: number, fh: number, rounded: boolean, uiScale: number): void {
    const pulse = 0.5 + 0.5 * Math.sin(Date.now() / 200);
    const txt = "You're Slacking!"; const px = Math.max(14, fh * 0.035 * uiScale);
    this.ctx.save(); this.ctx.font = `700 ${px}px 'Segoe UI', sans-serif`;
    const tw = this.ctx.measureText(txt).width, th = px * 0.9;
    const padX = 24, padY = 10; const bw = tw + 2 * padX, bh = th + 2 * padY;
    const bx1 = (fw - bw) / 2, by1 = fh * 0.18;
    const col: [number, number, number] = [40, 40, 230];
    const fill: [number, number, number] = [Math.round(col[0] * (0.8 + 0.2 * pulse)), Math.round(col[1] * (0.8 + 0.2 * pulse)), Math.round(col[2] * (0.8 + 0.2 * pulse))];
    const r = rounded ? bh * 0.3 : 0;
    styledRect(this.ctx, bx1, by1, bx1 + bw, by1 + bh, { fill, border: col, thickness: 2, radius: r });
    this.ctx.fillStyle = "white"; this.ctx.textAlign = "center"; this.ctx.textBaseline = "middle"; this.ctx.fillText(txt, bx1 + bw / 2, by1 + bh / 2);
    // border pulse
    const bt = 4 + 3 * pulse; this.ctx.fillStyle = `rgba(230,40,40,${0.6 + 0.4 * pulse})`;
    this.ctx.fillRect(0, 0, fw, bt); this.ctx.fillRect(0, fh - bt, fw, bt); this.ctx.fillRect(0, 0, bt, fh); this.ctx.fillRect(fw - bt, 0, bt, fh);
    this.ctx.restore();
  }

  private drawEditOverlay(fw: number, fh: number, theme: ReturnType<typeof getTheme>, rounded: boolean, uiScale: number): void {
    const lm = this.layout;
    const cellW = fw / lm.gridCols, cellH = fh / lm.gridRows;
    this.ctx.save(); this.ctx.strokeStyle = `rgb(${theme.divider[0]} ${theme.divider[1]} ${theme.divider[2]})`; this.ctx.lineWidth = 1; this.ctx.globalAlpha = 0.5;
    for (let i = 0; i <= lm.gridCols; i++) { const x = Math.round(i * cellW) + 0.5; this.ctx.beginPath(); this.ctx.moveTo(x, 0); this.ctx.lineTo(x, fh); this.ctx.stroke(); }
    for (let i = 0; i <= lm.gridRows; i++) { const y = Math.round(i * cellH) + 0.5; this.ctx.beginPath(); this.ctx.moveTo(0, y); this.ctx.lineTo(fw, y); this.ctx.stroke(); }
    this.ctx.globalAlpha = 1;
    for (const [name, el] of Object.entries(lm.config.elements)) {
      const [x1, y1, x2, y2] = lm.getElementRect(name, fw, fh, true);
      if (x2 <= x1 || y2 <= y1) continue;
      const isDrag = name === this.dragging;
      const col: [number, number, number] = isDrag ? theme.accent as [number, number, number] : !el.enabled ? theme.subtext as [number, number, number] : theme.onColor as [number, number, number];
      styledRect(this.ctx, x1, y1, x2, y2, { border: col, thickness: isDrag ? 3 : el.enabled ? 2 : 1, radius: rounded ? 6 : 0 });
      this.ctx.fillStyle = `rgb(${col[0]} ${col[1]} ${col[2]})`; this.ctx.font = `600 ${Math.max(10, 11 * uiScale)}px sans-serif`; this.ctx.textBaseline = "top"; this.ctx.fillText(name.replaceAll("_", " "), x1 + 5, y1 + 4);
      this.ctx.fillStyle = `rgb(${theme.subtext[0]} ${theme.subtext[1]} ${theme.subtext[2]})`; this.ctx.font = `400 ${Math.max(9, 10 * uiScale)}px sans-serif`;
      this.ctx.fillText(`(${el.x.toFixed(1)}, ${el.y.toFixed(1)})`, x1 + 5, y2 - 12);
    }
    // Done pill
    const txt = "[ Done ]"; this.ctx.font = `700 ${Math.max(11, 14 * uiScale)}px sans-serif`;
    const tw = this.ctx.measureText(txt).width; const pad = 10; const bw = tw + 2 * pad, bh = 28;
    const px1 = (fw - bw) / 2, py1 = 10;
    styledRect(this.ctx, px1, py1, px1 + bw, py1 + bh, { fill: theme.panelFill as [number, number, number], border: theme.accent as [number, number, number], thickness: 2, radius: rounded ? bh * 0.35 : 0 });
    this.ctx.fillStyle = `rgb(${theme.text[0]} ${theme.text[1]} ${theme.text[2]})`; this.ctx.textAlign = "center"; this.ctx.textBaseline = "middle"; this.ctx.fillText(txt, px1 + bw / 2, py1 + bh / 2);
    this.editDoneRect = [px1, py1, px1 + bw, py1 + bh];
    this.ctx.fillStyle = `rgb(${theme.subtext[0]} ${theme.subtext[1]} ${theme.subtext[2]})`; this.ctx.font = `400 ${Math.max(9, 10 * uiScale)}px sans-serif`; this.ctx.textAlign = "right"; this.ctx.fillText("Drag tiles to rearrange  |  [ Done ] or Esc to finish", fw - 8, fh - 8);
    this.ctx.restore();
  }

  private updateTopbarLabels(): void {
    const startBtn = this.qs<HTMLButtonElement>("btn-start");
    if (startBtn) startBtn.textContent = this.timer.isRunning ? "Running" : "Start";
    const editBtn = this.qs<HTMLButtonElement>("btn-edit");
    if (editBtn) editBtn.textContent = this.layoutEditMode ? "Exit Edit" : "Edit Layout (E)";
  }

  private visionBusy = false;
  private lastVisionTs = 0;
  private visionIntervalMs = 66; // ~15 fps vision, render stays 60fps

  async tick(): Promise<void> {
    this.timer.updateTimer();
    // Vision: non-blocking, throttled by time + busy flag — never stalls render
    const now = performance.now();
    const canRunVision = this.timer.cameraEnabled && this.deps.video.readyState >= 2 && !!this.stream
      && !this.visionBusy && (now - this.lastVisionTs >= this.visionIntervalMs);
    if (canRunVision) {
      this.visionBusy = true;
      this.lastVisionTs = now;
      const c = this.visionEveryN;
      this.vision.analyze(this.deps.video, this.timer.isRunning, c).then((score) => {
        this.timer.focusScore = score;
        this.timer.focusState = this.vision.classify(score, this.timer.isRunning);
      }).catch(() => {}).finally(() => { this.visionBusy = false; });
    } else if (!this.timer.cameraEnabled) {
      this.timer.focusState = FocusState.Neutral; this.timer.focusScore = 0;
    }
    this.draw();
  }

  async run(): Promise<void> {
    this.handleResize();
    this.renderOnboarding();
    this.renderHtmlTiles();
    await this.startCamera().catch(() => {});
    // Visibility-aware loop — pause vision when tab hidden
    let hidden = document.hidden;
    document.addEventListener("visibilitychange", () => { hidden = document.hidden; });
    const loop = async (): Promise<void> => {
      if (!hidden) await this.tick();
      else this.draw(); // still draw timer even when hidden
      this.rafId = requestAnimationFrame(() => { void loop(); });
    };
    void loop();
  }

  dispose(): void {
    cancelAnimationFrame(this.rafId);
    this.stopCamera();
  }
}

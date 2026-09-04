// src/pomodoro.ts — PomodoroTimer state machine (port of Python class)
import { BREAK_MIN_DEFAULT, POMODORO_MIN_DEFAULT, SETTINGS_STORAGE_KEY, UI_SCALE_DEFAULT, UI_SCALE_MAX, UI_SCALE_MIN } from "./constants.ts";
import { CornerStyle, DisplayMode, FocusState, Phase, ThemeName } from "./types.ts";
import type { AppSettings, ThemeName as TName } from "./types.ts";
import { THEMES } from "./theme.ts";

export class PomodoroTimer {
  sessionMinutes: number;
  breakMinutes: number;
  displayMode: string;

  currentPhase: string = Phase.Pomodoro;
  timeLeft: number;
  isRunning = false;
  phaseStartedAt: number | null = null;
  phaseDurationSec: number;

  showProgressBar: boolean;
  showTimerPopup: boolean;

  focusScore = 0;
  focusState: string = FocusState.Neutral;

  // settings
  uiScale: number = UI_SCALE_DEFAULT;
  theme: TName = ThemeName.Dark;
  cornerStyle: string = CornerStyle.Rounded;
  alertsEnabled = true;
  cameraEnabled = true;
  progressBarEnabled = false;

  private lastAlertTs = 0;

  constructor(sessionMin = POMODORO_MIN_DEFAULT, breakMin = BREAK_MIN_DEFAULT, mode: string = DisplayMode.Both) {
    this.sessionMinutes = sessionMin;
    this.breakMinutes = breakMin;
    this.displayMode = mode;
    this.timeLeft = sessionMin * 60;
    this.phaseDurationSec = this.timeLeft;
    this.showProgressBar = mode === DisplayMode.ProgressBar || mode === DisplayMode.Both;
    this.showTimerPopup = mode === DisplayMode.TimerPopup || mode === DisplayMode.Both;
    this.loadSettings();
  }

  // persistence — mirrors _load_settings / _save_settings
  loadSettings(): void {
    try {
      const raw = localStorage.getItem(SETTINGS_STORAGE_KEY);
      if (!raw) return;
      const d = JSON.parse(raw) as Partial<AppSettings & { theme: string; cornerStyle: string; alertsEnabled: boolean }>;
      if (typeof d.uiScale === "number") this.uiScale = Math.max(UI_SCALE_MIN, Math.min(UI_SCALE_MAX, d.uiScale));
      if (d.theme && d.theme in THEMES) this.theme = d.theme as TName;
      if (d.cornerStyle === CornerStyle.Rounded || d.cornerStyle === CornerStyle.Boxy) this.cornerStyle = d.cornerStyle;
      if (typeof d.alertsEnabled === "boolean") this.alertsEnabled = d.alertsEnabled;
    } catch {}
  }
  saveSettings(): void {
    try {
      localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify({
        uiScale: Math.round(this.uiScale * 1000) / 1000, theme: this.theme, cornerStyle: this.cornerStyle, alertsEnabled: this.alertsEnabled,
      }));
    } catch {}
  }

  applySettings(patch: { pomodoroMinutes?: number; breakMinutes?: number; displayMode?: string }): void {
    if (patch.pomodoroMinutes !== undefined) this.sessionMinutes = Math.max(1, Math.floor(patch.pomodoroMinutes));
    if (patch.breakMinutes !== undefined) this.breakMinutes = Math.max(1, Math.floor(patch.breakMinutes));
    if (patch.displayMode !== undefined) this.displayMode = patch.displayMode;
    this.showProgressBar = this.displayMode === DisplayMode.ProgressBar || this.displayMode === DisplayMode.Both;
    this.showTimerPopup = this.displayMode === DisplayMode.TimerPopup || this.displayMode === DisplayMode.Both;
    if (!this.isRunning) {
      this.timeLeft = this.sessionMinutes * 60; this.phaseDurationSec = this.timeLeft; this.currentPhase = Phase.Pomodoro;
    } else {
      const elapsed = this.phaseStartedAt ? Math.floor((Date.now() / 1000) - this.phaseStartedAt) : 0;
      this.phaseDurationSec = this.currentPhase === Phase.ShortBreak ? this.breakMinutes * 60 : this.sessionMinutes * 60;
      this.timeLeft = Math.max(0, this.phaseDurationSec - elapsed);
    }
  }

  startTimer(): void {
    if (this.isRunning) return;
    this.isRunning = true;
    this.phaseStartedAt = Date.now() / 1000;
    this.phaseDurationSec = this.currentPhase === Phase.ShortBreak ? this.breakMinutes * 60 : this.sessionMinutes * 60;
    this.timeLeft = this.phaseDurationSec;
  }
  stopTimer(): void { this.isRunning = false; this.phaseStartedAt = null; }
  toggleTimer(): void { this.isRunning ? this.stopTimer() : this.startTimer(); }
  resetTimer(): void { this.isRunning = false; this.phaseStartedAt = null; this.currentPhase = Phase.Pomodoro; this.timeLeft = this.sessionMinutes * 60; this.phaseDurationSec = this.timeLeft; }

  switchToShortBreak(): void { this.currentPhase = Phase.ShortBreak; this.timeLeft = this.breakMinutes * 60; this.phaseDurationSec = this.timeLeft; this.phaseStartedAt = Date.now() / 1000; this.isRunning = true; }
  switchToPomodoro(): void { this.currentPhase = Phase.Pomodoro; this.timeLeft = this.sessionMinutes * 60; this.phaseDurationSec = this.timeLeft; this.phaseStartedAt = Date.now() / 1000; this.isRunning = true; }

  updateTimer(): void {
    if (!this.isRunning || this.phaseStartedAt === null) return;
    const elapsed = Math.floor(Date.now() / 1000 - this.phaseStartedAt);
    this.timeLeft = Math.max(0, this.phaseDurationSec - elapsed);
    if (this.timeLeft <= 0) {
      if (this.currentPhase === Phase.Pomodoro) this.switchToShortBreak();
      else if (this.currentPhase === Phase.ShortBreak) this.switchToPomodoro();
    }
  }

  get timerText(): string {
    const m = Math.floor(this.timeLeft / 60), s = this.timeLeft % 60;
    return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  }

  resolvePhase(): { text: string; color: [number, number, number]; progress: number } {
    if (this.currentPhase === Phase.Pomodoro) {
      return { text: "Pomodoro", color: [231, 76, 60], progress: Math.max(0, Math.min(1, 1 - this.timeLeft / Math.max(1, this.sessionMinutes * 60))) };
    }
    if (this.currentPhase === Phase.ShortBreak) {
      return { text: "Short Break", color: [39, 174, 96], progress: Math.max(0, Math.min(1, 1 - this.timeLeft / Math.max(1, this.breakMinutes * 60))) };
    }
    return { text: "Long Break", color: [52, 152, 219], progress: 0.5 };
  }

  private audioCtx: AudioContext | null = null;
  private audioUnlocked = false;

  private getAudioContext(): AudioContext | null {
    if (this.audioCtx && this.audioCtx.state !== "closed") return this.audioCtx;
    try {
      const Ctx = (window as unknown as { AudioContext?: typeof AudioContext; webkitAudioContext?: typeof AudioContext }).AudioContext
        ?? (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
      if (!Ctx) return null;
      this.audioCtx = new Ctx();
      return this.audioCtx;
    } catch { return null; }
  }

  /** Call on first user gesture to unlock AudioContext (autoplay policy). */
  unlockAudio(): void {
    if (this.audioUnlocked) return;
    const ctx = this.getAudioContext();
    if (!ctx) return;
    if (ctx.state === "suspended") ctx.resume().catch(() => {});
    // Play silent buffer to unlock
    try {
      const buf = ctx.createBuffer(1, 1, 22050);
      const src = ctx.createBufferSource();
      src.buffer = buf;
      src.connect(ctx.destination);
      src.start(0);
    } catch {}
    this.audioUnlocked = true;
  }

  shouldAlert(): boolean {
    if (!this.alertsEnabled || !this.isRunning || this.focusState !== FocusState.Slacking) return false;
    const now = Date.now() / 1000;
    if (now - this.lastAlertTs < 8) return false;
    this.lastAlertTs = now;
    return true;
  }

  playAlertSound(): void {
    if (!this.shouldAlert()) return;
    const ctx = this.getAudioContext();
    if (!ctx) return;
    // Resume if suspended (browser autoplay policy)
    if (ctx.state === "suspended") ctx.resume().catch(() => {});
    try {
      const mk = (freq: number, dur: number, delay: number): void => {
        const o = ctx.createOscillator(), g = ctx.createGain();
        o.type = "sine";
        o.frequency.value = freq; o.connect(g); g.connect(ctx.destination);
        g.gain.setValueAtTime(0.18, ctx.currentTime + delay);
        g.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + delay + dur);
        o.start(ctx.currentTime + delay); o.stop(ctx.currentTime + delay + dur);
      };
      mk(880, 0.18, 0); mk(660, 0.22, 0.2); mk(880, 0.12, 0.42);
    } catch {}
  }
}

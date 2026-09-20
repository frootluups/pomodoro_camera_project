// src/pomodoro.ts — PomodoroTimer state machine (port of Python class)
import { BREAK_MIN_DEFAULT, LONG_BREAK_MIN_DEFAULT, POMODOROS_BEFORE_LONG_BREAK, POMODORO_MIN_DEFAULT, SETTINGS_STORAGE_KEY, UI_SCALE_DEFAULT, UI_SCALE_MAX, UI_SCALE_MIN } from "./constants.ts";
import { CornerStyle, DisplayMode, FocusState, Phase, ThemeName } from "./types.ts";
import type { AppSettings, ThemeName as TName } from "./types.ts";
import { THEMES } from "./theme.ts";
import { getHistory } from "./history.ts";

export class PomodoroTimer {
  sessionMinutes: number;
  breakMinutes: number;
  longBreakMinutes: number;
  pomodorosBeforeLongBreak: number;
  displayMode: string;

  currentPhase: string = Phase.Pomodoro;
  timeLeft: number;
  isRunning = false;
  phaseStartedAt: number | null = null;
  phaseDurationSec: number;
  completedPomodoros = 0;

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
  autoStartBreak = false;
  autoStartPomodoro = false;
  notificationsEnabled = false;
  volume = 0.7;

  private lastAlertTs = 0;
  private phaseStartMs: number | null = null;
  private focusSum = 0;
  private focusCount = 0;

  constructor(sessionMin = POMODORO_MIN_DEFAULT, breakMin = BREAK_MIN_DEFAULT, mode: string = DisplayMode.Both) {
    this.sessionMinutes = sessionMin;
    this.breakMinutes = breakMin;
    this.longBreakMinutes = LONG_BREAK_MIN_DEFAULT;
    this.pomodorosBeforeLongBreak = POMODOROS_BEFORE_LONG_BREAK;
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
      if (typeof d.longBreakMinutes === "number") this.longBreakMinutes = Math.max(5, Math.min(60, Math.floor(d.longBreakMinutes)));
      if (typeof d.pomodorosBeforeLongBreak === "number") this.pomodorosBeforeLongBreak = Math.max(2, Math.min(8, Math.floor(d.pomodorosBeforeLongBreak)));
      if (typeof d.autoStartBreak === "boolean") this.autoStartBreak = d.autoStartBreak;
      if (typeof d.autoStartPomodoro === "boolean") this.autoStartPomodoro = d.autoStartPomodoro;
      if (typeof d.notificationsEnabled === "boolean") this.notificationsEnabled = d.notificationsEnabled;
      if (typeof d.volume === "number") this.volume = Math.max(0, Math.min(1, d.volume));
    } catch {}
  }
  saveSettings(): void {
    try {
      localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify({
        uiScale: Math.round(this.uiScale * 1000) / 1000, theme: this.theme, cornerStyle: this.cornerStyle, alertsEnabled: this.alertsEnabled,
        longBreakMinutes: this.longBreakMinutes, pomodorosBeforeLongBreak: this.pomodorosBeforeLongBreak,
        autoStartBreak: this.autoStartBreak, autoStartPomodoro: this.autoStartPomodoro,
        notificationsEnabled: this.notificationsEnabled, volume: this.volume,
      }));
    } catch {}
  }

  applySettings(patch: { pomodoroMinutes?: number; breakMinutes?: number; longBreakMinutes?: number; pomodorosBeforeLongBreak?: number; displayMode?: string }): void {
    if (patch.pomodoroMinutes !== undefined) this.sessionMinutes = Math.max(1, Math.floor(patch.pomodoroMinutes));
    if (patch.breakMinutes !== undefined) this.breakMinutes = Math.max(1, Math.floor(patch.breakMinutes));
    if (patch.longBreakMinutes !== undefined) this.longBreakMinutes = Math.max(5, Math.min(60, Math.floor(patch.longBreakMinutes)));
    if (patch.pomodorosBeforeLongBreak !== undefined) this.pomodorosBeforeLongBreak = Math.max(2, Math.min(8, Math.floor(patch.pomodorosBeforeLongBreak)));
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
    this.phaseStartMs = Date.now();
    this.focusSum = 0; this.focusCount = 0;
    this.phaseDurationSec = this.phaseDurationFor(this.currentPhase);
    this.timeLeft = this.phaseDurationSec;
    this.emitPhase();
  }
  stopTimer(): void { this.isRunning = false; this.phaseStartedAt = null; this.recordSession(false); }
  toggleTimer(): void { this.isRunning ? this.stopTimer() : this.startTimer(); }
  resetTimer(): void { this.recordSession(false); this.isRunning = false; this.phaseStartedAt = null; this.currentPhase = Phase.Pomodoro; this.timeLeft = this.sessionMinutes * 60; this.phaseDurationSec = this.timeLeft; this.emitPhase(); }

  private phaseDurationFor(phase: string): number {
    if (phase === Phase.ShortBreak) return this.breakMinutes * 60;
    if (phase === Phase.LongBreak) return this.longBreakMinutes * 60;
    return this.sessionMinutes * 60;
  }

  switchToShortBreak(): void { this.recordSession(true); this.currentPhase = Phase.ShortBreak; this.timeLeft = this.breakMinutes * 60; this.phaseDurationSec = this.timeLeft; this.phaseStartedAt = Date.now() / 1000; this.phaseStartMs = Date.now(); this.focusSum = 0; this.focusCount = 0; this.isRunning = this.autoStartBreak; if (!this.isRunning) this.phaseStartedAt = null; this.emitPhase(); this.notifyPhase(); }
  switchToLongBreak(): void { this.recordSession(true); this.currentPhase = Phase.LongBreak; this.timeLeft = this.longBreakMinutes * 60; this.phaseDurationSec = this.timeLeft; this.phaseStartedAt = Date.now() / 1000; this.phaseStartMs = Date.now(); this.focusSum = 0; this.focusCount = 0; this.isRunning = this.autoStartBreak; if (!this.isRunning) this.phaseStartedAt = null; this.emitPhase(); this.notifyPhase(); }
  switchToPomodoro(): void { this.recordSession(true); this.currentPhase = Phase.Pomodoro; this.timeLeft = this.sessionMinutes * 60; this.phaseDurationSec = this.timeLeft; this.phaseStartedAt = Date.now() / 1000; this.phaseStartMs = Date.now(); this.focusSum = 0; this.focusCount = 0; this.isRunning = this.autoStartPomodoro; if (!this.isRunning) this.phaseStartedAt = null; this.emitPhase(); this.notifyPhase(); }

  updateTimer(): void {
    if (!this.isRunning || this.phaseStartedAt === null) return;
    const elapsed = Math.floor(Date.now() / 1000 - this.phaseStartedAt);
    this.timeLeft = Math.max(0, this.phaseDurationSec - elapsed);
    if (this.timeLeft <= 0) {
      if (this.currentPhase === Phase.Pomodoro) {
        this.completedPomodoros++;
        if (this.completedPomodoros % this.pomodorosBeforeLongBreak === 0) this.switchToLongBreak();
        else this.switchToShortBreak();
      } else {
        this.switchToPomodoro();
      }
    }
  }

  // focus sampling for history
  sampleFocus(score: number): void {
    this.focusScore = score;
    this.focusSum += score; this.focusCount++;
  }

  private recordSession(completed: boolean): void {
    if (this.phaseStartMs === null) return;
    const endedAt = Date.now();
    const durationSec = Math.round((endedAt - this.phaseStartMs) / 1000);
    if (durationSec < 5) { this.phaseStartMs = null; return; }
    try {
      const rec: Parameters<ReturnType<typeof getHistory>["add"]>[0] = {
        phase: this.currentPhase as "pomodoro" | "short_break" | "long_break",
        startedAt: this.phaseStartMs, endedAt, durationSec, completed,
        ...(this.focusCount ? { focusAvg: Math.round(this.focusSum / this.focusCount) } : {}),
      };
      getHistory().add(rec);
    } catch {
      try {
        const key = "pomodoro.history.v1";
        const raw = localStorage.getItem(key);
        const arr = raw ? JSON.parse(raw) as unknown[] : [];
        (arr as unknown[]).push({ id: Math.random().toString(36).slice(2, 9), phase: this.currentPhase, startedAt: this.phaseStartMs, endedAt, durationSec, completed, focusAvg: this.focusCount ? Math.round(this.focusSum / this.focusCount) : undefined });
        localStorage.setItem(key, JSON.stringify((arr as unknown[]).slice(-500)));
      } catch {}
    }
    this.phaseStartMs = null;
  }

  private emitPhase(): void {
    try {
      window.dispatchEvent(new CustomEvent("pomodoro:phase", { detail: { phase: this.currentPhase, timeLeft: this.timeLeft, isRunning: this.isRunning } }));
      window.parent?.postMessage({ type: "pomodoro:phase", phase: this.currentPhase, timeLeft: this.timeLeft, isRunning: this.isRunning }, "*");
    } catch {}
  }

  private notifyPhase(): void {
    if (!this.notificationsEnabled) return;
    if (typeof Notification === "undefined" || Notification.permission !== "granted") return;
    const title = this.currentPhase === Phase.Pomodoro ? "Break over — back to focus!" : this.currentPhase === Phase.LongBreak ? "Long break — recharge!" : "Pomodoro complete — break time!";
    try { new Notification("Pomodoro Camera", { body: title, icon: "/icon-192.png", tag: "pomodoro" }); } catch {}
  }

  async requestNotificationPermission(): Promise<boolean> {
    if (typeof Notification === "undefined") return false;
    if (Notification.permission === "granted") { this.notificationsEnabled = true; this.saveSettings(); return true; }
    if (Notification.permission === "denied") return false;
    const perm = await Notification.requestPermission();
    this.notificationsEnabled = perm === "granted";
    this.saveSettings();
    return this.notificationsEnabled;
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
    return { text: "Long Break", color: [52, 152, 219], progress: Math.max(0, Math.min(1, 1 - this.timeLeft / Math.max(1, this.longBreakMinutes * 60))) };
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
    if (now - this.lastAlertTs < 4) return false;
    this.lastAlertTs = now;
    return true;
  }

  playAlertSound(): void {
    if (!this.shouldAlert()) return;
    const ctx = this.getAudioContext();
    if (!ctx) return;
    if (ctx.state === "suspended") ctx.resume().catch(() => {});
    try {
      const vol = this.volume;
      const mk = (freq: number, dur: number, delay: number): void => {
        const o = ctx.createOscillator(), g = ctx.createGain();
        o.type = "sine";
        o.frequency.value = freq; o.connect(g); g.connect(ctx.destination);
        g.gain.setValueAtTime(0.22 * vol, ctx.currentTime + delay);
        g.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + delay + dur);
        o.start(ctx.currentTime + delay); o.stop(ctx.currentTime + delay + dur);
      };
      mk(880, 0.18, 0); mk(660, 0.22, 0.2); mk(880, 0.12, 0.42); mk(1100, 0.15, 0.56);
    } catch {}
  }

  playTestSound(): void {
    const ctx = this.getAudioContext();
    if (!ctx) return;
    if (ctx.state === "suspended") ctx.resume().catch(() => {});
    try {
      const o = ctx.createOscillator(), g = ctx.createGain();
      o.type = "sine"; o.frequency.value = 880; o.connect(g); g.connect(ctx.destination);
      g.gain.setValueAtTime(0.18 * this.volume, ctx.currentTime);
      g.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.25);
      o.start(); o.stop(ctx.currentTime + 0.25);
    } catch {}
  }
}

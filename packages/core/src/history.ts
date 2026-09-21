// src/history.ts — session history, streaks, export, focus samples
export interface SessionRecord {
  id: string;
  phase: "pomodoro" | "short_break" | "long_break";
  startedAt: number; // epoch ms
  endedAt: number;
  durationSec: number;
  completed: boolean;
  focusAvg?: number;
  person?: string;
}

export interface FocusSample {
  t: number; // epoch ms
  score: number; // 0..100
}

const HISTORY_KEY = "pomodoro.history.v1";
const FOCUS_KEY = "pomodoro.focus.v1";
const STREAK_KEY = "pomodoro.streak.v1";
const MAX_RECORDS = 500;
const MAX_FOCUS_SAMPLES = 600; // 10 min at 1Hz

function uid(): string {
  return Math.random().toString(36).slice(2, 9) + Date.now().toString(36);
}

export class HistoryStore {
  records: SessionRecord[] = [];
  focusSamples: FocusSample[] = [];

  constructor() {
    this.load();
  }

  private load(): void {
    try {
      const raw = localStorage.getItem(HISTORY_KEY);
      if (raw) this.records = JSON.parse(raw) as SessionRecord[];
      const fraw = localStorage.getItem(FOCUS_KEY);
      if (fraw) this.focusSamples = JSON.parse(fraw) as FocusSample[];
    } catch {}
  }

  private save(): void {
    try {
      localStorage.setItem(HISTORY_KEY, JSON.stringify(this.records.slice(-MAX_RECORDS)));
      localStorage.setItem(FOCUS_KEY, JSON.stringify(this.focusSamples.slice(-MAX_FOCUS_SAMPLES)));
    } catch {}
  }

  add(record: Omit<SessionRecord, "id">): SessionRecord {
    const r: SessionRecord = { id: uid(), ...record };
    this.records.push(r);
    if (this.records.length > MAX_RECORDS) this.records = this.records.slice(-MAX_RECORDS);
    this.save();
    this.updateStreak(r);
    return r;
  }

  addFocusSample(score: number): void {
    this.focusSamples.push({ t: Date.now(), score });
    if (this.focusSamples.length > MAX_FOCUS_SAMPLES) this.focusSamples = this.focusSamples.slice(-MAX_FOCUS_SAMPLES);
    // throttle save to every 5 samples
    if (this.focusSamples.length % 5 === 0) this.save();
  }

  getToday(): SessionRecord[] {
    const start = new Date(); start.setHours(0, 0, 0, 0);
    const s = start.getTime();
    return this.records.filter((r) => r.startedAt >= s);
  }

  getStats(): { todayPomodoros: number; todayFocusAvg: number; totalPomodoros: number; streak: number } {
    const today = this.getToday().filter((r) => r.phase === "pomodoro" && r.completed);
    const total = this.records.filter((r) => r.phase === "pomodoro" && r.completed).length;
    const focusAvg = today.length ? today.reduce((a, r) => a + (r.focusAvg ?? 0), 0) / today.length : 0;
    return { todayPomodoros: today.length, todayFocusAvg: Math.round(focusAvg), totalPomodoros: total, streak: this.getStreak() };
  }

  getStreak(): number {
    try {
      const raw = localStorage.getItem(STREAK_KEY);
      if (!raw) return 0;
      const d = JSON.parse(raw) as { streak: number; lastDate: string };
      const today = new Date().toISOString().slice(0, 10);
      const yesterday = new Date(Date.now() - 86400000).toISOString().slice(0, 10);
      if (d.lastDate === today || d.lastDate === yesterday) return d.streak;
      return 0;
    } catch { return 0; }
  }

  private updateStreak(record: SessionRecord): void {
    if (record.phase !== "pomodoro" || !record.completed) return;
    const today = new Date().toISOString().slice(0, 10);
    try {
      const raw = localStorage.getItem(STREAK_KEY);
      let streak = 1;
      let lastDate = today;
      if (raw) {
        const d = JSON.parse(raw) as { streak: number; lastDate: string };
        const yesterday = new Date(Date.now() - 86400000).toISOString().slice(0, 10);
        if (d.lastDate === today) streak = d.streak;
        else if (d.lastDate === yesterday) streak = d.streak + 1;
        else streak = 1;
      }
      localStorage.setItem(STREAK_KEY, JSON.stringify({ streak, lastDate }));
    } catch {}
  }

  exportJSON(): string {
    return JSON.stringify({ records: this.records, focusSamples: this.focusSamples, exportedAt: new Date().toISOString() }, null, 2);
  }

  exportCSV(): string {
    const header = "id,phase,startedAt,endedAt,durationSec,completed,focusAvg,person";
    const rows = this.records.map((r) =>
      [r.id, r.phase, new Date(r.startedAt).toISOString(), new Date(r.endedAt).toISOString(), r.durationSec, r.completed, r.focusAvg ?? "", r.person ?? ""].join(",")
    );
    return [header, ...rows].join("\n");
  }

  clear(): void {
    this.records = []; this.focusSamples = [];
    try { localStorage.removeItem(HISTORY_KEY); localStorage.removeItem(FOCUS_KEY); } catch {}
  }

  // for chart: last N minutes of focus
  getRecentFocus(minutes = 10): FocusSample[] {
    const cutoff = Date.now() - minutes * 60 * 1000;
    return this.focusSamples.filter((s) => s.t >= cutoff);
  }
}

let GLOBAL_HISTORY: HistoryStore | null = null;
export function getHistory(): HistoryStore {
  GLOBAL_HISTORY ??= new HistoryStore();
  return GLOBAL_HISTORY;
}

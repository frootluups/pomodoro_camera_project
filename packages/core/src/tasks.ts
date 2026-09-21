// src/tasks.ts — local task list + active task, mirrors gallery.ts persistence patterns.
export const TASKS_STORAGE_KEY = "pomodoro.tasks.v1";
export const ACTIVE_TASK_KEY = "pomodoro.activeTask.v1";
export const TASK_TITLE_MAX = 60;

export interface Task {
  id: string;
  title: string;
  done: boolean;
  createdAt: number; // epoch ms
  completedAt?: number; // epoch ms
}

/** Sanitize a task title: trim, collapse whitespace, cap length. Empty → null (rejected). */
export function sanitizeTaskTitle(title: string): string | null {
  const clean = title.replace(/\s+/g, " ").trim().slice(0, TASK_TITLE_MAX);
  return clean ? clean : null;
}

function uid(): string {
  return Math.random().toString(36).slice(2, 9) + Date.now().toString(36);
}

interface TasksSnapshot {
  version: 1;
  tasks: Task[];
}

export class TaskStore {
  tasks = new Map<string, Task>();
  activeId: string | null = null;

  constructor() {
    this.load();
  }

  list(): Task[] {
    return [...this.tasks.values()].sort((a, b) => a.createdAt - b.createdAt);
  }

  get(id: string): Task | null {
    return this.tasks.get(id) ?? null;
  }

  getActive(): Task | null {
    if (!this.activeId) return null;
    return this.tasks.get(this.activeId) ?? null;
  }

  add(title: string): Task | null {
    const clean = sanitizeTaskTitle(title);
    if (!clean) return null;
    const t: Task = { id: uid(), title: clean, done: false, createdAt: Date.now() };
    this.tasks.set(t.id, t);
    // first task becomes active automatically
    if (!this.activeId) this.activeId = t.id;
    this.save();
    return t;
  }

  rename(id: string, title: string): string | null {
    const t = this.tasks.get(id);
    if (!t) return null;
    const clean = sanitizeTaskTitle(title);
    if (!clean) return null;
    t.title = clean;
    this.save();
    return t.title;
  }

  toggleDone(id: string): boolean | null {
    const t = this.tasks.get(id);
    if (!t) return null;
    t.done = !t.done;
    if (t.done) t.completedAt = Date.now();
    else delete t.completedAt;
    this.save();
    return t.done;
  }

  remove(id: string): boolean {
    const had = this.tasks.delete(id);
    if (this.activeId === id) {
      // fall back to first open task, else null
      const next = this.list().find((t) => !t.done) ?? this.list()[0] ?? null;
      this.activeId = next ? next.id : null;
    }
    if (had) this.save();
    return had;
  }

  clearCompleted(): number {
    let n = 0;
    for (const [id, t] of this.tasks) {
      if (t.done) {
        this.tasks.delete(id);
        n++;
      }
    }
    if (this.activeId && !this.tasks.has(this.activeId)) {
      const next = this.list().find((t) => !t.done) ?? this.list()[0] ?? null;
      this.activeId = next ? next.id : null;
    }
    if (n) this.save();
    return n;
  }

  setActive(id: string | null): void {
    if (id !== null && !this.tasks.has(id)) return;
    this.activeId = id;
    this.save();
  }

  /** Persist tasks + active id. Best-effort, never throws. */
  save(): void {
    try {
      const snap: TasksSnapshot = { version: 1, tasks: this.list() };
      localStorage.setItem(TASKS_STORAGE_KEY, JSON.stringify(snap));
      if (this.activeId) localStorage.setItem(ACTIVE_TASK_KEY, this.activeId);
      else localStorage.removeItem(ACTIVE_TASK_KEY);
    } catch {}
  }

  /** Load persisted tasks + active id. Corrupt entries are skipped, never throws. */
  load(): void {
    try {
      const raw = localStorage.getItem(TASKS_STORAGE_KEY);
      if (raw) {
        const snap = JSON.parse(raw) as Partial<TasksSnapshot>;
        if (snap.version === 1 && Array.isArray(snap.tasks)) {
          for (const t of snap.tasks) {
            if (!t || typeof t !== "object") continue;
            const rec = t as Partial<Task>;
            if (typeof rec.id !== "string" || !rec.id) continue;
            const clean = typeof rec.title === "string" ? sanitizeTaskTitle(rec.title) : null;
            if (!clean) continue;
            if (typeof rec.createdAt !== "number" || !Number.isFinite(rec.createdAt)) continue;
            this.tasks.set(rec.id, {
              id: rec.id,
              title: clean,
              done: rec.done === true,
              createdAt: rec.createdAt,
              ...(typeof rec.completedAt === "number" ? { completedAt: rec.completedAt } : {}),
            });
          }
        }
      }
      const aid = localStorage.getItem(ACTIVE_TASK_KEY);
      if (aid && this.tasks.has(aid)) this.activeId = aid;
      else if (aid) localStorage.removeItem(ACTIVE_TASK_KEY);
    } catch {}
  }
}

let GLOBAL_TASKS: TaskStore | null = null;
export function getTasks(): TaskStore {
  GLOBAL_TASKS ??= new TaskStore();
  return GLOBAL_TASKS;
}

/** Test hook — drop the cached singleton so a fresh (re-loaded) store can be observed. */
export function resetTasksForTest(): void {
  GLOBAL_TASKS = null;
}

// src/gallery.ts — lightweight browser re-ID (HSV histogram via canvas). Mirrors FaceGallery.
import type { BBox } from "./types.ts";

export const GALLERY_STORAGE_KEY = "pomodoro.gallery.v1";
export const GALLERY_NAME_MAX = 24;

function histEmbedding(imageData: ImageData): Float32Array | null {
  if (imageData.width < 8 || imageData.height < 8) return null;
  const bins = 16;
  const hist = new Float32Array(bins * 3);
  const d = imageData.data;
  // Sample every 2nd pixel (stride 8) — 2× fewer ops, ~same accuracy for 48×48
  for (let i = 0; i < d.length; i += 8) {
    const r = d[i]! / 255, g = d[i + 1]! / 255, b = d[i + 2]! / 255;
    const max = Math.max(r, g, b), min = Math.min(r, g, b);
    const delta = max - min;
    let h = 0;
    if (delta !== 0) {
      if (max === r) h = ((g - b) / delta) % 6;
      else if (max === g) h = (b - r) / delta + 2;
      else h = (r - g) / delta + 4;
      h *= 60; if (h < 0) h += 360;
    }
    const s = max === 0 ? 0 : delta / max;
    const v = max;
    const hb = Math.min(bins - 1, Math.floor((h / 360) * bins));
    const sb = Math.min(bins - 1, Math.floor(s * bins));
    const vb = Math.min(bins - 1, Math.floor(v * bins));
    hist[hb]! += 1;
    hist[bins + sb]! += 1;
    hist[bins * 2 + vb]! += 1;
  }
  // L2 normalize
  let sum = 0;
  for (let i = 0; i < hist.length; i++) sum += hist[i]! * hist[i]!;
  const n = Math.sqrt(sum) || 1;
  for (let i = 0; i < hist.length; i++) hist[i]! /= n;
  return hist;
}

function cosine(a: Float32Array, b: Float32Array): number {
  let dot = 0; for (let i = 0; i < a.length; i++) dot += a[i]! * b[i]!;
  return dot;
}

/** Sanitize a display name: trim, collapse whitespace, cap length. Empty → null (reset to default). */
export function sanitizeGalleryName(name: string): string | null {
  const clean = name.replace(/\s+/g, " ").trim().slice(0, GALLERY_NAME_MAX);
  return clean ? clean : null;
}

export interface GalleryIdentity {
  pid: number;
  label: string;
  name: string | null;
  hasEmbedding: boolean;
}

interface GallerySnapshot {
  version: 1;
  entries: Record<string, number[]>;
  names: Record<string, string>;
}

export class FaceGallery {
  thresh: number;
  maxGallery: number;
  entries = new Map<number, Float32Array>();
  names = new Map<number, string>();

  constructor(thresh = 0.62, maxGallery = 12) {
    this.thresh = thresh; this.maxGallery = maxGallery;
    this.load();
  }

  private embedCanvas: HTMLCanvasElement | null = null;
  private embedCtx: CanvasRenderingContext2D | null = null;

  private getEmbedCanvas(): CanvasRenderingContext2D | null {
    if (!this.embedCanvas) {
      this.embedCanvas = document.createElement("canvas");
      this.embedCanvas.width = 48; this.embedCanvas.height = 48;
      this.embedCtx = this.embedCanvas.getContext("2d", { willReadFrequently: true });
    }
    return this.embedCtx;
  }

  embedFromCanvas(src: HTMLCanvasElement | OffscreenCanvas, bbox: BBox): Float32Array | null {
    const [x, y, w, h] = bbox;
    if (w < 12 || h < 12) return null;
    const ctx = this.getEmbedCanvas();
    if (!ctx) return null;
    const sx = Math.max(0, x), sy = Math.max(0, y);
    const sw = Math.min(src.width - sx, w), sh = Math.min(src.height - sy, h);
    if (sw <= 0 || sh <= 0) return null;
    ctx.clearRect(0, 0, 48, 48);
    ctx.drawImage(src as unknown as CanvasImageSource, sx, sy, sw, sh, 0, 0, 48, 48);
    const data = ctx.getImageData(0, 0, 48, 48);
    return histEmbedding(data);
  }

  match(emb: Float32Array | null): number | null {
    if (!emb || this.entries.size === 0) return null;
    let best: number | null = null; let bestScore = -1;
    for (const [pid, ref] of this.entries) {
      if (ref.length !== emb.length) continue;
      const s = cosine(ref, emb);
      if (s > bestScore) { bestScore = s; best = pid; }
    }
    return best !== null && bestScore >= this.thresh ? best : null;
  }

  enroll(pid: number, emb: Float32Array | null, name?: string): void {
    if (!emb) return;
    if (this.entries.size >= this.maxGallery && !this.entries.has(pid)) {
      const oldest = Math.min(...this.entries.keys());
      this.entries.delete(oldest); this.names.delete(oldest);
    }
    this.entries.set(pid, emb);
    if (name) {
      const clean = sanitizeGalleryName(name);
      if (clean) this.names.set(pid, clean);
    }
    this.save();
  }

  /** Rename a known identity. Empty/blank resets to the default `Person #pid` label. */
  rename(pid: number, name: string): string {
    const clean = sanitizeGalleryName(name);
    if (clean) this.names.set(pid, clean);
    else this.names.delete(pid);
    this.save();
    return this.label(pid);
  }

  /** Forget one identity (embedding + name). Returns true if anything was removed. */
  remove(pid: number): boolean {
    const had = this.entries.delete(pid);
    this.names.delete(pid);
    if (had) this.save();
    return had;
  }

  /** Forget all identities and clear persisted storage. */
  clear(): void {
    this.entries.clear();
    this.names.clear();
    try { localStorage.removeItem(GALLERY_STORAGE_KEY); } catch {}
  }

  /** Sorted identities for settings UI. Includes named-but-unenrolled pids. */
  list(): GalleryIdentity[] {
    const pids = new Set<number>([...this.entries.keys(), ...this.names.keys()]);
    return [...pids].sort((a, b) => a - b).map((pid) => ({
      pid,
      label: this.label(pid),
      name: this.names.get(pid) ?? null,
      hasEmbedding: this.entries.has(pid),
    }));
  }

  label(pid: number): string { return this.names.get(pid) ?? `Person #${pid}`; }

  /** Persist embeddings (rounded) + names to localStorage. Best-effort, never throws. */
  save(): void {
    try {
      const entries: Record<string, number[]> = {};
      for (const [pid, emb] of this.entries) {
        entries[String(pid)] = Array.from(emb, (v) => Math.round(v * 10000) / 10000);
      }
      const names: Record<string, string> = {};
      for (const [pid, n] of this.names) names[String(pid)] = n;
      const snap: GallerySnapshot = { version: 1, entries, names };
      localStorage.setItem(GALLERY_STORAGE_KEY, JSON.stringify(snap));
    } catch {}
  }

  /** Load persisted embeddings + names. Corrupt entries are skipped, never throws. */
  load(): void {
    try {
      const raw = localStorage.getItem(GALLERY_STORAGE_KEY);
      if (!raw) return;
      const snap = JSON.parse(raw) as Partial<GallerySnapshot>;
      if (snap.version !== 1) return;
      if (snap.entries && typeof snap.entries === "object") {
        for (const [k, arr] of Object.entries(snap.entries)) {
          const pid = Number(k);
          if (!Number.isInteger(pid) || pid < 0 || !Array.isArray(arr)) continue;
          if (arr.length !== 48) continue;
          const nums = (arr as unknown[]).filter((v): v is number => typeof v === "number" && Number.isFinite(v));
          if (nums.length !== 48) continue;
          this.entries.set(pid, new Float32Array(nums));
        }
        while (this.entries.size > this.maxGallery) {
          const oldest = Math.min(...this.entries.keys());
          this.entries.delete(oldest);
        }
      }
      if (snap.names && typeof snap.names === "object") {
        for (const [k, n] of Object.entries(snap.names)) {
          const pid = Number(k);
          if (!Number.isInteger(pid) || pid < 0 || typeof n !== "string") continue;
          const clean = sanitizeGalleryName(n);
          if (clean) this.names.set(pid, clean);
        }
      }
    } catch {}
  }
}

let GLOBAL: FaceGallery | null = null;
export function getGallery(): FaceGallery {
  GLOBAL ??= new FaceGallery();
  return GLOBAL;
}

/** Test hook — drop the cached singleton so a fresh (re-loaded) gallery can be observed. */
export function resetGalleryForTest(): void {
  GLOBAL = null;
}

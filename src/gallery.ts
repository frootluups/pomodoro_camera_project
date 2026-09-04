// src/gallery.ts — lightweight browser re-ID (HSV histogram via canvas). Mirrors FaceGallery.
import type { BBox } from "./types.ts";

function histEmbedding(imageData: ImageData): Float32Array | null {
  if (imageData.width < 8 || imageData.height < 8) return null;
  // Build 3×16 HSV histogram (approx via RGB → HSV per pixel)
  const bins = 16;
  const hist = new Float32Array(bins * 3);
  const d = imageData.data;
  for (let i = 0; i < d.length; i += 4) {
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

export class FaceGallery {
  thresh: number;
  maxGallery: number;
  entries = new Map<number, Float32Array>();
  names = new Map<number, string>();

  constructor(thresh = 0.62, maxGallery = 12) {
    this.thresh = thresh; this.maxGallery = maxGallery;
  }

  embedFromCanvas(src: HTMLCanvasElement | OffscreenCanvas, bbox: BBox): Float32Array | null {
    const [x, y, w, h] = bbox;
    if (w < 12 || h < 12) return null;
    const c = document.createElement("canvas");
    c.width = 48; c.height = 48;
    const ctx = c.getContext("2d");
    if (!ctx) return null;
    const sx = Math.max(0, x), sy = Math.max(0, y);
    const sw = Math.min(src.width - sx, w), sh = Math.min(src.height - sy, h);
    if (sw <= 0 || sh <= 0) return null;
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
    if (name) this.names.set(pid, name);
  }

  label(pid: number): string { return this.names.get(pid) ?? `Person #${pid}`; }
}

let GLOBAL: FaceGallery | null = null;
export function getGallery(): FaceGallery {
  GLOBAL ??= new FaceGallery();
  return GLOBAL;
}

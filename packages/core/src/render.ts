// src/render.ts — Canvas 2D equivalents of styled_rect, gradients, xp chrome from main.py
import type { RGB } from "./types.ts";

const _cssCache = new Map<string, string>();
function cssRgb([r, g, b]: RGB, a = 1): string {
  const key = `${r},${g},${b},${a}`;
  let v = _cssCache.get(key);
  if (v) return v;
  v = a >= 1 ? `rgb(${r} ${g} ${b})` : `rgba(${r},${g},${b},${a})`;
  if (_cssCache.size < 256) _cssCache.set(key, v);
  return v;
}

export function roundedPath(ctx: CanvasRenderingContext2D, x1: number, y1: number, x2: number, y2: number, r: number): void {
  const rad = Math.max(0, Math.min(r, (x2 - x1) / 2, (y2 - y1) / 2));
  ctx.beginPath();
  if (rad <= 0) { ctx.rect(x1, y1, x2 - x1, y2 - y1); return; }
  ctx.moveTo(x1 + rad, y1);
  ctx.arcTo(x2, y1, x2, y2, rad);
  ctx.arcTo(x2, y2, x1, y2, rad);
  ctx.arcTo(x1, y2, x1, y1, rad);
  ctx.arcTo(x1, y1, x2, y1, rad);
  ctx.closePath();
}

export function styledRect(
  ctx: CanvasRenderingContext2D,
  x1: number, y1: number, x2: number, y2: number,
  opts: { fill?: RGB | null; border?: RGB | null; thickness?: number; radius?: number } = {},
): void {
  const r = opts.radius ?? 0;
  const thickness = opts.thickness ?? 1;
  if (x2 <= x1 || y2 <= y1) return;
  if (opts.fill) {
    ctx.save();
    roundedPath(ctx, x1, y1, x2, y2, r);
    ctx.fillStyle = cssRgb(opts.fill);
    ctx.fill();
    ctx.restore();
  }
  if (opts.border) {
    ctx.save();
    roundedPath(ctx, x1, y1, x2, y2, r);
    ctx.strokeStyle = cssRgb(opts.border);
    ctx.lineWidth = thickness;
    ctx.stroke();
    ctx.restore();
  }
}

export function hGradient(
  ctx: CanvasRenderingContext2D,
  x1: number, y1: number, x2: number, y2: number,
  c1: RGB, c2: RGB, radius = 0,
): void {
  if (x2 <= x1 || y2 <= y1) return;
  const g = ctx.createLinearGradient(x1, 0, x2, 0);
  g.addColorStop(0, cssRgb(c1));
  g.addColorStop(1, cssRgb(c2));
  ctx.save();
  roundedPath(ctx, x1, y1, x2, y2, radius);
  ctx.fillStyle = g;
  ctx.fill();
  ctx.restore();
}

export function vGradient(
  ctx: CanvasRenderingContext2D,
  x1: number, y1: number, x2: number, y2: number,
  c1: RGB, c2: RGB,
): void {
  if (x2 <= x1 || y2 <= y1) return;
  const g = ctx.createLinearGradient(0, y1, 0, y2);
  g.addColorStop(0, cssRgb(c1));
  g.addColorStop(1, cssRgb(c2));
  ctx.fillStyle = g;
  ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
}

export function drawXpTitleBar(
  ctx: CanvasRenderingContext2D,
  x1: number, y1: number, x2: number, y2: number,
  title: string,
): { x1: number; y1: number; x2: number; y2: number } | null {
  const h = y2 - y1;
  if (h < 4) return null;
  hGradient(ctx, x1, y1, x2, y2, [180, 100, 10], [250, 180, 60]);
  const mid = (x1 + x2) / 2, stripeW = (x2 - x1) / 3;
  hGradient(ctx, mid - stripeW / 2, y1 + 1, mid + stripeW / 2, y2 - 1, [255, 200, 80], [230, 160, 40]);
  ctx.fillStyle = cssRgb([120, 70, 0]);
  ctx.fillRect(x1, y2 - 1, x2 - x1, 1);
  ctx.fillStyle = "white";
  ctx.font = `700 ${Math.max(11, h * 0.5)}px 'Segoe UI', sans-serif`;
  ctx.textAlign = "left"; ctx.textBaseline = "middle";
  ctx.fillText(title, x1 + 8, y1 + h / 2);
  const cbSize = h * 0.72;
  const cbX2 = x2 - 4, cbX1 = cbX2 - cbSize, cbY1 = y1 + (h - cbSize) / 2, cbY2 = cbY1 + cbSize;
  vGradient(ctx, cbX1, cbY1, cbX2, cbY2, [80, 80, 230], [40, 40, 180]);
  styledRect(ctx, cbX1, cbY1, cbX2, cbY2, { border: [20, 20, 120], thickness: 1 });
  ctx.fillStyle = "white"; ctx.font = `700 ${cbSize * 0.5}px sans-serif`;
  ctx.textAlign = "center"; ctx.fillText("×", (cbX1 + cbX2) / 2, (cbY1 + cbY2) / 2 + 1);
  return { x1: cbX1, y1: cbY1, x2: cbX2, y2: cbY2 };
}

export function drawXpProgressBar(ctx: CanvasRenderingContext2D, x1: number, y1: number, x2: number, y2: number, progress: number): void {
  const w = x2 - x1, h = y2 - y1;
  if (w < 8 || h < 4) return;
  styledRect(ctx, x1, y1, x2, y2, { fill: [80, 80, 80], radius: 2 });
  ctx.fillStyle = "rgba(0,0,0,0.35)"; ctx.fillRect(x1, y1, w, 1);
  if (progress > 0) {
    const fillW = Math.max(4, w * Math.min(1, progress));
    vGradient(ctx, x1 + 2, y1 + 2, x1 + fillW - 1, y2 - 1, [110, 210, 80], [50, 160, 20]);
  }
}

// src/tracker.ts — port of MultiPersonTracker from main.py:152-268
import {
  TRACK_CONFIRM_HITS,
  TRACK_MAX_MISS,
  TRACK_MAX_PERSONS,
  TRACK_SMOOTH_ALPHA,
  TRACK_IOU_THRESH,
} from "./constants.ts";
import type { BBox } from "./types.ts";

export interface TrackedPerson {
  pid: number;
  bbox: BBox;
  smooth: readonly [number, number, number, number]; // x1,y1,x2,y2
  color: readonly [number, number, number];
  hits: number;
  misses: number;
  lastUpdate: number;
  gid: number | null;
  label: string;
}

const PALETTE: readonly [number, number, number][] = [
  [120, 160, 255],
  [140, 220, 140],
  [255, 180, 60],
  [220, 140, 255],
  [0, 200, 220],
  [255, 120, 120],
];

function iou(a: BBox, b: BBox): number {
  const [ax, ay, aw, ah] = a;
  const [bx, by, bw, bh] = b;
  const ax2 = ax + aw, ay2 = ay + ah, bx2 = bx + bw, by2 = by + bh;
  const ix1 = Math.max(ax, bx), iy1 = Math.max(ay, by);
  const ix2 = Math.min(ax2, bx2), iy2 = Math.min(ay2, by2);
  const iw = Math.max(0, ix2 - ix1), ih = Math.max(0, iy2 - iy1);
  const inter = iw * ih;
  if (inter === 0) return 0;
  const union = aw * ah + bw * bh - inter;
  return union ? inter / union : 0;
}

function center(b: BBox): [number, number] {
  return [b[0] + b[2] * 0.5, b[1] + b[3] * 0.5];
}

export class MultiPersonTracker {
  maxMiss: number;
  iouThresh: number;
  nextId = 1;
  tracks = new Map<number, TrackedPerson>();
  private frame = 0;

  constructor(maxMiss = TRACK_MAX_MISS, iouThresh = TRACK_IOU_THRESH) {
    this.maxMiss = maxMiss;
    this.iouThresh = iouThresh;
  }

  update(detections: BBox[], frameIdx: number): TrackedPerson[] {
    this.frame = frameIdx;
    const dets = [...detections].sort((a, b) => b[2] * b[3] - a[2] * a[3]).slice(0, TRACK_MAX_PERSONS);

    const usedDet = new Set<number>();
    const usedTrk = new Set<number>();
    const matches: [pid: number, detIdx: number, score: number][] = [];

    for (const pid of this.tracks.keys()) {
      const trk = this.tracks.get(pid)!;
      let bestI = -1;
      let bestS = -1;
      const [cx, cy] = center(trk.bbox);
      for (let i = 0; i < dets.length; i++) {
        if (usedDet.has(i)) continue;
        const d = dets[i]!;
        const iouScore = iou(trk.bbox, d);
        const dcx = d[0] + d[2] * 0.5, dcy = d[1] + d[3] * 0.5;
        const dist = Math.hypot(cx - dcx, cy - dcy);
        const size = Math.max(trk.bbox[2], trk.bbox[3], d[2], d[3], 1);
        const distScore = Math.max(0, 1 - dist / (size * 1.4));
        const score = Math.max(iouScore, distScore * 0.6);
        if (score > bestS) { bestS = score; bestI = i; }
      }
      if (bestI >= 0 && bestS >= this.iouThresh) {
        matches.push([pid, bestI, bestS]);
        usedDet.add(bestI);
        usedTrk.add(pid);
      }
    }
    matches.sort((a, b) => b[2] - a[2]);

    for (const [pid, di] of matches) {
      const d = dets[di]!;
      const trk = this.tracks.get(pid)!;
      const [ax, ay, aw, ah] = trk.bbox;
      const [bx, by, bw, bh] = d;
      const move = Math.hypot((ax + aw * 0.5) - (bx + bw * 0.5), (ay + ah * 0.5) - (by + bh * 0.5));
      const avgSz = Math.max(1, (aw + ah + bw + bh) * 0.25);
      const aBbox = move > avgSz * 0.12 ? 0.78 : TRACK_SMOOTH_ALPHA;
      const nx = Math.round(ax * (1 - aBbox) + bx * aBbox);
      const ny = Math.round(ay * (1 - aBbox) + by * aBbox);
      const nw = Math.round(aw * (1 - aBbox) + bw * aBbox);
      const nh = Math.round(ah * (1 - aBbox) + bh * aBbox);
      trk.bbox = [nx, ny, nw, nh] as BBox;
      const [x1, y1, x2, y2] = [nx, ny, nx + nw, ny + nh] as const;
      const [sx1, sy1, sx2, sy2] = trk.smooth;
      const move2 = Math.hypot((x1 + x2) * 0.5 - (sx1 + sx2) * 0.5, (y1 + y2) * 0.5 - (sy1 + sy2) * 0.5);
      const avgSz2 = Math.max(1, (nw + nh) * 0.5);
      const a2 = move2 > avgSz2 * 0.08 ? 0.82 : 0.62;
      trk.smooth = [
        Math.round(sx1 * (1 - a2) + x1 * a2),
        Math.round(sy1 * (1 - a2) + y1 * a2),
        Math.round(sx2 * (1 - a2) + x2 * a2),
        Math.round(sy2 * (1 - a2) + y2 * a2),
      ];
      trk.hits += 1;
      trk.misses = 0;
      trk.lastUpdate = frameIdx;
    }

    for (let i = 0; i < dets.length; i++) {
      if (usedDet.has(i)) continue;
      const d = dets[i]!;
      const pid = this.nextId++;
      const [x, y, w, h] = d;
      const color = PALETTE[(pid - 1) % PALETTE.length]!;
      this.tracks.set(pid, { pid, bbox: d, smooth: [x, y, x + w, y + h], color, hits: 1, misses: 0, lastUpdate: frameIdx, gid: null, label: "" });
    }

    const matchedPids = new Set(matches.map(([m]) => m));
    const toDel: number[] = [];
    for (const [pid, trk] of this.tracks) {
      if (usedTrk.has(pid) || matchedPids.has(pid)) continue;
      if (trk.lastUpdate !== frameIdx) {
        trk.misses += 1;
        if (trk.misses > this.maxMiss) toDel.push(pid);
      }
    }
    for (const pid of toDel) this.tracks.delete(pid);

    return [...this.tracks.values()].sort((a, b) => a.pid - b.pid);
  }

  get active(): TrackedPerson[] {
    return [...this.tracks.values()].filter((t) => t.hits >= TRACK_CONFIRM_HITS).sort((a, b) => a.pid - b.pid);
  }

  get allTracks(): TrackedPerson[] {
    return [...this.tracks.values()].sort((a, b) => a.pid - b.pid);
  }

  clear(): void {
    this.tracks.clear();
    this.nextId = 1;
  }
}

export function iouForTest(a: BBox, b: BBox): number { return iou(a, b); }

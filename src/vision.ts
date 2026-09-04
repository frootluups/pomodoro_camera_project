// src/vision.ts — face detection (FaceDetector native → BlazeFace) + motion + focus scoring
import { TRACK_MAX_PERSONS, TRACK_MIN_SIZE_FRAC, TRACK_MAX_SIZE_FRAC, TRACK_NMS_IOU, TRACK_NMS_DIST_FRAC, FOCUS_CONCENTRATED_THRESHOLD, FOCUS_SLACKING_THRESHOLD } from "./constants.ts";
import type { BBox, FocusState } from "./types.ts";
import { MultiPersonTracker } from "./tracker.ts";
import { FaceGallery, getGallery } from "./gallery.ts";

type FaceDetectorNative = { detect(input: CanvasImageSource): Promise<{ boundingBox: DOMRectReadOnly }[]> };

function getNativeDetector(): FaceDetectorNative | null {
  const Ctor = (globalThis as unknown as { FaceDetector?: new (opts?: unknown) => FaceDetectorNative }).FaceDetector;
  if (!Ctor) return null;
  try { return new Ctor({ fastMode: true, maxDetectedFaces: 4 }); } catch { return null; }
}

let blazeModel: { estimateFaces(img: HTMLVideoElement | HTMLCanvasElement, flipHorizontal: boolean): Promise<{ topLeft: [number, number]; bottomRight: [number, number] }[]> } | null = null;
async function getBlazeModel() {
  if (blazeModel) return blazeModel;
  try {
    const tf = await import("@tensorflow/tfjs");
    await tf.ready();
    const blazeface = await import("@tensorflow-models/blazeface");
    const m = await blazeface.load();
    blazeModel = m as unknown as typeof blazeModel;
    return blazeModel;
  } catch { return null; }
}

function nms(dets: BBox[]): BBox[] {
  if (dets.length <= 1) return dets.slice(0, TRACK_MAX_PERSONS);
  const sorted = [...dets].sort((a, b) => b[2] * b[3] - a[2] * a[3]);
  const kept: BBox[] = [];
  for (const d of sorted) {
    let dup = false;
    for (const k of kept) {
      const ix = Math.max(0, Math.min(d[0] + d[2], k[0] + k[2]) - Math.max(d[0], k[0]));
      const iy = Math.max(0, Math.min(d[1] + d[3], k[1] + k[3]) - Math.max(d[1], k[1]));
      const inter = ix * iy; const union = d[2] * d[3] + k[2] * k[3] - inter;
      const iou = union ? inter / union : 0;
      if (iou >= TRACK_NMS_IOU) { dup = true; break; }
      const cx = d[0] + d[2] * 0.5, cy = d[1] + d[3] * 0.5;
      const kx = k[0] + k[2] * 0.5, ky = k[1] + k[3] * 0.5;
      const dist = Math.hypot(cx - kx, cy - ky);
      const avg = Math.max(d[2], d[3], k[2], k[3]);
      if (dist < avg * TRACK_NMS_DIST_FRAC) { dup = true; break; }
    }
    if (!dup) kept.push(d);
  }
  return kept.slice(0, TRACK_MAX_PERSONS);
}

function filterGeometric(dets: BBox[], frameW: number, frameH: number): BBox[] {
  const area = frameW * frameH;
  return dets.filter(([x, y, w, h]) => {
    const f = (w * h) / Math.max(1, area);
    if (!(TRACK_MIN_SIZE_FRAC <= f && f <= TRACK_MAX_SIZE_FRAC)) return false;
    const ar = w / (h || 1);
    if (!(0.68 <= ar && ar <= 1.42)) return false;
    if (x < -w * 0.12 || y < -h * 0.12 || x + w > frameW + w * 0.12 || y + h > frameH + h * 0.12) return false;
    if (y < frameH * 0.08 && f > 0.055) return false;
    if (y < frameH * 0.14 && f > 0.095) return false;
    return true;
  });
}

export class VisionEngine {
  tracker = new MultiPersonTracker();
  gallery: FaceGallery = getGallery();
  private nativeDetector = getNativeDetector();
  private frameIdx = 0;
  private prevSmall: ImageData | null = null;
  private smoothScore: number | null = null;
  // expose for UI
  lastFaces: BBox[] = [];
  motion = 0;

  async detectFaces(video: HTMLVideoElement, width: number, height: number): Promise<BBox[]> {
    // Try native FaceDetector first (fast, no model download)
    if (this.nativeDetector) {
      try {
        // draw to small offscreen canvas for detector
        const c = document.createElement("canvas");
        c.width = Math.min(640, width); c.height = Math.min(480, height);
        const ctx = c.getContext("2d");
        if (ctx) {
          ctx.drawImage(video, 0, 0, c.width, c.height);
          const faces = await this.nativeDetector.detect(c);
          if (faces.length) {
            const sx = width / c.width, sy = height / c.height;
            return faces.map((f) => [f.boundingBox.x * sx, f.boundingBox.y * sy, f.boundingBox.width * sx, f.boundingBox.height * sy] as BBox);
          }
        }
      } catch {}
    }
    // BlazeFace fallback (lazy-loaded)
    const model = await getBlazeModel();
    if (model) {
      try {
        const preds = await model.estimateFaces(video, false);
        return preds.map((p) => {
          const x = p.topLeft[0], y = p.topLeft[1];
          return [x, y, p.bottomRight[0] - x, p.bottomRight[1] - y] as BBox;
        });
      } catch {}
    }
    return [];
  }

  /** Motion + face combine → 0..100, mirrors PomodoroTimer.analyze_focus */
  async analyze(video: HTMLVideoElement, isRunning: boolean, everyNFrames = 2): Promise<number> {
    const w = video.videoWidth || 640, h = video.videoHeight || 480;
    // Build small grayscale 160×120 for motion
    const small = document.createElement("canvas");
    small.width = 160; small.height = 120;
    const sctx = small.getContext("2d", { willReadFrequently: true });
    if (!sctx) return 0;
    sctx.drawImage(video, 0, 0, 160, 120);
    const cur = sctx.getImageData(0, 0, 160, 120);

    // Face detection cadence
    this.frameIdx++;
    const shouldDetect = this.frameIdx % everyNFrames === 0 || this.lastFaces.length === 0;
    if (shouldDetect) {
      let raw = await this.detectFaces(video, w, h);
      raw = filterGeometric(raw, w, h);
      raw = nms(raw);
      this.tracker.update(raw, this.frameIdx);
      const active = this.tracker.active;
      // gallery bookkeeping (HSV hist)
      const used = new Set(active.filter((t) => t.gid !== null).map((t) => t.gid as number));
      for (const trk of active) {
        const newly = trk.hits === 2 && trk.lastUpdate === this.frameIdx;
        if (newly) {
          // canvas for embedding
          const embCanvas = document.createElement("canvas");
          embCanvas.width = w; embCanvas.height = h;
          const ectx = embCanvas.getContext("2d");
          if (ectx) {
            ectx.drawImage(video, 0, 0, w, h);
            const emb = this.gallery.embedFromCanvas(ectx.canvas as unknown as HTMLCanvasElement, trk.bbox);
            let gid: number | null = emb ? this.gallery.match(emb) : null;
            if (gid !== null && used.has(gid)) gid = null;
            if (gid !== null) { trk.gid = gid; trk.label = this.gallery.label(gid); used.add(gid); }
            else { if (emb) this.gallery.enroll(trk.pid, emb); trk.gid = trk.pid; trk.label = this.gallery.label(trk.pid); used.add(trk.pid); }
          }
        } else if (!trk.label) {
          trk.label = this.gallery.label(trk.gid ?? trk.pid);
        }
      }
      this.lastFaces = this.tracker.active.map((t) => t.bbox);
    } else {
      this.lastFaces = this.tracker.active.map((t) => t.bbox);
    }

    if (!isRunning) { this.prevSmall = cur; this.smoothScore = null; return 0; }
    if (!this.prevSmall) { this.prevSmall = cur; return this.lastFaces.length ? 100 : 25; }

    // motion = mean absdiff / 255
    let sum = 0;
    const a = this.prevSmall.data, b = cur.data;
    for (let i = 0; i < a.length; i += 4) {
      // grayscale approx: 0.299R+0.587G+0.114B but data is RGBA of small canvas drawn from video (already color)
      const ga = (a[i]! * 0.299 + a[i + 1]! * 0.587 + a[i + 2]! * 0.114);
      const gb = (b[i]! * 0.299 + b[i + 1]! * 0.587 + b[i + 2]! * 0.114);
      sum += Math.abs(ga - gb);
    }
    const motion = sum / (cur.width * cur.height) / 255;
    this.motion = motion;
    this.prevSmall = cur;

    const faceScore = this.lastFaces.length ? 1 : 0;
    const motionPenalty = Math.min(1, motion * (faceScore ? 1.6 : 2.4));
    let score = faceScore * 72 + (1 - motionPenalty) * 28;
    if (!faceScore) score *= 0.55;
    score = Math.max(0, Math.min(100, score));
    const prev = this.smoothScore ?? score;
    score = prev * 0.35 + score * 0.65;
    this.smoothScore = score;
    return score;
  }

  classify(score: number, isRunning: boolean): FocusState {
    if (!isRunning) return "neutral";
    if (score >= FOCUS_CONCENTRATED_THRESHOLD) return "concentrated";
    if (score <= FOCUS_SLACKING_THRESHOLD) return "slacking";
    return "neutral";
  }

  reset(): void {
    this.tracker.clear(); this.lastFaces = []; this.prevSmall = null; this.smoothScore = null; this.frameIdx = 0;
  }
}

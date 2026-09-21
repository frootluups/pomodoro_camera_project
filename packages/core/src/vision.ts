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
let blazeLoading: Promise<typeof blazeModel> | null = null;
async function getBlazeModel() {
  if (blazeModel) return blazeModel;
  if (blazeLoading) return blazeLoading;
  blazeLoading = (async () => {
    try {
      const tf = await import("@tensorflow/tfjs");
      await tf.ready();
      const blazeface = await import("@tensorflow-models/blazeface");
      const m = await blazeface.load();
      blazeModel = m as unknown as typeof blazeModel;
      return blazeModel;
    } catch (err) {
      console.warn("PomodoroCamera: BlazeFace model failed to load — face detection disabled (check network access to tfhub.dev):", err);
      return null;
    } finally { blazeLoading = null; }
  })();
  return blazeLoading;
}
export function preloadBlazeModel(): void { void getBlazeModel(); }

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
  // Reused canvases — avoid per-frame allocation/GC pressure
  private motionCanvas: HTMLCanvasElement | null = null;
  private motionCtx: CanvasRenderingContext2D | null = null;
  private detectCanvas: HTMLCanvasElement | null = null;
  private detectCtx: CanvasRenderingContext2D | null = null;
  private embedCanvas: HTMLCanvasElement | null = null;
  private embedCtx: CanvasRenderingContext2D | null = null;
  private eyeCanvas: HTMLCanvasElement | null = null;
  private eyeCtx: CanvasRenderingContext2D | null = null;
  // expose for UI
  lastFaces: BBox[] = [];
  motion = 0;
  eyeVerified: boolean[] = [];
  eyeVerifiedCount = 0;
  liveFocusHistory: { t: number; score: number }[] = [];
  liveStats = { frames: 0, focused: 0, slacking: 0 };

  private getMotionCanvas(): { canvas: HTMLCanvasElement; ctx: CanvasRenderingContext2D } | null {
    if (!this.motionCanvas) {
      this.motionCanvas = document.createElement("canvas");
      this.motionCanvas.width = 160; this.motionCanvas.height = 120;
      this.motionCtx = this.motionCanvas.getContext("2d", { willReadFrequently: true });
    }
    return this.motionCtx && this.motionCanvas ? { canvas: this.motionCanvas, ctx: this.motionCtx } : null;
  }

  private getDetectCanvas(w: number, h: number): { canvas: HTMLCanvasElement; ctx: CanvasRenderingContext2D } | null {
    const cw = Math.min(640, w), ch = Math.min(480, h);
    if (!this.detectCanvas) {
      this.detectCanvas = document.createElement("canvas");
      this.detectCtx = this.detectCanvas.getContext("2d");
    }
    if (this.detectCanvas.width !== cw || this.detectCanvas.height !== ch) {
      this.detectCanvas.width = cw; this.detectCanvas.height = ch;
      this.detectCtx = this.detectCanvas.getContext("2d");
    }
    return this.detectCtx && this.detectCanvas ? { canvas: this.detectCanvas, ctx: this.detectCtx } : null;
  }

  private getEmbedCanvas(w: number, h: number): { canvas: HTMLCanvasElement; ctx: CanvasRenderingContext2D } | null {
    if (!this.embedCanvas) {
      this.embedCanvas = document.createElement("canvas");
      this.embedCtx = this.embedCanvas.getContext("2d");
    }
    if (this.embedCanvas.width !== w || this.embedCanvas.height !== h) {
      this.embedCanvas.width = w; this.embedCanvas.height = h;
      this.embedCtx = this.embedCanvas.getContext("2d");
    }
    return this.embedCtx && this.embedCanvas ? { canvas: this.embedCanvas, ctx: this.embedCtx } : null;
  }

  private getEyeCanvas(): { canvas: HTMLCanvasElement; ctx: CanvasRenderingContext2D } | null {
    if (!this.eyeCanvas) {
      this.eyeCanvas = document.createElement("canvas");
      this.eyeCanvas.width = 64; this.eyeCanvas.height = 32;
      this.eyeCtx = this.eyeCanvas.getContext("2d", { willReadFrequently: true });
    }
    return this.eyeCtx && this.eyeCanvas ? { canvas: this.eyeCanvas, ctx: this.eyeCtx } : null;
  }

  /** Heuristic eye verification: upper half of face should have dark eye-like regions */
  private verifyEyes(video: HTMLVideoElement, bbox: BBox, frameW: number, frameH: number): boolean {
    const [x, y, w, h] = bbox;
    if (w < 24 || h < 24) return false;
    const ec = this.getEyeCanvas();
    if (!ec) return false;
    // upper 55% of face
    const eyeH = h * 0.55;
    const sx = Math.max(0, x), sy = Math.max(0, y);
    const sw = Math.min(frameW - sx, w), sh = Math.min(frameH - sy, eyeH);
    if (sw < 12 || sh < 8) return false;
    // draw upper face region to 64x32 eye canvas
    ec.ctx.clearRect(0, 0, 64, 32);
    try {
      ec.ctx.drawImage(video, sx, sy, sw, sh, 0, 0, 64, 32);
    } catch { return false; }
    const data = ec.ctx.getImageData(0, 0, 64, 32).data;
    // heuristic: look for dark horizontal bands (eyes) in upper face
    // split into left/right eye regions, check for dark pixels
    let leftDark = 0, rightDark = 0, leftTotal = 0, rightTotal = 0;
    for (let py = 6; py < 26; py++) {
      for (let px = 4; px < 28; px++) {
        const idx = (py * 64 + px) * 4;
        const luma = (data[idx]! * 77 + data[idx + 1]! * 150 + data[idx + 2]! * 29) >> 8;
        leftTotal++;
        if (luma < 70) leftDark++;
      }
      for (let px = 36; px < 60; px++) {
        const idx = (py * 64 + px) * 4;
        const luma = (data[idx]! * 77 + data[idx + 1]! * 150 + data[idx + 2]! * 29) >> 8;
        rightTotal++;
        if (luma < 70) rightDark++;
      }
    }
    const leftRatio = leftDark / Math.max(1, leftTotal);
    const rightRatio = rightDark / Math.max(1, rightTotal);
    // at least one eye region should have some dark pixels (eyes/pupils)
    // but not too many (would be shadow/hair)
    const hasLeftEye = leftRatio > 0.04 && leftRatio < 0.45;
    const hasRightEye = rightRatio > 0.04 && rightRatio < 0.45;
    // also check overall contrast in eye band — eyes create local dark spots
    let minLuma = 255, maxLuma = 0;
    for (let py = 8; py < 24; py++) {
      for (let px = 8; px < 56; px++) {
        const idx = (py * 64 + px) * 4;
        const luma = (data[idx]! * 77 + data[idx + 1]! * 150 + data[idx + 2]! * 29) >> 8;
        if (luma < minLuma) minLuma = luma;
        if (luma > maxLuma) maxLuma = luma;
      }
    }
    const contrast = maxLuma - minLuma;
    // need reasonable contrast (face has light skin + dark eyes)
    if (contrast < 35) return false;
    return hasLeftEye || hasRightEye;
  }

  async detectFaces(video: HTMLVideoElement, width: number, height: number): Promise<BBox[]> {
    // Try native FaceDetector first (fast, no model download)
    if (this.nativeDetector) {
      try {
        const dc = this.getDetectCanvas(width, height);
        if (dc) {
          dc.ctx.drawImage(video, 0, 0, dc.canvas.width, dc.canvas.height);
          const faces = await this.nativeDetector.detect(dc.canvas);
          if (faces.length) {
            const sx = width / dc.canvas.width, sy = height / dc.canvas.height;
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

  /** Preload BlazeFace in background when camera enabled — avoids first-frame stall */
  preload(): void { if (!this.nativeDetector) void getBlazeModel(); }

  /** Motion + face combine → 0..100, mirrors PomodoroTimer.analyze_focus */
  async analyze(video: HTMLVideoElement, isRunning: boolean, everyNFrames = 2): Promise<number> {
    const w = video.videoWidth || 640, h = video.videoHeight || 480;
    // Reused 160×120 canvas for motion
    const mc = this.getMotionCanvas();
    if (!mc) return 0;
    mc.ctx.drawImage(video, 0, 0, 160, 120);
    const cur = mc.ctx.getImageData(0, 0, 160, 120);

    // Face detection cadence
    this.frameIdx++;
    const shouldDetect = this.frameIdx % everyNFrames === 0 || this.lastFaces.length === 0;
    if (shouldDetect) {
      let raw = await this.detectFaces(video, w, h);
      raw = filterGeometric(raw, w, h);
      // eye verification — filter suspicious detections, track verification flags
      const verified: boolean[] = [];
      const eyeFiltered: BBox[] = [];
      for (const det of raw) {
        const [dx, dy, dw, dh] = det;
        const areaFrac = (dw * dh) / Math.max(1, w * h);
        const isBrightSuspect = dy < h * 0.14 && areaFrac > 0.055;
        const isLargeTop = dy < h * 0.18 && areaFrac > 0.08;
        const eyeOk = this.verifyEyes(video, det, w, h);
        // bright/large-top detections require eye evidence
        if ((isBrightSuspect || isLargeTop) && !eyeOk) continue;
        eyeFiltered.push(det);
        verified.push(eyeOk);
      }
      raw = eyeFiltered;
      this.eyeVerified = verified;
      raw = nms(raw);
      // re-align verified flags after NMS (keep flags for kept boxes)
      // NMS keeps largest first, so map by bbox identity
      if (verified.length !== raw.length) {
        // NMS may have dropped some — rebuild verified for kept
        const keptVerified: boolean[] = [];
        for (const k of raw) {
          const idx = eyeFiltered.findIndex((d) => d[0] === k[0] && d[1] === k[1] && d[2] === k[2] && d[3] === k[3]);
          keptVerified.push(idx >= 0 ? verified[idx]! : true);
        }
        this.eyeVerified = keptVerified;
      }
      this.tracker.update(raw, this.frameIdx);
      const active = this.tracker.active;
      // gallery bookkeeping (HSV hist)
      const used = new Set(active.filter((t) => t.gid !== null).map((t) => t.gid as number));
      for (const trk of active) {
        const newly = trk.hits === 2 && trk.lastUpdate === this.frameIdx;
        if (newly) {
          const ec = this.getEmbedCanvas(w, h);
          if (ec) {
            ec.ctx.drawImage(video, 0, 0, w, h);
            const emb = this.gallery.embedFromCanvas(ec.canvas as unknown as HTMLCanvasElement, trk.bbox);
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

    // motion = mean absdiff / 255 — integer luma (77,150,29)>>8 ≈ 0.299,0.587,0.114, sample every 2nd pixel
    let sum = 0;
    let sampled = 0;
    const a = this.prevSmall.data, b = cur.data;
    for (let i = 0; i < a.length; i += 8) {
      const ga = (a[i]! * 77 + a[i + 1]! * 150 + a[i + 2]! * 29) >> 8;
      const gb = (b[i]! * 77 + b[i + 1]! * 150 + b[i + 2]! * 29) >> 8;
      sum += Math.abs(ga - gb);
      sampled++;
    }
    const motion = sum / Math.max(1, sampled) / 255;
    this.motion = motion;
    this.prevSmall = cur;

    // Face presence + eye verification is primary signal — motion is secondary
    let faceScore: number;
    if (!this.lastFaces.length) {
      faceScore = 0;
      this.eyeVerifiedCount = 0;
    } else if (this.eyeVerified.length === this.lastFaces.length) {
      const verifiedCount = this.eyeVerified.filter(Boolean).length;
      this.eyeVerifiedCount = verifiedCount;
      const eyeConf = 0.55 + 0.45 * (verifiedCount / Math.max(1, this.eyeVerified.length));
      faceScore = eyeConf;
    } else {
      faceScore = this.lastFaces.length ? 0.85 : 0;
      this.eyeVerifiedCount = this.lastFaces.length;
    }
    const motionPenalty = Math.min(1, motion * (faceScore > 0.5 ? 1.0 : 2.0));
    let score = faceScore * 80 + (1 - motionPenalty) * 20;
    if (!faceScore) score *= 0.45;
    else if (faceScore < 0.7) score *= 0.92;
    score = Math.max(0, Math.min(100, score));
    const prev = this.smoothScore ?? score;
    score = prev * 0.30 + score * 0.70;
    this.smoothScore = score;
    // live stats tracking
    const now = Date.now();
    this.liveFocusHistory.push({ t: now, score });
    const cutoff = now - 120_000;
    this.liveFocusHistory = this.liveFocusHistory.filter((s) => s.t >= cutoff);
    this.liveStats.frames++;
    if (score >= FOCUS_CONCENTRATED_THRESHOLD) this.liveStats.focused++;
    else if (score <= FOCUS_SLACKING_THRESHOLD) this.liveStats.slacking++;
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
    this.eyeVerified = []; this.eyeVerifiedCount = 0; this.liveFocusHistory = []; this.liveStats = { frames: 0, focused: 0, slacking: 0 };
  }
}

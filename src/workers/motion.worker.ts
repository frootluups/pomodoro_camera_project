// src/workers/motion.worker.ts — off-main-thread motion diff (160×120)
// Integer luma (77,150,29)>>8 ≈ 0.299,0.587,0.114, sampled every 2nd pixel
export type MotionRequest = { id: number; a: Uint8ClampedArray; b: Uint8ClampedArray };
export type MotionResponse = { id: number; motion: number };

self.onmessage = (e: MessageEvent<MotionRequest>) => {
  const { id, a, b } = e.data;
  let sum = 0;
  let sampled = 0;
  // a,b are RGBA arrays from ImageData (160*120*4 = 76800)
  for (let i = 0; i < a.length; i += 8) {
    const ga = (a[i]! * 77 + a[i + 1]! * 150 + a[i + 2]! * 29) >> 8;
    const gb = (b[i]! * 77 + b[i + 1]! * 150 + b[i + 2]! * 29) >> 8;
    sum += Math.abs(ga - gb);
    sampled++;
  }
  const motion = sum / Math.max(1, sampled) / 255;
  (self as unknown as Worker).postMessage({ id, motion } satisfies MotionResponse);
};

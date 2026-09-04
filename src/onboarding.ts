// src/onboarding.ts — 6-step intro, HTML-driven but keeps Python's data parity
export interface OnboardingStep {
  title: string;
  body: string;
  accent: [number, number, number];
}

export const ONBOARDING_STEPS: readonly OnboardingStep[] = [
  { title: "Welcome to Pomodoro Camera", body: "A focus timer that watches your webcam to keep you concentrated. Work in short bursts, take breaks, and let the camera help you stay on track.", accent: [0, 180, 255] },
  { title: "Pomodoro Timer", body: "25-minute work sessions followed by 5-minute breaks.\nPress the Start button or hit S to begin.", accent: [80, 200, 120] },
  { title: "Camera Focus Tracking", body: "Your webcam detects your face and monitors movement.\nStay visible and still to stay Concentrated.", accent: [0, 200, 220] },
  { title: "Slacking Alerts", body: "When the timer is running, looking away or moving too much triggers a red alert with a soft beep. Toggle in Settings → Alerts.", accent: [0, 100, 255] },
  { title: "Custom Layout", body: "Drag tiles to reposition any element. Press E or use Settings → Layout Edit Mode to rearrange.", accent: [220, 180, 40] },
  { title: "Keyboard Shortcuts", body: "  S  start / pause      E  layout edit\n  Q  quit (pause)        Esc  exit edit", accent: [180, 140, 255] },
] as const;

export class OnboardingManager {
  step = 0;
  active = true;
  get current(): OnboardingStep { return ONBOARDING_STEPS[this.step]!; }
  get isFirst(): boolean { return this.step === 0; }
  get isLast(): boolean { return this.step >= ONBOARDING_STEPS.length - 1; }
  next(): void { this.step++; if (this.step >= ONBOARDING_STEPS.length) this.active = false; }
  back(): void { if (this.step > 0) this.step--; }
  skip(): void { this.active = false; }
}

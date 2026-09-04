// src/types.ts — enums + shared types (TypeScript 7: strict branded literals)

export const Phase = {
  Pomodoro: "pomodoro",
  ShortBreak: "short_break",
  LongBreak: "long_break",
} as const;
export type Phase = (typeof Phase)[keyof typeof Phase];

export const ThemeName = {
  Dark: "dark",
  Light: "light",
  XP: "xp",
} as const;
export type ThemeName = (typeof ThemeName)[keyof typeof ThemeName];

export const FocusState = {
  Concentrated: "concentrated",
  Slacking: "slacking",
  Neutral: "neutral",
} as const;
export type FocusState = (typeof FocusState)[keyof typeof FocusState];

export const CornerStyle = {
  Rounded: "rounded",
  Boxy: "boxy",
} as const;
export type CornerStyle = (typeof CornerStyle)[keyof typeof CornerStyle];

export const DisplayMode = {
  ProgressBar: "progress_bar",
  TimerPopup: "timer_popup",
  Both: "both",
} as const;
export type DisplayMode = (typeof DisplayMode)[keyof typeof DisplayMode];

export type BBox = readonly [x: number, y: number, w: number, h: number];
export type Rect = readonly [x1: number, y1: number, x2: number, y2: number];
export type RGB = readonly [number, number, number];

export interface ThemePalette {
  readonly backdrop: RGB;
  readonly panelFill: RGB;
  readonly panelBorder: RGB;
  readonly divider: RGB;
  readonly text: RGB;
  readonly subtext: RGB;
  readonly buttonFill: RGB;
  readonly buttonBorder: RGB;
  readonly accent: RGB;
  readonly onColor: RGB;
  readonly offColor: RGB;
  readonly popupFill: RGB;
  readonly barTrack: RGB;
  readonly widgetFill: RGB;
}

export interface AppSettings {
  uiScale: number;
  theme: ThemeName;
  cornerStyle: CornerStyle;
  alertsEnabled: boolean;
}

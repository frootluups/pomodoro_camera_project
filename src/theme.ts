import type { ThemeName, ThemePalette } from "./types.ts";

export const THEMES: Record<ThemeName, ThemePalette> = {
  dark: {
    backdrop: [28, 28, 30],
    panelFill: [34, 34, 38],
    panelBorder: [168, 168, 175],
    divider: [92, 92, 98],
    text: [240, 240, 240],
    subtext: [185, 185, 190],
    buttonFill: [58, 58, 64],
    buttonBorder: [150, 150, 158],
    accent: [120, 160, 255],
    onColor: [140, 220, 140],
    offColor: [110, 110, 230],
    popupFill: [24, 24, 28],
    barTrack: [48, 48, 54],
    widgetFill: [45, 45, 48],
  },
  light: {
    backdrop: [236, 236, 240],
    panelFill: [250, 250, 252],
    panelBorder: [70, 70, 78],
    divider: [165, 165, 172],
    text: [34, 34, 40],
    subtext: [95, 95, 102],
    buttonFill: [228, 228, 233],
    buttonBorder: [115, 115, 122],
    accent: [30, 90, 220],
    onColor: [40, 140, 60],
    offColor: [40, 60, 190],
    popupFill: [255, 255, 255],
    barTrack: [210, 210, 216],
    widgetFill: [215, 215, 220],
  },
  xp: {
    backdrop: [212, 208, 200],
    panelFill: [236, 233, 216],
    panelBorder: [104, 104, 104],
    divider: [172, 168, 160],
    text: [16, 16, 16],
    subtext: [90, 80, 60],
    buttonFill: [226, 223, 210],
    buttonBorder: [104, 104, 104],
    accent: [200, 130, 30],
    onColor: [80, 170, 50],
    offColor: [60, 60, 180],
    popupFill: [236, 233, 216],
    barTrack: [160, 160, 160],
    widgetFill: [212, 208, 200],
  },
};

export function getTheme(name: ThemeName): ThemePalette {
  return THEMES[name] ?? THEMES.dark;
}

export function rgbToCss([r, g, b]: readonly [number, number, number], alpha = 1): string {
  return alpha >= 1 ? `rgb(${r} ${g} ${b})` : `rgb(${r} ${g} ${b} / ${alpha})`;
}

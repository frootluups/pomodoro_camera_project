// src/layout.ts — port of LayoutConfig / LayoutManager + persistence via localStorage
import { GRID_COLS_DEFAULT, GRID_ROWS_DEFAULT, LAYOUT_STORAGE_KEY } from "./constants.ts";

export interface UIElementConfig {
  enabled: boolean;
  x: number; y: number; width: number; height: number;
  anchor: string;
  margin: number; padding: number; fontScale: number;
  customProps: Record<string, unknown>;
}

export interface LayoutConfigData {
  gridCols: number; gridRows: number;
  elements: Record<string, UIElementConfig>;
}

function defaultElements(): Record<string, UIElementConfig> {
  return {
    timer_popup: { enabled: true, x: 8.6, y: 0.42, width: 2.9, height: 0.88, anchor: "top-left", margin: 0.18, padding: 0, fontScale: 0.92, customProps: {} },
    phase_label: { enabled: false, x: 4.6, y: 0.55, width: 2.9, height: 0.7, anchor: "top-left", margin: 0.12, padding: 0, fontScale: 1, customProps: {} },
    progress_bar: { enabled: true, x: 0.30, y: 0.10, width: 11.4, height: 0.12, anchor: "top-left", margin: 0.06, padding: 0, fontScale: 1, customProps: {} },
    focus_display: { enabled: true, x: 0.30, y: 0.38, width: 2.7, height: 0.40, anchor: "top-left", margin: 0.06, padding: 0, fontScale: 1, customProps: {} },
    status_display: { enabled: false, x: 0.4, y: 1.0, width: 3.2, height: 0.42, anchor: "top-left", margin: 0.08, padding: 0, fontScale: 1, customProps: {} },
    main_buttons: { enabled: true, x: 3.2, y: 7.12, width: 5.6, height: 0.68, anchor: "top-left", margin: 0.12, padding: 0, fontScale: 1, customProps: {} },
    quit_hint: { enabled: false, x: 10.6, y: 7.45, width: 1.2, height: 0.4, anchor: "top-left", margin: 0.1, padding: 0, fontScale: 1, customProps: {} },
  };
}

export class LayoutConfig {
  gridCols: number;
  gridRows: number;
  elements: Record<string, UIElementConfig>;

  constructor(gridCols = GRID_COLS_DEFAULT, gridRows = GRID_ROWS_DEFAULT, elements?: Record<string, UIElementConfig>) {
    this.gridCols = gridCols; this.gridRows = gridRows;
    this.elements = elements ?? defaultElements();
  }

  static default(): LayoutConfig { return new LayoutConfig(GRID_COLS_DEFAULT, GRID_ROWS_DEFAULT, defaultElements()); }

  toDict(): LayoutConfigData { return { gridCols: this.gridCols, gridRows: this.gridRows, elements: this.elements }; }

  static fromDict(d: LayoutConfigData): LayoutConfig {
    return new LayoutConfig(d.gridCols, d.gridRows, d.elements);
  }

  save(): void {
    try { localStorage.setItem(LAYOUT_STORAGE_KEY, JSON.stringify(this.toDict())); } catch {}
  }
  static load(): LayoutConfig | null {
    try {
      const raw = localStorage.getItem(LAYOUT_STORAGE_KEY);
      if (!raw) return null;
      const data = JSON.parse(raw) as LayoutConfigData;
      if (!data.elements || data.gridCols < 1 || data.gridRows < 1) return null;
      return LayoutConfig.fromDict(data);
    } catch { return null; }
  }
}

export class LayoutManager {
  config: LayoutConfig;
  gridCols: number;
  gridRows: number;

  constructor(config: LayoutConfig) {
    this.config = config;
    this.gridCols = config.gridCols;
    this.gridRows = config.gridRows;
  }

  setGrid(cols: number, rows: number): void {
    this.gridCols = cols; this.gridRows = rows;
    this.config.gridCols = cols; this.config.gridRows = rows;
  }

  getElementRect(name: string, frameW: number, frameH: number, includeDisabled = false): [number, number, number, number] {
    const elem = this.config.elements[name];
    if (!elem || (!elem.enabled && !includeDisabled)) return [0, 0, 0, 0];
    const cellW = frameW / this.gridCols;
    const cellH = frameH / this.gridRows;
    const marginPx = Math.max(0, Math.floor(elem.margin * Math.min(cellW, cellH)));
    const paddingPx = Math.max(0, Math.floor(elem.padding * Math.min(cellW, cellH)));
    const w = Math.min(Math.round(elem.width * cellW), frameW - 2 * marginPx);
    const h = Math.min(Math.round(elem.height * cellH), frameH - 2 * marginPx);
    const x = Math.max(marginPx, Math.min(Math.round(elem.x * cellW), frameW - w - marginPx));
    const y = Math.max(marginPx, Math.min(Math.round(elem.y * cellH), frameH - h - marginPx));
    return [x + paddingPx, y + paddingPx, x + w - paddingPx, y + h - paddingPx];
  }

  updateElement(name: string, patch: Partial<UIElementConfig>): void {
    this.config.elements[name] ??= { enabled: true, x: 0, y: 0, width: 1, height: 1, anchor: "top-left", margin: 0, padding: 0, fontScale: 1, customProps: {} };
    Object.assign(this.config.elements[name]!, patch);
  }

  enableElement(name: string, enabled = true): void {
    if (this.config.elements[name]) this.config.elements[name]!.enabled = enabled;
  }

  getEnabledElements(): string[] { return Object.entries(this.config.elements).filter(([, v]) => v.enabled).map(([k]) => k); }
}

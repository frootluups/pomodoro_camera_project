// src/constants.ts — mirrors main.py constants
export const WINDOW_NAME = "Pomodoro Camera" as const;

export const DEFAULT_FRAME_W = 640;
export const DEFAULT_FRAME_H = 480;
export const MIN_RENDER_W = 320;
export const MIN_RENDER_H = 240;

export const GRID_COLS_DEFAULT = 12;
export const GRID_ROWS_DEFAULT = 8;

export const UI_SCALE_DEFAULT = 0.82;
export const UI_SCALE_MIN = 0.6;
export const UI_SCALE_MAX = 1.4;
export const UI_SCALE_STEP = 0.08;

export const POMODORO_MIN_DEFAULT = 25;
export const BREAK_MIN_DEFAULT = 5;
export const LONG_BREAK_MIN_DEFAULT = 15;
export const POMODOROS_BEFORE_LONG_BREAK = 4;

export const AUTO_START_POMODORO_DEFAULT = false;
export const AUTO_START_BREAK_DEFAULT = false;

export const FACE_DETECT_EVERY_N_FRAMES = 2;
export const FOCUS_CONCENTRATED_THRESHOLD = 70;
export const FOCUS_SLACKING_THRESHOLD = 35;

export const TRACK_MAX_MISS = 9;
export const TRACK_IOU_THRESH = 0.22;
export const TRACK_MAX_PERSONS = 2;
export const TRACK_SMOOTH_ALPHA = 0.68;
export const TRACK_CONFIRM_HITS = 2;
export const TRACK_MIN_SIZE_FRAC = 0.02;
export const TRACK_MAX_SIZE_FRAC = 0.28;
export const TRACK_NMS_IOU = 0.22;
export const TRACK_NMS_DIST_FRAC = 0.38;

export const LAYOUT_STORAGE_KEY = "pomodoro.layout.v1";
export const SETTINGS_STORAGE_KEY = "pomodoro.settings.v1";
export const HISTORY_STORAGE_KEY = "pomodoro.history.v1";
export const FOCUS_STORAGE_KEY = "pomodoro.focus.v1";

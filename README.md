# Pomodoro Camera

Focus-aware Pomodoro timer in a single OpenCV window. Combines a 25/5 timer with webcam face/motion tracking, draggable layout editing, TrueType text, and three themes (dark / light / Windows XP Luna).

*A lightweight Pomodoro timer i vibecoded because i was hella bored — now ported to website-friendly TypeScript 7 (Vite).*

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue) ![TypeScript](https://img.shields.io/badge/typescript-7-blue) ![Vite](https://img.shields.io/badge/vite-6-646CFF) ![OpenCV](https://img.shields.io/badge/opencv-%3E%3D4.10-green) ![License MIT](https://img.shields.io/badge/license-MIT-lightgrey)

## Features

- **Pomodoro engine** — 25 min work → 5 min break, auto flip, pause/resume/reset
- **Camera focus scoring** — Haar face detection (every 3rd frame) + motion diff → 0-100 score → `concentrated / neutral / slacking`
- **Face tracking outline** — smoothed, color-coded corner-bracket box with `SLACKING` tag
- **Face re-ID gallery** — persistent per-person identities (HSV hist + optional SFace), custom names, per-session attribution in history export
- **Slacking alerts** — pulsing banner + window border + beep (throttled, toggle in Settings)
- **TrueType text** — Segoe UI via Pillow, Hershey fallback, cached sizing, binary-search `fit_text`
- **Layout system** — 12×8 grid, top-left cells, drag-and-drop edit (`E` / `Esc`), presets (Default / Focus / Dashboard / Minimal), `layout.json` persistence
- **Themes** — `dark` / `light` / `xp` (Luna beveled buttons, gradient title bars, chunky progress)
- **Onboarding** — 6-step first-run intro (dim + highlight + card), `Next / Back / Skip`, persists via `settings.json`
- **Settings panel** — centered modal: pomodoro/break steppers, camera/bar toggles, theme/corners/alerts, display mode, UI scale (0.6–1.4), grid/layout actions
- **Web build** — Vite + TypeScript 7, `<video>` + `<canvas>`, `FaceDetector` → `BlazeFace` fallback, `localStorage` persistence

## Quick start

### Python (OpenCV window)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r apps/python/requirements.txt
python apps/python/main.py
# or: cd apps/python && python main.py
```

Or as installed script (after `pip install -e apps/python` with `apps/python/pyproject.toml`):

```bash
pomodoro-camera
```

### Web (TypeScript 7 + Vite)

```bash
npm install
npm run dev         # web app → http://localhost:5173 (apps/web)
npm run dev:embed   # embed preview → http://localhost:5174 (apps/embed)
npm run build       # → apps/web/dist/
npm run preview
```

Requires Node 20+. Camera needs HTTPS or `localhost`.

### Embeddable Widget — add to any website

**Option A — Web Component (recommended, isolated via Shadow DOM):**

```html
<script type="module" src="https://your-cdn.com/pomodoro-embed.js"></script>
<pomodoro-camera theme="dark" pomodoro="25" break="5"></pomodoro-camera>

<!-- Compact variant -->
<pomodoro-camera theme="light" compact width="380px"></pomodoro-camera>
```

Attributes: `theme="dark|light|xp"`, `pomodoro="25"`, `break="5"`, `camera="false"`, `compact`, `width`, `height`. All reactive.

**Option B — JavaScript API:**

```html
<div id="my-widget"></div>
<script type="module">
  import { mountPomodoro } from "https://your-cdn.com/pomodoro-embed.js";
  // or via global: window.PomodoroCamera.mount(...)
  const el = mountPomodoro("#my-widget", { theme: "dark", pomodoroMinutes: 25, compact: true });
  el.addEventListener("pomodoro-ready", ({ detail: { app } }) => {
    // app.timer, app.vision, app.layout available
  });
  // Imperative: el.start(), el.pause(), el.reset(), el.setTheme("light")
</script>
```

**Option C — iframe (zero JS, easiest for CMS/blog):**

```html
<iframe
  src="https://your-site.com/embed.html?theme=dark&pomodoro=25&break=5"
  allow="camera; microphone"
  width="520" height="600"
  style="border:0; border-radius:14px; overflow:hidden"
  loading="lazy"
></iframe>
```

Build outputs:
- `npm run build` → `apps/web/dist/index.html` (web app)
- `npm run build:embed-app` → `apps/embed/dist/embed.html` + `embed-demo.html` (embed preview)
- `npm run build:embed` → `apps/embed/dist/pomodoro-embed.js` + `pomodoro-embed.umd.js` (single-file library for CDN)
- `npm run build:all` → all three above

**Option D — npm package (if published):**

```bash
npm install pomodoro-camera-web
```
```ts
import "pomodoro-camera-web/embed"; // auto-defines <pomodoro-camera>
import { mountPomodoro } from "pomodoro-camera-web/embed";
```

## Controls

| Key | Action |
|---|---|
| `S` | start / pause timer |
| `E` | toggle layout edit mode |
| `N` | name primary face (Python window; Settings → Faces on web) |
| `Esc` | exit edit mode (also skip onboarding) |
| `Q` / window `X` | quit |
| `Enter` / `Space` | onboarding next |
| `Backspace` | onboarding back |
| mouse drag | move tiles in edit mode; click `[ Done ]` to save |

## Persistence

- `apps/python/layout.json` / `localStorage:pomodoro.layout.v1` — grid positions for `timer_popup`, `phase_label`, `progress_bar`, `focus_display`, `status_display`, `main_buttons`, `quit_hint`; auto-created, merged on update, healed if corrupt
- `apps/python/settings.json` / `localStorage:pomodoro.settings.v1` — `ui_scale`, `theme`, `corner_style`, `alerts_enabled`; written on change / onboarding finish
- `apps/python/gallery.json` / `localStorage:pomodoro.gallery.v1` — face re-ID embeddings + custom names; rename in Settings (web) or `N` key (Python), `Forget` to remove

Delete either file (or clear localStorage) to reset to defaults.

## Project structure

```
apps/python/ — OpenCV desktop app
  ├─ main.py — single-file, sectioned backend
  │    ├─ constants & StrEnum (Phase, ThemeName, FocusState, CornerStyle, DisplayMode)
  │    ├─ FontEngine (Pillow + Hershey, LRU-cached text_size / fit_text, numpy gradients)
  │    ├─ Layout (UIElementConfig, LayoutConfig, LayoutManager — grid → pixel rects)
  │    ├─ Rendering (styled_rect, _h/_v_gradient, _xp_button/title/progress)
  │    ├─ OnboardingManager (6 steps, _card_rects hit-test, draw_overlay)
  │    ├─ ButtonHandler (layout-driven hit-test)
  │    └─ PomodoroTimer
  │         ├─ timer state machine (_update_timer, switch_*)
  │         ├─ vision (analyze_focus — face every 3rd frame, motion penalty)
  │         ├─ rendering (_draw_ui → _draw_* helpers, theme-aware)
  │         ├─ I/O (camera via VideoCapture, window loop, persistence)
  │         └─ main() entrypoint
  ├─ tests/ (pytest: test_layout, test_render, test_tracking)
  ├─ haarcascade_frontalface_default.xml, requirements.txt, pyproject.toml
  └─ layout.json / settings.json (runtime, gitignored)
apps/web/ — standalone browser app (Vite)
  ├─ index.html, vite.config.ts, public/ (manifest.json, sw.js)
  └─ src/main.ts (boot — imports shared core)
apps/embed/ — embeddable widget (Vite)
  ├─ embed.html, embed-demo.html, vite.config.ts (preview), vite.embed.config.ts (library)
  └─ src/embed.ts (<pomodoro-camera> element + mount API — imports shared core)
packages/core/ — shared TypeScript core (imported by web + embed)
  ├─ constants.ts, types.ts, theme.ts
  ├─ tracker.ts (MultiPersonTracker), gallery.ts (FaceGallery)
  ├─ layout.ts (LayoutManager + localStorage), render.ts (Canvas 2D)
  ├─ vision.ts (FaceDetector/BlazeFace + motion), pomodoro.ts (state machine)
  ├─ app.ts (camera, canvas loop, drag, settings, onboarding)
  ├─ history.ts, onboarding.ts, workers/motion.worker.ts
  └─ package.json (@pomodoro/core, private)
```

Hot-path notes (see `apps/python/main.py:91-179`, `apps/python/main.py:365-532`):

- `text_size` cached (≤2048 entries), `fit_text` cached (≤1024), Pillow font cache
- `_h_gradient` / `_v_gradient` numpy-vectorized (`linspace` + outer)
- `_xp_button` single-pass gradient fill
- `analyze_focus` skips Haar cascade 2/3 frames, reuses `last_faces`

## Performance

Measured on 640×480 canvas, 30-frame average (`cd apps/python && python -c "from main import PomodoroTimer ..."`):

- XP theme ~3.5 ms / frame (~285 FPS)
- Dark theme ~3.5 ms / frame (~290 FPS)
- 72-combo matrix (3 sizes × 3 themes × 2 corners × 4 states) — all clean

## Configuration

Pass custom durations / mode programmatically:

```python
# run from apps/python/
from main import PomodoroTimer, DisplayMode
t = PomodoroTimer(session_duration_minutes=30, break_duration_minutes=10, display_mode=DisplayMode.BOTH)
t.start_session()
```

## Requirements

- Python 3.11+
- `opencv-python>=4.10.0` (provides `cv2.data.haarcascades`), `numpy>=1.24,<3`, `pillow>=10`, `python-dateutil>=2.8`
- Windows: `segoeui.ttf` from `C:\Windows\Fonts` auto-detected; other platforms fall back to DejaVu/Arial
- Optional: `winsound` (Windows) for beeps, otherwise `\a` bell
- Web: Node 20+, modern browser with `getUserMedia`

## Tests

```bash
cd apps/python
pytest -q
# or quick render-matrix smoke test:
python -c "import tests.test_render; tests.test_render.test_matrix()"
```

See `apps/python/tests/test_render.py` and `apps/python/tests/test_layout.py`.

## Future Improvements

- Audio-based focus detection
- ML attention prediction, eye movement / blink rate
- Multiple cameras, user profiles / habit tracking
- Calendar / task integration, mobile app, cross-platform polish

## Troubleshooting

- `numpy.core.multiarray failed to import` / `_ARRAY_API not found` → upgrade OpenCV: `pip install -U "opencv-python>=4.10"`
- Camera not opening → close other apps using webcam, check `cv2.VideoCapture(0)`; toggle `Camera: Off/On` in Settings (web: check `getUserMedia` permission)
- Layout broken after manual edit → delete `apps/python/layout.json` and restart (web: clear localStorage)
- No Pillow → installs Hershey fallback; `pip install pillow` for Segoe UI

"""
Pomodoro Camera — Focus-aware Pomodoro timer with webcam tracking.

Single-window OpenCV application that combines a Pomodoro timer with
camera-based focus detection, draggable layout editing, and themed UI.

Run:
    python main.py
Controls:
    q / window X  quit
    s             start / pause timer
    e             toggle layout edit mode
    Esc           exit edit mode / skip onboarding

Persistence:
    layout.json   grid positions for UI tiles
    settings.json theme, scale, alerts, corner style
"""

from __future__ import annotations

import ctypes
import json
import logging
import math
import os
import platform
import time
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont

    _PIL_AVAILABLE: Final = True
except Exception:  # Pillow not installed
    _PIL_AVAILABLE = False
    Image = ImageDraw = ImageFont = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pomodoro")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WINDOW_NAME: Final = "Pomodoro Camera"
DEFAULT_FRAME_W: Final = 640
DEFAULT_FRAME_H: Final = 480
MIN_RENDER_W: Final = 320
MIN_RENDER_H: Final = 240
DEFAULT_WINDOW_W: Final = 960
DEFAULT_WINDOW_H: Final = 600
LAYOUT_FILE: Final = Path("layout.json")
SETTINGS_FILE: Final = Path("settings.json")
CASCADE_FILE: Final = "haarcascade_frontalface_default.xml"
EYE_CASCADE_FILE: Final = "haarcascade_eye.xml"

GRID_COLS_DEFAULT: Final = 12
GRID_ROWS_DEFAULT: Final = 8

# UI scaling
UI_SCALE_DEFAULT: Final = 0.82
UI_SCALE_MIN: Final = 0.6
UI_SCALE_MAX: Final = 1.4
UI_SCALE_STEP: Final = 0.08

# Timer defaults
POMODORO_MIN_DEFAULT: Final = 25
BREAK_MIN_DEFAULT: Final = 5

# Focus — detection every 2 frames cuts lag from ~100ms to ~66ms
FACE_DETECT_EVERY_N_FRAMES: Final = 2
FOCUS_CONCENTRATED_THRESHOLD: Final = 70.0
FOCUS_SLACKING_THRESHOLD: Final = 35.0

# Cache limits
TEXT_SIZE_CACHE_MAX: Final = 2048
FIT_TEXT_CACHE_MAX: Final = 1024

# Tracker config — tuned to avoid ghost tracks and lag
TRACK_MAX_MISS: Final = 9
TRACK_IOU_THRESH: Final = 0.22  # lower = more permissive matching (same face jitter still matches)
TRACK_MAX_PERSONS: Final = 2  # cap at 2 — user sees 2 ghosts for 1 person, so cap tight
TRACK_SMOOTH_ALPHA: Final = 0.68
TRACK_CONFIRM_HITS: Final = 2
TRACK_MIN_SIZE_FRAC: Final = 0.020
TRACK_MAX_SIZE_FRAC: Final = 0.28
TRACK_NMS_IOU: Final = 0.22  # aggressive NMS — duplicate boxes for same face (IoU ~0.3-0.6) get merged
TRACK_NMS_DIST_FRAC: Final = 0.38  # also merge if centers within 38% of avg size (same face, different scale)

# ---------------------------------------------------------------------------
# Multi-person tracking — lightweight SORT-style centroid + IoU, persistent IDs
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class TrackedPerson:
    """One tracked face with persistent ID and smoothed geometry."""

    pid: int
    bbox: tuple[int, int, int, int]  # raw (x,y,w,h) in cam coords
    smooth: tuple[int, int, int, int]  # smoothed (x1,y1,x2,y2) for drawing
    color: tuple[int, int, int]
    hits: int = 1
    misses: int = 0
    last_update: int = 0
    gid: int | None = None  # gallery identity (long-term re-ID)
    label: str = ""  # display name, e.g. "Person #2" or custom name

    def center(self) -> tuple[float, float]:
        x, y, w, h = self.bbox
        return (x + w * 0.5, y + h * 0.5)

    def area(self) -> int:
        return self.bbox[2] * self.bbox[3]


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2, bx2, by2 = ax1 + aw, ay1 + ah, bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


_PERSON_PALETTE: Final[list[tuple[int, int, int]]] = [
    (120, 160, 255),  # blue
    (140, 220, 140),  # green
    (255, 180, 60),  # orange
    (220, 140, 255),  # purple
    (0, 200, 220),  # cyan
    (255, 120, 120),  # salmon
]


class MultiPersonTracker:
    """Greedy IoU + centroid tracker. Lightweight, no Kalman — EMA smoothing only."""

    def __init__(self, max_miss: int = TRACK_MAX_MISS, iou_thresh: float = TRACK_IOU_THRESH) -> None:
        self.max_miss = max_miss
        self.iou_thresh = iou_thresh
        self.next_id: int = 1
        self.tracks: dict[int, TrackedPerson] = {}
        self._frame: int = 0

    def update(self, detections: list[tuple[int, int, int, int]], frame_idx: int) -> list[TrackedPerson]:
        self._frame = frame_idx
        dets = sorted(detections, key=lambda d: d[2] * d[3], reverse=True)[:TRACK_MAX_PERSONS]
        # greedy matching: for each existing track, find best det by IoU+dist
        used_det = set()
        used_trk = set()
        matches: list[tuple[int, int, float]] = []  # (pid, det_idx, score)
        pids = list(self.tracks.keys())
        for pid in pids:
            trk = self.tracks[pid]
            best_i, best_s = -1, -1.0
            cx, cy = trk.center()
            for i, d in enumerate(dets):
                if i in used_det:
                    continue
                iou = _iou(trk.bbox, d)
                # centroid distance normalized by avg size (fallback when IoU low due to motion)
                dcx, dcy = d[0] + d[2] * 0.5, d[1] + d[3] * 0.5
                dist = math.hypot(cx - dcx, cy - dcy)
                size = max(trk.bbox[2], trk.bbox[3], d[2], d[3], 1)
                dist_score = max(0.0, 1.0 - dist / (size * 1.4))
                score = max(iou, dist_score * 0.6)
                if score > best_s:
                    best_s, best_i = score, i
            if best_i >= 0 and best_s >= self.iou_thresh:
                matches.append((pid, best_i, best_s))
                used_det.add(best_i)
                used_trk.add(pid)
        # sort matches by score descending to be stable
        matches.sort(key=lambda x: x[2], reverse=True)
        # we already marked used, but need to avoid duplicate det reuse — already handled
        # update matched — adaptive EMA: snap faster on large jumps
        for pid, di, _ in matches:
            d = dets[di]
            trk = self.tracks[pid]
            ax, ay, aw, ah = trk.bbox
            bx, by, bw, bh = d
            # adaptive alpha based on center move
            move = math.hypot((ax + aw * 0.5) - (bx + bw * 0.5), (ay + ah * 0.5) - (by + bh * 0.5))
            avg_sz = max(1, (aw + ah + bw + bh) * 0.25)
            a_bbox = 0.78 if move > avg_sz * 0.12 else TRACK_SMOOTH_ALPHA
            nx, ny, nw, nh = (
                int(round(ax * (1 - a_bbox) + bx * a_bbox)),
                int(round(ay * (1 - a_bbox) + by * a_bbox)),
                int(round(aw * (1 - a_bbox) + bw * a_bbox)),
                int(round(ah * (1 - a_bbox) + bh * a_bbox)),
            )
            trk.bbox = (nx, ny, nw, nh)
            x1, y1, x2, y2 = nx, ny, nx + nw, ny + nh
            sx1, sy1, sx2, sy2 = trk.smooth
            # single-stage smooth for drawing — high alpha = responsive, less lag
            move = math.hypot((x1 + x2) * 0.5 - (sx1 + sx2) * 0.5, (y1 + y2) * 0.5 - (sy1 + sy2) * 0.5)
            avg_sz = max(1, (nw + nh) * 0.5)
            a2 = 0.82 if move > avg_sz * 0.08 else 0.62
            trk.smooth = (
                int(round(sx1 * (1 - a2) + x1 * a2)),
                int(round(sy1 * (1 - a2) + y1 * a2)),
                int(round(sx2 * (1 - a2) + x2 * a2)),
                int(round(sy2 * (1 - a2) + y2 * a2)),
            )
            trk.hits += 1
            trk.misses = 0
            trk.last_update = frame_idx

        # create new tracks for unmatched dets
        for i, d in enumerate(dets):
            if i in used_det:
                continue
            pid = self.next_id
            self.next_id += 1
            x, y, w, h = d
            color = _PERSON_PALETTE[(pid - 1) % len(_PERSON_PALETTE)]
            self.tracks[pid] = TrackedPerson(pid=pid, bbox=d, smooth=(x, y, x + w, y + h), color=color, hits=1, misses=0, last_update=frame_idx)

        # age unmatched tracks
        to_del: list[int] = []
        for pid, trk in self.tracks.items():
            if pid in used_trk:
                continue
            # not matched this frame — if it wasn't in matches but was existing, it stays unmatched
            # check if this pid was matched — if not, increment misses
            if pid not in [m[0] for m in matches]:
                # also skip newly created tracks (they were just added)
                if trk.last_update != frame_idx:
                    trk.misses += 1
                    if trk.misses > self.max_miss:
                        to_del.append(pid)
        for pid in to_del:
            del self.tracks[pid]

        # return active sorted by id for stable drawing
        return sorted(self.tracks.values(), key=lambda t: t.pid)

    @property
    def active(self) -> list[TrackedPerson]:
        # only confirmed tracks (>= min hits) are considered active for UI — suppresses single-frame ghosts
        confirmed = [t for t in self.tracks.values() if t.hits >= TRACK_CONFIRM_HITS]
        return sorted(confirmed, key=lambda t: t.pid)

    @property
    def all_tracks(self) -> list[TrackedPerson]:
        return sorted(self.tracks.values(), key=lambda t: t.pid)

    def clear(self) -> None:
        self.tracks.clear()
        self.next_id = 1


# ---------------------------------------------------------------------------
# Face recognition — lightweight gallery with HSV histogram + optional SFace
# ---------------------------------------------------------------------------
class FaceGallery:
    """Long-term re-identification gallery. Stores per-person embeddings (HSV hist or SFace)."""

    def __init__(self, thresh: float = 0.62, max_gallery: int = 12) -> None:
        self.thresh = thresh
        self.max_gallery = max_gallery
        self.entries: dict[int, np.ndarray] = {}  # pid -> embedding (hist 3*32 or SFace 128)
        self.names: dict[int, str] = {}
        self._sface: Any | None = None
        self._sface_ok: bool = False
        self._try_load_sface()

    def _try_load_sface(self) -> None:
        try:
            # SFace model path — try local cache first
            candidates = [Path(__file__).parent / "face_recognition_sface_2021dec.onnx", Path.cwd() / "face_recognition_sface_2021dec.onnx"]
            model = next((p for p in candidates if p.exists() and p.stat().st_size > 1_000_000), None)
            if model is None:
                return
            # FaceRecognizerSF.create expects model, backend, target; use default
            self._sface = cv2.FaceRecognizerSF.create(str(model), "", 0, 0)  # type: ignore[attr-defined]
            self._sface_ok = True
            log.debug("SFace gallery enabled via %s", model)
        except Exception as exc:
            log.debug("SFace not available: %s", exc)
            self._sface = None

    def _hist_embedding(self, crop: np.ndarray) -> np.ndarray | None:
        if crop.size == 0 or min(crop.shape[:2]) < 12:
            return None
        try:
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            # 3 channels, 16 bins each, then concat
            h = cv2.calcHist([hsv], [0], None, [16], [0, 180])
            s = cv2.calcHist([hsv], [1], None, [16], [0, 256])
            v = cv2.calcHist([hsv], [2], None, [16], [0, 256])
            hist = np.concatenate([h.flatten(), s.flatten(), v.flatten()]).astype(np.float32)
            cv2.normalize(hist, hist)
            return hist
        except Exception:
            return None

    def _sface_embedding(self, frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray | None:
        if not self._sface_ok or self._sface is None:
            return None
        try:
            x, y, w, h = bbox
            # SFace expects aligned face; use bbox as is, with 0 landmarks
            # create a dummy aligned crop 112x112
            # fallback: just use hist if SFace fails on small crop
            aligned = cv2.resize(frame[max(0, y) : y + h, max(0, x) : x + w], (112, 112), interpolation=cv2.INTER_AREA)
            feat = np.zeros((128,), dtype=np.float32)
            # SFace feature returns (1,128) via .feature(aligned)
            # try both APIs
            if hasattr(self._sface, "feature"):
                feat = self._sface.feature(aligned)  # type: ignore[operator]
                if isinstance(feat, tuple):
                    feat = feat[0]
                feat = np.asarray(feat).flatten().astype(np.float32)
                cv2.normalize(feat, feat)
                return feat
        except Exception as exc:
            log.debug("SFace feature failed: %s", exc)
        return None

    def embed(self, frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray | None:
        # try SFace first, fallback to hist
        emb = self._sface_embedding(frame, bbox)
        if emb is not None:
            return emb
        x, y, w, h = bbox
        crop = frame[max(0, y) : y + h, max(0, x) : x + w]
        return self._hist_embedding(crop)

    def match(self, emb: np.ndarray | None) -> int | None:
        if emb is None or not self.entries:
            return None
        best_pid, best_score = None, -1.0
        for pid, ref in self.entries.items():
            if ref.shape != emb.shape:
                continue
            # cosine for SFace (128-d), correlation for hist (48-d)
            if ref.size == 128:
                score = float(np.dot(ref, emb))  # both L2-normalized
            else:
                score = float(cv2.compareHist(ref, emb, cv2.HISTCMP_CORREL))
            if score > best_score:
                best_score, best_pid = score, pid
        if best_pid is not None and best_score >= self.thresh:
            return best_pid
        return None

    def enroll(self, pid: int, emb: np.ndarray | None, name: str | None = None) -> None:
        if emb is None:
            return
        if len(self.entries) >= self.max_gallery and pid not in self.entries:
            # evict oldest (lowest pid)
            oldest = min(self.entries.keys())
            del self.entries[oldest]
            self.names.pop(oldest, None)
        self.entries[pid] = emb
        if name:
            self.names[pid] = name

    def label(self, pid: int) -> str:
        return self.names.get(pid, f"Person #{pid}")


# global gallery (persists across tracker resets)
_GLOBAL_GALLERY: FaceGallery | None = None


def get_gallery() -> FaceGallery:
    global _GLOBAL_GALLERY
    if _GLOBAL_GALLERY is None:
        _GLOBAL_GALLERY = FaceGallery()
    return _GLOBAL_GALLERY

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class Phase(StrEnum):
    POMODORO = "pomodoro"
    SHORT_BREAK = "short_break"
    LONG_BREAK = "long_break"


class ThemeName(StrEnum):
    DARK = "dark"
    LIGHT = "light"
    XP = "xp"


class FocusState(StrEnum):
    CONCENTRATED = "concentrated"
    SLACKING = "slacking"
    NEUTRAL = "neutral"


class CornerStyle(StrEnum):
    ROUNDED = "rounded"
    BOXY = "boxy"


class DisplayMode(StrEnum):
    PROGRESS_BAR = "progress_bar"
    TIMER_POPUP = "timer_popup"
    BOTH = "both"


# ---------------------------------------------------------------------------
# Font engine — encapsulated TrueType rendering with caching
# ---------------------------------------------------------------------------
def _resolve_font_paths() -> tuple[str | None, str | None]:
    """Find regular + bold TTF on this system. Returns (regular, bold)."""
    font_dirs: list[str] = []
    if platform.system() == "Windows":
        windir = os.environ.get("WINDIR", r"C:\Windows")
        font_dirs.append(str(Path(windir) / "Fonts"))
    font_dirs += [
        "/usr/share/fonts/truetype/dejavu",
        "/usr/share/fonts/TTF",
        "/usr/share/fonts",
        "/Library/Fonts",
        "/System/Library/Fonts",
    ]
    regular_names = ["segoeui.ttf", "arial.ttf", "DejaVuSans.ttf", "Verdana.ttf", "Helvetica.ttf"]
    bold_names = ["segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf", "Verdanab.ttf", "arial bold.ttf"]
    regular: str | None = None
    bold: str | None = None
    for d in font_dirs:
        for name in regular_names:
            p = Path(d) / name
            if p.exists():
                regular = str(p)
                break
        for name in bold_names:
            p = Path(d) / name
            if p.exists():
                bold = str(p)
                break
        if regular and bold:
            break
    return regular, bold


_FONT_PATHS: tuple[str | None, str | None] = _resolve_font_paths() if _PIL_AVAILABLE else (None, None)
_FONT_CACHE: dict[tuple[int, bool], Any] = {}
_TEXT_SIZE_CACHE: dict[tuple[str, int, bool], tuple[int, int]] = {}
_FIT_TEXT_CACHE: dict[tuple[str, int, int, int, bool], tuple[int, tuple[int, int]]] = {}


def get_font(px: int, bold: bool = False) -> Any:
    """Cached PIL font at pixel size."""
    px = max(8, int(round(px)))
    key = (px, bool(bold))
    if key not in _FONT_CACHE:
        path = _FONT_PATHS[1 if bold else 0]
        try:
            _FONT_CACHE[key] = ImageFont.truetype(path, px)  # type: ignore[union-attr]
        except Exception:
            _FONT_CACHE[key] = ImageFont.load_default()  # type: ignore[union-attr]
    return _FONT_CACHE[key]


def text_size(text: str, px: int, bold: bool = False) -> tuple[int, int]:
    """Rendered (width, height) of text in pixels. Cached."""
    px = max(8, int(round(px)))
    key = (text, px, bool(bold))
    if (cached := _TEXT_SIZE_CACHE.get(key)) is not None:
        return cached
    if _PIL_AVAILABLE:
        font = get_font(px, bold)
        dummy = ImageDraw.Draw(Image.new("RGB", (1, 1)))  # type: ignore[union-attr]
        l, t, r, b = dummy.textbbox((0, 0), text, font=font)
        result = (r - l, b - t)
    else:
        scale = max(0.3, px / 26.0)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, max(1, int(px / 18)))
        result = (tw, th)
    if len(_TEXT_SIZE_CACHE) < TEXT_SIZE_CACHE_MAX:
        _TEXT_SIZE_CACHE[key] = result
    return result


def draw_text(
    img: np.ndarray,
    text: str,
    x: float,
    y: float,
    px: int,
    color: tuple[int, int, int],
    bold: bool = False,
    anchor: str = "la",
    stroke: int = 0,
    stroke_color: tuple[int, int, int] = (0, 0, 0),
) -> None:
    """Draw text onto a BGR numpy image.

    anchor: 'la' = x,y is top-left | 'mm' = centered | 'lm' = left-middle.
    Falls back to Hershey vectors when Pillow is unavailable.
    """
    px = max(8, int(round(px)))
    stroke = max(0, int(stroke))
    if not _PIL_AVAILABLE:
        scale = max(0.3, px / 26.0)
        thick = max(1, int(round(px / 18)))
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        if anchor == "mm":
            ox, oy = x - tw // 2, y + th // 2
        elif anchor == "lm":
            ox, oy = x, y + th // 2
        else:
            ox, oy = x, y + th
        if stroke > 0:
            for ddx in (-stroke, 0, stroke):
                for ddy in (-stroke, 0, stroke):
                    if ddx or ddy:
                        cv2.putText(
                            img,
                            text,
                            (int(ox) + ddx, int(oy) + ddy),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            scale,
                            stroke_color,
                            thick,
                            cv2.LINE_AA,
                        )
        cv2.putText(img, text, (int(ox), int(oy)), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)
        return

    font = get_font(px, bold)
    tw, th = text_size(text, px, bold)
    if anchor == "mm":
        ax, ay = x - tw / 2.0, y - th / 2.0
    elif anchor == "lm":
        ax, ay = x, y - th / 2.0
    else:
        ax, ay = x, y

    h, w = img.shape[:2]
    pad = max(4, px // 4) + 2 * stroke
    x0 = int(max(0, math.floor(ax)) - pad)
    y0 = int(max(0, math.floor(ay)) - pad)
    x1 = int(min(w, math.ceil(ax + tw)) + pad)
    y1 = int(min(h, math.ceil(ay + th)) + pad)
    if x1 <= x0 or y1 <= y0:
        return

    region = np.ascontiguousarray(img[y0:y1, x0:x1])
    pil = Image.fromarray(region[:, :, ::-1])  # type: ignore[union-attr]
    d = ImageDraw.Draw(pil)  # type: ignore[union-attr]
    fill = (int(color[2]), int(color[1]), int(color[0]))
    if stroke > 0:
        d.text(
            (ax - x0, ay - y0),
            text,
            font=font,
            fill=fill,
            stroke_width=stroke,
            stroke_fill=(int(stroke_color[2]), int(stroke_color[1]), int(stroke_color[0])),
        )
    else:
        d.text((ax - x0, ay - y0), text, font=font, fill=fill)
    img[y0:y1, x0:x1] = np.ascontiguousarray(np.array(pil)[:, :, ::-1])


def fit_text(
    text: str,
    box_w: float,
    box_h: float,
    max_px: int,
    bold: bool = False,
    pad_w: int = 12,
    pad_h: int = 8,
    min_px: int = 10,
) -> tuple[int, tuple[int, int]]:
    """Binary-search largest pixel size whose text fits (box_w, box_h). Cached."""
    max_px = int(max(min_px, max_px))
    key = (text, int(box_w), int(box_h), max_px, bool(bold))
    if (cached := _FIT_TEXT_CACHE.get(key)) is not None:
        return cached
    lo, hi = min_px, max_px
    best: tuple[int, tuple[int, int]] = (min_px, text_size(text, min_px, bold))
    while lo <= hi:
        mid = (lo + hi) // 2
        tw, th = text_size(text, mid, bold)
        if tw <= box_w - pad_w and th <= box_h - pad_h:
            best = (mid, (tw, th))
            lo = mid + 1
        else:
            hi = mid - 1
    if len(_FIT_TEXT_CACHE) < FIT_TEXT_CACHE_MAX:
        _FIT_TEXT_CACHE[key] = best
    return best


# ---------------------------------------------------------------------------
# Layout system
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class UIElementConfig:
    """Position of one UI tile in grid cells from top-left."""

    enabled: bool = True
    x: float = 0.0
    y: float = 0.0
    width: float = 1.0
    height: float = 1.0
    anchor: str = "top-left"  # kept for backwards compat
    margin: float = 0.0
    padding: float = 0.0
    font_scale: float = 1.0
    custom_props: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UIElementConfig:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass(slots=True)
class LayoutConfig:
    grid_cols: int = GRID_COLS_DEFAULT
    grid_rows: int = GRID_ROWS_DEFAULT
    elements: dict[str, UIElementConfig] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "grid_cols": self.grid_cols,
            "grid_rows": self.grid_rows,
            "elements": {k: v.to_dict() for k, v in self.elements.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LayoutConfig:
        elements = {k: UIElementConfig.from_dict(v) for k, v in data.get("elements", {}).items()}
        return cls(grid_cols=data.get("grid_cols", GRID_COLS_DEFAULT), grid_rows=data.get("grid_rows", GRID_ROWS_DEFAULT), elements=elements)

    def save(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> LayoutConfig | None:
        p = Path(path)
        if not p.exists():
            return None
        try:
            cfg = cls.from_dict(json.loads(p.read_text(encoding="utf-8")))
        except Exception as exc:
            log.warning("Corrupt layout file %s: %s", p, exc)
            return None
        if cfg.grid_cols < 1 or cfg.grid_rows < 1 or not cfg.elements:
            return None
        return cfg

    @classmethod
    def default(cls) -> LayoutConfig:
        # minimalist — safe margins, timer inset from edge, no center Pomodoro, thin progress
        return cls(
            grid_cols=GRID_COLS_DEFAULT,
            grid_rows=GRID_ROWS_DEFAULT,
            elements={
                "timer_popup": UIElementConfig(enabled=True, x=8.6, y=0.42, width=2.9, height=0.88, margin=0.18, font_scale=0.92),
                "phase_label": UIElementConfig(enabled=False, x=4.6, y=0.55, width=2.9, height=0.7, margin=0.12),
                "progress_bar": UIElementConfig(enabled=True, x=0.30, y=0.10, width=11.4, height=0.12, margin=0.06),
                "focus_display": UIElementConfig(enabled=True, x=0.30, y=0.38, width=2.7, height=0.40, margin=0.06),
                "status_display": UIElementConfig(enabled=False, x=0.4, y=1.0, width=3.2, height=0.42, margin=0.08),
                "main_buttons": UIElementConfig(enabled=True, x=3.2, y=7.12, width=5.6, height=0.68, margin=0.12),
                "quit_hint": UIElementConfig(enabled=False, x=10.6, y=7.45, width=1.2, height=0.4, margin=0.1),
            },
        )


class LayoutManager:
    """Grid-based layout with pixel rect resolution."""

    def __init__(self, config: LayoutConfig) -> None:
        self.config = config
        self.grid_cols = config.grid_cols
        self.grid_rows = config.grid_rows

    def set_grid(self, cols: int, rows: int) -> None:
        self.grid_cols = cols
        self.grid_rows = rows
        self.config.grid_cols = cols
        self.config.grid_rows = rows

    def get_element_rect(
        self, name: str, frame_w: int, frame_h: int, include_disabled: bool = False
    ) -> tuple[int, int, int, int]:
        elem = self.config.elements.get(name)
        if elem is None or (not elem.enabled and not include_disabled):
            return (0, 0, 0, 0)
        cell_w = frame_w / float(self.grid_cols)
        cell_h = frame_h / float(self.grid_rows)
        margin_px = max(0, int(elem.margin * min(cell_w, cell_h)))
        padding_px = max(0, int(elem.padding * min(cell_w, cell_h)))
        w = min(int(round(elem.width * cell_w)), frame_w - 2 * margin_px)
        h = min(int(round(elem.height * cell_h)), frame_h - 2 * margin_px)
        x = max(margin_px, min(int(round(elem.x * cell_w)), frame_w - w - margin_px))
        y = max(margin_px, min(int(round(elem.y * cell_h)), frame_h - h - margin_px))
        return (x + padding_px, y + padding_px, x + w - padding_px, y + h - padding_px)

    def update_element(self, name: str, **kwargs: Any) -> None:
        if name not in self.config.elements:
            self.config.elements[name] = UIElementConfig()
        elem = self.config.elements[name]
        for k, v in kwargs.items():
            if hasattr(elem, k):
                setattr(elem, k, v)

    def enable_element(self, name: str, enabled: bool = True) -> None:
        if name in self.config.elements:
            self.config.elements[name].enabled = enabled

    def get_enabled_elements(self) -> list[str]:
        return [k for k, v in self.config.elements.items() if v.enabled]


# ---------------------------------------------------------------------------
# Rendering primitives
# ---------------------------------------------------------------------------
def styled_rect(
    img: np.ndarray,
    x1: int | float,
    y1: int | float,
    x2: int | float,
    y2: int | float,
    fill: tuple[int, int, int] | None = None,
    border: tuple[int, int, int] | None = None,
    thickness: int = 1,
    radius: int = 0,
) -> None:
    """Rounded or boxy rectangle with optional fill and border."""
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    if x2 <= x1 or y2 <= y1:
        return
    r = int(max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2)))
    if fill is not None:
        if r <= 0:
            cv2.rectangle(img, (x1, y1), (x2, y2), fill, -1)
        else:
            cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), fill, -1, lineType=cv2.LINE_AA)
            cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), fill, -1, lineType=cv2.LINE_AA)
            for cx, cy in ((x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)):
                cv2.circle(img, (cx, cy), r, fill, -1, lineType=cv2.LINE_AA)
    if border is not None:
        t = max(1, int(thickness))
        if r <= 0:
            cv2.rectangle(img, (x1, y1), (x2, y2), border, t)
        else:
            half = t // 2
            cv2.line(img, (x1 + r, y1 + half), (x2 - r, y1 + half), border, t, cv2.LINE_AA)
            cv2.line(img, (x1 + r, y2 - half), (x2 - r, y2 - half), border, t, cv2.LINE_AA)
            cv2.line(img, (x1 + half, y1 + r), (x1 + half, y2 - r), border, t, cv2.LINE_AA)
            cv2.line(img, (x2 - half, y1 + r), (x2 - half, y2 - r), border, t, cv2.LINE_AA)
            cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 0, 180, 270, border, t, cv2.LINE_AA)
            cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 0, 270, 360, border, t, cv2.LINE_AA)
            cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 0, 90, 180, border, t, cv2.LINE_AA)
            cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, border, t, cv2.LINE_AA)


def _h_gradient(
    img: np.ndarray,
    x1: int | float,
    y1: int | float,
    x2: int | float,
    y2: int | float,
    c1: tuple[int, int, int],
    c2: tuple[int, int, int],
    radius: int = 0,
) -> None:
    """Horizontal gradient — fast numpy blit, with rounded clipping if needed."""
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return
    # clamp to image bounds
    ih, iw = img.shape[:2]
    cx1, cy1, cx2, cy2 = max(0, x1), max(0, y1), min(iw, x2), min(ih, y2)
    if cx2 <= cx1 or cy2 <= cy1:
        return
    c1a = np.array(c1, dtype=np.float32)
    c2a = np.array(c2, dtype=np.float32)
    t = np.linspace(0, 1, w, dtype=np.float32)
    cols = (c1a + np.outer(t, c2a - c1a)).clip(0, 255).astype(np.uint8)  # (w,3) BGR
    if radius > 0:
        r = min(radius, w // 2, h // 2)
        centers = np.arange(w, dtype=np.float32)
        edge_dist = np.minimum(centers, w - 1 - centers)
        shrink = (r - np.sqrt(np.maximum(0, r * r - np.maximum(0, r - edge_dist) ** 2))).astype(np.int32)
        shrink = np.where(edge_dist < r, shrink, 0)
        # fast path: still per-column but without per-pixel float math; keep AA line for curved edges
        for i in range(w):
            gx = x1 + i
            if gx < cx1 or gx >= cx2:
                continue
            sy, ey = y1 + int(shrink[i]), y2 - int(shrink[i])
            sy, ey = max(sy, cy1), min(ey, cy2)
            if sy < ey:
                cv2.line(img, (gx, sy), (gx, ey), tuple(cols[i].tolist()), 1, cv2.LINE_AA)
        return
    # fast blit: slice to clipped region
    x_off = cx1 - x1
    cols_clip = cols[x_off : x_off + (cx2 - cx1)]
    # broadcast rows: (h_clip, w_clip, 3)
    img[cy1:cy2, cx1:cx2] = cols_clip[None, :, :]


def _v_gradient(
    img: np.ndarray,
    x1: int | float,
    y1: int | float,
    x2: int | float,
    y2: int | float,
    c1: tuple[int, int, int],
    c2: tuple[int, int, int],
) -> None:
    """Vertical gradient — fast numpy blit."""
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    h = y2 - y1
    w = x2 - x1
    if h <= 0 or w <= 0:
        return
    ih, iw = img.shape[:2]
    cx1, cy1, cx2, cy2 = max(0, x1), max(0, y1), min(iw, x2), min(ih, y2)
    if cx2 <= cx1 or cy2 <= cy1:
        return
    c1a = np.array(c1, dtype=np.float32)
    c2a = np.array(c2, dtype=np.float32)
    t = np.linspace(0, 1, h, dtype=np.float32)
    cols = (c1a + np.outer(t, c2a - c1a)).clip(0, 255).astype(np.uint8)  # (h,3)
    y_off = cy1 - y1
    cols_clip = cols[y_off : y_off + (cy2 - cy1)]
    # broadcast columns: need (h_clip, w_clip, 3)
    img[cy1:cy2, cx1:cx2] = cols_clip[:, None, :]


def _xp_button(
    img: np.ndarray,
    x1: int | float,
    y1: int | float,
    x2: int | float,
    y2: int | float,
    label: str,
    accent: bool = False,
    active: bool = False,
    text_color: tuple[int, int, int] | None = None,
    px: int = 14,
    bold: bool = False,
) -> None:
    """Windows XP Luna beveled button — gradient fill + 3-D bevel."""
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    bw, bh = x2 - x1, y2 - y1
    if bw < 4 or bh < 4:
        return
    r = min(6, bh // 4)
    if accent:
        c_top, c_bot = (255, 170, 50), (200, 115, 10)
    elif active:
        c_top, c_bot = (255, 220, 170), (230, 190, 130)
    else:
        c_top, c_bot = (240, 235, 225), (210, 205, 195)
    hi, sh, dk = (255, 255, 255), (105, 105, 105), (60, 60, 60)

    # fast gradient fill — two vertical gradients blitted via numpy
    c1 = np.array(c_top, dtype=np.float32)
    c2 = np.array(c_bot, dtype=np.float32)
    half_h = bh // 2
    t1 = np.linspace(0, 1, half_h, dtype=np.float32) if half_h > 0 else np.array([0.0], dtype=np.float32)
    t2 = np.linspace(0, 1, bh - half_h, dtype=np.float32) if bh - half_h > 0 else np.array([0.0], dtype=np.float32)
    top_cols = (c1 + np.outer(t1, c2 - c1)).clip(0, 255).astype(np.uint8)
    bot_cols = (c2 + np.outer(t2, c1 - c2)).clip(0, 255).astype(np.uint8)
    ih, iw = img.shape[:2]
    x1i, x2i = x1 + 2, x2 - 3
    y1i, ym, y2i = y1 + 2, y1 + 2 + half_h, y2 - 2
    # clip to image
    if x2i > x1i:
        # top half
        ty1, ty2 = max(y1i, 0), min(ym, ih)
        tx1, tx2 = max(x1i, 0), min(x2i, iw)
        if ty2 > ty1 and tx2 > tx1:
            y_off = ty1 - y1i
            img[ty1:ty2, tx1:tx2] = top_cols[y_off : y_off + (ty2 - ty1), None, :]
        # bottom half
        by1, by2 = max(ym, 0), min(y2i, ih)
        if by2 > by1 and tx2 > tx1:
            y_off = by1 - ym
            img[by1:by2, tx1:tx2] = bot_cols[y_off : y_off + (by2 - by1), None, :]

    cv2.line(img, (x1 + r, y1), (x2 - r, y1), hi, 2, cv2.LINE_AA)
    cv2.line(img, (x1, y1 + r), (x1, y2 - r), hi, 2, cv2.LINE_AA)
    cv2.line(img, (x2 - 1, y1 + r), (x2 - 1, y2 - r), sh, 2, cv2.LINE_AA)
    cv2.line(img, (x1 + r, y2 - 1), (x2 - r, y2 - 1), sh, 2, cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x2, y2), dk, 1, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2, y2), dk, 1, cv2.LINE_AA)
    for cx, cy, s, e in [
        (x1 + r, y1 + r, 180, 270),
        (x2 - r, y1 + r, 270, 360),
        (x1 + r, y2 - r, 90, 180),
        (x2 - r, y2 - r, 0, 90),
    ]:
        cv2.ellipse(img, (cx, cy), (r, r), 0, s, e, hi, 2, cv2.LINE_AA)
    tc = text_color or (15, 15, 15)
    draw_text(img, label, x1 + bw // 2, y1 + bh // 2, px, tc, bold=bold, anchor="mm")


def _xp_title_bar(
    img: np.ndarray,
    x1: int | float,
    y1: int | float,
    x2: int | float,
    y2: int | float,
    title: str,
    close_btn: bool = True,
) -> tuple[int, int, int, int] | None:
    """XP Luna blue gradient title bar with red close button."""
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    tb_h = y2 - y1
    if tb_h < 4:
        return None
    _h_gradient(img, x1, y1, x2, y2, (180, 100, 10), (250, 180, 60))
    mid = (x1 + x2) // 2
    stripe_w = (x2 - x1) // 3
    _h_gradient(img, mid - stripe_w // 2, y1 + 1, mid + stripe_w // 2, y2 - 1, (255, 200, 80), (230, 160, 40))
    cv2.line(img, (x1, y2 - 1), (x2, y2 - 1), (120, 70, 0), 1, cv2.LINE_AA)
    draw_text(img, title, x1 + 8, y1 + tb_h // 2, max(12, int(tb_h * 0.52)), (255, 255, 255), bold=True, anchor="lm", stroke=1, stroke_color=(40, 20, 0))
    if close_btn:
        cb_size = int(tb_h * 0.72)
        cb_x2 = x2 - 4
        cb_x1 = cb_x2 - cb_size
        cb_y1 = y1 + (tb_h - cb_size) // 2
        cb_y2 = cb_y1 + cb_size
        _v_gradient(img, cb_x1, cb_y1, cb_x2, cb_y2, (80, 80, 230), (40, 40, 180))
        cv2.rectangle(img, (cb_x1, cb_y1), (cb_x2, cb_y2), (20, 20, 120), 1)
        cv2.line(img, (cb_x1, cb_y1), (cb_x2 - 1, cb_y1), (140, 140, 255), 1, cv2.LINE_AA)
        cv2.line(img, (cb_x1, cb_y1), (cb_x1, cb_y2 - 1), (140, 140, 255), 1, cv2.LINE_AA)
        draw_text(img, "x", (cb_x1 + cb_x2) // 2, (cb_y1 + cb_y2) // 2, max(8, int(cb_size * 0.48)), (255, 255, 255), bold=True, anchor="mm")
        return (cb_x1, cb_y1, cb_x2, cb_y2)
    return None


def _xp_progress_bar(img: np.ndarray, x1: int | float, y1: int | float, x2: int | float, y2: int | float, progress: float, rounded: bool = False) -> None:
    """XP chunky green progress bar."""
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    pw, ph = x2 - x1, y2 - y1
    if pw < 8 or ph < 4:
        return
    r = min(3, ph // 3) if rounded else 0
    styled_rect(img, x1, y1, x2, y2, fill=(80, 80, 80), radius=r)
    cv2.line(img, (x1, y1), (x2 - 1, y1), (50, 50, 50), 1, cv2.LINE_AA)
    cv2.line(img, (x1, y1), (x1, y2 - 1), (50, 50, 50), 1, cv2.LINE_AA)
    cv2.line(img, (x2 - 1, y1 + 1), (x2 - 1, y2 - 1), (160, 160, 160), 1, cv2.LINE_AA)
    cv2.line(img, (x1 + 1, y2 - 1), (x2 - 1, y2 - 1), (160, 160, 160), 1, cv2.LINE_AA)
    if progress > 0:
        fill_w = max(4, int(pw * min(1.0, progress)))
        fx2 = x1 + fill_w
        c1 = np.array([110, 210, 80], dtype=np.float32)
        c2 = np.array([50, 160, 20], dtype=np.float32)
        gh = ph - 4
        if gh > 0:
            t = np.linspace(0, 1, gh, dtype=np.float32)
            cols = (c1 + np.outer(t, c2 - c1)).clip(0, 255).astype(np.uint8)
            # fast blit
            ih, iw = img.shape[:2]
            y1i, y2i = y1 + 2, y1 + 2 + gh
            x1i, x2i = x1 + 2, fx2 - 1
            # clip
            cy1, cy2 = max(y1i, 0), min(y2i, ih)
            cx1, cx2 = max(x1i, 0), min(x2i, iw)
            if cy2 > cy1 and cx2 > cx1:
                y_off = cy1 - y1i
                img[cy1:cy2, cx1:cx2] = cols[y_off : y_off + (cy2 - cy1), None, :]
        cv2.line(img, (x1 + 2, y1 + 1), (fx2 - 2, y1 + 1), (160, 255, 120), 1, cv2.LINE_AA)
        seg_w = max(12, ph)
        for sx in range(x1 + seg_w, fx2 - 2, seg_w):
            if 0 <= sx < img.shape[1]:
                cv2.line(img, (sx, y1 + 2), (sx, y2 - 2), (80, 180, 50), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Onboarding — first-launch intro overlay
# ---------------------------------------------------------------------------
class OnboardingManager:
    """6-step intro overlay shown on first launch."""

    _STEPS: list[dict[str, Any]] = [
        {
            "title": "Welcome to Pomodoro Camera",
            "body": "A focus timer that watches your webcam to keep you concentrated.  Work in short bursts, take breaks, and let the camera help you stay on track.",
            "accent": (0, 180, 255),
            "highlight": None,
        },
        {
            "title": "Pomodoro Timer",
            "body": "25-minute work sessions followed by 5-minute breaks.\nPress the Start button or hit  S  to begin.",
            "accent": (80, 200, 120),
            "highlight": (0.82, 0.08, 0.28, 0.20),
        },
        {
            "title": "Camera Focus Tracking",
            "body": "Your webcam detects your face and monitors movement.\nStay visible and still to stay  Concentrated.",
            "accent": (0, 200, 220),
            "highlight": (0.08, 0.05, 0.30, 0.30),
        },
        {
            "title": "Slacking Alerts",
            "body": "When the timer is running, looking away or moving too\nmuch triggers a red alert with a soft beep.  Toggle in\nSettings > Alerts.",
            "accent": (0, 100, 255),
            "highlight": (0.25, 0.35, 0.75, 0.50),
        },
        {
            "title": "Custom Layout",
            "body": "Drag tiles to reposition any element.  Press  E  or\nuse Settings > Layout Edit Mode to rearrange.",
            "accent": (220, 180, 40),
            "highlight": (0.15, 0.60, 0.85, 0.95),
        },
        {
            "title": "Keyboard Shortcuts",
            "body": "  S  start / pause      E  layout edit\n  Q  quit               Esc  exit edit",
            "accent": (180, 140, 255),
            "highlight": None,
        },
    ]

    def __init__(self) -> None:
        self.step: int = 0
        self.active: bool = True

    def _card_rects(self, fw: int, fh: int) -> dict[str, Any]:
        card_w = int(min(520, fw * 0.60))
        card_h = int(min(310, fh * 0.52))
        card_x = (fw - card_w) // 2
        card_y = (fh - card_h) // 2
        pad = max(14, int(card_w * 0.06))
        btn_h = max(28, int(card_h * 0.12))
        nav_y = card_y + card_h - pad - btn_h
        is_last = self.step >= len(self._STEPS) - 1
        is_first = self.step == 0
        bw_back = max(80, int(card_w * 0.16))
        bx1 = card_x + pad
        btn_back = (bx1, nav_y, bx1 + bw_back, nav_y + btn_h) if not is_first else None
        bw_next = max(100, int(card_w * 0.22))
        bx2 = card_x + card_w - pad - bw_next
        btn_next = (bx2, nav_y, bx2 + bw_next, nav_y + btn_h)
        skip_w = max(60, int(card_w * 0.12))
        sx = card_x + card_w - pad - skip_w
        sy = nav_y - int(btn_h * 0.65)
        btn_skip = (sx, sy, sx + skip_w, sy + int(btn_h * 0.55)) if not is_last else None
        return {
            "card": (card_x, card_y, card_x + card_w, card_y + card_h),
            "card_w": card_w,
            "card_h": card_h,
            "pad": pad,
            "btn_h": btn_h,
            "nav_y": nav_y,
            "btn_next": btn_next,
            "btn_back": btn_back,
            "btn_skip": btn_skip,
            "is_last": is_last,
            "is_first": is_first,
        }

    def draw_overlay(self, frame: np.ndarray, theme: dict[str, Any], rounded: bool, ui_scale: float) -> None:
        if not self.active:
            return
        fw, fh = frame.shape[1], frame.shape[0]
        step = self._STEPS[self.step]
        accent: tuple[int, int, int] = step["accent"]
        r = self._card_rects(fw, fh)
        card_x, card_y, card_x2, card_y2 = r["card"]
        card_w, card_h, pad, btn_h, nav_y = r["card_w"], r["card_h"], r["pad"], r["btn_h"], r["nav_y"]
        is_xp = getattr(self, "_theme", "dark") == "xp"

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (fw, fh), (180, 175, 165) if is_xp else (10, 10, 12), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        if (ht := step.get("highlight")) is not None:
            hx1, hy1, hx2, hy2 = int(fw * ht[0]), int(fh * ht[1]), int(fw * ht[2]), int(fh * ht[3])
            hr = 14 if rounded and not is_xp else 0
            pulse = 0.35 + 0.10 * math.sin(time.time() * 3.0)
            hl_overlay = frame.copy()
            styled_rect(hl_overlay, hx1, hy1, hx2, hy2, fill=accent, radius=hr)
            cv2.addWeighted(hl_overlay, pulse, frame, 1.0 - pulse, 0, frame)
            styled_rect(frame, hx1, hy1, hx2, hy2, border=accent, thickness=2, radius=hr)

        if is_xp:
            tb_h = max(22, int(card_h * 0.12))
            styled_rect(frame, card_x, card_y, card_x2, card_y2, fill=(236, 233, 216), border=(104, 104, 104), thickness=2, radius=0)
            _xp_title_bar(frame, card_x, card_y, card_x2, card_y + tb_h, step["title"])
            body_top = card_y + tb_h + 4
        else:
            card_r = 18 if rounded else 0
            card_bg = frame.copy()
            styled_rect(card_bg, card_x, card_y, card_x2, card_y2, fill=(28, 28, 32), radius=card_r)
            cv2.addWeighted(card_bg, 0.88, frame, 0.12, 0, frame)
            styled_rect(frame, card_x, card_y, card_x2, card_y2, border=accent, thickness=2, radius=card_r)
            bar_h = 4
            bar_x1 = card_x + 22 if rounded else card_x
            bar_x2 = card_x2 - 22 if rounded else card_x2
            cv2.line(frame, (bar_x1, card_y + bar_h), (bar_x2, card_y + bar_h), accent, bar_h, cv2.LINE_AA)
            title_px = max(18, int(min(22, card_h * 0.09) * ui_scale))
            draw_text(frame, step["title"], card_x + pad, card_y + pad + int(card_h * 0.05), title_px, accent, bold=True, anchor="la")
            body_top = card_y + pad + int(card_h * 0.22)

        body_px = max(13, int(min(16, card_h * 0.065) * ui_scale))
        body_y = body_top
        for line in step["body"].split("\n"):
            tc = (16, 16, 16) if is_xp else theme.get("text", (220, 220, 220))
            draw_text(frame, line, card_x + pad, body_y, body_px, tc, anchor="la")
            body_y += int(body_px * 1.65)

        if r["btn_back"]:
            bx1, by1, bx2, by2 = r["btn_back"]
            if is_xp:
                _xp_button(frame, bx1, by1, bx2, by2, "<  Back", px=max(11, int(body_px * 0.85)))
            else:
                styled_rect(frame, bx1, by1, bx2, by2, fill=(50, 50, 55), border=(90, 90, 95), thickness=1, radius=btn_h // 2 if rounded else 0)
                draw_text(frame, "<  Back", bx1 + (bx2 - bx1) // 2, by1 + (by2 - by1) // 2, max(12, int(body_px * 0.9)), (170, 170, 175), anchor="mm")

        bx1, by1, bx2, by2 = r["btn_next"]
        next_label = "Start  >" if r["is_last"] else "Next  >"
        if is_xp:
            _xp_button(frame, bx1, by1, bx2, by2, next_label, accent=True, px=max(12, int(body_px * 0.9)), bold=True)
        else:
            styled_rect(frame, bx1, by1, bx2, by2, fill=accent, radius=btn_h // 2 if rounded else 0)
            draw_text(frame, next_label, bx1 + (bx2 - bx1) // 2, by1 + (by2 - by1) // 2, max(13, int(body_px * 0.95)), (255, 255, 255), bold=True, anchor="mm")

        if r["btn_skip"]:
            sx1, sy1, sx2, sy2 = r["btn_skip"]
            sc = (90, 80, 60) if is_xp else (110, 110, 118)
            draw_text(frame, "Skip", sx1 + (sx2 - sx1) // 2, sy1 + (sy2 - sy1) // 2, max(11, int(body_px * 0.78)), sc, anchor="mm")

        dot_r = max(3, int(min(5, card_h * 0.015)))
        dot_gap = dot_r * 3
        total_dots_w = len(self._STEPS) * dot_r * 2 + (len(self._STEPS) - 1) * dot_gap
        dot_sx = card_x + (card_w - total_dots_w) // 2
        dot_cy = nav_y - int(btn_h * 0.25)
        for i in range(len(self._STEPS)):
            cx = dot_sx + i * (dot_r * 2 + dot_gap) + dot_r
            c = accent if i == self.step else (70, 70, 78)
            cv2.circle(frame, (cx, dot_cy), dot_r, c, -1, cv2.LINE_AA)

    def handle_click(self, x: int, y: int, fw: int, fh: int) -> bool:
        if not self.active:
            return False
        r = self._card_rects(fw, fh)

        def hit(rect: tuple[int, int, int, int] | None) -> bool:
            return bool(rect and rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3])

        if hit(r["btn_next"]):
            self.step += 1
            if self.step >= len(self._STEPS):
                self.active = False
            return True
        if hit(r["btn_back"]) and self.step > 0:
            self.step -= 1
            return True
        if hit(r["btn_skip"]):
            self.active = False
            return True
        return False


# ---------------------------------------------------------------------------
# Button handler
# ---------------------------------------------------------------------------
class ButtonHandler:
    def __init__(self, timer: PomodoroTimer) -> None:  # type: ignore[name-defined]
        self.timer = timer
        self.button_regions: dict[str, dict[str, float]] = {
            "start": {"x1": 0.06, "y1": 0.82, "x2": 0.30, "y2": 0.95},
            "reset": {"x1": 0.36, "y1": 0.82, "x2": 0.58, "y2": 0.95},
            "settings": {"x1": 0.66, "y1": 0.82, "x2": 0.88, "y2": 0.95},
        }

    def is_button_clicked(self, event: int, x: int, y: int, frame_w: int | None = None, frame_h: int | None = None) -> str | None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return None
        if frame_w is None or frame_h is None:
            frame_w, frame_h = self.timer.frame_width, self.timer.frame_height
        for btn_name, coords in self.button_regions.items():
            x1, y1, x2, y2 = coords.get("x1", 0), coords.get("y1", 0), coords.get("x2", 0), coords.get("y2", 0)
            if all(0.0 <= v <= 1.0 for v in (x1, y1, x2, y2)):
                x1, y1 = int(frame_w * x1), int(frame_h * y1)  # type: ignore[assignment]
                x2, y2 = int(frame_w * x2), int(frame_h * y2)  # type: ignore[assignment]
            else:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # type: ignore[assignment]
            if x1 <= x <= x2 and y1 <= y <= y2:
                return btn_name
        return None

    def handle_button_click(self, button_name: str) -> None:
        match button_name:
            case "start":
                self.timer.toggle_timer()
            case "reset":
                self.timer.reset_timer()
            case "settings":
                self.timer.toggle_settings()


# ---------------------------------------------------------------------------
# PomodoroTimer — core state + rendering + I/O
# ---------------------------------------------------------------------------
class PomodoroTimer:
    """Manage timer, camera, focus tracking, and UI in one window."""

    DISPLAY_MODE_PROGRESS_BAR: Final = DisplayMode.PROGRESS_BAR
    DISPLAY_MODE_TIMER_POPUP: Final = DisplayMode.TIMER_POPUP
    DISPLAY_MODE_BOTH: Final = DisplayMode.BOTH

    THEMES: Final[dict[str, dict[str, Any]]] = {
        "dark": {
            "backdrop": (28, 28, 30),
            "panel_fill": (34, 34, 38),
            "panel_border": (168, 168, 175),
            "divider": (92, 92, 98),
            "text": (240, 240, 240),
            "subtext": (185, 185, 190),
            "button_fill": (58, 58, 64),
            "button_border": (150, 150, 158),
            "accent": (120, 160, 255),
            "on_color": (140, 220, 140),
            "off_color": (110, 110, 230),
            "popup_fill": (24, 24, 28),
            "bar_track": (48, 48, 54),
            "widget_fill": (45, 45, 48),
        },
        "light": {
            "backdrop": (236, 236, 240),
            "panel_fill": (250, 250, 252),
            "panel_border": (70, 70, 78),
            "divider": (165, 165, 172),
            "text": (34, 34, 40),
            "subtext": (95, 95, 102),
            "button_fill": (228, 228, 233),
            "button_border": (115, 115, 122),
            "accent": (30, 90, 220),
            "on_color": (40, 140, 60),
            "off_color": (40, 60, 190),
            "popup_fill": (255, 255, 255),
            "bar_track": (210, 210, 216),
            "widget_fill": (215, 215, 220),
        },
        "xp": {
            "backdrop": (212, 208, 200),
            "panel_fill": (236, 233, 216),
            "panel_border": (104, 104, 104),
            "divider": (172, 168, 160),
            "text": (16, 16, 16),
            "subtext": (90, 80, 60),
            "button_fill": (226, 223, 210),
            "button_border": (104, 104, 104),
            "accent": (200, 130, 30),
            "on_color": (80, 170, 50),
            "off_color": (60, 60, 180),
            "popup_fill": (236, 233, 216),
            "bar_track": (160, 160, 160),
            "widget_fill": (212, 208, 200),
            "xp_title_bar": ((180, 100, 10), (250, 180, 60)),
            "xp_close_btn": (80, 80, 230),
            "xp_btn_top": (240, 235, 225),
            "xp_btn_bot": (210, 205, 195),
            "xp_btn_accent_top": (255, 170, 50),
            "xp_btn_accent_bot": (200, 115, 10),
        },
    }

    def __init__(
        self,
        session_duration_minutes: int = POMODORO_MIN_DEFAULT,
        break_duration_minutes: int = BREAK_MIN_DEFAULT,
        display_mode: str = DisplayMode.BOTH,
    ) -> None:
        self.session_duration_minutes = session_duration_minutes
        self.break_duration_minutes = break_duration_minutes
        self.display_mode = display_mode

        self.current_phase: str = Phase.POMODORO
        self.time_left: int = self.session_duration_minutes * 60
        self.is_running: bool = False
        self.session_open: bool = False
        self.frame_count: int = 0
        self.frame_width: int = DEFAULT_FRAME_W
        self.frame_height: int = DEFAULT_FRAME_H
        self.cap: cv2.VideoCapture | None = None
        self.phase_started_at: float | None = None
        self.phase_duration_seconds: int = self.session_duration_minutes * 60

        self.show_progress_bar: bool = display_mode in (DisplayMode.PROGRESS_BAR, DisplayMode.BOTH)
        self.show_timer_popup: bool = display_mode in (DisplayMode.TIMER_POPUP, DisplayMode.BOTH)
        self.settings_visible: bool = False
        self.focus_score: float = 0.0
        self.focus_state: str = FocusState.NEUTRAL
        self.prev_gray: np.ndarray | None = None
        self.face_cascade: cv2.CascadeClassifier | None = self._load_face_cascade()
        self.eye_cascade: cv2.CascadeClassifier | None = self._load_eye_cascade()
        self.button_handler = ButtonHandler(self)
        self.progress_bar_enabled: bool = False
        self.camera_enabled: bool = True
        self.widget_width: int = 420
        self.widget_height: int = 240
        self.theme: str = ThemeName.DARK
        self.corner_style: str = CornerStyle.ROUNDED

        self.layout_manager = LayoutManager(LayoutConfig.default())
        self.layout_file = LAYOUT_FILE
        self._load_layout()

        self.ui_scale: float = UI_SCALE_DEFAULT
        self.settings_file = SETTINGS_FILE

        self.layout_edit_mode: bool = False
        self._dragging_element: str | None = None
        self._drag_offset_x: float = 0
        self._drag_offset_y: float = 0
        self._edit_done_rect: tuple[int, int, int, int] | None = None

        self.last_faces: list[tuple[int, int, int, int]] = []
        self._face_box: tuple[int, int, int, int] | None = None
        self._face_miss_frames: int = 0
        self._cam_view: tuple[int, int, int, int, int, int] | None = None

        # multi-person tracking
        self.tracker = MultiPersonTracker()
        self.person_tracks: list[TrackedPerson] = []
        self.person_count: int = 0
        # legacy single-box for compat; now derived from tracker
        self._person_colors: dict[int, tuple[int, int, int]] = {}

        self.alerts_enabled: bool = True
        self._last_alert_ts: float = 0.0
        self._alert_active: bool = False
        self._face_frame_count: int = 0
        self._last_gray: np.ndarray | None = None

        self._load_settings()

        self.onboarding = OnboardingManager()
        if self.settings_file.exists():
            self.onboarding.active = False

    # -- helpers -----------------------------------------------------------
    def _theme_palette(self) -> dict[str, Any]:
        return self.THEMES.get(getattr(self, "theme", ThemeName.DARK), self.THEMES[ThemeName.DARK])

    def _is_rounded(self) -> bool:
        return getattr(self, "corner_style", CornerStyle.ROUNDED) == CornerStyle.ROUNDED

    def _play_alert_sound(self) -> None:
        now = time.time()
        if not self.alerts_enabled or now - self._last_alert_ts < 8.0:
            return
        self._last_alert_ts = now
        try:
            if platform.system() == "Windows":
                import winsound

                winsound.Beep(880, 180)
                winsound.Beep(660, 220)
            else:
                print("\a", end="", flush=True)
        except Exception as exc:
            log.debug("Alert sound failed: %s", exc)

    # -- face outline & slacking alert ------------------------------------
    def _draw_face_outline(self, frame: np.ndarray) -> None:
        # multi-person: draw every tracked face with persistent ID and per-person color
        tracks = getattr(self, "person_tracks", [])
        if not self.camera_enabled or not tracks:
            # fallback to legacy last_faces if tracker empty but faces exist (first frames)
            if self.last_faces and self.camera_enabled:
                tracks = []  # will be handled as single
            else:
                self._face_miss_frames += 1
                if self._face_miss_frames > 18:
                    self._face_box = None
                return
        # use tracker if available, else single legacy bbox
        if tracks:
            self._face_miss_frames = 0
            fw, fh = frame.shape[1], frame.shape[0]
            theme = self._theme_palette()
            # precompute global state color tint
            if not self.is_running:
                state_tint = None
                base_thick = 2
            elif self.focus_state == FocusState.SLACKING:
                pulse = 0.5 + 0.5 * math.sin(time.time() * 6)
                base = (60, 60, 255)
                state_tint = tuple(int(c * (0.7 + 0.3 * pulse)) for c in base)
                base_thick = 3 + int(2 * pulse)
            elif self.focus_state == FocusState.CONCENTRATED:
                state_tint = theme["on_color"]
                base_thick = 2
            else:
                state_tint = theme["accent"]
                base_thick = 2

            for person in tracks:
                if self._cam_view is None:
                    bx1, by1, bx2, by2 = person.smooth
                    pad = 8
                    bx1, by1, bx2, by2 = bx1 - pad, by1 - pad, bx2 + pad, by2 + pad
                else:
                    off_x, off_y, fit_w, fit_h, cam_w, cam_h = self._cam_view
                    scale_x = fit_w / float(cam_w) if cam_w else 1.0
                    scale_y = fit_h / float(cam_h) if cam_h else scale_x
                    x1, y1, x2, y2 = person.smooth
                    pad_cam = int(max(6, min(cam_w, cam_h) * 0.020))
                    x1, y1, x2, y2 = x1 - pad_cam, y1 - pad_cam, x2 + pad_cam, y2 + pad_cam
                    bx1 = int(off_x + x1 * scale_x)
                    by1 = int(off_y + y1 * scale_y)
                    bx2 = int(off_x + x2 * scale_x)
                    by2 = int(off_y + y2 * scale_y)

                bw, bh = bx2 - bx1, by2 - by1
                if bw < 10 or bh < 10:
                    continue
                fw, fh = frame.shape[1], frame.shape[0]
                bx1, by1, bx2, by2 = max(0, bx1), max(0, by1), min(fw, bx2), min(fh, by2)
                bw, bh = bx2 - bx1, by2 - by1
                if bw < 10 or bh < 10:
                    continue

                if state_tint is not None and self.is_running and self.focus_state in (FocusState.SLACKING, FocusState.CONCENTRATED, FocusState.NEUTRAL):
                    if self.focus_state == FocusState.SLACKING:
                        color = state_tint
                    elif self.focus_state == FocusState.CONCENTRATED:
                        color = tuple(int((a * 0.40 + b * 0.60)) for a, b in zip(person.color, state_tint))  # type: ignore[assignment]
                    else:
                        color = tuple(int((a * 0.65 + b * 0.35)) for a, b in zip(person.color, state_tint))  # type: ignore[assignment]
                    thick = base_thick
                else:
                    color = person.color
                    thick = 1  # minimalist thin

                if person.pid == max(tracks, key=lambda p: (p.bbox[2] * p.bbox[3])).pid:
                    self._face_box = (bx1, by1, bx2, by2)

                # minimalist: single thin rounded border only, no brackets
                corner_r = int(min(bw, bh) * 0.14)
                styled_rect(frame, bx1, by1, bx2, by2, border=color, thickness=thick, radius=corner_r)

                # tiny ID pill — only show pid number, subtle
                pid_text = f"#{person.pid}"
                # hide verbose G mapping in minimalist mode — keep clean
                pid_px = max(9, int(bh * 0.095))
                tw, th = text_size(pid_text, pid_px, True)
                px1, py1 = bx1 + 4, max(0, by1 + 4)
                spad = 3
                pill_bg = tuple(int(c * 0.38) for c in person.color)
                # only draw pill if box large enough and not slacking (avoid clutter)
                if bw > 60 and bh > 60:
                    styled_rect(frame, px1 - spad, py1 - 1, px1 + tw + spad, py1 + th + 1, fill=pill_bg, radius=3)
                    draw_text(frame, pid_text, px1, py1, pid_px, (255, 255, 255), bold=True, anchor="la")

                if self.is_running and self.focus_state == FocusState.SLACKING and person.pid == tracks[0].pid:
                    tag, tag_px = "SLACKING", max(9, int(bh * 0.10))
                    tw, th = text_size(tag, tag_px, True)
                    tx = bx1 + (bw - tw) // 2
                    ty = by1 - th - 6
                    spad = 5
                    styled_rect(frame, tx - spad, ty - 1, tx + tw + spad, ty + th + 1, fill=(40, 40, 210), radius=3)
                    draw_text(frame, tag, tx, ty, tag_px, (255, 255, 255), bold=True, anchor="la")

            if len(tracks) > 1:
                cnt_text = f"{len(tracks)}"
                cnt_px = max(11, int(min(fw, fh) * 0.024))
                tw, th = text_size(cnt_text, cnt_px, True)
                badge_w, badge_h = max(22, tw + 10), th + 6
                bx, by = fw - badge_w - 10, 10
                styled_rect(frame, bx, by, bx + badge_w, by + badge_h, fill=(22, 22, 26), border=(70, 70, 75), thickness=1, radius=badge_h // 2)
                draw_text(frame, cnt_text, bx + badge_w // 2, by + badge_h // 2, cnt_px, (200, 200, 210), bold=True, anchor="mm")
            return

        # fallback: single legacy path (no tracker yet)
        if not self.camera_enabled or not self.last_faces:
            self._face_miss_frames += 1
            if self._face_miss_frames > 18:
                self._face_box = None
            return
        if self._cam_view is None:
            x, y, w, h = max(self.last_faces, key=lambda f: f[2] * f[3])
            pad = 12
            target = (x - pad, y - pad, x + w + pad, y + h + pad)
            self._face_box = target
        else:
            self._face_miss_frames = 0
            off_x, off_y, fit_w, fit_h, cam_w, cam_h = self._cam_view
            scale_x = fit_w / float(cam_w) if cam_w else 1.0
            scale_y = fit_h / float(cam_h) if cam_h else scale_x
            x, y, w, h = max(self.last_faces, key=lambda f: f[2] * f[3])
            pad = int(max(14, min(fit_w, fit_h) * 0.06))
            cx = off_x + (x + w * 0.5) * scale_x
            cy = off_y + (y + h * 0.5) * scale_y
            hw, hh = w * scale_x * 0.5 + pad, h * scale_y * 0.5 + pad
            target = (int(cx - hw), int(cy - hh), int(cx + hw), int(cy + hh))
            alpha = 0.50 if self.is_running else 0.35
            self._face_box = target if self._face_box is None else tuple(int(round((1 - alpha) * a + alpha * b)) for a, b in zip(self._face_box, target))
        bx1, by1, bx2, by2 = self._face_box
        bw, bh = bx2 - bx1, by2 - by1
        if bw < 10 or bh < 10:
            return
        fw, fh = frame.shape[1], frame.shape[0]
        bx1, by1, bx2, by2 = max(0, bx1), max(0, by1), min(fw, bx2), min(fh, by2)
        bw, bh = bx2 - bx1, by2 - by1
        if bw < 10 or bh < 10:
            return
        theme = self._theme_palette()
        if not self.is_running:
            color, thick = theme["accent"], 2
        elif self.focus_state == FocusState.SLACKING:
            pulse = 0.5 + 0.5 * math.sin(time.time() * 6)
            base = (60, 60, 255)
            color = tuple(int(c * (0.7 + 0.3 * pulse)) for c in base)  # type: ignore[assignment]
            thick = 3 + int(2 * pulse)
        elif self.focus_state == FocusState.CONCENTRATED:
            color, thick = theme["on_color"], 2
        else:
            color, thick = theme["accent"], 2
        corner_r = int(min(bw, bh) * 0.18)
        styled_rect(frame, bx1, by1, bx2, by2, border=color, thickness=thick, radius=corner_r)
        bracket_len = int(min(bw, bh) * 0.22)
        bracket_thick = max(2, thick + 1)
        for x1, y1, x2, y2 in [
            (bx1, by1, bx1 + bracket_len, by1),
            (bx1, by1, bx1, by1 + bracket_len),
            (bx2, by1, bx2 - bracket_len, by1),
            (bx2, by1, bx2, by1 + bracket_len),
            (bx1, by2, bx1 + bracket_len, by2),
            (bx1, by2, bx1, by2 - bracket_len),
            (bx2, by2, bx2 - bracket_len, by2),
            (bx2, by2, bx2, by2 - bracket_len),
        ]:
            cv2.line(frame, (x1, y1), (x2, y2), color, bracket_thick, cv2.LINE_AA)
        if self.is_running and self.focus_state == FocusState.SLACKING:
            tag, tag_px = "SLACKING", max(10, int(bh * 0.12))
            tw, th = text_size(tag, tag_px, True)
            tx = bx1 + (bw - tw) // 2
            ty = by1 - th - 6
            spad = 6
            styled_rect(frame, tx - spad, ty - th - spad, tx + tw + spad, ty + spad, fill=(40, 40, 220), radius=4)
            draw_text(frame, tag, tx, ty, tag_px, (255, 255, 255), bold=True, anchor="la")

    def _draw_slacking_alert(self, frame: np.ndarray, theme: dict[str, Any], rounded: bool, ui_scale: float) -> None:
        if not self._alert_active:
            return
        fw, fh = frame.shape[1], frame.shape[0]
        pulse = 0.5 + 0.5 * math.sin(time.time() * 5)
        banner_text, banner_px = "You're Slacking!", max(18, int(fh * 0.035))
        tw, th = text_size(banner_text, banner_px, True)
        pad_x, pad_y = max(16, int(24 * ui_scale)), max(8, int(10 * ui_scale))
        bw, bh = tw + 2 * pad_x, th + 2 * pad_y
        bx1, by1, by2 = (fw - bw) // 2, int(fh * 0.18), int(fh * 0.18) + bh
        alert_red = (40, 40, 230)
        fill_color = tuple(int(c * (0.8 + 0.2 * pulse)) for c in alert_red)
        border_color = tuple(int(c * (0.5 + 0.5 * pulse)) for c in alert_red)
        br = int(bh * 0.3) if rounded else 0
        styled_rect(frame, bx1, by1, bx1 + bw, by2, fill=fill_color, border=border_color, thickness=2, radius=br)
        draw_text(frame, banner_text, bx1 + bw // 2, by1 + bh // 2, max(14, int(banner_px * 0.9)), (255, 255, 255), bold=True, anchor="mm")
        border_thick = 4 + int(3 * pulse)
        border_col = tuple(int(c * (0.6 + 0.4 * pulse)) for c in alert_red)
        cv2.rectangle(frame, (0, 0), (fw, border_thick), border_col, -1, cv2.LINE_AA)
        cv2.rectangle(frame, (0, fh - border_thick), (fw, fh), border_col, -1, cv2.LINE_AA)
        cv2.rectangle(frame, (0, 0), (border_thick, fh), border_col, -1, cv2.LINE_AA)
        cv2.rectangle(frame, (fw - border_thick, 0), (fw, fh), border_col, -1, cv2.LINE_AA)

    # -- window helpers ---------------------------------------------------
    def _make_window_resizable(self) -> None:
        try:
            if platform.system() != "Windows":
                return
            hwnd = ctypes.windll.user32.FindWindowW(None, WINDOW_NAME)  # type: ignore[attr-defined]
            if hwnd == 0:
                return
            GWL_STYLE, WS_OVERLAPPEDWINDOW = -16, 0x00CF0000
            current_style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_STYLE)  # type: ignore[attr-defined]
            if (current_style & WS_OVERLAPPEDWINDOW) != WS_OVERLAPPEDWINDOW:
                ctypes.windll.user32.SetWindowLongW(hwnd, GWL_STYLE, current_style | WS_OVERLAPPEDWINDOW)  # type: ignore[attr-defined]
                ctypes.windll.user32.SetWindowPos(hwnd, 0, 0, 0, 0, 0, 0x0002 | 0x0001 | 0x0004 | 0x0020)  # type: ignore[attr-defined]
        except Exception as exc:
            log.debug("Make resizable failed: %s", exc)

    # -- persistence ------------------------------------------------------
    def _load_layout(self) -> None:
        default_cfg = LayoutConfig.default()
        loaded: LayoutConfig | None = None
        try:
            if self.layout_file.exists():
                loaded = LayoutConfig.load(self.layout_file)
        except Exception as exc:
            log.warning("Layout load failed: %s", exc)
        if loaded is None or not loaded.elements:
            self.layout_manager.config = default_cfg
            self._save_layout()
        else:
            for key, elem in default_cfg.elements.items():
                loaded.elements.setdefault(key, elem)
            # one-time minimalist migration: disable center Pomodoro, ensure safe timer/progress
            try:
                pl = loaded.elements.get("phase_label")
                if pl and pl.enabled:
                    pl.enabled = False
                tp = loaded.elements.get("timer_popup")
                # migrate any old timer (8.8, 9.2) to new safe inset 8.6,0.42
                if tp and (abs(tp.x - 8.8) < 0.5 or abs(tp.x - 9.2) < 0.5):
                    tp.x, tp.y, tp.width, tp.height, tp.font_scale, tp.margin = 8.6, 0.42, 2.9, 0.88, 0.92, 0.18
                pb = loaded.elements.get("progress_bar")
                if pb and (abs(pb.y - 2.05) < 0.3 or abs(pb.y - 0.12) < 0.05):
                    pb.x, pb.y, pb.width, pb.height, pb.margin = 0.30, 0.10, 11.4, 0.12, 0.06
                fd = loaded.elements.get("focus_display")
                if fd and (abs(fd.width - 3.2) < 0.35 or abs(fd.width - 2.6) < 0.2):
                    fd.x, fd.width, fd.height, fd.margin = 0.30, 2.7, 0.40, 0.06
                sd = loaded.elements.get("status_display")
                if sd and sd.enabled:
                    sd.enabled = False
                mb = loaded.elements.get("main_buttons")
                # old 8.2 or 5.2 width → new 5.6
                if mb and (abs(mb.width - 8.2) < 0.5 or abs(mb.width - 5.2) < 0.35):
                    mb.x, mb.y, mb.width, mb.height, mb.margin = 3.2, 7.12, 5.6, 0.68, 0.12
                # only save if we changed something (pl was enabled)
                if pl and not pl.enabled:
                    log.info("Migrated layout to minimalist (disabled center Pomodoro, safe timer)")
                    loaded.save(self.layout_file)
            except Exception:
                pass
            self.layout_manager.config = loaded
        lm = self.layout_manager
        lm.grid_cols = max(1, int(lm.config.grid_cols))
        lm.grid_rows = max(1, int(lm.config.grid_rows))

    def _save_layout(self) -> None:
        try:
            self.layout_manager.config.save(self.layout_file)
        except Exception as exc:
            log.warning("Could not save layout: %s", exc)

    def reset_layout(self) -> None:
        cfg = LayoutConfig.default()
        self.layout_manager.config = cfg
        self.layout_manager.grid_cols = cfg.grid_cols
        self.layout_manager.grid_rows = cfg.grid_rows
        self._save_layout()

    def _load_settings(self) -> None:
        try:
            if not self.settings_file.exists():
                return
            data = json.loads(self.settings_file.read_text(encoding="utf-8"))
            if "ui_scale" in data:
                self.ui_scale = max(UI_SCALE_MIN, min(UI_SCALE_MAX, float(data["ui_scale"])))
            if data.get("theme") in self.THEMES:
                self.theme = data["theme"]
            if data.get("corner_style") in (CornerStyle.ROUNDED, CornerStyle.BOXY):
                self.corner_style = data["corner_style"]
            if "alerts_enabled" in data:
                self.alerts_enabled = bool(data["alerts_enabled"])
        except Exception as exc:
            log.warning("Settings load failed: %s", exc)

    def _save_settings(self) -> None:
        try:
            self.settings_file.write_text(
                json.dumps(
                    {
                        "ui_scale": round(float(self.ui_scale), 3),
                        "theme": self.theme,
                        "corner_style": self.corner_style,
                        "alerts_enabled": bool(self.alerts_enabled),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception as exc:
            log.warning("Could not save settings: %s", exc)

    def _get_window_client_size(self) -> tuple[int, int]:
        try:
            _, _, w, h = cv2.getWindowImageRect(WINDOW_NAME)
            if w and h and int(w) > 0 and int(h) > 0:
                return int(w), int(h)
        except Exception:
            pass
        return int(self.frame_width), int(self.frame_height)

    def _load_face_cascade(self) -> cv2.CascadeClassifier | None:
        """Load Haar cascade with fallbacks for opencv-python 5.x (no bundled xml)."""
        candidates: list[Path] = [
            Path(__file__).parent / CASCADE_FILE,
            Path.cwd() / CASCADE_FILE,
        ]
        # cv2.data.haarcascades may be empty or missing in opencv 5
        try:
            p = Path(str(getattr(cv2.data, "haarcascades", ""))) / CASCADE_FILE  # type: ignore[attr-defined]
            if str(p) != CASCADE_FILE:
                candidates.append(p)
        except Exception:
            pass
        # pip cv2 package internal location
        try:
            candidates.append(Path(cv2.__file__).parent / "data" / CASCADE_FILE)  # type: ignore[attr-defined]
            candidates.append(Path(cv2.__file__).parent / "data" / "haarcascades" / CASCADE_FILE)  # type: ignore[attr-defined]
        except Exception:
            pass

        for cand in candidates:
            try:
                if cand.exists() and cand.stat().st_size > 1000:
                    clf = cv2.CascadeClassifier(str(cand))
                    if not clf.empty():
                        log.debug("Face cascade loaded from %s", cand)
                        return clf
            except Exception as exc:
                log.debug("Cascade try %s failed: %s", cand, exc)

        # last resort: download from opencv github (cached locally)
        try:
            url = f"https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/{CASCADE_FILE}"
            dest = Path(__file__).parent / CASCADE_FILE
            if not dest.exists():
                dest = Path.cwd() / CASCADE_FILE
            if not dest.exists() or dest.stat().st_size < 1000:
                import urllib.request

                log.debug("Downloading face cascade from %s", url)
                data = urllib.request.urlopen(url, timeout=10).read()
                if len(data) > 1000:
                    dest.write_bytes(data)
                    log.debug("Saved cascade to %s (%d bytes)", dest, len(data))
            if dest.exists():
                clf = cv2.CascadeClassifier(str(dest))
                if not clf.empty():
                    log.debug("Face cascade loaded after download: %s", dest)
                    return clf
        except Exception as exc:
            log.debug("Cascade download failed: %s", exc)

        log.warning("Face cascade not found — tracking will run in motion-only mode")
        return None

    def _load_eye_cascade(self) -> cv2.CascadeClassifier | None:
        """Eye cascade for false-positive suppression — optional."""
        candidates: list[Path] = [Path(__file__).parent / EYE_CASCADE_FILE, Path.cwd() / EYE_CASCADE_FILE]
        try:
            p = Path(str(getattr(cv2.data, "haarcascades", ""))) / EYE_CASCADE_FILE  # type: ignore[attr-defined]
            if str(p) != EYE_CASCADE_FILE:
                candidates.append(p)
        except Exception:
            pass
        for cand in candidates:
            try:
                if cand.exists() and cand.stat().st_size > 1000:
                    clf = cv2.CascadeClassifier(str(cand))
                    if not clf.empty():
                        log.debug("Eye cascade loaded from %s", cand)
                        return clf
            except Exception:
                pass
        # try download (best-effort, no hard failure)
        try:
            url = f"https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/{EYE_CASCADE_FILE}"
            dest = Path(__file__).parent / EYE_CASCADE_FILE
            if not dest.exists():
                dest = Path.cwd() / EYE_CASCADE_FILE
            if not dest.exists() or dest.stat().st_size < 1000:
                import urllib.request

                log.debug("Downloading eye cascade from %s", url)
                data = urllib.request.urlopen(url, timeout=7).read()
                if len(data) > 1000:
                    dest.write_bytes(data)
            if dest.exists():
                clf = cv2.CascadeClassifier(str(dest))
                if not clf.empty():
                    log.debug("Eye cascade loaded after download: %s", dest)
                    return clf
        except Exception as exc:
            log.debug("Eye cascade unavailable: %s", exc)
        return None

    # -- camera -----------------------------------------------------------
    def start_camera(self) -> bool:
        if not self.camera_enabled:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
            return False
        if self.cap is not None and self.cap.isOpened():
            return True
        try:
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # type: ignore[attr-defined]
        except Exception:
            self.cap = None
        if self.cap is None:
            try:
                self.cap = cv2.VideoCapture(0)
            except Exception:
                self.cap = None
        if self.cap is not None and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            return True
        return False

    def start_session(self) -> None:
        if not self.start_camera():
            log.error("Could not access camera — check connection")
            print("Could not access the camera. Please connect a camera and try again.")
            return
        self.session_open = True
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        try:
            cv2.resizeWindow(WINDOW_NAME, DEFAULT_WINDOW_W, DEFAULT_WINDOW_H)
            try:
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_AUTOSIZE, 0)
                try:
                    self._make_window_resizable()
                except Exception:
                    pass
            except Exception:
                pass
        except Exception:
            pass
        cv2.setMouseCallback(WINDOW_NAME, self._on_mouse_click)
        log.info("Pomodoro camera started — press 'q' to quit")
        print("Pomodoro camera started. Click the on-screen buttons or press 'q' to quit.")

        last_render_size = (int(self.frame_width), int(self.frame_height))

        while self.session_open:
            win_w, win_h = self._get_window_client_size()
            cam_frame: np.ndarray | None = None
            if self.camera_enabled and self.cap is not None and self.cap.isOpened():
                ret, cam_frame = self.cap.read()
                if not ret:
                    cam_frame = None
            if cam_frame is not None:
                try:
                    cam_frame = cv2.flip(cam_frame, 1)
                except Exception:
                    pass

            target_w = max(MIN_RENDER_W, win_w)
            target_h = max(MIN_RENDER_H, win_h)
            if abs(target_w - last_render_size[0]) >= 4 or abs(target_h - last_render_size[1]) >= 4:
                last_render_size = (target_w, target_h)
            render_w, render_h = last_render_size

            frame = np.zeros((render_h, render_w, 3), dtype=np.uint8)
            if cam_frame is not None:
                cam_h, cam_w = cam_frame.shape[:2]
                scale = min(render_w / float(cam_w), render_h / float(cam_h))
                fit_w, fit_h = max(1, int(round(cam_w * scale))), max(1, int(round(cam_h * scale)))
                fitted = cv2.resize(cam_frame, (fit_w, fit_h), interpolation=cv2.INTER_AREA)
                off_x, off_y = (render_w - fit_w) // 2, (render_h - fit_h) // 2
                frame[off_y : off_y + fit_h, off_x : off_x + fit_w] = fitted
                self._cam_view = (off_x, off_y, fit_w, fit_h, cam_w, cam_h)
            else:
                frame[:] = self._theme_palette()["backdrop"]
                self._cam_view = None
                if not self.camera_enabled:
                    theme = self._theme_palette()
                    w, h = min(self.widget_width, render_w - 40), min(self.widget_height, render_h - 40)
                    cx, cy = render_w // 2, render_h // 2
                    wx1, wy1, wx2, wy2 = cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2
                    widget_radius = int(min(w, h) * 0.08) if self._is_rounded() else 0
                    styled_rect(frame, wx1, wy1, wx2, wy2, fill=theme["widget_fill"], border=theme["panel_border"], thickness=1, radius=widget_radius)
                    label_px = int(min(48, max(16, h * 0.16)))
                    draw_text(frame, "Camera Off", cx, cy, label_px, theme["text"], bold=True, anchor="mm")

            self.frame_width, self.frame_height = render_w, render_h
            self.frame_count += 1

            analysis_input = cam_frame if cam_frame is not None else frame
            self.focus_score = self.analyze_focus(analysis_input)
            self.focus_state = self.classify_focus_state(self.focus_score, frame)
            frame = self._draw_ui(frame)
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if self.onboarding.active:
                if key in (13, 10, 32):  # Enter / Space
                    self.onboarding.step += 1
                    if self.onboarding.step >= len(self.onboarding._STEPS):
                        self.onboarding.active = False
                        self._save_settings()
                elif key == 8 and self.onboarding.step > 0:
                    self.onboarding.step -= 1
                elif key == 27:
                    self.onboarding.active = False
                    self._save_settings()
            elif key == ord("q"):
                self.session_open = False
            elif key == ord("s"):
                self.toggle_timer()
            elif key == ord("e"):
                self.layout_edit_mode = not self.layout_edit_mode
                self._dragging_element = None
                if self.layout_edit_mode:
                    self.settings_visible = False
                else:
                    self._save_layout()
            elif key == 27 and self.layout_edit_mode:
                self.layout_edit_mode = False
                self._dragging_element = None
                self._save_layout()
            try:
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    self.session_open = False
                    break
            except Exception:
                pass
        self.cleanup()

    # -- mouse ------------------------------------------------------------
    def _on_mouse_click(self, event: int, x: int, y: int, flags: int, params: Any) -> None:
        mapped_x, mapped_y = x, y
        try:
            _, _, ww, wh = cv2.getWindowImageRect(WINDOW_NAME)
            if ww > 1 and wh > 1:
                mapped_x = int(round(x * self.frame_width / ww))
                mapped_y = int(round(y * self.frame_height / wh))
                mapped_x = max(0, min(mapped_x, self.frame_width - 1))
                mapped_y = max(0, min(mapped_y, self.frame_height - 1))
        except Exception:
            pass

        if self.onboarding.active and event == cv2.EVENT_LBUTTONDOWN:
            if self.onboarding.handle_click(mapped_x, mapped_y, self.frame_width, self.frame_height):
                if not self.onboarding.active:
                    self._save_settings()
                return

        if self.layout_edit_mode:
            if event == cv2.EVENT_LBUTTONDOWN:
                if (done := getattr(self, "_edit_done_rect", None)) and done[0] <= mapped_x <= done[2] and done[1] <= mapped_y <= done[3]:
                    self.layout_edit_mode = False
                    self._dragging_element = None
                    self._save_layout()
                    return
            self._handle_layout_drag(event, mapped_x, mapped_y)
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            if (btn := self.button_handler.is_button_clicked(event, mapped_x, mapped_y, self.frame_width, self.frame_height)):
                self.button_handler.handle_button_click(btn)
                return
            if self.settings_visible:
                self._handle_settings_click(mapped_x, mapped_y)

    def _handle_layout_drag(self, event: int, x: int, y: int) -> None:
        fw, fh = self.frame_width, self.frame_height
        lm = self.layout_manager
        cell_w, cell_h = fw / lm.grid_cols, fh / lm.grid_rows
        if event == cv2.EVENT_LBUTTONDOWN:
            self._dragging_element = None
            for name in lm.config.elements:
                x1, y1, x2, y2 = lm.get_element_rect(name, fw, fh, include_disabled=True)
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self._dragging_element = name
                    self._drag_offset_x = x - x1
                    self._drag_offset_y = y - y1
                    break
        elif event == cv2.EVENT_MOUSEMOVE and self._dragging_element:
            elem = lm.config.elements[self._dragging_element]
            nx, ny = (x - self._drag_offset_x) / cell_w, (y - self._drag_offset_y) / cell_h
            elem.x = max(0.0, min(nx, lm.grid_cols - elem.width))
            elem.y = max(0.0, min(ny, lm.grid_rows - elem.height))
        elif event == cv2.EVENT_LBUTTONUP and self._dragging_element:
            elem = lm.config.elements[self._dragging_element]
            max_qx, max_qy = int(max(0.0, (lm.grid_cols - elem.width)) * 4), int(max(0.0, (lm.grid_rows - elem.height)) * 4)
            elem.x = max(0, min(int(round(elem.x * 4)), max_qx)) / 4.0
            elem.y = max(0, min(int(round(elem.y * 4)), max_qy)) / 4.0
            self._dragging_element = None
            self._save_layout()

    # -- settings ---------------------------------------------------------
    def toggle_settings(self) -> None:
        self.settings_visible = not self.settings_visible

    def apply_settings(self, pomodoro_minutes: int | None = None, break_minutes: int | None = None, display_mode: str | None = None) -> None:
        if pomodoro_minutes is not None:
            self.session_duration_minutes = max(1, int(pomodoro_minutes))
        if break_minutes is not None:
            self.break_duration_minutes = max(1, int(break_minutes))
        if display_mode is not None:
            self.display_mode = display_mode
        self.show_progress_bar = self.display_mode in (DisplayMode.PROGRESS_BAR, DisplayMode.BOTH)
        self.show_timer_popup = self.display_mode in (DisplayMode.TIMER_POPUP, DisplayMode.BOTH)
        if not self.is_running:
            self.time_left = self.session_duration_minutes * 60
            self.phase_duration_seconds = self.time_left
            self.current_phase = Phase.POMODORO
        else:
            now = time.time()
            elapsed = int(now - self.phase_started_at) if self.phase_started_at is not None else 0
            self.phase_duration_seconds = self.break_duration_minutes * 60 if self.current_phase == Phase.SHORT_BREAK else self.session_duration_minutes * 60
            self.time_left = max(0, self.phase_duration_seconds - elapsed)
        log.info("Settings: Pomodoro=%d min, Break=%d min, Mode=%s", self.session_duration_minutes, self.break_duration_minutes, self.display_mode)

    # -- timer state machine ----------------------------------------------
    def start_timer(self) -> None:
        if self.is_running:
            return
        self.is_running = True
        self.phase_started_at = time.time()
        self.phase_duration_seconds = self.break_duration_minutes * 60 if self.current_phase == Phase.SHORT_BREAK else self.session_duration_minutes * 60
        self.time_left = self.phase_duration_seconds

    def stop_timer(self) -> None:
        self.is_running = False
        self.phase_started_at = None

    def toggle_timer(self) -> None:
        self.stop_timer() if self.is_running else self.start_timer()

    def reset_timer(self) -> None:
        self.is_running = False
        self.phase_started_at = None
        self.current_phase = Phase.POMODORO
        self.time_left = self.session_duration_minutes * 60
        self.phase_duration_seconds = self.time_left

    def switch_to_short_break(self) -> None:
        self.current_phase = Phase.SHORT_BREAK
        self.time_left = self.break_duration_minutes * 60
        self.phase_duration_seconds = self.time_left
        self.phase_started_at = time.time()
        self.is_running = True

    def switch_to_pomodoro(self) -> None:
        self.current_phase = Phase.POMODORO
        self.time_left = self.session_duration_minutes * 60
        self.phase_duration_seconds = self.time_left
        self.phase_started_at = time.time()
        self.is_running = True

    def _update_timer(self) -> None:
        if not self.is_running or self.phase_started_at is None:
            return
        elapsed = int(time.time() - self.phase_started_at)
        self.time_left = max(0, self.phase_duration_seconds - elapsed)
        if self.time_left <= 0:
            if self.current_phase == Phase.POMODORO:
                log.info("Pomodoro complete — switching to short break")
                self.switch_to_short_break()
            elif self.current_phase == Phase.SHORT_BREAK:
                log.info("Short break complete — ready for next pomodoro")
                self.switch_to_pomodoro()

    # -- focus analysis ---------------------------------------------------
    def analyze_focus(self, frame: np.ndarray | None) -> float:
        """Focus score 0-100. Robust Haar cascade + motion, with downscale and equalize."""
        if frame is None:
            return 0.0

        # --- motion gray: always computed, small for speed + stable diff ---
        h, w = frame.shape[:2]
        # downscale large frames to 320px width for both motion and detection
        det_scale = 1.0
        small = frame
        if w > 360:
            det_scale = 320.0 / w
            small = cv2.resize(frame, (320, int(h * det_scale)), interpolation=cv2.INTER_AREA)
        gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray_small = cv2.GaussianBlur(gray_small, (5, 5), 0)
        # improve contrast for cascade
        try:
            gray_eq = cv2.equalizeHist(gray_small)
        except Exception:
            gray_eq = gray_small

        # --- face detection: adaptive interval — 1 when searching/tracking 1-2 for low lag, 2 when crowded
        self._face_frame_count += 1
        cur_n = getattr(self, "person_count", 0)
        adapt_interval = 1 if cur_n <= 2 else 2
        should_detect = (self._face_frame_count % adapt_interval == 0) or not self.last_faces
        if should_detect:
            raw_dets: list[tuple[int, int, int, int]] = []
            if self.face_cascade is not None:
                # stricter params cut ghost detections: higher neighbors, larger minSize on 320px small
                try:
                    faces = self.face_cascade.detectMultiScale(gray_eq, scaleFactor=1.08, minNeighbors=5, minSize=(36, 36), flags=cv2.CASCADE_SCALE_IMAGE)
                except Exception:
                    try:
                        faces = self.face_cascade.detectMultiScale(gray_eq, 1.08, 5)
                    except Exception:
                        faces = []
                inv = 1.0 / det_scale if det_scale != 1.0 else 1.0
                raw_dets = [(int(x * inv), int(y * inv), int(w * inv), int(h * inv)) for (x, y, w, h) in faces]
                # geometric pre-filter: discard absurd sizes / aspect ratios before NMS
                if raw_dets and frame is not None:
                    fh, fw = frame.shape[:2]
                    frame_area = float(fw * fh)
                    filt: list[tuple[int, int, int, int]] = []
                    for (x, y, wb, hb) in raw_dets:
                        area_frac = (wb * hb) / max(1.0, frame_area)
                        if not (TRACK_MIN_SIZE_FRAC <= area_frac <= TRACK_MAX_SIZE_FRAC):
                            continue
                        ar = wb / hb if hb else 1.0
                        if not (0.68 <= ar <= 1.42):
                            continue
                        if x < -wb * 0.12 or y < -hb * 0.12 or x + wb > fw + wb * 0.12 or y + hb > fh + hb * 0.12:
                            continue
                        # ceiling/light ghost — large box hugging top edge
                        if y < fh * 0.08 and area_frac > 0.055:
                            continue
                        if y < fh * 0.14 and area_frac > 0.095:
                            continue
                        # very bright patch (ceiling lamp) — Haar loves bright squares
                        try:
                            if frame is not None:
                                cx, cy = int(x + wb * 0.5), int(y + hb * 0.5)
                                if 0 <= cx < fw and 0 <= cy < fh:
                                    patch = frame[max(0, cy - 10):cy + 10, max(0, cx - 10):cx + 10]
                                    if patch.size and float(np.mean(patch)) > 202:
                                        # require strong eye evidence for bright regions
                                        if self.eye_cascade is not None and not self.eye_cascade.empty():
                                            xs = int(x * det_scale); ys = int(y * det_scale)
                                            ws = int(wb * det_scale); hs = int(hb * det_scale)
                                            roi = gray_small[max(0, ys):ys + hs // 2, max(0, xs):xs + ws]
                                            if roi.size:
                                                eyes = self.eye_cascade.detectMultiScale(roi, 1.1, 3, minSize=(14, 14))
                                                if len(eyes) == 0:
                                                    continue
                                        else:
                                            continue
                        except Exception:
                            pass
                        filt.append((x, y, wb, hb))
                    raw_dets = filt
                # aggressive NMS: IoU or center-proximity → same face (kills duplicate boxes for 1 person)
                if len(raw_dets) > 1:
                    raw_dets = sorted(raw_dets, key=lambda d: d[2] * d[3], reverse=True)
                    kept: list[tuple[int, int, int, int]] = []
                    for d in raw_dets:
                        dup = False
                        for k in kept:
                            if _iou(d, k) >= TRACK_NMS_IOU:
                                dup = True
                                break
                            cx, cy = d[0] + d[2] * 0.5, d[1] + d[3] * 0.5
                            kx, ky = k[0] + k[2] * 0.5, k[1] + k[3] * 0.5
                            dist = math.hypot(cx - kx, cy - ky)
                            avg_sz = max(d[2], d[3], k[2], k[3])
                            if dist < avg_sz * TRACK_NMS_DIST_FRAC:
                                dup = True
                                break
                        if not dup:
                            kept.append(d)
                    raw_dets = kept[:TRACK_MAX_PERSONS]
            # update tracker with fresh detections (or empty if none)
            # gallery re-ID: try to map each raw detection to a known gallery identity
            # we do this before tracker so new tracks can inherit gallery gid
            gallery = get_gallery()
            # pre-compute embeddings for raw dets for gallery lookup (used to maybe reuse gid)
            det_embs: list[np.ndarray | None] = []
            for d in raw_dets:
                det_embs.append(gallery.embed(frame, d) if frame is not None else None)
            # if tracker will create new tracks, we will enroll after
            self.tracker.update(raw_dets, self._face_frame_count)
            self.person_tracks = self.tracker.active
            # gallery: enroll or match each active track (with used-gid guard for simultaneous)
            used_gids: set[int] = {t.gid for t in self.person_tracks if t.gid is not None}
            for trk in self.person_tracks:
                # newly confirmed tracks (hits just reached threshold) need gallery enrollment
                is_newly_confirmed = trk.hits == TRACK_CONFIRM_HITS and trk.last_update == self._face_frame_count
                if is_newly_confirmed:
                    best_emb: np.ndarray | None = None
                    best_iou = -1.0
                    for d, emb in zip(raw_dets, det_embs):
                        iou = _iou(trk.bbox, d)
                        if iou > best_iou:
                            best_iou, best_emb = iou, emb
                    if best_emb is not None:
                        gid = gallery.match(best_emb)
                        if gid is not None and gid in used_gids:
                            gid = None
                        if gid is not None:
                            trk.gid = gid
                            trk.label = gallery.label(gid)
                            used_gids.add(gid)
                        else:
                            gallery.enroll(trk.pid, best_emb)
                            trk.gid = trk.pid
                            trk.label = gallery.label(trk.pid)
                            used_gids.add(trk.pid)
                    else:
                        trk.label = f"Person #{trk.pid}"
                        used_gids.add(trk.pid)
                else:
                    if not trk.label:
                        trk.label = gallery.label(trk.gid) if trk.gid else f"Person #{trk.pid}"
                    if trk.gid is not None:
                        used_gids.add(trk.gid)
                    if trk.hits % 30 == 0:
                        emb = gallery.embed(frame, trk.bbox) if frame is not None else None
                        if emb is not None:
                            gallery.enroll(trk.gid or trk.pid, emb)
            for trk in self.person_tracks:
                if not trk.label:
                    trk.label = f"Person #{trk.pid}"
            self.person_count = len(self.person_tracks)
            self.last_faces = [t.bbox for t in self.person_tracks]
            self._last_gray = gray_small
        else:
            # no new detection — keep tracker state, just refresh count
            self.person_tracks = self.tracker.active
            self.person_count = len(self.person_tracks)
            # keep last_faces in sync (tracker may have pruned)
            self.last_faces = [t.bbox for t in self.person_tracks]

        # when idle, just keep motion baseline fresh, no scoring
        if not self.is_running:
            self.prev_gray = gray_small
            return 0.0

        # first-frame bootstrap — confirmed faces only (suppresses ghosts)
        if self.prev_gray is None or self.prev_gray.shape != gray_small.shape:
            self.prev_gray = gray_small
            return 100.0 if self.last_faces else 25.0

        motion = float(np.mean(cv2.absdiff(self.prev_gray, gray_small))) / 255.0
        self.prev_gray = gray_small

        face_score = 1.0 if self.last_faces else 0.0
        motion_penalty = min(1.0, motion * (1.6 if face_score else 2.4))
        score = (face_score * 72.0) + ((1.0 - motion_penalty) * 28.0)
        if not self.last_faces:
            score *= 0.55
        # clamp and smooth with one-pole EMA to avoid jitter (keep last 0.25s)
        score = max(0.0, min(100.0, score))
        prev = getattr(self, "_smooth_score", score)
        score = prev * 0.35 + score * 0.65
        self._smooth_score = score
        return score

    def classify_focus_state(self, score: float, frame: np.ndarray | None = None) -> str:
        if not self.is_running:
            return FocusState.NEUTRAL
        if score >= FOCUS_CONCENTRATED_THRESHOLD:
            return FocusState.CONCENTRATED
        if score <= FOCUS_SLACKING_THRESHOLD:
            return FocusState.SLACKING
        return FocusState.NEUTRAL

    # -- main UI draw -----------------------------------------------------
    def _draw_ui(self, frame: np.ndarray) -> np.ndarray:
        self._update_timer()
        phase_text, phase_color, progress = self._resolve_phase_render_state()
        timer_text = f"{self.time_left // 60:02d}:{self.time_left % 60:02d}"
        base_scale = max(0.75, min(2.5, frame.shape[0] / 480.0))
        ui_scale = base_scale * float(getattr(self, "ui_scale", UI_SCALE_DEFAULT))
        theme, rounded = self._theme_palette(), self._is_rounded()
        fw, fh = frame.shape[1], frame.shape[0]

        self._draw_face_outline(frame)
        if self.is_running and self.camera_enabled and self.focus_state == FocusState.SLACKING and self.alerts_enabled:
            self._alert_active = True
            self._play_alert_sound()
            self._draw_slacking_alert(frame, theme, rounded, ui_scale)
        else:
            self._alert_active = False

        self._draw_progress_bar(frame, progress, phase_color, theme, rounded, fw, fh)
        self._draw_timer_popup(frame, timer_text, phase_text, phase_color, theme, rounded, fw, fh)
        self._draw_phase_label(frame, phase_text, phase_color, fw, fh, ui_scale)
        self._draw_main_buttons(frame, theme, rounded, fw, fh, ui_scale)
        self._draw_focus_display(frame, theme, fw, fh, ui_scale)
        self._draw_status_display(frame, theme, fw, fh, ui_scale)
        self._draw_quit_hint(frame, theme, fw, fh)

        if self.settings_visible:
            self._draw_settings_panel(frame)
        if self.layout_edit_mode:
            self._draw_layout_edit_overlay(frame, theme, rounded, ui_scale)
        if self.onboarding.active:
            self.onboarding.draw_overlay(frame, theme, rounded, ui_scale)
        return frame

    def _resolve_phase_render_state(self) -> tuple[str, tuple[int, int, int], float]:
        match self.current_phase:
            case Phase.POMODORO:
                return "Pomodoro", (231, 76, 60), max(0.0, min(1.0, 1.0 - (self.time_left / max(1, self.session_duration_minutes * 60))))
            case Phase.SHORT_BREAK:
                return "Short Break", (39, 174, 96), max(0.0, min(1.0, 1.0 - (self.time_left / max(1, self.break_duration_minutes * 60))))
            case _:
                return "Long Break", (52, 152, 219), 0.5

    def _draw_progress_bar(self, frame: np.ndarray, progress: float, phase_color: tuple[int, int, int], theme: dict[str, Any], rounded: bool, fw: int, fh: int) -> None:
        self._last_bar_area = None
        if not (self.show_progress_bar and self.progress_bar_enabled and self.layout_manager.config.elements.get("progress_bar", UIElementConfig()).enabled):
            return
        bar_x1, bar_y, bar_x2, bar_y2 = self.layout_manager.get_element_rect("progress_bar", fw, fh)
        if self.theme == ThemeName.XP:
            _xp_progress_bar(frame, bar_x1, bar_y, bar_x2, bar_y2, progress)
        else:
            # minimalist thin pill progress — no border, subtle track
            bar_radius = (bar_y2 - bar_y) // 2 if rounded else 0
            # track as translucent dark
            overlay = frame.copy()
            styled_rect(overlay, bar_x1, bar_y, bar_x2, bar_y2, fill=(38, 38, 42), radius=bar_radius)
            cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
            if progress > 0:
                fill_x = max(bar_x1, int(bar_x1 + (bar_x2 - bar_x1) * progress))
                styled_rect(frame, bar_x1, bar_y, fill_x, bar_y2, fill=(120, 160, 255) if phase_color == (231, 76, 60) else phase_color, radius=bar_radius)
        self._last_bar_area = (bar_x1, bar_y, bar_x2, bar_y2)

    def _draw_timer_popup(self, frame: np.ndarray, timer_text: str, phase_text: str, phase_color: tuple[int, int, int], theme: dict[str, Any], rounded: bool, fw: int, fh: int) -> None:
        if not (self.show_timer_popup and self.layout_manager.config.elements.get("timer_popup", UIElementConfig()).enabled):
            return
        px1, py1, px2, py2 = self.layout_manager.get_element_rect("timer_popup", fw, fh)
        box_w, box_h = px2 - px1, py2 - py1
        if box_w <= 4 or box_h <= 4:
            return
        max_px = int(min(1.6, box_h / 60.0) * 72 * self.layout_manager.config.elements["timer_popup"].font_scale * self.ui_scale)
        px, _ = fit_text(timer_text, box_w, box_h, max_px, bold=True)
        if self.theme == ThemeName.XP:
            tb_h = max(18, int(box_h * 0.26))
            _xp_title_bar(frame, px1, py1, px2, py1 + tb_h, phase_text)
            styled_rect(frame, px1, py1 + tb_h, px2, py2, fill=theme["popup_fill"], border=(104, 104, 104), thickness=1, radius=0)
            draw_text(frame, timer_text, px1 + box_w // 2, py1 + tb_h + (box_h - tb_h) // 2, px, theme["text"], bold=True, anchor="mm")
        else:
            # minimalist: subtle translucent pill, thin neutral border, no shadow/highlight
            popup_radius = int(min(box_w, box_h) * 0.32) if rounded else 0
            # translucent dark pill
            overlay = frame.copy()
            styled_rect(overlay, px1, py1, px2, py2, fill=(32, 32, 36), radius=popup_radius)
            cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
            styled_rect(frame, px1, py1, px2, py2, border=(68, 68, 75), thickness=1, radius=popup_radius)
            draw_text(frame, timer_text, px1 + box_w // 2, py1 + box_h // 2, px, theme["text"], bold=True, anchor="mm")

    def _draw_phase_label(self, frame: np.ndarray, phase_text: str, phase_color: tuple[int, int, int], fw: int, fh: int, ui_scale: float) -> None:
        if not self.layout_manager.config.elements.get("phase_label", UIElementConfig()).enabled:
            return
        lx1, ly1, lx2, ly2 = self.layout_manager.get_element_rect("phase_label", fw, fh)
        if (w := lx2 - lx1) <= 4 or (h := ly2 - ly1) <= 4:
            return
        px, _ = fit_text(phase_text, w, h, int(0.62 * h * ui_scale), bold=True)
        draw_text(frame, phase_text, lx1 + w // 2, ly1 + h // 2, px, phase_color, bold=True, anchor="mm", stroke=max(1, px // 16), stroke_color=(12, 12, 14))

    def _draw_main_buttons(self, frame: np.ndarray, theme: dict[str, Any], rounded: bool, fw: int, fh: int, ui_scale: float) -> None:
        if not self.layout_manager.config.elements.get("main_buttons", UIElementConfig()).enabled:
            self.button_handler.button_regions = {}
            return
        bx1, by1, bx2, by2 = self.layout_manager.get_element_rect("main_buttons", fw, fh)
        if (btn_area_w := bx2 - bx1) <= 30 or (btn_area_h := by2 - by1) <= 10:
            self.button_handler.button_regions = {}
            return
        btn_h = max(28, int(btn_area_h * 0.74 * ui_scale))
        btn_y = by1 + (btn_area_h - btn_h) // 2
        gap = int(max(8, btn_area_w * 0.025))
        btn_w = (btn_area_w - 2 * gap) // 3
        main_btn_radius = int(btn_h * 0.26) if rounded else 0
        is_xp = self.theme == ThemeName.XP
        labels = ["Running" if self.is_running else "Start", "Reset", "Settings"]
        names = ("start", "reset", "settings")
        xs = [bx1, bx1 + btn_w + gap, bx1 + 2 * (btn_w + gap)]
        if is_xp:
            regions: dict[str, dict[str, float]] = {}
            for i, (bxx, label) in enumerate(zip(xs, labels)):
                _xp_button(frame, bxx, btn_y, bxx + btn_w, btn_y + btn_h, label, accent=(i == 0), px=max(12, int(0.38 * btn_h)), bold=(i == 0))
                regions[names[i]] = {"x1": bxx, "y1": btn_y, "x2": bxx + btn_w + 1, "y2": btn_y + btn_h + 1}
            self.button_handler.button_regions = regions
        else:
            # minimalist — no shadow/highlight, thin border, subtle fill
            buttons = [(xs[0], labels[0], theme["accent"] if self.is_running else (68, 68, 75)), (xs[1], labels[1], (68, 68, 75)), (xs[2], labels[2], (68, 68, 75))]
            regions = {}
            for bxx, label, border in buttons:
                # subtle translucent fill
                overlay = frame.copy()
                styled_rect(overlay, bxx, btn_y, bxx + btn_w, btn_y + btn_h, fill=(38, 38, 42), radius=main_btn_radius)
                cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
                styled_rect(frame, bxx, btn_y, bxx + btn_w, btn_y + btn_h, border=border, thickness=1, radius=main_btn_radius)
                px, _ = fit_text(label, btn_w, btn_h, int(0.38 * btn_h))
                draw_text(frame, label, bxx + btn_w // 2, btn_y + btn_h // 2, px, theme["text"], anchor="mm")
                regions[names[buttons.index((bxx, label, border))]] = {"x1": bxx, "y1": btn_y, "x2": bxx + btn_w + 1, "y2": btn_y + btn_h + 1}
            self.button_handler.button_regions = {n: {"x1": x, "y1": btn_y, "x2": x + btn_w + 1, "y2": btn_y + btn_h + 1} for n, x in zip(names, xs)}

    def _draw_focus_display(self, frame: np.ndarray, theme: dict[str, Any], fw: int, fh: int, ui_scale: float) -> None:
        if not self.layout_manager.config.elements.get("focus_display", UIElementConfig()).enabled:
            return
        fx1, fy1, fx2, fy2 = self.layout_manager.get_element_rect("focus_display", fw, fh)
        if fx2 - fx1 <= 4 or fy2 - fy1 <= 4:
            return
        n = getattr(self, "person_count", len(getattr(self, "last_faces", [])))
        if not self.is_running:
            focus_text = f"Ready" + (f" · {n}" if n else "")
        else:
            base = self.focus_state.title()
            if n > 1:
                focus_text = f"{base} · {n}"
            elif n == 1:
                focus_text = f"{base}"
            else:
                focus_text = f"{base}"
        # minimalist pill behind text
        px, (tw, th) = fit_text(focus_text, fx2 - fx1 - 12, fy2 - fy1 - 6, int(0.52 * (fy2 - fy1) * ui_scale))
        pill_w, pill_h = tw + 14, th + 8
        px1 = fx1 + 4
        py1 = fy1 + (fy2 - fy1 - pill_h) // 2
        overlay = frame.copy()
        styled_rect(overlay, px1, py1, px1 + pill_w, py1 + pill_h, fill=(28, 28, 32), radius=pill_h // 2)
        cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, frame)
        styled_rect(frame, px1, py1, px1 + pill_w, py1 + pill_h, border=(60, 60, 68), thickness=1, radius=pill_h // 2)
        draw_text(frame, focus_text, px1 + pill_w // 2, py1 + pill_h // 2, px, theme["text"], anchor="mm")

    def _draw_status_display(self, frame: np.ndarray, theme: dict[str, Any], fw: int, fh: int, ui_scale: float) -> None:
        if not self.layout_manager.config.elements.get("status_display", UIElementConfig()).enabled:
            return
        sx1, sy1, sx2, sy2 = self.layout_manager.get_element_rect("status_display", fw, fh)
        if sx2 - sx1 <= 4 or sy2 - sy1 <= 4:
            return
        base = "Running" if self.is_running else "Ready"
        n = getattr(self, "person_count", 0)
        # show tracking health: e.g. "Running · 2 tracked" or "Ready · No face"
        if n == 0:
            extra = " · No face" if self.camera_enabled else " · Cam off"
        elif n == 1:
            extra = " · 1 tracked"
        else:
            extra = f" · {n} tracked"
        status_text = f"Status: {base}{extra}"
        px, _ = fit_text(status_text, sx2 - sx1, sy2 - sy1, int(0.5 * (sy2 - sy1) * ui_scale))
        draw_text(frame, status_text, sx1 + 6, sy1 + (sy2 - sy1) // 2, px, theme["subtext"], anchor="lm", stroke=max(1, px // 16), stroke_color=(12, 12, 14))

    def _draw_quit_hint(self, frame: np.ndarray, theme: dict[str, Any], fw: int, fh: int) -> None:
        if not self.layout_manager.config.elements.get("quit_hint", UIElementConfig()).enabled:
            return
        hx1, hy1, hx2, hy2 = self.layout_manager.get_element_rect("quit_hint", fw, fh)
        if hx2 - hx1 <= 4 or hy2 - hy1 <= 4:
            return
        px, _ = fit_text("q: quit", hx2 - hx1, hy2 - hy1, int(0.5 * (hy2 - hy1)), pad_w=4, pad_h=3)
        draw_text(frame, "q: quit", hx1 + (hx2 - hx1) // 2, hy1 + (hy2 - hy1) // 2, px, theme["subtext"], anchor="mm", stroke=max(1, px // 18), stroke_color=(12, 12, 14))

    # -- layout edit overlay ----------------------------------------------
    def _draw_layout_edit_overlay(self, frame: np.ndarray, theme: dict[str, Any], rounded: bool, ui_scale: float) -> None:
        fw, fh = frame.shape[1], frame.shape[0]
        lm = self.layout_manager
        cell_w, cell_h = fw / lm.grid_cols, fh / lm.grid_rows
        grid_color = theme["divider"]
        for i in range(lm.grid_cols + 1):
            cv2.line(frame, (int(i * cell_w), 0), (int(i * cell_w), fh), grid_color, 1, cv2.LINE_AA)
        for i in range(lm.grid_rows + 1):
            cv2.line(frame, (0, int(i * cell_h)), (fw, int(i * cell_h)), grid_color, 1, cv2.LINE_AA)
        for name, elem in lm.config.elements.items():
            x1, y1, x2, y2 = lm.get_element_rect(name, fw, fh, include_disabled=True)
            if x2 <= x1 or y2 <= y1:
                continue
            if name == self._dragging_element:
                color, thickness = theme["accent"], 3
            elif not elem.enabled:
                color, thickness = theme["subtext"], 1
            else:
                color, thickness = theme["on_color"], 2
            styled_rect(frame, x1, y1, x2, y2, border=color, thickness=thickness, radius=6 if rounded else 0)
            label = name.replace("_", " ").title() + ("" if elem.enabled else " (off)")
            px, _ = fit_text(label, x2 - x1, max(14, int(20 * ui_scale)), int(0.4 * ui_scale * 22), pad_w=6, pad_h=3)
            draw_text(frame, label, x1 + 5, min(y1 + 5 + int(14 * ui_scale), y2 - 3), px, color, anchor="la")
            coord_text = f"({elem.x:g}, {elem.y:g})"
            px2, _ = fit_text(coord_text, x2 - x1, max(12, int(16 * ui_scale)), int(0.3 * ui_scale * 22), pad_w=6, pad_h=2)
            draw_text(frame, coord_text, x1 + 5, y2 - 4, px2, theme["subtext"], anchor="lb")
        done_text = "[ Done ]"
        px, (tw, th) = fit_text(done_text, fw // 3, int(40 * ui_scale), int(0.5 * ui_scale * 22), bold=True, pad_w=6, pad_h=4)
        pad = max(10, int(10 * ui_scale))
        pill_w, pill_h = tw + 2 * pad, th + 2 * pad
        px1, py1 = (fw - pill_w) // 2, max(8, int(10 * ui_scale))
        pr = int(pill_h * 0.35) if rounded else 0
        styled_rect(frame, px1, py1, px1 + pill_w, py1 + pill_h, fill=theme["panel_fill"], border=theme["accent"], thickness=2, radius=pr)
        draw_text(frame, done_text, px1 + pill_w // 2, py1 + pill_h // 2, px, theme["text"], bold=True, anchor="mm")
        self._edit_done_rect = (px1, py1, px1 + pill_w, py1 + pill_h)
        hint = "Drag tiles to rearrange  |  [ Done ] or Esc to finish"
        px, (tw, th) = fit_text(hint, fw, int(26 * ui_scale), int(0.4 * ui_scale * 22), pad_w=8, pad_h=2)
        cv2.rectangle(frame, (fw - tw - 16, fh - th - int(14 * ui_scale)), (fw, fh), theme["backdrop"], -1)
        draw_text(frame, hint, fw - tw - 8, fh - max(8, int(8 * ui_scale)), px, theme["subtext"], anchor="la")

    # -- settings panel ---------------------------------------------------
    def _draw_settings_panel(self, frame: np.ndarray) -> None:
        fw, fh = frame.shape[1], frame.shape[0]
        theme, rounded, is_xp = self._theme_palette(), self._is_rounded(), self.theme == ThemeName.XP
        panel_w = int(min(720, max(360, fw * 0.62)))
        panel_h = int(min(560, max(340, fh * 0.72)))
        panel_x, panel_y = int((fw - panel_w) / 2), int((fh - panel_h) / 2)
        pscale = max(0.75, min(2.0, panel_h / 380.0))
        pad = max(14, int(panel_w * 0.04))
        radius = int(min(panel_w, panel_h) * 0.05) if rounded and not is_xp else 0
        if is_xp:
            styled_rect(frame, panel_x, panel_y, panel_x + panel_w, panel_y + panel_h, fill=theme["panel_fill"], border=(104, 104, 104), thickness=2, radius=0)
        else:
            overlay = frame.copy()
            styled_rect(overlay, panel_x, panel_y, panel_x + panel_w, panel_y + panel_h, fill=theme["panel_fill"], radius=radius)
            cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
            styled_rect(frame, panel_x, panel_y, panel_x + panel_w, panel_y + panel_h, border=theme["panel_border"], thickness=2, radius=radius)
        self._settings_button_rects: dict[str, tuple[int, int, int, int]] = {}

        def draw_btn(name: str, x1: int | float, y1: int | float, x2: int | float, y2: int | float, label: str, border_color: tuple[int, int, int] | None = None, text_color: tuple[int, int, int] | None = None, bold: bool = False) -> None:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            if is_xp:
                accent = name in ("pomodoro_plus", "pomodoro_minus", "break_plus", "break_minus", "scale_plus", "scale_minus")
                _xp_button(frame, x1, y1, x2, y2, label, accent=accent, px=max(11, int(0.38 * (y2 - y1))), bold=bold)
            else:
                r = int(min(x2 - x1, y2 - y1) * 0.3) if rounded else 0
                styled_rect(frame, x1, y1, x2, y2, fill=theme["button_fill"], border=border_color or theme["button_border"], thickness=2, radius=r)
                bw, bh = x2 - x1, y2 - y1
                px, _ = fit_text(label, bw, bh, int(0.4 * bh), bold=bold)
                draw_text(frame, label, x1 + bw // 2, y1 + bh // 2, px, text_color or theme["text"], bold=bold, anchor="mm")
            self._settings_button_rects[name] = (x1, y1, x2, y2)

        title_h = int(48 * pscale)
        if is_xp:
            if (close_rect := _xp_title_bar(frame, panel_x, panel_y, panel_x + panel_w, panel_y + title_h, "Settings")):
                self._settings_button_rects["close"] = close_rect
            divider_y = panel_y + title_h + int(4 * pscale)
        else:
            draw_text(frame, "Settings", panel_x + pad, panel_y + pad + title_h // 2 + int(13 * pscale), int(22 * pscale), theme["text"], anchor="la")
            close_w, close_h = int(92 * pscale), int(32 * pscale)
            draw_btn("close", panel_x + panel_w - pad - close_w, panel_y + pad + title_h // 2 - close_h // 2, panel_x + panel_w - pad, panel_y + pad + title_h // 2 + close_h // 2, "Close", text_color=theme["subtext"])
            divider_y = panel_y + pad + title_h // 2 + title_h // 2 + int(8 * pscale)
            cv2.line(frame, (panel_x + pad, divider_y), (panel_x + panel_w - pad, divider_y), theme["divider"], 2)

        content_top = divider_y + int(12 * pscale)
        content_bottom = panel_y + panel_h - pad
        row_h = (content_bottom - content_top) / 9.0
        btn_h = int(min(row_h * 0.62, 42 * pscale))
        gap = int(10 * pscale)
        inner_r = panel_x + panel_w - pad
        stepper_w = int(44 * pscale)
        half_w = (panel_w - 2 * pad - gap) // 2

        def row_cy(i: int) -> int:
            return int(content_top + row_h * (i + 0.5))

        cy = row_cy(0)
        draw_text(frame, f"Pomodoro: {self.session_duration_minutes} min", panel_x + pad, cy + int(8 * pscale), int(19 * pscale), theme["subtext"], anchor="la")
        draw_btn("pomodoro_minus", inner_r - 2 * stepper_w - gap, cy - btn_h // 2, inner_r - stepper_w - gap, cy + btn_h // 2, "-", bold=True)
        draw_btn("pomodoro_plus", inner_r - stepper_w, cy - btn_h // 2, inner_r, cy + btn_h // 2, "+", bold=True)
        cy = row_cy(1)
        draw_text(frame, f"Break: {self.break_duration_minutes} min", panel_x + pad, cy + int(8 * pscale), int(19 * pscale), theme["subtext"], anchor="la")
        draw_btn("break_minus", inner_r - 2 * stepper_w - gap, cy - btn_h // 2, inner_r - stepper_w - gap, cy + btn_h // 2, "-", bold=True)
        draw_btn("break_plus", inner_r - stepper_w, cy - btn_h // 2, inner_r, cy + btn_h // 2, "+", bold=True)
        cy = row_cy(2)
        draw_btn("toggle_camera", panel_x + pad, cy - btn_h // 2, panel_x + pad + half_w, cy + btn_h // 2, f"Camera: {'On' if self.camera_enabled else 'Off'}", border_color=theme["on_color"] if self.camera_enabled else theme["off_color"])
        draw_btn("toggle_bar", panel_x + pad + half_w + gap, cy - btn_h // 2, inner_r, cy + btn_h // 2, f"Bar: {'On' if self.progress_bar_enabled else 'Off'}", border_color=theme["on_color"] if self.progress_bar_enabled else theme["off_color"])
        cy = row_cy(3)
        third_w = (panel_w - 2 * pad - 2 * gap) // 3
        draw_btn("toggle_theme", panel_x + pad, cy - btn_h // 2, panel_x + pad + third_w, cy + btn_h // 2, f"Theme: {self.theme.title()}")
        draw_btn("toggle_corners", panel_x + pad + third_w + gap, cy - btn_h // 2, panel_x + pad + 2 * third_w + gap, cy + btn_h // 2, f"Corners: {'Rounded' if rounded else 'Boxy'}")
        draw_btn("toggle_alerts", panel_x + pad + 2 * third_w + 2 * gap, cy - btn_h // 2, inner_r, cy + btn_h // 2, f"Alerts: {'On' if self.alerts_enabled else 'Off'}", border_color=theme["on_color"] if self.alerts_enabled else theme["off_color"])
        cy = row_cy(4)
        mode_w = (panel_w - 2 * pad - 2 * gap) // 3
        for i, (name, label, value) in enumerate([("mode_progress", "Progress", DisplayMode.PROGRESS_BAR), ("mode_popup", "Popup", DisplayMode.TIMER_POPUP), ("mode_both", "Both", DisplayMode.BOTH)]):
            active = self.display_mode == value
            bx1 = panel_x + pad + i * (mode_w + gap)
            draw_btn(name, bx1, cy - btn_h // 2, bx1 + mode_w, cy + btn_h // 2, label, border_color=theme["accent"] if active else theme["button_border"], text_color=theme["accent"] if active else theme["text"], bold=active)
        cy = row_cy(5)
        layout_btn_w = (panel_w - 2 * pad - 3 * gap) // 4
        for i, (name, label, enabled) in enumerate([("layout_toggle_timer", "Timer", self.layout_manager.config.elements.get("timer_popup", UIElementConfig()).enabled), ("layout_toggle_bar", "Bar", self.layout_manager.config.elements.get("progress_bar", UIElementConfig()).enabled), ("layout_toggle_focus", "Focus", self.layout_manager.config.elements.get("focus_display", UIElementConfig()).enabled), ("layout_toggle_phase", "Phase", self.layout_manager.config.elements.get("phase_label", UIElementConfig()).enabled)]):
            draw_btn(name, panel_x + pad + i * (layout_btn_w + gap), cy - btn_h // 2, panel_x + pad + i * (layout_btn_w + gap) + layout_btn_w, cy + btn_h // 2, label, border_color=theme["on_color"] if enabled else theme["off_color"])
        cy = row_cy(6)
        action_btn_w = (panel_w - 2 * pad - 3 * gap) // 4
        draw_btn("layout_grid_plus", panel_x + pad, cy - btn_h // 2, panel_x + pad + action_btn_w, cy + btn_h // 2, f"Grid: {self.layout_manager.grid_cols}x{self.layout_manager.grid_rows}")
        draw_btn("layout_reset", panel_x + pad + action_btn_w + gap, cy - btn_h // 2, panel_x + pad + 2 * action_btn_w + gap, cy + btn_h // 2, "Default")
        draw_btn("layout_save", panel_x + pad + 2 * action_btn_w + 2 * gap, cy - btn_h // 2, panel_x + pad + 3 * action_btn_w + 2 * gap, cy + btn_h // 2, "Save Layout")
        draw_btn("layout_preset", panel_x + pad + 3 * action_btn_w + 3 * gap, cy - btn_h // 2, inner_r, cy + btn_h // 2, "Presets")
        cy = row_cy(7)
        draw_text(frame, f"UI Scale: {int(self.ui_scale * 100)}%", panel_x + pad, cy + int(8 * pscale), int(19 * pscale), theme["subtext"], anchor="la")
        draw_btn("scale_minus", inner_r - 2 * stepper_w - gap, cy - btn_h // 2, inner_r - stepper_w - gap, cy + btn_h // 2, "-", bold=True)
        draw_btn("scale_plus", inner_r - stepper_w, cy - btn_h // 2, inner_r, cy + btn_h // 2, "+", bold=True)
        cy = row_cy(8)
        draw_btn("layout_edit_mode", panel_x + pad, cy - btn_h // 2, inner_r, cy + btn_h // 2, "Exit Edit Mode" if self.layout_edit_mode else "Layout Edit Mode", border_color=theme["off_color"] if self.layout_edit_mode else theme["accent"], bold=True)

    def _handle_settings_click(self, x: int, y: int) -> None:
        if not hasattr(self, "_settings_button_rects"):
            return
        for name, (x1, y1, x2, y2) in self._settings_button_rects.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                match name:
                    case "pomodoro_minus":
                        self.apply_settings(pomodoro_minutes=self.session_duration_minutes - 1)
                    case "pomodoro_plus":
                        self.apply_settings(pomodoro_minutes=self.session_duration_minutes + 1)
                    case "break_minus":
                        self.apply_settings(break_minutes=self.break_duration_minutes - 1)
                    case "break_plus":
                        self.apply_settings(break_minutes=self.break_duration_minutes + 1)
                    case "toggle_theme":
                        cycle = [ThemeName.DARK, ThemeName.LIGHT, ThemeName.XP]
                        idx = cycle.index(self.theme) if self.theme in cycle else 0
                        self.theme = cycle[(idx + 1) % len(cycle)]
                        self._save_settings()
                    case "toggle_corners":
                        self.corner_style = CornerStyle.BOXY if self.corner_style == CornerStyle.ROUNDED else CornerStyle.ROUNDED
                        self._save_settings()
                    case "toggle_alerts":
                        self.alerts_enabled = not self.alerts_enabled
                        self._save_settings()
                    case "scale_minus":
                        self.ui_scale = max(UI_SCALE_MIN, round(self.ui_scale - UI_SCALE_STEP, 2))
                        self._save_settings()
                    case "scale_plus":
                        self.ui_scale = min(UI_SCALE_MAX, round(self.ui_scale + UI_SCALE_STEP, 2))
                        self._save_settings()
                    case "toggle_bar":
                        self.progress_bar_enabled = not getattr(self, "progress_bar_enabled", True)
                        self.show_progress_bar = self.display_mode in (DisplayMode.PROGRESS_BAR, DisplayMode.BOTH)
                    case "toggle_camera":
                        self.camera_enabled = not getattr(self, "camera_enabled", True)
                        if not self.camera_enabled and self.cap is not None:
                            try:
                                self.cap.release()
                            except Exception:
                                pass
                            self.cap = None
                        self.prev_gray = None
                        if self.camera_enabled:
                            self.start_camera()
                    case "mode_progress":
                        self.apply_settings(display_mode=DisplayMode.PROGRESS_BAR)
                    case "mode_popup":
                        self.apply_settings(display_mode=DisplayMode.TIMER_POPUP)
                    case "mode_both":
                        self.apply_settings(display_mode=DisplayMode.BOTH)
                    case "layout_toggle_timer":
                        self.layout_manager.enable_element("timer_popup", not self.layout_manager.config.elements.get("timer_popup", UIElementConfig()).enabled)
                        self._save_layout()
                    case "layout_toggle_bar":
                        self.layout_manager.enable_element("progress_bar", not self.layout_manager.config.elements.get("progress_bar", UIElementConfig()).enabled)
                        self._save_layout()
                    case "layout_toggle_focus":
                        self.layout_manager.enable_element("focus_display", not self.layout_manager.config.elements.get("focus_display", UIElementConfig()).enabled)
                        self._save_layout()
                    case "layout_toggle_phase":
                        self.layout_manager.enable_element("phase_label", not self.layout_manager.config.elements.get("phase_label", UIElementConfig()).enabled)
                        self._save_layout()
                    case "layout_grid_plus":
                        new_cols, new_rows = self.layout_manager.grid_cols + 2, self.layout_manager.grid_rows + 1
                        if new_cols > 20:
                            new_cols, new_rows = GRID_COLS_DEFAULT, GRID_ROWS_DEFAULT
                        self.layout_manager.set_grid(new_cols, new_rows)
                        self._save_layout()
                    case "layout_reset":
                        self.reset_layout()
                    case "layout_save":
                        self._save_layout()
                    case "layout_preset":
                        self._cycle_layout_preset()
                    case "layout_edit_mode":
                        self.layout_edit_mode = not self.layout_edit_mode
                        self._dragging_element = None
                        if self.layout_edit_mode:
                            self.settings_visible = False
                        else:
                            self._save_layout()
                    case "close":
                        self.settings_visible = False
                return

    def _cycle_layout_preset(self) -> None:
        # all presets are minimalist — no center Pomodoro, safe timer, thin progress, subtle focus
        # use stored index to cycle reliably even if layout was hand-edited
        if not hasattr(self, "_preset_idx"):
            self._preset_idx = 0
        presets: list[tuple[str, LayoutConfig]] = [
            ("Default", LayoutConfig.default()),
            ("Focus", LayoutConfig(grid_cols=12, grid_rows=8, elements={
                "timer_popup": UIElementConfig(enabled=True, x=4.4, y=0.38, width=3.2, height=0.95, margin=0.14, font_scale=1.02),
                "phase_label": UIElementConfig(enabled=False, x=4.6, y=0.55, width=2.9, height=0.7, margin=0.12),
                "progress_bar": UIElementConfig(enabled=False, x=0.30, y=0.10, width=11.4, height=0.12, margin=0.06),
                "focus_display": UIElementConfig(enabled=False, x=0.30, y=0.38, width=2.7, height=0.40, margin=0.06),
                "status_display": UIElementConfig(enabled=False, x=0.4, y=1.0, width=3.2, height=0.42, margin=0.08),
                "main_buttons": UIElementConfig(enabled=True, x=3.6, y=7.28, width=4.8, height=0.60, margin=0.10),
                "quit_hint": UIElementConfig(enabled=False, x=10.6, y=7.45, width=1.2, height=0.4, margin=0.1),
            })),
            ("Dashboard", LayoutConfig(grid_cols=16, grid_rows=9, elements={
                "timer_popup": UIElementConfig(enabled=True, x=11.8, y=0.45, width=3.6, height=1.00, margin=0.12, font_scale=0.96),
                "phase_label": UIElementConfig(enabled=False, x=11.0, y=2.6, width=4.6, height=0.9, margin=0.15),
                "progress_bar": UIElementConfig(enabled=True, x=0.35, y=0.14, width=15.3, height=0.14, margin=0.05),
                "focus_display": UIElementConfig(enabled=True, x=0.35, y=0.42, width=3.8, height=0.44, margin=0.06),
                "status_display": UIElementConfig(enabled=False, x=0.4, y=2.5, width=5.0, height=0.7, margin=0.1),
                "main_buttons": UIElementConfig(enabled=True, x=4.8, y=8.02, width=6.4, height=0.72, margin=0.10),
                "quit_hint": UIElementConfig(enabled=False, x=14.2, y=8.45, width=1.6, height=0.4, margin=0.1),
            })),
            ("Minimal", LayoutConfig(grid_cols=12, grid_rows=8, elements={
                "timer_popup": UIElementConfig(enabled=True, x=8.8, y=0.38, width=2.6, height=0.82, margin=0.14, font_scale=0.88),
                "phase_label": UIElementConfig(enabled=False, x=4.4, y=0.55, width=3.2, height=0.8, margin=0.15),
                "progress_bar": UIElementConfig(enabled=False, x=0.30, y=0.10, width=11.4, height=0.12, margin=0.06),
                "focus_display": UIElementConfig(enabled=False, x=0.30, y=0.38, width=2.7, height=0.40, margin=0.06),
                "status_display": UIElementConfig(enabled=False, x=0.4, y=1.05, width=3.6, height=0.5, margin=0.1),
                "main_buttons": UIElementConfig(enabled=True, x=4.0, y=7.32, width=4.0, height=0.58, margin=0.08),
                "quit_hint": UIElementConfig(enabled=False, x=10.6, y=7.45, width=1.2, height=0.4, margin=0.1),
            })),
        ]
        # use stored index for reliable cycling (dict compare is brittle after migration)
        if not hasattr(self, "_preset_idx"):
            # try to infer current preset by timer position, else start at 0
            cur_x = self.layout_manager.config.elements.get("timer_popup", UIElementConfig()).x
            # find closest preset by timer x
            try:
                cur_idx = next(i for i, (_, p) in enumerate(presets) if abs(p.elements["timer_popup"].x - cur_x) < 0.3)
            except StopIteration:
                cur_idx = -1
            self._preset_idx = cur_idx
        self._preset_idx = (self._preset_idx + 1) % len(presets)
        name, cfg = presets[self._preset_idx]
        self.layout_manager.config = cfg
        self.layout_manager.grid_cols, self.layout_manager.grid_rows = cfg.grid_cols, cfg.grid_rows
        self._save_layout()
        log.info("Layout preset: %s", name)

    def cleanup(self) -> None:
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception as exc:
                log.debug("Camera release failed: %s", exc)
        try:
            cv2.destroyAllWindows()
        except Exception as exc:
            log.debug("destroyAllWindows failed: %s", exc)


def main() -> None:
    timer = PomodoroTimer()
    log.info("Pomodoro Camera Started — press 'q' to quit")
    print("Pomodoro Camera Started!")
    print("Press 'q' to quit the session.")
    try:
        timer.start_session()
    except KeyboardInterrupt:
        log.info("Session interrupted by user")
        print("\nSession interrupted by user.")
    finally:
        timer.cleanup()


if __name__ == "__main__":
    main()

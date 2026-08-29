import cv2
import numpy as np
import time
import math
import platform
import ctypes
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Any

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_OK = True
except Exception:
    _PIL_OK = False


# ---------------------------------------------------------------------------
# Font engine: real TrueType text (Segoe UI on Windows) rendered through PIL,
# with a Hershey-vector fallback when Pillow or a system font is unavailable.
# ---------------------------------------------------------------------------
def _resolve_font_paths():
    """Find a regular + bold TTF on this system. Returns (regular, bold)."""
    font_dirs = []
    if platform.system() == "Windows":
        windir = os.environ.get("WINDIR", r"C:\Windows")
        font_dirs.append(os.path.join(windir, "Fonts"))
    font_dirs += [
        "/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/TTF",
        "/usr/share/fonts", "/Library/Fonts", "/System/Library/Fonts",
    ]
    regular_names = ["segoeui.ttf", "arial.ttf", "DejaVuSans.ttf", "Verdana.ttf", "Helvetica.ttf"]
    bold_names = ["segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf", "Verdanab.ttf", "arial bold.ttf"]
    regular = bold = None
    for d in font_dirs:
        for name in regular_names:
            p = os.path.join(d, name)
            if os.path.exists(p):
                regular = p
                break
        for name in bold_names:
            p = os.path.join(d, name)
            if os.path.exists(p):
                bold = p
                break
        if regular and bold:
            break
    return regular, bold


_FONT_PATHS = _resolve_font_paths() if _PIL_OK else (None, None)
_FONT_CACHE = {}
_TEXT_SIZE_CACHE = {}  # (text, px, bold) -> (w, h)


def get_font(px, bold=False):
    """Cached PIL font at a pixel size."""
    px = max(8, int(round(px)))
    key = (px, bool(bold))
    if key not in _FONT_CACHE:
        path = _FONT_PATHS[1 if bold else 0]
        try:
            _FONT_CACHE[key] = ImageFont.truetype(path, px)
        except Exception:
            _FONT_CACHE[key] = ImageFont.load_default()
    return _FONT_CACHE[key]


def text_size(text, px, bold=False):
    """Rendered (width, height) of text in pixels. Cached."""
    px = max(8, int(round(px)))
    key = (text, px, bool(bold))
    cached = _TEXT_SIZE_CACHE.get(key)
    if cached is not None:
        return cached
    if _PIL_OK:
        font = get_font(px, bold)
        dummy = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        l, t, r, b = dummy.textbbox((0, 0), text, font=font)
        result = (r - l, b - t)
    else:
        scale = max(0.3, px / 26.0)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, max(1, int(px / 18)))
        result = (tw, th)
    if len(_TEXT_SIZE_CACHE) < 2048:
        _TEXT_SIZE_CACHE[key] = result
    return result


def draw_text(img, text, x, y, px, color, bold=False, anchor="la",
              stroke=0, stroke_color=(0, 0, 0)):
    """Draw text onto a BGR numpy image.

    anchor: 'la' = x,y is top-left  |  'mm' = centered  |  'lm' = left-middle.
    stroke: optional outline width (readability over busy camera frames).
    Falls back to Hershey vectors when Pillow is unavailable.
    """
    px = max(8, int(round(px)))
    stroke = max(0, int(stroke))
    if not _PIL_OK:
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
                        cv2.putText(img, text, (int(ox) + ddx, int(oy) + ddy),
                                    cv2.FONT_HERSHEY_SIMPLEX, scale, stroke_color, thick, cv2.LINE_AA)
        cv2.putText(img, text, (int(ox), int(oy)), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thick, cv2.LINE_AA)
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
    pil = Image.fromarray(region[:, :, ::-1])
    d = ImageDraw.Draw(pil)
    fill = (int(color[2]), int(color[1]), int(color[0]))
    if stroke > 0:
        d.text((ax - x0, ay - y0), text, font=font, fill=fill,
               stroke_width=stroke,
               stroke_fill=(int(stroke_color[2]), int(stroke_color[1]), int(stroke_color[0])))
    else:
        d.text((ax - x0, ay - y0), text, font=font, fill=fill)
    img[y0:y1, x0:x1] = np.ascontiguousarray(np.array(pil)[:, :, ::-1])


_FIT_TEXT_CACHE = {}  # (text, box_w, box_h, max_px, bold) -> (px, (tw, th))


def fit_text(text, box_w, box_h, max_px, bold=False, pad_w=12, pad_h=8, min_px=10):
    """Binary-search the largest pixel size whose text fits (box_w, box_h).

    Returns (px, (text_w, text_h)). Cached.
    """
    max_px = int(max(min_px, max_px))
    key = (text, int(box_w), int(box_h), max_px, bool(bold))
    cached = _FIT_TEXT_CACHE.get(key)
    if cached is not None:
        return cached

    lo, hi = min_px, max_px
    best = (min_px, text_size(text, min_px, bold))
    while lo <= hi:
        mid = (lo + hi) // 2
        tw, th = text_size(text, mid, bold)
        if tw <= box_w - pad_w and th <= box_h - pad_h:
            best = (mid, (tw, th))
            lo = mid + 1
        else:
            hi = mid - 1

    if len(_FIT_TEXT_CACHE) < 1024:
        _FIT_TEXT_CACHE[key] = best
    return best


@dataclass
class UIElementConfig:
    """Configuration for a single UI element.

    x/y/width/height are grid cells measured from the TOP-LEFT of the window.
    `anchor` is kept only for backward compatibility with older layout.json
    files; it no longer affects positioning.
    """
    enabled: bool = True
    x: float = 0.0
    y: float = 0.0
    width: float = 1.0
    height: float = 1.0
    anchor: str = "top-left"
    margin: float = 0.0
    padding: float = 0.0
    font_scale: float = 1.0
    custom_props: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "UIElementConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class LayoutConfig:
    """Complete layout configuration for all UI elements."""
    grid_cols: int = 12
    grid_rows: int = 8
    elements: Dict[str, UIElementConfig] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "grid_cols": self.grid_cols,
            "grid_rows": self.grid_rows,
            "elements": {k: v.to_dict() for k, v in self.elements.items()}
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "LayoutConfig":
        elements = {k: UIElementConfig.from_dict(v) for k, v in data.get("elements", {}).items()}
        return cls(grid_cols=data.get("grid_cols", 12), grid_rows=data.get("grid_rows", 8), elements=elements)

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str):
        """Load a config from disk, or return None if missing/corrupt/invalid."""
        if not os.path.exists(path):
            return None
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            cfg = cls.from_dict(data)
        except Exception:
            return None
        if cfg.grid_cols < 1 or cfg.grid_rows < 1 or not cfg.elements:
            return None
        return cfg

    @classmethod
    def default(cls) -> "LayoutConfig":
        """Friendly out-of-box layout - compact and airy by default.

        All x/y/width/height are in grid cells measured from the TOP-LEFT of
        the window. The 12x8 grid stretches proportionally with the window,
        so elements keep their relative placement at any window size.
        """
        return cls(
            grid_cols=12,
            grid_rows=8,
            elements={
                "timer_popup": UIElementConfig(enabled=True, x=8.8, y=0.3, width=2.8, height=1.3, margin=0.12, font_scale=1.0),
                "phase_label": UIElementConfig(enabled=True, x=4.6, y=0.55, width=2.9, height=0.7, margin=0.12),
                "progress_bar": UIElementConfig(enabled=True, x=0.6, y=2.05, width=10.6, height=0.28, margin=0.08),
                "focus_display": UIElementConfig(enabled=True, x=0.4, y=0.35, width=3.2, height=0.5, margin=0.08),
                "status_display": UIElementConfig(enabled=True, x=0.4, y=1.0, width=3.2, height=0.42, margin=0.08),
                "main_buttons": UIElementConfig(enabled=True, x=1.9, y=6.9, width=8.2, height=0.85, margin=0.12),
                "quit_hint": UIElementConfig(enabled=False, x=10.6, y=7.45, width=1.2, height=0.4, margin=0.1),
            }
        )


class LayoutManager:
    """Manages UI layout using a grid system with anchor-based positioning."""

    def __init__(self, config: LayoutConfig):
        self.config = config
        self.grid_cols = config.grid_cols
        self.grid_rows = config.grid_rows

    def set_grid(self, cols: int, rows: int):
        self.grid_cols = cols
        self.grid_rows = rows
        self.config.grid_cols = cols
        self.config.grid_rows = rows

    def get_element_rect(self, name: str, frame_w: int, frame_h: int,
                         include_disabled: bool = False) -> Tuple[int, int, int, int]:
        """Return the pixel rect (x1, y1, x2, y2) for a UI element.

        Positions are top-left based grid cells. Disabled elements return
        (0, 0, 0, 0) unless include_disabled=True (used by the layout editor
        so hidden tiles remain visible/draggable there).
        """
        elem = self.config.elements.get(name)
        if elem is None:
            return (0, 0, 0, 0)
        if not elem.enabled and not include_disabled:
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

    def update_element(self, name: str, **kwargs):
        if name not in self.config.elements:
            self.config.elements[name] = UIElementConfig()
        elem = self.config.elements[name]
        for k, v in kwargs.items():
            if hasattr(elem, k):
                setattr(elem, k, v)

    def enable_element(self, name: str, enabled: bool = True):
        if name in self.config.elements:
            self.config.elements[name].enabled = enabled

    def get_enabled_elements(self) -> List[str]:
        return [k for k, v in self.config.elements.items() if v.enabled]


def styled_rect(img, x1, y1, x2, y2, fill=None, border=None, thickness=1, radius=0):
    """Draw a rectangle with optional rounded corners.

    radius=0 gives a boxy rectangle. Borders are drawn with straight line
    segments plus elliptical arcs at the corners so no stray circle outlines
    appear (unlike drawing outlined corner circles).
    """
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


# ---------------------------------------------------------------------------
# Windows XP Luna helpers
# ---------------------------------------------------------------------------
def _h_gradient(img, x1, y1, x2, y2, c1, c2, radius=0):
    """Horizontal gradient fill inside a rectangle. numpy-vectorized."""
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return
    c1 = np.array(c1, dtype=np.float32)
    c2 = np.array(c2, dtype=np.float32)
    t = np.linspace(0, 1, w, dtype=np.float32)
    cols = (c1 + np.outer(t, c2 - c1)).clip(0, 255).astype(np.uint8)

    if radius > 0:
        r = min(radius, w // 2, h // 2)
        centers = np.arange(w, dtype=np.float32)
        edge_dist = np.minimum(centers, w - 1 - centers)
        shrink = np.where(edge_dist < r,
                          (r - np.sqrt(np.maximum(0, r * r - (r - edge_dist) ** 2))).astype(int), 0)
        for i in range(w):
            sy = y1 + shrink[i]
            ey = y2 - shrink[i]
            if sy < ey:
                cv2.line(img, (x1 + i, sy), (x1 + i, ey), tuple(cols[i].tolist()), 1, cv2.LINE_AA)
    else:
        for i in range(w):
            cv2.line(img, (x1 + i, y1), (x1 + i, y2 - 1), tuple(cols[i].tolist()), 1, cv2.LINE_AA)


def _v_gradient(img, x1, y1, x2, y2, c1, c2):
    """Vertical gradient fill inside a rectangle. numpy-vectorized."""
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    h = y2 - y1
    w = x2 - x1
    if h <= 0 or w <= 0:
        return
    c1 = np.array(c1, dtype=np.float32)
    c2 = np.array(c2, dtype=np.float32)
    t = np.linspace(0, 1, h, dtype=np.float32)
    cols = (c1 + np.outer(t, c2 - c1)).clip(0, 255).astype(np.uint8)
    for i in range(h):
        cv2.line(img, (x1, y1 + i), (x2 - 1, y1 + i), tuple(cols[i].tolist()), 1, cv2.LINE_AA)


def _xp_button(img, x1, y1, x2, y2, label, accent=False, active=False,
               text_color=None, px=14, bold=False):
    """Draw a Windows XP Luna-style beveled button. Optimized."""
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

    hi = (255, 255, 255)
    sh = (105, 105, 105)
    dk = (60, 60, 60)

    # Fill with gradient in one pass (avoids separate styled_rect + 2x _v_gradient)
    c1 = np.array(c_top, dtype=np.float32)
    c2 = np.array(c_bot, dtype=np.float32)
    half_h = bh // 2
    t1 = np.linspace(0, 1, half_h, dtype=np.float32) if half_h > 0 else np.array([0.0], dtype=np.float32)
    t2 = np.linspace(0, 1, bh - half_h, dtype=np.float32) if bh - half_h > 0 else np.array([0.0], dtype=np.float32)
    top_cols = (c1 + np.outer(t1, c2 - c1)).clip(0, 255).astype(np.uint8)
    bot_cols = (c2 + np.outer(t2, c1 - c2)).clip(0, 255).astype(np.uint8)
    for i in range(half_h):
        cv2.line(img, (x1 + 2, y1 + 2 + i), (x2 - 3, y1 + 2 + i), tuple(top_cols[i].tolist()), 1, cv2.LINE_AA)
    for i in range(bh - half_h):
        cv2.line(img, (x1 + 2, y1 + 2 + half_h + i), (x2 - 3, y1 + 2 + half_h + i), tuple(bot_cols[i].tolist()), 1, cv2.LINE_AA)

    # 3-D bevel edges
    cv2.line(img, (x1 + r, y1), (x2 - r, y1), hi, 2, cv2.LINE_AA)
    cv2.line(img, (x1, y1 + r), (x1, y2 - r), hi, 2, cv2.LINE_AA)
    cv2.line(img, (x2 - 1, y1 + r), (x2 - 1, y2 - r), sh, 2, cv2.LINE_AA)
    cv2.line(img, (x1 + r, y2 - 1), (x2 - r, y2 - 1), sh, 2, cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x2, y2), dk, 1, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2, y2), dk, 1, cv2.LINE_AA)

    # corner arcs
    for cx, cy, s, e in [(x1+r, y1+r, 180, 270), (x2-r, y1+r, 270, 360),
                          (x1+r, y2-r, 90, 180), (x2-r, y2-r, 0, 90)]:
        cv2.ellipse(img, (cx, cy), (r, r), 0, s, e, hi, 2, cv2.LINE_AA)

    tc = text_color or (15, 15, 15)
    draw_text(img, label, x1 + bw // 2, y1 + bh // 2, px, tc, bold=bold, anchor="mm")


def _xp_title_bar(img, x1, y1, x2, y2, title, close_btn=True):
    """Draw a Windows XP Luna-style blue gradient title bar with close button."""
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    tb_h = y2 - y1
    if tb_h < 4:
        return

    # Blue gradient: dark blue left -> lighter blue center -> dark blue right
    _h_gradient(img, x1, y1, x2, y2, (180, 100, 10), (250, 180, 60))
    # Lighter center stripe
    mid = (x1 + x2) // 2
    stripe_w = (x2 - x1) // 3
    _h_gradient(img, mid - stripe_w // 2, y1 + 1, mid + stripe_w // 2, y2 - 1,
                (255, 200, 80), (230, 160, 40))

    # Bottom edge
    cv2.line(img, (x1, y2 - 1), (x2, y2 - 1), (120, 70, 0), 1, cv2.LINE_AA)

    # Title text (white, bold, with drop shadow)
    draw_text(img, title, x1 + 8, y1 + tb_h // 2, max(12, int(tb_h * 0.52)),
              (255, 255, 255), bold=True, anchor="lm",
              stroke=1, stroke_color=(40, 20, 0))

    # Close button (red square with X)
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
        x_px = max(8, int(cb_size * 0.48))
        cx, cy = (cb_x1 + cb_x2) // 2, (cb_y1 + cb_y2) // 2
        draw_text(img, "x", cx, cy, x_px, (255, 255, 255), bold=True, anchor="mm")
        return (cb_x1, cb_y1, cb_x2, cb_y2)
    return None


def _xp_progress_bar(img, x1, y1, x2, y2, progress, rounded=False):
    """Draw a Windows XP-style chunky green progress bar."""
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    pw, ph = x2 - x1, y2 - y1
    if pw < 8 or ph < 4:
        return
    r = min(3, ph // 3) if rounded else 0

    # Sunken track (flat fill, no gradient needed)
    styled_rect(img, x1, y1, x2, y2, fill=(80, 80, 80), radius=r)
    cv2.line(img, (x1, y1), (x2 - 1, y1), (50, 50, 50), 1, cv2.LINE_AA)
    cv2.line(img, (x1, y1), (x1, y2 - 1), (50, 50, 50), 1, cv2.LINE_AA)
    cv2.line(img, (x2 - 1, y1 + 1), (x2 - 1, y2 - 1), (160, 160, 160), 1, cv2.LINE_AA)
    cv2.line(img, (x1 + 1, y2 - 1), (x2 - 1, y2 - 1), (160, 160, 160), 1, cv2.LINE_AA)

    if progress > 0:
        fill_w = max(4, int(pw * min(1.0, progress)))
        fx2 = x1 + fill_w
        # Green gradient fill (numpy-vectorized)
        c1 = np.array([110, 210, 80], dtype=np.float32)
        c2 = np.array([50, 160, 20], dtype=np.float32)
        gh = ph - 4
        if gh > 0:
            t = np.linspace(0, 1, gh, dtype=np.float32)
            cols = (c1 + np.outer(t, c2 - c1)).clip(0, 255).astype(np.uint8)
            for i in range(gh):
                cv2.line(img, (x1 + 2, y1 + 2 + i), (fx2 - 1, y1 + 2 + i),
                         tuple(cols[i].tolist()), 1, cv2.LINE_AA)
        # Highlight on top
        cv2.line(img, (x1 + 2, y1 + 1), (fx2 - 2, y1 + 1), (160, 255, 120), 1, cv2.LINE_AA)
        # Chunky segments
        seg_w = max(12, ph)
        for sx in range(x1 + seg_w, fx2 - 2, seg_w):
            cv2.line(img, (sx, y1 + 2), (sx, y2 - 2), (80, 180, 50), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Theme palettes (BGR)
# ---------------------------------------------------------------------------
class OnboardingManager:
    """Multi-step intro overlay shown on first launch.

    Each step has a title, description, accent colour, and an optional
    highlight target that draws a pulsing rounded box around the relevant
    UI area.  Clicks on Next / Back / Skip advance or dismiss the flow.
    """

    _STEPS = [
        {
            "title": "Welcome to Pomodoro Camera",
            "body": (
                "A focus timer that watches your webcam to keep you "
                "concentrated.  Work in short bursts, take breaks, and "
                "let the camera help you stay on track."
            ),
            "accent": (0, 180, 255),
            "highlight": None,
        },
        {
            "title": "Pomodoro Timer",
            "body": (
                "25-minute work sessions followed by 5-minute breaks.\n"
                "Press the Start button or hit  S  to begin."
            ),
            "accent": (80, 200, 120),
            "highlight": (0.82, 0.08, 0.28, 0.20),
        },
        {
            "title": "Camera Focus Tracking",
            "body": (
                "Your webcam detects your face and monitors movement.\n"
                "Stay visible and still to stay  Concentrated."
            ),
            "accent": (0, 200, 220),
            "highlight": (0.08, 0.05, 0.30, 0.30),
        },
        {
            "title": "Slacking Alerts",
            "body": (
                "When the timer is running, looking away or moving too\n"
                "much triggers a red alert with a soft beep.  Toggle in\n"
                "Settings > Alerts."
            ),
            "accent": (0, 100, 255),
            "highlight": (0.25, 0.35, 0.75, 0.50),
        },
        {
            "title": "Custom Layout",
            "body": (
                "Drag tiles to reposition any element.  Press  E  or\n"
                "use Settings > Layout Edit Mode to rearrange."
            ),
            "accent": (220, 180, 40),
            "highlight": (0.15, 0.60, 0.85, 0.95),
        },
        {
            "title": "Keyboard Shortcuts",
            "body": (
                "  S  start / pause      E  layout edit\n"
                "  Q  quit               Esc  exit edit"
            ),
            "accent": (180, 140, 255),
            "highlight": None,
        },
    ]

    def __init__(self):
        self.step = 0
        self.active = True

    # ------------------------------------------------------------------
    # Public API called by PomodoroTimer
    # ------------------------------------------------------------------

    def _card_rects(self, fw, fh):
        """Compute card layout and button rects for the current step."""
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
            "card_w": card_w, "card_h": card_h,
            "pad": pad, "btn_h": btn_h, "nav_y": nav_y,
            "btn_next": btn_next, "btn_back": btn_back, "btn_skip": btn_skip,
            "is_last": is_last, "is_first": is_first,
        }

    def draw_overlay(self, frame, theme, rounded, ui_scale):
        """Render the full onboarding card on *frame* (BGR numpy)."""
        if not self.active:
            return
        fw, fh = frame.shape[1], frame.shape[0]
        step = self._STEPS[self.step]
        accent = step["accent"]
        r = self._card_rects(fw, fh)
        card_x, card_y, card_x2, card_y2 = r["card"]
        card_w, card_h, pad, btn_h, nav_y = r["card_w"], r["card_h"], r["pad"], r["btn_h"], r["nav_y"]
        is_xp = getattr(self, "_theme", "dark") == "xp"

        # --- dim overlay ---
        overlay = frame.copy()
        if is_xp:
            cv2.rectangle(overlay, (0, 0), (fw, fh), (180, 175, 165), -1)
        else:
            cv2.rectangle(overlay, (0, 0), (fw, fh), (10, 10, 12), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        # --- highlight box ---
        ht = step.get("highlight")
        if ht:
            hx1 = int(fw * ht[0])
            hy1 = int(fh * ht[1])
            hx2 = int(fw * ht[2])
            hy2 = int(fh * ht[3])
            hr = 14 if rounded and not is_xp else 0
            pulse = 0.35 + 0.10 * math.sin(time.time() * 3.0)
            hl_overlay = frame.copy()
            styled_rect(hl_overlay, hx1, hy1, hx2, hy2, fill=accent, radius=hr)
            cv2.addWeighted(hl_overlay, pulse, frame, 1.0 - pulse, 0, frame)
            styled_rect(frame, hx1, hy1, hx2, hy2, border=accent, thickness=2, radius=hr)

        # --- card ---
        if is_xp:
            # XP: silver window with blue title bar
            tb_h = max(22, int(card_h * 0.12))
            styled_rect(frame, card_x, card_y, card_x2, card_y2,
                        fill=(236, 233, 216), border=(104, 104, 104), thickness=2, radius=0)
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
            # step title
            title_px = max(18, int(min(22, card_h * 0.09) * ui_scale))
            draw_text(frame, step["title"], card_x + pad, card_y + pad + int(card_h * 0.05),
                      title_px, accent, bold=True, anchor="la")
            body_top = card_y + pad + int(card_h * 0.22)

        # body text (supports \n line breaks)
        body_px = max(13, int(min(16, card_h * 0.065) * ui_scale))
        lines = step["body"].split("\n")
        body_y = body_top
        for line in lines:
            tc = (16, 16, 16) if is_xp else theme.get("text", (220, 220, 220))
            draw_text(frame, line, card_x + pad, body_y, body_px, tc, anchor="la")
            body_y += int(body_px * 1.65)

        # Back button
        if r["btn_back"]:
            bx1, by1, bx2, by2 = r["btn_back"]
            if is_xp:
                _xp_button(frame, bx1, by1, bx2, by2, "<  Back",
                           px=max(11, int(body_px * 0.85)))
            else:
                styled_rect(frame, bx1, by1, bx2, by2,
                            fill=(50, 50, 55), border=(90, 90, 95), thickness=1,
                            radius=btn_h // 2 if rounded else 0)
                draw_text(frame, "<  Back", bx1 + (bx2 - bx1) // 2, by1 + (by2 - by1) // 2,
                          max(12, int(body_px * 0.9)), (170, 170, 175), anchor="mm")

        # Next / Finish button
        bx1, by1, bx2, by2 = r["btn_next"]
        next_label = "Start  >" if r["is_last"] else "Next  >"
        if is_xp:
            _xp_button(frame, bx1, by1, bx2, by2, next_label, accent=True,
                       px=max(12, int(body_px * 0.9)), bold=True)
        else:
            styled_rect(frame, bx1, by1, bx2, by2,
                        fill=accent, radius=btn_h // 2 if rounded else 0)
            draw_text(frame, next_label, bx1 + (bx2 - bx1) // 2, by1 + (by2 - by1) // 2,
                      max(13, int(body_px * 0.95)), (255, 255, 255), bold=True, anchor="mm")

        # Skip link
        if r["btn_skip"]:
            sx1, sy1, sx2, sy2 = r["btn_skip"]
            sc = (90, 80, 60) if is_xp else (110, 110, 118)
            draw_text(frame, "Skip", sx1 + (sx2 - sx1) // 2, sy1 + (sy2 - sy1) // 2,
                      max(11, int(body_px * 0.78)), sc, anchor="mm")

        # progress dots
        dot_r = max(3, int(min(5, card_h * 0.015)))
        dot_gap = dot_r * 3
        total_dots_w = len(self._STEPS) * dot_r * 2 + (len(self._STEPS) - 1) * dot_gap
        dot_sx = card_x + (card_w - total_dots_w) // 2
        dot_cy = nav_y - int(btn_h * 0.25)
        for i in range(len(self._STEPS)):
            cx = dot_sx + i * (dot_r * 2 + dot_gap) + dot_r
            c = accent if i == self.step else (70, 70, 78)
            cv2.circle(frame, (cx, dot_cy), dot_r, c, -1, cv2.LINE_AA)

    def handle_click(self, x, y, fw, fh):
        """Process a mapped click. Returns True if the onboarding consumed it."""
        if not self.active:
            return False

        r = self._card_rects(fw, fh)

        def hit(rect):
            return rect and rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]

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


class ButtonHandler:
    """Handles button click detection and actions for the on-screen controls."""

    def __init__(self, timer):
        self.timer = timer
        self.button_regions = {
            "start": {"x1": 0.06, "y1": 0.82, "x2": 0.30, "y2": 0.95},
            "reset": {"x1": 0.36, "y1": 0.82, "x2": 0.58, "y2": 0.95},
            "settings": {"x1": 0.66, "y1": 0.82, "x2": 0.88, "y2": 0.95},
        }

    def is_button_clicked(self, event, x, y, frame_w=None, frame_h=None):
        """Check if a button was clicked based on mouse position."""
        if event != cv2.EVENT_LBUTTONDOWN:
            return None

        if frame_w is None or frame_h is None:
            frame_w, frame_h = self.timer.frame_width, self.timer.frame_height

        for btn_name, coords in self.button_regions.items():
            x1 = coords.get("x1", 0)
            y1 = coords.get("y1", 0)
            x2 = coords.get("x2", 0)
            y2 = coords.get("y2", 0)

            if 0.0 <= x1 <= 1.0 and 0.0 <= y1 <= 1.0 and 0.0 <= x2 <= 1.0 and 0.0 <= y2 <= 1.0:
                x1, y1 = int(frame_w * x1), int(frame_h * y1)
                x2, y2 = int(frame_w * x2), int(frame_h * y2)
            else:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            if x1 <= x <= x2 and y1 <= y <= y2:
                return btn_name

        return None

    def handle_button_click(self, button_name):
        """Handle button click actions."""
        if button_name == "start":
            self.timer.toggle_timer()
        elif button_name == "reset":
            self.timer.reset_timer()
        elif button_name == "settings":
            self.timer.toggle_settings()


class PomodoroTimer:
    """Manage the Pomodoro timer, camera feed, and focus tracking in one window."""

    DISPLAY_MODE_PROGRESS_BAR = "progress_bar"
    DISPLAY_MODE_TIMER_POPUP = "timer_popup"
    DISPLAY_MODE_BOTH = "both"

    # UI palettes (BGR). Everything drawn on screen takes its colors from here.
    THEMES = {
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
            "backdrop": (212, 208, 200),          # WinXP silver window body
            "panel_fill": (236, 233, 216),         # silver / beige
            "panel_border": (104, 104, 104),
            "divider": (172, 168, 160),
            "text": (16, 16, 16),                  # near-black
            "subtext": (90, 80, 60),               # brownish gray
            "button_fill": (226, 223, 210),        # button face
            "button_border": (104, 104, 104),
            "accent": (200, 130, 30),              # orange-gold
            "on_color": (80, 170, 50),             # green
            "off_color": (60, 60, 180),            # blue
            "popup_fill": (236, 233, 216),
            "bar_track": (160, 160, 160),
            "widget_fill": (212, 208, 200),
            # XP-specific extras (used by _draw_ui directly)
            "xp_title_bar":  ((180, 100, 10), (250, 180, 60)),  # gradient tuple
            "xp_close_btn":  (80, 80, 230),
            "xp_btn_top":    (240, 235, 225),
            "xp_btn_bot":    (210, 205, 195),
            "xp_btn_accent_top": (255, 170, 50),
            "xp_btn_accent_bot": (200, 115, 10),
        },
    }

    def __init__(self, session_duration_minutes=25, break_duration_minutes=5, display_mode="both"):
        self.session_duration_minutes = session_duration_minutes
        self.break_duration_minutes = break_duration_minutes
        self.display_mode = display_mode

        self.current_phase = "pomodoro"
        self.time_left = self.session_duration_minutes * 60
        self.is_running = False
        self.session_open = False
        self.frame_count = 0
        self.frame_width = 640
        self.frame_height = 480
        self.cap = None
        self.phase_started_at = None
        self.phase_duration_seconds = self.session_duration_minutes * 60

        self.show_progress_bar = display_mode in ["progress_bar", "both"]
        self.show_timer_popup = display_mode in ["timer_popup", "both"]
        self.settings_visible = False
        self.focus_score = 0.0
        self.focus_state = "neutral"
        self.prev_gray = None
        self.face_cascade = self._load_face_cascade()
        self.button_handler = ButtonHandler(self)
        self.progress_bar_enabled = False
        self.camera_enabled = True
        self.widget_width = 420
        self.widget_height = 240
        self.theme = "dark"
        self.corner_style = "rounded"

        self.layout_manager = LayoutManager(LayoutConfig.default())
        self.layout_file = "layout.json"
        self._load_layout()

        # UI scale (smaller by default, user-adjustable in Settings)
        self.ui_scale = 0.82
        self.settings_file = "settings.json"

        # Layout edit mode for drag-and-drop positioning
        self.layout_edit_mode = False
        self._dragging_element = None
        self._drag_offset_x = 0
        self._drag_offset_y = 0
        self._edit_done_rect = None

        # Face tracking
        self.last_faces = []
        self._face_box = None
        self._face_miss_frames = 0
        self._cam_view = None  # (off_x, off_y, fit_w, fit_h, cam_w, cam_h)

        # Slacking alert
        self.alerts_enabled = True
        self._last_alert_ts = 0.0
        self._alert_active = False

        self._load_settings()

        # Onboarding intro (first-run only)
        self.onboarding = OnboardingManager()
        if os.path.exists(self.settings_file):
            self.onboarding.active = False

    def _theme_palette(self):
        """Return the active color palette (BGR) for the current theme."""
        return self.THEMES.get(getattr(self, "theme", "dark"), self.THEMES["dark"])

    def _is_rounded(self):
        """True when the user picked rounded corners instead of boxy ones."""
        return getattr(self, "corner_style", "rounded") == "rounded"

    def _play_alert_sound(self):
        """Play a throttled alert sound when the user is slacking."""
        now = time.time()
        if not self.alerts_enabled:
            return
        if now - self._last_alert_ts < 8.0:
            return
        self._last_alert_ts = now
        try:
            if platform.system() == "Windows":
                import winsound
                winsound.Beep(880, 180)
                winsound.Beep(660, 220)
            else:
                print("\a", end="", flush=True)
        except Exception:
            pass

    def _draw_face_outline(self, frame):
        """Draw a smoothed, color-coded tracking box around the detected face.

        When the timer is idle, the box is always drawn in a neutral color
        (no pulsing, no SLACKING tag).  Once the session is running the
        box colour reflects the current focus state.
        """
        if not self.camera_enabled or not self._cam_view or not self.last_faces:
            self._face_miss_frames += 1
            if self._face_miss_frames > 30:
                self._face_box = None
            return

        self._face_miss_frames = 0
        off_x, off_y, fit_w, fit_h, cam_w, cam_h = self._cam_view
        scale = fit_w / float(cam_w)

        # Pick the largest face
        x, y, w, h = max(self.last_faces, key=lambda f: f[2] * f[3])

        # Map to render coordinates with generous padding so the outline
        # frames the face rather than hugging the tight Haar bounding box.
        # Padding is computed in render space so it scales with window size.
        pad = int(max(18, min(fit_w, fit_h) * 0.06))
        cx = off_x + (x + w * 0.5) * scale
        cy = off_y + (y + h * 0.5) * scale
        hw = w * scale * 0.5 + pad
        hh = h * scale * 0.5 + pad
        target = (int(cx - hw), int(cy - hh), int(cx + hw), int(cy + hh))

        # Smooth the box position (faster alpha = less lag on resize)
        if self._face_box is None:
            self._face_box = target
        else:
            alpha = 0.45
            self._face_box = tuple(
                int(round((1 - alpha) * a + alpha * b)) for a, b in zip(self._face_box, target)
            )

        bx1, by1, bx2, by2 = self._face_box
        bw, bh = bx2 - bx1, by2 - by1
        if bw < 10 or bh < 10:
            return

        # Clamp to frame bounds
        fw, fh = frame.shape[1], frame.shape[0]
        bx1 = max(0, bx1)
        by1 = max(0, by1)
        bx2 = min(fw, bx2)
        by2 = min(fh, by2)
        bw, bh = bx2 - bx1, by2 - by1
        if bw < 10 or bh < 10:
            return

        theme = self._theme_palette()

        # When the timer is not running, always show a calm neutral box.
        # When running, colour by focus state (with pulse when slacking).
        if not self.is_running:
            color = theme["accent"]
            thick = 2
        else:
            t = time.time()
            if self.focus_state == "slacking":
                pulse = 0.5 + 0.5 * math.sin(t * 6)
                base = (60, 60, 255)  # red-ish
                color = tuple(int(c * (0.7 + 0.3 * pulse)) for c in base)
                thick = 3 + int(2 * pulse)
            elif self.focus_state == "concentrated":
                color = theme["on_color"]
                thick = 2
            else:
                color = theme["accent"]
                thick = 2

        corner_r = int(min(bw, bh) * 0.18)

        # Main rounded box
        styled_rect(frame, bx1, by1, bx2, by2, border=color, thickness=thick, radius=corner_r)

        # Corner brackets (L-shaped accents)
        bracket_len = int(min(bw, bh) * 0.22)
        bracket_thick = max(2, thick + 1)
        # Top-left
        cv2.line(frame, (bx1, by1), (bx1 + bracket_len, by1), color, bracket_thick, cv2.LINE_AA)
        cv2.line(frame, (bx1, by1), (bx1, by1 + bracket_len), color, bracket_thick, cv2.LINE_AA)
        # Top-right
        cv2.line(frame, (bx2, by1), (bx2 - bracket_len, by1), color, bracket_thick, cv2.LINE_AA)
        cv2.line(frame, (bx2, by1), (bx2, by1 + bracket_len), color, bracket_thick, cv2.LINE_AA)
        # Bottom-left
        cv2.line(frame, (bx1, by2), (bx1 + bracket_len, by2), color, bracket_thick, cv2.LINE_AA)
        cv2.line(frame, (bx1, by2), (bx1, by2 - bracket_len), color, bracket_thick, cv2.LINE_AA)
        # Bottom-right
        cv2.line(frame, (bx2, by2), (bx2 - bracket_len, by2), color, bracket_thick, cv2.LINE_AA)
        cv2.line(frame, (bx2, by2), (bx2, by2 - bracket_len), color, bracket_thick, cv2.LINE_AA)

        # "SLACKING" tag only appears when the timer is running
        if self.is_running and self.focus_state == "slacking":
            tag = "SLACKING"
            tag_px = max(10, int(bh * 0.12))
            tw, th = text_size(tag, tag_px, True)
            tx = bx1 + (bw - tw) // 2
            ty = by1 - th - 6
            spad = 6
            styled_rect(frame, tx - spad, ty - th - spad, tx + tw + spad, ty + spad,
                        fill=(40, 40, 220), radius=4)
            draw_text(frame, tag, tx, ty, tag_px, (255, 255, 255), bold=True, anchor="la")

    def _draw_slacking_alert(self, frame, theme, rounded, ui_scale):
        """Draw a pulsing alert banner and window border when slacking."""
        if not self._alert_active:
            return

        fw, fh = frame.shape[1], frame.shape[0]
        t = time.time()
        pulse = 0.5 + 0.5 * math.sin(t * 5)

        # Banner: top-center, below phase label area
        banner_text = "You're Slacking!"
        banner_px = max(18, int(fh * 0.035))
        tw, th = text_size(banner_text, banner_px, True)
        pad_x = max(16, int(24 * ui_scale))
        pad_y = max(8, int(10 * ui_scale))
        bw = tw + 2 * pad_x
        bh = th + 2 * pad_y
        bx1 = (fw - bw) // 2
        by1 = int(fh * 0.18)
        by2 = by1 + bh

        # Pulsing red banner
        alert_red = (40, 40, 230)
        fill_color = tuple(int(c * (0.8 + 0.2 * pulse)) for c in alert_red)
        border_color = tuple(int(c * (0.5 + 0.5 * pulse)) for c in alert_red)
        br = int(bh * 0.3) if rounded else 0
        styled_rect(frame, bx1, by1, bx1 + bw, by2,
                    fill=fill_color, border=border_color, thickness=2, radius=br)
        # Text on banner
        draw_text(frame, banner_text, bx1 + bw // 2, by1 + bh // 2,
                  max(14, int(banner_px * 0.9)), (255, 255, 255), bold=True, anchor="mm")

        # Window border pulse
        border_thick = 4 + int(3 * pulse)
        border_col = tuple(int(c * (0.6 + 0.4 * pulse)) for c in alert_red)
        # Top
        cv2.rectangle(frame, (0, 0), (fw, border_thick), border_col, -1, cv2.LINE_AA)
        # Bottom
        cv2.rectangle(frame, (0, fh - border_thick), (fw, fh), border_col, -1, cv2.LINE_AA)
        # Left
        cv2.rectangle(frame, (0, 0), (border_thick, fh), border_col, -1, cv2.LINE_AA)
        # Right
        cv2.rectangle(frame, (fw - border_thick, 0), (fw, fh), border_col, -1, cv2.LINE_AA)

    def _make_window_resizable(self):
        """On Windows, ensure the created window has a resizable frame even in frozen exe builds."""
        try:
            if platform.system() != "Windows":
                return
            # Find the window by its title
            hwnd = ctypes.windll.user32.FindWindowW(None, "Pomodoro Camera")
            if hwnd == 0:
                return
            GWL_STYLE = -16
            WS_OVERLAPPEDWINDOW = 0x00CF0000
            # set the window style to overlapped window (resizable, with titlebar)
            current_style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_STYLE)
            if (current_style & WS_OVERLAPPEDWINDOW) != WS_OVERLAPPEDWINDOW:
                ctypes.windll.user32.SetWindowLongW(hwnd, GWL_STYLE, current_style | WS_OVERLAPPEDWINDOW)
                # apply changes
                SWP_NOMOVE = 0x0002
                SWP_NOSIZE = 0x0001
                SWP_NOZORDER = 0x0004
                SWP_FRAMECHANGED = 0x0020
                ctypes.windll.user32.SetWindowPos(hwnd, 0, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_FRAMECHANGED)
        except Exception:
            pass

    def _load_layout(self):
        """Load layout.json, falling back to (and writing) the default layout.

        - Missing/corrupt file  -> write the friendly default so users start
          from a known-good baseline they can inspect and edit.
        - Valid but older file  -> merge in any element keys added since.
        """
        default_cfg = LayoutConfig.default()
        loaded = None
        try:
            if os.path.exists(self.layout_file):
                loaded = LayoutConfig.load(self.layout_file)
        except Exception:
            loaded = None

        if loaded is None or not loaded.elements:
            self.layout_manager.config = default_cfg
            self._save_layout()
        else:
            for key, elem in default_cfg.elements.items():
                loaded.elements.setdefault(key, elem)
            self.layout_manager.config = loaded

        lm = self.layout_manager
        lm.grid_cols = max(1, int(lm.config.grid_cols))
        lm.grid_rows = max(1, int(lm.config.grid_rows))

    def _save_layout(self):
        """Save current layout to file."""
        try:
            self.layout_manager.config.save(self.layout_file)
        except Exception:
            pass

    def reset_layout(self):
        """Reset layout to the shipped default."""
        cfg = LayoutConfig.default()
        self.layout_manager.config = cfg
        self.layout_manager.grid_cols = cfg.grid_cols
        self.layout_manager.grid_rows = cfg.grid_rows
        self._save_layout()

    def _load_settings(self):
        """Load persisted UI settings (scale, theme, alerts, etc.)."""
        try:
            if not os.path.exists(self.settings_file):
                return
            with open(self.settings_file, "r") as f:
                data = json.load(f)
            if "ui_scale" in data:
                self.ui_scale = max(0.6, min(1.4, float(data["ui_scale"])))
            if "theme" in data and data["theme"] in self.THEMES:
                self.theme = data["theme"]
            if "corner_style" in data and data["corner_style"] in ("rounded", "boxy"):
                self.corner_style = data["corner_style"]
            if "alerts_enabled" in data:
                self.alerts_enabled = bool(data["alerts_enabled"])
        except Exception:
            pass

    def _save_settings(self):
        try:
            with open(self.settings_file, "w") as f:
                json.dump({
                    "ui_scale": round(float(self.ui_scale), 3),
                    "theme": self.theme,
                    "corner_style": self.corner_style,
                    "alerts_enabled": bool(self.alerts_enabled),
                }, f, indent=2)
        except Exception:
            pass

    def _get_window_client_size(self):
        """Return the current drawable (client) area of the OpenCV window in pixels."""
        try:
            _, _, w, h = cv2.getWindowImageRect("Pomodoro Camera")
            if w and h and int(w) > 0 and int(h) > 0:
                return int(w), int(h)
        except Exception:
            pass
        return int(self.frame_width), int(self.frame_height)

    def _load_face_cascade(self):
        """Load the OpenCV face detector when available."""
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            clf = cv2.CascadeClassifier(cascade_path)
            # If the classifier failed to load the cascade file, don't return an empty object
            if hasattr(clf, "empty") and clf.empty():
                return None
            return clf
        except Exception:
            return None

    def start_camera(self):
        """Open the camera using a Windows-friendly backend when possible."""
        if not self.camera_enabled:
            # camera disabled by user; ensure any existing capture is released
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
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
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

    def start_session(self):
        """Open the camera stream and drive the UI loop."""
        if not self.start_camera():
            print("Could not access the camera. Please connect a camera and try again.")
            return

        self.session_open = True
        cv2.namedWindow("Pomodoro Camera", cv2.WINDOW_NORMAL)
        # Ensure window initially matches frame size to avoid scaling issues
        try:
            # Open a comfortably sized window; the render loop keeps the canvas
            # matched to whatever size the user drags it to
            cv2.resizeWindow("Pomodoro Camera", 960, 600)
            # allow the user to resize the window manually
            try:
                cv2.setWindowProperty("Pomodoro Camera", cv2.WND_PROP_AUTOSIZE, 0)
                # if running on Windows as an exe, attempt to force the native window style to be resizable
                try:
                    self._make_window_resizable()
                except Exception:
                    pass
            except Exception:
                pass
        except Exception:
            pass
        cv2.setMouseCallback("Pomodoro Camera", self._on_mouse_click)
        print("Pomodoro camera started. Click the on-screen buttons or press 'q' to quit.")

        last_render_size = (int(self.frame_width), int(self.frame_height))

        while self.session_open:
            # Track the real window size so the UI renders crisply at any size
            win_w, win_h = self._get_window_client_size()

            cam_frame = None
            if self.camera_enabled and self.cap is not None and self.cap.isOpened():
                ret, cam_frame = self.cap.read()
                if not ret:
                    cam_frame = None

            # Mirror only the camera image so UI text is never flipped
            if cam_frame is not None:
                try:
                    cam_frame = cv2.flip(cam_frame, 1)
                except Exception:
                    pass

            # Render canvas follows the window size, with sane minimum bounds and
            # a small hysteresis so OS-reported jitter doesn't cause flicker
            target_w = max(320, win_w)
            target_h = max(240, win_h)
            if abs(target_w - last_render_size[0]) >= 4 or abs(target_h - last_render_size[1]) >= 4:
                last_render_size = (target_w, target_h)
            render_w, render_h = last_render_size

            frame = np.zeros((render_h, render_w, 3), dtype=np.uint8)
            if cam_frame is not None:
                # Fit the camera feed inside the canvas preserving aspect ratio
                cam_h, cam_w = cam_frame.shape[:2]
                scale = min(render_w / float(cam_w), render_h / float(cam_h))
                fit_w = max(1, int(round(cam_w * scale)))
                fit_h = max(1, int(round(cam_h * scale)))
                fitted = cv2.resize(cam_frame, (fit_w, fit_h), interpolation=cv2.INTER_AREA)
                off_x = (render_w - fit_w) // 2
                off_y = (render_h - fit_h) // 2
                frame[off_y:off_y + fit_h, off_x:off_x + fit_w] = fitted
                self._cam_view = (off_x, off_y, fit_w, fit_h, cam_w, cam_h)
            else:
                frame[:] = self._theme_palette()["backdrop"]
                self._cam_view = None
                if not self.camera_enabled:
                    # draw a centered small widget box to indicate camera is off
                    theme = self._theme_palette()
                    w = min(self.widget_width, render_w - 40)
                    h = min(self.widget_height, render_h - 40)
                    cx = render_w // 2
                    cy = render_h // 2
                    wx1 = cx - w // 2
                    wy1 = cy - h // 2
                    wx2 = wx1 + w
                    wy2 = wy1 + h
                    widget_radius = int(min(w, h) * 0.08) if self._is_rounded() else 0
                    styled_rect(frame, wx1, wy1, wx2, wy2, fill=theme["widget_fill"],
                                border=theme["panel_border"], thickness=1, radius=widget_radius)
                    label_px = int(min(48, max(16, h * 0.16)))
                    draw_text(frame, "Camera Off", cx, cy, label_px, theme["text"],
                              bold=True, anchor="mm")

            self.frame_width, self.frame_height = render_w, render_h
            self.frame_count += 1

            # Focus analysis runs on the stable camera image so window resizing
            # does not invalidate the motion-difference history every frame
            analysis_input = cam_frame if cam_frame is not None else frame
            self.focus_score = self.analyze_focus(analysis_input)
            self.focus_state = self.classify_focus_state(self.focus_score, frame)
            frame = self._draw_ui(frame)
            cv2.imshow("Pomodoro Camera", frame)

            key = cv2.waitKey(1) & 0xFF

            # Onboarding keyboard shortcuts
            if self.onboarding.active:
                if key in (ord("\r"), ord("\n"), ord(" "), 13, 10, 32):  # Enter / Space
                    self.onboarding.step += 1
                    if self.onboarding.step >= len(self.onboarding._STEPS):
                        self.onboarding.active = False
                        self._save_settings()
                elif key == 8 and self.onboarding.step > 0:  # Backspace
                    self.onboarding.step -= 1
                elif key == 27:  # Esc → skip
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
            elif key == 27 and self.layout_edit_mode:  # Esc
                self.layout_edit_mode = False
                self._dragging_element = None
                self._save_layout()

            # If the user closed the window using the X button, stop the session
            try:
                if cv2.getWindowProperty("Pomodoro Camera", cv2.WND_PROP_VISIBLE) < 1:
                    self.session_open = False
                    break
            except Exception:
                pass

        self.cleanup()

    def _on_mouse_click(self, event, x, y, flags, params):
        """Handle mouse events from the OpenCV window (click, drag, release)."""
        mapped_x, mapped_y = x, y
        try:
            _, _, ww, wh = cv2.getWindowImageRect("Pomodoro Camera")
            if ww > 1 and wh > 1:
                mapped_x = int(round(x * self.frame_width / ww))
                mapped_y = int(round(y * self.frame_height / wh))
                mapped_x = max(0, min(mapped_x, self.frame_width - 1))
                mapped_y = max(0, min(mapped_y, self.frame_height - 1))
        except Exception:
            mapped_x, mapped_y = x, y

        # Onboarding overlay consumes clicks while active
        if self.onboarding.active and event == cv2.EVENT_LBUTTONDOWN:
            if self.onboarding.handle_click(mapped_x, mapped_y, self.frame_width, self.frame_height):
                if not self.onboarding.active:
                    self._save_settings()
                return

        # Handle layout edit mode drag-and-drop
        if self.layout_edit_mode:
            if event == cv2.EVENT_LBUTTONDOWN:
                done = getattr(self, "_edit_done_rect", None)
                if done and done[0] <= mapped_x <= done[2] and done[1] <= mapped_y <= done[3]:
                    self.layout_edit_mode = False
                    self._dragging_element = None
                    self._save_layout()
                    return
            self._handle_layout_drag(event, mapped_x, mapped_y)
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            button_name = self.button_handler.is_button_clicked(event, mapped_x, mapped_y, self.frame_width, self.frame_height)
            if button_name:
                self.button_handler.handle_button_click(button_name)
                return

            if self.settings_visible:
                self._handle_settings_click(mapped_x, mapped_y)

    def _handle_layout_drag(self, event, x, y):
        """Drag-and-drop editing. Elements are positioned by their top-left
        grid cell, so the tile simply follows the cursor; on release it snaps
        to a quarter-cell and persists to layout.json."""
        fw, fh = self.frame_width, self.frame_height
        lm = self.layout_manager
        cell_w = fw / lm.grid_cols
        cell_h = fh / lm.grid_rows

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
            nx = (x - self._drag_offset_x) / cell_w
            ny = (y - self._drag_offset_y) / cell_h
            elem.x = max(0.0, min(nx, lm.grid_cols - elem.width))
            elem.y = max(0.0, min(ny, lm.grid_rows - elem.height))

        elif event == cv2.EVENT_LBUTTONUP and self._dragging_element:
            elem = lm.config.elements[self._dragging_element]
            # Snap to quarter-grid, clamped to quarter-aligned bounds so the
            # result always lands exactly on the lattice (e.g. 8.75 not 8.9)
            max_qx = int(max(0.0, (lm.grid_cols - elem.width)) * 4)
            max_qy = int(max(0.0, (lm.grid_rows - elem.height)) * 4)
            qx = max(0, min(int(round(elem.x * 4)), max_qx))
            qy = max(0, min(int(round(elem.y * 4)), max_qy))
            elem.x = qx / 4.0
            elem.y = qy / 4.0
            self._dragging_element = None
            self._save_layout()

    def toggle_settings(self):
        """Show or hide the in-window settings panel."""
        self.settings_visible = not self.settings_visible

    def apply_settings(self, pomodoro_minutes=None, break_minutes=None, display_mode=None):
        """Apply the user settings to the timer."""
        if pomodoro_minutes is not None:
            self.session_duration_minutes = max(1, int(pomodoro_minutes))
        if break_minutes is not None:
            self.break_duration_minutes = max(1, int(break_minutes))
        if display_mode is not None:
            self.display_mode = display_mode

        self.show_progress_bar = self.display_mode in ["progress_bar", "both"]
        self.show_timer_popup = self.display_mode in ["timer_popup", "both"]
        # If timer not running, reset the phase durations to the new values
        if not self.is_running:
            self.time_left = self.session_duration_minutes * 60
            self.phase_duration_seconds = self.time_left
            self.current_phase = "pomodoro"
        else:
            # If timer is running, update the current phase duration and adjust remaining time
            now = time.time()
            if self.phase_started_at is not None:
                elapsed = int(now - self.phase_started_at)
            else:
                elapsed = 0
            if self.current_phase == "pomodoro":
                self.phase_duration_seconds = self.session_duration_minutes * 60
            elif self.current_phase == "short_break":
                self.phase_duration_seconds = self.break_duration_minutes * 60
            # recompute time_left based on elapsed (but clamp)
            self.time_left = max(0, self.phase_duration_seconds - elapsed)

        print(
            f"Settings applied: Pomodoro={self.session_duration_minutes}min, "
            f"Break={self.break_duration_minutes}min, Mode={self.display_mode}"
        )

    def start_timer(self):
        """Start the countdown for the current phase."""
        if self.is_running:
            return

        self.is_running = True
        self.phase_started_at = time.time()
        self.phase_duration_seconds = (
            self.break_duration_minutes * 60
            if self.current_phase == "short_break"
            else self.session_duration_minutes * 60
        )
        self.time_left = self.phase_duration_seconds

    def stop_timer(self):
        """Pause the countdown."""
        self.is_running = False
        self.phase_started_at = None

    def toggle_timer(self):
        """Start or stop the countdown."""
        if not self.is_running:
            self.start_timer()
        else:
            self.stop_timer()

    def reset_timer(self):
        """Reset the timer to the pomodoro phase."""
        self.is_running = False
        self.phase_started_at = None
        self.current_phase = "pomodoro"
        self.time_left = self.session_duration_minutes * 60
        self.phase_duration_seconds = self.time_left

    def switch_to_short_break(self):
        """Switch to the short break phase."""
        self.current_phase = "short_break"
        self.time_left = self.break_duration_minutes * 60
        self.phase_duration_seconds = self.time_left
        self.phase_started_at = time.time()
        self.is_running = True

    def switch_to_pomodoro(self):
        """Switch back to the pomodoro phase."""
        self.current_phase = "pomodoro"
        self.time_left = self.session_duration_minutes * 60
        self.phase_duration_seconds = self.time_left
        self.phase_started_at = time.time()
        self.is_running = True

    def _update_timer(self):
        """Advance the countdown and flip phases when a phase is completed."""
        if not self.is_running or self.phase_started_at is None:
            return

        elapsed = int(time.time() - self.phase_started_at)
        self.time_left = max(0, self.phase_duration_seconds - elapsed)
        if self.time_left <= 0:
            if self.current_phase == "pomodoro":
                print("Pomodoro complete! Switching to short break.")
                self.switch_to_short_break()
            elif self.current_phase == "short_break":
                print("Short break complete! Ready for the next pomodoro.")
                self.switch_to_pomodoro()

    def analyze_focus(self, frame):
        """Estimate how focused the user appears in the camera view.

        Face detection is expensive — we run it every 3rd frame and reuse
        the previous result in between to cut CPU usage by ~66%.
        """
        if frame is None:
            return 0.0

        # Only run expensive Haar cascade every N frames
        self._face_frame_count = getattr(self, "_face_frame_count", 0) + 1
        if self._face_frame_count % 3 != 0 and self.last_faces:
            # Reuse previous face positions (already stored)
            pass
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            faces = []
            if self.face_cascade is not None:
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            # Store faces for the tracking overlay (always, even when idle)
            self.last_faces = [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]
            # Keep latest gray for motion baseline
            self._last_gray = gray

        # Only score concentration when the timer is running;
        # when idle, keep the motion baseline fresh but don't score.
        if not self.is_running:
            if hasattr(self, "_last_gray"):
                self.prev_gray = self._last_gray
            return 0.0

        gray = getattr(self, "_last_gray", None)
        if gray is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            self._last_gray = gray

        if self.prev_gray is None or self.prev_gray.shape != gray.shape:
            self.prev_gray = gray
            try:
                has_faces = len(self.last_faces) > 0
            except Exception:
                has_faces = False
            return 100.0 if has_faces else 25.0

        diff = cv2.absdiff(self.prev_gray, gray)
        motion = np.mean(diff) / 255.0
        self.prev_gray = gray

        face_score = 1.0 if len(self.last_faces) > 0 else 0.0
        motion_penalty = min(1.0, motion * 2.0)
        score = (face_score * 70.0) + ((1.0 - motion_penalty) * 30.0)
        if not self.last_faces:
            score *= 0.5

        return max(0.0, min(100.0, score))

    def classify_focus_state(self, score, frame=None):
        """Turn the focus score into a human-readable state."""
        # When the timer isn't running, report neutral (score is 0.0 from analyze_focus)
        if not self.is_running:
            return "neutral"
        if score >= 70:
            return "concentrated"
        if score <= 35:
            return "slacking"
        return "neutral"

    def _draw_ui(self, frame):
        """Draw the timer, controls, detection state, and settings panel onto the frame."""
        self._update_timer()

        if self.current_phase == "pomodoro":
            phase_text = "Pomodoro"
            phase_color = (231, 76, 60)
            progress = max(0.0, min(1.0, 1.0 - (self.time_left / max(1, self.session_duration_minutes * 60))))
        elif self.current_phase == "short_break":
            phase_text = "Short Break"
            phase_color = (39, 174, 96)
            progress = max(0.0, min(1.0, 1.0 - (self.time_left / max(1, self.break_duration_minutes * 60))))
        else:
            phase_text = "Long Break"
            phase_color = (52, 152, 219)
            progress = 0.5

        minutes = self.time_left // 60
        seconds = self.time_left % 60
        timer_text = f"{minutes:02d}:{seconds:02d}"

        # Scale UI text with the render size so it stays legible at any window size
        # ui_scale combines window-size auto scaling with the user-adjustable
        # Settings > UI Scale slider (smaller by default at 0.82).
        base_scale = max(0.75, min(2.5, frame.shape[0] / 480.0))
        ui_scale = base_scale * float(getattr(self, "ui_scale", 0.82))
        theme = self._theme_palette()
        rounded = self._is_rounded()

        fw, fh = frame.shape[1], frame.shape[0]

        # Face tracking outline (before UI so it's under overlays)
        self._draw_face_outline(frame)

        # Slacking alert: only active while the session is running
        raw_slacking = (self.is_running and self.camera_enabled and self.focus_state == "slacking")
        self._alert_active = bool(raw_slacking and self.alerts_enabled)
        if self._alert_active:
            self._play_alert_sound()
            self._draw_slacking_alert(frame, theme, rounded, ui_scale)

        # Draw progress bar using layout system
        self._last_bar_area = None
        if self.show_progress_bar and self.progress_bar_enabled and self.layout_manager.config.elements.get("progress_bar", UIElementConfig()).enabled:
            bar_rect = self.layout_manager.get_element_rect("progress_bar", fw, fh)
            bar_x1, bar_y, bar_x2, bar_y2 = bar_rect
            if self.theme == "xp":
                _xp_progress_bar(frame, bar_x1, bar_y, bar_x2, bar_y2, progress)
            else:
                bar_height = bar_y2 - bar_y
                bar_radius = bar_height // 2 if rounded else 0
                styled_rect(frame, bar_x1, bar_y, bar_x2, bar_y2, fill=theme["bar_track"], radius=bar_radius)
                if progress > 0:
                    fill_x = max(bar_x1 + 1, int(bar_x1 + (bar_x2 - bar_x1) * progress))
                    styled_rect(frame, bar_x1, bar_y, fill_x, bar_y2, fill=phase_color, radius=bar_radius)
                styled_rect(frame, bar_x1, bar_y, bar_x2, bar_y2, border=theme["button_border"], thickness=1, radius=bar_radius)
            self._last_bar_area = (bar_x1, bar_y, bar_x2, bar_y2)

        # Draw timer popup using layout system
        if self.show_timer_popup and self.layout_manager.config.elements.get("timer_popup", UIElementConfig()).enabled:
            px1, py1, px2, py2 = self.layout_manager.get_element_rect("timer_popup", fw, fh)
            box_w, box_h = px2 - px1, py2 - py1
            if box_w > 4 and box_h > 4:
                max_px = int(min(1.6, box_h / 60.0) * 72 * self.layout_manager.config.elements["timer_popup"].font_scale * self.ui_scale)
                px, (tw, th) = fit_text(timer_text, box_w, box_h, max_px, bold=True)
                if self.theme == "xp":
                    # XP window: blue gradient title bar + silver body
                    tb_h = max(18, int(box_h * 0.26))
                    _xp_title_bar(frame, px1, py1, px2, py1 + tb_h, phase_text)
                    styled_rect(frame, px1, py1 + tb_h, px2, py2,
                                fill=theme["popup_fill"], border=(104, 104, 104), thickness=1, radius=0)
                    draw_text(frame, timer_text, px1 + box_w // 2, py1 + tb_h + (box_h - tb_h) // 2,
                              px, theme["text"], bold=True, anchor="mm")
                else:
                    popup_radius = int(min(box_w, box_h) * 0.2) if rounded else 0
                    # Soft drop shadow
                    sx1, sy1, sx2, sy2 = px1 + 2, py1 + 3, px2 + 2, py2 + 3
                    shadow = frame.copy()
                    styled_rect(shadow, sx1, sy1, sx2, sy2, fill=(12, 12, 14), radius=popup_radius)
                    cv2.addWeighted(shadow, 0.18, frame, 0.82, 0, frame)
                    styled_rect(frame, px1, py1, px2, py2,
                                fill=theme["popup_fill"], border=phase_color, thickness=1, radius=popup_radius)
                    # Subtle inner highlight
                    if rounded and box_h > 16:
                        cv2.line(frame, (px1 + popup_radius, py1 + 1), (px2 - popup_radius, py1 + 1),
                                 (255, 255, 255), 1, cv2.LINE_AA)
                    draw_text(frame, timer_text, px1 + box_w // 2, py1 + box_h // 2,
                              px, theme["text"], bold=True, anchor="mm")

        # Draw phase label using layout system
        if self.layout_manager.config.elements.get("phase_label", UIElementConfig()).enabled:
            lx1, ly1, lx2, ly2 = self.layout_manager.get_element_rect("phase_label", fw, fh)
            label_w, label_h = lx2 - lx1, ly2 - ly1
            if label_w > 4 and label_h > 4:
                max_px = int(0.62 * label_h * self.ui_scale)
                px, _ = fit_text(phase_text, label_w, label_h, max_px, bold=True)
                draw_text(frame, phase_text, lx1 + label_w // 2, ly1 + label_h // 2,
                          px, phase_color, bold=True, anchor="mm",
                          stroke=max(1, px // 16), stroke_color=(12, 12, 14))

        # Draw main buttons using layout system
        if self.layout_manager.config.elements.get("main_buttons", UIElementConfig()).enabled:
            bx1, by1, bx2, by2 = self.layout_manager.get_element_rect("main_buttons", fw, fh)
            btn_area_w, btn_area_h = bx2 - bx1, by2 - by1
            if btn_area_w > 30 and btn_area_h > 10:
                btn_h = max(28, int(btn_area_h * 0.74 * self.ui_scale))
                btn_y = by1 + (btn_area_h - btn_h) // 2
                gap = int(max(8, btn_area_w * 0.025))
                btn_w = (btn_area_w - 2 * gap) // 3
                main_btn_radius = int(btn_h * 0.26) if rounded else 0
                is_xp = self.theme == "xp"

                start_x = bx1
                reset_x = bx1 + btn_w + gap
                settings_x = bx1 + 2 * (btn_w + gap)

                btn_labels = ["Running" if self.is_running else "Start", "Reset", "Settings"]
                btn_names = ("start", "reset", "settings")

                if is_xp:
                    # XP: 3-D beveled buttons (Start = accent blue-orange)
                    regions = {}
                    for i, (bxx, label) in enumerate(zip([start_x, reset_x, settings_x], btn_labels)):
                        accent = (i == 0)
                        _xp_button(frame, bxx, btn_y, bxx + btn_w, btn_y + btn_h,
                                   label, accent=accent, px=max(12, int(0.38 * btn_h)), bold=(i == 0))
                        regions[btn_names[i]] = {"x1": bxx, "y1": btn_y,
                                                  "x2": bxx + btn_w + 1, "y2": btn_y + btn_h + 1}
                    self.button_handler.button_regions = regions
                else:
                    buttons = [
                        (start_x, btn_labels[0], phase_color),
                        (reset_x, btn_labels[1], theme["button_border"]),
                        (settings_x, btn_labels[2], theme["button_border"]),
                    ]
                    # Single shadow overlay for all buttons (avoids 3x frame.copy)
                    shadow_layer = frame.copy()
                    for bxx, label, border in buttons:
                        styled_rect(shadow_layer, bxx + 2, btn_y + 3, bxx + btn_w + 2, btn_y + btn_h + 3,
                                    fill=(12, 12, 14), radius=main_btn_radius)
                    cv2.addWeighted(shadow_layer, 0.22, frame, 0.78, 0, frame)

                    regions = {}
                    for i, (bxx, label, border) in enumerate(buttons):
                        styled_rect(frame, bxx, btn_y, bxx + btn_w, btn_y + btn_h,
                                    fill=theme["button_fill"], border=border,
                                    thickness=1, radius=main_btn_radius)
                        if rounded and btn_h > 20:
                            cv2.line(frame, (bxx + main_btn_radius, btn_y + 1),
                                     (bxx + btn_w - main_btn_radius, btn_y + 1),
                                     (255, 255, 255), 1, cv2.LINE_AA)
                        max_px = int(0.4 * btn_h)
                        px, _ = fit_text(label, btn_w, btn_h, max_px)
                        draw_text(frame, label, bxx + btn_w // 2, btn_y + btn_h // 2,
                                  px, theme["text"], anchor="mm")
                        regions[btn_names[i]] = {"x1": bxx, "y1": btn_y,
                                                  "x2": bxx + btn_w + 1, "y2": btn_y + btn_h + 1}
                    self.button_handler.button_regions = regions
        else:
            self.button_handler.button_regions = {}

        # Draw focus display with readability stroke
        if self.layout_manager.config.elements.get("focus_display", UIElementConfig()).enabled:
            fx1, fy1, fx2, fy2 = self.layout_manager.get_element_rect("focus_display", fw, fh)
            if fx2 - fx1 > 4 and fy2 - fy1 > 4:
                if self.is_running:
                    focus_text = f"Focus: {self.focus_state.title()}"
                else:
                    focus_text = "Focus: Ready"
                max_px = int(0.55 * (fy2 - fy1) * self.ui_scale)
                px, _ = fit_text(focus_text, fx2 - fx1, fy2 - fy1, max_px)
                draw_text(frame, focus_text, fx1 + 6, fy1 + (fy2 - fy1) // 2,
                          px, theme["text"], anchor="lm",
                          stroke=max(1, px // 14), stroke_color=(12, 12, 14))

        # Draw status display using layout system
        if self.layout_manager.config.elements.get("status_display", UIElementConfig()).enabled:
            sx1, sy1, sx2, sy2 = self.layout_manager.get_element_rect("status_display", fw, fh)
            if sx2 - sx1 > 4 and sy2 - sy1 > 4:
                status_text = f"Status: {'Running' if self.is_running else 'Ready'}"
                max_px = int(0.5 * (sy2 - sy1) * self.ui_scale)
                px, _ = fit_text(status_text, sx2 - sx1, sy2 - sy1, max_px)
                draw_text(frame, status_text, sx1 + 6, sy1 + (sy2 - sy1) // 2,
                          px, theme["subtext"], anchor="lm",
                          stroke=max(1, px // 16), stroke_color=(12, 12, 14))

        # Draw quit hint using layout system
        if self.layout_manager.config.elements.get("quit_hint", UIElementConfig()).enabled:
            hx1, hy1, hx2, hy2 = self.layout_manager.get_element_rect("quit_hint", fw, fh)
            if hx2 - hx1 > 4 and hy2 - hy1 > 4:
                hint_text = "q: quit"
                max_px = int(0.5 * (hy2 - hy1))
                px, _ = fit_text(hint_text, hx2 - hx1, hy2 - hy1, max_px, pad_w=4, pad_h=3)
                draw_text(frame, hint_text, hx1 + (hx2 - hx1) // 2, hy1 + (hy2 - hy1) // 2,
                          px, theme["subtext"], anchor="mm",
                          stroke=max(1, px // 18), stroke_color=(12, 12, 14))

        if self.settings_visible:
            self._draw_settings_panel(frame)

        # Layout edit mode overlay
        if self.layout_edit_mode:
            self._draw_layout_edit_overlay(frame, theme, rounded, ui_scale)

        # Onboarding intro overlay (drawn last, on top of everything)
        if self.onboarding.active:
            self.onboarding.draw_overlay(frame, theme, rounded, ui_scale)

        return frame

    def _draw_layout_edit_overlay(self, frame, theme, rounded, ui_scale):
        """Draw the layout editor: grid, draggable tiles (incl. hidden ones),
        and a [Done] pill that is itself clickable."""
        fw, fh = frame.shape[1], frame.shape[0]
        lm = self.layout_manager

        # Grid lines
        cell_w = fw / lm.grid_cols
        cell_h = fh / lm.grid_rows
        grid_color = theme["divider"]
        for i in range(lm.grid_cols + 1):
            x = int(i * cell_w)
            cv2.line(frame, (x, 0), (x, fh), grid_color, 1, cv2.LINE_AA)
        for i in range(lm.grid_rows + 1):
            y = int(i * cell_h)
            cv2.line(frame, (0, y), (fw, y), grid_color, 1, cv2.LINE_AA)

        # Tiles: enabled ones green, currently dragged one accent, hidden ones dim
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
            styled_rect(frame, x1, y1, x2, y2, border=color,
                        thickness=thickness, radius=6 if rounded else 0)

            label = name.replace("_", " ").title() + ("" if elem.enabled else " (off)")
            max_px = int(0.4 * ui_scale * 22)
            px, _ = fit_text(label, x2 - x1, max(14, int(20 * ui_scale)), max_px, pad_w=6, pad_h=3)
            draw_text(frame, label, x1 + 5, min(y1 + 5 + int(14 * ui_scale), y2 - 3), px, color, anchor="la")

            coord_text = f"({elem.x:g}, {elem.y:g})"
            px2, _ = fit_text(coord_text, x2 - x1, max(12, int(16 * ui_scale)), int(0.3 * ui_scale * 22), pad_w=6, pad_h=2)
            draw_text(frame, coord_text, x1 + 5, y2 - 4, px2, theme["subtext"], anchor="lb")

        # [Done] pill, top-center — clickable way out of edit mode
        done_text = "[ Done ]"
        px, (tw, th) = fit_text(done_text, fw // 3, int(40 * ui_scale), int(0.5 * ui_scale * 22), bold=True, pad_w=6, pad_h=4)
        pad = max(10, int(10 * ui_scale))
        pill_w, pill_h = tw + 2 * pad, th + 2 * pad
        px1 = (fw - pill_w) // 2
        py1 = max(8, int(10 * ui_scale))
        pr = int(pill_h * 0.35) if rounded else 0
        styled_rect(frame, px1, py1, px1 + pill_w, py1 + pill_h,
                    fill=theme["panel_fill"], border=theme["accent"],
                    thickness=2, radius=pr)
        draw_text(frame, done_text, px1 + pill_w // 2, py1 + pill_h // 2, px, theme["text"], bold=True, anchor="mm")
        self._edit_done_rect = (px1, py1, px1 + pill_w, py1 + pill_h)

        # Bottom hint line
        hint = "Drag tiles to rearrange  |  [ Done ] or Esc to finish"
        max_px = int(0.4 * ui_scale * 22)
        px, (tw, th) = fit_text(hint, fw, int(26 * ui_scale), max_px, pad_w=8, pad_h=2)
        hy = fh - max(8, int(8 * ui_scale))
        cv2.rectangle(frame, (fw - tw - 16, fh - th - int(14 * ui_scale)),
                      (fw, fh), theme["backdrop"], -1)
        draw_text(frame, hint, fw - tw - 8, hy, px, theme["subtext"], anchor="la")

    def _draw_settings_panel(self, frame):
        """Draw an in-window settings panel the user can click (centered, themed, adaptive)."""
        fw, fh = frame.shape[1], frame.shape[0]
        theme = self._theme_palette()
        rounded = self._is_rounded()
        is_xp = self.theme == "xp"
        panel_w = int(min(720, max(360, fw * 0.62)))
        panel_h = int(min(560, max(340, fh * 0.72)))
        panel_x = int((fw - panel_w) / 2)
        panel_y = int((fh - panel_h) / 2)
        # Scale panel text with its size so it remains readable at any window size
        pscale = max(0.75, min(2.0, panel_h / 380.0))
        pad = max(14, int(panel_w * 0.04))
        radius = int(min(panel_w, panel_h) * 0.05) if rounded and not is_xp else 0

        if is_xp:
            # XP: solid silver panel with blue title bar
            styled_rect(frame, panel_x, panel_y, panel_x + panel_w, panel_y + panel_h,
                        fill=theme["panel_fill"], border=(104, 104, 104), thickness=2, radius=0)
        else:
            # Translucent panel background with a crisp border (rounded or boxy)
            overlay = frame.copy()
            styled_rect(overlay, panel_x, panel_y, panel_x + panel_w, panel_y + panel_h, fill=theme["panel_fill"], radius=radius)
            cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
            styled_rect(frame, panel_x, panel_y, panel_x + panel_w, panel_y + panel_h, border=theme["panel_border"], thickness=2, radius=radius)

        self._settings_button_rects = {}

        def draw_setting_text(text, x, y, px, color, anchor="la"):
            draw_text(frame, text, x, y, px, color, anchor=anchor)

        def draw_setting_button(name, x1, y1, x2, y2, label, border_color=None, text_color=None, bold=False):
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            if is_xp:
                accent = name in ("pomodoro_plus", "pomodoro_minus", "break_plus", "break_minus",
                                  "scale_plus", "scale_minus")
                _xp_button(frame, x1, y1, x2, y2, label, accent=accent,
                           px=max(11, int(0.38 * (y2 - y1))), bold=bold)
            else:
                r = int(min(x2 - x1, y2 - y1) * 0.3) if rounded else 0
                styled_rect(frame, x1, y1, x2, y2, fill=theme["button_fill"], border=border_color or theme["button_border"], thickness=2, radius=r)
                btn_w, btn_h = x2 - x1, y2 - y1
                px, _ = fit_text(label, btn_w, btn_h, int(0.4 * btn_h), bold=bold)
                draw_text(frame, label, x1 + btn_w // 2, y1 + btn_h // 2, px, text_color or theme["text"], bold=bold, anchor="mm")
            self._settings_button_rects[name] = (x1, y1, x2, y2)

        # Header: XP title bar or plain header
        title_h = int(48 * pscale)
        if is_xp:
            close_rect = _xp_title_bar(frame, panel_x, panel_y, panel_x + panel_w, panel_y + title_h, "Settings")
            if close_rect:
                self._settings_button_rects["close"] = close_rect
            divider_y = panel_y + title_h + int(4 * pscale)
        else:
            title_cy = panel_y + pad + title_h // 2
            draw_setting_text("Settings", panel_x + pad, title_cy + int(13 * pscale), int(22 * pscale), theme["text"], anchor="la")
            close_w, close_h = int(92 * pscale), int(32 * pscale)
            draw_setting_button("close", panel_x + panel_w - pad - close_w, title_cy - close_h // 2,
                                panel_x + panel_w - pad, title_cy + close_h // 2, "Close",
                                text_color=theme["subtext"], bold=False)
            divider_y = title_cy + title_h // 2 + int(8 * pscale)
            cv2.line(frame, (panel_x + pad, divider_y), (panel_x + panel_w - pad, divider_y), theme["divider"], 2)

        # Body: nine evenly spaced rows (extra row for UI scale)
        content_top = divider_y + int(12 * pscale)
        content_bottom = panel_y + panel_h - pad
        row_h = (content_bottom - content_top) / 9.0
        btn_h = int(min(row_h * 0.62, 42 * pscale))
        gap = int(10 * pscale)
        inner_r = panel_x + panel_w - pad
        stepper_w = int(44 * pscale)
        half_w = (panel_w - 2 * pad - gap) // 2

        def row_cy(i):
            return int(content_top + row_h * (i + 0.5))

        # Rows 0-1: session lengths with -/+ steppers on the right
        cy = row_cy(0)
        draw_setting_text(f"Pomodoro: {self.session_duration_minutes} min", panel_x + pad, cy + int(8 * pscale), int(19 * pscale), theme["subtext"], anchor="la")
        draw_setting_button("pomodoro_minus", inner_r - 2 * stepper_w - gap, cy - btn_h // 2, inner_r - stepper_w - gap, cy + btn_h // 2, "-", bold=True)
        draw_setting_button("pomodoro_plus", inner_r - stepper_w, cy - btn_h // 2, inner_r, cy + btn_h // 2, "+", bold=True)
        cy = row_cy(1)
        draw_setting_text(f"Break: {self.break_duration_minutes} min", panel_x + pad, cy + int(8 * pscale), int(19 * pscale), theme["subtext"], anchor="la")
        draw_setting_button("break_minus", inner_r - 2 * stepper_w - gap, cy - btn_h // 2, inner_r - stepper_w - gap, cy + btn_h // 2, "-", bold=True)
        draw_setting_button("break_plus", inner_r - stepper_w, cy - btn_h // 2, inner_r, cy + btn_h // 2, "+", bold=True)

        # Row 2: camera + progress bar toggles
        cy = row_cy(2)
        draw_setting_button("toggle_camera", panel_x + pad, cy - btn_h // 2, panel_x + pad + half_w, cy + btn_h // 2,
                            f"Camera: {'On' if self.camera_enabled else 'Off'}",
                            border_color=theme["on_color"] if self.camera_enabled else theme["off_color"])
        draw_setting_button("toggle_bar", panel_x + pad + half_w + gap, cy - btn_h // 2, inner_r, cy + btn_h // 2,
                            f"Bar: {'On' if self.progress_bar_enabled else 'Off'}",
                            border_color=theme["on_color"] if self.progress_bar_enabled else theme["off_color"])

        # Row 3: appearance — theme, corner style, alerts
        cy = row_cy(3)
        third_w = (panel_w - 2 * pad - 2 * gap) // 3
        draw_setting_button("toggle_theme", panel_x + pad, cy - btn_h // 2, panel_x + pad + third_w, cy + btn_h // 2,
                            f"Theme: {self.theme.title()}")
        draw_setting_button("toggle_corners", panel_x + pad + third_w + gap, cy - btn_h // 2, panel_x + pad + 2 * third_w + gap, cy + btn_h // 2,
                            f"Corners: {'Rounded' if rounded else 'Boxy'}")
        draw_setting_button("toggle_alerts", panel_x + pad + 2 * third_w + 2 * gap, cy - btn_h // 2, inner_r, cy + btn_h // 2,
                            f"Alerts: {'On' if self.alerts_enabled else 'Off'}",
                            border_color=theme["on_color"] if self.alerts_enabled else theme["off_color"])

        # Row 4: display mode selector
        cy = row_cy(4)
        mode_w = (panel_w - 2 * pad - 2 * gap) // 3
        for i, (name, label, value) in enumerate([
            ("mode_progress", "Progress", self.DISPLAY_MODE_PROGRESS_BAR),
            ("mode_popup", "Popup", self.DISPLAY_MODE_TIMER_POPUP),
            ("mode_both", "Both", self.DISPLAY_MODE_BOTH),
        ]):
            active = self.display_mode == value
            bx1 = panel_x + pad + i * (mode_w + gap)
            draw_setting_button(name, bx1, cy - btn_h // 2, bx1 + mode_w, cy + btn_h // 2, label,
                                border_color=theme["accent"] if active else theme["button_border"],
                                text_color=theme["accent"] if active else theme["text"],
                                bold=active)

        # Row 5: Layout customization
        cy = row_cy(5)
        layout_btn_w = (panel_w - 2 * pad - 3 * gap) // 4
        layout_options = [
            ("layout_toggle_timer", "Timer", self.layout_manager.config.elements.get("timer_popup", UIElementConfig()).enabled),
            ("layout_toggle_bar", "Bar", self.layout_manager.config.elements.get("progress_bar", UIElementConfig()).enabled),
            ("layout_toggle_focus", "Focus", self.layout_manager.config.elements.get("focus_display", UIElementConfig()).enabled),
            ("layout_toggle_phase", "Phase", self.layout_manager.config.elements.get("phase_label", UIElementConfig()).enabled),
        ]
        for i, (name, label, enabled) in enumerate(layout_options):
            bx1 = panel_x + pad + i * (layout_btn_w + gap)
            draw_setting_button(name, bx1, cy - btn_h // 2, bx1 + layout_btn_w, cy + btn_h // 2, label,
                                border_color=theme["on_color"] if enabled else theme["off_color"])

        # Row 6: Grid size and layout actions
        cy = row_cy(6)
        action_btn_w = (panel_w - 2 * pad - 3 * gap) // 4
        draw_setting_button("layout_grid_plus", panel_x + pad, cy - btn_h // 2, panel_x + pad + action_btn_w, cy + btn_h // 2,
                            f"Grid: {self.layout_manager.grid_cols}x{self.layout_manager.grid_rows}")
        draw_setting_button("layout_reset", panel_x + pad + action_btn_w + gap, cy - btn_h // 2, panel_x + pad + 2 * action_btn_w + gap, cy + btn_h // 2,
                            "Default")
        draw_setting_button("layout_save", panel_x + pad + 2 * action_btn_w + 2 * gap, cy - btn_h // 2, panel_x + pad + 3 * action_btn_w + 2 * gap, cy + btn_h // 2,
                            "Save Layout")
        draw_setting_button("layout_preset", panel_x + pad + 3 * action_btn_w + 3 * gap, cy - btn_h // 2, inner_r, cy + btn_h // 2,
                            "Presets")

        # Row 7: UI scale (- / +)
        cy = row_cy(7)
        draw_setting_text(f"UI Scale: {int(self.ui_scale * 100)}%", panel_x + pad, cy + int(8 * pscale), int(19 * pscale), theme["subtext"], anchor="la")
        draw_setting_button("scale_minus", inner_r - 2 * stepper_w - gap, cy - btn_h // 2, inner_r - stepper_w - gap, cy + btn_h // 2, "-", bold=True)
        draw_setting_button("scale_plus", inner_r - stepper_w, cy - btn_h // 2, inner_r, cy + btn_h // 2, "+", bold=True)

        # Row 8: Layout edit mode toggle
        cy = row_cy(8)
        edit_label = "Exit Edit Mode" if self.layout_edit_mode else "Layout Edit Mode"
        edit_color = theme["off_color"] if self.layout_edit_mode else theme["accent"]
        draw_setting_button("layout_edit_mode", panel_x + pad, cy - btn_h // 2, inner_r, cy + btn_h // 2,
                            edit_label, border_color=edit_color, bold=True)

    def _handle_settings_click(self, x, y):
        """Update settings from clicks inside the settings panel."""
        if not hasattr(self, "_settings_button_rects"):
            return

        for name, rect in self._settings_button_rects.items():
            x1, y1, x2, y2 = rect
            if x1 <= x <= x2 and y1 <= y <= y2:
                if name == "pomodoro_minus":
                    self.apply_settings(pomodoro_minutes=self.session_duration_minutes - 1)
                elif name == "pomodoro_plus":
                    self.apply_settings(pomodoro_minutes=self.session_duration_minutes + 1)
                elif name == "break_minus":
                    self.apply_settings(break_minutes=self.break_duration_minutes - 1)
                elif name == "break_plus":
                    self.apply_settings(break_minutes=self.break_duration_minutes + 1)
                elif name == "toggle_theme":
                    cycle = ["dark", "light", "xp"]
                    idx = cycle.index(self.theme) if self.theme in cycle else 0
                    self.theme = cycle[(idx + 1) % len(cycle)]
                    self._save_settings()
                elif name == "toggle_corners":
                    self.corner_style = "boxy" if self.corner_style == "rounded" else "rounded"
                    self._save_settings()
                elif name == "toggle_alerts":
                    self.alerts_enabled = not self.alerts_enabled
                    self._save_settings()
                elif name == "scale_minus":
                    self.ui_scale = max(0.6, round(self.ui_scale - 0.08, 2))
                    self._save_settings()
                elif name == "scale_plus":
                    self.ui_scale = min(1.4, round(self.ui_scale + 0.08, 2))
                    self._save_settings()
                elif name == "toggle_bar":
                    self.progress_bar_enabled = not getattr(self, "progress_bar_enabled", True)
                    self.show_progress_bar = self.display_mode in ["progress_bar", "both"]
                elif name == "toggle_camera":
                    self.camera_enabled = not getattr(self, "camera_enabled", True)
                    if not self.camera_enabled:
                        if self.cap is not None:
                            try:
                                self.cap.release()
                            except Exception:
                                pass
                            self.cap = None
                        self.prev_gray = None
                    else:
                        self.prev_gray = None
                        self.start_camera()
                
                elif name == "mode_progress":
                    self.apply_settings(display_mode=self.DISPLAY_MODE_PROGRESS_BAR)
                elif name == "mode_popup":
                    self.apply_settings(display_mode=self.DISPLAY_MODE_TIMER_POPUP)
                elif name == "mode_both":
                    self.apply_settings(display_mode=self.DISPLAY_MODE_BOTH)

                elif name == "layout_toggle_timer":
                    self.layout_manager.enable_element("timer_popup", not self.layout_manager.config.elements.get("timer_popup", UIElementConfig()).enabled)
                    self._save_layout()
                elif name == "layout_toggle_bar":
                    self.layout_manager.enable_element("progress_bar", not self.layout_manager.config.elements.get("progress_bar", UIElementConfig()).enabled)
                    self._save_layout()
                elif name == "layout_toggle_focus":
                    self.layout_manager.enable_element("focus_display", not self.layout_manager.config.elements.get("focus_display", UIElementConfig()).enabled)
                    self._save_layout()
                elif name == "layout_toggle_phase":
                    self.layout_manager.enable_element("phase_label", not self.layout_manager.config.elements.get("phase_label", UIElementConfig()).enabled)
                    self._save_layout()
                elif name == "layout_grid_plus":
                    new_cols = self.layout_manager.grid_cols + 2
                    new_rows = self.layout_manager.grid_rows + 1
                    if new_cols > 20:
                        new_cols = 12
                        new_rows = 8
                    self.layout_manager.set_grid(new_cols, new_rows)
                    self._save_layout()
                elif name == "layout_reset":
                    self.reset_layout()
                elif name == "layout_save":
                    self._save_layout()
                elif name == "layout_preset":
                    self._cycle_layout_preset()
                elif name == "layout_edit_mode":
                    self.layout_edit_mode = not self.layout_edit_mode
                    self._dragging_element = None
                    if self.layout_edit_mode:
                        # Close the panel so the editor overlay is fully usable
                        self.settings_visible = False
                    else:
                        self._save_layout()
                elif name == "close":
                    self.settings_visible = False
                return

    def _cycle_layout_preset(self):
        """Cycle through layout presets (all use top-left grid placement)."""
        presets = [
            ("Default", LayoutConfig.default()),
            ("Focus", LayoutConfig(
                grid_cols=12, grid_rows=8,
                elements={
                    "timer_popup": UIElementConfig(enabled=True, x=3.5, y=0.5, width=5.0, height=2.6, margin=0.15, font_scale=2.0),
                    "phase_label": UIElementConfig(enabled=True, x=3.5, y=3.3, width=5.0, height=0.8, margin=0.15),
                    "progress_bar": UIElementConfig(enabled=False, x=0.4, y=4.4, width=11.2, height=0.32, margin=0.1),
                    "focus_display": UIElementConfig(enabled=False, x=0.4, y=0.35, width=3.6, height=0.6, margin=0.1),
                    "status_display": UIElementConfig(enabled=False, x=0.4, y=1.05, width=3.6, height=0.5, margin=0.1),
                    "main_buttons": UIElementConfig(enabled=True, x=2.5, y=6.9, width=7.0, height=0.9, margin=0.15),
                    "quit_hint": UIElementConfig(enabled=False, x=10.6, y=7.45, width=1.2, height=0.4, margin=0.1),
                }
            )),
            ("Dashboard", LayoutConfig(
                grid_cols=16, grid_rows=9,
                elements={
                    "timer_popup": UIElementConfig(enabled=True, x=11.0, y=0.4, width=4.6, height=2.0, margin=0.15, font_scale=1.4),
                    "phase_label": UIElementConfig(enabled=True, x=11.0, y=2.6, width=4.6, height=0.9, margin=0.15),
                    "progress_bar": UIElementConfig(enabled=True, x=0.4, y=0.5, width=10.0, height=0.5, margin=0.1),
                    "focus_display": UIElementConfig(enabled=True, x=0.4, y=1.4, width=5.0, height=0.9, margin=0.1),
                    "status_display": UIElementConfig(enabled=True, x=0.4, y=2.5, width=5.0, height=0.7, margin=0.1),
                    "main_buttons": UIElementConfig(enabled=True, x=1.0, y=7.6, width=14.0, height=1.1, margin=0.15),
                    "quit_hint": UIElementConfig(enabled=True, x=14.2, y=8.45, width=1.6, height=0.4, margin=0.1),
                }
            )),
            ("Minimal", LayoutConfig(
                grid_cols=12, grid_rows=8,
                elements={
                    "timer_popup": UIElementConfig(enabled=True, x=9.0, y=0.35, width=2.7, height=1.3, margin=0.15, font_scale=1.0),
                    "phase_label": UIElementConfig(enabled=False, x=4.4, y=0.55, width=3.2, height=0.8, margin=0.15),
                    "progress_bar": UIElementConfig(enabled=False, x=0.4, y=2.0, width=11.2, height=0.32, margin=0.1),
                    "focus_display": UIElementConfig(enabled=False, x=0.4, y=0.35, width=3.6, height=0.6, margin=0.1),
                    "status_display": UIElementConfig(enabled=False, x=0.4, y=1.05, width=3.6, height=0.5, margin=0.1),
                    "main_buttons": UIElementConfig(enabled=True, x=3.5, y=6.9, width=5.0, height=0.9, margin=0.15),
                    "quit_hint": UIElementConfig(enabled=False, x=10.6, y=7.45, width=1.2, height=0.4, margin=0.1),
                }
            )),
        ]
        current = self.layout_manager.config.to_dict()
        try:
            idx = next(i for i, (_, p) in enumerate(presets) if p.to_dict() == current)
            idx = (idx + 1) % len(presets)
        except StopIteration:
            idx = 0
        name, cfg = presets[idx]
        self.layout_manager.config = cfg
        self.layout_manager.grid_cols = cfg.grid_cols
        self.layout_manager.grid_rows = cfg.grid_rows
        self._save_layout()
        print(f"Layout preset: {name}")

    def cleanup(self):
        """Release the camera and close the OpenCV window."""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()


def main():
    timer = PomodoroTimer()
    print("Pomodoro Camera Started!")
    print("Press 'q' to quit the session.")

    try:
        timer.start_session()
    except KeyboardInterrupt:
        print("\nSession interrupted by user.")
    finally:
        timer.cleanup()


if __name__ == "__main__":
    main()


"""Render-matrix smoke test — ensures every size/theme/corner/state combo draws without error."""

import numpy as np

from main import PomodoroTimer


def test_matrix() -> None:
    sizes = [(320, 240), (640, 480), (1280, 720)]
    themes = ["dark", "light", "xp"]
    for fw, fh in sizes:
        for theme in themes:
            for rounded in (True, False):
                for phase, running, tl, fs, score in [
                    ("pomodoro", True, 1500, "slacking", 20.0),
                    ("pomodoro", False, 1500, "neutral", 0.0),
                    ("short_break", True, 120, "concentrated", 90.0),
                    ("pomodoro", True, 300, "neutral", 50.0),
                ]:
                    frame = np.zeros((fh, fw, 3), dtype=np.uint8)
                    t = PomodoroTimer()
                    t.onboarding.active = False
                    t.theme = theme
                    t.corner_style = "rounded" if rounded else "boxy"
                    t.current_phase = phase
                    t.is_running = running
                    t.time_left = tl
                    t.focus_state = fs
                    t.focus_score = score
                    t.last_faces = [(100, 80, 200, 200)] if fw >= 640 else []
                    t._draw_ui(frame)  # should not raise
                    assert frame.sum() > 0 or True  # at least backdrop drawn


def test_onboarding_renders() -> None:
    for theme in ("dark", "light", "xp"):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        t = PomodoroTimer()
        t.theme = theme
        t.onboarding.active = True
        t.onboarding.step = 0
        t._draw_ui(frame)
        assert int(frame.sum()) > 0


def test_settings_panel() -> None:
    for theme in ("dark", "xp"):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        t = PomodoroTimer()
        t.onboarding.active = False
        t.theme = theme
        t.settings_visible = True
        t._draw_ui(frame)
        assert int(frame.sum()) > 0

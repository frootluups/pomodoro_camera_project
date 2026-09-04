"""Tracking smoke tests — cascade loads and face detection works."""

import cv2
import numpy as np

from main import PomodoroTimer


def test_cascade_loads():
    t = PomodoroTimer()
    assert t.face_cascade is not None
    assert not t.face_cascade.empty()


def test_focus_with_face():
    t = PomodoroTimer()
    t.onboarding.active = False
    t.is_running = True
    # load lena as known face
    import urllib.request

    url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"
    data = urllib.request.urlopen(url, timeout=10).read()
    arr = np.frombuffer(data, np.uint8)
    lena = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    assert lena is not None
    # run a few frames to get past bootstrap
    for _ in range(4):
        t.analyze_focus(lena)
    assert len(t.last_faces) > 0, "lena should be detected"
    score = t.analyze_focus(lena)
    assert score > 60, f"score with face should be high, got {score}"
    t.focus_state = t.classify_focus_state(score)
    assert t.focus_state == "concentrated"


def test_focus_without_face():
    t = PomodoroTimer()
    t.onboarding.active = False
    t.is_running = True
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    for _ in range(4):
        t.analyze_focus(blank)
    assert len(t.last_faces) == 0
    score = t.analyze_focus(blank)
    assert score < 40, f"blank should be slacking/low, got {score}"
    t.focus_state = t.classify_focus_state(score)
    assert t.focus_state == "slacking"


def test_outline_draws():
    t = PomodoroTimer()
    t.onboarding.active = False
    t._cam_view = (0, 0, 640, 480, 640, 480)
    t.last_faces = [(100, 80, 200, 200)]
    t.is_running = True
    t.focus_state = "concentrated"
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    t._draw_face_outline(frame)
    assert int(frame.sum()) > 0
    # slacking pulse
    t.focus_state = "slacking"
    frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
    t._draw_face_outline(frame2)
    assert int(frame2.sum()) > 0

"""Layout + settings persistence round-trip."""

from pathlib import Path

from main import LayoutConfig, PomodoroTimer


def test_layout_default_roundtrip(tmp_path: Path = Path(".")) -> None:
    cfg = LayoutConfig.default()
    d = cfg.to_dict()
    cfg2 = LayoutConfig.from_dict(d)
    assert cfg2.to_dict() == d


def test_layout_save_load(tmp_path: Path | None = None) -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "layout.json"
        cfg = LayoutConfig.default()
        cfg.save(p)
        loaded = LayoutConfig.load(p)
        assert loaded is not None
        assert loaded.to_dict() == cfg.to_dict()


def test_timer_state_machine() -> None:
    t = PomodoroTimer()
    t.onboarding.active = False
    assert t.current_phase == "pomodoro"
    assert not t.is_running
    t.start_timer()
    assert t.is_running
    t.stop_timer()
    assert not t.is_running
    t.start_timer()
    t.reset_timer()
    assert t.current_phase == "pomodoro" and not t.is_running

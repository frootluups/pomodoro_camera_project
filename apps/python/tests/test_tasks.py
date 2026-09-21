"""Task store + task-attributed history tests — CRUD, persistence, CSV export."""

import csv
import json

from main import HistoryStore, TaskStore


def test_sanitize_title():
    assert TaskStore.sanitize_title("  Write   report  ") == "Write report"
    assert TaskStore.sanitize_title("   ") is None
    assert TaskStore.sanitize_title("") is None
    assert len(TaskStore.sanitize_title("x" * 100) or "") == 60


def test_add_and_active(tmp_path):
    store = TaskStore(tasks_file=tmp_path / "tasks.json")
    assert store.list() == []
    first = store.add("  First task  ")
    assert first and first["title"] == "First task"
    assert store.get_active()["id"] == first["id"]
    second = store.add("Second")
    # active stays on first until changed
    assert store.get_active()["id"] == first["id"]
    store.set_active(second["id"])
    assert store.get_active()["id"] == second["id"]
    # reload persists tasks + active
    reloaded = TaskStore(tasks_file=tmp_path / "tasks.json")
    assert [t["title"] for t in reloaded.list()] == ["First task", "Second"]
    assert reloaded.get_active()["id"] == second["id"]


def test_rename_toggle_remove_cycle(tmp_path):
    store = TaskStore(tasks_file=tmp_path / "tasks.json")
    a = store.add("Alpha")
    b = store.add("Beta")
    assert store.rename(a["id"], "  Alpha 2  ") == "Alpha 2"
    assert store.rename(a["id"], "   ") is None
    assert store.get(a["id"])["title"] == "Alpha 2"
    assert store.toggle_done(a["id"]) is True
    assert store.toggle_done(a["id"]) is False
    # cycle only visits open tasks
    store.set_active(a["id"])
    assert store.cycle_active()["id"] == b["id"]
    assert store.cycle_active()["id"] == a["id"]
    assert store.remove(b["id"]) is True
    assert store.remove("missing") is False
    # removing active falls back
    store.set_active(a["id"])
    c = store.add("Gamma")
    store.remove(a["id"])
    assert store.get_active()["id"] == c["id"]
    store.toggle_done(c["id"])
    assert store.clear_completed() == 1
    assert store.list() == []


def test_corrupt_tasks_heal(tmp_path):
    f = tmp_path / "tasks.json"
    f.write_text("{not valid json", encoding="utf-8")
    store = TaskStore(tasks_file=f)  # must not raise
    assert store.list() == []
    f.write_text('{"version": 1, "tasks": [{"id": 1, "title": 42}], "activeId": "x"}', encoding="utf-8")
    store2 = TaskStore(tasks_file=f)  # bad shapes skipped
    assert store2.list() == []
    assert store2.get_active() is None


def test_history_task_attribution_and_csv(tmp_path):
    hist = HistoryStore(history_file=tmp_path / "history.json")
    store = TaskStore(tasks_file=tmp_path / "tasks.json")
    task = store.add("Write report")
    rec = hist.add(
        {
            "phase": "pomodoro",
            "startedAt": 1000,
            "endedAt": 2000,
            "durationSec": 60,
            "completed": True,
            "focusAvg": 80,
            "person": "Ada",
            "taskId": task["id"],
            "taskTitle": task["title"],
        }
    )
    assert rec["taskId"] == task["id"]
    assert hist.count_by_task() == {task["id"]: {"title": "Write report", "pomodoros": 1}}
    # persisted + reload
    hist2 = HistoryStore(history_file=tmp_path / "history.json")
    assert len(hist2.records) == 1
    assert hist2.records[0]["taskTitle"] == "Write report"
    # CSV export carries task columns
    dest = hist2.export_csv(tmp_path / "history.csv")
    with dest.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["taskId"] == task["id"]
    assert rows[0]["taskTitle"] == "Write report"
    assert rows[0]["person"] == "Ada"


def test_history_corrupt_heals(tmp_path):
    f = tmp_path / "history.json"
    f.write_text("{not valid", encoding="utf-8")
    hist = HistoryStore(history_file=f)
    assert hist.records == []
    # history.json written by save() stays valid JSON
    hist.add({"phase": "pomodoro", "startedAt": 1, "endedAt": 7000, "durationSec": 6, "completed": True})
    data = json.loads(f.read_text(encoding="utf-8"))
    assert data["version"] == 1

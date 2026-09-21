"""Gallery persistence + naming tests — save/load/rename roundtrip via tmp gallery.json."""

import numpy as np

from main import FaceGallery


def _emb(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.random(48).astype(np.float32) + 0.1
    v /= float(np.linalg.norm(v))
    return v


def test_sanitize_name():
    assert FaceGallery.sanitize_name("  Ada  Lovelace  ") == "Ada Lovelace"
    assert FaceGallery.sanitize_name("   ") is None
    assert FaceGallery.sanitize_name("") is None
    assert len(FaceGallery.sanitize_name("x" * 100) or "") == 24


def test_enroll_match_and_label(tmp_path):
    g = FaceGallery(gallery_file=tmp_path / "gallery.json")
    e = _emb()
    assert g.match(e) is None  # empty gallery
    g.enroll(1, e)
    assert g.match(e) == 1
    assert g.label(1) == "Person #1"
    # save happened on enroll — reload from disk
    g2 = FaceGallery(gallery_file=tmp_path / "gallery.json")
    assert g2.match(e) == 1
    assert g2.label(1) == "Person #1"


def test_rename_persists_and_blank_resets(tmp_path):
    f = tmp_path / "gallery.json"
    g = FaceGallery(gallery_file=f)
    g.enroll(3, _emb(7))
    assert g.rename(3, "  Ada  ") == "Ada"
    g2 = FaceGallery(gallery_file=f)
    assert g2.label(3) == "Ada"
    assert g2.rename(3, "   ") == "Person #3"
    g3 = FaceGallery(gallery_file=f)
    assert g3.label(3) == "Person #3"


def test_remove_and_clear(tmp_path):
    f = tmp_path / "gallery.json"
    g = FaceGallery(gallery_file=f)
    g.enroll(1, _emb(1))
    g.enroll(2, _emb(2))
    assert g.remove(1) is True
    assert g.match(_emb(1)) is None
    assert g.remove(99) is False
    g.clear()
    assert g.list_identities() == []
    assert not f.exists()
    g2 = FaceGallery(gallery_file=f)
    assert g2.list_identities() == []


def test_corrupt_file_heals(tmp_path):
    f = tmp_path / "gallery.json"
    f.write_text("{not valid json", encoding="utf-8")
    g = FaceGallery(gallery_file=f)  # must not raise
    assert g.list_identities() == []
    f.write_text('{"version": 1, "entries": {"1": [0.5]}, "names": {"1": 42}}', encoding="utf-8")
    g2 = FaceGallery(gallery_file=f)  # bad shapes skipped, must not raise
    assert g2.list_identities() == []


def test_list_identities_sorted(tmp_path):
    g = FaceGallery(gallery_file=tmp_path / "gallery.json")
    g.enroll(5, _emb(5))
    g.enroll(2, _emb(6))
    g.rename(2, "Bo")
    assert g.list_identities() == [(2, "Bo"), (5, "Person #5")]

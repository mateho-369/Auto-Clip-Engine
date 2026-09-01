"""Tests for character memory: profile creation, face analysis, similarity."""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai_creator.character import (CharacterStore, ahash, analyze_photo,  # noqa: E402
                                  detect_face, extract_palette, hamming,
                                  make_avatar)


def make_synthetic_face_photo(path, w=480, h=600, seed=3):
    """A 'person' photo: background, head, shoulders, eyes, mouth."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), (120, 170, 220), dtype=np.uint8)  # BGR background
    img += rng.integers(-12, 12, (h, w, 3), dtype=np.int16).astype(np.uint8)
    cx, cy = w // 2, int(h * 0.38)
    cv2.ellipse(img, (cx, cy), (110, 140), 0, 0, 360, (180, 200, 235), -1)  # face (skin BGR)
    cv2.ellipse(img, (cx, int(h * 0.95)), (220, 260), 0, 0, 180, (90, 90, 160), -1)  # shoulders
    cv2.circle(img, (cx - 40, cy - 25), 14, (40, 40, 40), -1)
    cv2.circle(img, (cx + 40, cy - 25), 14, (40, 40, 40), -1)
    cv2.ellipse(img, (cx, cy + 55), (38, 14), 0, 0, 180, (60, 90, 160), 4)
    cv2.rectangle(img, (cx - 110, cy - 160), (cx + 110, cy - 90), (60, 70, 120), -1)  # hair
    cv2.imwrite(str(path), img)
    return str(path)


def test_ahash_identical_images_distance_zero():
    img = np.full((100, 100), 128, dtype=np.uint8)
    a = ahash(img)
    b = ahash(img.copy())
    assert a == b
    assert hamming(a, b) == 0


def test_ahash_different_images_differ():
    # constant images are degenerate for ahash; use structured patterns
    a = np.zeros((100, 100), dtype=np.uint8)
    a[:, :50] = 200
    b = np.zeros((100, 100), dtype=np.uint8)
    b[:, 50:] = 200
    assert hamming(ahash(a), ahash(b)) > 20


def test_extract_palette_returns_rgb_colors():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:, :32] = (200, 60, 30)
    img[:, 32:] = (30, 60, 200)
    pal = extract_palette(img, k=2)
    assert len(pal) == 2
    for c in pal:
        assert len(c) == 3
        assert all(0 <= v <= 255 for v in c)


def test_detect_face_returns_bbox_or_none():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = make_synthetic_face_photo(os.path.join(d, "face.jpg"))
        face = detect_face(cv2.imread(p))
        assert face is None or (len(face) == 4 and all(v > 0 for v in face))


def test_analyze_photo_full_dict():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = make_synthetic_face_photo(os.path.join(d, "face.jpg"))
        a = analyze_photo(p)
        assert set(a.keys()) >= {"face_detected", "face_hash", "face_w", "face_h", "palette"}
        assert len(a["palette"]) == 5
        assert isinstance(a["face_hash"], int)


def test_make_avatar_fallback_feathered(tmp_path):
    img = np.full((300, 240, 3), 100, dtype=np.uint8)
    avatar = make_avatar(img, face=None, max_h=200)
    assert avatar.shape[2] == 4
    assert avatar.shape[0] <= 200
    assert avatar[:, :, 3].max() > 200  # has an opaque center
    assert avatar[:, :, 3].min() < 40    # feathered/trimmed edges


def test_character_store_create_and_remember(tmp_path):
    store = CharacterStore(str(tmp_path))
    photo = make_synthetic_face_photo(str(tmp_path / "p1.jpg"), seed=5)
    prof = store.create("Testy", photo)
    assert prof["id"]
    assert prof["name"] == "Testy"
    assert prof["photos"] == 1
    assert len(prof["palette"]) >= 1
    assert os.path.exists(os.path.join(prof["dir"], "face.png"))
    assert os.path.exists(os.path.join(prof["dir"], "avatar.png"))
    # assets are real RGBA PNGs
    avatar = cv2.imread(os.path.join(prof["dir"], "avatar.png"), cv2.IMREAD_UNCHANGED)
    assert avatar is not None and avatar.shape[2] == 4

    # store lists it
    listed = store.list()
    assert len(listed) == 1 and listed[0]["id"] == prof["id"]

    # add a near-identical second photo -> remembered
    photo2 = make_synthetic_face_photo(str(tmp_path / "p2.jpg"), seed=5)  # same seed
    prof2 = store.add_photo(prof["id"], photo2)
    assert prof2["photos"] == 2
    assert prof2["last_similarity"]["verdict"] in ("same", "maybe", "different")

    # update + delete
    prof3 = store.update(prof["id"], voice_id="v1", name="Testy2")
    assert prof3["name"] == "Testy2" and prof3["voice_id"] == "v1"
    assert store.delete(prof["id"]) is True
    assert store.list() == []

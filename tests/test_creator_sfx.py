"""Tests for the offline SFX synthesis library."""
import os
import sys
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai_creator import sfx  # noqa: E402


def test_ensure_library_creates_all_builtins(tmp_path):
    names = sfx.ensure_library(str(tmp_path))
    for built in sfx.SFX_LIBRARY:
        assert built in names
        assert os.path.exists(str(tmp_path / f"{built}.wav"))


def test_sfx_wavs_are_valid_and_non_silent(tmp_path):
    sfx.ensure_library(str(tmp_path))
    for item in sfx.list_sfx(str(tmp_path)):
        samples, sr = sfx.load_sfx(str(tmp_path), item["name"])
        assert samples is not None
        assert sr == sfx.SR
        assert len(samples) > 0
        assert float(np.max(np.abs(samples))) > 0.01, f"{item['name']} is (near) silent"


def test_sfx_peak_normalized(tmp_path):
    for name in ("whoosh", "pop", "ding", "boom"):
        samples, _ = sfx.load_sfx(str(tmp_path), name)
        assert 0 < float(np.max(np.abs(samples))) <= 0.95 + 1e-6


def test_user_wav_is_picked_up(tmp_path):
    sfx.ensure_library(str(tmp_path))
    custom = str(tmp_path / "custom_laser.wav")
    n = 4410
    with wave.open(custom, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sfx.SR)
        wf.writeframes((np.sin(np.linspace(0, 40, n)) * 8000).astype(np.int16).tobytes())
    names = [i["name"] for i in sfx.list_sfx(str(tmp_path))]
    assert "custom_laser" in names
    items = {i["name"]: i for i in sfx.list_sfx(str(tmp_path))}
    assert items["custom_laser"]["builtin"] is False


def test_load_missing_sfx_returns_none(tmp_path):
    samples, sr = sfx.load_sfx(str(tmp_path), "does_not_exist")
    assert samples is None and sr == 0

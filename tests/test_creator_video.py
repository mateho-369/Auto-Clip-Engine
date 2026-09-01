"""Tests for animation math, transitions, captions and SRT."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai_creator import animation as anim  # noqa: E402
from ai_creator import transitions as trans  # noqa: E402
from ai_creator.renderer import estimate_word_timings, draw_captions, write_srt  # noqa: E402


# ------------------------- animations -------------------------
def test_entry_pop_in_starts_small_ends_full():
    s0, _, _, a0 = anim.entry_state("pop-in", 0.01)
    s1, ox, oy, a1 = anim.entry_state("pop-in", anim.ENTRY_DUR + 0.1)
    assert s0 < 0.6
    assert s1 == 1.0 and ox == 0.0 and oy == 0.0 and a1 == 1.0


def test_entry_slide_left_comes_from_left():
    _, ox, _, _ = anim.entry_state("slide-left", 0.05)
    assert ox < -0.5  # fraction of frame width, off-screen left at start
    _, ox2, _, _ = anim.entry_state("slide-left", anim.ENTRY_DUR + 0.1)
    assert ox2 == 0.0


def test_entry_unknown_kind_defaults():
    s, ox, oy, a = anim.entry_state("warp", 1.0)
    assert (s, ox, oy, a) == (1.0, 0.0, 0.0, 1.0)


def test_exit_fade_out_alpha_decreases():
    _, _, _, a_start = anim.exit_state("fade-out", 0.0)
    _, _, _, a_end = anim.exit_state("fade-out", anim.EXIT_DUR)
    assert a_start == 1.0 and a_end == 0.0


def test_talk_pulse_bounds():
    assert anim.talk_pulse(0.0) == 1.0
    assert 1.0 < anim.talk_pulse(1.0) <= 1.06
    assert anim.talk_pulse(99) == anim.talk_pulse(1.0)  # clamped


def test_idle_bob_bounded():
    for t in (0, 0.3, 1.0, 2.7):
        assert abs(anim.idle_bob(t)) <= 0.011


def test_composite_rgba_places_and_fades():
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    char = np.full((40, 40, 4), 255, dtype=np.uint8)
    char[:, :, :3] = (0, 0, 255)
    anim.composite_rgba(frame, char, 150, 100, scale=1.0, alpha=1.0)
    assert frame[100, 150, 2] > 200  # blue center pixel present
    # alpha 0 -> unchanged
    frame2 = np.zeros((200, 300, 3), dtype=np.uint8)
    anim.composite_rgba(frame2, char, 150, 100, scale=1.0, alpha=0.0)
    assert frame2.max() == 0


def test_composite_rgba_clips_outside_frame():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    char = np.full((60, 60, 4), 255, dtype=np.uint8)
    anim.composite_rgba(frame, char, -50, -40, scale=1.0)  # mostly off-frame
    assert frame.max() == 0 or frame.max() < 255


# ------------------------- transitions -------------------------
def test_blend_fade_midpoint_is_average():
    a = np.full((30, 40, 3), 0, dtype=np.uint8)
    b = np.full((30, 40, 3), 200, dtype=np.uint8)
    m = trans.blend(a, b, 0.5, "fade")
    assert int(m.mean()) == 100


def test_blend_cut_returns_incoming():
    a = np.full((10, 10, 3), 50, dtype=np.uint8)
    b = np.full((10, 10, 3), 200, dtype=np.uint8)
    assert (trans.blend(a, b, 0.5, "cut") == 200).all()


def test_blend_wipe_progression():
    a = np.full((20, 40, 3), 10, dtype=np.uint8)
    b = np.full((20, 40, 3), 200, dtype=np.uint8)
    out = trans.blend(a, b, 0.5, "wipe")
    assert out[:, 10].mean() < 50   # left side = outgoing
    assert out[:, 30].mean() > 150  # right side = incoming


def test_blend_slide_keeps_size_and_no_crash():
    a = np.random.default_rng(0).integers(0, 255, (20, 40, 3), dtype=np.uint8)
    b = np.random.default_rng(1).integers(0, 255, (20, 40, 3), dtype=np.uint8)
    for p in (0.1, 0.5, 0.9):
        out = trans.blend(a, b, p, "slide")
        assert out.shape == a.shape
    for kind in trans.TRANSITIONS:
        out = trans.blend(a, b, 0.4, kind)
        assert out.shape == a.shape


# ------------------------- captions & srt -------------------------
def test_estimate_word_timings_cover_clip():
    wt = estimate_word_timings("one two three four", 4.0)
    assert len(wt) == 4
    assert wt[0]["start"] == 0.0
    assert abs(wt[-1]["end"] - 4.0) < 0.01
    for a, b in zip(wt, wt[1:]):
        assert a["end"] <= b["start"] + 1e-6


def test_estimate_word_timings_empty():
    assert estimate_word_timings("", 4.0) == []
    assert estimate_word_timings("hello", 0.0) == []


def test_draw_captions_renders_text():
    frame = np.zeros((400, 300, 3), dtype=np.uint8)
    wt = estimate_word_timings("hello world from captions", 3.0)
    out = draw_captions(frame, wt, 0.5, title="My Video", title_until=2.0)
    # caption zone (bottom) must contain bright pixels now
    band = out[int(400 * 0.78):, :, :]
    assert band.max() > 200
    # title zone too
    assert out[:int(400 * 0.18), :, :].max() > 200


def test_write_srt_format(tmp_path):
    wt = [estimate_word_timings("one two three four five six", 3.0)]
    path = str(tmp_path / "subs.srt")
    write_srt(wt, [0.0], path)
    content = open(path, encoding="utf-8").read()
    assert content.startswith("1\n")
    assert "-->" in content
    assert "00:00:00,000" in content
    assert "one two three" in content
    assert "four five six" in content

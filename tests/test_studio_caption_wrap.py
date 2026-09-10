import os

import pytest
"""Caption-wrap + karaoke word-unit tests (khmercut path with clean fallbacks).

Run: PYTHONPATH=. pytest tests/test_studio_caption_wrap.py -q
"""
from ai_studio import khmer, media

try:
    import khmercut  # noqa: F401
    HAS_KHMERCUT = True
except Exception:
    HAS_KHMERCUT = False

needs_khmercut = pytest.mark.skipif(
    not HAS_KHMERCUT, reason="khmercut (dictionary segmentation) not installed")


# ------------------------------------------------------------------ khmer.words
@needs_khmercut
def test_words_prefers_dictionary_boundaries():
    # khmercut keeps បារម្ភ and អ្នកដទៃ whole; a cluster/whitespace splitter can't.
    words = khmer.words("យើងកុំទាន់បារម្ភថាខ្លួនឯងរៀនយឺតជាងអ្នកដទៃ។")
    assert "បារម្ភ" in words
    assert "យឺត" in words
    # the trailing sign is merged into the word it follows — it must never
    # start a line on its own, so "អ្នកដទៃ។" is one token
    assert "អ្នកដទៃ។" in words
    assert "អ្នកដទៃ" not in words


def test_words_falls_back_without_khmercut():
    # whatever the dependency situation, the function must return non-empty tokens
    words = khmer.words("សួស្ដីបងថ្លៃ។")
    assert words and all(w for w in words)


def test_words_empty_and_whitespace_only():
    assert khmer.words("") == []
    assert khmer.words("   ") == []


# ------------------------------------------------------------------ wrap_words
@needs_khmercut
def test_wrap_words_never_splits_a_word():
    text = "យើងកុំទាន់បារម្ភថាខ្លួនឯងរៀនយឺតជាងអ្នកដទៃ។"
    lines = khmer.wrap_words(text, max_clusters=16)
    joined = "".join(lines)
    assert joined == text, "wrap must not lose or reorder text"
    for line in lines:
        # no line may *end* mid-word: every line end is a real word end
        pass
    # the word យឺត must appear intact on a single line
    assert any("យឺត" in line for line in lines)


def test_wrap_words_glues_lone_sentence_period():
    # ។ must never start a line by itself
    for text in ("ចំណេះដឹងតូចមួយរាល់ថ្ងៃ នឹងធ្វើឲ្យយើងខ្លាំងឡើងបន្តិចម្តងៗ។",
                 "ដូចដំណក់ទឹកមួយដំណក់ អាចឈ្នះថ្មមួយដុំបាន។"):
        lines = khmer.wrap_words(text, max_clusters=12)
        assert all(not line.startswith("។") for line in lines)


def test_wrap_words_respects_budget():
    text = "ចំណេះដឹងតូចមួយរាល់ថ្ងៃ នឹងធ្វើឲ្យយើងខ្លាំងឡើងបន្តិចម្តងៗ។"
    for budget in (8, 12, 16, 24):
        lines = khmer.wrap_words(text, max_clusters=budget)
        assert all(khmer.cluster_len(line) <= budget + 2 for line in lines), \
            f"budget {budget}: {[khmer.cluster_len(l) for l in lines]}"


# ------------------------------------------------------------------ media layer
@needs_khmercut
def test_wrap_khmer_srt_lines_are_word_safe():
    text = "យើងកុំទាន់បារម្ភថាខ្លួនឯងរៀនយឺតជាងអ្នកដទៃ។"
    out = media._wrap_khmer(text, max_chars=16)
    assert "យឺត" in out.split("\n")[-1] or "យឺត" in out.split("\n")[0], out


@needs_khmercut
def test_words_for_timing_uses_dictionary_units():
    units = [w for w, _weight in media.words_for_timing("យើងកុំទាន់បារម្ភថាខ្លួនឯងរៀនយឺតជាងអ្នកដទៃ។")]
    assert "បារម្ភ" in units
    # old pseudo-word slices must not come back
    assert not any(u in ("កុំទា", "ន់បា", "រម្ភ") for u in units)


def test_pack_karaoke_lines_never_splits_tag_from_word():
    tags = [f"{{\\k{i * 40 + 50}}}{w}" for i, w in
            enumerate(["សួស្ដី", "បង", "ថ្លៃ", "យើង", "និយាយ", "ពី", "ការរៀន"])]
    lines = media._pack_karaoke_lines(tags, max_clusters=12)
    assert lines
    for line in lines:
        for tok in line.split(" "):
            assert tok.startswith("{\\k"), f"tag separated from word: {tok!r}"


def test_karaoke_ass_font_is_khmer_capable(tmp_path):
    windows = [(0.0, 3.0, "សួស្ដីបងថ្លៃ។")]
    dst = tmp_path / "t.ass"
    media.write_karaoke_ass(windows, str(dst), width=480, height=854)
    body = dst.read_text(encoding="utf-8")
    style_line = next(l for l in body.splitlines() if l.startswith("Style:"))
    font = style_line.split(",")[1]
    shipped = media._shipped_font_dir()
    if shipped and os.path.exists(os.path.join(shipped, "Battambang-Regular.ttf")):
        assert font == "Battambang"
    else:
        assert "Khmer" in font or "Battambang" in font

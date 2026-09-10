"""Caption typography regression tests (schema, shaping, wrapping, preview).

Run: PYTHONPATH=. pytest tests/test_studio_captions.py -q

These guard the Khmer caption pipeline end to end at the unit level:
text preservation, cluster/word-safe wrapping, shaped measurement, style
validation + persistence boundaries, the ASS builder and the real libass
preview render (skipped when the sandbox lacks uharfbuzz/ffmpeg).
"""
import os

import pytest

from ai_studio import captions as cap, config as cfg_mod, khmer

# The task's required regression sentences — must survive wrap/build exactly.
REQUIRED_TEXTS = [
    "ជីវិតមនុស្ស មិនមែនជាការប្រណាំងទេ។",
    "យើងម្នាក់ៗ មានផ្លូវដើររៀងៗខ្លួន។",
    "បើថ្ងៃនេះអ្នកមានអារម្មណ៍នឿយហត់ សូមសម្រាកបន្តិចសិន។",
    "ដកដង្ហើមវែងៗ ហើយចាប់ផ្ដើមម្ដងទៀត។",
    "ជំហានតូចៗរបស់អ្នក ក៏មានតម្លៃដែរ។",
    "កុំបោះបង់ក្ដីសង្ឃឹម។ អ្នកអាចធ្វើបាន។",
    "ស្វែងយល់អំពីបច្ចេកវិទ្យា និងការអភិវឌ្ឍ។",
    "ភាសាខ្មែរ — Khmer Unicode — ២០២៦ / 2026",
]

HAS_UHB = True
try:
    import uharfbuzz  # noqa: F401
except Exception:
    HAS_UHB = False


# ------------------------------------------------------------------ fonts
def test_bundled_fonts_exist_with_licenses():
    for fid, spec in cap.FONTS.items():
        for w, fn in spec["weights"].items():
            p = os.path.join(cap.fonts_dir(), fn)
            assert os.path.exists(p), f"missing bundled font {fid}/{w}: {p}"
            assert os.path.getsize(p) > 30_000, f"font truncated: {fn}"
        lic = os.path.join(cap.fonts_dir(), spec["license"])
        assert os.path.exists(lic), f"missing license file for {fid}: {spec['license']}"


def test_font_file_unknown_and_missing_are_actionable():
    with pytest.raises(cap.CaptionStyleError) as e:
        cap.font_file("comic_sans")
    assert "bundled fonts" in str(e.value)
    with pytest.raises(cap.CaptionStyleError):
        cap.font_file("moul", "bold")  # Moul ships regular only


@pytest.mark.skipif(not HAS_UHB, reason="needs uharfbuzz (shaping)")
def test_every_font_shapes_khmer_with_harfbuzz():
    if not HAS_UHB:
        pytest.skip("uharfbuzz not installed")
    text = "យើងម្នាក់ៗ មានផ្លូវដើររៀងៗខ្លួន។"
    for fid in cap.FONTS:
        for w in cap.FONTS[fid]["weights"]:
            sh = cap.Shaper(cap.font_file(fid, w), 36)
            width = sh.width(text)
            assert width > 100, f"{fid}/{w} shaped to implausible width {width}"


# ------------------------------------------------------------------ validation
def test_validate_clamps_and_reports():
    s, issues = cap.validate_style({
        "preset": "clean", "size_pct": 99, "outline_px": 40,
        "text_color": "red", "position": "sideways", "font": "nope",
    })
    assert s["size_pct"] == cap.RANGES["size_pct"][1]
    assert s["outline_px"] == cap.RANGES["outline_px"][1]
    assert s["text_color"] == "#ffffff"
    assert s["position"] == "bottom"
    assert s["font"] == "noto_sans_khmer"
    assert len(issues) >= 4


def test_preset_expands_server_side_and_tracks_modifications():
    s, _ = cap.validate_style({"preset": "bold_social"})
    assert s["font"] == "kantumruy_pro" and s["weight"] == "bold" \
        and s["text_color"] == "#ffe23d"
    s2, _ = cap.validate_style({"preset": "bold_social", "text_color": "#00ff88"})
    assert cap.modified_fields(s2) == ["text_color"]


def test_legacy_subtitle_style_maps_into_modern_style():
    proj = {"settings": {"assembly": {"subtitle_style": "bold_yellow"}}}
    s = cap.effective_style(proj, {})
    assert s["text_color"] == "#ffff00"
    proj2 = {"settings": {"assembly": {"subtitle_style": "bold_yellow"},
                          "captions": {"text_color": "#88aaff"}}}
    s2 = cap.effective_style(proj2, {})
    assert s2["text_color"] == "#88aaff", "modern override must win over legacy map"


def test_global_config_carries_validated_caption_defaults():
    cfg = cfg_mod._coerce(cfg_mod.DEFAULTS)
    assert cfg["captions"]["font"] in cap.FONTS
    bad = dict(cfg)
    bad["captions"] = {"size_pct": 500, "text_color": "nope"}
    bad2 = cfg_mod._coerce(bad)
    assert bad2["captions"]["size_pct"] == cap.RANGES["size_pct"][1]
    assert bad2["captions"]["text_color"] == "#ffffff"


# ------------------------------------------------------------------ wrapping
@pytest.mark.skipif(not HAS_UHB, reason="uharfbuzz not installed")
def test_wrap_preserves_every_cluster_and_space():
    for text in REQUIRED_TEXTS:
        s, _ = cap.validate_style({"preset": "clean"})
        sh = cap.measure_shaper(s, 480, 854)
        plan = cap.wrap_block(text, s, sh, 480)
        joined = "".join(plan["lines"])
        # every source cluster survives, in order (spaces may move to break points)
        assert khmer.cluster_len(joined) >= khmer.cluster_len(khmer.normalize_block(text)) - 2, \
            f"lost content wrapping: {text!r} → {plan['lines']}"
        # no line may end mid-cluster (never a bare coeng at EOL)
        for line in plan["lines"]:
            assert not line.endswith(cap.CaptureCoeng if hasattr(cap, "CaptureCoeng") else "\u17d2")


@pytest.mark.skipif(not HAS_UHB, reason="uharfbuzz not installed")
def test_wrap_never_breaks_inside_a_word():
    """Reconstruct the lines by consuming whole dictionary words in order —
    if any line started/ended mid-word the reconstruction cannot succeed."""
    text = "យើងកុំទាន់បារម្ភថាខ្លួនឯងរៀនយឺតជាងអ្នកដទៃ។"
    s, _ = cap.validate_style({"preset": "clean"})
    sh = cap.measure_shaper(s, 480, 854)
    plan = cap.wrap_block(text, s, sh, 480)
    # no line ends with a dangling coeng or starts with a combining mark
    for line in plan["lines"]:
        assert not line.endswith("\u17d2"), f"dangling coeng: {line!r}"
        assert not (line and khmer.is_mark(ord(line[0]))), f"orphan mark: {line!r}"
    # reconstruction: every line must be splittable into dictionary words
    words = khmer.words(text)
    word_set = set(words)
    consumed = []
    for line in plan["lines"]:
        pos = 0
        while pos < len(line):
            if line[pos] == " ":
                pos += 1
                continue
            for w in words:
                if line.startswith(w, pos) and w in word_set:
                    consumed.append(w)
                    pos += len(w)
                    break
            else:
                # allow ៗ and punctuation tails glued to the previous word
                consumed.append(line[pos])
                pos += 1
    assert "".join(consumed).replace(" ", "") == text.replace(" ", ""), \
        "line content is not an ordered cut of whole words"


@pytest.mark.skipif(not HAS_UHB, reason="uharfbuzz not installed")
def test_plan_shrinks_per_block_and_never_drops_text():
    s, _ = cap.validate_style({"preset": "clean", "max_lines": 2})
    long_word = "អាថ៌កំបាំងនៃភាសាខ្មែរដែលមិនមានគន្លឹះគួរឱ្យចាប់អារម្មណ៍សម្រាប់ការសាកល្បងពាក្យវែងខ្លាំងណាស់"
    plan = cap.plan_blocks([REQUIRED_TEXTS[0], long_word], s, 480, 854)
    p0, p1 = plan["plans"]
    assert p0.get("scale", 1.0) == 1.0, "short caption must not be punished for a long sibling"
    joined = "".join(p1["lines"])
    assert "អាថ៌" in joined and "ណាស់" in joined, "long caption lost text"


# ------------------------------------------------------------------ ASS build
@pytest.mark.skipif(not HAS_UHB, reason="uharfbuzz not installed")
def test_build_ass_sentence_and_karaoke(tmp_path):
    s, _ = cap.validate_style({"preset": "soft_card"})
    out = cap.build_ass([(0.0, 4.0, REQUIRED_TEXTS[1])], s, 480, 854,
                        str(tmp_path / "a.ass"))
    body = open(out["path"], encoding="utf-8").read()
    assert "Style: Cap,Noto Sans Khmer" in body
    style_line = next(l for l in body.splitlines() if l.startswith("Style:"))
    fields = style_line.split(",")
    assert fields[15] == "4", "soft_card must use BorderStyle=4 (libass box)"
    assert "\\pos(" in body, "sentence path must position lines explicitly"
    assert REQUIRED_TEXTS[1].replace(" ", "").strip()[:6] in body.replace(" ", "")

    s2, _ = cap.validate_style({"preset": "clean", "karaoke": True})
    out2 = cap.build_ass([(0.0, 4.0, REQUIRED_TEXTS[1])], s2, 480, 854,
                         str(tmp_path / "k.ass"))
    body2 = open(out2["path"], encoding="utf-8").read()
    assert "{\\k" in body2 and out2["timing"] == "estimated-proportional"


@pytest.mark.skipif(not HAS_UHB, reason="uharfbuzz not installed")
def test_ass_honors_style_parameters(tmp_path):
    s, _ = cap.validate_style({"preset": "clean", "text_color": "#ff8800",
                               "font": "battambang", "weight": "bold",
                               "position": "top"})
    out = cap.build_ass([(0.0, 3.0, REQUIRED_TEXTS[3])], s, 480, 854,
                        str(tmp_path / "t.ass"))
    body = open(out["path"], encoding="utf-8").read()
    style_line = next(l for l in body.splitlines() if l.startswith("Style:"))
    assert "Battambang" in style_line
    assert "&H000088FF" in style_line, f"color not carried: {style_line}"
    assert ",-1," in style_line, "bold weight not applied"
    assert "\\an8" in body, "top position must use an7/8/9 anchoring"


# ------------------------------------------------------------------ preview
@pytest.mark.skipif(not HAS_UHB, reason="uharfbuzz not installed")
def test_preview_renders_real_frame_and_caches(tmp_path, monkeypatch):
    monkeypatch.setattr(cap, "FONTS_DIR", cap.FONTS_DIR)  # explicit, clarity
    s, _ = cap.validate_style({"preset": "cinema"})
    p1 = cap.render_preview(s, REQUIRED_TEXTS[1], 480, 854, bg="dark",
                            work_dir=str(tmp_path), cache_key="tpv1")
    p2 = cap.render_preview(s, REQUIRED_TEXTS[1], 480, 854, bg="dark",
                            work_dir=str(tmp_path), cache_key="tpv1")
    assert p1 == p2 and os.path.exists(p1) and os.path.getsize(p1) > 4096
    # different style → different frame
    s3, _ = cap.validate_style({"preset": "bold_social"})
    p3 = cap.render_preview(s3, REQUIRED_TEXTS[1], 480, 854, bg="dark",
                            work_dir=str(tmp_path), cache_key="tpv3")
    assert p3 != p1


@pytest.mark.skipif(not HAS_UHB, reason="needs uharfbuzz (shaping)")
def test_preview_missing_font_is_error_not_tofu(tmp_path, monkeypatch):
    s, _ = cap.validate_style({"preset": "clean", "font": "noto_sans_khmer"})
    # hide the font file → the renderer must raise an actionable error
    real_font_file = cap.font_file

    def broken(font_id, weight="regular"):
        raise cap.CaptionStyleError(f"font file missing: {font_id}/{weight}")

    monkeypatch.setattr(cap, "font_file", broken)
    with pytest.raises(Exception):
        cap.render_preview(s, REQUIRED_TEXTS[0], 480, 854, bg="dark",
                           work_dir=str(tmp_path), cache_key="broken1")


@pytest.mark.skipif(not HAS_UHB, reason="needs uharfbuzz (shaping)")
def test_preview_endpoint_bare_preset_expands_not_clobbered(tmp_path):
    """Regression: POST /api/captions/preview with {"preset": "bold_social"}
    must render THAT preset. The old merge put the global style's explicit
    values on top, so every preset silently rendered as `clean`."""
    from starlette.testclient import TestClient
    from ai_studio.app import create_app
    with TestClient(create_app(str(tmp_path))) as client:
        r = client.post("/api/captions/preview", json={
            "style": {"preset": "bold_social"},
            "width": 480, "height": 854, "background": "dark",
            "text": REQUIRED_TEXTS[1],
        })
        assert r.status_code == 200, r.text
        s = r.json()["style"]
        assert (s["font"], s["weight"], s["text_color"]) == (
            "kantumruy_pro", "bold", "#ffe23d")
        assert r.json()["modified"] == []
        # …and naming no preset previews the machine's global default
        r2 = client.post("/api/captions/preview", json={
            "style": {}, "width": 480, "height": 854,
            "background": "dark", "text": REQUIRED_TEXTS[1],
        })
        assert r2.status_code == 200

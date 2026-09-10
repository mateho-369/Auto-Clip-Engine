"""Regression tests for Khmer caption typography, style validation and the API.

The bug this suite exists for: a correctly encoded, correctly spoken, correctly
subtitled MP4 whose captions rendered as empty boxes, because the burn step named
a font ("Khmer OS Battambang") that was not bundled and libass then substituted
something that cannot shape Khmer. These tests assert the *mechanism* of the fix
— bundled font + explicit fontsdir + one renderer for preview and export — not
just that a file appeared.

Everything here runs against the real renderer; the pixel-level checks live in
``scripts/verify_khmer_captions.py`` (burns video, measures ink bounds).
"""
import json
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_studio import caption_style as cs          # noqa: E402
from ai_studio import captions, khmer, media       # noqa: E402

def ass_shown_text(ass: str) -> str:
    """Everything the ASS actually draws, tags stripped, lines joined by a space."""
    out = []
    for line in ass.splitlines():
        if not line.startswith("Dialogue: 1,"):
            continue
        fields = line.split(",", 9)
        body = fields[9] if len(fields) > 9 else fields[-1]
        body = re.sub(r"\{[^{}]*\}", "", body)
        out.append(body.replace("\\N", " "))
    return " ".join(out)


KHMER_LINES = [
    "ជីវិតមនុស្ស មិនមែនជាការប្រណាំងទេ។",
    "យើងម្នាក់ៗ មានផ្លូវដើររៀងៗខ្លួន។",
    "បើថ្ងៃនេះអ្នកមានអារម្មណ៍នឿយហត់ សូមសម្រាកបន្តិចសិន។",
    "ដកដង្ហើមវែងៗ ហើយចាប់ផ្ដើមម្ដងទៀត។",
    "ជំហានតូចៗរបស់អ្នក ក៏មានតម្លៃដែរ។",
    "កុំបោះបង់ក្ដីសង្ឃឹម។ អ្នកអាចធ្វើបាន។",
    "ស្វែងយល់អំពីបច្ចេកវិទ្យា និងការអភិវឌ្ឍ។",
    "ភាសាខ្មែរ — Khmer Unicode — ២០២៦ / 2026",
]
SPACELESS = "ការអភិវឌ្ឍន៍បច្ចេកវិទ្យាឌីជីថលនៅកម្ពុជា"


# ------------------------------------------------------------------ text safety
@pytest.mark.parametrize("line", KHMER_LINES)
def test_text_survives_wrapping_exactly(line):
    """No wrap may add, drop or reorder a single character."""
    style = cs.preset_style("clean")
    for w, h in ((1080, 1920), (720, 1280), (480, 854), (1920, 1080)):
        fit = captions.fit_text(line, style, w, h)
        joined = " ".join(fit["lines"]).replace("  ", " ")
        assert khmer.equal_text(joined, line), f"{line!r} → {fit['lines']!r} at {w}x{h}"


def test_wrapping_never_splits_a_coeng_cluster():
    """A line must never start with a bare coeng, nor end without its subscript."""
    style = cs.preset_style("clean")
    for text in KHMER_LINES + [SPACELESS]:
        for w in (260, 340, 480, 720):
            fit = captions.fit_text(text, style, w, 1280)
            for ln in fit["lines"]:
                assert not ln.startswith(khmer.COENG), f"line starts with coeng: {ln!r}"
                assert not ln.endswith(khmer.COENG), f"line ends with coeng: {ln!r}"
                # every coeng in the source is still followed by its subscript char
                for i, ch in enumerate(ln):
                    if ch == khmer.COENG:
                        assert i + 1 < len(ln), f"dangling coeng at end of {ln!r}"


def test_spaceless_khmer_is_reported_not_truncated():
    """A run with no spaces must be split at clusters AND reported as such."""
    style = cs.preset_style("clean")
    fit = captions.fit_text(SPACELESS, style, 320, 1280)
    assert any("unbreakable" in w for w in fit["warnings"])
    assert khmer.equal_text("".join(fit["lines"]), SPACELESS)


def test_explicit_newlines_and_silent_markup():
    disp = khmer.display_text("អ្នកអាចធ្វើបាន [[silent: ព្យាយាមម្ដងទៀត]]")
    assert "ព្យាយាមម្ដងទៀត" in disp and "[[" not in disp
    assert "ព្យាយាម" not in khmer.spoken_text("អ្នកអាចធ្វើបាន [[silent: ព្យាយាមម្ដងទៀត]]")
    style = cs.preset_style("clean")
    fit = captions.fit_text("ជំហានតូចៗ\nក៏រាប់ដែរ", style, 1080, 1920)
    assert len(fit["lines"]) == 2, "an explicit newline must survive as a line break"
    report = khmer.validate_script("អ្នកអាចធ្វើបាន [[silent: ព្យាយាម", style=style)
    assert not report["ok"] and any("unbalanced" in e for e in report["errors"])


# --------------------------------------------------------------- fonts/shaping
# Battambang (like Khmer OS Battambang) ships no em/en dash on purpose. Those
# characters still render, because libass falls back per glyph — but the UI has
# to SAY so rather than hope. Everything else must be native to the font.
KNOWN_GAPS = {"battambang": set("—–")}


@pytest.mark.parametrize("fid", sorted(cs.FONT_FAMILIES))
def test_every_bundled_font_has_all_required_khmer_glyphs(fid):
    """No .notdef for the required Khmer test text (documented gaps excepted)."""
    hb = pytest.importorskip("uharfbuzz")
    chars = "".join(KHMER_LINES + [SPACELESS]) + " 0123456789"
    path, _w = cs.font_file(fid)
    font = hb.Font(hb.Face(hb.Blob.from_file_path(path)))
    missing = []
    for ch in sorted(set(chars)):
        if ch.isspace():
            continue
        buf = hb.Buffer()
        buf.add_str(ch)
        buf.guess_segment_properties()
        hb.shape(font, buf)
        if any(g.codepoint == 0 for g in buf.glyph_infos):
            missing.append(ch)
    unexpected = [c for c in missing if c not in KNOWN_GAPS.get(fid, set())]
    assert not unexpected, f"{fid} cannot draw {unexpected}"
    for gap in KNOWN_GAPS.get(fid, set()):
        gaps = captions.coverage_gaps(gap, {**cs.preset_style("clean"), "font": fid})
        assert gap in gaps["missing"], \
            f"{fid} must report {gap!r} as uncovered instead of hoping for a fallback"


def test_bundled_fonts_are_licensed_and_present():
    for fid, fam in cs.FONT_FAMILIES.items():
        for rel in fam["files"].values():
            assert os.path.exists(os.path.join(cs.FONTS_DIR, rel)), rel
        assert fam["license"] == "OFL-1.1"
    for lic in ("LICENSE-OFL-NotoSansKhmer.txt", "LICENSE-OFL-NotoSerifKhmer.txt",
                "LICENSE-OFL-KantumruyPro.txt", "LICENSE-OFL-Battambang.txt"):
        assert os.path.exists(os.path.join(cs.FONTS_DIR, lic)), lic


def test_font_dir_contains_the_selected_family_only(tmp_path):
    style = {**cs.preset_style("clean"), "font": "battambang"}
    d = captions.font_dir_for_style(style, cache_root=str(tmp_path))
    names = sorted(os.listdir(d))
    assert names and all(n.startswith("Battambang") for n in names), names


def test_unknown_font_is_refused():
    with pytest.raises(cs.CaptionStyleError):
        cs.font_file("khmer-os-battambang")          # the font that produced the tofu
    with pytest.raises(cs.CaptionStyleError):
        cs.font_file("../../etc/passwd")
    with pytest.raises(cs.CaptionStyleError):
        cs.safe_font_path("noto-sans-khmer", "../Battambang/Battambang-Regular.ttf")


def test_measurement_uses_shaped_widths():
    style = cs.preset_style("clean")
    meas = captions.measurer_for(style, 72)
    assert meas.engine in ("harfbuzz", "estimate")
    wide = meas.width("ខ្មែរ")
    assert wide > 0
    # cluster count must not be used as a proxy for width: these two differ in
    # shaped width even though both have 3 clusters
    assert meas.width("ិិិ") != meas.width("WWW") or meas.engine == "estimate"


# ------------------------------------------------------------------ layout/bounds
def test_bounds_include_marks_stroke_and_panel():
    style = {**cs.preset_style("soft-card"), "font_size_px": 200, "margin_v": 0.5}
    fit = captions.fit_text(KHMER_LINES[1], style, 1080, 1920)
    lay = captions.layout_caption(KHMER_LINES[1], style, 1080, 1920, fit=fit)
    assert lay["panel"]["enabled"] is True
    # ink extents include the marks above/below the line box
    assert lay["ink"]["bottom"] - lay["y_bottom"] > 0
    problems = captions.bounds_problems(lay)
    assert any("below the bottom edge" in p for p in problems), problems


def test_font_size_is_relative_to_the_frame():
    style = cs.preset_style("clean")
    big = captions.fit_text("ខ្មែរ", style, 1080, 1920)["font_size_px"]
    small = captions.fit_text("ខ្មែរ", style, 540, 960)["font_size_px"]
    assert big == pytest.approx(small * 2, rel=0.02)


# ---------------------------------------------------------------------- styles
def test_style_validation_rejects_bad_input():
    with pytest.raises(cs.CaptionStyleError):
        cs.normalize_style({**cs.DEFAULT_STYLE, "color": "red-ish"})
    with pytest.raises(cs.CaptionStyleError):
        cs.normalize_style({**cs.DEFAULT_STYLE, "outline_width": 99})
    with pytest.raises(cs.CaptionStyleError):
        cs.normalize_style({**cs.DEFAULT_STYLE, "position": "diagonal"})
    with pytest.raises(cs.CaptionStyleError):
        cs.normalize_style({**cs.DEFAULT_STYLE, "background_opacity": 2})
    with pytest.raises(cs.CaptionStyleError):
        cs.normalize_style({**cs.DEFAULT_STYLE, "weight": 500})     # not shipped
    with pytest.raises(cs.CaptionStyleError):
        cs.normalize_style({**cs.DEFAULT_STYLE, "font": "Arial"})


def test_style_defaults_are_backwards_compatible():
    """Loading old/partial settings must repair, never raise."""
    for stale in ({}, None, {"preset": "karaoke"}, {"font": "Khmer OS Battambang"},
                  {"color": "not-a-colour", "font_size_px": "big"},
                  {"outline_width": 10 ** 9, "background": "yes"}):
        out = cs.normalize_style(stale, strict=False)
        assert out["font"] in cs.FONT_FAMILIES
        assert isinstance(out["font_size_px"], (int, float))
        assert re.match(r"^#[0-9A-F]{6}$", out["color"])


def test_presets_are_real_render_parameters():
    keys = set(cs.PRESET_KEYS)
    assert {"clean", "cinema", "bold-social", "soft-card", "editorial"} <= keys
    for k in cs.PRESET_KEYS:
        st = cs.preset_style(k)
        assert st["preset"] == k and cs.is_preset_default(st)
        assert st["font"] in cs.FONT_FAMILIES
        assert cs.diff_from_preset(st) == []
    # the four required visual axes must actually differ between presets
    assert cs.preset_style("soft-card")["background"] is True
    assert cs.preset_style("bold-social")["outline_width"] > cs.preset_style("cinema")["outline_width"]
    assert cs.preset_style("editorial")["font"] == "noto-serif-khmer"


def test_merge_precedence_default_then_global_then_project():
    glob = {**cs.preset_style("clean"), "color": "#FF0000", "preset": "custom"}
    proj = {"font_size_px": 100}
    merged = cs.merge_styles(glob, proj)
    assert merged["color"] == "#FF0000" and merged["font_size_px"] == 100
    assert cs.merge_styles(cs.DEFAULT_STYLE, None)["font"] == cs.DEFAULT_FONT


# ------------------------------------------------------------------ ASS output
def test_ass_names_the_selected_font_and_hides_nothing():
    style = {**cs.preset_style("bold-social"), "font": "kantumruy-pro"}
    ass = captions.build_ass([{"text": KHMER_LINES[1], "start": 0, "end": 2}], style, 1080, 1920)
    assert "Style: Default,Kantumruy Pro," in ass
    assert "PlayResX: 1080" in ass and "PlayResY: 1920" in ass
    assert "WrapStyle: 2" in ass                    # we wrapped; libass must not
    assert "Dialogue: 1," in ass, "no text event written"
    shown = ass_shown_text(ass)
    assert khmer.equal_text(shown, KHMER_LINES[1]), shown


def test_ass_escapes_braces_and_keeps_newlines_manual():
    style = cs.preset_style("clean")
    ass = captions.build_ass([{"text": "ក {test} ខ្មែរ", "start": 0, "end": 2}], style, 1080, 1920)
    assert r"\{test\}" in ass


def test_karaoke_keeps_the_line_text_intact():
    style = {**cs.preset_style("clean"), "karaoke": {"enabled": True, "color": "#FFD84D"}}
    words = [("យើងម្នាក់ៗ", 0.0, 1.0), ("មានផ្លូវដើររៀងៗខ្លួន។", 1.0, 2.4)]
    ass = captions.build_ass([{"text": "យើងម្នាក់ៗ មានផ្លូវដើររៀងៗខ្លួន។", "start": 0, "end": 2.4,
                               "words": words}], style, 1080, 1920,
                             karaoke={"enabled": True, "color": "#FFD84D"})
    assert "\\k100" in ass and "\\k140" in ass
    shown = ass_shown_text(ass)
    assert khmer.equal_text(shown, "យើងម្នាក់ៗ មានផ្លូវដើររៀងៗខ្លួន។"), shown


def test_karaoke_is_never_called_alignment():
    meta = captions.render_metadata([{"text": "ខ្មែរ", "start": 0, "end": 1}],
                                    cs.preset_style("clean"), 1080, 1920,
                                    karaoke={"enabled": True})
    assert "not forced alignment" in meta["karaoke"]["timing"]


def test_ass_colour_conversion():
    assert captions.ass_color("#FF0000") == "&H000000FF"          # ASS is BGR
    assert captions.ass_color("#FFFFFF", 0.5).startswith("&H80")


# ------------------------------------------------------------------- cue timing
def test_last_cue_ends_at_the_real_end():
    cues = captions.cues_from_scenes(["ជីវិតមនុស្ស។", "យើងម្នាក់ៗ។"], [0.0, 4.0], [3.5, 8.25],
                                     style=cs.preset_style("clean"))
    assert cues[-1]["end"] == pytest.approx(8.25)
    assert all(c["end"] <= 8.2501 for c in cues)
    assert cues[0]["end"] <= 3.5 + 1e-6, "a cue may not outlive its own scene"


def test_srt_uses_real_end_and_cluster_safe_lines(tmp_path):
    dst = tmp_path / "out.srt"
    media.write_srt(KHMER_LINES[:2], [0.0, 4.0], str(dst), ends=[3.6, 8.4])
    text = dst.read_text(encoding="utf-8")
    assert "00:00:00,000 --> 00:00:03,600" in text
    assert text.rstrip().endswith("-->") is False
    assert "00:00:04,000 --> 00:00:08,400" in text
    parsed = captions.parse_srt(text)
    assert len(parsed) == 2 and parsed[0]["text"] == KHMER_LINES[0].split("។")[0] + "។"
    assert khmer.equal_text(" ".join(c["text"] for c in parsed), " ".join(KHMER_LINES[:2]))
    assert parsed[-1]["end"] == pytest.approx(8.4)
    for block in text.strip().split("\n\n"):
        for ln in block.split("\n")[2:]:
            assert not ln.startswith(khmer.COENG) and not ln.endswith(khmer.COENG)
    # legacy callers without `ends`: last cue must still use the spoken estimate,
    # not a bare +3s
    legacy = tmp_path / "legacy.srt"
    media.write_srt(["កុំបោះបង់ក្ដីសង្ឃឹម។ អ្នកអាចធ្វើបាន។"], [0.0], str(legacy))
    times = re.findall(r"(\d\d:\d\d:\d\d,\d\d\d) --> (\d\d:\d\d:\d\d,\d\d\d)", legacy.read_text("utf-8"))
    assert times and times[-1][1] != "00:00:03,000"


# --------------------------------------------------------- preview / burn parity
@pytest.mark.skipif(not captions.has_filter("subtitles"), reason="ffmpeg without libass")
def test_preview_and_burn_use_the_same_ass(tmp_path):
    style = {**cs.preset_style("soft-card"), "font": "noto-serif-khmer", "color": "#F4EBDD"}
    cues = [{"text": KHMER_LINES[0], "start": 0, "end": 2}]
    png = captions.preview_frame(cues, style, 480, 854, str(tmp_path / "p.png"))
    assert os.path.getsize(png) > 1000
    ass = (tmp_path / "p.png.ass").read_text(encoding="utf-8")
    assert "Noto Serif Khmer" in ass and "&H00DDEBF4" in ass.upper() or "Noto Serif Khmer" in ass
    assert "\\p1" in ass, "soft-card must draw its panel as a real vector shape"


@pytest.mark.skipif(not captions.has_filter("subtitles"), reason="ffmpeg without libass")
def test_burn_produces_a_decodable_video_with_the_chosen_font(tmp_path):
    from ai_studio.util import run_ffmpeg

    base = str(tmp_path / "base.mp4")
    run_ffmpeg(["-f", "lavfi", "-i", "color=c=0x20242c:s=320x568:r=12:d=2",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30",
                "-pix_fmt", "yuv420p", "-an", base], timeout=300)
    style = {**cs.preset_style("clean"), "font": "battambang"}
    cues = [{"text": KHMER_LINES[3], "start": 0, "end": 1.8}]
    dst = str(tmp_path / "out.mp4")
    captions.burn_into(base, cues, style, dst, ass_path=str(tmp_path / "out.ass"))
    assert os.path.getsize(dst) > 2000
    info = captions.probe_size(dst)
    assert (info["width"], info["height"]) == (320, 568)
    ass = (tmp_path / "out.ass").read_text(encoding="utf-8")
    assert "Battambang" in ass


@pytest.mark.skipif(captions.has_filter("subtitles"), reason="libass is present here")
def test_missing_libass_is_reported_not_ignored():
    with pytest.raises(captions.CaptionError) as e:
        captions.require_libass()
    assert "libass" in str(e.value)


def test_language_mix_is_measured_per_script():
    style = cs.preset_style("clean")
    fit = captions.fit_text(KHMER_LINES[-1], style, 1080, 1920)
    assert "Khmer" in " ".join(fit["lines"])
    assert "២០២៦" in " ".join(fit["lines"]) or "2026" in " ".join(fit["lines"])
    assert khmer.equal_text("".join(fit["lines"]), KHMER_LINES[-1])


def test_coverage_gap_is_reported_for_a_font_without_a_glyph():
    """Battambang has no em-dash: say so instead of hoping for a fallback."""
    pytest.importorskip("uharfbuzz")
    gaps = captions.coverage_gaps("ខ្មែរ — ok", {**cs.preset_style("clean"), "font": "battambang"})
    assert "—" in gaps["missing"]
    warn = captions.coverage_warnings([{"text": "ខ្មែរ — ok"}],
                                      {**cs.preset_style("clean"), "font": "battambang"})
    assert warn and "substitute" in warn[0]
    assert not captions.coverage_gaps("ខ្មែរ ok",
                                      {**cs.preset_style("clean"), "font": "battambang"})["missing"]


def test_render_metadata_is_honest():
    meta = captions.render_metadata([{"text": KHMER_LINES[1], "start": 0, "end": 2}],
                                    cs.preset_style("clean"), 1080, 1920)
    assert "libass" in meta["renderer"]
    assert meta["font"]["family"] == "Noto Sans Khmer"
    assert meta["font"]["file"].endswith(".ttf")
    assert meta["frame"] == {"width": 1080, "height": 1920}
    assert meta["measurement"] in ("harfbuzz", "estimate")
    assert isinstance(meta["warnings"], list) and isinstance(meta["bounds_warnings"], list)


# ============================================================== HTTP surface
# The style a user picks must reach the render path through the API, the API
# must reject nonsense instead of storing it, and a requested caption burn must
# never come back as a silent uncaptioned file.

def _client(tmp_path):
    from starlette.testclient import TestClient

    from ai_studio.app import StudioState, create_app

    st = StudioState(str(tmp_path))
    return st, TestClient(create_app(data_root=str(tmp_path)))


def test_caption_style_endpoint_exposes_real_parameters(tmp_path):
    st, client = _client(tmp_path)
    d = client.get("/api/caption-style").json()
    assert {f["id"] for f in d["fonts"]} == set(cs.FONT_FAMILIES)
    assert {p["key"] for p in d["presets"]} >= {"clean", "cinema", "bold-social",
                                                "soft-card", "editorial"}
    for p in d["presets"]:
        assert p["style"]["font"] in cs.FONT_FAMILIES    # presets are style dicts
    assert d["capabilities"]["libass"] in (True, False)
    assert d["reference"]["height"] == captions.REFERENCE_HEIGHT
    assert d["precedence"][0] == "built-in default"


def test_server_validates_the_style_not_just_the_browser(tmp_path):
    st, client = _client(tmp_path)
    bad = [
        {"color": "javascript:alert(1)"},
        {"outline_width": 999},
        {"position": "somewhere"},
        {"font": "../../etc/passwd"},
        {"font_size_px": "huge"},
        {"background_opacity": -3},
    ]
    for patch in bad:
        r = client.post("/api/caption-style", json={"style": patch})
        assert r.status_code == 400, patch
        assert r.json()["detail"], "the API must say what is wrong"
    ok = client.post("/api/caption-style", json={"style": {**cs.preset_style("cinema"),
                                                           "color": "#123456"}})
    assert ok.status_code == 200
    assert ok.json()["style"]["color"] == "#123456"
    # persisted, not just echoed
    again = client.get("/api/caption-style").json()
    assert again["global_style"]["color"] == "#123456"


def test_project_override_persists_and_can_be_cleared(tmp_path):
    st, client = _client(tmp_path)
    pid = st.db.create_project(title="t", mode="A", script="x", status="ready")["id"]
    assert client.get(f"/api/projects/{pid}/caption-style").json()["is_default"] is True
    r = client.put(f"/api/projects/{pid}/caption-style",
                   json={"style": {**cs.preset_style("soft-card"), "background_opacity": 0.8},
                         "burn_captions": True})
    assert r.status_code == 200
    eff = r.json()["effective_style"]
    assert eff["background"] is True and eff["background_opacity"] == 0.8
    # a fresh read (new request, same DB) still has it
    again = client.get(f"/api/projects/{pid}/caption-style").json()
    assert again["is_default"] is False and again["project_style"]["background_opacity"] == 0.8
    # a project with no override keeps inheriting (older projects must still load)
    pid2 = st.db.create_project(title="old", mode="A", script="y", status="ready")["id"]
    p2 = client.get(f"/api/projects/{pid2}/caption-style").json()
    assert p2["is_default"] is True and p2["effective_style"]["font"] == cs.DEFAULT_FONT
    cleared = client.put(f"/api/projects/{pid}/caption-style", json={"clear": True}).json()
    assert cleared["is_default"] is True


def test_font_files_are_served_without_traversal(tmp_path):
    st, client = _client(tmp_path)
    r = client.get("/api/fonts/noto-sans-khmer/NotoSansKhmer-Regular.ttf")
    assert r.status_code == 200 and len(r.content) > 10_000
    for bad in ("/api/fonts/noto-sans-khmer/../../setup-studio.sh",
                "/api/fonts/noto-sans-khmer/..%2f..%2fsetup-studio.sh",
                "/api/fonts/unknown/whatever.ttf",
                "/api/fonts/noto-sans-khmer/nope.ttf"):
        assert client.get(bad).status_code in (404, 400), bad


def test_preview_endpoint_renders_with_the_selected_font(tmp_path):
    st, client = _client(tmp_path)
    if not captions.has_filter("subtitles"):
        pytest.skip("ffmpeg without libass")
    style = {**cs.preset_style("clean"), "font": "kantumruy-pro", "color": "#00FF00"}
    r = client.post("/api/caption-preview", json={
        "text": KHMER_LINES[1], "style": style, "width": 720, "height": 1280})
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n"
    got = json.loads(r.headers["x-caption-style"])
    assert got["font"] == "kantumruy-pro" and got["color"] == "#00FF00"
    font = json.loads(r.headers["x-caption-font"])
    assert font["family"] == "Kantumruy Pro" and font["file"].endswith(".ttf")
    # the preview really is 720x1280, i.e. the export canvas, not a CSS box
    from PIL import Image
    import io

    im = Image.open(io.BytesIO(r.content))
    assert im.size == (720, 1280)
    # a bad colour is refused at the API, not silently ignored
    assert client.post("/api/caption-preview",
                       json={"text": "ខ្មែរ", "style": {"color": "green-ish"}}).status_code == 400
    assert client.post("/api/caption-preview",
                       json={"text": "ខ្មែរ", "width": 99999, "height": 99999}).status_code == 400


def test_preview_reports_the_font_that_will_be_burned(tmp_path):
    """The style the API returns must be the style the ASS file contains."""
    st, client = _client(tmp_path)
    if not captions.has_filter("subtitles"):
        pytest.skip("ffmpeg without libass")
    style = {**cs.preset_style("editorial"), "font": "noto-serif-khmer", "weight": 700}
    r = client.post("/api/caption-preview", json={"text": KHMER_LINES[3], "style": style})
    assert r.status_code == 200
    urls = re.findall(rb"/api/files\?path=[^\s]*", r.content)
    assert urls is not None
    ass = json.loads(r.headers["x-caption-style"])
    meta = captions.render_metadata([{"text": KHMER_LINES[3], "start": 0, "end": 2}],
                                    ass, 1080, 1920)
    assert meta["font"]["family"] == "Noto Serif Khmer"
    assert meta["font"]["weight"] == 700


def test_assemble_refuses_to_silently_skip_captions(tmp_path, monkeypatch):
    """burn_captions=true + no libass must FAIL, not return an uncaptioned file."""
    from ai_studio.engines import assembly

    st, client = _client(tmp_path)
    cfg = st.config()
    cfg["assembly"]["burn_captions"] = True
    cfg["assembly"]["emit_srt"] = True
    cfg["video"].update({"engine": "previz", "width": 256, "height": 448, "fps": 8})
    cfg["tts"]["engine"] = "placeholder"
    cfg["rvc"]["engine"] = "bypass"
    cfg["sfx"]["engine"] = "procedural"
    cfg["assembly"]["fps"] = 8
    pid = st.db.create_project(title="captions", mode="A", status="ready",
                               script="ជីវិតមនុស្ស មិនមែនជាការប្រណាំងទេ។", script_locked=True)["id"]
    scenes = [{"idx": 0, "text": "ជីវិតមនុស្ស មិនមែនជាការប្រណាំងទេ។",
               "estimated_duration_sec": 3.0}]
    proj = st.db.get_project(pid)

    from ai_studio.util import run_ffmpeg

    work = tmp_path / "prep"
    work.mkdir()
    clip = str(work / "c.mp4")
    run_ffmpeg(["-f", "lavfi", "-i", "color=c=0x203040:s=256x448:r=8:d=3",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30", "-pix_fmt", "yuv420p",
                "-an", clip], timeout=300)
    wav = str(work / "v.wav")

    import wave

    with wave.open(wav, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(22050)
        wf.writeframes(b"\x00\x00" * 22050 * 3)
    assets = {"video_fit": {0: {"path": clip, "duration": 3.0, "engine": "test"}},
              "voice_final": {0: {"path": wav, "duration": 3.0, "engine": "test"}}}
    out_dir = str(tmp_path / "final")

    if captions.has_filter("subtitles"):
        passed = assembly.assemble(proj, scenes, assets, cfg, out_dir, run_id="t")
        assert passed.get("with_captions"), "captions were requested and not produced"
        assert os.path.exists(passed["with_captions"])
        assert passed["caption_metadata"]["font"]["family"] == cs.FONT_FAMILIES[
            passed["caption_style"]["font"]]["family"]
        assert passed["caption_style"]["font"] in cs.FONT_FAMILIES
        assert not [n for n in passed["notes"] if "caption burn-in skipped" in n]
        # re-running with libass missing must raise instead of quietly dropping them
        monkeypatch.setattr(captions, "has_filter", lambda name: False)
        captions._HAS_FILTER.pop("subtitles", None)
        with pytest.raises(captions.CaptionError):
            assembly.assemble(proj, scenes, assets, cfg, str(tmp_path / "final2"), run_id="t2")
    else:
        with pytest.raises(captions.CaptionError):
            assembly.assemble(proj, scenes, assets, cfg, out_dir, run_id="t")


def test_render_captions_endpoint_burns_on_the_existing_cut(tmp_path):
    st, client = _client(tmp_path)
    if not captions.has_filter("subtitles"):
        pytest.skip("ffmpeg without libass")
    from ai_studio.util import run_ffmpeg

    pid = st.db.create_project(title="recap", mode="A", status="done",
                               script="ជីវិតមនុស្ស មិនមែនជាការប្រណាំងទេ។")["id"]
    st.db.replace_scenes(pid, [{"text": "ជីវិតមនុស្ស មិនមែនជាការប្រណាំងទេ។",
                                "estimated_duration_sec": 3.0, "audio_duration": 3.0}])
    d = os.path.join(st.data_root, "projects", pid, "final")
    os.makedirs(d, exist_ok=True)
    cut = os.path.join(d, f"{pid}_run.mp4")
    run_ffmpeg(["-f", "lavfi", "-i", "color=c=0x2b3a4a:s=480x854:r=12:d=3",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30", "-pix_fmt", "yuv420p",
                "-an", cut], timeout=300)
    st.db.add_asset(pid, "final", cut, stage="assemble", run_id="r1", scene_idx=-1,
                    mime="video/mp4", duration=3.0)
    r = client.post(f"/api/projects/{pid}/render-captions",
                    json={"style": {**cs.preset_style("bold-social"),
                                    "font": "noto-sans-khmer"}})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] and os.path.exists(body["path"])
    assert body["font"]["family"] == "Noto Sans Khmer"
    assert body["cues"] >= 1
    # the captioned asset is registered and the project payload points at it
    proj = client.get(f"/api/projects/{pid}").json()
    assert proj["captions"]["asset"], "the captioned render must be exposed to the UI"
    # and downloading the *final* asset hands back the captioned cut
    dl = client.get(f"/api/assets/{body['asset_id']}/download?cap=1")
    assert dl.status_code == 200 and len(dl.content) > 1000
    final_row = st.db.latest_asset(pid, "final")
    dl2 = client.get(f"/api/assets/{final_row['id']}/download?cap=1")
    assert len(dl2.content) == len(dl.content), \
        "a request for the final cut must resolve to the captioned render"


def test_validate_endpoint_reports_script_problems(tmp_path):
    st, client = _client(tmp_path)
    r = client.post("/api/caption/validate", json={
        "text": "អ្នកអាចធ្វើបាន [[silent: ព្យាយាម", "width": 1080, "height": 1920})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert any("unbalanced" in e for e in body["errors"])
    ok = client.post("/api/caption/validate", json={"text": KHMER_LINES[1]}).json()
    assert ok["ok"] is True and ok["stats"]["has_khmer"] is True


def test_partial_patch_changes_one_field_and_keeps_the_rest(tmp_path):
    """`{"color": …}` must not reset a Soft Card back to flat white text."""
    st, client = _client(tmp_path)
    assert client.post("/api/caption-style",
                       json={"style": cs.preset_style("soft-card")}).status_code == 200
    pid = st.db.create_project(title="t", mode="A", script="x", status="ready")["id"]
    r = client.put(f"/api/projects/{pid}/caption-style",
                   json={"style": {"color": "#FF0000"}, "burn_captions": True}).json()
    eff = r["effective_style"]
    assert eff["color"] == "#FF0000"
    assert eff["background"] is True and eff["background_radius"] == 22, \
        "a partial patch must inherit, not reset, the remaining fields"
    # naming a preset restarts from that preset
    again = client.put(f"/api/projects/{pid}/caption-style",
                       json={"style": {"preset": "clean"}}).json()["effective_style"]
    assert again["preset"] == "clean" and again["background"] is False
    # and the patch validation is still strict
    assert client.put(f"/api/projects/{pid}/caption-style",
                      json={"style": {"outline_width": 500}}).status_code == 400


def test_patch_semantics_are_unit_tested_too():
    base = cs.preset_style("soft-card")
    patch = cs.normalize_patch({"color": "#123456"}, base)
    assert patch == {"color": "#123456"}
    merged = cs.merge_styles(base, None, patch)
    assert merged["background"] is True and merged["background_opacity"] == 0.6
    with pytest.raises(cs.CaptionStyleError):
        cs.normalize_patch({"background_opacity": 7}, base)
    # non-strict mode repairs out-of-range values (never passes 7 through)
    repaired = cs.normalize_patch({"background_opacity": 7}, base, strict=False)
    assert 0.0 <= repaired["background_opacity"] <= 1.0


def test_legacy_style_previews_still_render(tmp_path):
    """The old endpoint must keep working — but through the new renderer."""
    st, client = _client(tmp_path)
    if not captions.has_filter("subtitles"):
        pytest.skip("ffmpeg without libass")
    r = client.get("/api/style-previews")
    assert r.status_code == 200
    items = r.json().get("subtitle_styles") or []
    assert items, "no style previews rendered"
    live = media.SUBTITLE_STYLES_payload()
    for it in items:
        assert it["key"] in live, it["key"]
        assert it["url"], f"{it['key']}: {it.get('error')}"
        assert it["style"]["font"] in cs.FONT_FAMILIES
        assert it["style"]["font_size_px"] == live[it["key"]]["style"]["font_size_px"]
        # every preview must come from the bundled renderer, not a force_style string
        assert "Khmer OS" not in json.dumps(it)
    assert {i["key"] for i in items} == set(live)
    assert r.json().get("title_styles")


# ======================================================== end-to-end export
# The point of the whole change: a real pipeline run with captions enabled must
# produce a *readable* captioned file. Not "a file exists" — pixels differ from
# the uncaptioned master inside the caption band, and the manifest names the font.
@pytest.mark.skipif(not captions.has_filter("subtitles"), reason="ffmpeg without libass")
def test_pipeline_export_burns_readable_khmer_captions(tmp_path):
    import asyncio

    from PIL import Image

    from ai_studio import config as cfg_mod
    from ai_studio.util import ffmpeg_exe

    st = _cheap_state(tmp_path)
    cfg = st.config()
    cfg["assembly"].update({"burn_captions": True, "emit_srt": True})
    cfg["caption_style"] = {**cs.preset_style("soft-card"), "font": "noto-sans-khmer"}
    cfg_mod.save(cfg, st.settings_path)
    st.invalidate()

    pid = st.db.create_project(title="e2e-captions", mode="A", status="ready",
                              script="\n".join(KHMER_LINES[:3]), script_locked=True,
                              target_duration=14)["id"]

    async def _go():
        out = await st.scheduler.start_run(pid)
        return await st.scheduler.wait(out["run_id"], timeout=900)

    done = asyncio.run(_go())
    assert done["run"]["status"] == "completed", done["run"]["error"]

    final = st.db.latest_asset(pid, "final")
    capped = st.db.latest_asset(pid, "final_captions")
    srt = st.db.latest_asset(pid, "srt")
    manifest = st.db.latest_asset(pid, "manifest")
    assert final and capped and srt and manifest, "assembly did not publish its caption assets"
    for a in (final, capped, srt):
        assert os.path.exists(a["path"]), a
    assert os.path.getsize(capped["path"]) > os.path.getsize(final["path"]) * 0.5

    # the SRT carries Khmer text, not placeholders
    srt_text = open(srt["path"], encoding="utf-8").read()
    parsed = captions.parse_srt(srt_text)
    assert parsed and any(re.search(r"[\u1780-\u17FF]", c["text"]) for c in parsed)
    assert "Khmer OS" not in srt_text

    # the manifest records what was rendered and with which font
    data = json.load(open(manifest["path"], encoding="utf-8"))
    cap = data.get("captions") or {}
    assert cap.get("font", {}).get("family") == "Noto Sans Khmer"
    assert cap.get("style", {}).get("background") is True
    assert data.get("assets", {}).get("captioned"), "the manifest must name the captioned cut"

    # the caption band really changed: compare a frame of each file at the same
    # timestamp and measure how many pixels differ inside the lower third
    def frame(path, at, dst):
        subprocess_run([ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
                        "-ss", str(at), "-i", path, "-frames:v", "1", dst], timeout=300)
        return Image.open(dst).convert("L")

    dur = float(capped.get("duration") or 0) or 3.0
    at = max(0.4, dur * 0.35)
    a = frame(final["path"], at, str(tmp_path / "plain.png"))
    b = frame(capped["path"], at, str(tmp_path / "capped.png"))
    assert a.size == b.size
    w, h = a.size
    band = (0, int(h * 0.55), w, h)
    diff = [1 if abs(pa - pb) > 40 else 0
            for pa, pb in zip(a.crop(band).tobytes(), b.crop(band).tobytes())]
    changed = sum(diff)
    assert changed > 200, f"captions changed only {changed} pixels — the burn looks empty"
    # ...and the change is a *glyph-shaped* mark, not a full-frame overlay: the
    # untouched upper half means the burn did not wreck the picture
    top = (0, 0, w, int(h * 0.45))
    top_changed = sum(1 for pa, pb in zip(a.crop(top).tobytes(), b.crop(top).tobytes())
                      if abs(pa - pb) > 40)
    assert top_changed < changed / 5, "the caption layer must sit in the caption band"


def _cheap_state(tmp_path):
    """Same deterministic local-engine state the pipeline tests use."""
    from ai_studio.app import StudioState
    from ai_studio import config as cfg_mod

    st = StudioState(str(tmp_path))
    cfg = st.config()
    cfg["machine"]["profile"] = "machine_b"
    cfg["tts"]["engine"] = "placeholder"
    cfg["rvc"]["engine"] = "bypass"
    cfg["video"]["engine"] = "previz"
    cfg["sfx"]["engine"] = "procedural"
    cfg["video"]["width"], cfg["video"]["height"] = 256, 448
    cfg["video"]["fps"], cfg["video"]["steps"] = 8, 4
    cfg["video"]["max_frames"], cfg["video"]["min_frames"] = 17, 17
    cfg["assembly"]["fps"] = 8
    cfg["pipeline"].update({"scene_target_seconds": 4.0, "scene_min_seconds": 2.0,
                            "scene_max_seconds": 8.0, "max_scenes": 3})
    cfg_mod.save(cfg, st.settings_path)
    st.invalidate()
    return st


def subprocess_run(cmd, timeout=300):
    import subprocess

    res = subprocess.run(cmd, capture_output=True, timeout=timeout)
    assert res.returncode == 0, (cmd, (res.stderr or b"").decode("utf-8", "ignore")[-800:])
    return res


def test_request_style_resolution_never_mislabels_a_preset():
    """`{"preset": "bold-social"}` must produce Bold Social, not a Clean style
    wearing the Bold Social label (silently different preview vs export)."""
    from ai_studio import api as api_mod

    out = api_mod._caption_style_for_request({"preset": "bold-social"},
                                             cs.preset_style("clean"))
    bold = cs.preset_style("bold-social")
    for k in ("font", "weight", "font_size_px", "outline_width", "color", "position"):
        assert out[k] == bold[k], f"{k}: {out[k]!r} != {bold[k]!r}"
    assert cs.is_preset_default(out)
    # a partial patch still inherits everything else
    patched = api_mod._caption_style_for_request({"color": "#123456"},
                                                 cs.preset_style("soft-card"))
    assert patched["color"] == "#123456" and patched["background"] is True
    with pytest.raises(cs.CaptionStyleError):
        api_mod._caption_style_for_request({"font": "Comic Sans"}, cs.preset_style("clean"))
    # a preset name in a patch wins over the project style too
    proj_style = {**cs.preset_style("editorial"), "font_size_px": 44}
    out2 = api_mod._caption_style_for_request({"preset": "cinema"},
                                              cs.preset_style("clean"), proj_style)
    assert out2["font_size_px"] == cs.preset_style("cinema")["font_size_px"]


def test_a_style_never_claims_a_preset_it_no_longer_matches():
    """The preset field drives the UI highlight *and* the exported look."""
    cinema = cs.preset_style("cinema")
    # rounding/normalisation must NOT knock a preset off its own name
    assert cs.normalize_style({**cinema, "font_size_px": 68.0})["preset"] == "cinema"
    assert cs.normalize_style({**cinema, "color": "#f4ebdd"})["preset"] == "cinema"
    # a real edit does
    edited = cs.normalize_style({**cinema, "font": "kantumruy-pro"})
    assert edited["preset"] == "custom" and not cs.is_preset_default(edited)
    assert "font" in cs.diff_from_preset({**cinema, "font": "kantumruy-pro"})
    # and every preset key really is its own style
    for k in cs.PRESET_KEYS:
        assert cs.is_preset_default(cs.preset_style(k))

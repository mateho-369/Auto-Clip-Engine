"""Caption typography — the single source of truth for how captions look.

Why this module exists: burned Khmer captions are only "correct" when the whole
chain agrees — one font *file*, one shaped layout, one set of colors — across
the web preview, the SRT sidecar and the burned-in MP4. This module owns:

* the **font registry** (5 bundled, redistribution-permitted Khmer families
  with their real weights + license files, served locally — never a CDN);
* the **style schema** (validated, versioned, backwards compatible — legacy
  ``assembly.subtitle_style`` keys map into it);
* the **presets** (Clean / Cinema / Bold Social / Soft Card / Editorial);
* **shaped pixel measurement** via uharfbuzz (the same HarfBuzz shaping libass
  performs), so wrapping breaks between Khmer *words* at real pixel widths —
  never inside a coeng cluster, never mid-word;
* the **ASS builder** used for both the karaoke highlight style and the
  sentence styles, with explicit ``\\pos`` layout so line spacing, margins,
  max width and the soft-card panel are honoured exactly at burn-in;
* the **preview renderer** — the same ASS file through the same libass burn
  filter the exporter uses, so what you preview is what renders.

Honesty rules encoded here: karaoke timing is *proportional estimation*
(sherpa-onnx gives no word timestamps) and is labelled as such everywhere;
the panel's corner radius exists in the web preview but libass draws a square
box (documented limitation); missing fonts/shapers raise actionable errors
instead of silently exporting tofu.
"""
from __future__ import annotations

import hashlib
import os
import re
import threading

from .util import ensure_dir, run_ffmpeg

FONTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "fonts")

# ---------------------------------------------------------------- font registry
# Every family below ships in this repo (SIL OFL 1.1) — no CDN, no system-font
# roulette. ``family`` is the font's *internal* name (verified with fontTools);
# libass matches against it via the fontsdir we pass to the subtitles filter.
FONTS: dict[str, dict] = {
    "noto_sans_khmer": {
        "family": "Noto Sans Khmer", "label": "Noto Sans Khmer",
        "weights": {"regular": "NotoSansKhmer-Regular.ttf", "bold": "NotoSansKhmer-Bold.ttf"},
        "license": "OFL-NotoSansKhmer.txt",
        "note": "clean modern sans — the default for subtitles and UI",
    },
    "noto_serif_khmer": {
        "family": "Noto Serif Khmer", "label": "Noto Serif Khmer",
        "weights": {"regular": "NotoSerifKhmer_400Regular.ttf", "bold": "NotoSerifKhmer_700Bold.ttf"},
        "license": "OFL-NotoSerifKhmer.txt",
        "note": "editorial serif — formal, calm long-form narration",
    },
    "battambang": {
        "family": "Battambang", "label": "Battambang",
        "weights": {"regular": "Battambang-Regular.ttf", "bold": "Battambang-Bold.ttf"},
        "license": "OFL-Battambang.txt",
        "note": "familiar Khmer body typeface",
    },
    "kantumruy_pro": {
        "family": "Kantumruy Pro", "label": "Kantumruy Pro",
        "weights": {"regular": "KantumruyPro_400Regular.ttf", "bold": "KantumruyPro_700Bold.ttf"},
        "license": "OFL-KantumruyPro.txt",
        "note": "contemporary display/UI sans — strong at large sizes",
    },
    "moul": {
        "family": "Moul", "label": "Moul",
        "weights": {"regular": "Moul-Regular.ttf"},
        "license": "OFL-Moul.txt",
        "note": "traditional display face — titles and emphasis only (regular weight)",
    },
}

# The reference canvas for px-kind controls (outline, shadow offset, padding,
# radius). At other resolutions they scale by height/854 so proportions hold.
REF_H = 854.0

SCHEMA_VERSION = 1

DEFAULT_STYLE: dict = {
    "version": SCHEMA_VERSION,
    "preset": "clean",
    "font": "noto_sans_khmer",
    "weight": "regular",
    "size_pct": 4.6,                # % of output height → px at render time
    "text_color": "#ffffff",
    "outline_color": "#0a0c10",
    "outline_px": 2.0,
    "shadow": True,
    "shadow_strength": 0.55,
    "shadow_offset_px": 2,
    "panel": {"enabled": False, "color": "#0e1116", "opacity": 0.62,
              "padding_px": 12, "radius_px": 10},
    "position": "bottom",           # bottom | center | top
    "align": "center",              # left | center | right
    "margin_h_pct": 6.0,            # % of width
    "margin_v_pct": 7.0,            # % of height
    "line_spacing": 1.25,           # multiplier on natural leading (\pos layout)
    "max_line_width_pct": 88.0,     # % of width — wrap budget
    "max_lines": 2,
    "karaoke": False,               # word-highlight sweep (proportional timing)
}

# Presets must remain *complete* styles: selecting one re-derives every
# parameter (then the UI shows exactly which fields the user has modified).
PRESETS: dict[str, dict] = {
    "clean": {},
    "cinema": {
        "font": "noto_serif_khmer", "weight": "regular", "size_pct": 4.3,
        "text_color": "#f5efe0", "outline_color": "#1a1208", "outline_px": 1.5,
        "shadow": True, "shadow_strength": 0.65, "shadow_offset_px": 2,
        "panel": {"enabled": False, "color": "#0e1116", "opacity": 0.62,
                  "padding_px": 12, "radius_px": 10},
        "position": "bottom", "align": "center", "margin_h_pct": 7.0, "margin_v_pct": 8.0,
        "line_spacing": 1.3, "max_line_width_pct": 86.0, "max_lines": 2, "karaoke": False,
    },
    "bold_social": {
        "font": "kantumruy_pro", "weight": "bold", "size_pct": 5.4,
        "text_color": "#ffe23d", "outline_color": "#101014", "outline_px": 3.0,
        "shadow": False, "shadow_strength": 0.5, "shadow_offset_px": 2,
        "panel": {"enabled": False, "color": "#0e1116", "opacity": 0.62,
                  "padding_px": 12, "radius_px": 10},
        "position": "bottom", "align": "center", "margin_h_pct": 5.0, "margin_v_pct": 6.0,
        "line_spacing": 1.2, "max_line_width_pct": 90.0, "max_lines": 2, "karaoke": False,
    },
    "soft_card": {
        "font": "noto_sans_khmer", "weight": "regular", "size_pct": 4.2,
        "text_color": "#ffffff", "outline_color": "#0a0c10", "outline_px": 0.0,
        "shadow": False, "shadow_strength": 0.4, "shadow_offset_px": 1,
        "panel": {"enabled": True, "color": "#0e1116", "opacity": 0.62,
                  "padding_px": 12, "radius_px": 10},
        "position": "bottom", "align": "center", "margin_h_pct": 6.0, "margin_v_pct": 7.0,
        "line_spacing": 1.25, "max_line_width_pct": 84.0, "max_lines": 2, "karaoke": False,
    },
    "editorial": {
        "font": "noto_serif_khmer", "weight": "regular", "size_pct": 4.0,
        "text_color": "#ece7db", "outline_color": "#14100a", "outline_px": 1.0,
        "shadow": True, "shadow_strength": 0.4, "shadow_offset_px": 2,
        "panel": {"enabled": False, "color": "#0e1116", "opacity": 0.62,
                  "padding_px": 12, "radius_px": 10},
        "position": "bottom", "align": "left", "margin_h_pct": 8.0, "margin_v_pct": 9.0,
        "line_spacing": 1.35, "max_line_width_pct": 80.0, "max_lines": 2, "karaoke": False,
    },
}

# Legacy per-project keys (assembly.subtitle_style) → style overrides, so old
# projects keep their look after the upgrade. ``karaoke`` maps to clean+karaoke.
LEGACY_MAP: dict[str, dict] = {
    "clean": {},
    "bold_yellow": {"font": "battambang", "weight": "bold", "size_pct": 5.0,
                    "text_color": "#ffff00", "outline_px": 3.0},
    "minimal_top": {"position": "top", "size_pct": 3.8, "margin_v_pct": 4.5},
    "karaoke": {"karaoke": True, "text_color": "#ffff00"},
}

# Validation ranges (server-enforced; the UI reads them from /api/captions/schema).
RANGES = {
    "size_pct": (2.2, 9.5),
    "outline_px": (0.0, 8.0),
    "shadow_strength": (0.0, 1.0),
    "shadow_offset_px": (0, 8),
    "margin_h_pct": (2.0, 20.0),
    "margin_v_pct": (2.0, 20.0),
    "line_spacing": (0.9, 2.0),
    "max_line_width_pct": (50.0, 96.0),
    "max_lines": (1, 3),
    "panel.opacity": (0.0, 1.0),
    "panel.padding_px": (0, 28),
    "panel.radius_px": (0, 28),
}
POSITIONS = ("bottom", "center", "top")
ALIGNS = ("left", "center", "right")
HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


class CaptionStyleError(ValueError):
    """Raised with an actionable message when a style cannot render."""


# ---------------------------------------------------------------- resolution
def deep_merge(base: dict, over: dict) -> dict:
    out = dict(base or {})
    for k, v in (over or {}).items():
        if k == "version":
            continue
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        elif v is not None:
            out[k] = v
    return out


def effective_style(project: dict | None = None, global_settings: dict | None = None) -> dict:
    """DEFAULTS ← global settings.captions ← project legacy key ← project override.

    Precedence (documented in README + UI): the newest layer wins. A legacy
    ``assembly.subtitle_style`` project key applies only when the project has
    no modern ``settings.captions`` block, so nothing saved by older versions
    is silently discarded.
    """
    style = dict(DEFAULT_STYLE)
    g = (global_settings or {}).get("captions") or {}
    if g:
        style = deep_merge(style, g)
    proj = project or {}
    proj_settings = proj.get("settings") or {}
    modern = proj_settings.get("captions") or {}
    legacy_key = ((proj_settings.get("assembly") or {}).get("subtitle_style")
                  or (proj.get("settings") or {}).get("assembly", {}).get("subtitle_style")
                  if isinstance(proj.get("settings"), dict) else "")
    if modern:
        style = deep_merge(style, modern)
    elif legacy_key:
        style = deep_merge(style, LEGACY_MAP.get(str(legacy_key), {}))
    style, _issues = validate_style(style)
    return style


# ---------------------------------------------------------------- validation
def validate_style(raw: dict) -> tuple[dict, list[str]]:
    """Return (clean_style, issues). Clamps numerics, validates enums/colors,
    fonts and weights. Never raises for style input from the API — issues are
    surfaced so the caller can tell the user what was adjusted."""
    issues: list[str] = []
    raw = dict(raw or {})
    # a named preset expands server-side: DEFAULT ← preset ← explicit overrides.
    # A payload that only names the preset ("pick bold_social") therefore gets
    # the full look; a payload that also carries values keeps those values
    # (they are the user's modifications of that preset).
    base = deep_merge(DEFAULT_STYLE, PRESETS.get(str(raw.get("preset") or ""), {}))
    s = deep_merge(base, {k: v for k, v in raw.items() if k != "preset"})
    s["preset"] = str(raw.get("preset") or "clean")

    if s.get("font") not in FONTS:
        issues.append(f"font '{s.get('font')}' is not in the bundled registry — "
                      f"reset to noto_sans_khmer (options: {', '.join(FONTS)})")
        s["font"] = "noto_sans_khmer"
    weights = FONTS[s["font"]]["weights"]
    if s.get("weight") not in weights:
        issues.append(f"weight '{s.get('weight')}' does not exist for {s['font']} "
                      f"(available: {', '.join(weights)}) — using regular")
        s["weight"] = "regular"

    def _clamp_num(path, val, lo, hi):
        try:
            f = float(val)
        except (TypeError, ValueError):
            issues.append(f"{path}: not a number ({val!r}) — reset to {lo if path != 'line_spacing' else 1.25}")
            return lo if path != "line_spacing" else 1.25
        if f != f:  # NaN
            issues.append(f"{path}: NaN — reset")
            return lo if path != "line_spacing" else 1.25
        c = min(hi, max(lo, f))
        if c != f:
            issues.append(f"{path} {f:g} outside {lo:g}–{hi:g} — clamped to {c:g}")
        return c

    for key in ("size_pct", "outline_px", "shadow_strength", "shadow_offset_px",
                "margin_h_pct", "margin_v_pct", "line_spacing", "max_line_width_pct"):
        lo, hi = RANGES[key]
        s[key] = _clamp_num(key, s.get(key), lo, hi)
    s["max_lines"] = int(_clamp_num("max_lines", s.get("max_lines"), *RANGES["max_lines"]))

    for key in ("text_color", "outline_color"):
        v = str(s.get(key) or "")
        if not HEX_RE.match(v):
            issues.append(f"{key} '{v}' is not #rrggbb — reset to {DEFAULT_STYLE[key]}")
            s[key] = DEFAULT_STYLE[key]
    panel = s.get("panel") or {}
    if not isinstance(panel, dict):
        panel = dict(DEFAULT_STYLE["panel"])
    for key in ("opacity", "padding_px", "radius_px"):
        lo, hi = RANGES[f"panel.{key}"]
        panel[key] = _clamp_num(f"panel.{key}", panel.get(key), lo, hi)
    panel["enabled"] = bool(panel.get("enabled"))
    if not HEX_RE.match(str(panel.get("color") or "")):
        panel["color"] = DEFAULT_STYLE["panel"]["color"]
        issues.append("panel.color is not #rrggbb — reset")
    s["panel"] = panel

    if s.get("position") not in POSITIONS:
        issues.append(f"position '{s.get('position')}' invalid — bottom")
        s["position"] = "bottom"
    if s.get("align") not in ALIGNS:
        issues.append(f"align '{s.get('align')}' invalid — center")
        s["align"] = "center"
    s["shadow"] = bool(s.get("shadow"))
    s["karaoke"] = bool(s.get("karaoke"))
    preset = str(s.get("preset") or "clean")
    s["preset"] = preset if preset in PRESETS or preset == "custom" else "clean"
    s["version"] = SCHEMA_VERSION
    return s, issues


def modified_fields(style: dict) -> list[str]:
    """Which fields differ from the named preset (shown as 'modified' in the UI)."""
    base = dict(DEFAULT_STYLE)
    base.update(PRESETS.get(style.get("preset") or "clean", {}))
    diffs = []
    for k, v in style.items():
        if k in ("version", "preset"):
            continue
        if base.get(k) != v:
            diffs.append(k)
    return diffs


# ---------------------------------------------------------------- fonts
def font_file(font_id: str, weight: str = "regular") -> str:
    """Absolute path of the bundled TTF, or raise an actionable error."""
    spec = FONTS.get(font_id)
    if not spec:
        raise CaptionStyleError(
            f"unknown font '{font_id}' — bundled fonts: {', '.join(FONTS)}")
    fn = spec["weights"].get(weight) or spec["weights"].get("regular")
    if not fn or weight not in spec["weights"]:
        raise CaptionStyleError(
            f"font '{font_id}' has no {weight} file (available weights: "
            f"{', '.join(spec['weights'])})")
    p = os.path.join(FONTS_DIR, fn)
    if not os.path.exists(p):
        raise CaptionStyleError(
            f"font file missing: {fn} — the studio's bundled fonts were removed "
            f"from ai_studio/assets/fonts; re-install the repo or restore the file")
    return p


def font_family_name(font_id: str) -> str:
    return FONTS[font_id]["family"]


def fonts_dir() -> str:
    return FONTS_DIR


# ---------------------------------------------------------------- shaping
class Shaper:
    """HarfBuzz shaping widths for one font file (memoised per (file,size))."""

    def __init__(self, ttf_path: str, size_px: float):
        self.path = ttf_path
        self.size_px = float(size_px)
        import uharfbuzz as hb
        blob = hb.Blob.from_file_path(ttf_path)
        face = hb.Face(blob)
        self.upem = face.upem
        self._hb = hb
        self._font = hb.Font(face)
        self._scale = self.size_px / float(self.upem)
        # per-cluster advance cache for fallback wrap planning
        self._cache: dict[str, float] = {}

    def width(self, text: str) -> float:
        """Shaped advance width of `text` in pixels (matches libass closely:
        both shape with HarfBuzz and rasterise through FreeType)."""
        key = text
        got = self._cache.get(key)
        if got is not None:
            return got
        buf = self._hb.Buffer()
        buf.add_str(text)
        buf.guess_segment_properties()
        self._hb.shape(self._font, buf)
        units = sum(p.x_advance for p in buf.glyph_positions)
        w = units * self._scale
        self._cache[key] = w
        return w

    def widths_per_word(self, words: list[str]) -> list[float]:
        return [self.width(w) for w in words]


def measure_shaper(style: dict, out_w: int, out_h: int) -> Shaper:
    size_px = style["size_pct"] / 100.0 * float(out_h)
    return Shaper(font_file(style["font"], style["weight"]), size_px)


# ---------------------------------------------------------------- wrap plan
def wrap_block(text: str, style: dict, shaper: Shaper, out_w: int) -> dict:
    """Wrap one caption into lines ≤ the style's pixel budget via the shared
    khmer.wrap_words (word-safe, space-preserving, shaped widths). A single
    word wider than the whole budget stays on its own line and the block is
    flagged for bounded shrink (see plan_blocks). Explicit newlines in the
    source text are hard breaks."""
    from . import khmer as kh

    budget = max(40.0, out_w * style["max_line_width_pct"] / 100.0
                 - (2.0 * style["panel"]["padding_px"] if style["panel"]["enabled"] else 0.0))
    lines: list[str] = []
    for hard in (text or "").split("\n"):
        hard = hard.strip()
        if not hard:
            continue
        lines.extend(kh.wrap_words(hard, width_fn=shaper.width, budget_px=budget))
    out = lines or [""]
    longest = max((shaper.width(l) for l in out), default=0.0)
    return {"lines": out, "longest_px": longest,
            "overflow": longest > budget + 0.5, "budget_px": budget}


def plan_blocks(texts: list[str], style: dict, out_w: int, out_h: int) -> dict:
    """Wrap every caption; if any line still overflows (one huge word) or a
    block needs more than max_lines, apply a bounded *per-block* font shrink
    (≥0.72 of the chosen size — never letterspacing, which would break
    shaping). Blocks that cannot fit even at the floor keep all their lines
    and produce an explicit warning instead of losing text."""
    shaper = measure_shaper(style, out_w, out_h)
    plans = [wrap_block(t, style, shaper, out_w) for t in texts]
    # single-word overflow: shrink just enough that the widest line fits
    overflowing = [p for p in plans if p["overflow"]]
    if overflowing:
        need = max(p["longest_px"] / p["budget_px"] for p in overflowing)
        ov_scale = max(0.72, 1.0 / need)
        if ov_scale < 1.0:
            shaper2 = Shaper(shaper.path, shaper.size_px * ov_scale)
            plans = [wrap_block(t, style, shaper2, out_w) for t in texts]
    # max_lines: per-block bounded shrink (floor 0.72); below the floor keep
    # the lines and warn — never drop text to satisfy a layout rule.
    for i, p in enumerate(plans):
        p["scale"] = 1.0
        if len(p["lines"]) > style["max_lines"] and style["max_lines"] > 0:
            total_w = sum(shaper.width(l) for l in p["lines"])
            if total_w > 0:
                fit = max(0.72, min(1.0, (style["max_lines"] * p["budget_px"]) / total_w))
                if fit < 0.999:
                    sh = Shaper(shaper.path, shaper.size_px * fit)
                    p2 = wrap_block(texts[i], style, sh, out_w)
                    # accept the shrink only when it actually reduces the line
                    # count — otherwise keep the full size and warn honestly
                    if len(p2["lines"]) < len(p["lines"]):
                        p2["scale"] = fit
                        plans[i] = p2
    warnings = []
    for i, p in enumerate(plans):
        if len(p["lines"]) > style["max_lines"]:
            warnings.append(
                f"caption {i + 1} needs {len(p['lines'])} lines even at the minimum size "
                f"(max {style['max_lines']}) — all text is kept; raise max_lines, widen "
                "max_line_width_pct or lower font size")
        if p["overflow"]:
            warnings.append(f"caption {i + 1} contains a word wider than the text area — "
                            "rendered at reduced size instead of cutting it off")
    return {"plans": plans, "scale": 1.0, "warnings": warnings,
            "shaper_px": shaper.size_px, "shaper": shaper}


# ---------------------------------------------------------------- ASS build
def _ass_color(hex6: str, alpha01: float = 0.0) -> str:
    """#rrggbb + alpha(0=opaque) → ASS &HAABBGGRR."""
    h = hex6.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    a = int(round(max(0.0, min(1.0, alpha01)) * 255))
    return f"&H{a:02X}{b:02X}{g:02X}{r:02X}"


def _hex(h6: str) -> str:
    h = h6.lstrip("#")
    return f"&H00{h[4:6]}{h[2:4]}{h[0:2]}".upper()


def _fmt_ass_time(sec: float) -> str:
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h}:{m:02d}:{int(s):02d}.{int(round((s % 1) * 100)):02d}"


def scale_px(style: dict, v: float, out_h: int) -> float:
    return v * (float(out_h) / REF_H)


def style_ass_header(style: dict, out_w: int, out_h: int, karaoke: bool = False) -> str:
    """The Style: line honouring font/weight/size/colors/outline/shadow/panel."""
    fam = font_family_name(style["font"])
    size_px = style["size_pct"] / 100.0 * float(out_h) * 0.92  # libass size ≈ px height
    ml = style["margin_h_pct"] / 100.0 * out_w
    mr = ml
    mv = style["margin_v_pct"] / 100.0 * out_h
    outline = scale_px(style, style["outline_px"], out_h)
    shadow = scale_px(style, style["shadow_offset_px"], out_h) if style["shadow"] else 0.0
    border = 4 if style["panel"]["enabled"] else 1
    # BorderStyle=4 (libass): BackColour box behind each line; Outline adds box padding
    if border == 4:
        outline = scale_px(style, style["panel"]["padding_px"] * 0.6, out_h)
    align = {"bottom": 2, "center": 5, "top": 8}[style["position"]]
    if style["align"] == "left":
        align = {2: 1, 5: 4, 8: 7}[align]
    elif style["align"] == "right":
        align = {2: 3, 5: 6, 8: 9}[align]
    if karaoke:
        primary = _hex(style["text_color"])
        secondary = _hex("#ffffff") if _hex(style["text_color"]) != _hex("#ffffff") else _hex("#7fd0ff")
        bold = -1 if style["weight"] == "bold" else 0
    else:
        primary, secondary, bold = _hex(style["text_color"]), _hex("#ffffff"), \
            (-1 if style["weight"] == "bold" else 0)
    outline_col = _ass_color(style["outline_color"], 0.15)
    back = _ass_color(style["panel"]["color"], 1.0 - style["panel"]["opacity"]) \
        if style["panel"]["enabled"] else _ass_color("#000000", 1.0 - style["shadow_strength"])
    return (f"Style: Cap,{fam},{round(size_px,1)},{primary},{secondary},{outline_col},{back},"
            f"{bold},0,0,0,100,100,0,0,{border},{round(outline,1)},{round(shadow,1)},"
            f"{align},{round(ml)},{round(mr)},{round(mv)},1")


def _pack_karaoke_pixels(tags: list[str], shaper: "Shaper", budget_px: float) -> list[str]:
    """Pack `{\k…}word` tokens into lines within a shaped-pixel budget.

    Same rule as the sentence path (khmer.wrap_words) but operating on
    tag-glued tokens: width counts only the visible word plus a shaped space;
    a token never separates from its tag. Keeps the karaoke sweep inside the
    same max caption width as everything else (the old 64-cluster pack
    overflowed 480 px frames)."""
    lines: list[str] = []
    cur: list[str] = []
    width = 0.0
    space_w = shaper.width(" ")
    for tok in tags:
        word = tok.split("}", 1)[-1]
        w = shaper.width(word)
        sep = space_w if cur else 0.0
        if cur and width + sep + w > budget_px:
            lines.append(" ".join(cur))
            cur, width = [], 0.0
            sep = 0.0
        cur.append(tok)
        width += sep + w
    if cur:
        lines.append(" ".join(cur))
    return lines or [""]


def build_ass(blocks: list[tuple[float, float, str]], style: dict, out_w: int, out_h: int,
              dst: str, karaoke: bool | None = None) -> dict:
    """Write the caption .ass for `blocks` = [(start, end, text), …].

    Sentence path: per-line Dialogue events with explicit \\pos so
    line_spacing / margins / max width are honoured exactly. Karaoke path:
    one Dialogue per block, lines joined with \\N and {\\k} tags — natural
    leading (documented) and proportional word timing (estimated, not forced
    alignment).
    """
    from . import khmer as kh, media as media_mod

    karaoke = style.get("karaoke", False) if karaoke is None else karaoke
    plan = plan_blocks([b[2] for b in blocks], style, out_w, out_h)
    scale = plan["scale"]
    size_px = style["size_pct"] / 100.0 * float(out_h)
    line_h = size_px * float(style["line_spacing"])
    ml = style["margin_h_pct"] / 100.0 * out_w
    mr = ml
    mv = style["margin_v_pct"] / 100.0 * out_h
    panel_pad = scale_px(style, style["panel"]["padding_px"], out_h) if style["panel"]["enabled"] else 0.0

    header = [
        "[Script Info]", "ScriptType: v4.00+",
        f"PlayResX: {int(out_w)}", f"PlayResY: {int(out_h)}",
        "WrapStyle: 2", "ScaledBorderAndShadow: yes", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        style_ass_header(style, out_w, out_h, karaoke),
        "", "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    events: list[str] = []
    for bi, (start, end, text) in enumerate(blocks):
        disp = kh.display_text(text or "")
        if not disp:
            continue
        p = plan["plans"][bi]
        fsc = f"{{\\fscx{round(scale * 100)}\\fscy{round(scale * 100)}}}" if scale < 0.999 else ""
        if karaoke:
            words = media_mod.words_for_timing(disp)
            weights = [w for _wd, w in words] or [1.0]
            span = max(0.4, float(end) - float(start))
            cums, acc = [], 0.0
            for w in weights:
                acc += w
                cums.append(span * acc / sum(weights))
            tags, prev = [], 0.0
            for (word, _w), cum in zip(words, cums):
                k = max(1, int(round((cum - prev) * 100)))
                tags.append(f"{{\\k{k}}}{word}")
                prev = cum
            packed = _pack_karaoke_pixels(tags, plan["shaper"], p["budget_px"])
            body = "\\N".join(packed)
            events.append(f"Dialogue: 0,{_fmt_ass_time(start)},{_fmt_ass_time(end)},Cap,,0,0,0,,"
                          f"{fsc}{body}")
            continue
        # sentence path — one event per line, explicit layout
        n = len(p["lines"])
        anchor_x = {"left": ml, "center": out_w / 2.0, "right": out_w - mr}[style["align"]]
        if style["position"] == "bottom":
            block_bottom = out_h - mv - panel_pad
            top = block_bottom - n * line_h
            an = {"left": 4, "center": 5, "right": 6}[style["align"]]
        elif style["position"] == "top":
            top = mv + panel_pad
            an = {"left": 7, "center": 8, "right": 9}[style["align"]]
        else:
            top = (out_h - n * line_h) / 2.0
            an = {"left": 4, "center": 5, "right": 6}[style["align"]]
        for li, line in enumerate(p["lines"]):
            cy = top + (li + 0.5) * line_h
            events.append(f"Dialogue: 0,{_fmt_ass_time(start)},{_fmt_ass_time(end)},Cap,,0,0,0,,"
                          f"{fsc}{{\\an{an}\\pos({round(anchor_x,1)},{round(cy,1)})}}{line}")
    ensure_dir(os.path.dirname(dst) or ".")
    with open(dst, "w", encoding="utf-8") as f:
        f.write("\n".join(header + events) + "\n")
    return {"path": dst, "karaoke": karaoke, "scale": scale,
            "timing": "estimated-proportional" if karaoke else "sentence-window",
            "warnings": plan["warnings"], "font_file": font_file(style["font"], style["weight"]),
            "font_family": font_family_name(style["font"])}


# ---------------------------------------------------------------- preview
_PREVIEW_CACHE: "dict[str, str]" = {}
_PREVIEW_ORDER: "list[str]" = []
_PREVIEW_LOCK = threading.Lock()
_PREVIEW_LIMIT = 96


def render_preview(style: dict, text: str, out_w: int, out_h: int, bg: str = "dark",
                   cache_key: str | None = None, work_dir: str | None = None) -> str:
    """Render ONE frame through the exporter's real burn path (ASS + libass).

    Same builder, same filter, same font file as the final export — the
    preview cannot drift from the render. Results are cached by style hash.
    """
    style, _issues = validate_style(style)
    key = cache_key or hashlib.sha1(
        repr((sorted(style.items(), key=lambda kv: str(kv[0]))), ).encode()
        + text.encode() + f"{out_w}x{out_h}:{bg}".encode()).hexdigest()
    with _PREVIEW_LOCK:
        if key in _PREVIEW_CACHE and os.path.exists(_PREVIEW_CACHE[key]):
            return _PREVIEW_CACHE[key]
    work = ensure_dir(work_dir or os.path.join(os.path.dirname(FONTS_DIR), "..", "data", "caption_previews"))
    ass_path = os.path.join(work, f"pv_{key[:12]}.ass")
    png_path = os.path.join(work, f"pv_{key[:12]}.png")
    build_ass([(0.0, 6.0, text)], style, out_w, out_h, ass_path)
    W, H = int(out_w), int(out_h)
    bgf = {
        "light": f"gradients=s={W}x{H}:c0=#d8dee6:c1=#aeb6c0:x0=0:y0=0:x1=0:y1={H}",
        "dark": f"gradients=s={W}x{H}:c0=#14181f:c1=#2a3040:x0=0:y0=0:x1=0:y1={H}",
        "busy": (f"gradients=s={W}x{H}:c0=#7a3f2a:c1=#2d6f5a:x0={W // 4}:y0={H // 14}:"
                 f"x1={3 * W // 4}:y1={H - H // 14},noise=alls=18:allf=t"),
        "videoish": f"gradients=s={W}x{H}:c0=#33475f:c1=#0f1620:x0=0:y0=0:x1={W}:y1={H},noise=alls=8:allf=t",
    }.get(bg, f"color=c=0x1a2634:s={W}x{H}")
    ass_esc = ass_path.replace("\\", "/").replace(":", r"\:").replace("'", r"\'")
    run_ffmpeg(["-f", "lavfi", "-i", bgf, "-frames:v", "1", "-vf",
                f"ass='{ass_esc}':fontsdir='{FONTS_DIR.replace(chr(92), '/').replace(':', chr(92) + ':')}'",
                "-update", "1", "-y", png_path], timeout=120)
    if not os.path.exists(png_path) or os.path.getsize(png_path) < 512:
        raise CaptionStyleError("caption preview render failed — libass produced no frame "
                                "(check the studio log for the exact ffmpeg error)")
    with _PREVIEW_LOCK:
        _PREVIEW_CACHE[key] = png_path
        _PREVIEW_ORDER.append(key)
        if len(_PREVIEW_ORDER) > _PREVIEW_LIMIT:
            old = _PREVIEW_ORDER.pop(0)
            old_png = _PREVIEW_CACHE.pop(old, None)
            if old_png and os.path.exists(old_png):
                try:
                    os.remove(old_png)
                except OSError:
                    pass
    return png_path

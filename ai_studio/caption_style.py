"""Caption style schema — validated, persisted, and used by BOTH preview and export.

Why a module of its own: the studio used to carry exactly one hardcoded libass
``force_style`` string per style key (``media.SUBTITLE_STYLES``), with the font
name ``Khmer OS Battambang`` written into it. That font is not shipped with the
studio and not present on most machines, so libass silently substituted an
arbitrary font — which renders Khmer as empty boxes (reproduced: a caption frame
from the old default style is a row of tofu glyphs).

Everything a caption needs is one typed dict here:

* **font** — an id from :data:`FONT_FAMILIES` (bundled, licence-checked, OFL),
  plus a weight the family actually ships as a static instance.
* **size/colour/outline/shadow** — explicit numbers, not a string blob.
* **background panel** — toggle, colour, opacity, padding, corner radius.
* **layout** — position, safe margins, line spacing, max line count, max width.

The same dict is:

* validated server-side (:func:`normalize_style`) — the browser is not trusted;
* stored in settings (``caption_style``) and per project
  (``projects.settings_json.caption_style``);
* turned into an ASS ``[V4+ Styles]`` line by :mod:`ai_studio.captions`;
* returned to the UI by ``GET /api/caption-style``.

Precedence (documented in ``docs/KHMER-CAPTIONS.md``):

    built-in default  →  global ``cfg["caption_style"]``  →  project override

Older projects simply have no ``caption_style`` key: they load with the global
style and nothing is discarded, because nothing was ever saved for them.
"""
from __future__ import annotations

import copy
import os
import re

FONTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")

# --------------------------------------------------------------------- colours
_HEX_RE = re.compile(r"^#?([0-9a-fA-F]{6}|[0-9a-fA-F]{8})$")
_NAMED = {
    "white": "#FFFFFF", "black": "#000000", "yellow": "#FFD84D",
    "gold": "#E8C87A", "ivory": "#F4EBDD", "cream": "#F7F1E4",
    "red": "#FF5A5A", "green": "#5AD18B", "grey": "#9AA4B2", "gray": "#9AA4B2",
}


class CaptionStyleError(ValueError):
    """Raised for a style the renderer would have to guess about."""


def parse_color(value, field="color"):
    """'#rrggbb' / '#aarrggbb' / 'ivory' → '#rrggbb' (alpha is a separate field).

    ASS colours are ``&HAABBGGRR`` and the UI shows a hex picker, so we keep the
    canonical form as 6-digit ``#rrggbb`` and carry opacity as its own number —
    one representation, no ambiguity between "opacity 0.5" and "alpha 80".
    """
    if value is None or value == "":
        raise CaptionStyleError(f"{field}: a colour is required (e.g. #FFFFFF)")
    if isinstance(value, (int, float)):
        raise CaptionStyleError(f"{field}: use a hex string like #FFFFFF, not a number")
    s = str(value).strip()
    low = s.lower()
    if low in _NAMED:
        s = _NAMED[low]
    if not _HEX_RE.match(s):
        raise CaptionStyleError(f"{field}: '{value}' is not a hex colour (#RRGGBB)")
    h = s.lstrip("#")
    if len(h) == 8:                      # #aarrggbb → drop alpha, keep rgb
        h = h[2:]
    return "#" + h.upper()


def parse_opacity(value, field="opacity"):
    try:
        f = float(value)
    except (TypeError, ValueError):
        raise CaptionStyleError(f"{field}: must be a number between 0 and 1")
    if not 0.0 <= f <= 1.0:
        raise CaptionStyleError(f"{field}: must be between 0 and 1 (got {value})")
    return round(f, 3)


def _num(value, lo, hi, field, default=None, integer=False):
    if value is None:
        if default is None:
            raise CaptionStyleError(f"{field}: a value is required")
        return default
    try:
        f = float(value)
    except (TypeError, ValueError):
        raise CaptionStyleError(f"{field}: must be a number (got {value!r})")
    if f != f or f in (float("inf"), float("-inf")):
        raise CaptionStyleError(f"{field}: must be a finite number")
    if not lo <= f <= hi:
        raise CaptionStyleError(f"{field}: must be between {lo} and {hi} (got {value})")
    return int(round(f)) if integer else round(f, 3)


def _enum(value, allowed, field, default=None):
    if value is None:
        if default is None:
            raise CaptionStyleError(f"{field}: a value is required")
        return default
    s = str(value).strip().lower().replace("_", "-")
    aliases = {"centre": "center", "middle": "center", "bold-social": "bold-social",
               "soft card": "soft-card"}
    s = aliases.get(s, s)
    if s not in allowed:
        raise CaptionStyleError(f"{field}: '{value}' is not one of {', '.join(sorted(allowed))}")
    return s


# ---------------------------------------------------------------------- fonts
# Bundled families. Every file here ships in ai_studio/fonts/ with its OFL
# licence (see fonts/README.md) and is loaded into libass through an explicit
# `fontsdir` — never through a system-wide font lookup, which is exactly what
# produced the "tofu" report. Weights listed are the weights that exist as a
# *static* file: a variable font put on disk unnamed is not selectable by
# weight, so the studio ships pre-instanced statics and offers only those.
FONT_FAMILIES = {
    "noto-sans-khmer": {
        "id": "noto-sans-khmer",
        "family": "Noto Sans Khmer",
        "label": "Noto Sans Khmer · ខ្មែរ",
        "blurb": "Clean modern sans — safest default for subtitles.",
        "files": {"400": "NotoSansKhmer/NotoSansKhmer-Regular.ttf",
                  "700": "NotoSansKhmer/NotoSansKhmer-Bold.ttf"},
        "weights": [400, 700],
        "license": "OFL-1.1",
    },
    "noto-serif-khmer": {
        "id": "noto-serif-khmer",
        "family": "Noto Serif Khmer",
        "label": "Noto Serif Khmer · ខ្មែរ",
        "blurb": "Editorial serif with generous marks — quotes, essays, formal films.",
        "files": {"400": "NotoSerifKhmer/NotoSerifKhmer-Regular.ttf",
                  "700": "NotoSerifKhmer/NotoSerifKhmer-Bold.ttf"},
        "weights": [400, 700],
        "license": "OFL-1.1",
    },
    "kantumruy-pro": {
        "id": "kantumruy-pro",
        "family": "Kantumruy Pro",
        "label": "Kantumruy Pro · ខ្មែរ",
        "blurb": "Contemporary Khmer UI/display face — strong at large sizes.",
        "files": {"400": "KantumruyPro/KantumruyPro-Regular.ttf",
                  "700": "KantumruyPro/KantumruyPro-Bold.ttf"},
        "weights": [400, 700],
        "license": "OFL-1.1",
    },
    "battambang": {
        "id": "battambang",
        "family": "Battambang",
        "label": "Battambang · ខ្មែរ",
        "blurb": "Familiar Khmer body text (Khmer OS Battambang's free sibling). "
                 "No em-dash glyph — the fallback chain covers it.",
        "files": {"300": "Battambang/Battambang-Light.ttf",
                  "400": "Battambang/Battambang-Regular.ttf",
                  "700": "Battambang/Battambang-Bold.ttf"},
        "weights": [300, 400, 700],
        "license": "OFL-1.1",
    },
}

DEFAULT_FONT = "noto-sans-khmer"


def font_list():
    """UI payload for the font picker (with a live Khmer sample per family)."""
    out = []
    for key, f in FONT_FAMILIES.items():
        out.append({"id": key, "family": f["family"], "label": f["label"],
                    "blurb": f["blurb"], "weights": list(f["weights"]),
                    "license": f["license"], "bundled": True,
                    "files": [os.path.join(FONTS_DIR, rel) for rel in f["files"].values()],
                    "sample": "ជីវិតមនុស្ស មិនមែនជាការប្រណាំងទេ"})
    return out


def font_files(font_id):
    """Absolute paths of every static file of a bundled family (regular first)."""
    fam = FONT_FAMILIES.get(font_id)
    if not fam:
        raise CaptionStyleError(f"font: '{font_id}' is not a bundled font family")
    paths = []
    for w in sorted(fam["files"], key=lambda k: abs(int(k) - 400)):
        p = os.path.join(FONTS_DIR, fam["files"][w])
        if not os.path.exists(p):
            raise CaptionStyleError(
                f"font: bundled file missing for '{font_id}' ({fam['files'][w]}) — "
                f"reinstall the studio or restore ai_studio/fonts/")
        paths.append(p)
    return paths


def safe_font_path(font_id, filename):
    """Resolve a bundled font filename for HTTP serving — no path traversal.

    Only the exact files listed in :data:`FONT_FAMILIES` can be returned, and the
    resolved path must stay inside ``ai_studio/fonts``; every other request is
    refused. Nothing here reads a user-supplied path or a system font.
    """
    fam = FONT_FAMILIES.get(font_id)
    if not fam:
        raise CaptionStyleError(f"font: '{font_id}' is not a bundled font family")
    wanted = os.path.basename(str(filename or ""))
    for rel in fam["files"].values():
        if os.path.basename(rel) == wanted:
            full = os.path.abspath(os.path.join(FONTS_DIR, rel))
            root = os.path.abspath(FONTS_DIR) + os.sep
            if not full.startswith(root) or not os.path.isfile(full):
                raise CaptionStyleError("font file is not available")
            return full
    raise CaptionStyleError(f"font: '{wanted}' is not part of {fam['family']}")


def font_file(font_id, weight=None):
    """The single static file that matches `weight` (nearest shipped weight)."""
    fam = FONT_FAMILIES.get(font_id)
    if not fam:
        raise CaptionStyleError(f"font: '{font_id}' is not a bundled font family")
    weights = sorted(int(w) for w in fam["files"])
    want = int(weight) if weight else 400
    best = min(weights, key=lambda w: (abs(w - want), w))
    path = os.path.join(FONTS_DIR, fam["files"][str(best)])
    if not os.path.exists(path):
        raise CaptionStyleError(f"font: bundled file missing for '{font_id}'")
    return path, best


# ------------------------------------------------------------------- defaults
DEFAULT_STYLE = {
    "version": 1,
    "preset": "clean",
    "font": DEFAULT_FONT,
    "weight": 400,
    "font_size_px": 72,          # px on the 1080x1920 portrait master
    "color": "#FFFFFF",
    "outline_color": "#000000",
    "outline_width": 3,          # px at reference height (ASS Outline)
    "shadow": 0.35,              # 0 = off; strength in px at reference height
    "shadow_color": "#000000",
    "shadow_opacity": 0.65,
    "background": False,
    "background_color": "#0B0F14",
    "background_opacity": 0.55,
    "background_padding": 26,
    "background_radius": 18,
    "position": "bottom",        # bottom | center | top
    "alignment": "center",       # center | left | right  (horizontal)
    "margin_h": 8.0,             # % of frame width, kept clear on both sides
    "margin_v": 9.0,             # % of frame height, from the chosen edge
    "line_spacing": 1.25,
    "max_lines": 3,
    "max_width_pct": 88.0,       # % of frame width the text box may occupy
    "letter_spacing": 0.0,
}

# A preset IS the style dict, so "apply preset" and "export" can never disagree.
PRESETS = {
    "clean": {
        "label": "Clean",
        "desc": "White Noto Sans Khmer, restrained dark contrast, no clutter.",
        "style": {"font": "noto-sans-khmer", "weight": 400, "color": "#FFFFFF",
                  "outline_color": "#000000", "outline_width": 3, "shadow": 0.35,
                  "shadow_opacity": 0.6, "background": False,
                  "position": "bottom", "margin_h": 8.0, "margin_v": 9.0,
                  "line_spacing": 1.25, "max_lines": 3, "font_size_px": 72},
    },
    "cinema": {
        "label": "Cinema",
        "desc": "Warm ivory serif, soft shadow, no hard outline — quiet and filmic.",
        "style": {"font": "noto-serif-khmer", "weight": 400, "color": "#F4EBDD",
                  "outline_color": "#100C08", "outline_width": 2, "shadow": 2.2,
                  "shadow_opacity": 0.55, "background": False,
                  "position": "bottom", "margin_h": 10.0, "margin_v": 11.0,
                  "line_spacing": 1.4, "max_lines": 2, "font_size_px": 68},
    },
    "bold-social": {
        "label": "Bold Social",
        "desc": "High-contrast yellow, sized outline — readable over busy b-roll.",
        "style": {"font": "kantumruy-pro", "weight": 700, "color": "#FFD84D",
                  "outline_color": "#140F00", "outline_width": 5, "shadow": 1.2,
                  "shadow_opacity": 0.7, "background": False,
                  "position": "center", "margin_h": 7.0, "margin_v": 12.0,
                  "line_spacing": 1.2, "max_lines": 3, "font_size_px": 88},
    },
    "soft-card": {
        "label": "Soft Card",
        "desc": "White text on a dark translucent rounded panel — mobile-safe.",
        "style": {"font": "noto-sans-khmer", "weight": 400, "color": "#FFFFFF",
                  "outline_color": "#000000", "outline_width": 1, "shadow": 0.0,
                  "shadow_opacity": 0.0, "background": True, "background_color": "#0B0F14",
                  "background_opacity": 0.6, "background_padding": 30,
                  "background_radius": 22, "position": "bottom",
                  "margin_h": 7.0, "margin_v": 8.0, "line_spacing": 1.3,
                  "max_lines": 3, "font_size_px": 68},
    },
    "editorial": {
        "label": "Editorial",
        "desc": "Serif Khmer, calm warm colours, generous spacing — quotes and essays.",
        "style": {"font": "noto-serif-khmer", "weight": 400, "color": "#EFE7DA",
                  "outline_color": "#000000", "outline_width": 1, "shadow": 1.0,
                  "shadow_opacity": 0.45, "background": False, "position": "bottom",
                  "margin_h": 12.0, "margin_v": 12.0, "line_spacing": 1.5,
                  "max_lines": 3, "font_size_px": 66, "letter_spacing": 0.0},
    },
}
PRESET_KEYS = tuple(PRESETS)


def _preset_spec(preset):
    """Raw (un-normalized) spec of a preset: its style keys over the defaults."""
    return {**DEFAULT_STYLE, **PRESETS[preset]["style"]}


def _preset_match(style, preset):
    """Does a *normalized* style equal the preset it claims? (no recursion)"""
    if preset not in PRESETS:
        return False
    ref, cur = _preset_spec(preset), (style or {})
    for k, want in ref.items():
        if k in ("preset", "version"):
            continue
        got = cur.get(k)
        if isinstance(want, bool) or isinstance(got, bool):
            if bool(got) != bool(want):
                return False
        elif isinstance(want, (int, float)) and isinstance(got, (int, float)):
            if abs(float(got) - float(want)) > 1e-6:
                return False
        elif k.endswith("color"):
            if str(got or "").upper() != str(want).upper():
                return False
        elif got != want:
            return False
    return True


def preset_style(key):
    """Full style dict for a preset key (validated), or raise."""
    key = _enum(key, set(PRESETS), "preset", default="clean")
    return normalize_style({**PRESETS[key]["style"], "preset": key})


def preset_payload():
    return [{"key": k, "label": PRESETS[k]["label"], "desc": PRESETS[k]["desc"],
             "style": preset_style(k)} for k in PRESET_KEYS]


# ------------------------------------------------------------- style handling
def normalize_style(raw, base=None, strict=True):
    """Validate + fill a partial style dict. Extra keys are dropped, never kept.

    ``strict=True`` (default) raises :class:`CaptionStyleError` on a bad value so
    the API can answer 400 with a specific message. ``strict=False`` repairs what
    it can and is only used when *loading old settings*, where the studio must
    never refuse to start because of a stale value.
    """
    src = dict(base or DEFAULT_STYLE)
    for k, v in (raw or {}).items():
        if k in DEFAULT_STYLE:
            src[k] = v

    out = {}
    errs = []

    def ok(fn, *a, **kw):
        try:
            return fn(*a, **kw), None
        except CaptionStyleError as e:
            return None, str(e)

    font = str(src.get("font") or DEFAULT_FONT)
    if font not in FONT_FAMILIES:
        errs.append(f"font: '{font}' is not a bundled font family "
                    f"({', '.join(FONT_FAMILIES)})")
        font = DEFAULT_FONT
    out["font"] = font

    weights = FONT_FAMILIES[font]["weights"]
    try:
        w = int(float(src.get("weight", 400)))
    except (TypeError, ValueError):
        errs.append(f"weight: '{src.get('weight')}' is not a number")
        w = 400
    if w not in weights:
        errs.append(f"weight: {w} is not shipped for {FONT_FAMILIES[font]['family']} "
                    f"(available: {', '.join(str(x) for x in weights)})")
        w = min(weights, key=lambda x: abs(x - w))
    out["weight"] = w

    for field, key in (("font_size_px", "font_size_px"),):
        v, e = ok(_num, src.get(key), 14, 220, field)
        if e:
            errs.append(e)
            v = DEFAULT_STYLE[key]
        out[key] = v

    # reference-height scaling: a style written for a portrait 1080p master is
    # rescaled by ai_studio.captions for any other canvas, so the number the UI
    # shows is always "px on a 1080-tall frame".
    v, e = ok(parse_color, src.get("color"), "color")
    errs.append(e) if e else None
    out["color"] = v or DEFAULT_STYLE["color"]
    v, e = ok(parse_color, src.get("outline_color"), "outline_color")
    errs.append(e) if e else None
    out["outline_color"] = v or DEFAULT_STYLE["outline_color"]
    v, e = ok(_num, src.get("outline_width"), 0, 10, "outline_width")
    errs.append(e) if e else None
    out["outline_width"] = v if v is not None else DEFAULT_STYLE["outline_width"]

    v, e = ok(_num, src.get("shadow"), 0, 12, "shadow")
    errs.append(e) if e else None
    out["shadow"] = v if v is not None else DEFAULT_STYLE["shadow"]
    v, e = ok(parse_color, src.get("shadow_color"), "shadow_color")
    errs.append(e) if e else None
    out["shadow_color"] = v or DEFAULT_STYLE["shadow_color"]
    v, e = ok(parse_opacity, src.get("shadow_opacity"), "shadow_opacity")
    errs.append(e) if e else None
    out["shadow_opacity"] = v if v is not None else DEFAULT_STYLE["shadow_opacity"]

    bg = src.get("background", False)
    out["background"] = bool(bg) if isinstance(bg, (bool, int, str)) else False
    v, e = ok(parse_color, src.get("background_color"), "background_color")
    errs.append(e) if e else None
    out["background_color"] = v or DEFAULT_STYLE["background_color"]
    v, e = ok(parse_opacity, src.get("background_opacity"), "background_opacity")
    errs.append(e) if e else None
    out["background_opacity"] = v if v is not None else DEFAULT_STYLE["background_opacity"]
    v, e = ok(_num, src.get("background_padding"), 0, 120, "background_padding")
    errs.append(e) if e else None
    out["background_padding"] = v if v is not None else DEFAULT_STYLE["background_padding"]
    v, e = ok(_num, src.get("background_radius"), 0, 120, "background_radius")
    errs.append(e) if e else None
    out["background_radius"] = v if v is not None else DEFAULT_STYLE["background_radius"]

    v, e = ok(_enum, src.get("position"), {"bottom", "center", "top"}, "position")
    errs.append(e) if e else None
    out["position"] = v or DEFAULT_STYLE["position"]
    v, e = ok(_enum, src.get("alignment"), {"center", "left", "right"}, "alignment")
    errs.append(e) if e else None
    out["alignment"] = v or DEFAULT_STYLE["alignment"]

    v, e = ok(_num, src.get("margin_h"), 0, 30, "margin_h")
    errs.append(e) if e else None
    out["margin_h"] = v if v is not None else DEFAULT_STYLE["margin_h"]
    v, e = ok(_num, src.get("margin_v"), 0, 40, "margin_v")
    errs.append(e) if e else None
    out["margin_v"] = v if v is not None else DEFAULT_STYLE["margin_v"]
    v, e = ok(_num, src.get("line_spacing"), 0.8, 2.4, "line_spacing")
    errs.append(e) if e else None
    out["line_spacing"] = v if v is not None else DEFAULT_STYLE["line_spacing"]
    v, e = ok(_num, src.get("max_lines"), 1, 5, "max_lines", integer=True)
    errs.append(e) if e else None
    out["max_lines"] = v if v is not None else DEFAULT_STYLE["max_lines"]
    v, e = ok(_num, src.get("max_width_pct"), 40, 100, "max_width_pct")
    errs.append(e) if e else None
    out["max_width_pct"] = v if v is not None else DEFAULT_STYLE["max_width_pct"]
    v, e = ok(_num, src.get("letter_spacing"), 0, 6, "letter_spacing")
    errs.append(e) if e else None
    out["letter_spacing"] = v if v is not None else DEFAULT_STYLE["letter_spacing"]

    preset = src.get("preset") or "custom"
    out["preset"] = preset if preset in PRESETS else "custom"
    out["version"] = 1
    # never claim to be a preset it no longer matches: the UI highlights a preset
    # cell from this field, and a style that *says* Bold Social while rendering
    # Clean is exactly the silent disagreement this module exists to prevent
    if out["preset"] != "custom" and not _preset_match(out, out["preset"]):
        out["preset"] = "custom"

    if errs and strict:
        raise CaptionStyleError("; ".join(errs))
    return out


def normalize_patch(patch, base=None, strict=True):
    """Validate ONLY the keys present in ``patch``, on top of ``base``.

    A request-level override is a patch, not a full style: sending ``{color}``
    must change the colour and nothing else, and a *bad* ``{color}`` must still
    be a 400 instead of a silently repaired default. Returns the canonical
    subset, so ``merge_styles`` can layer it without resetting the rest.
    """
    base_n = normalize_style(base, strict=False)
    keys = [k for k in (patch or {}) if k in DEFAULT_STYLE or k == "preset"]
    full = normalize_style({**base_n, **{k: patch[k] for k in keys}},
                           base=base_n, strict=strict)
    return {k: full[k] for k in keys}


def style_errors(raw):
    """List (never raise) of problems — used when loading settings/config."""
    try:
        normalize_style(raw)
        return []
    except CaptionStyleError as e:
        return [str(e)]


def is_preset_default(style):
    """True when `style` currently equals the preset it claims (UI shows a dot)."""
    preset = (style or {}).get("preset")
    return _preset_match(normalize_style(style, strict=False), preset)


def diff_from_preset(style):
    """Human-readable list of fields changed away from the stated preset."""
    preset = (style or {}).get("preset")
    if preset not in PRESETS:
        return []
    base = preset_style(preset)
    cur = normalize_style(style)
    return [k for k in sorted(cur) if k != "preset" and cur[k] != base.get(k)]


def merge_styles(global_style, project_style=None, override=None):
    """Global default → project override → (optional) explicit patch."""
    merged = copy.deepcopy(normalize_style(global_style, strict=False))
    for layer in (project_style, override):
        if not layer or not isinstance(layer, dict):
            continue
        merged = normalize_style({**merged, **normalize_patch(layer, merged, strict=False)},
                                 base=merged, strict=False)
    return merged

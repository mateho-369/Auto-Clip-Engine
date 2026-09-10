"""Burned-in caption rendering: layout, wrapping, ASS generation, burn-in, preview.

One renderer for preview **and** export. The old code path built a libass
``force_style`` string per style key, so what the Style Gallery showed, what the
SRT produced and what the final MP4 contained were three different things. Here
there is exactly one function, :func:`build_ass`, and every consumer (the burn,
the still-frame preview, the style gallery) calls it.

How text gets on screen:

1. ``khmer.split_clusters`` gives Khmer character clusters — the only safe
   positions to break a line (a coeng + subscript pair is one cluster).
2. :class:`TextMeasurer` measures candidate lines with HarfBuzz, so wrapping is
   decided by *shaped advance widths*, not by counting characters. Without
   uharfbuzz the code degrades to an estimate and says so in the render
   metadata (``measurement: "estimate"``) instead of pretending.
3. Lines are laid out on the exact pixel canvas of the video (``PlayResX/Y`` =
   frame size), with the style's safe margins, then written as ASS.
4. ffmpeg/libass renders that ASS with ``fontsdir`` pointing at the bundled
   font family. libass then shapes Khmer with HarfBuzz — correct coeng
   stacking, no dropped marks.

If anything makes honest rendering impossible (ffmpeg without libass, a missing
bundled font, a fit that cannot be achieved) the code raises
:class:`CaptionError` with an actionable message. It never silently returns an
uncaptioned file, and it never silently truncates the Director's words.
"""
from __future__ import annotations

import math
import os
import re
import shutil
import struct
import subprocess
import threading

from . import caption_style as cs
from . import khmer
from .util import ensure_dir, ffmpeg_exe, run_ffmpeg

# Font sizes are authored against the studio's portrait master (1080x1920) and
# scaled by frame height for every other canvas (see `scale_style`). The UI
# shows the user's own canvas size next to the slider, so the number is never
# mysterious: 72px on the master is 72px on a 1080x1920 export and 40px on a
# 1920x1080 one.
REFERENCE_HEIGHT = 1920.0
REFERENCE_WIDTH = 1080.0

_HAS_FILTER = {}
_FONT_DIRS = {}
_FONT_DIRS_LOCK = threading.Lock()


class CaptionError(RuntimeError):
    """Rendering cannot be done honestly — the caller must surface this."""


# ------------------------------------------------------------------ ffmpeg
def has_filter(name):
    if name in _HAS_FILTER:
        return _HAS_FILTER[name]
    ok = False
    ff = ffmpeg_exe()
    if ff:
        try:
            res = subprocess.run([ff, "-hide_banner", "-filters"], capture_output=True, timeout=60)
            text = (res.stdout or b"").decode("utf-8", "ignore")
            ok = re.search(r"^\s*[TSC.]+\s+%s\s" % re.escape(name), text, re.M) is not None
        except Exception:
            ok = False
    _HAS_FILTER[name] = ok
    return ok


def require_libass():
    if not has_filter("subtitles"):
        raise CaptionError(
            "this ffmpeg build has no 'subtitles' filter (libass missing) — install an "
            "ffmpeg build with libass (e.g. the gyan.dev 'essentials' Windows build) or "
            "export SRT + the .ass file and burn captions elsewhere")

    return True


# --------------------------------------------------------- bundled font dirs
def font_dir_for_style(style, cache_root=None):
    """Directory holding the static files of the selected family.

    libass resolves a font by *family name* from the directories it is given;
    pointing it at our own folder is what guarantees the caption font is the one
    the UI selected, on Windows, macOS and Linux alike (and it is why the old
    'Khmer OS Battambang' name rendered as empty boxes: the name resolved to
    nothing).
    """
    fam = cs.FONT_FAMILIES.get(style.get("font"))
    if not fam:
        raise CaptionError(f"unknown caption font '{style.get('font')}'")
    files = cs.font_files(fam["id"])
    key = fam["id"]
    if cache_root is None:
        with _FONT_DIRS_LOCK:
            cached = _FONT_DIRS.get(key)
            if cached and os.path.isdir(cached):
                return cached
    root = ensure_dir(os.path.join(cache_root or _default_font_cache(), key))
    for src in files:
        dst = os.path.join(root, os.path.basename(src))
        if not os.path.exists(dst) or os.path.getmtime(dst) < os.path.getmtime(src):
            shutil.copyfile(src, dst)
    if cache_root is None:
        with _FONT_DIRS_LOCK:
            _FONT_DIRS[key] = root
    return root


def _default_font_cache():
    try:
        from . import config as cfg_mod
        return os.path.join(cfg_mod.data_root(), "caption-fonts")
    except Exception:
        import tempfile
        return os.path.join(tempfile.gettempdir(), "studio-caption-fonts")


def font_report(style):
    """What font actually got used + whether the file really is on disk."""
    fam = cs.FONT_FAMILIES.get(style.get("font"))
    path, weight = cs.font_file(style.get("font"), style.get("weight"))
    return {"family": fam["family"] if fam else "", "font_id": style.get("font"),
            "weight": weight, "requested_weight": style.get("weight"),
            "file": os.path.basename(path), "license": fam["license"] if fam else "",
            "size_bytes": os.path.getsize(path) if os.path.exists(path) else 0}


# ------------------------------------------------------------- font metrics
_METRIC_CACHE = {}


def font_metrics(font_path):
    """(units_per_em, ascent, descent, ink_above, ink_below) from the raw tables.

    Read straight from the sfnt tables (no fontTools dependency): ``hhea`` for
    the typographic extents, ``OS/2`` for the Windows extents and ``head`` for
    the true bbox. We take the *maximum* of these so a caption with marks far
    above the ascender (Khmer sign muusikatoan, ៉) or below the baseline (coeng
    subscripts, ្ត) is still measured with room for its ink — the old layout
    measured nothing and let marks collide with the screen edge.
    """
    if font_path in _METRIC_CACHE:
        return _METRIC_CACHE[font_path]
    with open(font_path, "rb") as f:
        data = f.read()
    if data[:4] == b"ttcf":
        data = data[12:]                       # first face of a collection
    num_tables = struct.unpack(">H", data[4:6])[0]
    tables = {}
    for i in range(num_tables):
        off = 12 + 16 * i
        tag = data[off:off + 4].decode("latin-1")
        toff, _tlen = struct.unpack(">II", data[off + 8:off + 16])
        tables[tag] = toff

    def _u16(tag, rel):
        return struct.unpack(">H", data[tables[tag] + rel:tables[tag] + rel + 2])[0]

    def _i16(tag, rel):
        return struct.unpack(">h", data[tables[tag] + rel:tables[tag] + rel + 2])[0]

    upem = _u16("head", 18) or 1000
    y_min, y_max = _i16("head", 38), _i16("head", 42)
    asc = _i16("hhea", 4)
    desc = _i16("hhea", 6)
    if "OS/2" in tables and len(data) >= tables["OS/2"] + 74:
        try:
            win_a, win_d = _u16("OS/2", 74), _u16("OS/2", 76)
            asc = max(asc, win_a)
            desc = min(desc, -win_d)
        except Exception:
            pass
    above = max(asc, y_max, 0)
    below = -min(desc, y_min, 0)
    out = (float(upem), float(asc), float(desc), float(above), float(below))
    _METRIC_CACHE[font_path] = out
    return out


# --------------------------------------------------------------- measuring
class TextMeasurer:
    """Shaped pixel widths for one font + size.

    HarfBuzz when available (the same shaper libass uses → preview and export
    agree), otherwise a documented estimate that is only used to pick break
    points; libass still does the real shaping at burn time.
    """

    def __init__(self, font_path, size_px, letter_spacing=0.0):
        self.font_path = font_path
        self.size_px = float(size_px)
        self.letter_spacing = float(letter_spacing or 0.0)
        self.upem, self.asc, self.desc, self.ink_above, self.ink_below = font_metrics(font_path)
        self.engine = "estimate"
        self._hb_font = None
        self._hb = None
        try:
            import uharfbuzz as hb
            blob = hb.Blob.from_file_path(font_path)
            face = hb.Face(blob)
            font = hb.Font(face)
            font.scale = (int(self.upem), int(self.upem))
            self._hb, self._hb_font = hb, font
            self.engine = "harfbuzz"
        except Exception:
            self._hb = None

    # -- helpers
    def _shape(self, text):
        hb, font = self._hb, self._hb_font
        buf = hb.Buffer()
        buf.add_str(text)
        buf.guess_segment_properties()
        hb.shape(font, buf)
        return buf.glyph_positions

    def width(self, text):
        """Advance width of `text` in pixels (shaped, including letter spacing)."""
        if not text:
            return 0.0
        s = self.size_px / self.upem
        if self._hb is not None:
            try:
                total = sum(p.x_advance for p in self._shape(text)) * s
                return total + self.letter_spacing * max(0, len(text) - 1)
            except Exception:
                pass
        # Estimate: Khmer clusters are wide (subscripts + vowels), Latin runs
        # narrower. Only used to choose line breaks when HarfBuzz is absent.
        km = sum(1 for c in text if 0x1780 <= ord(c) <= 0x17FF)
        lat = len(text) - km
        clusters = len(khmer.split_clusters(text)) if km else len(text)
        return (clusters * 0.72 + lat * 0.08) * self.size_px

    def line_height(self, spacing):
        """One line box in pixels: shaped ascent + descent, times the spacing."""
        return (self.ink_above + self.ink_below) / self.upem * self.size_px * float(spacing)

    def block_height(self, lines, spacing):
        return self.line_height(spacing) * max(1, len(lines))


def measurer_for(style, size_px):
    path, weight = cs.font_file(style.get("font"), style.get("weight"))
    return TextMeasurer(path, size_px, style.get("letter_spacing", 0.0))


# ------------------------------------------------------------------ wrapping
_BREAK_AFTER = set(" \u200b")                       # spaces + zero-width space
_SENTENCE_ENDS = set("។៕៖!?.,;:—–-")
_NO_LINE_START = set("។៕៖!?,;:)]}»…")               # never orphan these


def _break_units(text):
    """Cluster list split into space-delimited words, each word's clusters."""
    clusters = khmer.split_clusters(text)
    words, cur = [], []
    for cl in clusters:
        cur.append(cl)
        if cl in _BREAK_AFTER:
            words.append(cur)
            cur = []
    if cur:
        words.append(cur)
    return words


def wrap_text(text, measurer, usable_width, max_lines=None, hard_split_words=True):
    """Wrap `text` to `usable_width` px, never inside a Khmer cluster.

    Breaks are preferred at spaces (Khmer's phrase boundaries), then after
    punctuation. A single word that is wider than the line is split at cluster
    boundaries — the smallest unit the shaper can still render correctly — and
    that is reported to the caller so the UI can show a warning instead of
    silently producing an unreadable line.

    Returns ``(lines, meta)`` where meta has ``split_words`` and ``max_width``.
    """
    text = (text or "").replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    lines: list[str] = []
    meta = {"split_words": [], "max_width": 0.0}

    for raw_line in text.split("\n"):
        para = raw_line.strip()
        if not para:
            continue
        words = _break_units(para)
        cur = ""
        for wi, wcl in enumerate(words):
            word = "".join(wcl)
            if word.strip() == "" and not cur:
                continue                                  # collapse leading space
            candidate = (cur + word) if cur else word.lstrip()
            if measurer.width(candidate.rstrip()) <= usable_width or not cur:
                if measurer.width(candidate.rstrip()) > usable_width and not cur:
                    # a single word wider than the line: split it on clusters
                    for piece in _split_word(word, measurer, usable_width, meta):
                        if measurer.width((cur + piece).rstrip()) <= usable_width or not cur:
                            cur = (cur + piece)
                        else:
                            lines.append(cur.rstrip())
                            cur = piece
                else:
                    cur = candidate
                continue
            lines.append(cur.rstrip())
            cur = word.lstrip()
        if cur.strip():
            lines.append(cur.rstrip())
        # trailing window: keep the last word of a paragraph from being alone
        if len(lines) >= 2 and len(lines[-1].strip()) <= 2 and len(lines[-2]) > 8:
            lines[-2], lines[-1] = (lines[-2] + " " + lines[-1].strip()).strip(), ""
            lines = [l for l in lines if l]

    lines = [l for l in lines if l.strip()]
    meta["max_width"] = max((measurer.width(l) for l in lines), default=0.0)
    return lines, meta


def _split_word(word, measurer, usable_width, meta, hard=True):
    """Split one over-long word at cluster boundaries (coeng-safe)."""
    clusters = khmer.split_clusters(word)
    pieces, cur = [], ""
    for cl in clusters:
        if cur and measurer.width(cur + cl) > usable_width:
            pieces.append(cur)
            cur = cl
        else:
            cur += cl
    if cur:
        pieces.append(cur)
    if len(pieces) > 1:
        meta["split_words"].append(word)
    return pieces or [word]


def fit_text(text, style, frame_w, frame_h, min_scale=0.62):
    """Wrap + size-fit one caption without ever dropping characters.

    Returns ``{lines, font_size_px, scale, warnings[]}``. The style's
    ``font_size_px`` is interpreted on a 1080-tall master and scaled to the
    actual frame, then reduced in 4% steps (never below ``min_scale``) until the
    text fits ``max_lines`` inside the safe box. If it still does not fit, extra
    lines are allowed and a warning is returned — the words are never cut.
    """
    warnings: list[str] = []
    base_size = scale_number(style.get("font_size_px", cs.DEFAULT_STYLE["font_size_px"]), frame_h)
    usable = min(frame_w * float(style.get("max_width_pct", 88)) / 100.0,
                 frame_w * (1.0 - 2.0 * float(style.get("margin_h", 8)) / 100.0))
    usable = max(40.0, usable)
    max_lines = int(style.get("max_lines", 3))
    spacing = float(style.get("line_spacing", 1.25))

    size = base_size
    result = None
    for _step in range(16):
        meas = measurer_for(style, size)
        lines, meta = wrap_text(text, meas, usable)
        lh = meas.line_height(spacing)
        block_h = lh * max(1, len(lines))
        avail_h = frame_h * (1.0 - 2.0 * float(style.get("margin_v", 9)) / 100.0)
        fits = len(lines) <= max_lines and block_h <= avail_h
        result = {"lines": lines, "meta": meta, "measurer": meas, "line_height": lh,
                  "block_h": block_h, "font_size_px": size, "usable_w": usable,
                  "available_h": avail_h, "fits": fits,
                  "ink_above_px": meas.ink_above / meas.upem * size,
                  "ink_below_px": meas.ink_below / meas.upem * size}
        if fits:
            break
        if size <= base_size * min_scale:
            break
        size = max(base_size * min_scale, size * 0.96)

    r = result
    if r["meta"]["split_words"]:
        warnings.append("long unbreakable run split at cluster boundaries: "
                        + ", ".join(r["meta"]["split_words"][:3]))
    if not r["fits"]:
        if len(r["lines"]) > max_lines:
            warnings.append(
                f"text needs {len(r['lines'])} lines at {r['font_size_px']:.0f}px "
                f"(max_lines={max_lines}) — increase Max lines or shorten the line")
        if r["block_h"] > r["available_h"]:
            warnings.append(
                f"caption block is {r['block_h']:.0f}px tall in a {frame_h}px frame with "
                f"margin_v={style.get('margin_v')}% — reduce the font size, max lines or margin")
    if r["font_size_px"] < base_size * 0.999:
        warnings.append(f"font auto-reduced {base_size:.0f}px → {r['font_size_px']:.0f}px to fit")
    if r["measurer"].engine != "harfbuzz":
        warnings.append("shaping measurement unavailable (install uharfbuzz) — line breaks "
                        "were chosen from an estimate; the burn itself is still shaped by libass")
    if r["font_size_px"] > 0.16 * frame_h:
        warnings.append(f"font size {r['font_size_px']:.0f}px is over 16% of the "
                        f"{frame_h}px frame height — it will dominate the picture")
    elif r["font_size_px"] < 0.014 * frame_h:
        warnings.append(f"font size {r['font_size_px']:.0f}px is under 1.4% of the "
                        f"{frame_h}px frame height — hard to read on a phone")
    return {"lines": r["lines"], "font_size_px": r["font_size_px"],
            "scale": r["font_size_px"] / base_size if base_size else 1.0,
            "line_height": r["line_height"], "block_h": r["block_h"],
            "ink_above_px": r["ink_above_px"], "ink_below_px": r["ink_below_px"],
            "max_line_width": r["meta"]["max_width"], "usable_w": r["usable_w"],
            "available_h": r["available_h"], "warnings": warnings,
            "measurement": r["measurer"].engine}


def scale_number(value, frame_h):
    """A 1080-master number → this frame's pixels (never below 1)."""
    return max(1.0, float(value) * float(frame_h) / REFERENCE_HEIGHT)


def scale_style(style, frame_h):
    """The numeric parts of a style, scaled to the frame. Layout % values are not."""
    s = dict(style)
    s["font_size_px"] = scale_number(style.get("font_size_px", 64), frame_h)
    s["outline_width"] = scale_number(style.get("outline_width", 3), frame_h)
    s["shadow"] = scale_number(style.get("shadow", 0), frame_h)
    s["background_padding"] = scale_number(style.get("background_padding", 0), frame_h)
    s["background_radius"] = scale_number(style.get("background_radius", 0), frame_h)
    s["letter_spacing"] = scale_number(style.get("letter_spacing", 0), frame_h)
    return s


# -------------------------------------------------------------------- colours
def ass_color(hex_color, opacity=1.0):
    """'#RRGGBB' (+opacity) → ASS '&HAABBGGRR'. ASS alpha: 00 opaque, FF clear."""
    c = cs.parse_color(hex_color).lstrip("#")
    r, g, b = c[0:2], c[2:4], c[4:6]
    a = int(round((1.0 - max(0.0, min(1.0, float(opacity)))) * 255))
    return f"&H{a:02X}{b}{g}{r}"


def _ass_escape(text):
    """Escape the characters ASS would otherwise interpret."""
    return (text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
                .replace("\n", "\\N"))


def _ass_time(sec):
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h}:{m:02d}:{s:05.2f}"


# ------------------------------------------------------------------ ASS build
_ALIGN_ASS = {
    ("bottom", "left"): 1, ("bottom", "center"): 2, ("bottom", "right"): 3,
    ("center", "left"): 4, ("center", "center"): 5, ("center", "right"): 6,
    ("top", "left"): 7, ("top", "center"): 8, ("top", "right"): 9,
}


def _rounded_rect(x0, y0, x1, y1, r):
    """ASS vector drawing for a rounded rectangle (real corner radius)."""
    r = max(0.0, min(r, (x1 - x0) / 2.0, (y1 - y0) / 2.0))
    k = r * 0.5523
    cmds = []
    if r <= 0.6:
        return f"m {x0:.1f} {y0:.1f} l {x1:.1f} {y0:.1f} l {x1:.1f} {y1:.1f} l {x0:.1f} {y1:.1f}"
    cmds.append(f"m {x0 + r:.1f} {y0:.1f}")
    cmds.append(f"l {x1 - r:.1f} {y0:.1f}")
    cmds.append(f"b {x1 - r + k:.1f} {y0:.1f} {x1:.1f} {y0 + r - k:.1f} {x1:.1f} {y0 + r:.1f}")
    cmds.append(f"l {x1:.1f} {y1 - r:.1f}")
    cmds.append(f"b {x1:.1f} {y1 - r + k:.1f} {x1 - r + k:.1f} {y1:.1f} {x1 - r:.1f} {y1:.1f}")
    cmds.append(f"l {x0 + r:.1f} {y1:.1f}")
    cmds.append(f"b {x0 + r - k:.1f} {y1:.1f} {x0:.1f} {y1 - r + k:.1f} {x0:.1f} {y1 - r:.1f}")
    cmds.append(f"l {x0:.1f} {y0 + r:.1f}")
    cmds.append(f"b {x0:.1f} {y0 + r - k:.1f} {x0 + r - k:.1f} {y0:.1f} {x0 + r:.1f} {y0:.1f}")
    return " ".join(cmds)


def layout_caption(text, style, frame_w, frame_h, fit=None):
    """Compute the drawn geometry for one caption: lines, boxes, anchors.

    Pure geometry — used by the ASS writer and by the tests that assert bounds
    include marks, outline and padding.
    """
    fit = fit or fit_text(text, style, frame_w, frame_h)
    st = scale_style(style, frame_h)
    spacing = float(style.get("line_spacing", 1.25))
    lines = fit["lines"] or [""]
    lh = fit["line_height"]
    block_h = lh * len(lines)
    margin_h = frame_w * float(style.get("margin_h", 8)) / 100.0
    margin_v = frame_h * float(style.get("margin_v", 9)) / 100.0
    position = style.get("position", "bottom")
    alignment = style.get("alignment", "center")
    pad = float(st["background_padding"]) if style.get("background") else 0.0
    radius = float(st["background_radius"]) if style.get("background") else 0.0
    outline = float(st["outline_width"])
    shadow = float(st["shadow"])
    # ink above/below the line box (marks!) — included in every bounds check
    need_top = fit["ink_above_px"]
    need_bottom = fit["ink_below_px"]

    box_w = min(frame_w - 2 * margin_h, max(fit["usable_w"], fit["max_line_width"] + 2 * pad))
    if alignment == "center":
        x_anchor = frame_w / 2.0
        box_x0 = x_anchor - box_w / 2.0
    elif alignment == "left":
        box_x0 = margin_h
        x_anchor = box_x0
    else:
        box_x0 = frame_w - margin_h - box_w
        x_anchor = box_x0 + box_w
    box_x1 = box_x0 + box_w

    if position == "bottom":
        y_bottom = frame_h - margin_v
        y_top = y_bottom - block_h
    elif position == "top":
        y_top = margin_v
        y_bottom = y_top + block_h
    else:
        y_top = frame_h / 2.0 - block_h / 2.0
        y_bottom = y_top + block_h

    panel = {"enabled": bool(style.get("background")),
             "x0": box_x0 if alignment != "left" else margin_h,
             "y0": y_top - pad, "x1": box_x1 if alignment != "right" else frame_w - margin_h,
             "y1": y_bottom + pad, "radius": radius}
    if alignment == "center":
        panel["x0"], panel["x1"] = x_anchor - box_w / 2.0, x_anchor + box_w / 2.0
        if style.get("background"):
            panel["x0"], panel["x1"] = x_anchor - max(
                box_w, fit["max_line_width"] + 2 * pad) / 2.0, x_anchor + max(
                box_w, fit["max_line_width"] + 2 * pad) / 2.0
    ink = {"top": y_top - need_top - outline - shadow + (0 if not panel["enabled"] else 0),
           "bottom": y_bottom + need_bottom + outline + shadow,
           "left": panel["x0"] - outline - shadow if panel["enabled"] else box_x0 - outline,
           "right": panel["x1"] + outline + shadow if panel["enabled"] else box_x1 + outline}
    return {"fit": fit, "style": st, "lines": lines, "line_height": lh, "block_h": block_h,
            "y_top": y_top, "y_bottom": y_bottom, "x_anchor": x_anchor,
            "box_w": box_w, "box_x0": box_x0, "box_x1": box_x1, "panel": panel,
            "alignment": alignment, "position": position, "ink": ink,
            "frame_w": frame_w, "frame_h": frame_h}


def bounds_problems(layout):
    """Out-of-frame warnings for a laid-out caption (marks/stroke/panel included)."""
    ink, w, h = layout["ink"], layout["frame_w"], layout["frame_h"]
    out = []
    if ink["top"] < -0.5:
        out.append(f"caption ink is {-ink['top']:.0f}px above the top edge")
    if ink["bottom"] > h + 0.5:
        out.append(f"caption ink is {ink['bottom'] - h:.0f}px below the bottom edge")
    if ink["left"] < -0.5:
        out.append(f"caption ink is {-ink['left']:.0f}px outside the left edge")
    if ink["right"] > w + 0.5:
        out.append(f"caption ink is {ink['right'] - w:.0f}px outside the right edge")
    return out


def build_ass(cues, style, frame_w, frame_h, karaoke=None, title=""):
    """ASS text for a list of cues — the single source of truth for rendering.

    ``cues`` = ``[{"text", "start", "end", "words"?}, …]``. ``words`` (when
    ``karaoke`` is enabled) is ``[(word, start, end), …]`` from
    :func:`media.words_for_timing` — proportional timing, and the render
    metadata says so rather than calling it forced alignment.
    """
    frame_w, frame_h = int(frame_w), int(frame_h)
    st = scale_style(style, frame_h)
    base = cs.normalize_style(style, strict=False)
    primary = ass_color(base["color"])
    outline_c = ass_color(base["outline_color"])
    shadow_c = ass_color(base["shadow_color"], base["shadow_opacity"])
    secondary = ass_color(base.get("karaoke_color", base["color"]))
    panel_c = ass_color(base["background_color"], base["background_opacity"])
    family = cs.FONT_FAMILIES[base["font"]]["family"]
    weight = int(st.get("weight", base.get("weight", 400)))
    spacing = float(base.get("line_spacing", 1.25))
    karaoke = karaoke or {}
    k_on = bool(karaoke.get("enabled"))
    k_primary = ass_color(karaoke.get("color", "#FFD84D"))
    header = [
        "[Script Info]",
        "; Generated by Khmer AI Content Studio — do not edit: the studio rebuilds this file",
        "ScriptType: v4.00+",
        "Title: " + re.sub(r"[\r\n]+", " ", str(title or ""))[:120],
        f"PlayResX: {frame_w}",
        f"PlayResY: {frame_h}",
        "WrapStyle: 2",                 # 2 = no automatic wrapping: we wrapped already
        "ScaledBorderAndShadow: yes",
        "YCbCr Matrix: None",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        # BorderStyle 1 = outline+shadow. Panels are drawn as vector shapes instead,
        # which is the only way to get the requested rounded corners in ASS.
        f"Style: Default,{family},{st['font_size_px']:.1f},"
        f"{k_primary if k_on else primary},{secondary},{outline_c},{panel_c},"
        f"{-1 if weight >= 600 else 0},0,0,0,100,100,{st['letter_spacing']:.2f},0,1,"
        f"{st['outline_width']:.2f},{st['shadow']:.2f},{_ALIGN_ASS[(base['position'], base['alignment'])]},"
        f"0,0,0,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    events = []
    for idx, cue in enumerate(cues):
        text = cue.get("text", "")
        if not text.strip():
            continue
        start, end = float(cue.get("start", 0.0)), float(cue.get("end", 0.0))
        if end - start < 0.04:
            end = start + 0.04
        lay = layout_caption(text, base, frame_w, frame_h)
        panel = lay["panel"]
        if panel["enabled"]:
            box = _rounded_rect(panel["x0"], panel["y0"], panel["x1"], panel["y1"],
                                panel["radius"])
            events.append(f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Default,,0,0,0,,"
                          f"{{\\an7\\pos(0,0)\\p1\\bord0\\shad0\\1c{panel_c}\\1a&H00&}}"
                          f"{box}{{\\p0}}")
        lines = lay["lines"]
        for li, line in enumerate(lines):
            y = lay["y_top"] + lay["line_height"] * (li + 0.5)
            if lay["alignment"] == "left":
                pos, an = f"{lay['box_x0']:.1f}", 4
            elif lay["alignment"] == "right":
                pos, an = f"{lay['box_x1']:.1f}", 6
            else:
                pos, an = f"{lay['x_anchor']:.1f}", 5
            body = _karaoke_line(line, cue, k_on, start, idx, frame_w, lay) if k_on \
                else _ass_escape(line)
            events.append(f"Dialogue: 1,{_ass_time(start)},{_ass_time(end)},Default,,0,0,0,,"
                          f"{{\\an{an}\\pos({pos},{y:.1f})}}{body}")
    if not events:
        events.append("Dialogue: 0,0:00:00.00,0:00:00.04,Default,,0,0,0,,{\\alpha&HFF&}")
    return "\n".join(header + events) + "\n"


def _karaoke_line(line, cue, enabled, start, idx, frame_w, lay):
    """``\\k`` tags inside one already-wrapped, cluster-safe line."""
    if not enabled:
        return _ass_escape(line)
    words = cue.get("words") or []
    if not words:
        return _ass_escape(line)
    plain = "".join(w for w, _s, _e in words)
    if plain.replace(" ", "") != line.replace(" ", ""):
        return _ass_escape(line)          # words and text disagree → never distort the text
    out, cur = [], 0
    for w, ws, we in words:
        gap = line.find(w, cur)
        if gap < 0:
            continue
        if gap > cur:
            out.append(_ass_escape(line[cur:gap]))
        dur = max(1, int(round((float(we) - float(ws)) * 100)))
        out.append(f"{{\\k{dur}}}{_ass_escape(w)}")
        cur = gap + len(w)
    if cur < len(line):
        out.append(_ass_escape(line[cur:]))
    # everything before the first word (leading spaces) is preserved verbatim
    return "".join(out)


def ass_style_line(style, frame_w, frame_h, karaoke=False):
    """The ``Style: Default,…`` line for a style dict (back-compat + diagnostics)."""
    text = build_ass([{"text": "ក", "start": 0, "end": 1}], style, frame_w, frame_h,
                     karaoke={"enabled": karaoke})
    for line in text.splitlines():
        if line.startswith("Style: Default,"):
            return line[len("Style: Default,"):]
    return ""


def write_ass(cues, style, frame_w, frame_h, dst, karaoke=None, title=""):
    ensure_dir(os.path.dirname(dst) or ".")
    text = build_ass(cues, style, frame_w, frame_h, karaoke=karaoke, title=title)
    with open(dst, "w", encoding="utf-8") as f:
        f.write(text)
    return dst


def cues_from_scenes(scene_texts, starts, ends, style=None):
    """Scene text → cues, split per sentence and timed inside the scene window.

    Scene boundaries are real audio boundaries (see ``engines.assembly``), so a
    sentence never outlives its own scene and the last cue ends at the real end
    of the narration instead of `start + 3s`.
    """
    if style is None:
        style = cs.normalize_style({}, strict=False)
    cues = []
    for i, text in enumerate(scene_texts):
        start = float(starts[i])
        end = float(ends[i]) if i < len(ends) else start + 3.0
        end = max(end, start + 0.35)
        sentences = split_caption_sentences(text)
        span = (end - start) / max(1, len(sentences))
        for j, sent in enumerate(sentences):
            s0, s1 = start + j * span, start + (j + 1) * span if j + 1 < len(sentences) else end
            cues.append({"text": sent, "start": round(s0, 3), "end": round(max(s1, s0 + 0.2), 3)})
    return cues


def split_caption_sentences(text):
    """Sentence-sized caption blocks (respects explicit newlines and [[silent:]])."""
    text = khmer.display_text(text or "").strip()
    if not text:
        return []
    out = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        parts, cur = [], ""
        for ch in para:
            cur += ch
            if ch in "។៕!?.":
                if cur.strip():
                    parts.append(cur.strip())
                cur = ""
        if cur.strip():
            parts.append(cur.strip())
        out.extend(parts)
    return out or [text]


# --------------------------------------------------------------------- SRT
_SRT_CUE_RE = re.compile(
    r"(?P<idx>\d+)\s*\n(?P<s>\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*"
    r"(?P<e>\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})\s*\n(?P<text>.*?)(?=\n\s*\n|\Z)", re.S)


def _parse_srt_time(t):
    h, m, rest = t.replace(",", ".").split(":")
    return int(h) * 3600 + int(m) * 60 + float(rest)


def parse_srt(text_or_path):
    """Read an SRT file/text into the studio's cue dicts (used by legacy callers)."""
    if os.path.exists(str(text_or_path)):
        with open(text_or_path, "r", encoding="utf-8-sig") as f:
            raw = f.read()
    else:
        raw = str(text_or_path or "")
    cues = []
    for m in _SRT_CUE_RE.finditer(raw.replace("\ufeff", "")):
        body = m.group("text").strip("\n")
        body = body.replace("\r", "")
        if not body.strip():
            continue
        cues.append({"text": body, "start": _parse_srt_time(m.group("s")),
                     "end": _parse_srt_time(m.group("e"))})
    return cues


def _srt_time(sec):
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:02d}:{m:02d}:{int(s):02d},{int(round((s % 1) * 1000)):03d}"


def write_srt(cues, dst, style=None, width=1920, height=1080):
    """Cluster-safe SRT with the *actual* end time of every cue.

    Lines are wrapped with the same shaped measurement the burn uses, and never
    inside a Khmer cluster (the old writer cut ``ខ្លួន។`` into ``ខ្លួ`` + ``ន។``).
    """
    style = style or cs.normalize_style({}, strict=False)
    blocks, n = [], 0
    for cue in cues:
        text = cue.get("text", "")
        if not text.strip():
            continue
        fit = fit_text(text, style, width, height)
        lines = fit["lines"] or [text]
        n += 1
        blocks.append(f"{n}\n{_srt_time(cue.get('start', 0))} --> {_srt_time(cue.get('end', 0))}\n"
                      + "\n".join(lines) + "\n")
    ensure_dir(os.path.dirname(dst) or ".")
    with open(dst, "w", encoding="utf-8") as f:
        f.write("\n".join(blocks))
    return dst


# ------------------------------------------------------------------- burning
def _escape_filter_path(path):
    """Path for a ffmpeg filter option value (always forward slashes)."""
    return (str(path).replace("\\", "/")
            .replace(":", r"\:")
            .replace("'", r"\'"))


def ass_filter(ass_path, font_dir=None):
    vf = f"subtitles='{_escape_filter_path(ass_path)}'"
    if font_dir:
        vf += f":fontsdir='{_escape_filter_path(font_dir)}'"
    return vf


def burn(video, ass_path, dst, style, frame_w=0, frame_h=0, crf=22, preset="veryfast",
         audio="copy"):
    """Burn an ASS file onto a video with the bundled font directory."""
    require_libass()
    font_dir = font_dir_for_style(style)
    vf = ass_filter(ass_path, font_dir)
    args = ["-i", video, "-vf", vf, "-c:v", "libx264", "-preset", preset, "-crf",
            str(int(crf)), "-pix_fmt", "yuv420p"]
    if audio == "copy":
        args += ["-c:a", "copy"]
    else:
        args += ["-an"]
    args += [dst]
    run_ffmpeg(args, timeout=3600)
    if not os.path.exists(dst) or os.path.getsize(dst) < 1024:
        raise CaptionError("caption burn produced no readable file")
    return dst


def burn_into(video, cues, style, dst, karaoke=None, ass_path=None, title=""):
    """Convenience: build the ASS then burn it (single code path with the preview)."""
    from .util import media_duration
    info = probe_size(video)
    w, h = info["width"] or 1080, info["height"] or 1920
    ass_path = ass_path or (dst + ".ass")
    write_ass(cues, style, w, h, ass_path, karaoke=karaoke, title=title)
    out = burn(video, ass_path, dst, style)
    return {"path": out, "ass": ass_path, "width": w, "height": h,
            "duration": media_duration(out, 0.0)}


def probe_size(path):
    ff = ffmpeg_exe()
    out = {"width": 0, "height": 0}
    if not ff or not path or not os.path.exists(path):
        return out
    try:
        res = subprocess.run([ff, "-hide_banner", "-i", path], capture_output=True, timeout=60)
        # NB: bytes.decode("ignore") is a LookupError (the first argument is the
        # *encoding*), which is how media.probe() managed to always report 0x0.
        m = re.search(r"Video:.*?(\d{2,5})x(\d{2,5})",
                      (res.stderr or b"").decode("utf-8", "ignore"))
        if m:
            out["width"], out["height"] = int(m.group(1)), int(m.group(2))
    except Exception:
        pass
    return out


# ------------------------------------------------------------------ preview
def preview_frame(cues, style, frame_w, frame_h, dst_png, at_sec=None, backdrop=None,
                  karaoke=None):
    """Render one caption frame at full video resolution — the export renderer.

    ``backdrop`` is a video/image to composite onto (a real frame of the project,
    so the preview shows the actual contrast); with no backdrop a studio-style
    gradient/grid is generated, which is what the style gallery uses.
    """
    require_libass()
    cues = [c for c in cues if c.get("text", "").strip()]
    if not cues:
        raise CaptionError("nothing to preview")
    at = float(at_sec if at_sec is not None else cues[0].get("start", 0.0) + 0.05)
    active = [c for c in cues if float(c.get("start", 0)) - 0.001 <= at <= float(c.get("end", 0)) + 0.001]
    if not active:
        # show the cue starting after `at` (the caller previews "what will this line look like")
        nxt = [c for c in cues if float(c.get("start", 0)) >= at]
        active = nxt[:1] or cues[:1]
        at = float(active[0].get("start", 0)) + 0.05
    ass = dst_png + ".ass"
    write_ass(cues, style, frame_w, frame_h, ass, karaoke=karaoke, title="caption preview")
    font_dir = font_dir_for_style(style)
    ff = ffmpeg_exe()
    if not ff:
        raise CaptionError("no ffmpeg available for preview")
    vf = ass_filter(ass, font_dir)
    backdrop = str(backdrop) if backdrop and os.path.exists(str(backdrop)) else ""
    if backdrop:
        still = dst_png + ".still.png"
        subprocess.run([ff, "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{at:.3f}",
                        "-i", backdrop, "-frames:v", "1", still], check=True, timeout=300)
        base_args = ["-i", still]
    else:
        base_args = ["-f", "lavfi", "-i",
                     f"color=c=0x1B2028:s={int(frame_w)}x{int(frame_h)}:r=25:d=1"]
    try:
        subprocess.run([ff, "-hide_banner", "-loglevel", "error", "-y", *base_args,
                        "-vf", vf, "-frames:v", "1", dst_png], check=True,
                       capture_output=True, timeout=300)
    except subprocess.CalledProcessError as e:
        raise CaptionError("preview render failed: "
                           + (e.stderr or b"").decode("utf-8", "ignore")[:300])
    finally:
        for tmp in (dst_png + ".still.png",):
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
    if not os.path.exists(dst_png):
        raise CaptionError("preview render produced no image")
    return dst_png


_COVERAGE_CACHE = {}


def coverage_gaps(text, style):
    """Characters of `text` the selected font has no glyph for (.notdef).

    Reported, never hidden: libass substitutes another installed font for these,
    which *usually* looks fine (Battambang has no em-dash on purpose and the
    dash still appears) — but it is the user's machine that decides, so the
    studio says which characters are not carried by the chosen family.
    """
    font_id = style.get("font")
    weight = style.get("weight")
    key = (font_id, weight)
    try:
        path, used = cs.font_file(font_id, weight)
    except cs.CaptionStyleError as e:
        return {"font": font_id, "missing": [], "error": str(e)}
    cache = _COVERAGE_CACHE.setdefault(key, {})
    hb = None
    try:
        import uharfbuzz as hb          # noqa: F401
        blob = hb.Blob.from_file_path(path)
        font = hb.Font(hb.Face(blob))
    except Exception:
        return {"font": font_id, "weight": used, "missing": [],
                "note": "coverage check needs uharfbuzz (pip install uharfbuzz)"}
    missing = []
    for ch in sorted({c for c in (text or "") if not c.isspace()}):
        if ch not in cache:
            buf = hb.Buffer()
            buf.add_str(ch)
            buf.guess_segment_properties()
            hb.shape(font, buf)
            cache[ch] = any(g.codepoint == 0 for g in buf.glyph_infos)
        if cache[ch]:
            missing.append(ch)
    return {"font": font_id, "weight": used,
            "family": cs.FONT_FAMILIES[font_id]["family"], "missing": missing}


def coverage_warnings(cues, style):
    text = " ".join(c.get("text", "") for c in cues)
    gaps = coverage_gaps(text, style)
    if gaps.get("missing"):
        shown = "".join(gaps["missing"][:8])
        return [f"{gaps.get('family', gaps['font'])} has no glyph for {shown!r} — libass "
                f"will substitute another installed font for those characters; check the "
                f"preview, or pick a font that covers them"]
    return []


def render_metadata(cues, style, frame_w, frame_h, karaoke=None):
    """Honest description of what a render will contain (goes in the manifest + API)."""
    st = cs.normalize_style(style, strict=False)
    rep = font_report(st)
    fits = []
    for c in cues:
        fit = fit_text(c.get("text", ""), st, frame_w, frame_h)
        lay = layout_caption(c.get("text", ""), st, frame_w, frame_h, fit=fit)
        fits.append({"lines": len(fit["lines"]), "font_size_px": round(fit["font_size_px"], 1),
                     "warnings": fit["warnings"], "bounds": bounds_problems(lay),
                     "measurement": fit["measurement"]})
    return {
        "renderer": "ffmpeg/libass (HarfBuzz shaping)",
        "measurement": fits[0]["measurement"] if fits else "n/a",
        "font": rep,
        "style": st,
        "frame": {"width": frame_w, "height": frame_h},
        "cues": len(cues),
        "per_cue": fits,
        "warnings": sorted({w for f in fits for w in f["warnings"]} |
                           set(coverage_warnings(cues, st))),
        "bounds_warnings": sorted({w for f in fits for w in f["bounds"]}),
        "coverage": coverage_gaps(" ".join(c.get("text", "") for c in cues), st),
        "karaoke": {"enabled": bool((karaoke or {}).get("enabled")),
                    "timing": "proportional estimate (not forced alignment)"},
    }


_CAPS = {}


def capabilities(cfg=None, refresh=False):
    """What caption rendering can do on this machine — with actionable fixes.

    ``ok`` is False when burned-in captions cannot be produced correctly. The UI
    shows ``problems`` verbatim, and the assemble stage refuses to pretend.
    """
    if _CAPS.get("data") and not refresh:
        return _CAPS["data"]
    ff = ffmpeg_exe()
    libass = has_filter("subtitles")
    shaping = "harfbuzz" if _uharfbuzz_available() else "estimate"
    probed = {"ffmpeg": ff or None}
    fonts = []
    missing = []
    for key, fam in cs.FONT_FAMILIES.items():
        try:
            path, weight = cs.font_file(key)
            fonts.append({"id": key, "family": fam["family"], "file": os.path.basename(path),
                          "weight": weight, "license": fam["license"], "ok": True})
        except cs.CaptionStyleError as e:
            fonts.append({"id": key, "family": fam["family"], "ok": False, "error": str(e)})
            missing.append(fam["family"])
    problems = []
    if not ff:
        problems.append("ffmpeg was not found — install ffmpeg (Windows: winget install Gyan.FFmpeg) "
                        "or `pip install imageio-ffmpeg`")
    elif not libass:
        problems.append("this ffmpeg has no libass ('subtitles' filter) — burned-in captions cannot "
                        "be rendered; install the gyan.dev Windows build or keep SRT as a sidecar")
    if missing:
        problems.append("bundled font files are missing: " + ", ".join(missing)
                        + " — restore ai_studio/fonts/ from the repository")
    if shaping != "harfbuzz":
        problems.append("uharfbuzz is not installed — line breaks fall back to an estimate "
                        "(the burn itself is still shaped by libass). Optional: pip install uharfbuzz")
    data = {"ok": not [p for p in problems if "uharfbuzz" not in p],
            "libass": libass, "shaping": shaping, "ffmpeg": probed["ffmpeg"],
            "fonts": fonts, "problems": problems,
            "reference_height": REFERENCE_HEIGHT,
            "font_cache": _default_font_cache(),
            "font_dir_for": {k: font_dir_for_style(cs.preset_style("clean") | {"font": k})
                             for k in cs.FONT_FAMILIES} if libass else {}}
    _CAPS["data"] = data
    return data


def _uharfbuzz_available():
    try:
        import uharfbuzz  # noqa: F401
        return True
    except Exception:
        return False


def palettes():
    """Named colours the UI offers next to the pickers."""
    return {"colors": [{"name": k, "hex": v} for k, v in sorted(cs._NAMED.items())],
            "reference_height": REFERENCE_HEIGHT}

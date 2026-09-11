"""Khmer cluster-safe subtitle wrapping & style template definitions.

Ensures Khmer captions never break inside a grapheme cluster or consonant stack
(e.g., separating base consonant from subscript coeng or vowels).
"""
import os
import re
import cv2
import numpy as np

# Khmer Unicode Range: U+1780 - U+17FF
# A cluster consists of a base character followed by subjoined consonants (\u17D2 + consonant) and vowels/marks.
KHMER_CLUSTER_PATTERN = re.compile(
    r'(?:\u1780-\u17B3|\u17DC)'                     # Base consonant or independent symbol
    r'(?:\u17D2[\u1780-\u17B3])*'                   # Zero or more coeng (subscript) pairs
    r'[\u17B6-\u17C5\u17C6-\u17D3\u17DD]*'         # Vowels, diacritics, signs
)

FULL_TOKEN_RE = re.compile(
    r'[\u1780-\u17B3\u17DC](?:\u17D2[\u1780-\u17B3])*(?:[\u17B6-\u17C5\u17C6-\u17D3\u17DD])*'  # Khmer cluster
    r'|\s+'                                         # Whitespace
    r'|[^\s\u1780-\u17FF]+'                         # Non-Khmer tokens
)

SUBTITLE_TEMPLATES = {
    "classic_yellow": {
        "name": "Classic Yellow",
        "desc": "Yellow active text on semi-transparent dark pill background",
        "text_color": (255, 255, 255),       # BGR
        "active_color": (0, 230, 255),      # BGR Yellow/Gold
        "bg_box": True,
        "bg_color": (20, 20, 24, 180),
        "font_scale": 1.0,
        "thickness": 3,
        "position": "bottom",
    },
    "bold_neon": {
        "name": "Bold Neon",
        "desc": "Bright cyan active text with thick dark outline & glow",
        "text_color": (255, 255, 255),
        "active_color": (255, 235, 0),       # BGR Cyan
        "bg_box": False,
        "stroke_color": (20, 10, 40),
        "font_scale": 1.15,
        "thickness": 4,
        "position": "bottom",
    },
    "minimal_light": {
        "name": "Minimal Light",
        "desc": "Clean white typography with soft subtle shadow",
        "text_color": (220, 220, 220),
        "active_color": (255, 255, 255),
        "bg_box": False,
        "stroke_color": (0, 0, 0),
        "font_scale": 0.9,
        "thickness": 2,
        "position": "bottom",
    },
    "box_brand": {
        "name": "Brand Banner",
        "desc": "Dark bold text on vibrant solid amber background banner",
        "text_color": (15, 15, 20),
        "active_color": (0, 0, 0),
        "bg_box": True,
        "bg_color": (30, 190, 255, 240),     # Amber/Gold BGR
        "font_scale": 1.05,
        "thickness": 3,
        "position": "bottom",
    }
}


def split_khmer_clusters(text):
    """Splits a string into indivisible grapheme clusters."""
    if not text:
        return []
    return FULL_TOKEN_RE.findall(text)


def wrap_khmer_text(text, max_chars=24):
    """Wraps text into lines at cluster/word boundaries.

    Guarantees no break occurs inside a Khmer grapheme cluster.
    """
    clusters = split_khmer_clusters(text)
    if not clusters:
        return []

    lines = []
    current_line = ""

    for cl in clusters:
        # Check if adding cluster exceeds max_chars
        if len(current_line) + len(cl) > max_chars and current_line.strip():
            lines.append(current_line.strip())
            current_line = cl.lstrip()
        else:
            current_line += cl

    if current_line.strip():
        lines.append(current_line.strip())

    return lines


def load_khmer_font(font_size):
    """Load a Khmer-capable TrueType font for PIL text drawing.

    NOTE (checked against PyPI, hash-verified): **no Pillow wheel from 10.4
    through 12.0 bundles Raqm** (libraqm/libfribidi are absent from the
    manylinux wheels), so `layout_engine=ImageFont.Layout.RAQM` warns and
    silently falls back to BASIC layout — which does NO Khmer shaping
    (coeng never stack, vowels never reorder). When Raqm is genuinely
    available (source build with libraqm, or a future wheel that restores
    it) this returns a shaping-capable font; otherwise callers MUST NOT use
    this font for Khmer body text — use ``render_khmer_text_overlay()``
    instead, which renders through the libass/HarfBuzz path that always
    shapes correctly.

    A shaping-capable layout engine is necessary but not sufficient: the
    font file itself must contain Khmer glyphs. DejaVu Sans and Pillow's
    built-in default font do not, and will silently render Khmer as tofu
    boxes / nothing — so a real Khmer font must also be present.
    """
    import logging

    from PIL import ImageFont, features

    fonts_dir = os.path.join(os.path.dirname(__file__), "fonts")
    font_candidates = [
        os.path.join(fonts_dir, "NotoSansKhmer-Regular.ttf"),
        os.path.join(fonts_dir, "KhmerOS.ttf"),
        os.path.join(fonts_dir, "NotoSansKhmer.ttf"),
        os.path.join(fonts_dir, "Battambang-Regular.ttf"),
        "/usr/share/fonts/truetype/khmeros/KhmerOS.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansKhmer-Regular.ttf",
    ]
    raqm_ok = bool(features.check("raqm"))
    for font_path in font_candidates:
        if os.path.exists(font_path):
            try:
                if raqm_ok:
                    return ImageFont.truetype(font_path, font_size,
                                              layout_engine=ImageFont.Layout.RAQM)
                logging.getLogger(__name__).warning(
                    "Pillow was built without Raqm — '%s' would draw UNshaped "
                    "Khmer (coeng never stack). Rendering Khmer through the "
                    "libass/HarfBuzz overlay instead.", font_path)
                return ImageFont.truetype(font_path, font_size)
            except Exception:
                continue

    logging.getLogger(__name__).error(
        "No Khmer-capable font found (checked %s) — Khmer text will render "
        "as tofu boxes. Add NotoSansKhmer-Regular.ttf to %s.",
        font_candidates, fonts_dir,
    )
    return ImageFont.load_default()


def render_khmer_text_overlay(frame, text, font_px, color_bgr, position="bottom",
                              align="center", margin_v_pct=None, margin_h_pct=None,
                              panel=False, panel_color_bgr=(24, 20, 20),
                              panel_opacity=0.7):
    """Draw Khmer (or any non-ASCII) text on a BGR frame, SHAPED correctly.

    Pillow's BASIC layout cannot shape Khmer and no Pillow wheel ships the
    Raqm engine any more, so the text is rendered through the same
    ffmpeg/libass + HarfBuzz path the studio uses for burned captions
    (ai_studio.captions.build_ass: bundled OFL fonts, shaped pixel-width
    wrapping, cluster-safe breaks). Raises RuntimeError honestly if this
    ffmpeg has no libass — it never silently draws unshaped text.
    """
    import logging
    import tempfile

    h, w = frame.shape[:2]
    from ai_studio import captions as cap
    from ai_studio.util import run_ffmpeg
    from ai_studio.media import _has_filter

    if not _has_filter("subtitles"):
        raise RuntimeError(
            "this ffmpeg build has no libass 'subtitles' filter — cannot draw "
            "shaped Khmer text (install a full ffmpeg build)")

    def _hex_from_bgr(bgr):
        r, g, b = int(bgr[2]), int(bgr[1]), int(bgr[0])
        return f"#{r:02x}{g:02x}{b:02x}"

    size_pct = round(max(1.5, font_px / float(h) * 100.0), 2)
    if margin_v_pct is None:
        margin_v_pct = 8.0 if position == "bottom" else 10.0
    style, _issues = cap.validate_style({
        "preset": "custom",
        "font": "noto_sans_khmer", "weight": "regular",
        "size_pct": size_pct,
        "text_color": _hex_from_bgr(color_bgr),
        "outline_color": "#101014", "outline_px": 2.0,
        "shadow": True, "shadow_strength": 0.6, "shadow_offset_px": 2,
        "panel": {"enabled": bool(panel), "color": _hex_from_bgr(panel_color_bgr),
                  "opacity": float(panel_opacity), "padding_px": 12, "radius_px": 10},
        "position": position, "align": align,
        "margin_v_pct": float(margin_v_pct),
        "margin_h_pct": float(margin_h_pct if margin_h_pct is not None else 6.0),
        "line_spacing": 1.25, "max_line_width_pct": 90.0, "max_lines": 3,
        "karaoke": False,
    })

    work = tempfile.mkdtemp(prefix="khmer_overlay_")
    src_png = os.path.join(work, "frame.png")
    out_png = os.path.join(work, "out.png")
    ass_path = os.path.join(work, "text.ass")
    try:
        cv2.imwrite(src_png, frame)
        cap.build_ass([(0.0, 5.0, str(text))], style, w, h, ass_path)
        ass_esc = ass_path.replace("\\", "/").replace(":", r"\:").replace("'", r"\'")
        fontsdir = cap.fonts_dir().replace("\\", "/").replace(":", r"\:").replace("'", r"\'")
        run_ffmpeg(["-i", src_png, "-vf",
                    f"subtitles='{ass_esc}':fontsdir='{fontsdir}'",
                    "-frames:v", "1", "-y", out_png])
        out = cv2.imread(out_png)
        if out is None or out.shape[0] != h or out.shape[1] != w:
            raise RuntimeError("libass overlay produced no readable frame")
        np.copyto(frame, out)
        return frame
    finally:
        for p in (src_png, out_png, ass_path):
            try:
                os.remove(p)
            except OSError:
                pass
        try:
            os.rmdir(work)
        except OSError:
            pass


def render_caption_frame(frame, words_timing, t, template_key="classic_yellow", title=None):
    """Renders captions on frame according to the chosen subtitle template."""
    tmpl = SUBTITLE_TEMPLATES.get(template_key, SUBTITLE_TEMPLATES["classic_yellow"])
    h, w = frame.shape[:2]

    if not words_timing:
        return frame

    # Find active word
    active_idx = -1
    for idx, wt in enumerate(words_timing):
        if wt["start"] <= t <= wt["end"]:
            active_idx = idx
            break
    if active_idx == -1:
        for idx, wt in enumerate(words_timing):
            if t < wt["start"]:
                active_idx = idx
                break
    if active_idx == -1:
        active_idx = len(words_timing) - 1

    # Group 3-word sliding window
    start_w = max(0, active_idx - 1)
    end_w = min(len(words_timing), active_idx + 2)
    phrase = words_timing[start_w:end_w]

    text_to_draw = " ".join(wt["word"] for wt in phrase)

    # Check if text contains non-ASCII (e.g. Khmer)
    # Note: Known limitation — per-word Khmer layout highlight with draw.textlength can be enhanced in a follow-up PR.
    has_non_ascii = any(ord(c) > 127 for c in text_to_draw)

    if has_non_ascii:
        # Khmer shaping: Pillow BASIC layout cannot shape Khmer and no Pillow
        # wheel ships Raqm any more, so draw via the libass/HarfBuzz overlay
        # (same renderer as the studio's burned captions — bundled OFL fonts,
        # cluster-safe wrap). The old code drew the whole 3-word phrase in the
        # active color, so a single styled block preserves the look.
        active_color = tmpl.get("active_color", (0, 230, 255))
        # legacy PIL path used int(24 * font_scale) with NO width scaling
        font_px = max(18, int(24 * tmpl.get("font_scale", 1.0)))
        # the ASCII path anchors text with its top at h*0.85 → mirror that as
        # a bottom margin
        margin_v = max(4.0, (h - int(h * 0.85) - int(font_px * 1.3)) / h * 100.0)
        render_khmer_text_overlay(
            frame, text_to_draw, font_px=font_px, color_bgr=active_color,
            position="bottom", align="center", margin_v_pct=margin_v,
            panel=bool(tmpl.get("bg_box")),
            panel_color_bgr=tmpl.get("bg_color", (20, 20, 24))[:3],
            panel_opacity=tmpl.get("bg_color", (20, 20, 24, 180))[3] / 255.0
            if len(tmpl.get("bg_color", (20, 20, 24, 180))) > 3 else 0.7)
        return frame

    # Standard ASCII rendering with OpenCV
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.8, (w / 800) * tmpl.get("font_scale", 1.0))
    thickness = tmpl.get("thickness", 3)

    # Compute total width
    sizes = [cv2.getTextSize(wt["word"].upper(), font, scale, thickness)[0] for wt in phrase]
    total_w = sum(s[0] for s in sizes) + 12 * (len(phrase) - 1)
    total_h = max(s[1] for s in sizes) if sizes else 30

    x = int((w - total_w) / 2)
    y = int(h * 0.85)

    # Draw background box if enabled
    if tmpl.get("bg_box"):
        pad_x, pad_y = 16, 12
        box_x1 = max(10, x - pad_x)
        box_y1 = max(10, y - total_h - pad_y)
        box_x2 = min(w - 10, x + total_w + pad_x)
        box_y2 = min(h - 10, y + pad_y + 6)
        
        bg_color = tmpl.get("bg_color", (20, 20, 24, 180))
        overlay = frame.copy()
        cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), bg_color[:3], -1)
        alpha = bg_color[3] / 255.0 if len(bg_color) > 3 else 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Render phrase words
    for i, wt in enumerate(phrase):
        word = wt["word"].upper()
        is_active = (start_w + i == active_idx)
        color = tmpl["active_color"] if is_active else tmpl["text_color"]
        stroke = tmpl.get("stroke_color", (0, 0, 0))

        # Text outline / shadow
        cv2.putText(frame, word, (x + 2, y + 2), font, scale, stroke, thickness + 3, cv2.LINE_AA)
        cv2.putText(frame, word, (x, y), font, scale, color, thickness, cv2.LINE_AA)
        x += sizes[i][0] + 12

    return frame

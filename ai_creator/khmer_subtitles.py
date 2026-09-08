"""Khmer cluster-safe subtitle wrapping & style template definitions.

Ensures Khmer captions never break inside a grapheme cluster or consonant stack
(e.g., separating base consonant from subscript coeng or vowels).
"""
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


def render_caption_frame(frame, words_timing, t, template_key="classic_yellow", title=None):
    """Renders captions on frame according to the chosen subtitle template."""
    tmpl = SUBTITLE_TEMPLATES.get(template_key, SUBTITLE_TEMPLATES["classic_yellow"])
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

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

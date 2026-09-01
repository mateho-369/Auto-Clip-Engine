"""Character animation math: entry, exit, idle bob and talk-pulse transforms.

All transforms are pure functions of (kind, t) so they are trivially
testable and deterministic. The renderer maps the returned
(scale, offset_x_frac, offset_y_frac, alpha) onto the character sprite.
"""
import math
import numpy as np

ENTRIES = ["pop-in", "slide-left", "slide-right", "zoom", "bounce", "fade-in"]
EXITS = ["fade-out", "slide-down", "shrink"]
ENTRY_DUR = 0.6
EXIT_DUR = 0.5


def ease_out_cubic(p):
    return 1 - (1 - p) ** 3


def ease_out_back(p):
    c1 = 1.70158
    c3 = c1 + 1
    return 1 + c3 * (p - 1) ** 3 + c1 * (p - 1) ** 2


def entry_state(kind, t, dur=ENTRY_DUR):
    """Returns (scale, ox_frac, oy_frac, alpha) during an entry animation."""
    p = max(0.0, min(1.0, t / dur))
    kind = kind if kind in ENTRIES else "pop-in"
    if p >= 1.0:
        return (1.0, 0.0, 0.0, 1.0)
    if kind == "pop-in":
        s = max(0.05, ease_out_back(p))
        return (s, 0.0, 0.0, min(1.0, p * 6))
    if kind == "slide-left":
        e = ease_out_cubic(p)
        return (1.0, -1.15 * (1 - e), 0.0, min(1.0, p * 4))
    if kind == "slide-right":
        e = ease_out_cubic(p)
        return (1.0, 1.15 * (1 - e), 0.0, min(1.0, p * 4))
    if kind == "zoom":
        s = 1.35 - 0.35 * ease_out_cubic(p)
        return (s, 0.0, 0.0, min(1.0, p * 3))
    if kind == "bounce":
        e = ease_out_cubic(p)
        oy = -0.22 * (1 - p) * abs(math.sin(p * 3 * math.pi))
        return (max(0.05, e), 0.0, oy, min(1.0, p * 5))
    # fade-in
    return (1.0, 0.0, 0.0, p)


def exit_state(kind, t, dur=EXIT_DUR):
    """t = time elapsed since the exit window started."""
    p = max(0.0, min(1.0, t / dur))
    kind = kind if kind in EXITS else "fade-out"
    if kind == "fade-out":
        return (1.0, 0.0, 0.0, 1.0 - p)
    if kind == "slide-down":
        e = p ** 2
        return (1.0, 0.0, e, 1.0 - p)
    if kind == "shrink":
        return (max(0.02, 1.0 - p), 0.0, 0.5 * p, 1.0 - p)
    return (1.0, 0.0, 0.0, 1.0 - p)


def talk_pulse(env_value):
    """Scale multiplier from a 0..1 audio envelope (the 'talking' bounce)."""
    return 1.0 + 0.055 * max(0.0, min(1.0, float(env_value)))


def idle_bob(t, amp=0.010, freq=1.7):
    """Gentle breathing/bobbing offset_y fraction while idle."""
    return amp * math.sin(2 * math.pi * freq * t)


def composite_rgba(frame_bgr, char_rgba, cx, cy, scale=1.0, ox_frac=0.0, oy_frac=0.0, alpha=1.0,
                   frame_w=None, frame_h=None):
    """Paste an RGBA character sprite onto a BGR frame with transform.

    cx/cy = anchor point (character center). ox_frac/oy_frac are fractions
    of frame width/height. Returns the (modified) frame.
    """
    if char_rgba is None or char_rgba.size == 0:
        return frame_bgr
    fh, fw = frame_bgr.shape[:2]
    if frame_w is None:
        frame_w = fw
    if frame_h is None:
        frame_h = fh
    ch, cw = char_rgba.shape[:2]
    new_w = max(2, int(cw * scale))
    new_h = max(2, int(ch * scale))
    resized = cv2_resize(char_rgba, new_w, new_h)
    if alpha < 1.0:
        resized = resized.copy()
        resized[:, :, 3] = (resized[:, :, 3] * alpha).astype(np.uint8)
    if resized.shape[2] == 3:  # plain BGR -> add alpha
        a = np.full((resized.shape[0], resized.shape[1], 1), 255, dtype=np.uint8)
        resized = np.dstack([resized, a])
    x0 = int(cx + ox_frac * frame_w - new_w / 2)
    y0 = int(cy + oy_frac * frame_h - new_h / 2)
    # clip to frame
    sx0, sy0 = max(0, -x0), max(0, -y0)
    sx1, sy1 = min(new_w, fw - x0), min(new_h, fh - y0)
    if sx1 <= sx0 or sy1 <= sy0:
        return frame_bgr
    dx0, dy0 = x0 + sx0, y0 + sy0
    region = frame_bgr[dy0:dy0 + (sy1 - sy0), dx0:dx0 + (sx1 - sx0)]
    patch = resized[sy0:sy1, sx0:sx1]
    a = patch[:, :, 3:4].astype(np.float32) / 255.0
    frame_bgr[dy0:dy0 + (sy1 - sy0), dx0:dx0 + (sx1 - sx0)] = (
        patch[:, :, :3].astype(np.float32) * a + region.astype(np.float32) * (1 - a)
    ).astype(np.uint8)
    return frame_bgr


def cv2_resize(img, w, h):
    import cv2
    flag = cv2.INTER_AREA if (w < img.shape[1] or h < img.shape[0]) else cv2.INTER_LINEAR
    return cv2.resize(img, (w, h), interpolation=flag)


def load_character_rgba(path):
    """Loads an RGBA character asset (falls back to a generated placeholder
    when the file is missing so renders never crash)."""
    import cv2
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is not None:
        if img.shape[2] == 4:
            return img
        a = np.full((img.shape[0], img.shape[1], 1), 255, dtype=np.uint8)
        return np.dstack([img, a])
    # placeholder: simple presenter avatar
    h, w = 640, 480
    canvas = np.zeros((h, w, 4), dtype=np.uint8)
    cv2.ellipse(canvas, (w // 2, h // 2), (w // 2 - 8, h // 2 - 8), 0, 0, 360, (200, 160, 90, 255), -1)
    cv2.circle(canvas, (w // 2, int(h * 0.36)), 95, (80, 220, 255, 255), -1)
    cv2.circle(canvas, (w // 2 - 35, int(h * 0.33)), 16, (0, 0, 0, 255), -1)
    cv2.circle(canvas, (w // 2 + 35, int(h * 0.33)), 16, (0, 0, 0, 255), -1)
    cv2.ellipse(canvas, (w // 2, int(h * 0.42)), (42, 18), 0, 0, 180, (0, 0, 0, 255), 4)
    return canvas

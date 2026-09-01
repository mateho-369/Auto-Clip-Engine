"""Scene transition blending between two BGR frames.

fade   — crossfade
slide  — outgoing pushed left, incoming slides in from the right
zoom   — outgoing zooms in while crossfading
wipe   — left-to-right reveal
cut    — hard cut (no blend)
"""
import numpy as np
from .animation import cv2_resize

TRANSITIONS = ["fade", "slide", "zoom", "wipe", "cut"]


def blend(a, b, p, kind="fade"):
    """Blend frame a (outgoing) into frame b (incoming) at progress p in (0,1)."""
    if kind not in TRANSITIONS:
        kind = "fade"
    if kind == "cut":
        return b.copy()
    p = max(0.0, min(1.0, float(p)))
    if kind == "fade":
        return (a.astype(np.float32) * (1 - p) + b.astype(np.float32) * p).astype(np.uint8)
    if kind == "zoom":
        h, w = a.shape[:2]
        za = cv2_resize(a, int(w * (1 + 0.28 * p)), int(h * (1 + 0.28 * p)))
        za = _center_crop(za, h, w)
        zb = cv2_resize(b, int(w * (1.18 - 0.18 * p)), int(h * (1.18 - 0.18 * p)))
        zb = _center_crop(zb, h, w)
        return (za.astype(np.float32) * (1 - p) + zb.astype(np.float32) * p).astype(np.uint8)
    if kind == "slide":
        h, w = a.shape[:2]
        out = np.zeros_like(a)
        off = int(w * p)
        # outgoing: visible left part shifted left
        if w - off > 0:
            out[:, :w - off] = a[:, off:]
        # incoming: enters from the right
        if off < w:
            out[:, w - off:] = b[:, :off] if off <= w else b[:, 0]
        return out
    if kind == "wipe":
        h, w = a.shape[:2]
        x0 = int(w * p)
        out = np.empty_like(a)
        out[:, :x0] = a[:, :x0]
        out[:, x0:] = b[:, x0:]
        return out
    return (a.astype(np.float32) * (1 - p) + b.astype(np.float32) * p).astype(np.uint8)


def _center_crop(img, h, w):
    ih, iw = img.shape[:2]
    if ih == h and iw == w:
        return img
    y0 = max(0, (ih - h) // 2)
    x0 = max(0, (iw - w) // 2)
    y1 = min(ih, y0 + h)
    x1 = min(iw, x0 + w)
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[: y1 - y0, : x1 - x0] = img[y0:y1, x0:x1]
    return out

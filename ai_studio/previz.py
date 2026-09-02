"""Procedural previz renderer — a calm animated clip, drawn with numpy+PIL.

This is what makes the studio *never* dead-end:

* Machine B (no CUDA) still gets a watchable full video;
* on Machine A it is the fallback when ComfyUI/Wan is offline, OOM'd, or the
  Director wants a timing draft before spending GPU minutes on real renders;
* it is also the `video` stage when `video.engine = "previz"`.

It deliberately matches the house style (Section 5): soft light, slow motion,
sky/hills/water, drifting petals — never random noise. Each mood tag from
Stage 1 selects a palette and a particle behaviour, and the clip is rendered at
the scene's real duration so the assembly stays frame-accurate.
"""
import math
import os
import subprocess

from .util import ensure_dir, ffmpeg_exe

# sky_top, sky_mid, horizon, sun, hills[3], water, mist_alpha, particle, particle_colour
PALETTES = {
    "sunrise-warm": ((24, 40, 92), (120, 92, 120), (236, 168, 118), (255, 226, 176),
                     [(60, 58, 84), (38, 40, 60), (20, 22, 36)], (40, 60, 92),
                     0.16, "petals", (255, 214, 214)),
    "water-calm": ((58, 96, 128), (128, 168, 186), (206, 224, 226), (255, 250, 226),
                   [(74, 96, 96), (48, 68, 74), (26, 42, 50)], (52, 92, 116),
                   0.20, "ripples", (236, 248, 255)),
    "forest-mist": ((46, 74, 70), (96, 130, 112), (178, 196, 168), (242, 246, 214),
                    [(36, 62, 52), (22, 44, 40), (12, 26, 26)], (44, 74, 70),
                    0.30, "leaves", (206, 226, 178)),
    "birds-dawn": ((40, 62, 110), (150, 150, 170), (238, 202, 170), (255, 236, 200),
                   [(64, 70, 88), (40, 46, 62), (24, 28, 40)], (58, 84, 106),
                   0.14, "birds", (36, 36, 42)),
    "flowers-still": ((74, 96, 140), (178, 160, 190), (240, 206, 214), (255, 240, 230),
                      [(96, 110, 96), (66, 82, 74), (40, 54, 56)], (76, 110, 126),
                      0.12, "petals", (255, 226, 236)),
    "night-quiet": ((6, 10, 26), (14, 24, 48), (34, 52, 78), (226, 236, 250),
                    [(10, 16, 26), (6, 10, 18), (3, 6, 12)], (10, 22, 40),
                    0.10, "stars", (236, 244, 255)),
    "rain-soft": ((54, 62, 74), (92, 104, 116), (140, 152, 160), (196, 208, 216),
                  [(48, 58, 62), (32, 42, 48), (20, 28, 34)], (64, 84, 96),
                  0.24, "rain", (208, 224, 236)),
    "path-walking": ((78, 96, 132), (176, 168, 150), (232, 210, 168), (255, 240, 206),
                     [(96, 92, 74), (66, 66, 56), (42, 44, 38)], (84, 92, 78),
                     0.12, "dust", (236, 224, 196)),
    "home-warm": ((44, 30, 30), (110, 74, 56), (200, 146, 96), (255, 214, 156),
                  [(70, 48, 40), (48, 34, 30), (28, 20, 18)], (60, 44, 36),
                  0.08, "dust", (255, 226, 180)),
    "study-calm": ((52, 56, 72), (120, 124, 140), (196, 190, 180), (250, 244, 226),
                   [(70, 72, 84), (50, 52, 62), (34, 36, 44)], (72, 74, 86),
                   0.06, "dust", (250, 246, 230)),
    "still-lake": ((58, 86, 122), (136, 168, 190), (214, 226, 232), (255, 252, 236),
                   [(70, 92, 96), (46, 66, 74), (26, 42, 52)], (64, 104, 138),
                   0.18, "ripples", (240, 250, 255)),
    "effort-dawn": ((44, 62, 104), (146, 140, 148), (232, 194, 156), (255, 234, 196),
                    [(70, 70, 76), (46, 50, 58), (28, 32, 40)], (66, 84, 96),
                    0.12, "dust", (255, 236, 210)),
    "kind-warm": ((52, 68, 100), (132, 146, 164), (210, 200, 196), (252, 240, 220),
                  [(72, 84, 86), (48, 60, 66), (30, 40, 46)], (68, 96, 112),
                  0.16, "rain", (226, 238, 246)),
    "calm-warm": ((38, 58, 106), (146, 138, 154), (238, 196, 158), (255, 232, 190),
                  [(64, 68, 86), (42, 46, 62), (24, 28, 42)], (56, 90, 116),
                  0.15, "petals", (255, 230, 220)),
}

PARTICLE_COUNTS = {"petals": 26, "leaves": 30, "dust": 44, "rain": 130, "birds": 4,
                   "ripples": 0, "stars": 0}


def palette_for(mood_tag, visual_prompt=""):
    mood = str(mood_tag or "").strip().lower()
    if mood in PALETTES:
        return PALETTES[mood]
    hay = f"{mood} {visual_prompt}".lower()
    table = [("star", "night-quiet"), ("night", "night-quiet"), ("moon", "night-quiet"),
             ("rain", "rain-soft"), ("tin roof", "rain-soft"),
             ("river", "water-calm"), ("water", "water-calm"), ("lake", "still-lake"),
             ("pond", "still-lake"), ("forest", "forest-mist"), ("tree", "forest-mist"),
             ("mist", "forest-mist"), ("fog", "forest-mist"), ("bird", "birds-dawn"),
             ("flower", "flowers-still"), ("lotus", "flowers-still"),
             ("sunrise", "sunrise-warm"), ("dawn", "sunrise-warm"), ("morning", "sunrise-warm"),
             ("kitchen", "home-warm"), ("home", "home-warm"), ("tea", "home-warm"),
             ("desk", "study-calm"), ("book", "study-calm"), ("notebook", "study-calm"),
             ("path", "path-walking"), ("walk", "path-walking"), ("road", "path-walking"),
             ("climb", "effort-dawn"), ("breath", "effort-dawn"), ("umbrella", "kind-warm")]
    for key, pal in table:
        if key in hay:
            return PALETTES[pal]
    return PALETTES["calm-warm"]


def _ridge(rng, width, base_y, amp, roughness=0.06):
    """Smooth 1-D silhouette (random-phase sine sum) → a hill/mountain line."""
    import numpy as np

    x = np.linspace(0, 1, width, dtype=np.float32)
    y = np.zeros(width, dtype=np.float32)
    for k in range(1, 7):
        freq = k * (1.0 / max(0.25, roughness * 8))
        y += np.sin(2 * math.pi * freq * x + float(rng.uniform(0, 6.28))) / (k ** 1.3)
    y /= max(1e-6, float(np.max(np.abs(y))))
    return (base_y - y * amp).astype(np.int32)


def render_clip(dst, duration=6.0, width=480, height=854, fps=16, mood_tag="", visual_prompt="",
                seed=0, motion=0.75, progress=None):
    """Render one previz clip. Returns {path, duration, width, height, fps, engine}."""
    import numpy as np

    ensure_dir(os.path.dirname(dst) or ".")
    duration = max(0.6, float(duration))
    width, height, fps = int(width), int(height), max(8, int(fps))
    n = max(int(round(duration * fps)), 8)
    sky_top, sky_mid, horizon_c, sun_c, hill_cols, water_c, mist_a, ptype, pcol = \
        palette_for(mood_tag, visual_prompt)
    rng = np.random.default_rng(abs(int(seed or 0)) % 9973 + 17)

    # Render oversized by 10% so the drifting crop never shows an edge.
    W, H = int(width * 1.10) | 1, int(height * 1.10) | 1
    horizon_y = int(H * (0.66 if ptype == "stars" else 0.58))
    water_h = max(2, H - horizon_y)

    img = np.zeros((H, W, 3), dtype=np.float32)
    g = np.linspace(0, 1, horizon_y, dtype=np.float32)
    top, mid, hor = (np.array(c, dtype=np.float32) for c in (sky_top, sky_mid, horizon_c))
    grad = np.where(g[:, None] < 0.55,
                    top[None, :] * (1 - g[:, None] / 0.55) + mid[None, :] * (g[:, None] / 0.55),
                    mid[None, :] * (1 - (g[:, None] - 0.55) / 0.45) + hor[None, :] *
                    ((g[:, None] - 0.55) / 0.45))
    img[:horizon_y] = grad[:, None, :]
    water = np.array(water_c, dtype=np.float32)
    img[horizon_y:] = (hor * 0.55 + water * 0.45)[None, None, :]

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    sun_x = int(W * (0.68 if ptype != "stars" else 0.30))
    sun_y = int(H * (0.20 if ptype != "stars" else 0.16))
    r = np.sqrt((xx - sun_x) ** 2 + ((yy - sun_y) * 1.5) ** 2) / max(1.0, W * 0.5)
    glow = np.clip(1.0 - r, 0, 1) ** (2.4 if ptype != "stars" else 3.6)
    disc = np.clip(1.0 - (r * W * 0.5) / (W * 0.05), 0, 1)
    sun3 = np.array(sun_c, dtype=np.float32)[None, None, :]
    img += (glow[..., None] * (0.42 if ptype != "stars" else 0.22) + disc[..., None] * 0.8) * sun3
    if ptype == "stars":
        stars = np.zeros((H, W), dtype=np.float32)
        for _ in range(int(W * H / 4800)):
            stars[int(rng.integers(0, max(2, horizon_y - 10))), int(rng.integers(0, W))] = \
                float(rng.uniform(0.3, 1.0))
        img += stars[..., None] * sun3 * 0.5

    layers = []
    for i, hc in enumerate(hill_cols):
        base_y = horizon_y - int(H * (0.015 + 0.05 * (len(hill_cols) - i)))
        amp = int(H * (0.030 + 0.026 * i))
        line = _ridge(rng, W, base_y, amp, roughness=0.05 + 0.035 * i)
        mask = np.zeros((H, W), dtype=bool)
        cols = np.arange(W)
        mask[line, cols] = True
        mask = np.cumsum(mask, axis=0).astype(bool)
        mask[horizon_y + 2:] = False
        layers.append((mask, np.array(hc, dtype=np.float32), 0.35 + 0.32 * i))
    for mask, col, _s in layers:
        img[mask] = col

    px = rng.uniform(0, W, PARTICLE_COUNTS.get(ptype, 20)).astype(np.float32)
    py = rng.uniform(0, H, PARTICLE_COUNTS.get(ptype, 20)).astype(np.float32)
    pv = rng.uniform(0.10, 0.85, PARTICLE_COUNTS.get(ptype, 20)).astype(np.float32)
    psz = rng.uniform(1.1, 3.2, PARTICLE_COUNTS.get(ptype, 20)).astype(np.float32)
    pph = rng.uniform(0, 6.28, PARTICLE_COUNTS.get(ptype, 20)).astype(np.float32)
    pcol3 = np.array(pcol, dtype=np.float32)

    vig = (1.0 - 0.28 * (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2))
    vig = np.clip(vig, 0.62, 1.0)[..., None]
    img *= vig
    img_reflect = img[horizon_y:][::-1].copy() * 0.16   # faint reflection: water stays deep

    ff = ffmpeg_exe()
    if not ff:
        raise RuntimeError("ffmpeg is required to render previz clips (scoop install ffmpeg)")
    cmd = [ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-an",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", dst]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE)
    written = 0
    try:
        for k in range(n):
            t = k / float(max(1, n - 1))
            frame = img.copy()
            dx = (t - 0.5) * 2.0 * motion              # -1..1 pan
            # horizon reflection shimmers under the water line
            band = img_reflect.copy()
            wob = np.clip(int(round(np.sin(t * 3.1) * 2)), -3, 3)
            ys = horizon_y + np.arange(band.shape[0])
            valid = ys < H
            if valid.any():
                frame[ys[valid]] = np.clip(frame[ys[valid]] + np.roll(band[valid],
                                                                      int(wob * 6), axis=1) * 0.55, 0, 255)
            # water ripple lines
            if water_h > 6:
                for rr in range(0, water_h, 6):
                    y2 = horizon_y + rr + int(round(2.0 * math.sin(t * 2.2 + rr * 0.4)))
                    if 0 <= y2 < H:
                        ph = np.sin(np.linspace(0, 6.28 * 5, W, dtype=np.float32) + t * 2.8 + rr * 0.5)
                        a = np.clip(0.05 + 0.05 * ph, 0.0, 0.13) * (1.0 - rr / max(1, water_h))
                        frame[y2] = np.clip(frame[y2] + a[..., None] * pcol3, 0, 255)
            # drifting mist across the horizon
            if mist_a > 0.02:
                mh = max(3, int(H * 0.055))
                xs = np.linspace(0, 6.28 * 3, W, dtype=np.float32)
                m = (0.55 + 0.45 * np.sin(xs + t * 1.1 + np.arange(mh)[:, None] * 0.4))
                m *= np.linspace(1.0, 0.15, mh)[:, None]
                m = np.roll(m, -int((t * 0.35 * W) % W), axis=1) * mist_a
                y0 = max(0, horizon_y - mh // 2)
                sl = frame[y0:y0 + mh]
                if sl.shape[0]:
                    mm = m[: sl.shape[0], :W][..., None]
                    frame[y0:y0 + sl.shape[0]] = sl * (1 - mm) + np.array(sun_c, dtype=np.float32)[None, None, :] * mm
            # particles
            if px.size:
                if ptype == "rain":
                    py = (py + pv * 9.0) % H
                    for i in range(0, px.size, 1):
                        x0 = int(px[i]) % W
                        y0 = int(py[i]) % H
                        ln = 10 + int(pv[i] * 22)
                        seg = np.arange(max(0, min(ln, H - y0)))
                        if seg.size:
                            ys = y0 + seg
                            xs = np.clip(x0 - (seg * 0.12).astype(np.int32), 0, W - 1)
                            frame[ys, xs] = np.clip(frame[ys, xs] + pcol3 * 0.5, 0, 255)
                elif ptype == "birds":
                    for i in range(px.size):
                        bx = (px[i] + t * (18 + pv[i] * 30)) % (W + 60) - 30
                        by = py[i] * 0.55 + H * 0.06 + math.sin(t * 6 + pph[i]) * 5
                        w = 6 + int(3 * math.sin(t * 9 + pph[i]))
                        for sx in (-w, w):
                            pts = np.arange(max(1, w))
                            yy2 = np.clip(int(by) + (pts * 0.6).astype(int), 0, H - 1)
                            xx2 = np.clip(int(bx) + sx + np.sign(sx) * pts, 0, W - 1)
                            frame[yy2, xx2] = pcol3
                else:
                    py = (py + pv * 1.1) % H
                    px = (px + np.sin(t * 2.0 + pph) * 0.7 + (0.5 if ptype != "dust" else 1.1)) % W
                    for i in range(px.size):
                        s = max(1, int(psz[i] * (1.0 if ptype != "petals" else 1.25)))
                        x0, y0 = int(px[i]) % W, int(py[i]) % H
                        frame[y0:min(H, y0 + s), x0:min(W, x0 + s)] = \
                            frame[y0:min(H, y0 + s), x0:min(W, x0 + s)] * 0.35 + pcol3 * 0.65
            proc.stdin.write(_crop_resample(frame, dx, t, width, height))
            written += 1
            if progress and (k % max(1, n // 16) == 0):
                progress(100.0 * (k + 1) / n, f"previz frame {k + 1}/{n}")
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass
        err = proc.stderr.read().decode(errors="ignore") if proc.stderr else ""
        rc = proc.wait(timeout=900)
    if rc != 0 or not os.path.exists(dst) or written == 0:
        raise RuntimeError(f"previz encode failed: {err[-400:] or 'no frames written'}")
    return {"path": dst, "duration": round(written / float(fps), 3), "width": width,
            "height": height, "fps": fps, "frames": written, "engine": "previz",
            "mood": mood_tag or "calm-warm"}


def _crop_resample(frame, dx, t, width, height):
    """Crop a slowly drifting, gently breathing window out of the big buffer."""
    import numpy as np
    from PIL import Image

    H, W = frame.shape[0], frame.shape[1]
    zx = 1.0 + 0.018 * math.sin(t * math.pi)
    cw = int(np.clip(width * zx, width, W))
    ch = int(np.clip(height * zx, height, H))
    x0 = int(np.clip((W - cw) / 2 + dx * (W - cw) * 0.5, 0, W - cw))
    y0 = int(np.clip((H - ch) / 2 - 0.10 * (H - ch) * t, 0, H - ch))
    sub = frame[y0:y0 + ch, x0:x0 + cw]
    img = Image.fromarray(np.clip(sub, 0, 255).astype("uint8"), "RGB")
    if (cw, ch) != (width, height):
        img = img.resize((width, height), Image.BILINEAR)
    return img.tobytes()

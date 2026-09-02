"""Stage-5 fallback: procedurally synthesised natural ambience (numpy only).

MMAudio is the intended video-to-audio model, but the pipeline must still
deliver sound when it is unavailable (Machine B, ComfyUI down, VRAM pressure,
first draft). Rather than silence, the studio synthesises the ambience the SFX
Director asked for — birds, water, wind in leaves, rain on a roof, crickets —
with the same "no music, no stingers, no manipulation" rule from the house
style guideline.

Pure numpy: deterministic per seed, no downloads, works air-gapped.
"""
import math
import os
import re

from .util import ensure_dir, write_wav

SR = 44100

# keyword → (component weights). Weights are [0..1] levels per layer.
LAYERS = {
    "birds":   dict(kinds=("birds",), level=0.55),
    "water":   dict(kinds=("water",), level=0.5),
    "wind":    dict(kinds=("wind",), level=0.4),
    "leaves":  dict(kinds=("leaves",), level=0.35),
    "rain":    dict(kinds=("rain",), level=0.55),
    "night":   dict(kinds=("crickets",), level=0.4),
    "room":    dict(kinds=("room",), level=0.22),
    "distant": dict(kinds=("village",), level=0.2),
}

TEXT_MAP = [
    (r"bird|chirp|សត្វបក្សី|បក្សី", ("birds",)),
    (r"water|river|stream|flow|brook|ទឹក|ទន្លេ|ស្ទឹង", ("water",)),
    (r"rain|drizzle|ភ្លៀង", ("rain", "room")),
    (r"wind|breeze|wind in the leaves", ("wind",)),
    (r"leav|rustl|ស្លឹកឈើ", ("leaves", "wind")),
    (r" cricket|night|evening|យប់", ("night",)),
    (r"forest|jungle|tree|ព្រៃ|ដើមឈើ", ("leaves", "birds")),
    (r"morning|dawn|sunrise|ព្រឹក", ("birds", "wind")),
    (r"ocean|sea|wave|សមុទ្រ", ("water", "wind")),
    (r"temple|bell|village|ភូមិ", ("village", "room")),
    (r"quiet|still|calm|peace|ស្ងប់|សន្តិភាព", ("room", "water")),
    (r"footstep|walk|path", ("leaves", "room")),
    (r"breath", ("room",)),
    (r"kitchen|tea|home|ផ្ទះ", ("room", "water")),
]

MOOD_MAP = {
    "sunrise-warm": ("birds", "wind"), "water-calm": ("water", "birds"),
    "forest-mist": ("leaves", "birds", "wind"), "birds-dawn": ("birds",),
    "flowers-still": ("birds", "leaves"), "night-quiet": ("night", "wind"),
    "rain-soft": ("rain", "room"), "path-walking": ("leaves", "wind", "room"),
    "home-warm": ("room", "water"), "study-calm": ("room",),
    "still-lake": ("water", "birds"), "effort-dawn": ("wind", "leaves"),
    "kind-warm": ("rain", "room"), "calm-warm": ("birds", "water"),
}


def plan_for(prompt_text="", mood_tag=""):
    """Decide which ambience layers to synthesise from the SFX prompt + mood."""
    hay = f"{prompt_text or ''} {mood_tag or ''}".lower()
    kinds = []
    if mood_tag and mood_tag in MOOD_MAP:
        kinds.extend(MOOD_MAP[mood_tag])
    for pat, ks in TEXT_MAP:
        if re.search(pat, hay):
            kinds.extend(ks)
    if not kinds:
        kinds = ["birds", "water"]      # the house default: peaceful morning
    out, level = [], 0.0
    for k in dict.fromkeys(kinds):      # de-dupe, keep order
        lvl = LAYERS.get(k, {}).get("level", 0.3)
        out.append((k, round(lvl, 3)))
        level += lvl
    return out


def _pink(n, rng, alpha=1.0):
    """Filtered noise (1/f^alpha-ish) — the base for wind/water/rain."""
    import numpy as np

    x = rng.normal(0.0, 1.0, n).astype(np.float32)
    # cheap smoothing: cascaded moving averages ≈ lowpass; alpha picks the cascade count
    for _ in range(int(round(1 + alpha * 2))):
        k = 9
        x = np.convolve(x, np.ones(k, dtype=np.float32) / k, mode="same")
    for _ in range(int(round(3 - alpha))):
        x = np.diff(x, prepend=x[0])
    peak = float(np.max(np.abs(x))) or 1.0
    return x / peak


def _envelope(n, attack=0.25, release=0.5, sr=SR):
    import numpy as np

    a = max(1, int(attack * sr))
    r = max(1, int(release * sr))
    env = np.ones(n, dtype=np.float32)
    if a + r >= n:
        return np.linspace(0, 1, n, dtype=np.float32) * np.linspace(1, 0, n, dtype=np.float32)
    env[:a] = np.linspace(0, 1, a, dtype=np.float32)
    env[-r:] = np.linspace(1, 0, r, dtype=np.float32)
    return env


def _birds(n, sr, rng, density=0.9):
    import numpy as np

    out = np.zeros(n, dtype=np.float32)
    gap = max(int(sr * 0.18), 1)
    pos = int(rng.integers(0, gap))
    while pos < n - gap:
        dur = float(rng.uniform(0.06, 0.22))
        m = int(dur * sr)
        if m < 8 or pos + m >= n:
            pos += int(rng.integers(gap, gap * 4))
            continue
        t = np.arange(m, dtype=np.float32) / sr
        f0 = float(rng.uniform(1900, 4600))
        sweep = float(rng.uniform(-1.4, 1.4)) * f0 * t
        frq = f0 + sweep + np.sin(2 * np.pi * rng.uniform(9, 26) * t) * f0 * 0.16
        phase = 2 * np.pi * np.cumsum(frq) / sr
        chirp = np.sin(phase) * np.hanning(m)
        # a second harmonic gives it body
        chirp = (chirp + 0.35 * np.sin(phase * 2.02) * np.hanning(m)) * 0.5
        out[pos:pos + m] += chirp * float(rng.uniform(0.35, 1.0))
        pos += m + int(rng.integers(int(gap / density), int(gap * 5 / density)))
    return np.clip(out, -1, 1)


def _water(n, sr, rng):
    import numpy as np

    x = _pink(n, rng, alpha=1.15)
    slow = np.sin(2 * np.pi * np.cumsum(np.linspace(0.05, 0.14, n)) / sr) * 0.18
    gurg = np.sin(2 * np.pi * np.cumsum(rng.uniform(280, 900, n)) / sr) * 0.06
    out = x * (0.6 + slow) + gurg * np.abs(x)
    return out.astype(np.float32)


def _wind(n, sr, rng, leaves=False):
    import numpy as np

    x = _pink(n, rng, alpha=1.6)
    t = np.arange(n, dtype=np.float32) / sr
    swell = 0.45 + 0.55 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.09 * t + float(rng.uniform(0, 6))))
    gust = (0.5 + 0.5 * np.sin(2 * np.pi * 0.37 * t + 1.1)) ** 2.0
    out = x * (0.35 + 0.65 * swell * gust)
    if leaves:
        ticks = np.zeros(n, dtype=np.float32)
        pos = int(rng.integers(0, int(sr * 0.05)))
        while pos < n:
            m = int(sr * float(rng.uniform(0.005, 0.02)))
            ticks[pos:pos + m] = rng.normal(0, 1, m).astype(np.float32) * 0.5
            pos += int(rng.integers(int(sr * 0.02), int(sr * 0.18)))
        ticks = np.convolve(ticks, np.array([0.4, 0.6], dtype=np.float32), mode="same")
        out = out * 0.75 + ticks * 0.35
    return out.astype(np.float32)


def _rain(n, sr, rng, on_roof=True):
    import numpy as np

    x = _pink(n, rng, alpha=0.55)
    t = np.arange(n, dtype=np.float32)
    hiss = 0.55 + 0.45 * np.sin(2 * np.pi * 0.13 * t / sr)
    out = x * hiss
    if on_roof:
        taps = np.zeros(n, dtype=np.float32)
        pos = int(rng.integers(0, sr // 10))
        while pos < n:
            m = int(sr * 0.02)
            if pos + m < n:
                tt = np.arange(m, dtype=np.float32) / sr
                f = float(rng.uniform(700, 2100))
                taps[pos:pos + m] += np.sin(2 * np.pi * f * tt) * np.exp(-tt * 180) * 0.35
            pos += int(rng.integers(int(sr * 0.012), int(sr * 0.14)))
        out = out * 0.7 + taps
    return out.astype(np.float32)


def _crickets(n, sr, rng):
    import numpy as np

    out = np.zeros(n, dtype=np.float32)
    t = np.arange(n, dtype=np.float32) / sr
    carrier = np.sin(2 * np.pi * float(rng.uniform(4200, 5200)) * t)
    rate = float(rng.uniform(22, 34))
    pulse = np.clip(np.sin(2 * np.pi * rate * t), 0.0, 1.0) ** 3.0
    spacing = (0.5 + 0.5 * np.sin(2 * np.pi * 0.31 * t))
    out = carrier * pulse * spacing * 0.6
    return out.astype(np.float32)


def _room(n, sr, rng):
    import numpy as np

    x = _pink(n, rng, alpha=1.9) * 0.35
    hum = np.sin(2 * np.pi * 50.0 * np.arange(n, dtype=np.float32) / sr) * 0.03
    return (x + hum).astype(np.float32)


def _village(n, sr, rng):
    import numpy as np

    out = _room(n, sr, rng) * 0.5
    # a couple of far, soft motorbike-ish passes + a distant chime
    for _ in range(int(rng.integers(1, 3))):
        pos = int(rng.integers(0, max(1, n - int(sr * 2))))
        m = min(int(sr * float(rng.uniform(1.2, 2.4))), n - pos)
        if m < 100:
            continue
        t = np.arange(m, dtype=np.float32) / sr
        f = 70 + 25 * np.sin(np.pi * t / max(1e-6, t[-1]))
        seg = np.sin(2 * np.pi * np.cumsum(f) / sr) * np.hanning(m) * 0.18
        out[pos:pos + m] += seg
    return out.astype(np.float32)


BUILDERS = {
    "birds": _birds, "water": _water, "wind": _wind, "leaves": lambda n, sr, rng: _wind(n, sr, rng, True),
    "rain": _rain, "night": _crickets, "room": _room, "village": _village,
}


def synthesize(dst, duration=6.0, prompt_text="", mood_tag="", seed=0, level=1.0, sr=SR,
               stereo=True):
    """Build an ambience bed. Returns {path, duration, layers, engine, sample_rate}."""
    import numpy as np

    ensure_dir(os.path.dirname(dst) or ".")
    duration = max(0.5, float(duration))
    n = int(round(duration * sr))
    rng = np.random.default_rng(abs(int(seed or 0)) % 999983 + 7)
    plan = plan_for(prompt_text, mood_tag)
    mix = np.zeros(n, dtype=np.float32)
    used = []
    for kind, w in plan:
        builder = BUILDERS.get(kind)
        if builder is None:
            continue
        try:
            x = builder(n, sr, rng)
        except Exception:
            continue
        x = np.asarray(x, dtype=np.float32)
        if x.shape[0] < n:
            x = np.concatenate([x, np.zeros(n - x.shape[0], dtype=np.float32)])
        x = x[:n] * float(w)
        mix += x * _envelope(n, attack=min(0.6, duration * 0.15), release=min(1.0, duration * 0.2), sr=sr)
        used.append(kind)
    if not used:
        mix = _room(n, sr, rng) * 0.3
    peak = float(np.max(np.abs(mix))) or 1.0
    mix = mix / peak * float(np.clip(0.55 * level, 0.05, 0.95))
    if stereo:
        # decorrelate the right channel slightly: width without phase weirdness
        d = int(sr * 0.011)
        right = np.concatenate([np.zeros(d, dtype=np.float32), mix[:-d]]) if d < n else mix
        two = np.stack([mix, right * 0.94 + mix * 0.06], axis=1)
    else:
        two = mix
    write_wav(dst, two, sr, channels=(2 if stereo else 1), normalize_to=0.55)
    return {"path": dst, "duration": round(duration, 3), "layers": used, "engine": "procedural",
            "sample_rate": sr, "stereo": bool(stereo)}

"""100% offline sound-effect library.

Every SFX is synthesized with numpy (no downloads, no assets, no network),
so sound design works even on a fully air-gapped machine. Users can also
drop their own .wav files into the SFX folder, which are picked up too.
"""
import os
import wave
import numpy as np

SR = 44100


def _write_wav(path, samples, sr=SR):
    samples = np.asarray(samples, dtype=np.float64)
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak > 0:
        samples = samples / peak * 0.9
    pcm = (samples * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def _t(n, sr=SR):
    return np.arange(n, dtype=np.float64) / sr


def _one_pole_filter(x, fc, sr=SR):
    """Simple one-pole lowpass; fc may be a per-sample array (sweep)."""
    r = 1.0 / (1.0 + np.exp(-2 * np.pi * fc / sr))
    r = np.asarray(r, dtype=np.float64)
    out = np.empty_like(x)
    y = 0.0
    for i in range(len(x)):
        y = r[i] * x[i] + (1.0 - r[i]) * y
        out[i] = y
    return out


def sfx_whoosh(dur=0.9, sr=SR):
    n = int(dur * sr)
    rng = np.random.default_rng(7)
    noise = rng.uniform(-1, 1, n)
    fc = 900 * (1 - _t(n) / dur) + 120  # sweep 1020 -> 120 Hz
    x = _one_pole_filter(noise, fc, sr)
    env = np.sin(np.pi * np.linspace(0, 1, n)) ** 0.7
    return x * env


def sfx_pop(dur=0.14, sr=SR):
    n = int(dur * sr)
    t = _t(n, sr)
    f = 480 * np.exp(-t * 30) + 90
    phase = 2 * np.pi * np.cumsum(f) / sr
    env = np.exp(-t * 28)
    return np.sin(phase) * env


def sfx_ding(dur=0.9, sr=SR):
    n = int(dur * sr)
    t = _t(n, sr)
    env = np.exp(-t * 5.0)
    return (np.sin(2 * np.pi * 880 * t) * 0.7 + np.sin(2 * np.pi * 1318.5 * t) * 0.3) * env


def sfx_click(dur=0.05, sr=SR):
    n = int(dur * sr)
    rng = np.random.default_rng(11)
    x = rng.uniform(-1, 1, n)
    x = np.diff(x, prepend=0.0)  # crude highpass
    return x * np.exp(-_t(n, sr) * 120)


def sfx_riser(dur=1.5, sr=SR):
    n = int(dur * sr)
    t = _t(n, sr)
    p = t / dur
    f = 200 + 1200 * p ** 1.6
    phase = 2 * np.pi * np.cumsum(f) / sr
    tone = np.sin(phase) * 0.6
    rng = np.random.default_rng(13)
    noise = rng.uniform(-1, 1, n) * (0.2 + 0.8 * p)
    return (tone + noise * 0.5) * (0.25 + 0.75 * p)


def sfx_boom(dur=1.1, sr=SR):
    n = int(dur * sr)
    t = _t(n, sr)
    tone = np.sin(2 * np.pi * (55 - 20 * t) * t) * np.exp(-t * 3.2)
    rng = np.random.default_rng(17)
    thump = rng.uniform(-1, 1, n) * np.exp(-t * 14) * 0.7
    return tone * 0.9 + thump


def sfx_applause(dur=2.2, sr=SR):
    n = int(dur * sr)
    rng = np.random.default_rng(19)
    t = _t(n, sr)
    attack = np.clip(t / 0.25, 0, 1)
    decay = np.exp(-np.clip(t - 0.35, 0, None) * 0.9)
    env = attack * (1 - 0.7 * decay)
    # random clap amplitude modulation (8-14 Hz irregular)
    rate = rng.uniform(8, 14, n // 64 + 1)
    mod = np.ones(n)
    i = 0
    while i < n:
        seg = min(64, n - i)
        lvl = rng.uniform(0.35, 1.0)
        mod[i:i + seg] = lvl
        i += seg
    x = rng.uniform(-1, 1, n)
    return x * env * mod


def sfx_sparkle(dur=0.8, sr=SR):
    n = int(dur * sr)
    out = np.zeros(n)
    for k, f in enumerate((1568.0, 1244.5, 987.8, 1568.0)):
        start = int(k * 0.11 * sr)
        m = min(n - start, int(0.5 * sr))
        if m <= 0:
            continue
        t = _t(m, sr)
        out[start:start + m] += np.sin(2 * np.pi * f * t) * np.exp(-t * 6) * 0.6
    return out


def sfx_typing(dur=0.8, sr=SR):
    n = int(dur * sr)
    out = np.zeros(n)
    rng = np.random.default_rng(23)
    i = 0
    while i < n - int(0.02 * sr):
        gap = int(rng.uniform(0.05, 0.12) * sr)
        i += gap
        m = min(int(0.018 * sr), n - i)
        t = _t(m, sr)
        out[i:i + m] += rng.uniform(0.5, 1.0) * np.sin(2 * np.pi * rng.uniform(1800, 3200) * t) * np.exp(-t * 220)
    return out


SFX_LIBRARY = {
    "whoosh": {"make": sfx_whoosh, "desc": "Fast air sweep — entrances & scene whooshes"},
    "pop": {"make": sfx_pop, "desc": "Snappy pop — element appears"},
    "ding": {"make": sfx_ding, "desc": "Bright notification chime — good idea / payoff"},
    "click": {"make": sfx_click, "desc": "Tight UI click — quick cuts"},
    "riser": {"make": sfx_riser, "desc": "Building tension sweep — before a reveal"},
    "boom": {"make": sfx_boom, "desc": "Deep impact — big reveals / emphasis"},
    "applause": {"make": sfx_applause, "desc": "Crowd clap — endings & CTA"},
    "sparkle": {"make": sfx_sparkle, "desc": "Magical arpeggio — tips & tricks"},
    "typing": {"make": sfx_typing, "desc": "Keyboard taps — code / list moments"},
}


def ensure_library(sfx_dir):
    """Materializes all built-in SFX as .wav files (idempotent)."""
    os.makedirs(sfx_dir, exist_ok=True)
    names = []
    for name, info in SFX_LIBRARY.items():
        path = os.path.join(sfx_dir, f"{name}.wav")
        if not os.path.exists(path):
            _write_wav(path, info["make"]())
        names.append(name)
    # user-supplied wavs are picked up automatically
    for fn in sorted(os.listdir(sfx_dir)):
        if fn.lower().endswith(".wav"):
            stem = os.path.splitext(fn)[0]
            if stem not in names:
                names.append(stem)
    return names


def list_sfx(sfx_dir):
    ensure_library(sfx_dir)
    out = []
    for fn in sorted(os.listdir(sfx_dir)):
        if fn.lower().endswith(".wav"):
            stem = os.path.splitext(fn)[0]
            out.append({
                "name": stem,
                "desc": SFX_LIBRARY.get(stem, {}).get("desc", "Custom sound effect"),
                "builtin": stem in SFX_LIBRARY,
            })
    return out


def load_sfx(sfx_dir, name):
    """Returns (float32 mono samples, sample_rate) or (None, 0)."""
    ensure_library(sfx_dir)
    path = os.path.join(sfx_dir, f"{name}.wav")
    if not os.path.exists(path):
        return None, 0
    try:
        with wave.open(path, "rb") as wf:
            sr = wf.getframerate()
            n = wf.getnframes()
            raw = wf.readframes(n)
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if wf.getnchannels() > 1:
            data = data.reshape(-1, wf.getnchannels()).mean(axis=1)
        return data, sr
    except Exception as e:
        print(f"Failed to load SFX {name}: {e}")
        return None, 0

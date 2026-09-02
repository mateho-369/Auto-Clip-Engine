"""Small shared helpers: ids, time, JSON, wav IO, ffmpeg discovery, paths.

Deliberately dependency-light (stdlib + numpy) so the studio core installs on a
bare Windows box next to the existing ``ai_creator`` stack.
"""
import io
import json
import math
import os
import re
import shutil
import subprocess
import time
import uuid
import wave


# ---------------------------------------------------------------- ids / time
def new_id(n=8):
    return uuid.uuid4().hex[:n]


def now():
    return time.time()


def iso(ts=None):
    ts = time.time() if ts is None else ts
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts))


def fmt_dur(sec):
    """127.4 -> '2m 07s', 42.0 -> '42s'."""
    try:
        sec = float(sec)
    except Exception:
        return "—"
    if sec < 60:
        return f"{sec:.0f}s"
    m, s = divmod(int(round(sec)), 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


def elapsed_str(start, end=None):
    return fmt_dur((time.time() if end is None else end) - start)


# ---------------------------------------------------------------- JSON
def jdump(obj):
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        return json.dumps({"unserializable": str(obj)[:2000]})


def jload(text, default=None):
    if not text:
        return default
    try:
        return json.loads(text)
    except Exception:
        return default


def read_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def write_json(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
    return path


# ---------------------------------------------------------------- paths
_SAFE = re.compile(r"[^A-Za-z0-9._\-]+")


def safe_name(name, maxlen=80):
    name = _SAFE.sub("_", str(name or "").strip()).strip("_") or "untitled"
    return name[:maxlen]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def rel(path, root):
    """POSIX-style path relative to root (used for asset URLs)."""
    try:
        return os.path.relpath(path, root).replace(os.sep, "/")
    except Exception:
        return os.path.basename(path)


def human_size(n):
    try:
        n = float(n)
    except Exception:
        return "—"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}GB"


# ---------------------------------------------------------------- audio
def write_wav(path, samples, sr=44100, channels=1, normalize_to=0.92):
    """Write float (-1..1) samples as 16-bit PCM wav, peak-normalised."""
    import numpy as np

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    x = np.asarray(samples, dtype=np.float32)
    if x.ndim == 1 and channels > 1:
        x = np.repeat(x[:, None], channels, axis=1)
    if x.ndim == 2 and channels == 1:
        x = x.mean(axis=1)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 1e-6:
        x = x / peak * float(normalize_to)
    pcm = (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(int(channels))
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(pcm.tobytes())
    return path


def read_wav(path, mono=True, target_sr=None):
    """Read a wav into (float32 samples in -1..1, samplerate).

    Falls back to ffmpeg decoding for non-wav/odd input, so callers can treat
    any media file as audio.
    """
    import numpy as np

    try:
        with wave.open(path, "rb") as wf:
            sr = wf.getframerate()
            n = wf.getnframes()
            sw = wf.getsampwidth()
            ch = wf.getnchannels()
            raw = wf.readframes(n)
        dtype = {1: np.uint8, 2: np.int16, 4: np.int32}.get(sw, np.int16)
        data = np.frombuffer(raw, dtype=dtype).astype(np.float32)
        if sw == 1:
            data = (data - 128.0) / 128.0
        else:
            data = data / float(2 ** (8 * sw - 1))
        if ch > 1 and mono:
            data = data.reshape(-1, ch).mean(axis=1)
        if target_sr and sr != target_sr:
            data = resample_linear(data, sr, target_sr)
            sr = target_sr
        return np.ascontiguousarray(data, dtype=np.float32), sr
    except Exception:
        pass
    # any other container/codec -> decode to a temp wav with ffmpeg
    ff = ffmpeg_exe()
    if not ff:
        raise ValueError(f"cannot decode audio (no ffmpeg): {path}")
    tmp = os.path.join(os.path.dirname(path) or ".", f".dec_{new_id(4)}.wav")
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", path, "-vn", "-ac", "1",
                    "-ar", str(target_sr or 44100), "-c:a", "pcm_s16le", tmp],
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
    try:
        out = read_wav(tmp, mono=mono, target_sr=target_sr)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return out


def resample_linear(x, src_sr, dst_sr):
    import numpy as np

    x = np.asarray(x, dtype=np.float32)
    if src_sr == dst_sr or x.size == 0:
        return x
    n_out = max(1, int(round(x.shape[-1] * dst_sr / float(src_sr))))
    pos = np.linspace(0, x.shape[-1] - 1, n_out)
    i0 = np.floor(pos).astype(np.int64)
    i1 = np.minimum(i0 + 1, x.shape[-1] - 1)
    frac = (pos - i0).astype(np.float32)
    return (x[i0] * (1 - frac) + x[i1] * frac).astype(np.float32)


def wav_duration(path, default=0.0):
    try:
        with wave.open(path, "rb") as wf:
            return wf.getnframes() / float(wf.getframerate())
    except Exception:
        return default


def rms(x):
    import numpy as np

    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return 0.0
    return float(math.sqrt(float(np.mean(x * x))))


def dbfs(x):
    v = rms(x)
    return -120.0 if v < 1e-7 else 20.0 * math.log10(v)


def wav_peaks(path, bins=400):
    """Normalised 0..1 envelope of a wav — the UI draws a mini waveform."""
    import numpy as np

    try:
        x, _ = read_wav(path)
    except Exception:
        return []
    if x.size == 0:
        return []
    chunk = max(1, x.size // max(1, int(bins)))
    n = int(math.ceil(x.size / chunk))
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        out[i] = rms(x[i * chunk:(i + 1) * chunk])
    peak = float(out.max()) if out.size else 0.0
    if peak > 0:
        out = out / peak
    return [round(float(v), 3) for v in out]


def leading_trailing_silence(path, thresh=0.02, frame=0.02):
    """(head_s, tail_s) of silence below `thresh` RMS — used by QA."""
    import numpy as np

    try:
        x, sr = read_wav(path)
    except Exception:
        return 0.0, 0.0
    if x.size == 0:
        return 0.0, 0.0
    win = max(1, int(sr * frame))
    n = int(math.ceil(x.size / win))
    env = np.array([rms(x[i * win:(i + 1) * win]) for i in range(n)], dtype=np.float32)
    loud = np.where(env > thresh)[0]
    if loud.size == 0:
        return float(n) * frame, float(n) * frame
    return float(loud[0]) * frame, float(n - 1 - loud[-1]) * frame


def has_audio_stream(path):
    """Cheap check used before muxing: does the file carry audio already?"""
    ff = ffmpeg_exe()
    if not ff:
        return False
    try:
        res = subprocess.run([ff, "-hide_banner", "-i", path], capture_output=True, timeout=30)
        return b"Audio:" in (res.stderr or b"")
    except Exception:
        return False


# ---------------------------------------------------------------- ffmpeg
_FF_EXE = None


def ffmpeg_exe():
    """PATH ffmpeg, else the imageio-ffmpeg bundled binary, else None."""
    global _FF_EXE
    if _FF_EXE:
        return _FF_EXE
    exe = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
    if not exe:
        try:
            import imageio_ffmpeg

            exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            exe = None
    _FF_EXE = exe
    return exe


def ffprobe_exe():
    exe = shutil.which("ffprobe") or shutil.which("ffprobe.exe")
    if exe:
        return exe
    ff = ffmpeg_exe()
    if ff and "imageio" in ff:  # imageio bundle ships ffmpeg only
        cand = os.path.join(os.path.dirname(ff), "ffprobe")
        return cand if os.path.exists(cand) else None
    return shutil.which("ffprobe")


def run_ffmpeg(args, timeout=1800, quiet=True):
    ff = ffmpeg_exe()
    if not ff:
        raise RuntimeError("ffmpeg not found — install it (scoop install ffmpeg) or pip install imageio-ffmpeg")
    cmd = [ff, "-y", "-hide_banner", "-loglevel", "error" if quiet else "info"] + list(args)
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    if res.returncode != 0:
        err = (res.stderr or b"").decode(errors="ignore").strip()
        raise RuntimeError(f"ffmpeg failed ({res.returncode}): {err[-800:] or 'no stderr'}")
    return res.stdout or b""


def media_duration(path, default=0.0):
    """Seconds for wav/mp4/mp3 — ffprobe when present, else parse ffmpeg."""
    if not path or not os.path.exists(path):
        return default
    if path.lower().endswith(".wav"):
        d = wav_duration(path, default=-1)
        if d and d > 0:
            return d
    probe = ffprobe_exe()
    if probe:
        try:
            res = subprocess.run([probe, "-v", "error", "-show_entries", "format=duration",
                                  "-of", "csv=p=0", path], capture_output=True, timeout=60)
            txt = res.stdout.decode(errors="ignore").strip()
            if txt:
                return float(txt.splitlines()[-1])
        except Exception:
            pass
    ff = ffmpeg_exe()
    if ff:
        try:
            res = subprocess.run([ff, "-hide_banner", "-i", path], capture_output=True, timeout=60)
            m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)",
                          (res.stderr or b"").decode(errors="ignore"))
            if m:
                return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
        except Exception:
            pass
    return default


def clamp(v, lo, hi):
    try:
        v = float(v)
    except Exception:
        return lo
    return max(lo, min(hi, v))


def pct(done, total):
    if not total:
        return 0.0
    return round(clamp(100.0 * done / total, 0.0, 100.0), 1)


def scrub(obj, maxlen=2000):
    """Truncate huge prompt/response blobs before storing them in the DB."""
    s = obj if isinstance(obj, str) else str(obj)
    return s if len(s) <= maxlen else s[:maxlen] + f"\n…[truncated {len(s) - maxlen} chars]"


def zip_paths(out_zip, paths, arc_root="", skip_internal=True):
    """Zip files/dirs (used by 'download everything for this scene/project').

    `skip_internal` drops dot-prefixed names — the studio keeps transient work
    (assembly concat/duck files, `.part` downloads) in `.folders` inside the
    project dir, and those are not part of "download all intermediates".
    """
    import zipfile

    ensure_dir(os.path.dirname(out_zip) or ".")
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in paths:
            if not p or not os.path.exists(p):
                continue
            if os.path.isdir(p):
                for root, dirs, files in os.walk(p):
                    if skip_internal:
                        dirs[:] = [d for d in dirs if not d.startswith(".")]
                        files = [f for f in files if not f.startswith(".")]
                    for fn in sorted(files):
                        full = os.path.join(root, fn)
                        arc = os.path.join(arc_root, os.path.relpath(full, p))
                        zf.write(full, arc)
            else:
                zf.write(p, os.path.join(arc_root, os.path.basename(p)) if arc_root
                         else os.path.basename(p))
    return out_zip

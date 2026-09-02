"""ffmpeg media operations — duration fitting, mixing, concat, mux, thumbs.

The brief's Stage 7 is "ffmpeg/MoviePy (already in the existing stack)". We use
ffmpeg directly: it is one less heavy import, it is what moviepy shells out to
anyway, and it keeps precise control over the thing that actually matters here —
**every scene clip must match its voice duration**, and the ambience must sit
*under* the narration rather than next to it.

Audio mixing is done with numpy (exact ducking/fades, no fragile filter graphs),
video work is done with ffmpeg (encode/concat/mux/thumbnails).
"""
import math
import os
import re
import shutil
import subprocess
import threading

from .util import (ensure_dir, ffmpeg_exe, media_duration, read_wav, rel, run_ffmpeg,
                   wav_duration, write_wav)

SR = 44100


def probe(path):
    """{duration, width, height, fps} for a video file — best effort."""
    out = {"duration": media_duration(path, 0.0), "width": 0, "height": 0, "fps": 0.0}
    ff = ffmpeg_exe()
    if not ff or not path or not os.path.exists(path):
        return out
    try:
        res = subprocess.run([ff, "-hide_banner", "-i", path], capture_output=True, timeout=60)
        txt = (res.stderr or b"").decode(errors="ignore")
        m = re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", txt)
        if m:
            out["width"], out["height"] = int(m.group(1)), int(m.group(2))
        m = re.search(r"([\d.]+)\s*fps", txt)
        if m:
            out["fps"] = float(m.group(1))
    except Exception:
        pass
    return out


# ------------------------------------------------------------------- fitting
def fit_audio(src, dst, target_sec, mode="pad", sr=SR, allow_rate=True):
    """Make an audio file exactly `target_sec` long.

    mode: 'trim' (cut), 'pad' (silence tail), 'fit' (trim/pad), 'rate' (gentle
    atempo correction inside ±8% before padding — keeps lip-ish sync without
    sounding chipmunked).
    """
    x, s = read_wav(src, mono=True, target_sr=sr)
    need = int(round(max(0.02, float(target_sec)) * sr))
    have = x.shape[0]
    if have > need:
        if mode == "rate" and allow_rate and need / have > 0.92:
            factor = have / float(need)
            x = _timestretch(x, 1.0 / factor)
            have = x.shape[0]
        if have > need:
            x = x[:need]
    elif have < need:
        if mode == "rate" and allow_rate and have / need > 0.92:
            x = _timestretch(x, need / float(have))
            have = x.shape[0]
        if have < need:
            x = numpy_pad(x, need - have)
    write_wav(dst, x[:need] if x.shape[0] > need else _pad_to(x, need), sr)
    return {"duration": need / float(sr), "src_duration": have / float(sr)}


def _pad_to(x, n):
    import numpy as np
    if x.shape[0] >= n:
        return x
    return np.concatenate([x, np.zeros(n - x.shape[0], dtype=np.float32)])


def numpy_pad(x, n):
    import numpy as np
    return np.concatenate([np.asarray(x, dtype=np.float32), np.zeros(int(n), dtype=np.float32)])


def _timestretch(x, rate, sr=SR):
    """WSOLA-free, good-enough resample stretch (voice only, ±8%)."""
    import numpy as np

    rate = max(0.85, min(1.18, float(rate)))
    if abs(rate - 1.0) < 1e-3:
        return x
    n_out = max(1, int(x.shape[0] * rate))
    pos = np.linspace(0, x.shape[0] - 1, n_out)
    i0 = np.floor(pos).astype(np.int64)
    i1 = np.minimum(i0 + 1, x.shape[0] - 1)
    f = (pos - i0).astype(np.float32)
    return (x[i0] * (1 - f) + x[i1] * f).astype(np.float32)


def fit_video(src, dst, target_sec, width=0, height=0, fps=24, mode="auto",
              freeze_tail=True, fade=0.0):
    """Duration-match a silent clip to the voice.

    auto = trim if longer, freeze-last-frame if shorter (never loops: a looping
    background reads as a glitch in calm content). Optional re-scale/fps normalise
    so every scene segment is concat-compatible.
    """
    src_dur = media_duration(src, 0.0)
    target = max(0.5, float(target_sec))
    args = ["-i", src]
    vf = []
    if width and height:
        vf.append(f"scale={int(width)}:{int(height)}:force_original_aspect_ratio=increase")
        vf.append(f"crop={int(width)}:{int(height)}")
    if fps:
        vf.append(f"fps={int(fps)}")
    shorter = target > src_dur + 0.06
    if shorter and freeze_tail:
        vf.append("tpad=stop_mode=clone:stop_duration=%.3f" % (target - src_dur))
    if fade:
        vf.append(f"fade=t=in:st=0:d={fade:.2f}")
        vf.append(f"fade=t=out:st={max(0.0, target - fade):.2f}:d={fade:.2f}")
    if vf:
        args += ["-vf", ",".join(vf)]
    args += ["-t", f"{target:.3f}", "-an", "-c:v", "libx264", "-preset", "veryfast",
             "-crf", "22", "-pix_fmt", "yuv420p", "-r", str(int(fps or 24)), dst]
    run_ffmpeg(args, timeout=1800)
    return {"duration": media_duration(dst, target), "src_duration": src_dur,
            "trimmed": src_dur > target + 0.06, "froze_tail": shorter and freeze_tail}


# ------------------------------------------------------------------- mixing
def mix_audio(tracks, dst, total_sec=None, sr=SR, normalize_to=0.92, duck=None, max_total_sec=1800.0):
    """Mix `tracks` = [{path, gain, delay, fade_in, fade_out, is_voice}] into dst.

    `duck` lowers everything that is *not* the voice while the voice speaks
    (windowed RMS gate, smoothed so the ambience glides instead of pumping) —
    that one behaviour is what makes calm narration sound produced rather than
    two files stacked on top of each other.
    """
    import numpy as np

    total = float(total_sec or 0)
    for t in tracks:
        if not t.get("path"):
            continue
        d = media_duration(t["path"], 0.0)
        total = max(total, d + float(t.get("delay", 0.0)))
    total = max(0.2, min(float(max_total_sec), total))       # never allocate a monster
    n = int(total * sr)
    mix = np.zeros(n, dtype=np.float32)
    voice = np.zeros(n, dtype=np.float32)
    for t in tracks:
        path = t.get("path")
        if not path or not os.path.exists(path):
            continue
        x, s = read_wav(path, mono=True, target_sr=sr)
        off = int(float(t.get("delay", 0.0)) * sr)
        if off >= n:
            continue
        end = min(n, off + x.shape[0])
        seg = np.asarray(x[: end - off], dtype=np.float32)
        fade_in = min(seg.shape[0], int(float(t.get("fade_in", 0.02)) * sr))
        fade_out = min(seg.shape[0], int(float(t.get("fade_out", 0.05)) * sr))
        if fade_in > 1:
            seg = seg.copy()
            seg[:fade_in] *= np.linspace(0, 1, fade_in, dtype=np.float32)
        if fade_out > 1:
            seg = seg.copy()
            seg[-fade_out:] *= np.linspace(1, 0, fade_out, dtype=np.float32)
        mix[off:end] += seg * float(t.get("gain", 1.0))
        if t.get("is_voice"):
            voice[off:end] = np.maximum(voice[off:end], seg)
    if duck and float(np.max(np.abs(voice))) > 1e-5:
        win = max(1, int(sr * float(duck.get("window", 0.03))))
        nw = int(math.ceil(n / win))
        pad = max(1, int(float(duck.get("pad", 0.25)) * sr / win))
        voiced = np.zeros(nw, dtype=bool)
        for i in range(nw):
            lo, hi = i * win, min(n, (i + 1) * win)
            if hi > lo and float(np.max(np.abs(voice[lo:hi]))) > float(duck.get("threshold", 0.02)):
                voiced[i] = True
        if voiced.any():
            gate = np.ones(nw, dtype=np.float32)
            idx = np.where(voiced)[0]
            lo, hi = max(0, int(idx[0]) - pad), min(nw, int(idx[-1]) + pad + 1)
            gate[lo:hi] = float(duck.get("gain", 0.35))       # one smooth ramp across the
            k = max(3, int(0.15 * sr / win))                   # narration block, not per word
            ramp = np.hanning(k)
            ramp = ramp / ramp.sum()
            gate = np.convolve(gate, ramp, mode="same").astype(np.float32)
            gate = np.clip(gate, float(duck.get("gain", 0.35)), 1.0)
            mix *= np.repeat(gate, win)[:n]
    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak > 1e-6:
        mix = mix / peak * float(normalize_to)
    write_wav(dst, mix, sr, normalize_to=normalize_to)
    return {"duration": n / float(sr), "path": dst, "peak": peak, "tracks": len(tracks)}


def to_stereo(src_wav, dst_wav, width=0.35):
    """Tiny stereo decorrelation so ambience doesn't sound 'mono phone'."""
    import numpy as np

    x, sr = read_wav(src_wav, mono=False, target_sr=SR)
    if x.ndim == 1:
        x = np.stack([x, x], axis=1)
    h = int(sr * width / 2)
    left = x[:, 0].copy()
    right = x[:, 1].copy() if x.shape[1] > 1 else x[:, 0].copy()
    if h > 1 and left.shape[0] > h * 2:
        d = np.zeros_like(left)
        d[h:] = left[:-h] * 0.35
        right = np.clip(right + d, -1, 1)
    os.makedirs(os.path.dirname(dst_wav) or ".", exist_ok=True)
    write_wav(dst_wav, np.stack([left, right], axis=1), SR, channels=2, normalize_to=0.9)
    return dst_wav


def loudnorm(src, dst, target_lufs=-16.0):
    """Single-pass EBU R128-ish normalisation; falls back to peak gain."""
    try:
        run_ffmpeg(["-i", src, "-af", f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11",
                    "-c:a", "pcm_s16le", dst], timeout=900)
        if os.path.exists(dst) and wav_duration(dst, 0) > 0.1:
            return dst
    except Exception:
        pass
    x, sr = read_wav(src, mono=False, target_sr=SR)
    peak = float(max(1e-6, np_max(abs(x))))
    write_wav(dst, x * (0.9 / peak), sr, channels=(2 if x.ndim > 1 else 1), normalize_to=0.9)
    return dst


def np_max(x):
    import numpy as np
    return float(np.max(x)) if x.size else 0.0


# ------------------------------------------------------------- concat / mux
def normalize_clip(src, dst, width, height, fps, duration=None, silent=True):
    vf = (f"scale={int(width)}:{int(height)}:force_original_aspect_ratio=decrease,"
          f"pad={int(width)}:{int(height)}:(ow-iw)/2:(oh-ih)/2:color=black,format=yuv420p")
    args = ["-i", src, "-vf", vf, "-r", str(int(fps)), "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "21", "-pix_fmt", "yuv420p"]
    if silent:
        args += ["-an"]
    if duration:
        args += ["-t", f"{float(duration):.3f}"]
    args += [dst]
    run_ffmpeg(args, timeout=1800)
    return dst


def concat_clips(clips, dst, fps=24, transition="cut", fade=0.0, work_dir=None):
    """Concatenate video-only clips. `transition='crossfade'` uses xfade when the
    installed ffmpeg supports it, else falls back to a hard cut (never fails)."""
    clips = [c for c in clips if c and os.path.exists(c)]
    if not clips:
        raise RuntimeError("no clips to concatenate")
    if len(clips) == 1:
        shutil.copyfile(clips[0], dst)
        return dst
    if transition == "crossfade" and fade and fade > 0.02 and len(clips) > 1 and _has_filter("xfade"):
        try:
            return _concat_xfade(clips, dst, fade)
        except Exception:
            pass
    work = ensure_dir(work_dir or (os.path.dirname(dst) + "/.concat"))
    listing = os.path.join(work, "list.txt")
    with open(listing, "w", encoding="utf-8") as f:
        for c in clips:
            f.write("file '" + os.path.abspath(c).replace("'", "'\\''") + "'\n")
    run_ffmpeg(["-f", "concat", "-safe", "0", "-i", listing, "-c", "copy", "-movflags", "+faststart", dst],
               timeout=1800)
    if not os.path.exists(dst) or os.path.getsize(dst) < 1024:
        run_ffmpeg(["-f", "concat", "-safe", "0", "-i", listing, "-r", str(int(fps)),
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p", dst],
                   timeout=1800)
    return dst


def _concat_xfade(clips, dst, fade):
    inputs, filters = [], []
    for i, c in enumerate(clips):
        inputs += ["-i", c]
    prev = "[0:v]"
    durs = [media_duration(c, 4.0) for c in clips]
    for i in range(1, len(clips)):
        off = max(0.0, sum(durs[:i]) - fade * i)
        out = f"[v{i}]"
        filters.append(f"{prev}[{i}:v]xfade=transition=fade:duration={fade:.2f}:offset={off:.2f}{out}")
        prev = out
    args = inputs + ["-filter_complex", ";".join(filters), "-map", prev, "-an",
                     "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p", dst]
    run_ffmpeg(args, timeout=2400)
    return dst


_FILTERS_CACHE = {}


def _has_filter(name):
    if name in _FILTERS_CACHE:
        return _FILTERS_CACHE[name]
    ok = False
    ff = ffmpeg_exe()
    if ff:
        try:
            res = subprocess.run([ff, "-hide_banner", "-filters"], capture_output=True, timeout=60)
            ok = (f" {name} ".encode() in (res.stdout or b"")) or (f"{name}          ".encode()
                                                                    in (res.stdout or b""))
        except Exception:
            ok = False
    _FILTERS_CACHE[name] = ok
    return ok


def mux(video, audio, dst, crf=23, preset="veryfast", audio_kbps=160, faststart=True,
        video_codec="libx264", max_dur=None, extra=None):
    """Mux a silent video + a mixed audio track into the final MP4."""
    args = ["-i", video, "-i", audio, "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", video_codec, "-preset", preset, "-crf", str(int(crf)), "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", f"{int(audio_kbps)}k", "-shortest"]
    if max_dur:
        args += ["-t", f"{float(max_dur):.3f}"]
    if faststart:
        args += ["-movflags", "+faststart"]
    if extra:
        args += list(extra)
    args += [dst]
    run_ffmpeg(args, timeout=3600)
    return dst


def make_silent_video_from_image(image, dst, duration=6.0, width=480, height=854, fps=24,
                                 zoom=0.06, motion="kenburns"):
    """Still image → gentle Ken Burns clip (the deep fallback when nothing else works)."""
    d = max(1.0, float(duration))
    if motion == "kenburns" and _has_filter("zoompan"):
        vf = (f"scale={int(width * 1.25)}:-2,zoompan=z='min(zoom+0.0008,1.08)':x='iw/2-(iw/zoom/2)':"
              f"y='ih/2-(ih/zoom/2)':d={int(d * fps)}:s={int(width)}x{int(height)}:fps={int(fps)}")
    else:
        vf = f"scale={int(width)}:{int(height)}:force_original_aspect_ratio=decrease," \
             f"pad={int(width)}:{int(height)}:(ow-iw)/2:(oh-ih)/2,format=yuv420p"
    args = ["-loop", "1", "-framerate", str(int(fps)), "-t", f"{d:.2f}", "-i", image,
            "-vf", vf, "-r", str(int(fps)), "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
            "-pix_fmt", "yuv420p", "-an", "-t", f"{d:.2f}", dst]
    run_ffmpeg(args, timeout=1800)
    return dst


def thumbnail(video, dst_png, at_sec=0.6, width=320):
    try:
        run_ffmpeg(["-ss", f"{float(at_sec):.2f}", "-i", video, "-frames:v", "1",
                    "-vf", f"scale={int(width)}:-2", dst_png], timeout=300)
        return dst_png if os.path.exists(dst_png) else None
    except Exception:
        return None


def burn_subtitles(video, srt, dst, force_style="FontName=Noto Sans Khmer,FontSize=18,PrimaryColour=&H00FFFFFF,OutlineColour=&H60000000,BorderStyle=1,Outline=2,Shadow=1,MarginV=48"):
    """Only used when assembly.burn_captions is on and libass is available."""
    if not _has_filter("subtitles"):
        raise RuntimeError("this ffmpeg build has no 'subtitles' filter (needs libass) — "
                           "keep SRT as a sidecar file instead")
    srt_esc = str(srt).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")
    run_ffmpeg(["-i", video, "-vf", f"subtitles='{srt_esc}':force_style='{force_style}'",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-c:a", "copy", dst],
               timeout=3600)
    return dst


def write_srt(scene_texts, scene_starts, dst, words_per_line=6):
    """Khmer-safe SRT: whole sentence per scene (word timing on Khmer is unreliable
    because it has no spaces, so we keep subtitle granularity = scene)."""
    def fmt(sec):
        sec = max(0.0, float(sec))
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = sec % 60
        return f"{h:02d}:{m:02d}:{int(s):02d},{int(round((s % 1) * 1000)):03d}"

    lines = []
    for i, (txt, start) in enumerate(zip(scene_texts, scene_starts)):
        end = scene_starts[i + 1] if i + 1 < len(scene_starts) else start + 3.0
        lines.append(f"{i + 1}\n{fmt(start)} --> {fmt(max(end, start + 0.8))}\n{txt}\n")
    ensure_dir(os.path.dirname(dst) or ".")
    with open(dst, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return dst


def extract_audio(video, dst_wav, sr=SR):
    tmp = dst_wav + ".raw.wav"
    run_ffmpeg(["-i", video, "-vn", "-ac", "1", "-ar", str(sr), "-c:a", "pcm_s16le", tmp], timeout=900)
    shutil.move(tmp, dst_wav)
    return dst_wav


class FFLock:
    """Serialize ffmpeg launches so two stages can't spawn 8 encodes at once on a
    16GB laptop. Cheap, and it keeps RAM/threads predictable."""

    def __init__(self, n=2):
        self._sem = threading.Semaphore(max(1, int(n)))

    def __enter__(self):
        self._sem.acquire()
        return self

    def __exit__(self, *a):
        self._sem.release()
        return False


def asset_url(path, data_root):
    return "/assets-file/" + rel(path, data_root)

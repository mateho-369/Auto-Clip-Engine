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
import unicodedata

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
def normalize_clip(src, dst, width, height, fps, duration=None, silent=True, tail_pad=0.0):
    vf = (f"scale={int(width)}:{int(height)}:force_original_aspect_ratio=decrease,"
          f"pad={int(width)}:{int(height)}:(ow-iw)/2:(oh-ih)/2:color=black,format=yuv420p")
    tail = max(0.0, float(tail_pad or 0))
    if tail > 0.02:
        # deterministic pause between lines: freeze the last frame (never loops,
        # never black) — assembly uses tts.line_gap_sec for this
        vf += f",tpad=stop_mode=clone:stop_duration={tail:.3f}"
    args = ["-i", src, "-vf", vf, "-r", str(int(fps)), "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "21", "-pix_fmt", "yuv420p"]
    if silent:
        args += ["-an"]
    if duration:
        args += ["-t", f"{float(duration) + tail:.3f}"]
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


# ------------------------------------------------------ subtitle / title styles
# Style library replacing the single hardcoded force_style string. Each entry
# is the full libass inline style (or karaoke metadata) the frontend's Style
# Gallery shows by *preview*, never by name alone.
SUBTITLE_STYLE_KEYS = ("clean", "bold_yellow", "minimal_top", "karaoke")
_SUB_BASE = "FontName=Khmer OS Battambang,FontSize=15,PrimaryColour=&H00FFFFFF," \
            "OutlineColour=&HC0000000,BorderStyle=1,Outline=2,Shadow=0,MarginV=56," \
            "MarginL=28,MarginR=28,Alignment=2"
SUBTITLE_STYLES = {
    "clean": {"label": "Clean", "desc": "Today's look — white, centred, soft outline.",
              "force_style": _SUB_BASE, "karaoke": False},
    "bold_yellow": {"label": "Bold yellow",
                    "desc": "High-contrast bold yellow — readable over bright b-roll.",
                    "force_style": "FontName=Khmer OS Battambang,FontSize=17,"
                                   "PrimaryColour=&H0000FFFF,Bold=1,"
                                   "OutlineColour=&H80000000,BorderStyle=1,Outline=3,"
                                   "Shadow=1,MarginV=56,MarginL=24,MarginR=24,Alignment=2",
                    "karaoke": False},
    "minimal_top": {"label": "Minimal top",
                    "desc": "Small clean text pinned at the top — leaves the picture open.",
                    "force_style": "FontName=Khmer OS Battambang,FontSize=13,"
                                   "PrimaryColour=&H00FFFFFF,OutlineColour=&H80000000,"
                                   "BorderStyle=1,Outline=1,Shadow=0,MarginV=36,MarginL=28,"
                                   "MarginR=28,Alignment=8",
                    "karaoke": False},
    "karaoke": {"label": "Karaoke", "desc": "Word-by-word highlight (proportional timing).",
                "force_style": "FontName=Khmer OS Battambang,FontSize=15,"
                               "PrimaryColour=&H0000FFFF,Bold=1,"
                               "SecondaryColour=&H00FFFFFF,OutlineColour=&HC0000000,"
                               "BorderStyle=1,Outline=2,Shadow=0,MarginV=56,MarginL=28,"
                               "MarginR=28,Alignment=2",
                "karaoke": True},
}
TITLE_STYLE_KEYS = ("centered_fade", "bottom_left_minimal", "bold_pop")
TITLE_STYLES = {
    "centered_fade": {"label": "Centered fade",
                      "desc": "Title centre-frame, fades in and out.",
                      "layout": "center", "fontsize": 52, "yellow": False},
    "bottom_left_minimal": {"label": "Bottom-left minimal",
                            "desc": "Small title in the lower-left corner, stays quiet.",
                            "layout": "bottom_left", "fontsize": 34, "yellow": False},
    "bold_pop": {"label": "Bold pop",
                 "desc": "Big bold yellow title with a hard outline.",
                 "layout": "center", "fontsize": 58, "yellow": True},
}


def subtitle_force_style(style_key):
    st = SUBTITLE_STYLES.get(style_key or "clean", SUBTITLE_STYLES["clean"])
    return st["force_style"]


def _fmt_ass_time(sec):
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h}:{m:02d}:{int(s):02d}.{int(round((s % 1) * 100)):02d}"


def words_for_timing(text):
    """(word, weight) list for karaoke timing.

    Khmer is *scriptio continua*: when a sentence has spaces we use them as
    word separators (the studio's scripts do); otherwise we fall back to
    character-cluster groups of 2, which are the smallest pieces a viewer can
    plausibly read. Weight = the same syllable estimate the whole pipeline
    uses for pacing, so the karaoke sweep and the spoken rhythm agree.
    """
    from . import khmer

    t = khmer.display_text(text or "")
    if not t:
        return []
    if khmer.is_khmer(t):
        toks = [w for w in re.split(r"\s+", t) if w]
        if len(toks) > 1 or " " in t:
            return [(w, max(0.2, khmer.syllable_estimate(w))) for w in toks]
        units = khmer.split_clusters(t)
        words, cur = [], []
        for u in units:
            cur.append(u)
            if len(cur) >= 2:
                words.append("".join(cur))
                cur = []
        if cur:
            words.append("".join(cur))
        return [(w, max(0.2, khmer.syllable_estimate(w))) for w in words]
    toks = re.split(r"\s+", t)
    return [(w, max(1, len(re.findall(r"[aeiouy]+", w, re.I)))) for w in toks if w]


def write_karaoke_ass(scene_windows, dst, style="karaoke", width=480, height=854):
    """ASS with ``\\k`` karaoke tags — burned by the same libass `subtitles`
    filter (it supports karaoke natively).

    Timing is an honest approximation: sherpa-onnx gives no real word
    timestamps, so each scene's known audio window is distributed across its
    words proportionally to :func:`khmer.syllable_estimate` weight. This is
    NOT forced alignment — see README-STUDIO.md (real ASR alignment is the
    documented future upgrade).
    """
    from . import khmer

    st = SUBTITLE_STYLES.get(style, SUBTITLE_STYLES["karaoke"])
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {int(width)}", f"PlayResY: {int(height)}",
        "WrapStyle: 0", "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, "
        "Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, "
        "Encoding",
        f"Style: Default,{st.get('font', 'Khmer OS Battambang')},15,"
        f"{st.get('primary', '&H00FFFFFF')},{st.get('secondary', '&H0000FFFF')},"
        f"{st.get('outline', '&HC0000000')},&H80000000,{int(st.get('bold', 1))},0,0,0,100,100,0,0,"
        f"1,2,1,2,28,28,56,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    for start, end, text in scene_windows:
        words = words_for_timing(text)
        weights = [w for _w, w in words]
        total = sum(weights) or 1.0
        span = max(0.4, float(end) - float(start))
        cums, acc = [], 0.0
        for w in weights:
            acc += w
            cums.append(span * acc / total)
        tags = []
        prev = 0.0
        for (word, _w), cum in zip(words, cums):
            k = max(1, int(round((cum - prev) * 100)))
            tags.append(f"{{\\k{k}}}{word}")
            prev = cum
        text_line = " ".join(tags)
        # manual line wrap (spaceless script) using cluster breaks
        wrapped = khmer.wrap_clusters(text_line, max_clusters=64)
        text_line = "\\N".join(wrapped)
        lines.append(f"Dialogue: 0,{_fmt_ass_time(start)},{_fmt_ass_time(end)},Default,,0,0,0,,{text_line}")
    ensure_dir(os.path.dirname(dst) or ".")
    with open(dst, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return dst


def burn_subtitles(video, srt, dst, force_style="FontName=Khmer OS Battambang,FontSize=15,PrimaryColour=&H00FFFFFF,OutlineColour=&HC0000000,BorderStyle=1,Outline=2,Shadow=0,MarginV=56,MarginL=28,MarginR=28,Alignment=2",
                   style="clean"):
    """Only used when assembly.burn_captions is on and libass is available.

    Font choice matters here, not just cosmetically: this Windows ffmpeg
    build's libass won't discover installed system fonts on its own (no
    `fontsdir` = it silently falls back to *something*, and with "Noto Sans
    Khmer" specifically that fallback doesn't shape Khmer script correctly —
    dependent vowels and coeng-stacked consonants render unshaped, reading as
    scrambled text even though the underlying SRT is correct). Pointing
    fontsdir at the Windows font directory and picking a font confirmed (by
    rendering a test frame) to shape correctly fixes it.
    """
    if not _has_filter("subtitles"):
        raise RuntimeError("this ffmpeg build has no 'subtitles' filter (needs libass) — "
                           "keep SRT as a sidecar file instead")
    srt_esc = str(srt).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")
    vf = f"subtitles='{srt_esc}'"
    fontsdir = os.environ.get("SystemRoot", r"C:\Windows") + r"\Fonts" if os.name == "nt" else ""
    if fontsdir and os.path.isdir(fontsdir):
        vf += f":fontsdir='{fontsdir.replace(chr(92), '/').replace(':', chr(92) + ':')}'"
    if style and style not in ("clean",) and style in SUBTITLE_STYLES:
        force_style = subtitle_force_style(style)
    vf += f":force_style='{force_style}'"
    run_ffmpeg(["-i", video, "-vf", vf,
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-c:a", "copy", dst],
               timeout=3600)
    return dst


def burn_ass(video, ass, dst, style="karaoke"):
    """Burn an .ass file (karaoke ``\\k`` tags) via the same libass filter."""
    if not _has_filter("subtitles"):
        raise RuntimeError("this ffmpeg build has no 'subtitles' filter (needs libass)")
    ass_esc = str(ass).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")
    vf = f"subtitles='{ass_esc}'"
    fontsdir = os.environ.get("SystemRoot", r"C:\Windows") + r"\Fonts" if os.name == "nt" else ""
    if fontsdir and os.path.isdir(fontsdir):
        vf += f":fontsdir='{fontsdir.replace(chr(92), '/').replace(':', chr(92) + ':')}'"
    run_ffmpeg(["-i", video, "-vf", vf,
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-c:a", "copy", dst],
               timeout=3600)
    return dst


# -------------------------------------------------------------- title cards
def _find_font():
    """A usable font (Khmer-capable preferred) for title rendering."""
    cands = []
    if os.name == "nt":
        root = os.environ.get("WINDIR", r"C:\Windows")
        for d in (os.path.join(root, "Fonts"),):
            if os.path.isdir(d):
                cands += [os.path.join(d, f) for f in os.listdir(d)
                          if f.lower().endswith((".ttf", ".otf")) and any(
                              k in f.lower() for k in ("khmer", "noto", "battambang"))]
                cands += [os.path.join(d, f) for f in os.listdir(d)
                          if f.lower().endswith((".ttf", ".otf"))]
    for d in ("/usr/share/fonts/truetype/noto", "/usr/share/fonts/truetype/dejavu",
              "/usr/share/fonts", "/usr/local/share/fonts"):
        if os.path.isdir(d):
            for root, _dirs, files in os.walk(d):
                for f in files:
                    if f.lower().endswith((".ttf", ".otf")):
                        cands.append(os.path.join(root, f))
    for c in cands:
        if os.path.exists(c):
            return c
    return None


def render_title_card(dst, title, style="centered_fade", width=480, height=854, fps=24,
                      duration=2.6):
    """A title intro clip (ffmpeg drawtext; PIL fallback if no font renders).

    ``style`` is one of :data:`TITLE_STYLES`. Safe for Khmer: with no
    Khmer-capable font installed the PIL fallback draws the Latin/ASCII part and
    the notes say so — the title card is optional and never breaks the cut.
    """
    spec = TITLE_STYLES.get(style, TITLE_STYLES["centered_fade"])
    duration = max(1.2, float(duration))
    font = _find_font()
    if font:
        try:
            return _render_title_drawtext(dst, title, spec, font, width, height, fps, duration)
        except Exception:
            pass
    try:
        return _render_title_pil(dst, title, spec, width, height, fps, duration, font)
    except Exception as e:
        raise RuntimeError(f"title card render failed (no font?): {str(e)[:120]}")


def _esc_drawtext(s):
    return (str(s).replace("\\", "\\\\").replace(":", r"\:").replace("'", r"\'")
            .replace("%", r"\%").replace(",", r"\,"))


def _render_title_drawtext(dst, title, spec, font, width, height, fps, duration):
    fs = int(spec.get("fontsize", 48))
    colour = "0xFFFF00" if spec.get("yellow") else "0xFFFFFF"
    layout = spec.get("layout", "center")
    esc = _esc_drawtext(title)
    if layout == "bottom_left":
        pos = f"x=36:y=h-{int(height * 0.22)}"
        align = 1
    else:
        pos = "x=(w-text_w)/2:y=(h-text_h)/2"
        align = 5
    border = 4 if spec.get("yellow") else 2
    fade = duration > 1.4
    vf = (f"drawtext=fontfile='{font}':text='{esc}':{pos}:fontsize={fs}:fontcolor={colour}:"
          f"borderw={border}:bordercolor=black:shadow=1:shadowcolor=black@0.6")
    if fade:
        out_at = max(0.1, duration - 0.5)
        vf += f":alpha='if(lt(t,0.4),t/0.4,if(lt(t,{out_at:.2f}),1,({duration:.2f}-t)/0.5))'"
    args = ["-f", "lavfi", "-i", f"color=c=0x101418:s={width}x{height}:r={fps}:d={duration:.3f}",
            "-vf", vf, "-r", str(int(fps)), "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "21", "-pix_fmt", "yuv420p", "-an", dst]
    run_ffmpeg(args, timeout=900)
    return dst


def _render_title_pil(dst, title, spec, width, height, fps, duration, font):
    """PIL-drawn title card (still image) → Ken Burns clip."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        raise RuntimeError("Pillow not installed for title-card fallback")
    img = Image.new("RGB", (int(width), int(height)), (16, 20, 24))
    d = ImageDraw.Draw(img)
    fs = int(spec.get("fontsize", 48))
    try:
        f = ImageFont.truetype(font, fs) if font else ImageFont.load_default()
    except Exception:
        f = ImageFont.load_default()
    wrapped = _wrap_khmer(title, max_chars=max(10, int(width / 38)))
    lines = wrapped.split("\n")
    line_h = int(fs * 1.35)
    total_h = line_h * len(lines)
    if spec.get("layout") == "bottom_left":
        x, y = 36, int(height * 0.78) - total_h
    else:
        x, y = int(width * 0.08), int((height - total_h) / 2)
    colour = (255, 255, 0) if spec.get("yellow") else (255, 255, 255)
    for i, ln in enumerate(lines):
        try:
            bbox = d.textbbox((0, 0), ln, font=f)
            tw = bbox[2] - bbox[0]
        except Exception:
            tw = len(ln) * fs // 2
        cx = x if spec.get("layout") == "bottom_left" else x + max(0, (width - 2 * x - tw) // 2)
        d.text((cx + 2, y + i * line_h + 2), ln, font=f, fill=(0, 0, 0))
        d.text((cx, y + i * line_h), ln, font=f, fill=colour)
    png = dst + ".png"
    img.save(png)
    return make_silent_video_from_image(png, dst, duration=duration, width=width, height=height,
                                        fps=fps, motion="kenburns")


_KHMER_BREAK_CHARS = "។៕៖,.!? "


def _safe_khmer_cut(text, cut):
    """Deprecated-shim kept for callers: cluster-safe cut via ``khmer``.

    The real wrapper below uses :func:`ai_studio.khmer.split_clusters`, which
    treats ``base + ្ + subscript`` as ONE unit — the old codepoint-arithmetic
    version could still produce a line starting with a lone ``្``. This helper
    snapshots the same guarantee: it only ever returns a boundary between two
    character clusters.
    """
    from . import khmer as khmer_mod

    cut = max(1, min(int(cut), len(text)))
    prefixes = khmer_mod.split_clusters(text)
    if cut >= len(prefixes):
        return len(text)
    return len("".join(prefixes[:cut]))


def _wrap_khmer(text, max_chars=16):
    """Insert manual line breaks for burned captions — CLUSTER-SAFE.

    Khmer script has no spaces between words, so libass's whitespace-based
    auto-wrap has nowhere to break a long line — it silently overflows the
    frame instead. Break by hand using :func:`ai_studio.khmer.split_clusters`:
    a coeng subscript pair (``ស្ + វ``) is a single unit and can never be
    split across a line boundary, which is exactly the corruption that made
    ``ស្វែងយល់`` render as ``ស្ វែងយល់``. Natural punctuation breaks near the
    target width are preferred, then a hard cluster-count cut.
    """
    from . import khmer as khmer_mod

    text = text.strip()
    if not text:
        return text
    units = khmer_mod.split_clusters(text)
    budget = max(1, int(max_chars))
    lines, cur = [], []
    for cl in units:
        cur.append(cl)
        # break at punctuation once the line is reasonably full, or at the
        # hard cluster budget — either way the break is BETWEEN clusters
        if len(cur) >= budget or (cl in _KHMER_BREAK_CHARS and len(cur) >= max(2, int(budget * 0.6))):
            lines.append("".join(cur).strip())
            cur = []
    if cur:
        lines.append("".join(cur).strip())
    return "\n".join(l for l in lines if l) or text


def _split_sentences(text):
    """Break a scene's (possibly multi-sentence) text at Khmer/Latin sentence
    boundaries so one caption block is one thought, not a whole paragraph."""
    parts, cur = [], ""
    for ch in text:
        cur += ch
        if ch in "។!?." and cur.strip():
            parts.append(cur.strip())
            cur = ""
    if cur.strip():
        parts.append(cur.strip())
    return parts or [text]


def write_srt(scene_texts, scene_starts, dst, words_per_line=6):
    """Khmer-safe SRT: one sentence per caption block, time-sliced within the
    scene's window (word timing on Khmer is unreliable — it has no spaces —
    so sentence, not word, is the smallest unit we sync to), each block
    manually line-wrapped since libass can't auto-wrap spaceless script."""
    def fmt(sec):
        sec = max(0.0, float(sec))
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = sec % 60
        return f"{h:02d}:{m:02d}:{int(s):02d},{int(round((s % 1) * 1000)):03d}"

    blocks = []
    n = 0
    for i, (txt, start) in enumerate(zip(scene_texts, scene_starts)):
        end = scene_starts[i + 1] if i + 1 < len(scene_starts) else start + 3.0
        end = max(end, start + 0.8)
        sentences = _split_sentences(txt)
        span = (end - start) / len(sentences)
        for j, sent in enumerate(sentences):
            s0, s1 = start + j * span, start + (j + 1) * span
            n += 1
            blocks.append(f"{n}\n{fmt(s0)} --> {fmt(s1)}\n{_wrap_khmer(sent)}\n")
    ensure_dir(os.path.dirname(dst) or ".")
    with open(dst, "w", encoding="utf-8") as f:
        f.write("\n".join(blocks))
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

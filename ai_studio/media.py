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
    # `decode("ignore")` is a LookupError — the first positional argument of
    # bytes.decode is the encoding, not the error handler. That typo made this
    # function return 0x0/fps=0 for every video, silently.
    out = {"duration": media_duration(path, 0.0), "width": 0, "height": 0, "fps": 0.0}
    ff = ffmpeg_exe()
    if not ff or not path or not os.path.exists(path):
        return out
    try:
        res = subprocess.run([ff, "-hide_banner", "-i", path], capture_output=True, timeout=60)
        txt = (res.stderr or b"").decode("utf-8", "ignore")
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
# The old style library here was a `force_style` *string* per key, with the font
# name "Khmer OS Battambang" hardcoded in it. That font is not bundled and not
# installed on most machines, so libass substituted an arbitrary font and the
# Khmer text came out as empty boxes. Styles are now real dicts
# (ai_studio/caption_style.py) rendered by ai_studio/captions.py, and these two
# names survive only as the legacy keys the UI and old settings may still hold.
LEGACY_STYLE_KEYS = {
    "clean": "clean",
    "bold_yellow": "bold-social",
    "minimal_top": "editorial",
    "karaoke": "clean",
}
SUBTITLE_STYLE_KEYS = tuple(LEGACY_STYLE_KEYS)


def subtitle_style(key, karaoke=False):
    """Legacy style key → a validated caption-style dict (never a font blob).

    Unknown keys fall back to the studio default *and* say so in the render
    metadata; they never silently pick a font that might not exist.
    """
    from . import caption_style as cs

    preset = LEGACY_STYLE_KEYS.get(str(key or "clean"), "clean")
    style = cs.preset_style(preset)
    if karaoke:
        style = {**style, "karaoke": {"enabled": True, "color": "#FFD84D"}}
    return style


def SUBTITLE_STYLES_payload():
    """What `GET /api/settings` reports (labels + the real parameters)."""
    from . import caption_style as cs

    out = {}
    for legacy, preset in LEGACY_STYLE_KEYS.items():
        p = cs.PRESETS[preset]
        style = cs.preset_style(preset)
        if legacy == "karaoke":
            # the legacy key meant "word-by-word highlight" — keep that meaning by
            # carrying the karaoke settings in the style dict itself
            style = {**style, "karaoke": {"enabled": True, "color": "#FFD84D"}}
        out[legacy] = {"label": p["label"] + (" · word highlight" if legacy == "karaoke" else ""),
                       "desc": p["desc"], "preset": preset, "style": style,
                       "legacy": True, "karaoke": bool(legacy == "karaoke")}
    return out


SUBTITLE_STYLES = None      # built lazily by SUBTITLE_STYLES_payload()
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
    """Deprecated shim: legacy key → the ASS style line used by the new renderer.

    Kept only so old callers/notebooks do not explode; new code passes a caption
    style dict (``ai_studio/caption_style.py``) to :mod:`ai_studio.captions`.
    """
    from . import captions

    style = subtitle_style(style_key)
    return captions.ass_style_line(style, 1920, 1080, karaoke=False)


def write_karaoke_ass(scene_windows, dst, style="karaoke", width=480, height=854):
    """ASS with ``\\k`` karaoke tags — burned by the same libass filter.

    Timing is an honest approximation: sherpa-onnx gives no real word
    timestamps, so each scene's known audio window is distributed across its
    words proportionally to :func:`khmer.syllable_estimate` weight. This is
    NOT forced alignment and the render metadata says so. Wrapping is shared
    with the normal captions, so a karaoke line can no more split a coeng
    stack than a plain one.
    """
    from . import captions

    cstyle = subtitle_style(style, karaoke=True)
    cues = []
    for start, end, text in scene_windows:
        span = max(0.4, float(end) - float(start))
        words = words_for_timing(text)
        total = sum(w for _t, w in words) or 1.0
        acc, timed = 0.0, []
        for word, w in words:
            s0 = float(start) + span * acc / total
            acc += w
            s1 = float(start) + span * acc / total
            timed.append((word, s0, s1))
        cues.append({"text": khmer.display_text(text), "start": float(start),
                     "end": float(end), "words": timed})
    return captions.write_ass(cues, cstyle, width, height, dst, karaoke=cstyle["karaoke"])


def burn_subtitles(video, srt, dst, force_style="", style="clean", caption_style=None,
                   karaoke=None):
    """Burn captions through the studio's single renderer (``ai_studio.captions``).

    ``force_style`` is accepted for backwards compatibility but IGNORED — the
    old behaviour injected a raw libass style string naming a font that is not
    bundled ("Khmer OS Battambang"), and libass then silently substituted a
    font that cannot shape Khmer, which is how the reported "tofu" frames were
    produced. Captions now always come from a validated style dict, so what the
    UI previewed is what lands in the MP4.

    Raises :class:`ai_studio.captions.CaptionError` when the burn is impossible
    (no libass, missing bundled font) — the caller must surface that instead of
    shipping an uncaptioned file.
    """
    from . import captions, caption_style as cs

    style_dict = caption_style or (cs.normalize_style(caption_style or {}, strict=False)
                                   if isinstance(caption_style, dict) else None)
    if style_dict is None:
        style_dict = subtitle_style(style, karaoke=bool(karaoke))
    cues = captions.parse_srt(srt)
    if not cues:
        raise captions.CaptionError(f"no cues found in {srt}")
    info = captions.probe_size(video)
    w, h = info["width"] or 1920, info["height"] or 1080
    ass = os.path.splitext(dst)[0] + ".ass"
    captions.write_ass(cues, style_dict, w, h, ass,
                       karaoke=karaoke or style_dict.get("karaoke"))
    return captions.burn(video, ass, dst, style_dict)


def burn_ass(video, ass, dst, style="karaoke", caption_style=None):
    """Burn a prepared .ass (karaoke ``\\k`` tags or plain) with bundled fonts."""
    from . import captions, caption_style as cs

    style_dict = caption_style if isinstance(caption_style, dict) else None
    if style_dict is None:
        style_dict = cs.normalize_style({}, strict=False) if style is None else subtitle_style(style)
    return captions.burn(video, ass, dst, style_dict)


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


def write_srt(scene_texts, scene_starts, dst, ends=None, style=None,
              width=1920, height=1080):
    """Khmer-safe SRT written by the same renderer that burns the video.

    * wraps with *shaped pixel measurement* (HarfBuzz) and never inside a Khmer
      cluster — the previous writer cut ``ខ្លួន។`` into ``ខ្លួ`` + ``ន។``;
    * ends the last cue at its real end (``ends``), never at ``start + 3s``;
    * splits a scene into sentence-sized cues and keeps ``[[silent: …]]`` words
      on screen while they stay out of the spoken audio.
    """
    from . import captions, khmer

    starts = [float(s) for s in scene_starts]
    if ends is None:
        # legacy callers pass only starts: use the next start, and for the final
        # scene the estimated speech duration (never a bare fixed 3s that can
        # cut the last line short or leave it on screen after the audio ends)
        ends = [starts[i + 1] if i + 1 < len(starts)
                else starts[i] + max(1.2, khmer.estimate_speech_seconds(scene_texts[i]))
                for i in range(len(starts))]
    cues = captions.cues_from_scenes([khmer.display_text(t) for t in scene_texts],
                                     starts, [float(e) for e in ends], style=style)
    return captions.write_srt(cues, dst, style=style, width=width, height=height)


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

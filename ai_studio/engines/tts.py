"""Stage 3a — Khmer voice synthesis.

Primary engine: **sherpa-onnx** running a VITS model built from Meta MMS
(`facebook/mms-tts-khm`, converted to ONNX — see `scripts/setup_khmer_tts.sh`).
That combination is the practical 2026 answer for *local* Khmer TTS: Kokoro,
Piper and XTTS have no Khmer voice, MMS does, and ONNX keeps it fast on CPU
(so Machine B narrates at the same quality as Machine A).

Order of attempts, each fully optional:

1. ``sherpa_onnx`` Python API (int8 model preferred → smaller RAM, 16GB box)
2. the ``sherpa-onnx-offline-tts`` CLI binary (version-stable flags)
3. ``piper`` CLI, if someone drops a Khmer piper model in (not shipped by piper)
4. Kokoro-82M via the existing ``ai_creator.voice`` engine — only for non-Khmer
   text or when the Director explicitly allows a non-native voice
5. ``placeholder`` — synthetic speech-shaped audio with *realistic duration*
   (per-syllable envelopes). Everything downstream (timing, video length, QA)
   still exercises, and the asset is flagged `real_speech: false` so the UI,
   the QA stage and the final manifest all say "this is not your voice yet".

Long lines are chunked per sentence (VITS quality collapses past ~200 chars)
and stitched with a short crossfade.
"""
import glob
import os
import shutil
import subprocess
import time

from .. import khmer
from ..util import (ensure_dir, ffmpeg_exe, media_duration, read_wav, write_wav)

SR_DEFAULT = 44100


def resolve_model(cfg):
    """(model.onnx, tokens.txt, dir) for the configured sherpa VITS model."""
    from ..config import find_model_onnx, sherpa_model_dir

    d = sherpa_model_dir(cfg)
    if not d or not os.path.isdir(d):
        return None, None, d
    onnx = find_model_onnx(d)
    tokens = os.path.join(d, "tokens.txt")
    if onnx and os.path.exists(tokens):
        return onnx, tokens, d
    return None, None, d


def sherpa_bin(cfg):
    """Locate the sherpa-onnx CLI (explicit setting → PATH → model dir → ../bin)."""
    cfg_bin = (cfg.get("tts") or {}).get("sherpa_cli") or ""
    cand = []
    if cfg_bin:
        cand.append(cfg_bin)
    for name in ("sherpa-onnx-offline-tts", "sherpa-onnx-offline-tts.exe"):
        w = shutil.which(name)
        if w:
            cand.append(w)
    _m, _t, d = resolve_model(cfg)
    if d:
        cand += [os.path.join(d, "sherpa-onnx-offline-tts"),
                 os.path.join(d, "bin", "sherpa-onnx-offline-tts"),
                 os.path.join(d, "sherpa-onnx-offline-tts.exe")]
    for c in cand:
        if c and os.path.exists(c) and os.access(c, os.X_OK | os.path.R_OK):
            return c
    w = shutil.which("sherpa-onnx-offline-tts")
    return w or None


def available_engines(cfg):
    """What this machine can do right now (used by /api/status and the settings UI)."""
    out = {"sherpa_model": bool(resolve_model(cfg)[0]),
           "sherpa_python": _python_api(),
           "sherpa_cli": bool(sherpa_bin(cfg)),
           "piper": bool(shutil.which("piper")),
           "kokoro": False,
           "placeholder": True}
    try:
        from ai_creator.voice import kokoro_available
        from ..config import data_root
        out["kokoro"] = kokoro_available(data_root(cfg)) or kokoro_available(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    except Exception:
        pass
    return out


def _python_api():
    try:
        import sherpa_onnx  # noqa: F401
        return hasattr(sherpa_onnx, "OfflineTts")
    except Exception:
        return False


_engine_cache = {}


def _sherpa_tts(cfg, model_onnx, tokens, model_dir):
    """Build (and memoise) the OfflineTts instance — loading it costs ~1s."""
    key = (model_onnx, tokens, float(cfg.get("tts", {}).get("sample_rate") or 0))
    if key in _engine_cache:
        return _engine_cache[key]
    import sherpa_onnx

    tcfg = cfg.get("tts", {})
    vits_kwargs = dict(model=model_onnx, tokens=tokens)
    lex = os.path.join(model_dir, "lexicon.txt")
    if os.path.exists(lex):
        vits_kwargs["lexicon"] = lex
    espeak = os.path.join(model_dir, "espeak-ng-data")
    if os.path.isdir(espeak):
        vits_kwargs["data_dir"] = espeak
    vits_kwargs["noise_scale"] = 0.667
    # The installed sherpa-onnx (1.13.7)'s OfflineTtsVitsModelConfig ctor
    # takes `noise_scale_w`, not `noise_scale_dur` — the old kwarg name made
    # every real-model call raise "incompatible constructor arguments" and
    # silently fall back to the placeholder voice (confirmed: this was the
    # actual reason real TTS never fired once the model was installed).
    vits_kwargs["noise_scale_w"] = 0.8
    num_threads = int(os.cpu_count() or 4)
    num_threads = max(1, min(4, num_threads // 2))       # leave RAM/cores for ffmpeg
    kwargs_list = [
        dict(model=sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(**vits_kwargs),
            num_threads=num_threads, debug=False, provider="cpu")),
        dict(model=sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(**{k: v for k, v in vits_kwargs.items()
                                                           if k in ("model", "tokens")}),
            num_threads=num_threads, provider="cpu")),
    ]
    last = None
    for kw in kwargs_list:
        try:
            kw.update(max_num_sentences=1)
            engine = sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(**kw))
        except TypeError as e:
            last = e
            try:
                kw.pop("max_num_sentences", None)
                engine = sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(**kw))
            except Exception as e2:
                last = e2
                continue
        except Exception as e:
            last = e
            continue
        _engine_cache[key] = engine
        return engine
    raise RuntimeError(f"could not init sherpa-onnx TTS: {last}")


# ------------------------------------------------------------------ synthesis
def synthesize(text, out_wav, cfg, engine="auto", progress=None, seed=0):
    """Speak `text` into `out_wav`. Returns a result dict (never raises)."""
    cfg_t = cfg.get("tts", {})
    text = khmer.strip_emoji_and_marks(text or "")
    if not text:
        return {"ok": False, "reason": "empty text", "engine": "none"}
    ensure_dir(os.path.dirname(out_wav) or ".")
    want = engine if engine in ("sherpa", "piper", "kokoro", "placeholder") else "auto"
    chain = (["sherpa", "piper", "kokoro", "placeholder"] if want == "auto" else [want, "placeholder"])
    attempts = []
    for choice in chain:
        if choice == "sherpa":
            res = _try_sherpa(text, out_wav, cfg, progress, attempts)
        elif choice == "piper":
            res = _try_piper(text, out_wav, cfg, progress, attempts)
        elif choice == "kokoro":
            res = _try_kokoro(text, out_wav, cfg, attempts)
        else:
            res = _placeholder(text, out_wav, cfg, attempts, seed=seed)
        if res.get("ok"):
            res.setdefault("chunks", 1)
            return res
    return {"ok": False, "reason": "; ".join(attempts) or "no TTS engine available",
            "engine": "none", "attempts": attempts}


def _try_sherpa(text, out_wav, cfg, progress, attempts):
    model_onnx, tokens, model_dir = resolve_model(cfg)
    if not model_onnx:
        attempts.append("sherpa: model dir has no model.onnx + tokens.txt")
        return {"ok": False}
    chunks = khmer.tts_chunks(text, max_chars=190)
    speed = float(cfg.get("tts", {}).get("speed", 1.0))
    sid = int(cfg.get("tts", {}).get("speaker_id", 0))
    tmp = ensure_dir(os.path.join(os.path.dirname(out_wav), ".tts_tmp"))
    parts, sr = [], 0
    if _python_api():
        try:
            engine = _sherpa_tts(cfg, model_onnx, tokens, model_dir)
            for i, ch in enumerate(chunks):
                audio = _generate(engine, ch, sid, speed)
                if audio is None:
                    raise RuntimeError("sherpa returned no audio")
                arr, sr = _as_array(audio)
                p = os.path.join(tmp, f"part{i:03d}.wav")
                write_wav(p, arr, sr or SR_DEFAULT, normalize_to=1.0)
                parts.append(p)
                if progress:
                    progress(100.0 * (i + 1) / len(chunks), f"sherpa chunk {i + 1}/{len(chunks)}")
        except Exception as e:
            attempts.append(f"sherpa python api failed ({str(e)[:120]}); trying CLI")
            parts, sr = [], 0
    if not parts:
        exe = sherpa_bin(cfg)
        if not exe:
            attempts.append("sherpa: no usable Python API and no CLI binary")
            return {"ok": False}
        for i, ch in enumerate(chunks):
            d = ensure_dir(os.path.join(tmp, f"c{i:03d}"))
            cmd = [exe, f"--vits-model={model_onnx}", f"--vits-tokens={tokens}",
                   f"--output-filename={os.path.join(d, 'out.wav')}", "--silence-scale=0.25"]
            if sid:
                cmd.append(f"--vits-speaker-id={sid}")
            if speed and abs(speed - 1.0) > 1e-3:
                cmd.append(f"--speed={speed}")
            if os.path.isdir(os.path.join(model_dir, "espeak-ng-data")):
                cmd.append(f"--vits-data-dir={os.path.join(model_dir, 'espeak-ng-data')}")
            cmd.append(ch)
            try:
                r = subprocess.run(cmd, cwd=d, capture_output=True, timeout=1200)
            except Exception as e:
                attempts.append(f"sherpa CLI error: {str(e)[:120]}")
                return {"ok": False}
            found = sorted(glob.glob(os.path.join(d, "*.wav")))
            if not found:
                attempts.append("sherpa CLI produced no wav: "
                                + (r.stderr.decode(errors="ignore")[-160:] if r.stderr else ""))
                return {"ok": False}
            parts.extend(found)
            if progress:
                progress(100.0 * (i + 1) / len(chunks), f"sherpa CLI chunk {i + 1}/{len(chunks)}")
    if not parts:
        return {"ok": False}
    joined = _stitch(parts, out_wav, gap=0.10, crossfade_ms=int(cfg.get("tts", {}).get("crossfade_ms", 30)))
    for p in parts:
        try:
            if os.path.exists(p) and not p.startswith(os.path.dirname(out_wav) + os.sep + ".tts_tmp" + os.sep):
                os.remove(p)
        except Exception:
            pass
    dur = media_duration(joined, 0.0)
    return {"ok": True, "engine": "sherpa-onnx-vits", "duration": dur, "real_speech": True,
            "model": os.path.basename(model_onnx), "model_dir": model_dir,
            "chunks": len(parts), "sample_rate": sr or SR_DEFAULT,
            "language": cfg.get("tts", {}).get("language", "km")}


def _generate(engine, text, sid, speed):
    for kwargs in (dict(sid=sid, speed=speed), {}):
        try:
            fn = getattr(engine, "generate", None)
            if fn is None:
                return None
            out = fn(text, **kwargs) if kwargs else fn(text)
            if out is not None and getattr(out, "samples", None) is not None:
                return out
        except TypeError:
            continue
        except Exception:
            return None
    return None


def _as_array(audio):
    import numpy as np

    sr = int(getattr(audio, "sample_rate", SR_DEFAULT) or SR_DEFAULT)
    samples = getattr(audio, "samples", None)
    if samples is None and isinstance(audio, (list, tuple)) and len(audio) == 2:
        samples, sr = audio[0], int(audio[1])
    if samples is None:
        raise RuntimeError("unsupported sherpa output")
    return np.asarray(samples, dtype=np.float32), sr


def _stitch(parts, out_wav, gap=0.1, crossfade_ms=30):
    import numpy as np

    if len(parts) == 1:
        x, sr = read_wav(parts[0])
        write_wav(out_wav, x, sr, normalize_to=0.9)
        return out_wav
    chunks, sr = [], 44100
    for p in parts:
        x, s = read_wav(p)
        sr = s or sr
        chunks.append(x)
    gapn = max(1, int(gap * sr))
    xf = max(0, int(crossfade_ms / 1000.0 * sr))
    total = sum(c.shape[0] for c in chunks) + gapn * (len(chunks) - 1)
    out = np.zeros(total, dtype=np.float32)
    pos = 0
    for i, c in enumerate(chunks):
        if i and xf > 1 and pos >= xf:
            a = out[pos - xf:pos]
            b = c[:xf]
            w = np.linspace(0, 1, xf, dtype=np.float32)
            out[pos - xf:pos] = a * (1 - w) + b * w
            out[pos:pos + c.shape[0] - xf] = c[xf:]
            pos += c.shape[0] - xf
        else:
            out[pos:pos + c.shape[0]] = c
        pos += c.shape[0]
        if i < len(chunks) - 1:
            pos += gapn
    write_wav(out_wav, out[:pos], sr, normalize_to=0.9)
    return out_wav


def _try_piper(text, out_wav, cfg, progress, attempts):
    exe = shutil.which("piper")
    model = os.environ.get("PIPER_MODEL", "")
    if not exe or not os.path.exists(model):
        attempts.append("piper: binary or PIPER_MODEL not set")
        return {"ok": False}
    try:
        p = subprocess.run([exe, f"--output_file={out_wav}", "--sentence_silence=0.12"],
                           input=text.encode("utf-8"), capture_output=True, timeout=900)
        if p.returncode == 0 and os.path.exists(out_wav):
            return {"ok": True, "engine": "piper", "duration": media_duration(out_wav, 0),
                    "real_speech": True}
    except Exception as e:
        attempts.append(f"piper error {str(e)[:120]}")
    return {"ok": False}


def _try_kokoro(text, out_wav, cfg, attempts):
    if not cfg.get("tts", {}).get("allow_nonnative_fallback") and khmer.is_khmer(text):
        attempts.append("kokoro skipped: it has no Khmer voice "
                        "(enable tts.allow_nonnative_fallback to use it anyway)")
        return {"ok": False}
    try:
        from ai_creator.voice import TTSEngine
        from ..config import data_root

        tts = TTSEngine(weights_dir=data_root(cfg))
        ok, engine_name = tts.speak(text, out_wav, kokoro_voice="af_bella")
        if ok:
            return {"ok": True, "engine": f"kokoro({engine_name})", "real_speech": True,
                    "duration": media_duration(out_wav, 0),
                    "note": "non-Khmer voice engine — pronunciation will be wrong for Khmer text"}
    except Exception as e:
        attempts.append(f"kokoro: {str(e)[:120]}")
    return {"ok": False}


def _placeholder(text, out_wav, cfg, attempts, seed=0):
    """Speech-shaped synthetic audio with honest, realistic timing.

    Vowel-ish formant pairs per syllable + consonant noise bursts, so the clip
    has the rhythm of calm Khmer narration. Duration = the same estimate every
    other stage uses, so the whole pipeline can be exercised (and video
    length-matched) with no model installed.
    """
    import numpy as np

    try:
        syl = max(1.0, khmer.syllable_estimate(text))
        calm = float(cfg.get("pipeline", {}).get("pace_calm", 1.15))
        rate = 3.5 / max(0.5, calm)
        dur = float(np.clip((syl / rate) + 0.5, 0.9, 90.0))
        sr = SR_DEFAULT
        n = int(dur * sr)
        rng = np.random.default_rng((abs(int(seed)) * 7919 + 13) % (2 ** 31))
        nsyl = max(2, int(round(syl)))
        per = n / nsyl
        out = np.zeros(n, dtype=np.float32)
        base = float(rng.uniform(118, 148))          # a calm male-ish fundamental
        for i in range(nsyl):
            s0 = int(i * per)
            s1 = min(n, int((i + 1) * per))
            m = s1 - s0
            if m < 40:
                continue
            t = np.arange(m, dtype=np.float32) / sr
            f0 = base * (1.0 + 0.05 * float(rng.normal())) * (1 - 0.06 * (i / nsyl))
            # two formants + f0 harmonics ≈ a soft 'oa'-like vowel
            fw1, fw2 = 520 + 260 * float(rng.normal(scale=0.5)), 1180 + 240 * float(rng.normal(scale=0.5))
            env = np.hanning(m).astype(np.float32)
            voiced = (np.sin(2 * np.pi * f0 * t) * 0.55
                      + 0.30 * np.sin(2 * np.pi * f0 * 2 * t + 0.4)
                      + 0.18 * np.sin(2 * np.pi * fw1 * t)
                      + 0.10 * np.sin(2 * np.pi * fw2 * t)).astype(np.float32)
            voiced *= env
            if float(rng.random()) < 0.3 and m > 900:      # consonant onset
                k = int(min(m * 0.12, sr * 0.045))
                voiced[:k] += rng.normal(0, 0.16, k).astype(np.float32) * np.linspace(1, 0, k)
            out[s0:s1] += voiced
            if i < nsyl - 1 and float(rng.random()) < 0.14:   # phrase-final pause
                gap = int(sr * float(rng.uniform(0.09, 0.2)))
                out[s1 - min(gap, m):s1] *= np.linspace(1, 0.15, min(gap, m), dtype=np.float32)
        out = np.clip(out, -1, 1) * 0.6
        breath = rng.normal(0, 0.012, n).astype(np.float32)
        write_wav(out_wav, out + breath, sr, normalize_to=0.62)
        return {"ok": True, "engine": "placeholder", "duration": dur, "real_speech": False,
                "syllables": nsyl,
                "note": ("no Khmer TTS engine found — synthetic speech-shaped audio keeps the "
                         "pipeline/timings honest. Run scripts/setup_khmer_tts.sh to fix this.")}
    except Exception as e:
        attempts.append(f"placeholder failed: {str(e)[:120]}")
        return {"ok": False}


def probe(cfg):
    """Status for /api/status + the settings page."""
    eng = available_engines(cfg)
    onnx, tokens, d = resolve_model(cfg)
    est = None
    try:
        est = os.path.getsize(onnx) if onnx else None
    except Exception:
        pass
    return {"engines": eng, "model_dir": d, "model": onnx, "tokens": tokens,
            "model_bytes": est, "sherpa_cli": sherpa_bin(cfg),
            "ready": bool(eng["sherpa_model"] and (eng["sherpa_python"] or eng["sherpa_cli"])),
            "fallback": "placeholder" if not eng["sherpa_model"] else "none"}

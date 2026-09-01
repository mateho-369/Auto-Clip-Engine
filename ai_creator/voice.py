"""Voice store + TTS engines.

Voice cloning: upload a 10s–2min recording of YOUR voice. When the
Coqui/XTTS engine (optional `TTS` package) is available locally it is
used to clone that voice — the most human-like result. Otherwise:

  1. Kokoro-82M ONNX (local, weights must be downloaded — setup script)
  2. gTTS (online fallback)
  3. no voice — render still succeeds (SFX only), UI reports it clearly

All engines produce a mono WAV the renderer mixes.
"""
import json
import os
import shutil
import subprocess
import time
import uuid
import wave

import numpy as np

MIN_VOICE_SEC = 8
MAX_VOICE_SEC = 180

_XTTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
_xtts_instance = None
_xtts_failed = False


def _ffmpeg_exe():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _probe_wav(path):
    try:
        with wave.open(path, "rb") as wf:
            return wf.getnframes() / float(wf.getframerate())
    except Exception:
        return 0.0


class VoiceStore:
    def __init__(self, root):
        self.root = os.path.join(root, "voices")
        os.makedirs(self.root, exist_ok=True)

    def _dir(self, vid):
        return os.path.join(self.root, vid)

    def list(self):
        out = []
        for d in sorted(os.listdir(self.root)):
            mp = os.path.join(self._dir(d), "meta.json")
            if os.path.exists(mp):
                try:
                    with open(mp, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    meta["dir"] = self._dir(d)
                    out.append(meta)
                except Exception:
                    continue
        return out

    def get(self, vid):
        mp = os.path.join(self._dir(vid), "meta.json")
        if not os.path.exists(mp):
            return None
        with open(mp, "r", encoding="utf-8") as f:
            meta = json.load(f)
        meta["dir"] = self._dir(vid)
        return meta

    def add(self, name, upload_path, source="upload"):
        """Converts any audio upload to 16k mono WAV (via ffmpeg) and stores it."""
        meta_path_probe = os.path.join(self._dir("probe"), "meta.json")
        vid = str(uuid.uuid4())[:8]
        vdir = self._dir(vid)
        os.makedirs(vdir, exist_ok=True)
        wav_path = os.path.join(vdir, "recording.wav")
        ff = _ffmpeg_exe()
        if ff is None:
            raise ValueError("ffmpeg not found — cannot process voice recording.")
        res = subprocess.run(
            [ff, "-y", "-loglevel", "error", "-i", upload_path,
             "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", wav_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120,
        )
        if res.returncode != 0 or not os.path.exists(wav_path):
            shutil.rmtree(vdir, ignore_errors=True)
            raise ValueError(f"Could not decode audio: {res.stderr.decode(errors='ignore')[:300]}")
        dur = _probe_wav(wav_path)
        if dur < MIN_VOICE_SEC:
            shutil.rmtree(vdir, ignore_errors=True)
            raise ValueError(f"Recording is {dur:.1f}s — send at least {MIN_VOICE_SEC}s (ideally 30-60s).")
        if dur > MAX_VOICE_SEC:
            # trim to max (ffmpeg re-run)
            res = subprocess.run(
                [ff, "-y", "-loglevel", "error", "-i", upload_path,
                 "-t", str(MAX_VOICE_SEC), "-ar", "16000", "-ac", "1",
                 "-c:a", "pcm_s16le", wav_path],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120,
            )
            dur = _probe_wav(wav_path)
        meta = {
            "id": vid,
            "name": name or "My Voice",
            "created": time.time(),
            "duration": round(dur, 2),
            "source": source,
            "clone_engine": "xtts" if xtts_available() else "pending-xtts-or-fallback",
        }
        with open(os.path.join(vdir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        return meta

    def delete(self, vid):
        d = self._dir(vid)
        if os.path.exists(d):
            shutil.rmtree(d, ignore_errors=True)
            return True
        return False


def kokoro_available(weights_dir="."):
    return (os.path.exists(os.path.join(weights_dir, "kokoro-v0_19.onnx"))
            and os.path.exists(os.path.join(weights_dir, "voices.bin")))


def gtts_available():
    try:
        import gtts  # noqa: F401
        return True
    except Exception:
        return False


def xtts_available():
    """True when the optional Coqui TTS package with an XTTS v2 model is usable."""
    global _xtts_instance, _xtts_failed
    if _xtts_instance is not None:
        return True
    if _xtts_failed:
        return False
    try:
        from TTS.api import TTS  # noqa: F401
        import torch
        # model must be present locally (downloaded by setup script) — avoid
        # silently pulling 1.9GB at render time
        from huggingface_hub import try_to_load_from_cache  # type: ignore
        found = try_to_load_from_cache("hf.co", _XTTS_MODEL, "config.json") is not None or \
                _XTTS_MODEL in os.listdir(os.path.expanduser("~/.cache")) if os.path.isdir(os.path.expanduser("~/.cache")) else False
        _xtts_failed = not (found or _env_forced())
        return not _xtts_failed
    except Exception:
        _xtts_failed = True
        return False


def _env_forced():
    return os.environ.get("AI_CREATOR_FORCE_XTTS") == "1"


def _get_xtts():
    global _xtts_instance
    if _xtts_instance is None:
        from TTS.api import TTS
        _xtts_instance = TTS(_XTTS_MODEL)
    return _xtts_instance


class TTSEngine:
    def __init__(self, weights_dir=".", gtts_tld="com"):
        self.weights_dir = weights_dir
        self.gtts_tld = gtts_tld
        self.kokoro = None
        if kokoro_available(weights_dir):
            try:
                from kokoro_onnx import Kokoro
                self.kokoro = Kokoro(
                    os.path.join(weights_dir, "kokoro-v0_19.onnx"),
                    os.path.join(weights_dir, "voices.bin"),
                )
            except Exception as e:
                print(f"Kokoro init failed: {e}")
                self.kokoro = None

    def probe(self):
        return {
            "kokoro": self.kokoro is not None,
            "xtts": xtts_available(),
            "gtts": gtts_available(),
        }

    def speak(self, text, out_wav, clone_voice_wav=None, kokoro_voice="af_bella"):
        """Speaks text to out_wav. Returns (ok, engine_name)."""
        text = (text or "").strip()
        if not text:
            return False, "none"
        # 1) voice cloning with XTTS
        if clone_voice_wav and os.path.exists(clone_voice_wav) and xtts_available():
            try:
                model = _get_xtts()
                model.tts_to_file(text=text, speaker_wav=clone_voice_wav,
                                  language="en", file_path=out_wav)
                if os.path.exists(out_wav) and _probe_wav(out_wav) > 0.2:
                    return True, "xtts-clone"
            except Exception as e:
                print(f"XTTS clone failed: {e}; trying next engine.")
        # 2) Kokoro (local, very human-like)
        if self.kokoro is not None:
            try:
                import soundfile as sf
                samples, sr = self.kokoro.create(text, voice=kokoro_voice, speed=1.0, lang="en-us")
                sf.write(out_wav, np.asarray(samples, dtype=np.float32), sr)
                if os.path.exists(out_wav) and _probe_wav(out_wav) > 0.2:
                    return True, "kokoro"
            except Exception as e:
                print(f"Kokoro failed: {e}; trying gTTS.")
        # 3) gTTS (online)
        try:
            from gtts import gTTS
            tmp_mp3 = out_wav + ".mp3"
            gTTS(text=text, lang="en", tld=self.gtts_tld, slow=False).save(tmp_mp3)
            if os.path.exists(tmp_mp3):
                ff = _ffmpeg_exe()
                if ff:
                    subprocess.run([ff, "-y", "-loglevel", "error", "-i", tmp_mp3,
                                    "-ar", "22050", "-ac", "1", "-c:a", "pcm_s16le", out_wav],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
                else:
                    os.replace(tmp_mp3, out_wav)
                os.remove(tmp_mp3) if os.path.exists(tmp_mp3) else None
                if os.path.exists(out_wav) and _probe_wav(out_wav) > 0.2:
                    return True, "gtts"
        except Exception as e:
            print(f"gTTS failed: {e}")
        return False, "none"


def audio_envelope(wav_path, fps=24):
    """RMS envelope (0..1) of a wav at fps — drives the talk-pulse animation."""
    try:
        with wave.open(wav_path, "rb") as wf:
            sr = wf.getframerate()
            n = wf.getnframes()
            raw = wf.readframes(n)
            data = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
            if wf.getnchannels() > 1:
                data = data.reshape(-1, wf.getnchannels()).mean(axis=1)
        if data.size == 0:
            return np.zeros(1, dtype=np.float32)
        chunk = max(1, int(sr / fps))
        env = np.array([
            float(np.sqrt(np.mean(data[i:i + chunk] ** 2)) / 32768.0)
            for i in range(0, len(data), chunk)
        ], dtype=np.float32)
        peak = float(env.max())
        if peak > 0:
            env = env / peak
        return env
    except Exception:
        return np.zeros(1, dtype=np.float32)


def wav_duration(path):
    return _probe_wav(path)

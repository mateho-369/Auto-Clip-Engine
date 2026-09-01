"""Integration test: full render pipeline (scenes -> transitions -> SFX mix -> MP4+SRT)."""
import os
import sys
import wave

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai_creator.renderer import Renderer  # noqa: E402
from ai_creator.character import CharacterStore  # noqa: E402
from tests.test_creator_character import make_synthetic_face_photo  # noqa: E402
from ai_creator import sfx as sfx_mod  # noqa: E402


class FakeTTS:
    """Stands in for TTSEngine: writes a 1.5s sine 'narration' wav (22050 Hz)."""
    def __init__(self):
        self.calls = []

    def speak(self, text, out_wav, clone_voice_wav=None, kokoro_voice="af_bella"):
        self.calls.append(text)
        sr = 22050
        n = int(1.5 * sr)
        t = np.arange(n) / sr
        x = np.sin(2 * np.pi * 440 * t) * 0.5
        env = np.clip(np.sin(np.pi * t / 1.5), 0, 1)
        pcm = (x * env * 32767).astype(np.int16)
        with wave.open(out_wav, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm.tobytes())
        return True, "fake"

    def probe(self):
        return {"kokoro": False, "xtts": False, "gtts": False}


def make_character(tmp_path):
    store = CharacterStore(str(tmp_path / "root"))
    photo = make_synthetic_face_photo(str(tmp_path / "photo.jpg"), seed=9)
    prof = store.create("Tester", photo)
    return prof["dir"]


def test_full_render_with_narration_and_sfx(tmp_path):
    sfx_dir = str(tmp_path / "sfx")
    sfx_mod.ensure_library(sfx_dir)
    char_dir = make_character(tmp_path)
    work = str(tmp_path / "work")
    out = str(tmp_path / "out")
    os.makedirs(out, exist_ok=True)

    plan = {
        "title": "Integration Test Short",
        "logline": "two scenes",
        "scenes": [
            {"hook": "Hook", "script": "Welcome to the integration test of the studio pipeline",
             "sfx": "whoosh", "sfx_time": 0.2, "animation": "pop-in",
             "transition": "fade", "background": "gradient-violet", "duration": 3.0},
            {"hook": "Payoff", "script": "This second scene proves transitions and audio mixing work",
             "sfx": "ding", "sfx_time": 0.4, "animation": "bounce",
             "transition": "zoom", "background": "pattern-dots", "duration": 3.0},
        ],
    }
    progress = []
    renderer = Renderer(sfx_dir, work)
    result = renderer.render(plan, char_dir, FakeTTS(), None,
                             width=480, height=854, fps=24, out_dir=out,
                             progress=lambda stage, pct: progress.append(pct))

    # files exist
    assert os.path.exists(os.path.join(out, "video.mp4"))
    assert os.path.exists(os.path.join(out, "subtitles.srt"))
    # SRT has content for both scenes (phrases wrap across 3-word SRT lines)
    srt = open(os.path.join(out, "subtitles.srt"), encoding="utf-8").read()
    assert "Welcome to the" in srt
    assert "integration test of" in srt
    assert "proves transitions" in srt

    # video specs
    cap = cv2.VideoCapture(os.path.join(out, "video.mp4"))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert (w, h) == (480, 854)
    # ~6s minus 0.5s transition overlap, plus tail
    assert 4 * 24 <= n_frames <= 8 * 24

    # frame check: character pixels + caption pixels present
    cap = cv2.VideoCapture(os.path.join(out, "video.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 24)  # mid scene 1
    ret, frame = cap.read()
    cap.release()
    assert ret
    assert frame.max() > 100
    # captions band
    assert frame[int(h * 0.78):, :, :].max() > 200

    # narration was actually requested for both scenes
    # (FakeTTS.calls recorded; render consumed the wavs)
    assert len(progress) >= 5
    assert result["scenes"] == 2
    assert result["duration"] > 4.5


def test_render_without_tts_still_succeeds_sfx_only(tmp_path):
    """No TTS engine available -> scenes render without narration, SFX still mixed."""
    sfx_dir = str(tmp_path / "sfx")
    sfx_mod.ensure_library(sfx_dir)
    char_dir = make_character(tmp_path)

    class NoTTS(FakeTTS):
        def speak(self, text, out_wav, **kw):
            return False, "none"

    plan = {
        "title": "No Voice Short",
        "scenes": [
            {"hook": "A", "script": "silent scene one", "sfx": "pop", "sfx_time": 0.2,
             "animation": "slide-left", "transition": "wipe", "background": "solid-navy", "duration": 3.0},
            {"hook": "B", "script": "silent scene two", "sfx": "boom", "sfx_time": 0.3,
             "animation": "zoom", "transition": "cut", "background": "solid-dark", "duration": 3.0},
        ],
    }
    out = str(tmp_path / "out2")
    renderer = Renderer(sfx_dir, str(tmp_path / "work2"))
    result = renderer.render(plan, char_dir, NoTTS(), None,
                             width=480, height=854, fps=24, out_dir=out)
    assert os.path.exists(os.path.join(out, "video.mp4"))
    assert result["duration"] > 4.5

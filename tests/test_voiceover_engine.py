import os
import pytest
from unittest.mock import patch
from src.voiceover_engine import VoiceoverEngine

def test_voiceover_engine_graceful_fallback():
    """Verify that when Kokoro model files are missing, VoiceoverEngine defaults
    to the gTTS fallback path without crashing. The gTTS call itself is mocked —
    it's an unofficial, frequently-fragile reverse-engineered API and a test
    suite shouldn't depend on its live uptime to verify our own fallback logic."""
    engine = VoiceoverEngine(kokoro_model_path="non_existent_model.onnx", kokoro_voices_path="non_existent_voices.bin")

    assert not engine.kokoro_available

    temp_voice = "temp_test_voice.mp3"

    def fake_save(self, path):
        # Simulate gTTS writing an output file, without a live network call
        with open(path, "wb") as f:
            f.write(b"\x00" * 128)

    try:
        with patch("src.voiceover_engine.gTTS.save", fake_save):
            success = engine.generate_voiceover_mp3(
                text="Testing local voiceover synthesis fallback",
                output_path=temp_voice,
                use_kokoro=True  # Kokoro unavailable -> should fall through to gTTS path
            )
        assert success
        assert os.path.exists(temp_voice)
        assert os.path.getsize(temp_voice) > 0
    finally:
        if os.path.exists(temp_voice):
            os.remove(temp_voice)

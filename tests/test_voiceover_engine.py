import os
import pytest
from src.voiceover_engine import VoiceoverEngine

def test_voiceover_engine_graceful_fallback():
    """Verify that when Kokoro model files are missing, VoiceoverEngine defaults to gTTS gracefully without crashing."""
    # Ensure any previous model files are bypassed
    engine = VoiceoverEngine(kokoro_model_path="non_existent_model.onnx", kokoro_voices_path="non_existent_voices.bin")
    
    assert not engine.kokoro_available
    
    # Run voiceover generation (should fallback and succeed via gTTS)
    temp_voice = "temp_test_voice.mp3"
    try:
        success = engine.generate_voiceover_mp3(
            text="Testing local voiceover synthesis fallback",
            output_path=temp_voice,
            use_kokoro=True # It should fallback to gTTS
        )
        assert success
        assert os.path.exists(temp_voice)
        assert os.path.getsize(temp_voice) > 0
    finally:
        if os.path.exists(temp_voice):
            os.remove(temp_voice)

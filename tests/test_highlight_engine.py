import os
import pytest
import json
import numpy as np
from unittest.mock import patch, MagicMock
from src.highlight_engine import HighlightEngine

def test_scoring_returns_native_floats():
    """Verify that scoring values and breakdowns are strictly native Python floats, not numpy scalars."""
    engine = HighlightEngine("non_existent_video.mp4")
    
    # Mock duration and metrics
    engine.duration = 60
    
    # We will run a mocked detect_highlights pass by mocking the internal extract and analysis functions
    with patch.object(engine, 'extract_audio', return_value=True), \
         patch.object(engine, 'analyze_audio_energy', return_value=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6] * 10), \
         patch.object(engine, 'transcribe_audio_segments', return_value=[{"start": 0, "end": 15, "text": "wow amazing laugh joke"}] * 4), \
         patch.object(engine, 'analyze_visual_motion', return_value=[5.0, 10.0, 15.0] * 10):
         
        highlights = engine.detect_highlights(use_whisper=False, llm_provider="Off")
        
        assert len(highlights) > 0
        for hl in highlights:
            # Check type of score
            assert isinstance(hl["score"], float)
            assert not isinstance(hl["score"], (np.float32, np.float64, np.integer))
            
            # Check type of each breakdown metric
            for k, v in hl["breakdown"].items():
                assert isinstance(v, (float, int))
                assert not isinstance(v, (np.float32, np.float64, np.integer))

@patch("urllib.request.urlopen")
def test_ollama_graceful_fallback(mock_urlopen):
    """Verify that if Ollama is offline or returns error, the app gracefully degrades to heuristic-only scoring."""
    # Mock urlopen to raise an exception representing a network error or offline status
    mock_urlopen.side_effect = Exception("Ollama connection refused")
    
    engine = HighlightEngine("non_existent_video.mp4")
    engine.duration = 40
    
    with patch.object(engine, 'extract_audio', return_value=True), \
         patch.object(engine, 'analyze_audio_energy', return_value=[0.2] * 40), \
         patch.object(engine, 'transcribe_audio_segments', return_value=[]), \
         patch.object(engine, 'analyze_visual_motion', return_value=[]):
         
        # This call should complete smoothly with NO exception propagating, even though Ollama is enabled
        highlights = engine.detect_highlights(use_whisper=False, llm_provider="Ollama", ollama_model="llama3.2:3b")
        
        assert len(highlights) > 0
        for hl in highlights:
            # Verify fallback was activated
            assert "fallback" in hl["explanation"] or "offline" in hl["explanation"]
            assert hl["breakdown"]["llm_semantic_score"] == 0.0

@patch("urllib.request.urlopen")
def test_openai_compatible_success(mock_urlopen):
    """Verify that a successful OpenAI-compatible (like OpenCode Zen/9Router) API response is parsed and blended correctly."""
    # Mock return value of OpenAI completions API
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": json.dumps([
                        {
                            "index": 0,
                            "semantic_score": 95,
                            "reason": "This is an extremely engaging clip!"
                        }
                    ])
                }
            }
        ]
    }).encode("utf-8")
    mock_urlopen.return_value.__enter__.return_value = mock_response
    
    engine = HighlightEngine("non_existent_video.mp4")
    engine.duration = 40
    
    with patch.object(engine, 'extract_audio', return_value=True), \
         patch.object(engine, 'analyze_audio_energy', return_value=[0.2] * 40), \
         patch.object(engine, 'transcribe_audio_segments', return_value=[{"start": 0, "end": 15, "text": "funny moment"}] * 3), \
         patch.object(engine, 'analyze_visual_motion', return_value=[]):
         
        highlights = engine.detect_highlights(
            use_whisper=False, 
            llm_provider="OpenAI", 
            openai_base_url="http://localhost:20128/v1", 
            openai_model="oc/deepseek-v4-flash-free",
            openai_api_key="sk-test-key"
        )
        
        assert len(highlights) > 0
        # The first element should have been blended with the LLM semantic score
        assert "This is an extremely engaging clip!" in highlights[0]["explanation"]
        assert highlights[0]["breakdown"]["llm_semantic_score"] == 95.0

@patch("urllib.request.urlopen")
def test_openai_compatible_graceful_fallback(mock_urlopen):
    """Verify that if the OpenAI endpoint is unreachable or unauthorized, the app gracefully degrades without failing."""
    mock_urlopen.side_effect = Exception("HTTP Error 401: Unauthorized")
    
    engine = HighlightEngine("non_existent_video.mp4")
    engine.duration = 40
    
    with patch.object(engine, 'extract_audio', return_value=True), \
         patch.object(engine, 'analyze_audio_energy', return_value=[0.2] * 40), \
         patch.object(engine, 'transcribe_audio_segments', return_value=[]), \
         patch.object(engine, 'analyze_visual_motion', return_value=[]):
         
        highlights = engine.detect_highlights(
            use_whisper=False, 
            llm_provider="OpenAI", 
            openai_base_url="http://localhost:20128/v1", 
            openai_model="oc/deepseek-v4-flash-free",
            openai_api_key="bad-key"
        )
        
        assert len(highlights) > 0
        for hl in highlights:
            assert "fallback" in hl["explanation"] or "offline" in hl["explanation"]
            assert hl["breakdown"]["llm_semantic_score"] == 0.0

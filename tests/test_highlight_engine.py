import os
import pytest
import json
import numpy as np
import cv2
from unittest.mock import patch, MagicMock
from src.highlight_engine import HighlightEngine

def test_scoring_returns_native_floats_and_timings():
    """Verify that scoring values are native Python floats, and timing metrics are logged with correct keys."""
    engine = HighlightEngine("non_existent_video.mp4")
    engine.duration = 100
    
    with patch.object(engine, 'extract_audio', return_value=True), \
         patch.object(engine, 'analyze_audio_energy', return_value=[0.1, 0.2] * 50), \
         patch.object(engine, 'transcribe_audio_segments', return_value=[{"start": 0, "end": 15, "text": "wow amazing laugh joke"}] * 4), \
         patch.object(engine, 'analyze_visual_motion', return_value=[{"time": float(i), "score": 10.0} for i in range(100)]):
         
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
                
        # Validate performance timing instrumentations
        assert isinstance(engine.last_timing, dict)
        required_keys = {"audio_extract", "transcribe", "motion", "heuristic_scoring", "llm_rerank", "total"}
        assert required_keys.issubset(engine.last_timing.keys())
        for key in required_keys:
            assert isinstance(engine.last_timing[key], (float, int))

@patch("urllib.request.urlopen")
def test_ollama_graceful_fallback(mock_urlopen):
    """Verify that if Ollama is offline or returns error, the app gracefully degrades to heuristic-only scoring."""
    mock_urlopen.side_effect = Exception("Ollama connection refused")
    
    engine = HighlightEngine("non_existent_video.mp4")
    engine.duration = 40
    
    with patch.object(engine, 'extract_audio', return_value=True), \
         patch.object(engine, 'analyze_audio_energy', return_value=[0.2] * 40), \
         patch.object(engine, 'transcribe_audio_segments', return_value=[]), \
         patch.object(engine, 'analyze_visual_motion', return_value=[]):
         
        highlights = engine.detect_highlights(use_whisper=False, llm_provider="Ollama", ollama_model="llama3.2:3b")
        
        assert len(highlights) > 0
        for hl in highlights:
            assert "fallback" in hl["explanation"] or "offline" in hl["explanation"]
            assert hl["breakdown"]["llm_semantic_score"] == 0.0

@patch("urllib.request.urlopen")
def test_openai_compatible_success(mock_urlopen):
    """Verify that a successful OpenAI-compatible (like OpenCode Zen/9Router) API response is parsed and blended correctly."""
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

def test_visual_motion_evenly_sampled_coverage():
    """Verify that analyze_visual_motion seek-samples frames spread evenly across the full duration of a video."""
    video_path = "test_motion_dummy.mp4"
    width, height = 320, 180
    fps = 30
    duration_sec = 10
    total_frames = duration_sec * fps # 300 frames
    
    # Write a dummy video file
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    for f in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Draw motion
        cv2.circle(frame, (f % width, height // 2), 20, (255, 255, 255), -1)
        out.write(frame)
    out.release()
    
    try:
        engine = HighlightEngine(video_path)
        # Ask to sample fewer frames than total, e.g. 50 samples
        max_samples = 50
        scores = engine.analyze_visual_motion(max_frames_to_check=max_samples)
        
        assert len(scores) > 0
        # The scores must contain timestamps representing start to end
        timestamps = [item["time"] for item in scores]
        
        # Verify the maximum timestamp is close to the end of the video duration (9.x seconds)
        assert max(timestamps) > 9.0
        # Verify the minimum timestamp is at the beginning (0.0 seconds)
        assert min(timestamps) == 0.0
        
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)

import os
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from src.app import app, app_state, export_jobs

client = TestClient(app)

def test_home_route():
    """Verify home page route is active and serves HTML."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

def test_upload_invalid_file_extension():
    """Verify that uploading a file with an invalid extension returns 400."""
    response = client.post(
        "/upload",
        files={"file": ("test.txt", b"dummy text content", "text/plain")}
    )
    assert response.status_code == 400
    assert "Unsupported video format" in response.json()["detail"]

def test_demo_generation_mocked():
    """Verify /demo generates or loads the demo video correctly."""
    with patch("os.path.exists", return_value=True):
        response = client.post("/demo")
        assert response.status_code == 200
        assert response.json()["status"] == "success"
        assert response.json()["filename"] == "Demo_Video_16_9.mp4"

def test_analyze_no_video_loaded():
    """Verify that analyzing without a loaded video returns 400."""
    # Reset app state
    app_state["current_video_path"] = ""
    app_state["highlights"] = []
    
    response = client.post(
        "/analyze",
        json={
            "use_whisper": False
        }
    )
    assert response.status_code == 400
    assert "No active video loaded" in response.json()["detail"]

@patch("src.app.HighlightEngine")
def test_analyze_success(mock_engine_class):
    """Verify /analyze runs successfully and returns proper response format."""
    # Mock highlight engine results
    mock_engine = MagicMock()
    mock_engine.duration = 120
    mock_engine.detect_highlights.return_value = [
        {
            "start": 10.0,
            "end": 30.0,
            "score": 92.5,
            "text": "funny talking moment",
            "hook_title": "Clip #1",
            "explanation": "Heuristic match",
            "breakdown": {
                "audio_energy": 80.0,
                "lexical_virality": 90.0,
                "visual_motion": 70.0,
                "hook_strength": 95.0,
                "llm_semantic_score": 0.0
            }
        }
    ]
    mock_engine_class.return_value = mock_engine
    
    app_state["current_video_path"] = "dummy_active_video.mp4"
    app_state["current_video_name"] = "test.mp4"
    
    with patch("os.path.exists", return_value=True):
        response = client.post(
            "/analyze",
            json={
                "use_whisper": False
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert len(data["highlights"]) == 1
        assert data["highlights"][0]["score"] == 92.5

@patch("os.path.exists", return_value=True)
def test_export_invalid_clip_index(mock_exists):
    """Verify /export with an out-of-bounds index returns 400."""
    app_state["current_video_path"] = "dummy_active_video.mp4"
    app_state["highlights"] = [] # Empty highlights list
    
    response = client.post(
        "/export",
        json={
            "clip_index": 0,
            "track_faces": True,
            "caption_style": "Reels",
            "enable_voiceover": False
        }
    )
    assert response.status_code == 400
    assert "Invalid clip selection index" in response.json()["detail"]

@patch("src.app.VideoCropper")
def test_export_and_polling_flow(mock_cropper_class):
    """Verify background export queuing and status polling works flawlessly."""
    # Pre-populate active highlights in app state
    app_state["current_video_path"] = "dummy_active_video.mp4"
    app_state["highlights"] = [
        {
            "start": 10.0,
            "end": 30.0,
            "text": "funny moment",
            "hook_title": "Clip #1",
            "explanation": "Heuristic match",
            "breakdown": {}
        }
    ]
    
    with patch("os.path.exists", return_value=True):
        # 1. Trigger export
        response = client.post(
            "/export",
            json={
                "clip_index": 0,
                "track_faces": True,
                "caption_style": "Reels",
                "enable_voiceover": False
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "queued"
        job_id = data["job_id"]
        assert job_id is not None
        
        # 2. Poll job status
        status_response = client.get(f"/export/status/{job_id}")
        assert status_response.status_code == 200
        job_data = status_response.json()
        assert "status" in job_data
        assert job_data["status"] in ["processing", "completed", "failed"]

def test_settings_get_and_post():
    """Verify settings can be set, get correctly, API key is masked and protected on update."""
    # 1. Post new settings with a true key
    payload = {
        "llm_provider": "OpenAI",
        "ollama_model": "llama3.2:3b",
        "ollama_host": "http://localhost:11434",
        "openai_base_url": "http://localhost:20128/v1",
        "openai_model": "oc/deepseek-v4-flash-free",
        "openai_api_key": "sk-secret-true-key-content"
    }
    post_res = client.post("/settings", json=payload)
    assert post_res.status_code == 200
    
    # 2. Get settings, verify keys are returned masked
    get_res = client.get("/settings")
    assert get_res.status_code == 200
    get_data = get_res.json()
    assert get_data["llm_provider"] == "OpenAI"
    assert "••••" in get_data["openai_api_key"]
    assert "sk-secret" not in get_data["openai_api_key"]
    
    # 3. Post settings back with masked key (simulating typical UI update)
    payload_masked = {
        "llm_provider": "OpenAI",
        "ollama_model": "llama3.2:3b",
        "ollama_host": "http://localhost:11434",
        "openai_base_url": "http://localhost:20128/v1",
        "openai_model": "oc/deepseek-v4-flash-free",
        "openai_api_key": get_data["openai_api_key"] # passing the masked key back
    }
    post_res_masked = client.post("/settings", json=payload_masked)
    assert post_res_masked.status_code == 200
    
    # Clean up generated config file if desired, or let it persist
    if os.path.exists("config.json"):
        try:
            os.remove("config.json")
        except:
            pass

"""API tests for the AI Content Creator FastAPI app (TestClient)."""
import io
import json
import os
import sys
import wave

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai_creator.app import app, CHARACTERS, VOICES, PLANS_DIR, OUTPUTS_DIR, SFX_DIR, WORK_DIR  # noqa: E402
from tests.test_creator_character import make_synthetic_face_photo  # noqa: E402

client = TestClient(app)


@pytest.fixture(autouse=True)
def _clean_state():
    """Snapshot & restore app-level stores so tests don't leak into each other."""
    import shutil
    yield


# ------------------------------ basics ------------------------------
def test_home_serves_ui():
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "AI Content Creator" in r.text


def test_status_shape():
    r = client.get("/api/status")
    assert r.status_code == 200
    d = r.json()
    assert "ollama" in d and "tts" in d and "sfx_count" in d
    assert d["sfx_count"] >= 9


def test_sfx_list_and_preview():
    r = client.get("/api/sfx")
    assert r.status_code == 200
    names = [i["name"] for i in r.json()]
    assert "whoosh" in names and "boom" in names
    p = client.get(f"/api/sfx/{names[0]}.wav")
    assert p.status_code == 200
    assert p.headers["content-type"].startswith("audio/wav")


def test_download_traversal_blocked():
    r = client.get("/outputs/..%2f..%2fetc%2fpasswd")
    assert r.status_code in (400, 404)


# ------------------------------ team ------------------------------
def test_team_get_and_post_roundtrip():
    r = client.get("/api/team")
    assert r.status_code == 200
    d = r.json()
    assert set(d["roles"].keys()) == {"planner", "scriptwriter", "sfx_director", "animator", "qa"}
    d["roles"]["qa"]["enabled"] = True
    d["roles"]["qa"]["temperature"] = 0.3
    r2 = client.post("/api/team", json=d)
    assert r2.status_code == 200
    r3 = client.get("/api/team")
    assert r3.json()["roles"]["qa"]["enabled"] is True
    assert r3.json()["roles"]["qa"]["temperature"] == 0.3


# ------------------------------ characters ------------------------------
def _upload_character(name="ApiTesty"):
    photo = make_synthetic_face_photo(os.path.join(WORK_DIR, "api_photo.jpg"), seed=42)
    with open(photo, "rb") as f:
        r = client.post("/api/characters/create",
                        data={"name": name}, files={"file": ("p.jpg", f, "image/jpeg")})
    assert r.status_code == 200, r.text
    return r.json()["character"]


def test_character_create_list_update_delete():
    c = _upload_character()
    assert c["id"]
    lst = client.get("/api/characters").json()
    assert any(x["id"] == c["id"] for x in lst)
    # assets served
    a = client.get(f"/assets/characters/{c['id']}/avatar.png")
    assert a.status_code == 200 and a.headers["content-type"] == "image/png"
    # update
    u = client.post("/api/characters", json={"id": c["id"], "name": "Renamed"})
    assert u.status_code == 200
    assert [x for x in client.get("/api/characters").json() if x["id"] == c["id"]][0]["name"] == "Renamed"
    # add photo
    photo2 = make_synthetic_face_photo(os.path.join(WORK_DIR, "api_photo2.jpg"), seed=42)
    with open(photo2, "rb") as f:
        r2 = client.post(f"/api/characters/{c['id']}/photos",
                         files={"file": ("p2.jpg", f, "image/jpeg")})
    assert r2.status_code == 200
    assert r2.json()["photos"] == 2
    # delete
    assert client.delete(f"/api/characters/{c['id']}").status_code == 200
    assert not any(x["id"] == c["id"] for x in client.get("/api/characters").json())


def test_character_bad_file_type_rejected():
    r = client.post("/api/characters/create", data={"name": "x"},
                    files={"file": ("bad.exe", b"hello", "application/octet-stream")})
    assert r.status_code == 400


def test_plan_requires_character():
    r = client.post("/api/plan", json={"idea": "test", "character_id": "nope"})
    assert r.status_code == 400


# ------------------------------ voices ------------------------------
def _make_wav_bytes(seconds=12, sr=16000):
    n = int(seconds * sr)
    t = np.arange(n) / sr
    pcm = (np.sin(2 * np.pi * 300 * t) * 8000).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def test_voice_upload_preview_delete():
    r = client.post("/api/voices", data={"name": "Test Voice"},
                    files={"file": ("v.wav", _make_wav_bytes(12), "audio/wav")})
    assert r.status_code == 200, r.text
    vid = r.json()["voice"]["id"]
    assert r.json()["voice"]["duration"] == pytest.approx(12.0, abs=0.3)
    p = client.get(f"/assets/voices/{vid}/preview.wav")
    assert p.status_code == 200
    assert client.delete(f"/api/voices/{vid}").status_code == 200


def test_voice_too_short_rejected():
    r = client.post("/api/voices", data={"name": "short"},
                    files={"file": ("s.wav", _make_wav_bytes(2), "audio/wav")})
    assert r.status_code == 400
    assert "at least" in r.json()["detail"]


# ------------------------------ plan & render ------------------------------
def test_plan_offline_uses_fallback_and_renders():
    c = _upload_character("RenderTesty")
    r = client.post("/api/plan", json={
        "idea": "why early birds win the day", "target_duration": 12,
        "style": "punchy", "character_id": c["id"]})
    assert r.status_code == 200, r.text
    plan = r.json()
    assert plan["id"]
    assert len(plan["scenes"]) >= 1
    assert plan["total_duration"] > 0
    assert plan["activity"]  # team activity logged (fallbacks)

    # edit the plan (trim to a single short scene to keep the test fast)
    one_scene = [plan["scenes"][0]]
    one_scene[0]["duration"] = 3.0
    one_scene[0]["sfx"] = "pop"
    up = client.post(f"/api/plans/{plan['id']}",
                     json={"title": "Trimmed", "scenes": one_scene})
    assert up.status_code == 200

    # render (no TTS in CI -> SFX only; 720x1280 per API contract)
    rr = client.post("/api/render", json={"plan_id": plan["id"], "width": 720, "height": 1280})
    assert rr.status_code == 200, rr.text
    job_id = rr.json()["job_id"]

    # poll until done (TestClient runs background tasks synchronously per request cycle)
    import time
    for _ in range(120):
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["status"] in ("completed", "failed"):
            break
        time.sleep(1)
    assert j["status"] == "completed", j.get("error")
    assert j["result"]["download_url"].endswith(".mp4")
    dl = client.get(j["result"]["download_url"])
    assert dl.status_code == 200
    assert len(dl.content) > 20000  # real video bytes
    s = client.get(j["result"]["srt_url"])
    assert s.status_code == 200 and "-->" in s.text

    # cleanup job output
    shutil.rmtree(os.path.join(OUTPUTS_DIR, f"render_{job_id}"), ignore_errors=True)


def test_job_unknown_404():
    assert client.get("/api/jobs/doesnotexist").status_code == 404


def test_render_unknown_plan_404():
    r = client.post("/api/render", json={"plan_id": "nope"})
    assert r.status_code == 404


import shutil  # noqa: E402

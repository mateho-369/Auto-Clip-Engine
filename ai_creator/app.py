"""AI Content Creator — FastAPI server.

Local studio: character memory + AI team (Ollama) + TTS/voice cloning +
SFX + animation/transitions + renderer. Everything is served from one
process; state (characters, voices, plans) persists on disk.
"""
import os
import json
import shutil
import uuid
import time
import threading
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional as Opt

from ai_creator.team import load_config, save_config, normalize_config, ROLES, ROLE_LABELS, ROLE_DESCRIPTIONS
from ai_creator.ollama_client import OllamaClient
from ai_creator.planner import Studio, fallback_plan, new_plan_id, validate_plan
from ai_creator.character import CharacterStore
from ai_creator.voice import VoiceStore, TTSEngine
from ai_creator.renderer import Renderer
from ai_creator import sfx as sfx_mod

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

app = FastAPI(title="AI Content Creator — Local AI + Bot Video Studio")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

CHARACTERS = CharacterStore(ROOT)
VOICES = VoiceStore(ROOT)
TTS = TTSEngine(weights_dir=ROOT)
TEAM_CONFIG_PATH = os.path.join(ROOT, "team_config.json")
PLANS_DIR = os.path.join(ROOT, "plans")
OUTPUTS_DIR = os.path.join(ROOT, "outputs")
SFX_DIR = os.path.join(ROOT, "sfx_library")
WORK_DIR = os.path.join(ROOT, "work")
os.makedirs(PLANS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(WORK_DIR, exist_ok=True)
sfx_mod.ensure_library(SFX_DIR)

jobs = {}


def team_config():
    return load_config(TEAM_CONFIG_PATH)


def ollama_client():
    return OllamaClient(team_config().get("ollama_host", "http://localhost:11434"))


# ------------------------------ UI ------------------------------
@app.get("/")
async def home():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates", "index.html")
    if os.path.exists(path):
        return FileResponse(path, media_type="text/html")
    return {"message": "AI Content Creator running. UI template missing."}


# ------------------------------ status ------------------------------
@app.get("/api/status")
async def api_status():
    client = ollama_client()
    online = client.is_online()
    return {
        "ollama": {"online": online,
                   "host": team_config().get("ollama_host"),
                   "models": client.list_models() if online else []},
        "tts": TTS.probe(),
        "sfx_count": len(sfx_mod.list_sfx(SFX_DIR)),
        "characters": len(CHARACTERS.list()),
        "voices": len(VOICES.list()),
        "ffmpeg": shutil.which("ffmpeg") is not None or _ffmpeg_present(),
    }


def _ffmpeg_present():
    try:
        import imageio_ffmpeg
        return os.path.exists(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        return False


# --------------------------- AI team ---------------------------
@app.get("/api/team")
async def api_team_get():
    cfg = team_config()
    return {
        "ollama_host": cfg["ollama_host"],
        "controller": cfg["controller"],
        "roles": cfg["roles"],
        "roles_meta": ROLE_LABELS,
        "roles_desc": ROLE_DESCRIPTIONS,
    }


class TeamConfigModel(BaseModel):
    ollama_host: str = "http://localhost:11434"
    controller: str = "llama3.2:3b"
    roles: dict = {}


@app.post("/api/team")
async def api_team_post(req: TeamConfigModel):
    cfg = normalize_config({"ollama_host": req.ollama_host, "controller": req.controller,
                            "roles": req.roles or {}})
    if save_config(TEAM_CONFIG_PATH, cfg):
        return {"status": "success", "message": "AI team updated."}
    raise HTTPException(500, "Could not save team config.")


@app.get("/api/ollama/models")
async def api_ollama_models():
    client = ollama_client()
    if not client.is_online():
        return {"online": False, "models": []}
    return {"online": True, "models": client.list_models()}


# --------------------------- characters ---------------------------
@app.get("/api/characters")
async def api_characters():
    out = []
    for prof in CHARACTERS.list():
        out.append({
            "id": prof["id"], "name": prof["name"], "photos": prof.get("photos", 1),
            "palette": prof.get("palette", []),
            "face_detected": prof.get("face", {}).get("detected", False),
            "voice_id": prof.get("voice_id", ""),
            "style_notes": prof.get("style_notes", ""),
            "created": prof.get("created"),
            "assets": {"face": f"/assets/characters/{prof['id']}/face.png",
                       "avatar": f"/assets/characters/{prof['id']}/avatar.png"},
        })
    return out


class CharacterModel(BaseModel):
    id: Opt[str] = None
    name: Opt[str] = None
    voice_id: Opt[str] = None
    style_notes: Opt[str] = None


@app.post("/api/characters")
async def api_character_update(req: CharacterModel):
    if not req.id:
        raise HTTPException(400, "Character id required.")
    prof = CHARACTERS.update(req.id, name=req.name, voice_id=req.voice_id, style_notes=req.style_notes)
    if prof is None:
        raise HTTPException(404, "Character not found.")
    return {"status": "success"}


@app.delete("/api/characters/{char_id}")
async def api_character_delete(char_id: str):
    if not CHARACTERS.delete(char_id):
        raise HTTPException(404, "Character not found.")
    return {"status": "success"}


@app.post("/api/characters/{char_id}/photos")
async def api_character_photo(char_id: str, file: UploadFile = File(...)):
    prof = CHARACTERS.get(char_id)
    if prof is None:
        raise HTTPException(404, "Create the character first (upload the first photo).")
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
        raise HTTPException(400, "Send a photo (JPG/PNG/WEBP).")
    tmp = os.path.join(WORK_DIR, f"photo_{uuid.uuid4().hex[:6]}{ext}")
    with open(tmp, "wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        new_prof = CHARACTERS.add_photo(char_id, tmp)
    except ValueError as e:
        raise HTTPException(400, str(e))
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    if new_prof is None:
        raise HTTPException(500, "Could not process photo.")
    return {"status": "success",
            "similarity": new_prof.get("last_similarity"),
            "photos": new_prof.get("photos")}


@app.post("/api/characters/create")
async def api_character_create(name: str = Form(...), file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
        raise HTTPException(400, "Send a photo (JPG/PNG/WEBP).")
    tmp = os.path.join(WORK_DIR, f"photo_{uuid.uuid4().hex[:6]}{ext}")
    with open(tmp, "wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        prof = CHARACTERS.create(name.strip() or "My Character", tmp)
    except ValueError as e:
        raise HTTPException(400, str(e))
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return {"status": "success",
            "character": {"id": prof["id"], "name": prof["name"],
                          "face_detected": prof.get("face", {}).get("detected", False),
                          "palette": prof.get("palette", [])}}


@app.get("/assets/characters/{char_id}/{asset}")
async def api_character_asset(char_id: str, asset: str):
    prof = CHARACTERS.get(char_id)
    if prof is None:
        raise HTTPException(404, "Character not found.")
    if asset not in ("face.png", "avatar.png"):
        raise HTTPException(400, "Unknown asset.")
    path = os.path.join(prof["dir"], asset)
    if not os.path.exists(path):
        raise HTTPException(404, "Asset not built yet.")
    return FileResponse(path, media_type="image/png")


@app.get("/assets/voices/{voice_id}/preview.wav")
async def api_voice_preview(voice_id: str):
    meta = VOICES.get(voice_id)
    if meta is None:
        raise HTTPException(404, "Voice not found.")
    path = os.path.join(meta["dir"], "recording.wav")
    if not os.path.exists(path):
        raise HTTPException(404, "Recording missing.")
    return FileResponse(path, media_type="audio/wav")


# ----------------------------- voices -----------------------------
@app.get("/api/voices")
async def api_voices():
    out = []
    for meta in VOICES.list():
        out.append({"id": meta["id"], "name": meta["name"], "duration": meta.get("duration"),
                    "created": meta.get("created"), "source": meta.get("source"),
                    "clone_engine": meta.get("clone_engine"),
                    "preview": f"/assets/voices/{meta['id']}/preview.wav"})
    return out


@app.post("/api/voices")
async def api_voice_upload(name: str = Form(""), file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in (".wav", ".mp3", ".m4a", ".webm", ".ogg", ".flac"):
        raise HTTPException(400, "Send a voice recording (WAV/MP3/M4A/WEBM/OGG).")
    tmp = os.path.join(WORK_DIR, f"voice_{uuid.uuid4().hex[:6]}{ext}")
    with open(tmp, "wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        meta = VOICES.add(name.strip() or "My Voice", tmp)
    except ValueError as e:
        raise HTTPException(400, str(e))
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return {"status": "success", "voice": meta}


@app.delete("/api/voices/{voice_id}")
async def api_voice_delete(voice_id: str):
    if not VOICES.delete(voice_id):
        raise HTTPException(404, "Voice not found.")
    return {"status": "success"}


# ----------------------------- SFX -----------------------------
@app.get("/api/sfx")
async def api_sfx():
    return sfx_mod.list_sfx(SFX_DIR)


@app.get("/api/sfx/{name}.wav")
async def api_sfx_play(name: str):
    samples, sr = sfx_mod.load_sfx(SFX_DIR, name)
    if samples is None:
        raise HTTPException(404, "SFX not found.")
    import wave
    import io
    import numpy as np
    from fastapi import Response
    buf = io.BytesIO()
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return Response(content=buf.getvalue(), media_type="audio/wav",
                    headers={"Content-Disposition": f'attachment; filename="{name}.wav"'})


# ----------------------------- planning -----------------------------
class PlanRequest(BaseModel):
    idea: str
    target_duration: int = 25
    style: str = ""
    character_id: Opt[str] = None


@app.post("/api/plan")
async def api_plan(req: PlanRequest):
    idea = (req.idea or "").strip()
    if not idea:
        raise HTTPException(400, "Describe your video idea first.")
    char = CHARACTERS.get(req.character_id) if req.character_id else None
    if char is None:
        raise HTTPException(400, "Pick a character first — the video stars your character.")

    cfg = team_config()
    client = ollama_client()
    studio = Studio(cfg, client)
    plan = studio.plan(idea, req.target_duration, req.style, char.get("name", ""))
    plan_id = new_plan_id()
    plan["id"] = plan_id
    plan["character_id"] = char["id"]
    plan["created"] = time.time()
    with open(os.path.join(PLANS_DIR, f"{plan_id}.json"), "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)
    return plan


class PlanUpdateModel(BaseModel):
    title: Opt[str] = None
    logline: Opt[str] = None
    scenes: List[dict] = []


@app.post("/api/plans/{plan_id}")
async def api_plan_update(plan_id: str, req: PlanUpdateModel):
    path = os.path.join(PLANS_DIR, f"{plan_id}.json")
    if not os.path.exists(path):
        raise HTTPException(404, "Plan not found (server may have restarted — re-plan).")
    with open(path, "r", encoding="utf-8") as f:
        plan = json.load(f)
    if req.title is not None:
        plan["title"] = req.title[:120]
    if req.logline is not None:
        plan["logline"] = req.logline[:300]
    if req.scenes:
        plan["scenes"] = req.scenes
    try:
        # merge: validate returns a fresh dict — keep identity fields
        # (character_id, activity, idea, created) from the stored plan.
        validated = validate_plan(plan)
        plan = {**plan, **validated}
        plan["total_duration"] = round(sum(s["duration"] for s in plan["scenes"]), 2)
    except ValueError as e:
        raise HTTPException(400, str(e))
    plan["id"] = plan_id
    with open(path, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)
    return {"status": "success", "total_duration": plan["total_duration"]}


# ----------------------------- rendering -----------------------------
class RenderRequest(BaseModel):
    plan_id: str
    voice_id: Opt[str] = None
    kokoro_voice: str = "af_bella"
    width: int = 720
    height: int = 1280


def perform_render(job_id, req: RenderRequest, plan, char, voice_meta):
    try:
        def progress(stage, pct):
            jobs[job_id]["stage"] = stage
            jobs[job_id]["progress"] = int(pct)

        renderer = Renderer(SFX_DIR, WORK_DIR)
        voice_cfg = None
        if voice_meta:
            voice_cfg = {"voice_id": voice_meta["id"], "kokoro_voice": req.kokoro_voice,
                         "voices_root": VOICES.root}
        result = renderer.render(
            plan, char["dir"], TTS, voice_cfg,
            width=req.width, height=req.height, fps=24,
            out_dir=OUTPUTS_DIR, progress=progress,
        )
        # keep a per-render copy so downloads don't collide
        run_dir = os.path.join(OUTPUTS_DIR, f"render_{job_id}")
        os.makedirs(run_dir, exist_ok=True)
        for fn in (result["mp4"], result["srt"]):
            src = os.path.join(OUTPUTS_DIR, fn)
            if os.path.exists(src):
                shutil.copyfile(src, os.path.join(run_dir, fn))
        jobs[job_id] = {
            "status": "completed", "stage": "Render complete!", "progress": 100,
            "result": {
                "download_url": f"/outputs/{job_id}/{result['mp4']}",
                "srt_url": f"/outputs/{job_id}/{result['srt']}",
                **result,
            },
            "error": None,
        }
    except Exception as e:
        print(f"Render job {job_id} failed: {e}")
        import traceback
        traceback.print_exc()
        jobs[job_id] = {"status": "failed", "stage": "Render failed", "progress": 0,
                        "result": None, "error": str(e)}


@app.post("/api/render")
async def api_render(req: RenderRequest):
    path = os.path.join(PLANS_DIR, f"{req.plan_id}.json")
    if not os.path.exists(path):
        raise HTTPException(404, "Plan not found — run the planner first.")
    with open(path, "r", encoding="utf-8") as f:
        plan = json.load(f)
    char = CHARACTERS.get(plan.get("character_id", ""))
    if char is None:
        raise HTTPException(400, "The plan's character no longer exists.")
    voice_meta = None
    if req.voice_id:
        voice_meta = VOICES.get(req.voice_id)
    elif char.get("voice_id"):
        voice_meta = VOICES.get(char["voice_id"])
    if (req.width, req.height) not in ((720, 1280), (1080, 1920), (1280, 720)):
        raise HTTPException(400, "Use 720x1280, 1080x1920 or 1280x720.")

    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {"status": "processing", "stage": "Starting render...", "progress": 0,
                    "result": None, "error": None}
    # Run in a dedicated thread (NOT a BackgroundTask): starlette executes
    # sync background tasks on the event loop, which would freeze the whole
    # server (and the UI's status polling) for the entire render.
    threading.Thread(target=perform_render, args=(job_id, req, plan, char, voice_meta),
                     daemon=True).start()
    return {"status": "queued", "job_id": job_id}


@app.get("/api/jobs/{job_id}")
async def api_job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found.")
    return jobs[job_id]


# ----------------------------- downloads -----------------------------
@app.get("/outputs/{job_id}/{filename}")
async def api_output(job_id: str, filename: str):
    if "/" in filename or "\\" in filename or filename.startswith("."):
        raise HTTPException(400, "Invalid filename.")
    path = os.path.join(OUTPUTS_DIR, f"render_{job_id}", filename)
    if not os.path.exists(path):
        raise HTTPException(404, "File not found.")
    media = "video/mp4" if filename.endswith(".mp4") else "application/x-subrip"
    return FileResponse(path, media_type=media, filename=filename)

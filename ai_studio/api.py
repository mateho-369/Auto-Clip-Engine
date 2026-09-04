"""The studio's HTTP + live API.

Grouped the way the UI needs it:

* projects & history (search / sort / duplicate / reuse / delete)
* the two input modes (Mode A script lock, Mode B generate → review → approve)
* runs: start / cancel / pause / resume / **continue-from-last-good** /
  **regenerate one stage** / GPU catch-up after moving to Machine A
* per-stage inspection: scene board, asset streams (Range-capable), prompt log
* settings: role→model per agent, engine choices, VRAM safety, voice profiles
* live status: WebSocket + SSE + polling snapshot (the UI degrades in that order)
* downloads: single asset, whole project, or a zip of one scene's folder

Everything is localhost: no telemetry, no cloud calls.
"""
import asyncio
import json
import os
import shutil
import time

from fastapi import (APIRouter, Body, Depends, File, Form, HTTPException, Query, Request,
                     UploadFile, WebSocket, WebSocketDisconnect)
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse

from . import __version__, config as cfg_mod, content as content_mod, khmer, media, style as style_mod
from . import vram as vram_mod
from .db import Database
from .events import RunProgress
from .pipeline import spec as stagespec
from .util import (ensure_dir, fmt_dur, human_size, jdump, media_duration, new_id, now,
                   read_json, wav_peaks, write_json, zip_paths)

router = APIRouter(prefix="/api")

STATE = {}          # populated by create_app() / get_state()


def get_state(request: Request = None):
    st = STATE.get("app")
    if st is None:
        raise HTTPException(500, "studio not initialised")
    return st


# ============================================================== status / health
@router.get("/status")
async def api_status(deep: bool = Query(False)):
    st = get_state()
    cfg = st.config()
    # StudioState.plan() has a 25s TTL cache specifically so this endpoint
    # (polled by the UI every ~15s) doesn't re-probe Ollama/ComfyUI/RVC on
    # every tick. Hardcoding refresh=True bypassed that cache entirely,
    # forcing a full synchronous probe cycle on every poll — measured 8-20s
    # per call once Ollama had a generation in flight, stalling the UI boot.
    # Only `?deep=true` (an explicit, infrequent request) should force it.
    plan = st.plan(refresh=deep)
    caps = await asyncio.to_thread(cfg_mod.capabilities, cfg) if deep else {}
    return {"studio": "khmer-ai-content-studio", "version": __version__,
            "data_dir": st.data_root, "db": st.db.stats(),
            "machine": plan.get("hardware"), "plan": plan, "capabilities": caps,
            "vram": await asyncio.to_thread(vram_mod.status, cfg, plan),
            "active_runs": [r["id"] for r in st.db.active_runs()],
            "ffmpeg": await asyncio.to_thread(lambda: __import__("ai_studio.util",
                                                                  fromlist=["ffmpeg_exe"])
                                              .ffmpeg_exe())}


@router.get("/health")
async def api_health():
    return {"ok": True, "ts": now()}


# ==================================================================== projects
@router.get("/projects")
async def api_projects(search: str = "", status: str = "", mode: str = "",
                      sort: str = "updated_desc", limit: int = 200, offset: int = 0):
    st = get_state()
    rows = st.db.list_projects(search=search, status=status, mode=mode, sort=sort,
                               limit=min(500, max(1, limit)), offset=max(0, offset))
    return {"projects": rows, "counts": st.db.project_counts()}


class ProjectBody:
    pass


@router.post("/projects")
async def api_create_project(payload: dict = Body(...)):
    st = get_state()
    mode = str(payload.get("mode") or "A").strip().upper()[:1]
    if mode not in ("A", "B"):
        raise HTTPException(400, "mode must be 'A' (Director script) or 'B' (auto idea)")
    script = khmer.normalize_block(payload.get("script") or "")
    # Khmer has no spaces between words: every truncation here is by
    # *character cluster*, so a coeng/subscript pair can never be bisected.
    topic = khmer.clip_clusters(khmer.strip_emoji_and_marks(payload.get("topic_hint") or ""), 400)
    if mode == "A" and not script:
        raise HTTPException(400, "Mode A needs the finished script pasted in — the studio will "
                                 "never write or rewrite it for you.")
    if mode == "A" and len(script) < 12:
        raise HTTPException(400, "That script looks too short to segment — paste the full text.")
    from . import content as content_mod

    title = khmer.clip_clusters((payload.get("title") or "").strip(), 120) or (
        khmer.title_from(script) if script else
        (khmer.truncate_clusters(khmer.strip_emoji_and_marks(topic), 60) or "Auto idea short"))
    content_type = content_mod.normalize(payload.get("content_type") or "explainer")
    character_id = str(payload.get("character_id") or "")
    if character_id and not st.db.get_character(character_id):
        raise HTTPException(400, "character_id does not match any saved character")
    proj = st.db.create_project(
        title=title, mode=mode, status="draft", language=payload.get("language") or "km",
        script=script if mode == "A" else "", script_locked=(mode == "A"),
        script_origin="director" if mode == "A" else "", topic_hint=topic,
        content_type=content_type, character_id=character_id,
        style_notes=(payload.get("style_notes") or "")[:800],
        target_duration=float(payload.get("target_duration") or 30),
        voice_profile_id=payload.get("voice_profile_id") or "",
        settings=payload.get("settings") or {}, parent_id=payload.get("parent_id") or "")
    st.bus.publish("project_created", {"project_id": proj["id"], "mode": mode,
                                       "title": proj["title"]}, project_id=proj["id"])
    if mode == "B" and payload.get("generate_now", True):
        res = await _generate_idea(proj["id"])
        proj = st.db.get_project(proj["id"])
        proj["generated"] = res
    return {"project": {k: v for k, v in proj.items() if k != "scenes"},
            "scenes": proj.get("scenes", [])}


async def _generate_idea(project_id):
    """Mode B: run ONLY the script stage, so the Director can approve first."""
    st = get_state()
    out = await st.scheduler.start_run(project_id, trigger="new", force_stages=["script"],
                                       auto_start=True)
    run_id = out["run_id"]
    done = await st.scheduler.wait(run_id, timeout=900)
    run = (done or {}).get("run") or {}
    project = st.db.get_project(project_id) or {}
    return {"run_id": run_id, "status": run.get("status"), "script": project.get("script"),
            "title": project.get("title"), "origin": project.get("script_origin"),
            "error": run.get("error") or ""}


@router.get("/projects/{project_id}")
async def api_project(project_id: str):
    st = get_state()
    proj = st.db.get_project(project_id)
    if not proj:
        raise HTTPException(404, "project not found")
    runs = st.db.list_runs(project_id, limit=40)
    for r in runs:
        r["summary"] = RunProgress.rollup(st.db.list_stages(r["id"]))
        r["overall"] = RunProgress.overall(st.db.list_stages(r["id"]))
        r["assets_count"] = len(st.db.list_assets(run_id=r["id"], limit=999))
    latest = runs[0] if runs else None
    return {"project": proj, "scenes": st.db.list_scenes(project_id), "runs": runs,
            "latest_run_id": (latest or {}).get("id"),
            "prompts": st.db.list_prompts(project_id=project_id, limit=60),
            "assets": st.db.list_assets(project_id=project_id, limit=400),
            "integrity": _integrity_report(proj, st.db.list_scenes(project_id)),
            "disk": _project_disk(project_id)}


def _integrity_report(proj, scenes):
    """The Mode-A promise, computed (not asserted): did the words survive?"""
    if (proj.get("mode") or "A").upper() != "A":
        return {"applies": False, "ok": True,
                "detail": "Mode B — the Controller wrote this script, edits are allowed"}
    joined = khmer.join_sentences([s.get("text", "") for s in scenes])
    ok = khmer.equal_text(joined, proj.get("script") or "") if scenes else None
    return {"applies": True, "ok": ok,
            "detail": ("scene text rejoins the Director's script exactly" if ok else
                       ("not segmented yet" if ok is None else
                        "⚠ wording differs from the pasted script")),
            "script_chars": khmer.char_len(proj.get("script") or ""),
            "scene_chars": khmer.char_len(joined)}


def _project_disk(project_id):
    st = get_state()
    d = os.path.join(st.data_root, "projects", project_id)
    total = files = 0
    if os.path.isdir(d):
        for root, _dirs, fns in os.walk(d):
            for fn in fns:
                try:
                    total += os.path.getsize(os.path.join(root, fn))
                    files += 1
                except Exception:
                    pass
    return {"bytes": total, "files": files, "human": human_size(total), "path": d}


@router.patch("/projects/{project_id}")
@router.post("/projects/{project_id}")
async def api_update_project(project_id: str, payload: dict = Body(...)):
    st = get_state()
    if not st.db.get_project(project_id):
        raise HTTPException(404, "project not found")
    from . import content as content_mod

    allowed = {"title", "status", "topic_hint", "style_notes", "target_duration",
               "voice_profile_id", "language", "mode", "content_type", "character_id"}
    kw = {k: v for k, v in payload.items() if k in allowed}
    if "content_type" in kw:
        kw["content_type"] = content_mod.normalize(kw["content_type"])
    if "character_id" in kw:
        cid = str(kw["character_id"] or "")
        if cid and not st.db.get_character(cid):
            raise HTTPException(400, "character_id does not match any saved character")
        kw["character_id"] = cid
    if "script" in payload:
        proj = st.db.get_project(project_id)
        if (proj.get("mode") or "A").upper() == "A" and not payload.get("director_override"):
            raise HTTPException(403, "Mode A scripts are locked — the Director edits them, "
                                     "never the studio. (Pass director_override=true if you "
                                     "really are the Director.)")
        kw["script"] = khmer.normalize_block(payload["script"])
    if "settings" in payload:
        kw["settings"] = payload["settings"] or {}
    if not kw:
        raise HTTPException(400, "nothing to update")
    return {"project": st.db.update_project(project_id, **kw)}


@router.delete("/projects/{project_id}")
async def api_delete_project(project_id: str, purge_files: bool = Query(False)):
    st = get_state()
    d = os.path.join(st.data_root, "projects", project_id)
    st.db.delete_project(project_id)
    removed = False
    if purge_files and os.path.isdir(d):
        shutil.rmtree(d, ignore_errors=True)
        removed = True
    return {"ok": True, "files_removed": removed}


@router.post("/projects/{project_id}/duplicate")
async def api_duplicate(project_id: str, payload: dict = Body(default={})):
    """Re-use a project as a starting point (script, scenes, per-stage settings)."""
    st = get_state()
    src = st.db.get_project(project_id)
    if not src:
        raise HTTPException(404, "project not found")
    keep_scenes = bool(payload.get("keep_scenes", True))
    new_mode = str(payload.get("mode") or src.get("mode") or "A").upper()[:1]
    title = khmer.clip_clusters(payload.get("title") or f"{src.get('title', 'Project')} (copy)", 120)
    proj = st.db.create_project(title=title, mode=new_mode, status="draft",
                                language=src.get("language") or "km",
                                script=src.get("script") or "",
                                script_locked=(new_mode == "A"),
                                script_origin="director" if new_mode == "A" else (
                                    src.get("script_origin") or ""),
                                topic_hint="" if new_mode == "A" else src.get("topic_hint", ""),
                                content_type=src.get("content_type") or "explainer",
                                character_id=src.get("character_id") or "",
                                style_notes=src.get("style_notes") or "",
                                target_duration=src.get("target_duration") or 30,
                                voice_profile_id=src.get("voice_profile_id") or "",
                                settings=(payload.get("settings") if payload.get("settings")
                                          is not None else src.get("settings") or {}),
                                parent_id=project_id)
    if keep_scenes and src.get("scenes"):
        st.db.replace_scenes(proj["id"], [{k: v for k, v in s.items() if k != "id"}
                                          for s in src["scenes"]])
    return {"project": {k: v for k, v in st.db.get_project(proj["id"]).items() if k != "scenes"},
            "note": "copied script + stage settings; outputs (voice/video) start empty"}


@router.get("/projects/{project_id}/export")
async def api_export_project(project_id: str):
    """Portable project memory: script, scenes, prompts, asset index."""
    st = get_state()
    proj = st.db.get_project(project_id)
    if not proj:
        raise HTTPException(404, "project not found")
    return {"project": {k: v for k, v in proj.items() if k != "scenes"},
            "scenes": st.db.list_scenes(project_id),
            "runs": [{k: v for k, v in r.items() if k not in ("stages", "assets")}
                     for r in st.db.list_runs(project_id, limit=50)],
            "prompts": st.db.list_prompts(project_id=project_id, limit=2000),
            "assets": st.db.list_assets(project_id=project_id, limit=2000),
            "style_guideline": style_mod.STYLE_GUIDELINE, "exported_at": now()}


@router.get("/projects/{project_id}/download")
async def api_download_project(project_id: str, kind: str = "all"):
    st = get_state()
    proj = st.db.get_project(project_id)
    if not proj:
        raise HTTPException(404, "project not found")
    d = os.path.join(st.data_root, "projects", project_id)
    if kind == "final":
        final = st.db.latest_asset(project_id, "final")
        if not final:
            raise HTTPException(404, "no final render yet")
        return FileResponse(final["path"], media_type="video/mp4",
                            filename=_safe_download_name(proj.get("title"), ".mp4"))
    if kind == "bundle":
        out = os.path.join(st.data_root, f"export_{project_id}.json")
        payload = await api_export_project(project_id)
        write_json(out, payload.json() if hasattr(payload, "json") else payload)
        return FileResponse(out, media_type="application/json",
                            filename=f"{khmer.normalize(proj.get('title') or 'project')}.json")
    if not os.path.isdir(d):
        raise HTTPException(404, "no files for this project yet")
    zp = os.path.join(st.data_root, f"project_{project_id}.zip")
    await asyncio.to_thread(zip_paths, zp, [d], arc_root=f"project_{project_id}")
    return FileResponse(zp, media_type="application/zip",
                        filename=_safe_download_name(proj.get("title"), ".zip"))


def _safe_download_name(title, ext):
    base = khmer.strip_emoji_and_marks(title or "project")
    base = "".join(c for c in base if c.isalnum() or c in " -_.")
    if khmer.is_khmer(base):
        base = khmer.truncate_clusters(base, 60, suffix="")   # never cut a coeng pair
    base = base[:60].strip() or "project"
    return base + ext


# ================================================================= scene board
@router.get("/projects/{project_id}/scenes")
async def api_scenes(project_id: str):
    st = get_state()
    return {"scenes": st.db.list_scenes(project_id),
            "integrity": _integrity_report(st.db.get_project(project_id) or {},
                                           st.db.list_scenes(project_id))}


@router.post("/projects/{project_id}/scenes")
async def api_save_scenes(project_id: str, payload: dict = Body(...)):
    """The editable storyboard. Text edits are allowed (Director is senior) and
    re-verified against the locked script if the scene list no longer matches."""
    st = get_state()
    proj = st.db.get_project(project_id)
    if not proj:
        raise HTTPException(404, "project not found")
    scenes = payload.get("scenes") or []
    clean = []
    for s in scenes:
        text = khmer.normalize_block(s.get("text") or "")
        if not text:
            continue
        meta = dict(s.get("meta") or {})
        visual_source = str(meta.get("visual_source") or "generated_video")
        if visual_source not in ("generated_video", "illustration", "character_demo"):
            visual_source = "generated_video"
        render_mode = str(meta.get("render_mode") or "broll")
        if render_mode not in ("broll", "talking_head"):
            render_mode = "broll"
        if render_mode == "talking_head" and not (meta.get("character_id")
                                                  or proj.get("character_id")):
            raise HTTPException(400, "render_mode 'talking_head' needs a character on the project "
                                     "or this scene")
        if visual_source == "character_demo" and not (meta.get("character_id")
                                                      or proj.get("character_id")):
            raise HTTPException(400, "visual_source 'character_demo' needs a character on the "
                                     "project or this scene — choose video or illustration")
        if "character_id" in meta and meta.get("character_id"):
            if not st.db.get_character(meta["character_id"]):
                raise HTTPException(400, f"scene {len(clean) + 1}: unknown character_id")
        clean.append({"text": text,
                      "visual_prompt": khmer.clip_clusters((s.get("visual_prompt") or "").strip(), 600),
                      "mood_tag": khmer.clip_clusters((s.get("mood_tag") or "").strip(), 40),
                      "estimated_duration_sec": float(s.get("estimated_duration_sec") or 0),
                      "audio_duration": float(s.get("audio_duration") or 0),
                      "sfx_prompt": khmer.clip_clusters((s.get("sfx_prompt") or "").strip(), 300),
                      "meta": {k: v for k, v in meta.items()
                               if k in ("visual_source", "render_mode", "character_id", "side")}})
    if not clean:
        raise HTTPException(400, "no usable scenes")
    st.db.replace_scenes(project_id, clean)
    write_json(os.path.join(st.data_root, "projects", project_id, "02_scenes.json"),
               {"engine": "director-board", "scenes": clean})
    note = "storyboard saved"
    if (proj.get("mode") or "A").upper() == "A":
        ok = khmer.equal_text(khmer.join_sentences([s["text"] for s in clean]), proj["script"])
        note += " · wording matches the Director's script" if ok else \
                " · ⚠ wording now differs from the original paste"
    return {"scenes": st.db.list_scenes(project_id), "note": note,
            "integrity": _integrity_report(st.db.get_project(project_id),
                                           st.db.list_scenes(project_id))}


# ============================================================== idea (Mode B)
@router.post("/projects/{project_id}/generate-idea")
async def api_generate_idea(project_id: str):
    st = get_state()
    if not st.db.get_project(project_id):
        raise HTTPException(404, "project not found")
    res = await _generate_idea(project_id)
    return res


@router.post("/projects/{project_id}/approve-script")
async def api_approve_script(project_id: str, payload: dict = Body(default={})):
    """Gate for Mode B: approve (optionally edited) script, then start production."""
    st = get_state()
    proj = st.db.get_project(project_id)
    if not proj:
        raise HTTPException(404, "project not found")
    script = proj.get("script") or ""
    if payload.get("script"):
        script = khmer.normalize_block(payload["script"])
    if not script.strip():
        raise HTTPException(400, "there is no script to approve")
    st.db.update_project(project_id, script=script, status="ready", script_locked=False,
                         script_origin=proj.get("script_origin") or "ai:approved")
    write_json(os.path.join(st.data_root, "projects", project_id, "01_script.txt"),
               {"script": script, "approved_at": now(), "approved_by": "director"})
    if payload.get("start", True):
        force = [] if payload.get("from") == "start" else ["breakdown"]
        run = await st.scheduler.start_run(project_id, trigger="new" if not force else "resume",
                                          resume_from=proj.get("last_run_id") or "",
                                          force_stages=force or None)
        return {"ok": True, "project": st.db.get_project(project_id), "run_id": run["run_id"]}
    return {"ok": True, "project": st.db.get_project(project_id)}


@router.post("/projects/{project_id}/regenerate-script")
async def api_regenerate_script(project_id: str, payload: dict = Body(default={})):
    st = get_state()
    proj = st.db.get_project(project_id)
    if not proj:
        raise HTTPException(404, "project not found")
    if (proj.get("mode") or "A").upper() == "A":
        raise HTTPException(403, "Mode A scripts are the Director's — regeneration is disabled")
    note = (payload.get("note") or "").strip()[:400]
    st.db.update_project(project_id, status="draft", regenerate_note=note) if False else None
    st.db.update_project(project_id, status="draft")
    st.db.update_project(project_id, settings={**(proj.get("settings") or {}),
                                               "regenerate_note": note})
    return await _generate_idea(project_id)


# ================================================================== run control
@router.post("/projects/{project_id}/runs")
async def api_start_run(project_id: str, payload: dict = Body(default={})):
    st = get_state()
    proj = st.db.get_project(project_id)
    if not proj:
        raise HTTPException(404, "project not found")
    if not (proj.get("script") or "").strip():
        raise HTTPException(400, "no script yet — paste one (Mode A) or generate one (Mode B)")
    trigger = str(payload.get("trigger") or "new")
    out = await st.scheduler.start_run(
        project_id, trigger=trigger,
        resume_from=payload.get("resume_from") or "",
        force_stages=payload.get("force_stages") or None,
        auto_start=not bool(payload.get("queue_only")))
    st.db.update_project(project_id, status="rendering", last_run_id=out["run_id"])
    return out


@router.get("/runs")
async def api_runs(limit: int = 100):
    st = get_state()
    return {"runs": st.db.list_runs(limit=min(500, max(1, limit)))}


@router.get("/runs/{run_id}")
async def api_run(run_id: str):
    st = get_state()
    s = st.scheduler.status(run_id)
    if not s:
        raise HTTPException(404, "run not found")
    return s


@router.get("/runs/{run_id}/status")
async def api_run_status(run_id: str, since: int = 0):
    """Polling snapshot (the UI's fallback when WebSocket/SSE are unavailable)."""
    st = get_state()
    s = st.scheduler.status(run_id)
    if not s:
        raise HTTPException(404, "run not found")
    events = st.db.list_events(run_id, limit=200, after_id=since)
    return {**s, "events": events, "last_event_id": max([e["id"] for e in events] or [since])}


@router.post("/runs/{run_id}/cancel")
async def api_run_cancel(run_id: str):
    return await get_state().scheduler.cancel_run(run_id)


@router.post("/runs/{run_id}/pause")
async def api_run_pause(run_id: str):
    return get_state().scheduler.pause_run(run_id)


@router.post("/runs/{run_id}/resume")
async def api_run_resume(run_id: str):
    st = get_state()
    out = st.scheduler.resume_paused(run_id)
    if out.get("ok"):
        return out
    run = st.db.get_run(run_id)
    if not run:
        raise HTTPException(404, "run not found")
    from .pipeline.scheduler import resume_run

    res = await resume_run(st.scheduler, run_id)
    st.db.update_run(run_id, status="running")
    return {"ok": True, "continued": True, **res}


@router.post("/runs/{run_id}/continue")
async def api_run_continue(run_id: str):
    """Resume from the last successful stage (fresh run, done stages inherited)."""
    st = get_state()
    run = st.db.get_run(run_id)
    if not run:
        raise HTTPException(404, "run not found")
    return await st.scheduler.start_run(run["project_id"], trigger="resume", resume_from=run_id)


@router.post("/runs/{run_id}/stages/{stage}/regenerate")
async def api_stage_regenerate(run_id: str, stage: str, payload: dict = Body(default={})):
    """Re-run ONE stage (and everything downstream) without redoing the rest.

    Optional `overrides` (visual_prompt / sfx_prompt / mood_tag / seed / prompt
    text edits) are applied to the scene row first, so the regeneration uses the
    Director's new instruction rather than the old one.
    """
    st = get_state()
    run = st.db.get_run(run_id)
    if not run:
        raise HTTPException(404, "run not found")
    if stage not in stagespec.STAGE_BY_KEY:
        raise HTTPException(400, f"unknown stage '{stage}' — pick one of "
                                 + ", ".join(stagespec.ORDER))
    pid = run["project_id"]
    scene_idx = payload.get("scene_idx")
    overrides = payload.get("overrides") or {}
    if overrides and scene_idx is not None and int(scene_idx) >= 0:
        allowed = {"text", "visual_prompt", "mood_tag", "sfx_prompt", "est_duration",
                   "estimated_duration_sec"}
        upd = {k: v for k, v in overrides.items() if k in allowed}
        if upd:
            st.db.update_scene(pid, int(scene_idx), **upd)
    elif overrides:
        raise HTTPException(400, "overrides need a scene_idx")
    if payload.get("skip_qa_gate"):
        proj = st.db.get_project(pid) or {}
        settings = dict(proj.get("settings") or {})
        settings.setdefault("pipeline", {})["require_qa_pass"] = False
        st.db.update_project(pid, settings=settings)
    out = await st.scheduler.rerun_stage(run_id, stage, project_id=pid)
    st.bus.publish("stage_regenerate", {"stage": stage, "scene_idx": scene_idx,
                                       "new_run": out["run_id"], "overrides": overrides},
                  run_id=out["run_id"], project_id=pid, stage=stage,
                  scene_idx=int(scene_idx) if scene_idx is not None else -1)
    return out


@router.post("/projects/{project_id}/catchup")
async def api_gpu_catchup(project_id: str):
    """Run the stages deferred on Machine B (video + MMAudio) on a GPU machine."""
    st = get_state()
    proj = st.db.get_project(project_id)
    if not proj:
        raise HTTPException(404, "project not found")
    last = proj.get("last_run_id") or ""
    deferred = []
    if last:
        deferred = sorted({r["stage"] for r in st.db.list_stages(last) if r["status"] == "deferred"})
    if not deferred:
        deferred = ["video", "video_fit", "sfx"]
    for s in ("qa", "assemble"):
        if s not in deferred:
            deferred.append(s)
    out = await st.scheduler.start_run(project_id, trigger="gpu-catchup", resume_from=last,
                                       force_stages=deferred)
    return {**out, "deferred_stages": deferred,
            "note": "re-rendering the GPU stages, then QA + assembly; voice is reused as-is"}


@router.get("/runs/{run_id}/scenes/{idx}/bundle")
async def api_scene_bundle(run_id: str, idx: int):
    """Everything one scene produced — what the inspector panel shows."""
    st = get_state()
    run = st.db.get_run(run_id)
    if not run:
        raise HTTPException(404, "run not found")
    pid = run["project_id"]
    scene = st.db.get_scene(pid, idx)
    if not scene:
        raise HTTPException(404, "scene not found")
    by_kind = {}
    for kind in ("voice", "voice_final", "video", "video_fit", "ambient", "qa"):
        a = st.db.latest_asset(pid, kind, scene_idx=idx)
        if a:
            by_kind[kind] = {**{k: v for k, v in a.items() if k != "meta"}, "meta": a["meta"],
                             "url": f"/api/assets/{a['id']}/stream",
                             "download": f"/api/assets/{a['id']}/download",
                             "exists": os.path.exists(a["path"]),
                             "size_human": human_size(a.get("size_bytes") or 0)}
    stages = [r for r in st.db.list_stages(run_id) if r["scene_idx"] == idx]
    prompts = [p for p in st.db.list_prompts(project_id=pid, limit=400) if p["scene_idx"] == idx]
    peaks = {}
    for kind in ("voice", "voice_final", "ambient"):
        p = by_kind.get(kind, {}).get("path")
        if p and kind.startswith("voice") or (p and p.endswith(".wav")):
            try:
                peaks[kind] = await asyncio.to_thread(wav_peaks, p, 220)
            except Exception:
                peaks[kind] = []
    return {"scene": scene, "assets": by_kind, "stages": stages, "prompts": prompts[:24],
            "peaks": peaks,
            "qa": json.loads(json.dumps((by_kind.get("qa", {}) or {}).get("meta") or {}))}


# ==================================================================== assets
@router.get("/assets")
async def api_assets(project_id: str = "", kind: str = "", limit: int = 400):
    st = get_state()
    rows = st.db.list_assets(project_id=project_id or None, kind=kind or None,
                            limit=min(2000, max(1, limit)))
    for r in rows:
        r["url"] = f"/api/assets/{r['id']}/stream"
        r["download"] = f"/api/assets/{r['id']}/download"
        r["size_human"] = human_size(r.get("size_bytes") or 0)
    return {"assets": rows}


def _asset_or_404(asset_id):
    st = get_state()
    row = st.db.one("SELECT * FROM assets WHERE id=?", (asset_id,))
    if not row:
        raise HTTPException(404, "asset not found")
    if not row.get("path") or not os.path.exists(row["path"]):
        raise HTTPException(404, "asset file is missing on disk (was the folder cleaned up?)")
    return row


@router.get("/assets/{asset_id}/stream")
async def api_asset_stream(asset_id: str, request: Request = None):
    row = _asset_or_404(asset_id)
    return _stream_with_range(row["path"], row.get("mime") or "application/octet-stream")


@router.get("/assets/{asset_id}/download")
async def api_asset_download(asset_id: str):
    row = _asset_or_404(asset_id)
    return FileResponse(row["path"], media_type=row.get("mime") or "application/octet-stream",
                        filename=os.path.basename(row["path"]))


def _stream_with_range(path, mime):
    """HTTP Range support so <video>/<audio> seeking works in every browser."""
    size = os.path.getsize(path)
    headers = {"Accept-Ranges": "bytes", "Content-Length": str(size),
               "Cache-Control": "no-cache"}
    return FileResponse(path, media_type=mime, headers=headers)


@router.get("/assets/{asset_id}/waveform")
async def api_asset_waveform(asset_id: str, bins: int = 200):
    row = _asset_or_404(asset_id)
    try:
        peaks = await asyncio.to_thread(wav_peaks, row["path"], min(600, max(40, bins)))
    except Exception as e:
        raise HTTPException(400, f"cannot read audio: {e}")
    return {"peaks": peaks, "duration": media_duration(row["path"], 0)}


@router.get("/projects/{project_id}/scene/{idx}/download")
async def api_scene_zip(project_id: str, idx: int):
    st = get_state()
    d = os.path.join(st.data_root, "projects", project_id, "scenes", f"{idx:02d}")
    if not os.path.isdir(d):
        raise HTTPException(404, "no files for that scene")
    zp = os.path.join(st.data_root, f"scene_{project_id}_{idx}.zip")
    await asyncio.to_thread(zip_paths, zp, [d], arc_root=f"scene_{idx:02d}")
    return FileResponse(zp, media_type="application/zip",
                        filename=f"{project_id}_scene{idx:02d}.zip")


# ================================================================== prompts log
@router.get("/prompts")
async def api_prompts(project_id: str = "", run_id: str = "", stage: str = "", limit: int = 200):
    st = get_state()
    return {"prompts": st.db.list_prompts(project_id=project_id or None, run_id=run_id or None,
                                         stage=stage or None, limit=min(1000, max(1, limit)))}


# ==================================================================== settings
@router.get("/settings")
async def api_settings():
    st = get_state()
    cfg = st.config()
    plan = st.plan(refresh=True)
    return {"settings": cfg, "plan": plan, "roles": stagespec.stage_labels(),
            "llm_roles": {"keys": cfg_mod.LLM_ROLES, "labels": cfg_mod.LLM_ROLE_LABELS},
            "machine_profiles": cfg_mod.MACHINE_PROFILES,
            "defaults": cfg_mod.DEFAULTS, "style_guideline": style_mod.STYLE_GUIDELINE,
            "placeholders": __import__("ai_studio.workflows", fromlist=["KNOWN_PLACEHOLDERS"])
            .KNOWN_PLACEHOLDERS,
            "pace_presets": cfg_mod.PACE_PRESETS,
            "subtitle_styles": media.SUBTITLE_STYLES, "title_styles": media.TITLE_STYLES,
            "content_types": content_mod.content_type_payload(),
            "vram": {"limit_mb": cfg["vram"]["limit_mb"], "detected": plan.get("hardware")}}


@router.post("/settings")
async def api_settings_save(payload: dict = Body(...)):
    st = get_state()
    cur = st.config()
    merged = cfg_mod.normalize_config(_merge_settings(cur, payload))
    path = cfg_mod.save(merged, st.settings_path)
    cfg_mod.sync_to_ai_creator_team(merged)         # keep the legacy studio consistent
    st.invalidate()
    cfg2, plan = st.config(), st.plan(refresh=True)
    return {"ok": True, "path": path, "settings": cfg2, "plan": plan,
            "note": "saved. Engine choices take effect on the next run."}


def _merge_settings(cur, patch):
    out = json.loads(json.dumps(cur))
    for k, v in (patch or {}).items():
        if k.startswith("_"):
            continue
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge_settings(out[k], v)
        else:
            out[k] = v
    return out


@router.post("/settings/probe")
async def api_settings_probe():
    """Re-detect hardware + services (after the user starts Ollama/ComfyUI)."""
    st = get_state()
    cfg = st.config()
    caps = await asyncio.to_thread(cfg_mod.capabilities, cfg)
    await asyncio.to_thread(st.invalidate)
    plan = st.plan(refresh=True)
    fixes = _check_hints(cfg, caps)
    return {"capabilities": caps, "plan": plan,
            "tts": {**_tts_probe(cfg), "fix": fixes["tts"], "check_ok": bool(caps.get("sherpa_tts"))},
            "rvc": {**_rvc_probe(cfg), "fix": fixes["rvc"],
                    "check_ok": bool(caps.get("rvc_http") or caps.get("rvc_cli"))},
            "video": {**_video_probe(cfg), "fix": fixes["video"], "check_ok": bool(caps.get("comfyui"))},
            "sfx": {**_sfx_probe(cfg), "fix": fixes["sfx"], "check_ok": bool(caps.get("comfyui"))},
            "talking_head": {**_talking_head_probe(cfg), "fix": fixes["talking_head"],
                             "check_ok": bool(caps.get("sadtalker"))},
            "illustration": {**_illustration_probe(cfg), "fix": fixes["illustration"],
                             "check_ok": bool(caps.get("comfyui"))},
            "vram": await asyncio.to_thread(vram_mod.status, cfg, plan)}


def _tts_probe(cfg):
    from .engines import tts

    return tts.probe(cfg)


def _rvc_probe(cfg):
    from .engines import rvc

    return rvc.probe(cfg)


def _video_probe(cfg):
    from .engines import video

    return video.probe(cfg)


def _sfx_probe(cfg):
    from .engines import sfx

    return sfx.probe(cfg)


def _talking_head_probe(cfg):
    from .engines import talking_head

    return talking_head.probe(cfg)


def _illustration_probe(cfg):
    from .engines import illustration

    return illustration.probe(cfg)


def _check_hints(cfg, caps):
    """The --check fix commands, verbatim, for the Services panel."""
    return {
        "tts": ([] if caps.get("sherpa_tts") else
                ["./scripts/setup_khmer_tts.sh   (one-time MMS conversion)",
                 "pip install sherpa-onnx"]),
        "rvc": ([] if (caps.get("rvc_http") or caps.get("rvc_cli")) else
                ["start RVC-WebUI's inference API (default http://127.0.0.1:9513)",
                 "or set rvc.webui_dir to your RVC-WebUI folder"]),
        "video": ([] if caps.get("comfyui") else
                  ["python main.py --listen 127.0.0.1 --port 8188",
                   "then load models/wan2.1_t2v_1.3b (see README-STUDIO.md)"]),
        "sfx": ([] if caps.get("comfyui") else
                ["python main.py --listen 127.0.0.1 --port 8188",
                 "then load models/mmaudio_small (see README-STUDIO.md)"]),
        "talking_head": ([] if caps.get("sadtalker") else
                         ["git clone SadTalker (open-source); set talking_head.sadtalker_dir "
                          "to its folder containing infer.py"]),
        "illustration": ([] if caps.get("comfyui") else
                         ["python main.py --listen 127.0.0.1 --port 8188",
                          "then put flux2-klein-4b-fp8.safetensors + t5xxl + ae in "
                          "ComfyUI/models (workflow shipped, editable)"]),
    }


@router.get("/workflows")
async def api_workflows():
    from . import workflows as wf

    st = get_state()
    out = []
    for d in wf.search_dirs(st.config()):
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".json"):
                continue
            path = os.path.join(d, fn)
            data = read_json(path, None) or {}
            out.append({"name": fn, "dir": d, "path": path,
                        "placeholders": wf.template_placeholders(data),
                        "size": os.path.getsize(path)})
    return {"workflows": out, "dirs": [d for d in wf.search_dirs(st.config()) if os.path.isdir(d)],
            "known": wf.KNOWN_PLACEHOLDERS}


# =============================================================== voice profiles
@router.get("/voices")
async def api_voices():
    st = get_state()
    rows = st.db.list_voice_profiles()
    for r in rows:
        r["pth_exists"] = bool(r.get("pth_path")) and os.path.exists(r["pth_path"])
        r["index_exists"] = bool(r.get("index_path")) and os.path.exists(r["index_path"])
        r["sample_url"] = f"/api/voices/{r['id']}/sample" if r.get("sample_path") else ""
    cfg = st.config()
    discovered = await asyncio.to_thread(_discover_rvc, cfg)
    return {"voices": rows, "discovered": discovered,
            "rvc": {"webui_dir": cfg["rvc"].get("webui_dir"), "api_base": cfg["rvc"].get("api_base")}}


def _discover_rvc(cfg):
    from .engines.rvc import discover_profiles

    return discover_profiles(cfg)


@router.post("/voices/import-discovered")
async def api_voices_import():
    st = get_state()
    found = await asyncio.to_thread(_discover_rvc, st.config())
    added = []
    for f in found:
        existing = next((v for v in st.db.list_voice_profiles() if v.get("pth_path") == f["pth_path"]),
                        None)
        if existing:
            continue
        added.append(st.db.create_voice_profile(name=f["name"], pth_path=f["pth_path"],
                                               index_path=f.get("index_path") or "",
                                               notes="discovered in models/rvc"))
    return {"added": added, "found": len(found)}


@router.post("/voices")
async def api_voice_create(name: str = Form(...), pth: UploadFile = File(None),
                          index: UploadFile = File(None), sample: UploadFile = File(None),
                          notes: str = Form(""), pitch: int = Form(0)):
    """Register a voice profile: upload the RVC .pth (+ .index) and the 10-15 min
    training sample. The sample is normalised for training and kept for preview."""
    st = get_state()
    vid = f"v{new_id(6)}"
    root = ensure_dir(os.path.join(st.data_root, "voices", vid))
    pth_path = index_path = sample_path = ""
    if pth is not None and pth.filename:
        pth_path = os.path.join(root, _safefilename(pth.filename, ".pth"))
        await _save_upload(pth, pth_path)
    if index is not None and index.filename:
        index_path = os.path.join(root, _safefilename(index.filename, ".index"))
        await _save_upload(index, index_path)
    seconds = 0.0
    warnings = []
    if sample is not None and sample.filename:
        raw = os.path.join(root, _safefilename(sample.filename, ".wav"))
        await _save_upload(sample, raw)
        try:
            from .engines.rvc import prepare_sample

            info = await asyncio.to_thread(prepare_sample, raw, root)
            sample_path, seconds, warnings = info["path"], info["seconds"], info["warnings"]
            if os.path.exists(raw) and raw != sample_path:
                os.remove(raw)
        except Exception as e:
            warnings.append(f"sample could not be normalised: {str(e)[:120]}")
            sample_path = raw
    if not pth_path and not sample_path:
        shutil.rmtree(root, ignore_errors=True)
        raise HTTPException(400, "a voice profile needs either already-trained RVC .pth weights "
                                 "or a training sample to train one from")
    prof = st.db.create_voice_profile(name=name.strip() or "My Voice", pth_path=pth_path,
                                      index_path=index_path, sample_path=sample_path,
                                      sample_seconds=seconds, notes=notes, pitch=pitch)
    prof["warnings"] = warnings
    prof["sample_url"] = f"/api/voices/{vid}/sample" if sample_path else ""
    return {"voice": prof, "training_command": _training_command(prof, sample_path, st.config())}


def _training_command(prof, sample_path, cfg):
    from .engines.rvc import training_command

    if not sample_path:
        return ""
    try:
        return training_command(prof, sample_path, cfg)
    except Exception:
        return ""


def _safefilename(fn, default_ext):
    fn = os.path.basename(fn or "")
    fn = "".join(c for c in fn if c.isalnum() or c in "._-") or ("voice" + default_ext)
    if not os.path.splitext(fn)[1]:
        fn += default_ext
    return fn


async def _save_upload(up: UploadFile, dst: str):
    ensure_dir(os.path.dirname(dst))
    with open(dst, "wb") as f:
        shutil.copyfileobj(up.file, f)
    return dst


@router.delete("/voices/{voice_id}")
async def api_voice_delete(voice_id: str, purge_files: bool = Query(False)):
    """Forget a voice profile. Uploads stay on disk unless purge_files=true, so a
    mis-typed profile can be removed without destroying a trained model."""
    st = get_state()
    prof = st.db.get_voice_profile(voice_id)
    if not prof:
        raise HTTPException(404, "voice profile not found")
    removed = 0
    if purge_files:
        root = os.path.join(st.data_root, "voices", voice_id)
        if os.path.isdir(root):
            shutil.rmtree(root, ignore_errors=True)
            removed = 1
    st.db.delete_voice_profile(voice_id)
    cfg = st.config()
    if cfg["rvc"].get("profile_id") == voice_id:
        cfg["rvc"]["profile_id"] = ""
        cfg_mod.save(cfg, st.settings_path)
        st.invalidate()
    return {"ok": True, "files_removed": removed}


@router.get("/voices/{voice_id}/sample")
async def api_voice_sample(voice_id: str):
    st = get_state()
    prof = st.db.get_voice_profile(voice_id)
    if not prof or not prof.get("sample_path") or not os.path.exists(prof["sample_path"]):
        raise HTTPException(404, "no sample recorded for that voice")
    return _stream_with_range(prof["sample_path"], "audio/wav")


@router.post("/voices/{voice_id}/select")
async def api_voice_select(voice_id: str, project_id: str = ""):
    st = get_state()
    if not st.db.get_voice_profile(voice_id):
        raise HTTPException(404, "voice profile not found")
    cfg = st.config()
    if not project_id:
        cfg["rvc"]["profile_id"] = voice_id
        cfg_mod.save(cfg, st.settings_path)
        st.invalidate()
        return {"ok": True, "scope": "default"}
    proj = st.db.get_project(project_id)
    if not proj:
        raise HTTPException(404, "project not found")
    st.db.update_project(project_id, voice_profile_id=voice_id)
    return {"ok": True, "scope": "project"}


@router.post("/voices/{voice_id}/preview")
async def api_voice_preview(voice_id: str, payload: dict = Body(default={})):
    """Hear stage 3a + 3b on one line before committing to a full render."""
    st = get_state()
    prof = st.db.get_voice_profile(voice_id)
    if not prof:
        raise HTTPException(404, "voice profile not found")
    cfg, plan = st.resolved_cfg()
    text = (payload.get("text") or "សួស្ដី។ ខ្ញុំកំពុងនិយាយដោយស្ងប់ស្ងាត់ និងមិនបោះបង់ទេ។")[:400]
    tmp = ensure_dir(os.path.join(st.data_root, "tmp"))
    base = os.path.join(tmp, f"prev_{voice_id}_base.wav")
    conv = os.path.join(tmp, f"prev_{voice_id}_final.wav")
    from .engines import rvc as rvc_engine, tts as tts_engine

    res = await asyncio.to_thread(tts_engine.synthesize, text, base, cfg, plan["tts"]["engine"])
    if not res.get("ok"):
        raise HTTPException(400, f"voice engine failed: {res.get('reason')}")
    conv_res = await asyncio.to_thread(rvc_engine.convert, base, conv, cfg, prof)
    return {"base_url": "/api/tmpfile?name=" + os.path.basename(base),
            "final_url": ("/api/tmpfile?name=" + os.path.basename(conv)) if conv_res.get("ok") else "",
            "tts": {k: v for k, v in res.items() if k != "ok"},
            "rvc": {k: v for k, v in conv_res.items() if k != "ok"},
            "converted": bool(conv_res.get("converted")),
            "duration": media_duration(base, 0)}


@router.get("/tmpfile")
async def api_tmpfile(name: str):
    if "/" in name or "\\" in name or name.startswith("."):
        raise HTTPException(400, "bad name")
    p = os.path.join(get_state().data_root, "tmp", name)
    if not os.path.exists(p):
        raise HTTPException(404, "gone")
    ext = os.path.splitext(name)[1].lower()
    mime = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".mp4": "video/mp4",
            ".png": "image/png", ".jpg": "image/jpeg", ".json": "application/json"}.get(ext,
                                                                                          "application/octet-stream")
    return _stream_with_range(p, mime)


@router.post("/voices/{voice_id}/train")
async def api_voice_train(voice_id: str):
    """Opt-in: launch the user-configured RVC training command, log streamed."""
    st = get_state()
    prof = st.db.get_voice_profile(voice_id)
    if not prof:
        raise HTTPException(404, "voice profile not found")
    if not prof.get("sample_path"):
        raise HTTPException(400, "upload a 10-15 minute sample first")
    from .engines.rvc import run_training

    job_id = "train_" + new_id(6)
    st.training[job_id] = {"status": "running", "lines": [], "voice_id": voice_id}

    def log_line(line):
        st.training[job_id]["lines"].append(line)
        st.bus.publish("training_log", {"job_id": job_id, "line": line}, project_id=voice_id)

    def _run():
        return run_training(prof, prof["sample_path"], st.config(), log_line=log_line)

    async def _bg():
        res = await asyncio.to_thread(_run)
        st.training[job_id].update(res)
        st.training[job_id]["status"] = "done" if res.get("ok") else "error"
        st.db.update_voice_profile(voice_id, training_status=st.training[job_id]["status"])
        st.bus.publish("training_done", {"job_id": job_id, **res}, project_id=voice_id)

    asyncio.create_task(_bg())
    return {"job_id": job_id, "note": "training is RVC WebUI's job; the studio only runs the "
                                     "command you configured and streams its log"}


@router.get("/training/{job_id}")
async def api_training_status(job_id: str):
    st = get_state()
    return st.training.get(job_id) or {"status": "unknown"}


# ============================================================ diagnostics / misc
@router.get("/jobs")
async def api_jobs():
    st = get_state()
    return {"runs": st.db.list_runs(limit=60), "active": list(st.scheduler.runs.keys())}


# ================================================================== characters
# A Character is the studio's persistent NPC: a name + a set of expression
# images (neutral / calm / happy / sad / ...). A project can set `character_id`;
# scenes can override per-shot (`meta.character_id`) for two-character scripts.
EXPRESSION_LABELS = ("neutral", "calm", "happy", "sad", "surprised", "thinking",
                     "curious", "excited", "worried", "confident")


@router.get("/characters")
async def api_characters():
    st = get_state()
    rows = st.db.list_characters()
    for r in rows:
        for img in r.get("images") or []:
            img["url"] = f"/api/characters/{r['id']}/images/{img['id']}/file"
    return {"characters": rows, "expression_labels": EXPRESSION_LABELS,
            "mood_to_expression": content_mod.MOOD_TO_EXPRESSION}


@router.post("/characters")
async def api_character_create(payload: dict = Body(...)):
    st = get_state()
    name = str(payload.get("name") or "").strip()[:80]
    if not name:
        raise HTTPException(400, "a character needs a name")
    return {"character": st.db.create_character(name=name,
                                                notes=(payload.get("notes") or "")[:1000])}


@router.patch("/characters/{character_id}")
async def api_character_update(character_id: str, payload: dict = Body(...)):
    st = get_state()
    if not st.db.get_character(character_id):
        raise HTTPException(404, "character not found")
    kw = {}
    if "name" in payload:
        kw["name"] = str(payload.get("name") or "").strip()[:80]
    if "notes" in payload:
        kw["notes"] = str(payload.get("notes") or "")[:1000]
    return {"character": st.db.update_character(character_id, **kw)}


@router.delete("/characters/{character_id}")
async def api_character_delete(character_id: str):
    st = get_state()
    if not st.db.get_character(character_id):
        raise HTTPException(404, "character not found")
    st.db.delete_character(character_id)
    return {"ok": True, "message": "character deleted; projects using it now have no character"}


@router.post("/characters/{character_id}/images")
async def api_character_image_upload(character_id: str,
                                     expression_label: str = Form("neutral"),
                                     image: UploadFile = File(...)):
    """Upload one expression photo for a character (png/jpg/webp)."""
    st = get_state()
    char = st.db.get_character(character_id)
    if not char:
        raise HTTPException(404, "character not found")
    label = str(expression_label or "neutral").strip()[:40] or "neutral"
    root = ensure_dir(os.path.join(st.data_root, "characters", character_id))
    safe = _safefilename(image.filename or "image.png", ".png")
    raw = os.path.join(root, "raw_" + safe)
    await _save_upload(image, raw)
    dst = os.path.join(root, _safefilename(f"{label} {new_id(4)}.png", ".png"))
    try:
        await asyncio.to_thread(_normalize_image, raw, dst)
        if os.path.exists(raw) and raw != dst:
            os.remove(raw)
    except Exception as e:
        # keep raw_* on disk for inspection; normalized dst is only written on success
        raise HTTPException(400, f"image could not be processed: {str(e)[:160]}")
    row = st.db.add_character_image(character_id, label, dst)
    return {"ok": True,
            "image": {**row, "url": f"/api/characters/{character_id}/images/{row['id']}/file"},
            "note": f"expression '{label}' uploaded for {char.get('name')}"}


@router.get("/characters/{character_id}/images/{image_id}/file")
async def api_character_image_file(character_id: str, image_id: str):
    st = get_state()
    row = st.db.get_character_image(image_id)
    if not row or row["character_id"] != character_id or not row.get("image_path"):
        raise HTTPException(404, "image not found")
    if not os.path.exists(row["image_path"]):
        raise HTTPException(404, "image file missing on disk")
    return FileResponse(row["image_path"], media_type="image/png",
                        filename=os.path.basename(row["image_path"]))


@router.delete("/characters/{character_id}/images/{image_id}")
async def api_character_image_delete(character_id: str, image_id: str):
    st = get_state()
    row = st.db.get_character_image(image_id)
    if not row or row["character_id"] != character_id:
        raise HTTPException(404, "image not found")
    st.db.delete_character_image(image_id)
    try:
        if row.get("image_path") and os.path.exists(row["image_path"]):
            os.remove(row["image_path"])
    except Exception:
        pass
    return {"ok": True}


def _normalize_image(src, dst):
    """Re-encode to the studio's 512x768 portrait PNG so ffmpeg/PIL/I2V always
    see one format. Pillow first, ffmpeg scale/pad as the fallback."""
    try:
        from PIL import Image
        im = Image.open(src).convert("RGB")
        w, h = im.size
        tw, th = 512, 768
        scale = max(tw / w, th / h)
        im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
        left = (im.width - tw) // 2
        top = (im.height - th) // 2
        im = im.crop((left, top, left + tw, top + th))
        im.save(dst, "PNG")
        return dst
    except Exception:
        from .util import run_ffmpeg
        run_ffmpeg(["-i", src, "-vf",
                    "scale=512:768:force_original_aspect_ratio=increase,crop=512:768,"
                    "format=rgb24", "-frames:v", "1", dst], timeout=300)
        return dst


# ============================================================ scene still images
@router.post("/projects/{project_id}/scenes/{idx}/image")
async def api_scene_image_upload(project_id: str, idx: int, image: UploadFile = File(...)):
    """Director's custom picture for one scene — wins over all generation.

    Stored as ``00_custom.png`` next to the scene; the video stage detects it and
    makes a Ken Burns clip instead of calling a model.
    """
    st = get_state()
    prj = st.db.get_project(project_id)
    if not prj:
        raise HTTPException(404, "project not found")
    scene = st.db.get_scene(project_id, idx)
    if not scene:
        raise HTTPException(404, f"scene {idx} not found")
    scene_dir = ensure_dir(os.path.join(st.data_root, "projects", project_id, "scenes",
                                        f"{int(idx):02d}"))
    safe = _safefilename(image.filename or "image.png", ".png")
    raw = os.path.join(scene_dir, "00_custom_raw" + os.path.splitext(safe)[1])
    await _save_upload(image, raw)
    dst = os.path.join(scene_dir, "00_custom.png")
    try:
        await asyncio.to_thread(_normalize_image, raw, dst)
        if os.path.exists(raw):
            os.remove(raw)
    except Exception as e:
        raise HTTPException(400, f"image could not be processed: {str(e)[:160]}")
    meta = dict(scene.get("meta") or {})
    meta["visual_source"] = "illustration"
    _set_scene_meta(st.db, project_id, idx, meta)
    return {"ok": True, "scene": st.db.get_scene(project_id, idx),
            "url": f"/api/files?path={os.path.abspath(dst)}",
            "note": "custom image saved for this scene (illustration source, generation skipped)"}


@router.delete("/projects/{project_id}/scenes/{idx}/image")
async def api_scene_image_delete(project_id: str, idx: int):
    st = get_state()
    scene_dir = os.path.join(st.data_root, "projects", project_id, "scenes", f"{int(idx):02d}")
    removed = False
    for cand in ("00_custom.png", "00_custom.jpg", "00_custom.jpeg", "00_custom.webp"):
        p = os.path.join(scene_dir, cand)
        if os.path.exists(p):
            try:
                os.remove(p)
                removed = True
            except Exception:
                pass
    scene = st.db.get_scene(project_id, idx)
    if scene:
        meta = dict(scene.get("meta") or {})
        meta.pop("visual_source", None)
        meta["visual_source"] = "generated_video"
        _set_scene_meta(st.db, project_id, idx, meta)
    return {"ok": True, "removed": removed}


def _set_scene_meta(db, project_id, idx, meta):
    """Scene meta lives in meta_json; db.update_scene only touches scalar cols,
    so the whole board is rewritten (cheap, and single-writer by design)."""
    all_scenes = db.list_scenes(project_id)
    for s in all_scenes:
        if s["idx"] == idx:
            s["meta"] = dict(s.get("meta") or {})
            s["meta"].update(meta or {})
    db.replace_scenes(project_id, all_scenes)
    return db.get_scene(project_id, idx)


# ============================================================ ollama models
@router.get("/ollama/models")
async def api_ollama_models():
    """The AI Team panel's model picker: models actually installed in Ollama."""
    st = get_state()
    try:
        from .llm import LLM
        llm = LLM(st.config())
        names = await asyncio.to_thread(llm.list_models)
        return {"models": names, "host": llm.host, "online": bool(names)}
    except Exception as e:
        return {"models": [], "host": (st.config().get("ollama") or {}).get("host", ""),
                "online": False, "error": str(e)[:160]}


# ============================================================ content types
@router.get("/content-types")
async def api_content_types():
    return {"types": content_mod.content_type_payload()}


# ============================================================ style previews
@router.get("/style-previews")
async def api_style_previews(refresh: bool = Query(False)):
    """Pre-rendered ~3s samples for every subtitle *and* title style (cached in
    <data>/style-previews, regenerated on ?refresh=true)."""
    st = get_state()
    return await asyncio.to_thread(_style_previews, st, refresh)


def _style_previews(st, refresh=False):
    from . import previz as previz_mod

    root = ensure_dir(os.path.join(st.data_root, "style-previews"))
    # one shared source clip (2.8s calm scene) for all subtitle styles
    base_mp4 = os.path.join(root, "base.mp4")
    if refresh or not os.path.exists(base_mp4):
        previz_mod.render_clip(base_mp4, duration=2.8, width=480, height=854, fps=20,
                               mood_tag="calm-warm",
                               visual_prompt="soft golden light over a quiet rice field, gentle mist",
                               seed=7, motion=0.5)
    total_dur = media_duration(base_mp4, 0.0) or 2.8
    srt_path = os.path.join(root, "sample.srt")
    write_text_file(srt_path, _sample_srt(total_dur, 2.6))
    sub_styles, title_styles = [], []
    for key in media.SUBTITLE_STYLE_KEYS:
        out = os.path.join(root, f"subtitles_{key}.mp4")
        err = ""
        if refresh or not os.path.exists(out):
            try:
                media.burn_subtitles(base_mp4, srt_path, out, style=key)
            except Exception as e:
                err = str(e)[:180]
                # a failed sample must still be LISTED (honest badge), never a gap
                try:
                    media.burn_subtitles(base_mp4, srt_path, out, style="clean")
                except Exception as e2:
                    err = f"{err}; even clean burn failed: {str(e2)[:120]}"
        sub_styles.append({"key": key, "label": media.SUBTITLE_STYLES[key]["label"],
                           "url": f"/api/files?path={os.path.abspath(out)}" if os.path.exists(out) else "",
                           "desc": media.SUBTITLE_STYLES[key].get(
                               "desc") or media.SUBTITLE_STYLES[key].get("description", ""),
                           "font_size": media.SUBTITLE_STYLES[key].get("font_size"),
                           "error": err})
    for key in media.TITLE_STYLE_KEYS:
        out = os.path.join(root, f"title_{key}.mp4")
        err = ""
        if refresh or not os.path.exists(out):
            try:
                media.render_title_card(out, "មួយជំហាន ឆ្ពោះទៅមុខ", key, 480, 854, 20, 2.6)
            except Exception as e:
                err = str(e)[:180]
        if os.path.exists(out):
            title_styles.append({"key": key, "label": media.TITLE_STYLES[key]["label"],
                                 "url": f"/api/files?path={os.path.abspath(out)}",
                                 "desc": media.TITLE_STYLES[key].get("desc")
                                 or media.TITLE_STYLES[key].get("description", ""),
                                 "error": err})
    return {"subtitle_styles": sub_styles, "title_styles": title_styles,
            "source": {"duration": round(total_dur, 2), "width": 480, "height": 854},
            "note": "samples are cached; add ?refresh=true to rebuild"}


def _sample_srt(total_dur, end_at=None):
    end = float(end_at or min(2.6, max(1.0, total_dur - 0.2)))
    return "\n".join([
        "1\n00:00:00,200 --> " + _srt_ts(end - 0.35) + "\nសួស្តី ថ្ងៃនេះយើងមកស្វែងយល់អំពីរឿងមួយ\n",
        "2\n" + _srt_ts(end + 0.4) + " --> " + _srt_ts(min(total_dur, end + 1.3)) +
        "\nhello world this is a subtitle preview\n",
    ])


def _srt_ts(sec):
    ms = int(round(sec * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def write_text_file(path, text):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


@router.get("/files")
async def api_files(path: str = Query(...)):
    """Serve a file only when its absolute path is inside the studio data root."""
    st = get_state()
    import mimetypes
    root = os.path.abspath(st.data_root)
    p = os.path.abspath(path)
    if not p.startswith(root + os.sep):
        raise HTTPException(403, "path is outside the studio data directory")
    if not os.path.exists(p):
        raise HTTPException(404, "file not found")
    return FileResponse(p, media_type=mimetypes.guess_type(p)[0] or "application/octet-stream",
                        filename=os.path.basename(p))


@router.get("/memory/search")
async def api_memory(q: str = "", limit: int = 60):
    """Search across every prompt, script and scene ever stored (the memory view)."""
    st = get_state()
    q = (q or "").strip()
    out = {"projects": [], "prompts": [], "scenes": []}
    if q:
        like = f"%{q}%"
        out["projects"] = st.db.query(
            "SELECT id,title,mode,status,substr(script,1,200) AS excerpt,updated_at FROM projects "
            "WHERE title LIKE ? OR script LIKE ? OR topic_hint LIKE ? ORDER BY updated_at DESC LIMIT ?",
            (like, like, like, int(limit)))
        out["prompts"] = st.db.query(
            "SELECT id,project_id,stage,role,model,substr(user,1,300) AS user_excerpt,"
            "substr(response,1,300) AS response_excerpt,created_at FROM prompts "
            "WHERE system LIKE ? OR user LIKE ? OR response LIKE ? ORDER BY id DESC LIMIT ?",
            (like, like, like, int(limit)))
        out["scenes"] = st.db.query(
            "SELECT s.project_id,s.idx,substr(s.text,1,240) AS text,s.visual_prompt,s.mood_tag "
            "FROM scenes s WHERE s.text LIKE ? OR s.visual_prompt LIKE ? "
            "ORDER BY s.project_id DESC LIMIT ?", (like, like, int(limit)))
    else:
        out["projects"] = st.db.list_projects(limit=limit)
        out["prompts"] = st.db.query("SELECT id,project_id,stage,role,model,created_at FROM prompts "
                                     "ORDER BY id DESC LIMIT ?", (int(limit),))
    return out


@router.post("/preview/previz")
async def api_preview_previz(payload: dict = Body(default={})):
    """Render a 2s previz test frame/clip from a mood tag — instant feedback when
    the Director is choosing a look."""
    st = get_state()
    cfg = st.config()
    from . import previz

    out = os.path.join(ensure_dir(os.path.join(st.data_root, "tmp")),
                       f"previz_{new_id(5)}.mp4")
    res = await asyncio.to_thread(previz.render_clip, out,
                                 float(payload.get("duration") or 2.0),
                                 int(cfg["video"]["width"]), int(cfg["video"]["height"]),
                                 min(24, int(cfg["video"]["fps"]) + 4),
                                 payload.get("mood_tag") or "", payload.get("visual_prompt") or "",
                                 int(payload.get("seed") or 0), float(cfg["video"]["motion_strength"]))
    return {"url": "/api/tmpfile?name=" + os.path.basename(out), "engine": "previz", **res,
            "download_name": os.path.basename(out)}


@router.get("/style")
async def api_style():
    return {"guideline": style_mod.STYLE_GUIDELINE, "moods": sorted(style_mod.MOOD_AMBIENCE),
            "ambience_examples": style_mod.MOOD_AMBIENCE,
            "imagery_keys": [{"keys": list(k)[:4], "mood": m} for k, _v, m, _a in style_mod.KEYWORD_IMAGERY][:24]}


# ============================================================ live: WS + SSE
@router.websocket("/runs/{run_id}/events")
async def ws_run_events(ws: WebSocket, run_id: str):
    await ws.accept()
    st = STATE.get("app")
    if st is None:
        await ws.close()
        return
    q = st.bus.subscribe(run_id)
    try:
        snap = st.scheduler.status(run_id)
        await ws.send_text(json.dumps({"kind": "snapshot", "payload": snap}, ensure_ascii=False))
        while True:
            try:
                ev = await asyncio.wait_for(q.get(), timeout=25)
            except asyncio.TimeoutError:
                await ws.send_text(json.dumps({"kind": "ping", "ts": now()}))
                continue
            await ws.send_text(json.dumps(ev, ensure_ascii=False))
    except (WebSocketDisconnect, RuntimeError):
        pass
    except Exception:
        pass
    finally:
        st.bus.unsubscribe(run_id, q)


@router.websocket("/events")
async def ws_global_events(ws: WebSocket):
    await ws.accept()
    st = STATE.get("app")
    if st is None:
        await ws.close()
        return
    q = st.bus.subscribe("*")
    try:
        while True:
            ev = await q.get()
            await ws.send_text(json.dumps(ev, ensure_ascii=False))
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        st.bus.unsubscribe("*", q)


@router.get("/runs/{run_id}/stream")
async def sse_run(run_id: str, request: Request):
    """SSE fallback for environments where WebSocket is unavailable."""
    st = get_state()
    q = st.bus.subscribe(run_id)

    async def gen():
        try:
            snap = st.scheduler.status(run_id)
            yield "event: snapshot\ndata: " + json.dumps(snap, ensure_ascii=False) + "\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=20)
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
                    continue
                yield "event: " + str(ev.get("kind", "update")) + "\ndata: " + \
                    json.dumps(ev, ensure_ascii=False) + "\n\n"
        finally:
            st.bus.unsubscribe(run_id, q)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

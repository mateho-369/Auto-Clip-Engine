"""The nine stages: what each one actually does.

Contract — every stage returns a dict the scheduler applies verbatim::

    {ok, status?, engine, message, error?, progress?, assets[], scene_update{},
     project_update{}, run_update{}, requires_review?, notes[]}

* ``status`` may be ``deferred`` (queued for Machine A) — a *successful* outcome
  that downstream stages treat as satisfied, which is how a CPU-only machine
  still finishes a draft cut;
* no stage raises for expected problems (engine missing, model offline). Raising
  here means "bug or hard failure", and the scheduler retries once and then
  reports it on that stage's card.
"""
import asyncio
import json
import os

from .. import khmer, media, style as style_mod
from ..util import jdump, media_duration, write_json
from .context import scene_dict_for_prompt


async def run_stage(stage, ctx, job):
    fn = STAGE_IMPL.get(stage)
    if fn is None:
        raise RuntimeError(f"no implementation for stage '{stage}'")
    return await fn(ctx, job.scene_idx)


# --------------------------------------------------------------------- 0 · script
async def stage_script(ctx, _idx):
    """Mode A: lock the Director's script. Mode B: let the Controller write it."""
    project = ctx.project
    mode = (project.get("mode") or "A").upper()
    out_dir = ctx.project_dir()
    if mode == "A":
        script = khmer.normalize_block(project.get("script") or "")
        if not script:
            return {"ok": False, "error": "Mode A needs the Director's script — the project "
                                          "has none. Paste it on the project page."}
        path = ctx.write_text(ctx.asset_path("script", -1, ".txt"), script + "\n")
        # mechanical sentence pass so the board is never empty even before Stage 1
        sentences = khmer.split_sentences(script, max_chars=style_mod.SCENE_MAX_CHARS)
        return {
            "ok": True, "engine": "director-lock", "progress": 100.0,
            "message": f"Script locked: {len(sentences)} sentence(s), ~"
                       f"{khmer.estimate_speech_seconds(script, calm=ctx.cfg['pipeline']['pace_calm']):.0f}s. "
                       "No agent may rewrite it.",
            "assets": [{"kind": "script", "path": path, "scene_idx": -1, "meta": {
                "mode": "A", "sentences": len(sentences), "chars": khmer.char_len(script),
                "locked": True, "sha": _sha(script)}}],
            "project_update": {"script": script, "script_locked": True, "script_origin": "director",
                               "status": "ready", "title": project.get("title")
                               or khmer.title_from(script)},
            "notes": ["mode A — wording is ground truth; AI does segmentation only"],
        }

    # ---- Mode B
    script = (project.get("script") or "").strip()
    already = bool(script) and project.get("status") in ("ready", "rendering", "done", "approved")
    if already and "script" not in ctx.force_stages:
        return {"ok": True, "engine": "reused", "progress": 100.0,
                "message": f"Reusing the approved script ({khmer.char_len(script)} chars)",
                "assets": [{"kind": "script", "path": ctx.write_text(
                    ctx.asset_path("script", -1, ".txt"), script), "scene_idx": -1,
                    "meta": {"mode": "B", "reused": True}}]}
    from ..agents.auto_idea import generate

    llm = ctx.llm()
    cfg = dict(ctx.cfg)
    cfg["target_duration"] = float(project.get("target_duration") or 30)
    res = await generate(llm, project.get("topic_hint") or "", cfg,
                         style_notes=project.get("style_notes") or "",
                         regenerate_note=project.get("regenerate_note") or "")
    script = khmer.normalize_block(res.get("script") or "")
    if not script:
        return {"ok": False, "error": "auto-idea produced no script"}
    path = ctx.write_text(ctx.asset_path("script", -1, ".txt"), script + "\n")
    gate = ctx.cfg["pipeline"].get("review_gate", "auto")
    needs_review = gate in ("auto", "always") and not ctx.cfg["pipeline"].get("auto_approve_mode_b")
    out = {"ok": True, "engine": res.get("engine", "?"), "progress": 100.0,
           "message": f"{res.get('title') or 'Auto script'} · ~{res.get('estimated_seconds', 0):.0f}s "
                      f"({res.get('engine')})",
           "assets": [{"kind": "script", "path": path, "scene_idx": -1,
                       "meta": {"mode": "B", "engine": res.get("engine"), "topic": project.get("topic_hint"),
                                "khmer_ratio": res.get("khmer_ratio"),
                                "notes": res.get("notes") or []}}],
           "project_update": {"script": script, "script_locked": False,
                              "script_origin": res.get("origin") or "ai",
                              "title": (res.get("title") or khmer.title_from(script))[:120],
                              "status": "review" if needs_review else "ready"},
           "notes": res.get("notes") or [], "requires_review": bool(needs_review),
           "generated_logline": res.get("logline", "")}
    ctx.db.log_prompt(project_id=ctx.project_id, run_id=ctx.run_id, stage="script", scene_idx=-1,
                      role="auto_idea", model=res.get("model") or res.get("engine") or "",
                      engine=res.get("engine") or "", system="", user=json.dumps(
                          {"topic_hint": project.get("topic_hint"),
                           "target_duration": cfg["target_duration"]}, ensure_ascii=False),
                      response=script[:6000], ok=True, latency_ms=int(res.get("latency_ms") or 0))
    return out


def _sha(text):
    import hashlib
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:12]


# ------------------------------------------------------------------ 1 · breakdown
async def stage_breakdown(ctx, _idx):
    project = ctx.project
    script = (project.get("script") or "").strip()
    if not script:
        return {"ok": False, "error": "no script to break down"}
    board = ctx.db.list_scenes(ctx.project_id) or None
    from ..agents.controller import break_down

    scenes, meta = await break_down(ctx.llm(), script, ctx.cfg, plan_scenes=board)
    if not scenes:
        return {"ok": False, "error": "segmentation produced no scenes"}
    limit = int(ctx.cfg["pipeline"].get("max_scenes", 12))
    scenes = scenes[:limit]
    for i, s in enumerate(scenes):
        s["idx"] = i
        s.setdefault("sfx_prompt", style_mod.ambience_for(s.get("mood_tag"), s.get("visual_prompt")))
    ctx.db.replace_scenes(ctx.project_id, scenes)
    ctx.reload_scenes()
    path = write_json(ctx.asset_path("scenes", -1, ".json"),
                      {"engine": meta.get("engine"), "script_sha": _sha(script),
                       "scenes": scenes, "integrity": meta.get("integrity") or {}})
    ctx.log_engine_prompt("breakdown", -1, meta.get("engine", "deterministic"),
                          model=meta.get("model") or meta.get("engine") or "deterministic",
                          system="segment the Director's script verbatim; emit JSON scenes",
                          user=json.dumps({"script_chars": len(script), "max_scenes": limit,
                                           "script_sha": _sha(script)}, ensure_ascii=False),
                          response=jdump([{k: sc.get(k) for k in
                                          ("idx", "text", "visual_prompt", "mood_tag",
                                           "estimated_duration_sec")} for sc in scenes])[:4000])
    est = round(sum(float(s.get("estimated_duration_sec") or 0) for s in scenes), 1)
    notes = list(meta.get("notes") or [])
    integrity = meta.get("integrity") or {}
    if integrity.get("ok") is False:
        notes.append("⚠️ scene text did not match the Director's script exactly")
    elif (project.get("mode") or "A").upper() == "A":
        notes.append("integrity verified: scene texts rejoin the Director's script byte-for-byte")
    return {"ok": True, "engine": meta.get("engine", "deterministic"), "progress": 100.0,
            "message": f"{len(scenes)} scene(s), ~{est:.0f}s · model={meta.get('engine')}",
            "assets": [{"kind": "scenes", "path": path, "scene_idx": -1,
                        "meta": {"count": len(scenes), "est_total": est, "engine": meta.get("engine"),
                                 "integrity": integrity}}],
            "run_update": {"scene_count": len(scenes), "estimated_total": est},
            "notes": notes}


# ------------------------------------------------------------------ 3a · voice
async def stage_voice_base(ctx, idx):
    scene = ctx.db.get_scene(ctx.project_id, idx)
    if not scene or not (scene.get("text") or "").strip():
        return {"ok": False, "error": f"scene {idx} has no text"}
    from ..engines import tts

    engine = (ctx.plan.get("tts") or {}).get("engine", "placeholder")
    out = ctx.asset_path("voice", idx, ".wav")
    if os.path.exists(out) and "voice_base" not in ctx.force_stages:
        os.remove(out)
    res = await asyncio.to_thread(tts.synthesize, scene["text"], out, ctx.cfg, engine,
                                  ctx.progress_cb("voice_base", idx, 5, 92, "synthesising · "),
                                  idx + _run_seed(ctx))
    if not res.get("ok"):
        return {"ok": False, "error": f"voice synthesis failed: {res.get('reason')}",
                "engine": engine}
    facts = ctx.voice_facts(out)
    ctx.log_engine_prompt("voice_base", idx, res.get("engine", engine),
                          model=str(res.get("model") or engine),
                          system=f"lang_id/sid={res.get('sid', 0)} speed={res.get('speed', 1.0)}",
                          user=scene["text"],
                          response=f"{facts['duration']:.2f}s wav · peaks={facts['peak']:.3f}",
                          latency_ms=int(res.get("latency_ms") or 0))
    meta = {k: v for k, v in res.items() if k not in ("ok",)}
    meta.update(facts)
    return {"ok": True, "engine": res.get("engine", engine), "progress": 100.0,
            "message": (f"{facts['duration']:.2f}s voice · {res.get('engine')}"
                        + ("" if res.get("real_speech") else " ⚠ placeholder")),
            "assets": [{"kind": "voice", "path": out, "scene_idx": idx, "duration": facts["duration"],
                        "meta": meta}],
            "scene_update": {"audio_duration": 0},      # authoritative duration arrives with 3b
            "notes": [res.get("note")] if res.get("note") else []}


# ------------------------------------------------------------------ 3b · timbre
async def stage_voice_final(ctx, idx):
    scene = ctx.db.get_scene(ctx.project_id, idx)
    base = ctx.latest_asset("voice", scene_idx=idx)
    if not base:
        return {"ok": False, "error": "no 3a voice to convert"}
    from ..engines import rvc

    plan_rvc = (ctx.plan.get("rvc") or {}).get("engine", "bypass")
    out = ctx.asset_path("voice_final", idx, ".wav")
    profile = _resolve_profile(ctx)
    if plan_rvc in ("off",):
        import shutil
        shutil.copyfile(base["path"], out)
        return {"ok": True, "status": "skipped", "engine": "off", "progress": 100.0,
                "message": "timbre stage off — using the 3a Khmer voice as the final voice",
                "assets": [{"kind": "voice_final", "path": out, "scene_idx": idx,
                            "duration": media_duration(out), "meta": {"converted": False,
                                                                      "reason": "rvc disabled"}}],
                "scene_update": {"audio_duration": media_duration(out)}}
    res = await asyncio.to_thread(rvc.convert, base["path"], out, ctx.cfg, profile,
                                  ctx.progress_cb("voice_final", idx, 5, 95, "RVC · "))
    if not res.get("ok"):
        return {"ok": False, "error": f"timbre conversion failed: {res.get('reason')}",
                "engine": plan_rvc}
    dur = media_duration(out, 0.0)
    facts = ctx.voice_facts(out)
    meta = {"converted": bool(res.get("converted")), "engine": res.get("engine"),
            "profile": (profile or {}).get("name", ""), "reason": res.get("reason", ""),
            "pitch": int(ctx.cfg["rvc"].get("pitch") or 0), **facts}
    msg = (f"voice in '{(profile or {}).get('name') or 'base'}' timbre · {facts['duration']:.2f}s"
           if res.get("converted") else
           f"timbre not converted ({res.get('engine')}): {str(res.get('reason') or '')[:90]}")
    return {"ok": True, "engine": res.get("engine") or plan_rvc, "progress": 100.0,
            "message": msg,
            "assets": [{"kind": "voice_final", "path": out, "scene_idx": idx,
                        "duration": facts["duration"], "meta": meta}],
            "scene_update": {"audio_duration": round(facts["duration"], 3)},
            "notes": [res.get("reason")] if res.get("reason") and not res.get("converted") else []}


def _resolve_profile(ctx):
    """Selected voice profile (per-project override → settings → first available)."""
    pid = ctx.project.get("voice_profile_id") or ctx.cfg["rvc"].get("profile_id") or ""
    if pid:
        prof = ctx.db.get_voice_profile(pid)
        if prof:
            return prof
    profs = ctx.db.list_voice_profiles()
    if profs:
        return profs[0]
    found = []
    try:
        from ..engines.rvc import discover_profiles

        found = discover_profiles(ctx.cfg)
    except Exception:
        found = []
    return found[0] if found else None


# --------------------------------------------------------------------- 4 · video
async def stage_video(ctx, idx):
    scene = ctx.db.get_scene(ctx.project_id, idx)
    if not scene:
        return {"ok": False, "error": f"scene {idx} missing"}
    plan_video = ctx.plan.get("video") or {}
    engine = plan_video.get("engine", "previz")
    if engine in ("defer", "off"):
        return {"ok": True, "status": "deferred", "engine": engine, "progress": 0.0,
                "message": plan_video.get("reason") or "video queued for the GPU machine",
                "notes": [f"scene {idx + 1}: the final cut will use a CPU previz draft until "
                          f"you run the GPU catch-up"]}
    from ..engines import video as video_engine

    out = ctx.asset_path("video", idx, ".mp4")
    est = float(scene.get("estimated_duration_sec") or 4.0)
    aud = float(scene.get("audio_duration") or 0)
    target = max(1.0, aud or est)          # whatever is known now; 4b fixes it exactly
    seed = idx + _run_seed(ctx) if int(ctx.cfg["video"].get("seed", -1)) == -1 else int(
        ctx.cfg["video"].get("seed") or 0)
    res = await asyncio.to_thread(video_engine.render_scene_clip, scene, out, ctx.cfg, ctx.plan,
                                  target, ctx.progress_cb("video", idx, 3, 96, f"{engine} · "),
                                  seed, _reference_image(ctx))
    if not res.get("ok"):
        return {"ok": False, "error": f"video render failed: {str(res.get('reason'))[:240]}",
                "engine": engine}
    ctx.log_engine_prompt("video", idx, res.get("engine") or engine,
                          model=str(res.get("model") or ""),
                          system=str(res.get("negative_prompt") or "")[:1500],
                          user=jdump({k: res.get(k) for k in ("prompt", "width", "height", "fps",
                                                               "frames", "steps", "cfg", "seed")
                                      if res.get(k) is not None})[:4000],
                          response=f"{res.get('duration', 0):.2f}s clip · {os.path.basename(out)}",
                          latency_ms=int(res.get("latency_ms") or 0))
    meta = {k: v for k, v in res.items() if k not in ("ok", "path")}
    meta["target_duration"] = round(target, 2)
    meta["prompt"] = res.get("prompt", "")
    if res.get("prompt_id"):
        meta["comfy_prompt_id"] = res["prompt_id"]
    return {"ok": True, "engine": res.get("engine") or engine, "progress": 100.0,
            "message": f"{res.get('duration', 0):.2f}s clip · {res.get('engine')}"
                       + (f" · {len(res.get('vram_notes') or [])} VRAM adjust" if res.get("vram_notes") else ""),
            "assets": [{"kind": "video", "path": out, "scene_idx": idx,
                        "duration": res.get("duration", 0), "meta": meta}],
            "notes": (res.get("vram_notes") or []) + ([res.get("fallback_reason")]
                                                      if res.get("fallback_reason") else [])}


def _reference_image(ctx):
    """A character/key frame to seed Wan2.2 TI2V, when the project has one."""
    d = os.path.join(ctx.project_dir(), "reference")
    if not os.path.isdir(d):
        return None
    for fn in sorted(os.listdir(d)):
        if fn.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            return os.path.join(d, fn)
    return None


# ----------------------------------------------------------------- 4b · duration match
async def stage_video_fit(ctx, idx):
    scene = ctx.db.get_scene(ctx.project_id, idx)
    video = ctx.latest_asset("video", scene_idx=idx)
    voice = ctx.latest_asset("voice_final", scene_idx=idx) or ctx.latest_asset("voice", scene_idx=idx)
    plan_video = ctx.plan.get("video") or {}
    if not video:
        if plan_video.get("engine") in ("defer", "off"):
            return {"ok": True, "status": "deferred", "engine": "defer",
                    "message": "nothing to fit — video render deferred to the GPU machine"}
        return {"ok": False, "error": "no video clip to fit"}
    target = float((voice or {}).get("duration") or 0) or float(
        scene.get("estimated_duration_sec") or 4.0)
    tol = float(ctx.cfg["pipeline"].get("duration_tolerance_sec", 0.9))
    have = float(video.get("duration") or 0)
    out = ctx.asset_path("video_fit", idx, ".mp4")
    if abs(have - target) <= max(0.12, tol * 0.35):
        # close enough: reference the existing clip, no re-encode (saves 10-30s/scene)
        meta = {"duration": have or target, "adjusted": False, "drift_sec": round(have - target, 3),
                "source": "video", "engine": "passthrough"}
        return {"ok": True, "engine": "passthrough", "progress": 100.0,
                "message": f"picture already matches voice ({have:.2f}s vs {target:.2f}s) — no re-encode",
                "assets": [{"kind": "video_fit", "path": video["path"], "scene_idx": idx,
                            "duration": have or target, "meta": meta}]}
    res = await asyncio.to_thread(media.fit_video, video["path"], out, target,
                                  int(ctx.cfg["video"]["width"]), int(ctx.cfg["video"]["height"]),
                                  int(ctx.cfg["assembly"]["fps"]), "auto", True, 0.0)
    if not os.path.exists(out):
        return {"ok": False, "error": f"duration fit failed: {res}"}
    dur = _dur(out)
    return {"ok": True, "engine": "ffmpeg-fit", "progress": 100.0,
            "message": f"fitted {have:.2f}s → {dur:.2f}s (voice {target:.2f}s)",
            "assets": [{"kind": "video_fit", "path": out, "scene_idx": idx, "duration": dur,
                        "meta": {"adjusted": True, "drift_sec": round(dur - target, 3),
                                 "source_duration": have, "target": round(target, 3)}}]}


def _run_seed(ctx):
    """Stable per-run offset so 'regenerate' gives a *different* take, and a
    resumed run gives the same one as its predecessor."""
    src = (ctx.resume_from or ctx.run_id or "run") + str(ctx.cfg.get("video", {}).get("steps", 0))
    return int(_sha(src)[:5], 16) % 997


def _dur(path):
    from ..util import media_duration

    return media_duration(path, 0.0)


# ----------------------------------------------------------------------- 5 · sfx
async def stage_sfx(ctx, idx):
    scene = ctx.db.get_scene(ctx.project_id, idx)
    plan_sfx = ctx.plan.get("sfx") or {}
    engine = plan_sfx.get("engine", "procedural")
    if engine in ("defer", "off"):
        return {"ok": True, "status": "deferred", "engine": engine, "progress": 0.0,
                "message": plan_sfx.get("reason") or "MMAudio ambience queued for the GPU machine"}
    video = ctx.latest_asset("video", scene_idx=idx) or ctx.latest_asset("video_fit", scene_idx=idx)
    fit = ctx.latest_asset("video_fit", scene_idx=idx)
    voice = ctx.latest_asset("voice_final", scene_idx=idx) or ctx.latest_asset("voice", scene_idx=idx)
    target = max(0.8, float((fit or video or {}).get("duration") or 0) or
                 float((voice or {}).get("duration") or 0) or
                 float(scene.get("estimated_duration_sec") or 4.0))
    out = ctx.asset_path("ambient", idx, ".wav")
    from ..engines import sfx as sfx_engine

    res = await asyncio.to_thread(sfx_engine.render, (video or {}).get("path"), out, scene, ctx.cfg,
                                  target, ctx.progress_cb("sfx", idx, 5, 95, f"{engine} · "),
                                  idx * 7 + 3)
    if not res.get("ok"):
        return {"ok": True, "status": "skipped", "engine": "none", "progress": 100.0,
                "message": f"no ambience for this scene ({str(res.get('reason'))[:110]}) — "
                           "voice-only cut continues",
                "notes": [str(res.get("reason"))[:200]]}
    ctx.log_engine_prompt("sfx", idx, res.get("engine") or engine,
                          model=str(res.get("model") or ""),
                          system=str(res.get("video_conditioned") and "conditioned on the clip" or ""),
                          user=str(res.get("prompt") or scene.get("sfx_prompt") or ""),
                          response=f"{res.get('duration', 0):.2f}s ambience · "
                                   f"{','.join(res.get('layers') or [])}")
    meta = {k: v for k, v in res.items() if k not in ("ok", "path")}
    meta["video_conditioned"] = bool(video and res.get("engine") == "mmaudio")
    return {"ok": True, "engine": res.get("engine") or engine, "progress": 100.0,
            "message": f"{res.get('duration', 0):.1f}s ambience · {res.get('engine')}"
                       + (f" [{','.join(res.get('layers') or [])}]" if res.get("layers") else ""),
            "assets": [{"kind": "ambient", "path": out, "scene_idx": idx,
                        "duration": res.get("duration", 0), "meta": meta}],
            "notes": [res.get("fallback_reason")] if res.get("fallback_reason") else []}


# ----------------------------------------------------------------------- 6 · QA
async def stage_qa(ctx, idx):
    scene = ctx.db.get_scene(ctx.project_id, idx)
    if not scene:
        return {"ok": False, "error": f"scene {idx} missing"}
    facts = {"voice_duration": 0.0, "video_duration": 0.0, "ambient_duration": 0.0,
             "voice_engine": "", "video_engine": "", "voice_engine_ok": True,
             "ambient_engine": "", "head_silence": 0.0, "tail_silence": 0.0,
             "ambient_layers": "", "peak": 0.0}
    assets = {}
    slots = {}
    for kind in ("voice", "voice_final", "video", "video_fit", "ambient"):
        a = ctx.latest_asset(kind, scene_idx=idx)
        if a:
            slots[kind] = {"path": a["path"], "duration": float(a.get("duration") or 0),
                           **{k: v for k, v in (a.get("meta") or {}).items()}}
    assets.update(slots)
    voice = slots.get("voice_final") or slots.get("voice")
    if voice:
        facts["voice_duration"] = float(voice.get("duration") or 0)
        facts["voice_engine"] = str(voice.get("engine") or "")
        facts["voice_engine_ok"] = "placeholder" not in facts["voice_engine"]
        facts["head_silence"] = float(voice.get("head_silence") or 0)
        facts["tail_silence"] = float(voice.get("tail_silence") or 0)
        facts["peak"] = float(voice.get("peak") or 0)
    for key in ("voice", "voice_final"):
        if key in slots:
            slots[key]["peak"] = facts["peak"]
    assets["voice_engine"] = facts["voice_engine"]
    assets["voice_engine_ok"] = facts["voice_engine_ok"]
    assets["_head_silence"] = facts["head_silence"]
    assets["_tail_silence"] = facts["tail_silence"]
    vid = slots.get("video_fit") or slots.get("video")
    if vid:
        facts["video_duration"] = float(vid.get("duration") or 0)
        facts["video_engine"] = str(vid.get("engine") or "").replace("+chunked", "")
        assets["video_engine"] = facts["video_engine"]
        assets["video_fit"] = dict(vid, duration=facts["video_duration"]) if "video_fit" in vid else \
            assets.get("video_fit") or {"path": vid.get("path"), "duration": facts["video_duration"]}
        assets.setdefault("video", {"path": vid.get("path"), "duration": facts["video_duration"]})
    amb = slots.get("ambient")
    if amb:
        facts["ambient_duration"] = float(amb.get("duration") or 0)
        facts["ambient_engine"] = str(amb.get("engine") or "")
        facts["ambient_layers"] = ",".join(amb.get("layers") or [])
        assets["ambient"] = {"duration": facts["ambient_duration"]}

    from ..agents.qa import review_scene

    res = await review_scene(ctx.llm(), scene, assets, ctx.cfg, idx, run_id=ctx.run_id,
                             project_id=ctx.project_id)
    out = ctx.asset_path("qa", idx, ".json")
    write_json(out, {"facts": facts, **res})
    if res.get("engine") == "deterministic":     # LLM path logs itself inside llm.ask
        ctx.log_engine_prompt("qa", idx, "deterministic", model="deterministic",
                              system="check duration, loudness, engine substitution; emit JSON",
                              user=jdump(facts)[:2000],
                              response=jdump({"approved": res.get("approved"),
                                              "issues": res.get("issues")})[:3000])
    hard = [i for i in res["issues"] if i.get("severity") == "fail"]
    warn = [i for i in res["issues"] if i.get("severity") != "fail"]
    blocked = bool(hard) and ctx.cfg["pipeline"].get("require_qa_pass")
    status = "failed" if blocked else "done"
    msg = ("✅ clean" if not res["issues"] else
           f"{len(hard)} fail · {len(warn)} warn: " + (res["issues"][0]["issue"][:110]))
    return {"ok": not blocked, "status": status, "engine": res.get("engine", "deterministic"),
            "progress": 100.0, "message": msg, "error": msg if blocked else "",
            "assets": [{"kind": "qa", "path": out, "scene_idx": idx,
                        "meta": {"approved": bool(res.get("approved")), "issues": res["issues"],
                                 "engine": res.get("engine"), "summary": res.get("summary", "")}}],
            "qa": res}


# --------------------------------------------------------------------- 7 · assembly
async def stage_assemble(ctx, _idx):
    scenes = ctx.db.list_scenes(ctx.project_id)
    if not scenes:
        return {"ok": False, "error": "no scenes to assemble"}
    stage_assets = {}
    for kind in ("voice", "voice_final", "video", "video_fit", "ambient", "qa"):
        stage_assets[kind] = {}
        for s in scenes:
            a = ctx.latest_asset(kind, scene_idx=s["idx"])
            if a:
                stage_assets[kind][s["idx"]] = {"path": a["path"], "duration": a["duration"],
                                               "engine": (a.get("meta") or {}).get("engine"),
                                               "meta": a.get("meta") or {}}
    from ..engines import assembly

    out_dir = ctx.final_dir()
    res = await asyncio.to_thread(assembly.assemble, ctx.project, scenes, stage_assets, ctx.cfg,
                                  out_dir, ctx.run_id,
                                  ctx.progress_cb("assemble", -1, 5, 95, "render · "))
    final_path = res.get("path")
    if not final_path or not os.path.exists(final_path):
        return {"ok": False, "error": "assembly produced no file"}
    assets = [{"kind": "final", "path": final_path, "scene_idx": -1,
               "duration": res.get("duration", 0),
               "meta": {k: res.get(k) for k in ("width", "height", "fps", "notes", "scenes")}
                       | {"draft": any("previz" in str(n) or "black slate" in str(n)
                                       for n in (res.get("notes") or []))}}]
    for key, kind in (("srt", "srt"), ("poster", "poster"), ("manifest", "manifest"),
                      ("with_captions", "final_captions")):
        p = res.get(key)
        if p and os.path.exists(p):
            assets.append({"kind": kind, "path": p, "scene_idx": -1,
                           "duration": 0, "meta": {"of": os.path.basename(final_path)}})
    dur = float(res.get("duration") or 0)
    want = float(ctx.project.get("target_duration") or 0)
    note = f"{dur:.1f}s · {res.get('width')}x{res.get('height')}@{res.get('fps')}"
    if want:
        note += f" (target {want:.0f}s, {100.0 * dur / want:.0f}%)"
    return {"ok": True, "engine": "ffmpeg", "progress": 100.0, "message": note,
            "assets": assets,
            "project_update": {"status": "done"},
            "run_update": {"final_path": final_path, "duration": round(dur, 2),
                           "size_bytes": os.path.getsize(final_path)},
            "notes": res.get("notes") or [],
            "final_asset_path": final_path}


STAGE_IMPL = {
    "script": stage_script,
    "breakdown": stage_breakdown,
    "voice_base": stage_voice_base,
    "voice_final": stage_voice_final,
    "video": stage_video,
    "video_fit": stage_video_fit,
    "sfx": stage_sfx,
    "qa": stage_qa,
    "assemble": stage_assemble,
}

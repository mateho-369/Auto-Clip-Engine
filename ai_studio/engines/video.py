"""Stage 4 — the Animator: one silent video clip per scene, Wan via ComfyUI.

Primary: ComfyUI + **Wan2.1 T2V 1.3B** at 480p (the pick for a 5070 8GB — it is
the most reliable option in that envelope). A Wan2.2 TI2V-5B workflow is shipped
too; `auto` stays on 1.3B unless the Director selects it, because the 5B only
fits 8GB through ComfyUI's offloading and costs 3-5× the time per clip.

Because the brief wants voice and video generated *concurrently*, the clip length
here comes from Stage 1's estimate; `pipeline.stages.video_fit` then does a cheap
CPU re-timing so the picture matches the finished voice exactly. That is the
whole reason the two branches can overlap without drifting apart.

Fallbacks, in order: ComfyUI error that looks like OOM → retry with the VRAM
guard's reduced frames/size → procedural previz (:mod:`ai_studio.previz`).
"""
import os
import time

from .. import khmer, workflows
from .. import previz
from ..comfy import ComfyUIClient, ComfyError
from ..util import ensure_dir, media_duration, rel
from ..vram import free_vram_mb, frames_for, guard_request

STYLE_TAIL = ("calm documentary look, soft natural light, gentle slow camera drift, "
              "muted warm palette, peaceful natural scenery, film-like, no text, no captions")


def compose_prompt(scene, cfg):
    """Positive prompt = the scene's visual tag + house look + the spoken line's mood."""
    vp = (scene.get("visual_prompt") or "").strip().rstrip(".")
    if not vp:
        from .. import style as style_mod
        vp = style_mod.DEFAULT_VISUAL
    mood = (scene.get("mood_tag") or "").replace("-", " ")
    parts = [vp]
    if mood and mood.lower() not in vp.lower():
        parts.append(f"{mood} atmosphere")
    style_tail = (cfg.get("video", {}) or {}).get("style_tail")
    parts.append(style_tail if style_tail is not None else STYLE_TAIL)
    return ", ".join(p for p in parts if p)[:900]


def render(scene, out_path, cfg, plan, target_duration, progress=None, seed=0,
           reference_image=None, attempt=1):
    """Produce one silent clip for a scene. Returns a dict (never raises)."""
    v = cfg.get("video", {})
    engine = (plan.get("video") or {}).get("engine", "previz") if plan else "previz"
    if engine == "comfyui":
        res = _comfyui(scene, out_path, cfg, target_duration, progress, seed, reference_image, attempt)
        if res.get("ok"):
            return res
        if not res.get("oom") and "fallback_to_previz" not in res:
            res["fallback_to_previz"] = True
        prev = previz_clip(scene, out_path, cfg, target_duration, progress, seed)
        prev["fallback_from"] = "comfyui"
        prev["fallback_reason"] = str(res.get("reason", ""))[:300]
        return prev
    if engine == "previz":
        return previz_clip(scene, out_path, cfg, target_duration, progress, seed)
    return {"ok": False, "engine": engine, "reason": f"video engine '{engine}' does not render here"}


def _values(scene, cfg, target_duration, seed, out_prefix, frames, width, height):
    v = cfg.get("video", {})
    fps = int(v.get("fps", 16))
    sec = max(0.5, float(target_duration))
    my_seed = int(v.get("seed", -1))
    if my_seed == -1:
        my_seed = (abs(int(seed or 0)) * 7919 + int(time.time()) % 9973) % 2**31
    return {
        "PROMPT": compose_prompt(scene, cfg),
        "NEGATIVE": v.get("negative_prompt") or "",
        "WIDTH": int(width), "HEIGHT": int(height),
        "FRAMES": int(frames), "FPS": int(fps),
        "DURATION": round(sec, 3),
        "STEPS": int(v.get("steps", 20)), "CFG": float(v.get("cfg", 6.0)),
        "SHIFT": float(v.get("shift", 8.0)), "SEED": int(my_seed),
        "MOTION": float(v.get("motion_strength", 0.75)),
        "TEXT": khmer.truncate_clusters(scene.get("text") or "", 400),
        "MOOD": scene.get("mood_tag") or "",
        "OUT_PREFIX": out_prefix,
    }


def _comfyui(scene, out_path, cfg, target_duration, progress, seed, reference_image, attempt):
    v = cfg.get("video", {})
    host = v.get("comfy_host") or "http://127.0.0.1:8188"
    client = ComfyUIClient(host)
    if not client.is_online():
        return {"ok": False, "engine": "comfyui", "reason": f"ComfyUI not reachable at {host}"}
    fps = int(v.get("fps", 16))
    frames = frames_for(target_duration, fps, v.get("min_frames", 17), v.get("max_frames", 81))
    free = free_vram_mb(cfg, client)
    frames, w, h, notes = guard_request(cfg, frames, v.get("width", 480), v.get("height", 854), free)
    if notes and progress:
        progress(2.0, "VRAM guard: " + "; ".join(notes))
    try:
        wf_path, template, _ = workflows.resolve_workflow(
            v.get("workflow"), cfg, default="wan2.1_t2v_1.3b_480p")
    except Exception as e:
        return {"ok": False, "engine": "comfyui", "reason": f"workflow: {e}"}
    prefix = "ai_studio/" + os.path.splitext(os.path.basename(out_path))[0]
    values = _values(scene, cfg, target_duration, seed, prefix, frames, w, h)
    if reference_image and "{{START_IMAGE}}" in str(template):
        try:
            up = client.upload_image(reference_image)
            values["START_IMAGE"] = f"{up['subfolder']}/{up['name']}" if up.get("subfolder") else up["name"]
        except Exception as e:
            values["START_IMAGE"] = ""
            notes.append(f"start-frame upload failed ({str(e)[:60]})")
    wf, report = workflows.render(template, values)
    if report.get("unresolved"):
        return {"ok": False, "engine": "comfyui",
                "reason": "workflow has unresolved placeholders: " + ", ".join(report["unresolved"])
                          + f" (edit {wf_path})"}
    ensure_dir(os.path.dirname(out_path) or ".")
    try:
        pid = client.queue_prompt(wf)
    except ComfyError as e:
        msg = str(e)
        return {"ok": False, "engine": "comfyui", "reason": msg,
                "oom": "out of memory" in msg.lower() or "cuda" in msg.lower()}
    try:
        if progress:
            progress(6.0, f"queued at ComfyUI ({pid[:8]}) · {w}x{h} · {frames}f")
        outputs = client.wait(pid, timeout=int(v.get("timeout_sec", 2400)),
                              on_progress=lambda pct, node: progress and progress(
                                  8.0 + pct * 0.86, node or "sampling"),
                              cancel=None)
    except ComfyError as e:
        msg = str(e)
        low = msg.lower()
        return {"ok": False, "engine": "comfyui", "reason": msg,
                "oom": "out of memory" in low or "allocat" in low or "vae" in low and "size" in low}
    refs = ComfyUIClient.output_refs(outputs)
    vids = [r for r in refs if r["filename"].lower().endswith((".mp4", ".webm", ".mov", ".mkv"))]
    if not vids:
        return {"ok": False, "engine": "comfyui",
                "reason": "job finished but produced no video file (check SaveVideo node)"}
    try:
        client.download(vids[0], out_path)
    except Exception as e:
        return {"ok": False, "engine": "comfyui", "reason": f"download failed: {str(e)[:150]}"}
    dur = media_duration(out_path, 0.0)
    client.free_memory()
    return {"ok": True, "engine": "comfyui-wan", "path": out_path, "duration": dur,
            "width": w, "height": h, "frames": frames, "fps": fps,
            "prompt": values["PROMPT"], "negative": values["NEGATIVE"], "seed": values["SEED"],
            "workflow": os.path.basename(wf_path), "prompt_id": pid, "vram_notes": notes,
            "applied": report.get("used"), "target_duration": float(target_duration)}


def previz_clip(scene, out_path, cfg, target_duration, progress=None, seed=0):
    """CPU-only animated clip — the draft / Machine-B / OOM-fallback renderer."""
    v = cfg.get("video", {})
    ensure_dir(os.path.dirname(out_path) or ".")
    try:
        info = previz.render_clip(
            out_path, duration=max(1.0, float(target_duration)),
            width=int(v.get("width", 480)), height=int(v.get("height", 854)),
            fps=min(24, int(v.get("fps", 16)) + 4),
            mood_tag=scene.get("mood_tag") or "", visual_prompt=scene.get("visual_prompt") or "",
            seed=int(seed or 0), motion=float(v.get("motion_strength", 0.75)),
            progress=progress)
        info["prompt"] = compose_prompt(scene, cfg)
        info["duration"] = media_duration(out_path, info.get("duration", 0.0))
        info["target_duration"] = float(target_duration)
        info["engine"] = "previz"
        info["ok"] = True
        return info
    except Exception as e:
        return {"ok": False, "engine": "previz", "reason": f"previz render failed: {str(e)[:200]}"}


def probe(cfg, plan=None):
    v = cfg.get("video", {})
    host = v.get("comfy_host") or ""
    client = ComfyUIClient(host)
    online = client.is_online()
    out = {"host": host, "online": online, "workflow": v.get("workflow"),
           "engine": (plan or {}).get("video", {}).get("engine"),
           "models": {}, "free_vram_mb": None}
    if online:
        try:
            stats = client.system_stats()
            devs = stats.get("devices") or []
            out["free_vram_mb"] = int((devs[0].get("vram_free") or 0) / 1e6) if devs else None
            out["comfy_version"] = stats.get("system", {}).get("version")
        except Exception:
            pass
        for cls in ("Wan22ImageToVideoLatent", "MMAudio", "CreateVideo", "SaveVideo"):
            try:
                out["models"][cls] = client.has_node(cls)
            except Exception:
                out["models"][cls] = False
    try:
        path, wf, _ = workflows.resolve_workflow(v.get("workflow"), cfg)
        out["workflow_path"] = path
        out["placeholders"] = workflows.template_placeholders(wf)
        out["missing_required"] = workflows.missing_required(wf)
    except Exception as e:
        out["workflow_error"] = str(e)[:200]
    return out


def clip_capacity_sec(cfg):
    """How long one generated clip can be with the current settings (seconds)."""
    v = cfg.get("video", {})
    fps = max(6, int(v.get("fps", 16)))
    return max(1.0, int(v.get("max_frames", 81)) / float(fps))


def render_scene_clip(scene, out_path, cfg, plan, target_duration, progress=None, seed=0,
                      reference_image=None):
    """Whole-scene picture, chunked when the clip budget is shorter than the scene.

    An 8GB card gives ~5s per Wan clip at 480p while a narrated scene is often
    8-12s, so we render up to `max_clips` sequential clips (different seeds, the
    same prompt/mood) and concatenate them. Previz chunks the same way, which keeps
    the two paths' behaviour identical.
    """
    from ..media import concat_clips
    from ..util import ensure_dir, media_duration

    cap = clip_capacity_sec(cfg)
    total = max(0.6, float(target_duration))
    n_clips = 1 if total <= cap * 1.05 else min(6, int(-(-total // cap)))
    if n_clips <= 1:
        return render(scene, out_path, cfg, plan, total, progress=progress, seed=seed,
                      reference_image=reference_image)
    per = total / float(n_clips)
    parts, notes, engines, prompts = [], [], set(), []
    ensure_dir(os.path.dirname(out_path) or ".")
    work = ensure_dir(os.path.join(os.path.dirname(out_path), ".clips"))
    ok_any = False
    for c in range(n_clips):
        part = os.path.join(work, f"{os.path.splitext(os.path.basename(out_path))[0]}.c{c}.mp4")
        lo = 100.0 * c / n_clips
        hi = 100.0 * (c + 1) / n_clips
        cb = None
        if progress:
            def cb(pct, note="", _lo=lo, _hi=hi, _c=c):
                progress(_lo + (_hi - _lo) * max(0.0, min(100.0, float(pct))) / 100.0,
                         f"clip {_c + 1}/{n_clips} · {note}")
        res = render(scene, part, cfg, plan, per, progress=cb, seed=int(seed or 0) + c * 101,
                     reference_image=reference_image if c == 0 else None)
        if res.get("ok") and os.path.exists(part):
            parts.append(part)
            engines.add(str(res.get("engine")))
            if res.get("prompt"):
                prompts.append(res["prompt"])
            notes.extend(res.get("vram_notes") or [])
            ok_any = True
        else:
            notes.append(f"clip {c + 1} failed: {str(res.get('reason'))[:120]}")
        if not ok_any and c == 0 and engines == set():
            # first chunk already failed in every engine — bail out with that reason
            return {"ok": False, "engine": "comfyui", "reason": str(res.get("reason"))[:300]}
    if not parts:
        return {"ok": False, "engine": "previz", "reason": "all clips failed: " + "; ".join(notes)[:260]}
    try:
        concat_clips(parts, out_path, work_dir=work)
    except Exception as e:
        return {"ok": False, "engine": "concat", "reason": f"clip concat failed: {str(e)[:180]}"}
    for p in parts:
        try:
            os.remove(p)
        except Exception:
            pass
    dur = media_duration(out_path, total)
    return {"ok": True, "engine": "+".join(sorted(engines)) or "chunked", "path": out_path,
            "duration": dur, "width": int(cfg.get("video", {}).get("width", 480)),
            "height": int(cfg.get("video", {}).get("height", 854)),
            "fps": int(cfg.get("video", {}).get("fps", 16)), "clips": len(parts),
            "prompt": prompts[0] if prompts else compose_prompt(scene, cfg),
            "vram_notes": notes, "target_duration": total, "chunked": True}

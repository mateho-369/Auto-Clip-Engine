"""Stage 5 — SFX Director: natural ambience *from the rendered picture*.

Primary engine is **MMAudio** (video-to-audio) through ComfyUI, using the small
checkpoint — the "below 8GB VRAM" flavour. The SFX prompt is not free-form noise:
it is derived from the scene's mood tag + the script's own imagery, and the house
rule is "ambience only, no music, no stingers", which is exactly the calm-narration
genre this studio makes.

The video clip is uploaded into ComfyUI's input dir (it doubles as conditioning
*and* timing reference), MMAudio runs, and the wav is pulled back and trimmed to
the clip length with fades so scene joins never click.

Fallback: :mod:`ai_studio.ambience` (numpy, deterministic) so a Machine-B draft
still has weather, birds and water under the voice.
"""
import os

from .. import ambience, workflows
from ..comfy import ComfyUIClient, ComfyError
from ..media import fit_audio
from ..util import ensure_dir, media_duration


def prompt_for(scene, cfg):
    from .. import style as style_mod

    sfx = (scene.get("sfx_prompt") or "").strip()
    if not sfx:
        sfx = style_mod.ambience_for(scene.get("mood_tag"), scene.get("visual_prompt"))
    sfx = sfx.strip().rstrip(".")
    return (sfx + ", no music, no melody, no stingers, no speech, natural field "
            "recording, quiet and gentle")[:400]


def render(video_path, out_wav, scene, cfg, target_duration, progress=None, seed=0):
    """Ambience for one scene: MMAudio if it's up, procedural numpy bed if not."""
    res = _mmaudio(video_path, out_wav, scene, cfg, target_duration, progress, seed)
    if res.get("ok"):
        return res
    proc = _procedural(out_wav, scene, cfg, target_duration, progress, seed)
    if proc.get("ok"):
        proc["fallback_from"] = "mmaudio" if res.get("attempted") else None
        proc["fallback_reason"] = str(res.get("reason", ""))[:200]
    else:
        proc["mmaudio_reason"] = str(res.get("reason", ""))[:200]
    return proc


def _mmaudio(video_path, out_wav, scene, cfg, target_duration, progress, seed):
    s = cfg.get("sfx", {})
    host = cfg.get("video", {}).get("comfy_host") or ""
    client = ComfyUIClient(host)
    out = {"engine": "mmaudio", "attempted": True}
    if not video_path or not os.path.exists(video_path):
        return {"ok": False, **out, "reason": "no video clip to condition on", "attempted": False}
    if not client.is_online():
        return {"ok": False, **out, "reason": f"ComfyUI not reachable at {host}", "attempted": False}
    try:
        wf_path, template, _ = workflows.resolve_workflow(s.get("workflow"), cfg)
    except Exception as e:
        return {"ok": False, **out, "reason": f"workflow: {e}"}
    dur = float(target_duration) + float(s.get("duration_pad_sec", 0.35))
    up = None
    try:
        up = client.upload_image(video_path, subfolder="ai_studio")
    except Exception as e:
        return {"ok": False, **out, "reason": f"could not upload video to ComfyUI: {str(e)[:150]}"}
    values = {
        "VIDEO_PATH": f"{up['subfolder']}/{up['name']}" if up.get("subfolder") else up["name"],
        "SFX_PROMPT": prompt_for(scene, cfg),
        "NEGATIVE": "music, melody, singing, speech, narration, stinger, whoosh, "
                    "impact, dramatic riser, loud, distortion",
        "DURATION": round(dur, 2),
        "STEPS": 20, "CFG": 6.0,
        "SEED": (abs(int(seed or 0)) * 104729 + 7) % 2**31,
        "SAMPLE_RATE": 44100,
        "OUT_PREFIX": "ai_studio/" + os.path.splitext(os.path.basename(out_wav))[0],
        "MOOD": scene.get("mood_tag") or "",
    }
    wf, report = workflows.render(template, values)
    if report.get("unresolved"):
        return {"ok": False, **out, "reason": "unresolved placeholders: "
                + ", ".join(report["unresolved"]) + f" (edit {wf_path})"}
    try:
        pid = client.queue_prompt(wf)
        if progress:
            progress(10, "MMAudio queued")
        outputs = client.wait(pid, timeout=int(s.get("timeout_sec", 900)),
                              on_progress=lambda pct, node: progress and progress(
                                  12 + pct * 0.8, node or "diffusing"))
    except ComfyError as e:
        return {"ok": False, **out, "reason": str(e)[:300]}
    refs = ComfyUIClient.output_refs(outputs, exts=(".wav", ".flac", ".mp3", ".ogg"))
    if not refs:
        return {"ok": False, **out, "reason": "MMAudio produced no audio file (check the "
                                             "workflow's SaveAudio node)"}
    try:
        tmp = out_wav + ".mmaudio.wav"
        ensure_dir(os.path.dirname(out_wav) or ".")
        client.download(refs[0], tmp)
        fit_audio(tmp, out_wav, max(0.5, float(target_duration)), mode="trim")
        os.remove(tmp)
    except Exception as e:
        return {"ok": False, **out, "reason": f"audio fetch failed: {str(e)[:150]}"}
    client.free_memory()
    return {"ok": True, "engine": "mmaudio", "path": out_wav, "duration": media_duration(out_wav, 0),
            "prompt": values["SFX_PROMPT"], "workflow": os.path.basename(wf_path),
            "prompt_id": pid, "layers": ["model-generated"], "sample_rate": 44100,
            "stereo": True}


def _procedural(out_wav, scene, cfg, target_duration, progress, seed):
    s = cfg.get("sfx", {})
    ensure_dir(os.path.dirname(out_wav) or ".")
    if progress:
        progress(40, "synthesising ambience (numpy)")
    try:
        info = ambience.synthesize(out_wav, duration=max(0.8, float(target_duration)),
                                   prompt_text=prompt_for(scene, cfg),
                                   mood_tag=scene.get("mood_tag") or "",
                                   seed=int(seed or 0), level=float(s.get("ambient_gain", 1.0)))
        info["ok"] = True
        info["prompt"] = prompt_for(scene, cfg)
        if progress:
            progress(100, "ambience ready")
        return info
    except Exception as e:
        return {"ok": False, "engine": "procedural",
                "reason": f"ambience synthesis failed: {str(e)[:200]}"}


def probe(cfg, plan=None):
    s = cfg.get("sfx", {})
    out = {"engine": (plan or {}).get("sfx", {}).get("engine"), "workflow": s.get("workflow"),
           "duck_gain": s.get("voice_duck_gain")}
    try:
        _p, wf, _ = workflows.resolve_workflow(s.get("workflow"), cfg)
        out["placeholders"] = workflows.template_placeholders(wf)
    except Exception as e:
        out["workflow_error"] = str(e)[:180]
    return out

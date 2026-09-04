"""NPC mode: Talking Head — one character image animated speaking the audio.

Integration: **SadTalker** (open-source, audio-driven single-image talking-face
animation). Inputs: one character expression image (already matched to the
scene's mood) + the scene's final voice wav. Output: a clip of that image
talking, with natural head motion.

Hardware note: SadTalker is the right fit for an RTX 5070 8GB + 16GB box — no
multi-subject identity tracking needed, single input image, runs locally. The
studio points at the SadTalker checkout's ``infer.py`` via
``talking_head.sadtalker_dir`` (see README-STUDIO.md for setup) and uses its
own Python venv when configured.

Fallback ("still"): the same image gets the existing Ken Burns motion so a
project with talking_head configured but SadTalker not installed still renders
— flagged ``real_talking_head: false`` in the manifest, never silent.
"""
import os
import shutil
import subprocess

from ..util import ensure_dir, media_duration


def resolve(cfg):
    """('sadtalker'|'still', reason) plan for the configured talking_head engine."""
    th = cfg.get("talking_head", {}) or {}
    want = str(th.get("engine") or "auto")
    exe = _sadtalker_exe(th)
    if want in ("sadtalker", "auto") and exe:
        return "sadtalker", "SadTalker found at " + exe
    if want == "sadtalker":
        return "still", "SadTalker configured but not found — using still fallback"
    return "still", "attempting SadTalker none/degraded — Ken Burns still"


def _sadtalker_exe(th):
    d = (th.get("sadtalker_dir") or os.environ.get("SADTALKER_DIR") or "").strip()
    if not d:
        return None
    python = (th.get("python") or os.environ.get("PYTHON") or "python").strip()
    for cand in (os.path.join(d, "infer.py"), os.path.join(d, "scripts", "inference.py")):
        if os.path.exists(cand):
            return {"python": python, "script": cand, "dir": d}
    return None


def render(image_path, wav_path, out_path, cfg, progress=None):
    """One talking-head clip. Returns dict {ok, engine, path, duration, ...}."""
    th = cfg.get("talking_head", {}) or {}
    plan, reason = resolve(cfg)
    ensure_dir(os.path.dirname(out_path) or ".")
    if not image_path or not os.path.exists(image_path):
        return {"ok": False, "engine": "talking_head",
                "reason": "no character image for this scene (character_id unset?)"}
    if plan == "sadtalker":
        res = _sadtalker(image_path, wav_path, out_path, cfg, progress)
        if res.get("ok"):
            return res
        return {"ok": False, "engine": "sadtalker", "reason": res.get("reason"),
                "fallback": "still"}
    # still fallback: Ken Burns on the matched image, length = the voice
    from .. import media

    try:
        dur = max(0.8, media_duration(wav_path, 0.0) or 0.0)
        media.make_silent_video_from_image(
            image_path, out_path, duration=dur,
            width=int(cfg.get("video", {}).get("width", 480)),
            height=int(cfg.get("video", {}).get("height", 854)),
            fps=min(24, int(cfg.get("video", {}).get("fps", 16)) + 4),
            motion="kenburns")
        return {"ok": True, "engine": "still", "path": out_path,
                "duration": media_duration(out_path, dur),
                "real_talking_head": False,
                "note": "SadTalker not installed — the character image is animated "
                        "with gentle motion but is not lip-synced. "
                        "Install SadTalker (README-STUDIO.md) and set "
                        "talking_head.sadtalker_dir to get real audio-driven lip sync.",
                "fallback_reason": reason}
    except Exception as e:
        return {"ok": False, "engine": "still", "reason": f"still render failed: {str(e)[:200]}"}


def _sadtalker(image_path, wav_path, out_path, cfg, progress):
    th = cfg.get("talking_head", {}) or {}
    exe = _sadtalker_exe(th)
    if not exe:
        return {"ok": False, "reason": "SadTalker checkout not configured"}
    if progress:
        progress(5.0, "SadTalker starting")
    out_dir = ensure_dir(os.path.join(os.path.dirname(out_path), ".sadtalker"))
    cmd = [exe["python"], exe["script"], "--driven_audio", wav_path,
           "--source_image", image_path, "--result_dir", out_dir,
           "--still", "--cpu" if str(th.get("device", "cuda")).startswith("cpu") else "--cuda"]
    try:
        r = subprocess.run(cmd, cwd=exe["dir"], capture_output=True, timeout=int(
            th.get("timeout_sec", 1800)), text=True, errors="replace")
    except Exception as e:
        return {"ok": False, "reason": f"SadTalker failed to run: {str(e)[:160]}"}
    # SadTalker writes result.mp4 (or *_result.mp4) into result_dir
    found = None
    for root, _dirs, files in os.walk(out_dir):
        for fn in files:
            if fn.lower().endswith(".mp4"):
                found = os.path.join(root, fn)
                break
        if found:
            break
    if not found:
        tail = (r.stderr or r.stdout or "")[-300:]
        return {"ok": False, "reason": f"SadTalker produced no mp4 ({tail[:200]})"}
    try:
        shutil.move(found, out_path)
    except Exception:
        import shutil as _s
        _s.copyfile(found, out_path)
    if progress:
        progress(95.0, "SadTalker done")
    return {"ok": True, "engine": "sadtalker", "path": out_path,
            "duration": media_duration(out_path, 0.0), "real_talking_head": True}


def probe(cfg):
    th = cfg.get("talking_head", {}) or {}
    plan, reason = resolve(cfg)
    return {"engine": th.get("engine", "auto"), "plan": plan, "reason": reason,
            "sadtalker_dir": th.get("sadtalker_dir"), "python": th.get("python"),
            "ready": plan == "sadtalker"}

"""Illustration stills — FLUX.2 klein 4B via ComfyUI, with a local fallback.

``visual_source: "illustration"`` needs a single still picture per scene (the
video stage then animates it with the existing Ken Burns helper — cheaper and
a better fit for static comparisons). The picture comes from, in order:

1. the Director's own upload (handled by the API, stored as ``00_custom.png``
   next to the scene — the video stage skips generation entirely then);
2. ComfyUI + a FLUX.2 klein text-to-image workflow (``flux2_klein_t2i_480p``,
   shipped as an API-format template; klein is the 4B Apache-2.0 model that
   fits alongside a loaded Wan checkpoint on an 8GB card only when run
   sequentially — the studio already serialises GPU jobs, and ComfyUI
   unloads models between prompts via the existing ``/free`` call);
3. a deterministic PIL gradient/mood still (no model at all) — flagged
   ``engine: pil`` so the manifest is honest.

If FLUX cannot be co-resident with Wan, the Director can set
``illustration.engine: pil`` to skip ComfyUI entirely and stay sequential.
"""
import os
import time

from ..comfy import ComfyUIClient, ComfyError
from ..util import ensure_dir
from .. import workflows


def generate_still(prompt, out_path, cfg, negative_prompt=None, progress=None,
                   seed=0, reference_image=None):
    """Render one still to ``out_path`` (png/jpg). Returns a dict, never raises."""
    ill = cfg.get("illustration", {}) or {}
    engine = str(ill.get("engine") or "auto")
    ensure_dir(os.path.dirname(out_path) or ".")
    if engine in ("auto", "comfyui"):
        res = _comfyui(prompt, out_path, cfg, negative_prompt, progress, seed)
        if res.get("ok"):
            return res
        if engine == "comfyui":
            return res                       # explicit choice: no silent fallback
    return _pil_fallback(prompt, out_path, cfg, negative_prompt, seed)


def _comfyui(prompt, out_path, cfg, negative_prompt, progress, seed):
    ill = cfg.get("illustration", {}) or {}
    host = ill.get("comfy_host") or (cfg.get("video", {}) or {}).get("comfy_host", "http://127.0.0.1:8188")
    client = ComfyUIClient(host)
    if not client.is_online():
        return {"ok": False, "engine": "comfyui",
                "reason": f"ComfyUI not reachable at {host} — falling back to local still"}
    try:
        wf_path, template, _t = workflows.resolve_workflow(
            ill.get("workflow") or "flux2_klein_t2i_480p", cfg, default="flux2_klein_t2i_480p")
    except Exception as e:
        return {"ok": False, "engine": "comfyui", "reason": f"workflow: {e}"}
    my_seed = int(ill.get("seed", -1))
    if my_seed == -1:
        my_seed = (abs(int(seed or 0)) * 7919 + int(time.time()) % 9973) % 2**31
    values = {
        "PROMPT": str(prompt or "")[:1200],
        "NEGATIVE": negative_prompt or ill.get("negative_prompt") or "",
        "WIDTH": int(ill.get("width", 854)),
        "HEIGHT": int(ill.get("height", 480)),
        "STEPS": int(ill.get("steps", 12)),
        "CFG": float(ill.get("cfg", 3.5)),
        "SEED": int(my_seed),
        "OUT_PREFIX": "ai_studio/" + os.path.splitext(os.path.basename(out_path))[0],
    }
    wf, report = workflows.render(template, values)
    if report.get("unresolved"):
        return {"ok": False, "engine": "comfyui",
                "reason": "unresolved placeholders: " + ", ".join(report["unresolved"])}
    try:
        pid = client.queue_prompt(wf)
    except ComfyError as e:
        return {"ok": False, "engine": "comfyui", "reason": str(e)[:200]}
    if progress:
        progress(8.0, "queued at ComfyUI (FLUX.2 klein)")
    try:
        outputs = client.wait(pid, timeout=int(ill.get("timeout_sec", 900)),
                              on_progress=lambda pct, node: progress and progress(
                                  8.0 + pct * 0.86, node or "sampling"))
    except ComfyError as e:
        return {"ok": False, "engine": "comfyui", "reason": str(e)[:200]}
    refs = ComfyUIClient.output_refs(outputs)
    imgs = [r for r in refs if r["filename"].lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
    if not imgs:
        return {"ok": False, "engine": "comfyui", "reason": "no image produced"}
    try:
        client.download(imgs[0], out_path)
    except Exception as e:
        return {"ok": False, "engine": "comfyui", "reason": f"download failed: {str(e)[:160]}"}
    client.free_memory()
    return {"ok": True, "engine": "comfyui-flux", "path": out_path,
            "prompt": values["PROMPT"], "seed": my_seed,
            "workflow": os.path.basename(wf_path), "prompt_id": pid}


def _pil_fallback(prompt, out_path, cfg, negative_prompt, seed):
    """Deterministic mood still: warm gradient + soft vignette + subtle texture.

    Never a plain black frame — with no model the Director still sees which
    scene is which, and the manifest says ``engine: pil``.
    """
    from .. import previz
    import numpy as np

    width = int((cfg.get("illustration", {}) or {}).get("width", 854))
    height = int((cfg.get("illustration", {}) or {}).get("height", 480))
    mood = (prompt or "").lower()
    palette = previz.palette_for(mood, prompt)
    rng = np.random.default_rng((abs(int(seed or 0)) * 7919 + 11) % (2 ** 31))
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, width, dtype=np.float32)[None, :]
    c0 = np.array(palette[0], dtype=np.float32)
    c1 = np.array(palette[1], dtype=np.float32)
    base = c0[None, None, :] * (1 - y[..., None] * 0.55) + c1[None, None, :] * (y[..., None] * 0.55)
    # soft diagonal light + gentle noise so it never reads as a flat swatch
    light = 0.10 * np.exp(-(((x - 0.62) ** 2 + (y - 0.3) ** 2) * 5.0))[..., None]
    texture = (rng.normal(0, 0.018, (height, width, 1)).astype(np.float32))
    frame = np.clip(base + light + texture, 0, 1)
    from ..util import write_wav  # noqa  (keep import graph light in real use)
    try:
        from PIL import Image
        img = Image.fromarray((frame * 255).astype("uint8"))
        img.save(out_path)
    except Exception:
        # pillow missing: let ffmpeg do a gradient frame
        from ..util import run_ffmpeg
        run_ffmpeg(["-f", "lavfi", "-i",
                    f"gradients=s={width}x{height}:c0=0x{'%02x%02x%02x' % tuple(int(v * 255) for v in palette[0])}"
                    f":c1=0x{'%02x%02x%02x' % tuple(int(v * 255) for v in palette[1])}:d=0.1",
                    "-frames:v", "1", out_path], timeout=300)
    return {"ok": True, "engine": "pil", "path": out_path,
            "prompt": (prompt or "")[:1200],
            "seed": int(seed or 0), "note": "no FLUX model reachable — deterministic mood still",
            "fallback_reason": "comfyui offline or workflow unavailable"}


def probe(cfg):
    ill = cfg.get("illustration", {}) or {}
    host = ill.get("comfy_host") or "http://127.0.0.1:8188"
    client = ComfyUIClient(host)
    online = client.is_online()
    out = {"host": host, "online": online, "engine": ill.get("engine", "auto"),
           "workflow": ill.get("workflow") or "flux2_klein_t2i_480p",
           "flux_node": None}
    if online:
        for cls in ("Flux2KleinModelLoader", "UNETLoader", "CLIPLoader", "VAELoader", "SaveImage"):
            try:
                if client.has_node(cls):
                    out["flux_node"] = cls
                    break
            except Exception:
                pass
    try:
        path, wf, _ = workflows.resolve_workflow(ill.get("workflow") or "flux2_klein_t2i_480p", cfg,
                                                 default="flux2_klein_t2i_480p")
        out["workflow_path"] = path
        out["placeholders"] = workflows.template_placeholders(wf)
    except Exception as e:
        out["workflow_error"] = str(e)[:180]
    return out


def find_custom_image(scene_dir):
    """The Director's own upload for a scene, if any (always wins over gen)."""
    cand = os.path.join(scene_dir, "00_custom.png")
    if os.path.exists(cand):
        return cand
    for ext in (".jpg", ".jpeg", ".webp"):
        p = os.path.join(scene_dir, "00_custom" + ext)
        if os.path.exists(p):
            return p
    return None

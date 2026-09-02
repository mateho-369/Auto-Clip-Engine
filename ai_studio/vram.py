"""VRAM safety — the 8GB budget, enforced in one place.

Machine A has exactly one RTX 5070 (8GB) and 16GB of system RAM. The three heavy
models (Ollama 8B, Wan, MMAudio) *cannot* be resident at the same time, so:

* the scheduler holds a single GPU lock (`vram.serialize_gpu`);
* each LLM stage ends with `keep_alive: 0` so Ollama unloads before ComfyUI loads
  (see :mod:`ai_studio.llm`);
* before a GPU job we ask ComfyUI `/system_stats` (authoritative — it owns the
  card) and fall back to `nvidia-smi`, then *downscale instead of failing*:
  fewer frames first, then a smaller frame size, matching the community rule of
  thumb for Wan on 8GB (480p · 17–81 frames · fp16/quantized · tiled VAE).
"""
from .config import nvidia_gpus
from .util import clamp


def free_vram_mb(cfg, comfy_client=None):
    """MB free on the GPU as far as we can tell (None = unknown)."""
    if comfy_client is not None:
        try:
            v = comfy_client.free_vram_mb()
            if v:
                return int(v)
        except Exception:
            pass
    gpus = nvidia_gpus()
    if gpus:
        return int(min(g.get("memory_free_mb", 0) or 0 for g in gpus))
    return None


COMFORT_MPXF_PER_8GB = 42.0      # mega-pixel-frames an 8GB card survives (offload + tiled VAE)


def guard_request(cfg, frames, width, height, free_mb=None):
    """Return (frames, width, height, notes) adjusted to fit the card.

    The unit is mega-pixel-frames (width × height × frames / 1e6) because that is
    what actually drives Wan's activation memory: 480x854 x 81 ≈ 33 Mpx·f (the tier
    community guides confirm fits 8GB with offloading), while 1280x720 x 81 ≈ 75
    Mpx·f wants 16GB+. So the budget is ~42 Mpx·f per 8GB, never dropping below the
    480p tier, and it only bites on 720p / 121-frame / "I pasted a 24GB tutorial
    config" requests. Nothing here is physics — it is a budget that errs
    conservative and always keeps the clip watchable (>= video.min_frames,
    >= ~352px on the short side).
    """
    v = cfg.get("video", {})
    vm = cfg.get("vram", {})
    notes = []
    frames = int(max(5, frames))
    width, height = int(width), int(height)
    limit = int(vm.get("limit_mb", 8192))
    reserve = int(vm.get("reserve_free_mb", 900))
    cap_frames = int(v.get("max_frames", 81))
    if frames > cap_frames:
        notes.append(f"frames {frames}→{cap_frames} (video.max_frames)")
        frames = cap_frames
    if free_mb is None:
        return frames, width, height, notes        # card unknown: only the hard cap applies

    usable = max(600.0, float(free_mb) - reserve)
    allowed = COMFORT_MPXF_PER_8GB * (usable / max(1024.0, limit)) * (limit / 8192.0)
    allowed = max(34.0, allowed)                   # never below the 480p comfort tier
    px_frames = (width * height * frames) / 1_000_000.0
    if px_frames <= allowed:
        return frames, width, height, notes
    if not vm.get("downscale_on_pressure", True):
        notes.append(f"VRAM: request is {px_frames:.0f} Mpx·f, card has room for {allowed:.0f} "
                     "(auto-reduce is off)")
        return frames, width, height, notes
    min_frames = int(v.get("min_frames", 17))
    want = int(allowed * 1e6 / max(1, width * height))
    want = max(min_frames, min(cap_frames, want - (want % 4 - 1)))
    if want < frames:
        notes.append(f"VRAM {free_mb}MB free → frames {frames}→{want}")
        frames = want
    px_frames = (width * height * frames) / 1e6
    if px_frames > allowed and min(width, height) > 352:
        k = 0.85
        while px_frames > allowed and min(int(width * k), int(height * k)) >= 352:
            width, height = int(width * k), int(height * k)
            width -= width % 2
            height -= height % 2
            px_frames = (width * height * frames) / 1e6
            k = max(0.55, k - 0.06)
        notes.append(f"VRAM pressure → frame size {width}x{height}")
    # Never hand back a frame much bigger than the house tier, even when the frame
    # budget would allow it: a 17-frame 1632x918 clip is not a usable short.
    base = int(v.get("width", 480)) * int(v.get("height", 854))
    cap_area = max(base, base * float(vm.get("max_area_ratio", 2.0)))
    if width * height > cap_area:
        k2 = (cap_area / float(width * height)) ** 0.5
        width, height = int(width * k2) - int(width * k2) % 2, int(height * k2) - int(height * k2) % 2
        notes.append(f"capped to {width}x{height} (house tier is {v.get('width')}x{v.get('height')})")
    return int(frames), int(width), int(height), notes


def frames_for(duration_sec, fps, min_frames=17, max_frames=81):
    """Wan wants 4n+1 frames; clamp to the safe window for the clip length."""
    d = float(clamp(duration_sec, 0.4, 600))
    f = int(round(d * float(fps)))
    f = int(clamp(f, int(min_frames), int(max_frames)))
    f = f - ((f - 1) % 4)               # 1, 5, 9, ... (VAE temporal stride 4 + 1)
    if f < int(min_frames):
        f = int(min_frames)
    return max(5, f)


def status(cfg, plan=None):
    """For /api/status: what the guard would do right now."""
    free = free_vram_mb(cfg)
    limit = int(cfg.get("vram", {}).get("limit_mb", 8192))
    out = {"free_mb": free, "limit_mb": limit, "reserve_mb": int(cfg.get("vram", {}).get("reserve_free_mb", 900)),
           "serialize_gpu": bool(cfg.get("vram", {}).get("serialize_gpu", True)),
           "pressure": None}
    if free is not None:
        out["pressure"] = "tight" if free < limit * 0.25 else "ok"
    if plan:
        f, w, h, notes = guard_request(cfg, frames_for(6.0, cfg["video"]["fps"],
                                                       cfg["video"]["min_frames"],
                                                       cfg["video"]["max_frames"]),
                                      cfg["video"]["width"], cfg["video"]["height"], free)
        out["plan_6s"] = {"frames": f, "width": w, "height": h, "notes": notes}
    return out

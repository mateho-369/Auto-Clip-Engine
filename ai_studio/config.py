"""Studio settings: role→model mapping, per-stage engines, VRAM safety.

Follows the ``ai_creator/team.py`` pattern (role → {enabled, model, temperature})
but for a *pipeline*: every stage also carries an engine choice, because the
heavy stages are not LLMs (sherpa-onnx, RVC, ComfyUI/Wan, MMAudio, ffmpeg).

Two hardware profiles are first-class citizens:

``machine_a``  RTX 5070 8GB · Ryzen 9 · 16GB RAM   → everything runs locally
``machine_b``  Ryzen 5 6600H + iGPU · 16GB RAM      → CPU only: no NVIDIA/CUDA

On ``machine_b`` the GPU stages (video, MMAudio SFX) default to ``defer``: they
are recorded as queued-for-machine-A instead of pretending to run, and the rest
of the pipeline (breakdown, Khmer voice, QA, assembly from previz) still works.

All numbers here were chosen to stay inside 8GB VRAM: quantized/small model
variants, 480p, standard step counts, one GPU job at a time.
"""
import copy
import json
import os
import shutil
import subprocess

from . import style as style_mod
from .util import clamp, read_json, write_json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MACHINE_PROFILES = {
    "auto": {"label": "Auto-detect", "desc": "Probe for an NVIDIA GPU and pick the matching profile.",
             "vram_limit_mb": 0},
    "machine_a": {"label": "Machine A — RTX 5070 8GB",
                  "desc": "Full pipeline: Wan video + MMAudio + RVC on CUDA.",
                  "vram_limit_mb": 8192},
    "machine_b": {"label": "Machine B — Ryzen 5 6600H (CPU only)",
                  "desc": "No NVIDIA GPU: video/SFX deferred (or cheap CPU previz), 3B LLM, 3a/3b voice still local.",
                  "vram_limit_mb": 0},
}

LLM_ROLES = ["controller", "auto_idea", "qa"]
LLM_ROLE_LABELS = {
    "controller": "Controller / Scene Breakdown (Stage 1)",
    "auto_idea": "Auto-Idea Generator (Stage 2 · Mode B)",
    "qa": "QA Reviewer (Stage 6)",
}

DEFAULTS = {
    "version": 1,
    "machine": {
        "profile": "auto",            # auto | machine_a | machine_b
        "force_cpu_only": False,       # manual override of detection
        "gpus": [],                      # filled by detect()
        "vram_total_mb": 0,
    },
    "ollama": {
        "host": "http://127.0.0.1:11434",
        "request_timeout_sec": 300,
        "keep_alive": "0",              # "0" = unload after each call → frees VRAM between stages
        "num_ctx": 4096,
        "roles": {
            "controller": {"enabled": True, "model": "sailor2:8b", "fallback_model": "llama3.2:3b",
                           "temperature": 0.55},
            "auto_idea": {"enabled": True, "model": "sailor2:8b", "fallback_model": "llama3.2:3b",
                          "temperature": 0.85},
            "qa": {"enabled": True, "model": "sailor2:8b", "fallback_model": "llama3.2:3b",
                   "temperature": 0.2},
        },
    },
    "tts": {                                        # Stage 3a
        "engine": "auto",                           # auto | sherpa | piper | kokoro | placeholder
        "model_dir": "models/tts/vits-mms-khm",     # sherpa-onnx VITS dir (model.onnx + tokens.txt)
        "sherpa_cli": "",                           # path to sherpa-onnx-offline-tts (optional)
        "language": "km",
        "speed": 1.0,                               # calm narration: <= 1.0 recommended
        "sample_rate": 0,                           # 0 = take it from the model
        "speaker_id": 0,
        "crossfade_ms": 30,
        "allow_nonnative_fallback": False,          # Kokoro (en) may speak Khmer text? never by default
    },
    "rvc": {                                        # Stage 3b
        "engine": "auto",                           # auto | http | cli | bypass
        "enabled": True,
        "api_base": "http://127.0.0.1:9513",
        "api_endpoint": "/sync",                    # Applio / RVC-WebUI inference endpoint
        "api_upload_field": "audio_file",
        "api_fields": {"rvc_model": "{model_name}", "pitch": "{pitch}",
                       "index_rate": "{index_rate}", "rms_mix_rate": "{rms_mix_rate}",
                       "f0_method": "{f0_method}", "clean_audio": "False",
                       "export_format": "WAV", "train_id": "0"},
        "api_result_mode": "auto",                   # auto | audio | json
        "webui_dir": "",                            # RVC-WebUI install dir
        # matches the RVC-Project mainline's offline CLI (infer/cli.py) — what
        # README-STUDIO.md actually has users install. {python} resolves to
        # that install's own venv interpreter, not this process's.
        "cli_template": ["{python}", "-m", "infer.cli", "--model", "{pth}",
                         "--input", "{input}", "--output", "{output}",
                         "--pitch", "{pitch}", "--f0-method", "{f0_method}",
                         "--index", "{index}", "--index-rate", "{index_rate}",
                         "--overwrite"],
        "models_dir": "models/rvc",                 # scanned for *.pth + assets/*.index
        "profile_id": "",                           # chosen voice profile (multi-profile support)
        "pitch": 0,
        "index_rate": 0.75,
        "rms_mix_rate": 0.25,
        "f0_method": "rmvpe",
        "crepe_hop_length": 128,
        "formant_shift": 0.0,
        "clean": False,
        "device": "cuda:0",
        "threads": 4,
        "fp16": True,
        "timeout_sec": 900,
        "train_command": "",                         # optional: launch training from the UI
    },
    "video": {                                      # Stage 4
        "engine": "auto",                           # auto | comfyui | previz | defer | off
        "comfy_host": "http://127.0.0.1:8188",
        "workflow": "wan2.1_t2v_1.3b_480p",         # file name in workflows/ (or absolute path)
        "width": 480,
        "height": 854,
        "fps": 16,
        "steps": 20,
        "cfg": 6.0,
        "shift": 8.0,
        "sampler": "euler",
        "scheduler": "simple",
        "max_frames": 81,                           # hard cap: 81 @480p is the safe 8GB window
        "min_frames": 17,
        "frames_per_sec_budget": 16,
        "motion_strength": 0.75,
        "negative_prompt": style_mod.DEFAULT_NEGATIVE,
        "style_tail": "calm documentary look, soft natural light, gentle slow camera drift, "
                      "muted warm palette, peaceful natural scenery, film-like, no text, no captions",
        # ^ appended to every scene's visual prompt after the Controller's own
        # text — override per-project for a different visual style (e.g. a
        # stick-figure/line-art look instead of the nature-documentary default).
        "seed": -1,
        "timeout_sec": 2400,
        "upload_start_frame": True,                 # TI2V: send a reference image when available
    },
    "sfx": {                                        # Stage 5
        "engine": "auto",                           # auto | mmaudio | procedural | defer | off
        "workflow": "mmaudio_small_480p",
        "duration_pad_sec": 0.35,
        "voice_duck_gain": 0.32,                    # ambience sits *under* narration
        "ambient_gain": 1.0,
        "timeout_sec": 900,
    },
    "vram": {
        "limit_mb": 8192,
        "reserve_free_mb": 900,                     # refuse a GPU job if free VRAM < this
        "serialize_gpu": True,                      # one GPU model resident at a time (8GB!)
        "unload_llm_after_stage": True,
        "downscale_on_pressure": True,              # halve frames/res before failing
        "max_scene_seconds_for_model": 14,          # longer than this → split into 2 clips
    },
    "pipeline": {
        "scene_target_seconds": style_mod.SCENE_TARGET_SECONDS,
        "scene_min_seconds": style_mod.SCENE_MIN_SECONDS,
        "scene_max_seconds": style_mod.SCENE_MAX_SECONDS,
        "max_scenes": 12,
        "review_gate": "auto",                      # auto|always|never — Mode B script approval
        "auto_approve_mode_b": False,               # "run fully autonomously"
        "require_qa_pass": False,                   # block assembly on QA fail?
        "retry_limit": 1,                           # one automatic retry per stage
        "retry_backoff_sec": 3,
        "duration_tolerance_sec": 0.9,              # QA: voice vs video mismatch
        "pace_calm": 1.15,                          # >1 = slower narration estimate
        "concurrency": {"llm": 1, "tts": 1, "gpu": 1, "cpu": 2, "io": 4},
        "keep_intermediate": True,
    },
    "assembly": {
        "fps": 24,
        "crf": 23,
        "preset": "veryfast",
        "video_codec": "libx264",
        "audio_kbps": 160,
        "loudnorm_target_lufs": -16.0,
        "emit_srt": True,
        "emit_manifest": True,
        "fade_sec": 0.35,
        "transition": "crossfade",                  # crossfade | cut
        "burn_captions": False,
    },
    # "" = resolve at runtime: STUDIO_DATA_DIR, else <repo>/data/studio.
    # Kept empty on purpose so a custom --data-dir also moves model/voice lookups.
    "paths": {"data_dir": ""},
    "ui": {"theme": "dark", "items_per_page": 24},
}


# --------------------------------------------------------------- deep merge
def _deep_merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        elif v is not None:
            out[k] = v
    return out


def _coerce(cfg):
    """Repair values coming from the settings UI (strings, NaN, wild ranges)."""
    c = copy.deepcopy(cfg)
    c["tts"]["speed"] = clamp(c["tts"].get("speed", 1.0), 0.5, 2.0)
    c["pipeline"]["scene_target_seconds"] = clamp(c["pipeline"].get("scene_target_seconds"), 2.5, 20.0)
    c["pipeline"]["scene_min_seconds"] = min(c["pipeline"]["scene_min_seconds"],
                                             c["pipeline"]["scene_target_seconds"])
    c["pipeline"]["scene_max_seconds"] = max(c["pipeline"]["scene_max_seconds"],
                                             c["pipeline"]["scene_target_seconds"])
    c["pipeline"]["max_scenes"] = int(clamp(c["pipeline"].get("max_scenes"), 1, 40))
    c["pipeline"]["retry_limit"] = int(clamp(c["pipeline"].get("retry_limit"), 0, 3))
    c["video"]["width"] = int(clamp(c["video"].get("width"), 256, 1280))
    c["video"]["height"] = int(clamp(c["video"].get("height"), 256, 1280))
    c["video"]["fps"] = int(clamp(c["video"].get("fps"), 8, 30))
    c["video"]["steps"] = int(clamp(c["video"].get("steps"), 4, 40))
    c["video"]["cfg"] = clamp(c["video"].get("cfg"), 1.0, 15.0)
    c["video"]["max_frames"] = int(clamp(c["video"].get("max_frames"), 17, 121))
    c["video"]["min_frames"] = int(min(c["video"]["min_frames"], c["video"]["max_frames"]))
    # 8GB hard limit — refuse to *ever* be configured into an OOM machine
    if int(c["video"]["width"]) * int(c["video"]["height"]) > 1280 * 720:
        c["video"]["width"], c["video"]["height"] = 480, 854
    c["sfx"]["voice_duck_gain"] = clamp(c["sfx"].get("voice_duck_gain"), 0.0, 1.0)
    if not isinstance(c["rvc"].get("api_fields"), dict):
        c["rvc"]["api_fields"] = dict(DEFAULTS["rvc"]["api_fields"])
    c["rvc"]["timeout_sec"] = int(clamp(c["rvc"].get("timeout_sec"), 60, 3600))
    c["vram"]["limit_mb"] = int(clamp(c["vram"].get("limit_mb"), 1024, 49152))
    # Machine A is an RTX 5070 with 8GB. A stale/guest "16308 MB total" reading must
    # not become permission to allocate 16 GB, so the profile's cap wins whenever it
    # is the lower number, unless the user picked 'auto'/'machine_a'-agnostic settings.
    prof = MACHINE_PROFILES.get(c.get("machine", {}).get("profile") or "auto", {})
    prof_cap = int(prof.get("vram_limit_mb") or 0)
    detected = int(c.get("machine", {}).get("vram_total_mb") or 0)
    if prof_cap and c["vram"]["limit_mb"] > prof_cap:
        c["vram"]["limit_mb"] = prof_cap
    elif detected and detected >= 1024 and c["vram"]["limit_mb"] > detected:
        c["vram"]["limit_mb"] = detected
    c["vram"]["reserve_free_mb"] = int(clamp(c["vram"].get("reserve_free_mb"), 0, 8192))
    for role in LLM_ROLES:
        r = c["ollama"]["roles"].setdefault(role, {})
        r["enabled"] = bool(r.get("enabled", True))
        r["model"] = str(r.get("model") or "sailor2:8b").strip() or "sailor2:8b"
        r["fallback_model"] = str(r.get("fallback_model") or "llama3.2:3b").strip() or "llama3.2:3b"
        r["temperature"] = clamp(r.get("temperature", 0.6), 0.0, 2.0)
    for k in c["pipeline"]["concurrency"]:
        c["pipeline"]["concurrency"][k] = int(clamp(c["pipeline"]["concurrency"][k], 1, 8))
    for engine_key, allowed in (
        (("video", "engine"), ("auto", "comfyui", "previz", "defer", "off")),
        (("sfx", "engine"), ("auto", "mmaudio", "procedural", "defer", "off")),
        (("tts", "engine"), ("auto", "sherpa", "piper", "kokoro", "placeholder")),
        (("rvc", "engine"), ("auto", "http", "cli", "bypass")),
    ):
        section, key = engine_key
        if str(c[section][key]) not in allowed:
            c[section][key] = "auto"
    c["pipeline"]["review_gate"] = str(c["pipeline"].get("review_gate") or "auto")
    if c["pipeline"]["review_gate"] not in ("auto", "always", "never"):
        c["pipeline"]["review_gate"] = "auto"
    return c


def default_config():
    return _coerce(copy.deepcopy(DEFAULTS))


def normalize_config(cfg):
    return _coerce(_deep_merge(DEFAULTS, cfg or {}))


# ------------------------------------------------------------- hardware probe
def nvidia_gpus():
    """[{name, memory_total_mb, memory_free_mb}] via nvidia-smi; [] if none."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        res = subprocess.run(
            [exe, "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, timeout=15)
        out = (res.stdout or b"").decode(errors="ignore").strip()
        gpus = []
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                try:
                    gpus.append({"name": parts[0], "memory_total_mb": int(float(parts[1])),
                                 "memory_free_mb": int(float(parts[2]))})
                except Exception:
                    continue
        return gpus
    except Exception:
        return []


def detect_machine(cfg=None):
    """Resolve `machine.profile` → effective {profile, cpu_only, gpus, vram_*}."""
    cfg = cfg or default_config()
    gpus = nvidia_gpus()
    want = str(cfg.get("machine", {}).get("profile") or "auto").lower()
    forced_cpu = bool(cfg.get("machine", {}).get("force_cpu_only"))
    if want in ("machine_a", "machine_b"):
        profile = want
    else:
        profile = "machine_a" if (gpus and not forced_cpu) else "machine_b"
    if forced_cpu:
        profile = "machine_b"
    total = sum(g.get("memory_total_mb", 0) for g in gpus) if profile == "machine_a" else 0
    return {"profile": profile, "cpu_only": profile == "machine_b", "gpus": gpus,
            "vram_total_mb": total, "vram_free_mb": sum(g.get("memory_free_mb", 0) for g in gpus),
            "cpu_cores": os.cpu_count() or 0, "ram_total_gb": _total_ram_gb()}


def _total_ram_gb():
    """System RAM in GB (0.0 when we cannot tell) — used for honest warnings only."""
    try:
        if os.name == "nt":
            import ctypes

            class _MSX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            msx = _MSX()
            msx.dwLength = ctypes.sizeof(_MSX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(msx)):
                return round(msx.ullTotalPhys / 1024 ** 3, 1)
            return 0.0
        if hasattr(os, "sysconf") and hasattr(os, "SC_PHYS_PAGES"):
            return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024 ** 3, 1)
    except Exception:
        pass
    return 0.0


def machine_defaults(profile):
    """What the profile implies for engine choices (user settings override these)."""
    if profile == "machine_b":
        return {
            "video": {"engine": "defer"},
            "sfx": {"engine": "defer"},
            "rvc": {"engine": "auto", "device": "cpu", "fp16": False},
            "ollama": {"roles": {r: {"model": "llama3.2:3b"} for r in LLM_ROLES}},
            "vram": {"serialize_gpu": False, "reserve_free_mb": 0},
        }
    return {
        "video": {"engine": "auto"},
        "sfx": {"engine": "auto"},
        "rvc": {"engine": "auto", "device": "cuda:0", "fp16": True},
        "ollama": {"roles": {r: {"model": "sailor2:8b"} for r in LLM_ROLES}},
    }


def resolve(cfg):
    """Fold hardware profile + availability into the effective engine choices.

    Returns (cfg, plan). ``plan`` is what the scheduler and the UI read:
    per-stage {engine, reason} plus capability flags. No stage is ever left
    ambiguous: 'auto' always resolves to a concrete engine or 'defer', and every
    choice carries a human reason so the settings page can explain itself.
    """
    raw = cfg or {}
    user_choice = {sec: str(((raw.get(sec) or {}) or {}).get("engine") or "").strip()
                   for sec in ("tts", "rvc", "video", "sfx")}
    cfg = normalize_config(cfg)
    hw = detect_machine(cfg)
    profile_default = {sec: str((((machine_defaults(hw["profile"]) or {}).get(sec) or {})
                                .get("engine")) or "") for sec in ("tts", "rvc", "video", "sfx")}
    cfg = _deep_merge(cfg, machine_defaults(hw["profile"]))
    for _sec in ("tts", "rvc", "video", "sfx"):
        # a profile default must never overrule an explicit Director choice
        if user_choice.get(_sec) and user_choice[_sec] != "auto":
            cfg[_sec]["engine"] = user_choice[_sec]
    cfg = _coerce(cfg)
    caps = capabilities(cfg)
    plan = {}

    def pick(stage, allowed, order, mapping):
        want = str(cfg[stage].get("engine") or "auto")
        explicit = bool(user_choice.get(stage)) and user_choice[stage] != "auto"
        profile_forced = (not explicit) and want == profile_default.get(stage)
        if want != "auto" and want in allowed:
            # 'defer' = the job runs but only records the hand-off to Machine A, so it
            # still appears in the stepper; 'off' skips it entirely.
            plan[stage] = {"engine": want, "run": want not in ("off",),
                           "reason": ("user choice" if explicit else
                                      f"{hw['profile']} profile default" if profile_forced
                                      else "auto → " + want)}
            return
        for cand, cap in order:
            if cap is None or caps.get(cap, False):
                reason = f"auto → {cand} ({'detected' if cap else 'always available'})"
                if want != "auto":
                    reason = f"'{want}' unavailable here → {cand}"
                plan[stage] = {"engine": cand, "reason": reason, "run": True}
                return
        fallback = allowed[-1]
        plan[stage] = {"engine": fallback, "reason": f"no engine available → {fallback}",
                       "run": True}

    pick("tts", ["sherpa", "piper", "kokoro", "placeholder"],
         [("sherpa", "sherpa_tts"), ("piper", "piper"), ("kokoro", "kokoro"), ("placeholder", None)],
         {"sherpa": "sherpa_tts", "piper": "piper", "kokoro": "kokoro"})
    pick("rvc", ["http", "cli", "bypass"],
         [("http", "rvc_http"), ("cli", "rvc_cli"), ("bypass", None)],
         {"http": "rvc_http", "cli": "rvc_cli"})
    if not cfg["rvc"].get("enabled", True):
        plan["rvc"] = {"engine": "off", "reason": "timbre stage disabled in settings", "run": False}
    pick("video", ["comfyui", "previz", "defer"],
         [("comfyui", "comfyui"), ("previz", None), ("defer", None)], {"comfyui": "comfyui"})
    if hw["cpu_only"] and plan["video"]["engine"] == "comfyui":
        plan["video"] = {"engine": "defer", "run": False,
                         "reason": "CPU-only machine — model render queued for Machine A"}
    pick("sfx", ["mmaudio", "procedural", "defer"],
         [("mmaudio", "comfyui"), ("procedural", None), ("defer", None)], {"mmaudio": "comfyui"})
    if hw["cpu_only"] and plan["sfx"]["engine"] == "mmaudio":
        plan["sfx"] = {"engine": "defer", "run": False,
                       "reason": "CPU-only machine — MMAudio queued for Machine A"}

    plan["ollama"] = {"available": bool(caps.get("ollama")),
                      "reason": "online" if caps.get("ollama") else "Ollama offline — deterministic fallbacks"}
    plan["ffmpeg"] = {"available": bool(caps.get("ffmpeg"))}
    plan["hardware"] = hw
    plan["vram_pressure"] = bool(hw.get("vram_free_mb") and hw["vram_free_mb"]
                                 < cfg["vram"]["limit_mb"] - cfg["vram"]["reserve_free_mb"])
    return cfg, plan


def capabilities(cfg=None):
    """What is actually present on this machine (cached by callers, not here)."""
    cfg = normalize_config(cfg or DEFAULTS)
    root = data_root(cfg)
    out = {
        "ffmpeg": shutil.which("ffmpeg") is not None or _imageio_ffmpeg(),
        "ffprobe": shutil.which("ffprobe") is not None,
        "python": True,
    }
    # Ollama (lazy import: never blocks on network in tests)
    try:
        from .llm import ollama_online
        out["ollama"] = ollama_online(cfg["ollama"]["host"])
    except Exception:
        out["ollama"] = False
    # TTS: sherpa-onnx model dir (model.onnx + tokens.txt)
    tdir = _abspath(root, cfg["tts"]["model_dir"])
    out["sherpa_tts"] = _sherpa_model_ok(tdir)
    out["sherpa_python"] = _module_present("sherpa_onnx")
    out["sherpa_cli"] = bool(shutil.which("sherpa-onnx-offline-tts")
                             or (cfg["tts"].get("sherpa_cli") and os.path.exists(cfg["tts"]["sherpa_cli"])))
    out["piper"] = shutil.which("piper") is not None or _module_present("piper")
    out["kokoro"] = (os.path.exists(os.path.join(root, "kokoro-v0_19.onnx"))
                     and os.path.exists(os.path.join(root, "voices.bin")))
    # RVC
    out["rvc_cli"] = bool(_rvc_webui_dir(cfg))
    try:
        from .engines.rvc import http_reachable
        out["rvc_http"] = http_reachable(cfg)
    except Exception:
        out["rvc_http"] = False
    out["rvc_profiles"] = len(_scan_rvc_profiles(cfg, root))
    # ComfyUI
    try:
        from .comfy import ComfyUIClient
        out["comfyui"] = ComfyUIClient(cfg["video"]["comfy_host"]).is_online()
    except Exception:
        out["comfyui"] = False
    gpus = nvidia_gpus()
    out["nvidia"] = bool(gpus)
    out["vram_free_mb"] = sum(g.get("memory_free_mb", 0) for g in gpus)
    return out


# ------------------------------------------------------------------- helpers
def _imageio_ffmpeg():
    try:
        import imageio_ffmpeg
        return os.path.exists(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        return False


def _module_present(name):
    try:
        return __import__(name) is not None
    except Exception:
        return False


def _sherpa_model_ok(tdir):
    if not tdir or not os.path.isdir(tdir):
        return False
    return bool(find_model_onnx(tdir)) and os.path.exists(os.path.join(tdir, "tokens.txt"))


def find_model_onnx(tdir):
    """Locate the VITS model file inside a sherpa-onnx model dir (any name)."""
    if not tdir or not os.path.isdir(tdir):
        return None
    pref = [f for f in os.listdir(tdir) if f.endswith(".onnx") and "int8" not in f]
    other = [f for f in os.listdir(tdir) if f.endswith(".onnx")]
    for cand in (pref or other):
        return os.path.join(tdir, cand)
    return None


def _abspath(root, p):
    if not p:
        return ""
    p = os.path.expanduser(str(p))
    return p if os.path.isabs(p) else os.path.join(root, p)


def data_root(cfg=None):
    """All studio state lives here (DB, media, settings) — copy it to move everything.

    Order: an explicit `paths.data_dir` in settings, then $STUDIO_DATA_DIR, then
    <repo>/data/studio. Every caller — API, scheduler, TTS/RVC/ComfyUI engines —
    resolves through here, so a relocated data dir cannot half-apply.
    """
    cfg = cfg or {}
    d = (cfg.get("paths") or {}).get("data_dir") or ""
    if not d:
        d = os.environ.get("STUDIO_DATA_DIR") or "data/studio"
    d = os.path.expanduser(str(d))
    return os.path.abspath(d) if os.path.isabs(d) else os.path.join(ROOT, d)


def _has_rvc_cli(d):
    # "infer_cli.py" at root is an older fork's layout; the RVC-Project
    # mainline (what README-STUDIO.md actually has users install) ships its
    # offline CLI at infer/cli.py instead — accept either.
    return os.path.exists(os.path.join(d, "infer_cli.py")) or \
        os.path.exists(os.path.join(d, "infer", "cli.py"))


def _rvc_webui_dir(cfg):
    d = (cfg.get("rvc") or {}).get("webui_dir") or os.environ.get("RVC_WEBUI_DIR") or ""
    d = os.path.expanduser(d)
    if d and _has_rvc_cli(d):
        return d
    for guess in (os.path.join(ROOT, "RVC-WebUI"), os.path.expanduser("~/RVC-WebUI"),
                  os.path.expanduser("~/Retrieval-based-Voice-Conversion-WebUI")):
        if _has_rvc_cli(guess):
            return guess
    return None


def _scan_rvc_profiles(cfg, root=None):
    """*.pth in models/rvc (plus a matching assets/*.index) = a voice profile."""
    root = root or data_root(cfg)
    d = _abspath(root, (cfg.get("rvc") or {}).get("models_dir") or "models/rvc")
    found = []
    if not os.path.isdir(d):
        return found
    for fn in sorted(os.listdir(d)):
        if not fn.lower().endswith(".pth"):
            continue
        name = os.path.splitext(fn)[0]
        idx = None
        for cand in (os.path.join(d, "assets", f"{name}.index"),
                     os.path.join(d, f"{name}.index"),
                     os.path.join(d, "assets", name, f"added_{name}.index")):
            if os.path.exists(cand):
                idx = cand
                break
        if idx is None:  # RVC layout: logs/<name>/added_*.index
            for base in (os.path.join(d, "logs", name), os.path.join(_rvc_webui_dir(cfg) or "",
                                                                       "assets", "indices", name)):
                if os.path.isdir(base):
                    for f2 in os.listdir(base):
                        if f2.endswith(".index"):
                            idx = os.path.join(base, f2)
                            break
        found.append({"name": name, "pth": os.path.join(d, fn), "index": idx})
    return found


# ---------------------------------------------------------------- persistence
def load(path=None):
    path = path or os.path.join(data_root(), "settings.json")
    raw = read_json(path, None)
    cfg = normalize_config(raw) if raw else default_config()
    cfg["_config_path"] = path
    return cfg


def save(cfg, path=None):
    path = path or cfg.get("_config_path") or os.path.join(data_root(), "settings.json")
    cfg = normalize_config({k: v for k, v in cfg.items() if not k.startswith("_")})
    write_json(path, cfg)
    return path


# ------------------------------------------------- ai_creator team.py bridge
def sync_to_ai_creator_team(cfg, team_path=None):
    """Mirror the LLM roles into ``ai_creator/team_config.json``.

    The older studio reads that file, so assigning sailor2 here also makes the
    legacy "AI Team" page consistent — one brain, two front-ends.
    """
    team_path = team_path or os.path.join(ROOT, "team_config.json")
    roles = cfg.get("ollama", {}).get("roles", {})
    data = {
        "ollama_host": cfg.get("ollama", {}).get("host"),
        "controller": roles.get("controller", {}).get("model"),
        "roles": {
            "planner": {"enabled": True, "model": roles.get("controller", {}).get("model"),
                        "temperature": roles.get("controller", {}).get("temperature")},
            "scriptwriter": {"enabled": True, "model": roles.get("auto_idea", {}).get("model"),
                             "temperature": roles.get("auto_idea", {}).get("temperature")},
            "sfx_director": {"enabled": True, "model": roles.get("qa", {}).get("model"),
                             "temperature": 0.4},
            "animator": {"enabled": True, "model": roles.get("controller", {}).get("model"),
                         "temperature": 0.6},
            "qa": {"enabled": roles.get("qa", {}).get("enabled", True),
                   "model": roles.get("qa", {}).get("model"),
                   "temperature": roles.get("qa", {}).get("temperature")},
        },
    }
    try:
        write_json(team_path, data)
        return team_path
    except Exception:
        return None


def as_json(cfg):
    return json.dumps({k: v for k, v in cfg.items() if not k.startswith("_")},
                      ensure_ascii=False, indent=2)


# ------------------------------------------------------- public aliases (engines)
def scan_rvc_profiles(cfg, root=None):
    return _scan_rvc_profiles(cfg, root or data_root(cfg))


def rvc_webui_dir(cfg):
    return _rvc_webui_dir(cfg)


def sherpa_model_dir(cfg, root=None):
    return _abspath(root or data_root(cfg), (cfg.get("tts") or {}).get("model_dir") or "")

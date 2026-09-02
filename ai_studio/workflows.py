"""ComfyUI workflow templates + placeholder injection.

Hard-coding someone else's node graph is fragile (custom-node packs rename their
inputs every few months), so the studio never guesses node ids. Instead:

* we ship example **API-format** workflows in ``ai_studio/workflows/``;
* the app injects per-scene values by replacing ``{{PLACEHOLDER}}`` markers, so
  a user's own exported workflow only has to contain the markers where the script
  text, prompt, resolution, frame count, seed, video path or output name go;
* a run reports *exactly* which placeholders were left unresolved instead of
  silently rendering the wrong thing.

Placeholders may be the whole string value (replaced with a typed
int/float/bool/list) or embedded inside it (string-substituted).
"""
import copy
import json
import os
import re

from .util import read_json

PLACEHOLDER = re.compile(r"\{\{\s*([A-Z0-9_]+)\s*\}\}")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILTIN_DIR = os.path.join(ROOT, "ai_studio", "workflows")

# Every value the studio knows how to inject. Documented once, used in the UI
# help text and by `validate_template`.
KNOWN_PLACEHOLDERS = {
    "PROMPT": "positive visual prompt for this scene (English, ComfyUI-friendly)",
    "NEGATIVE": "negative prompt (no text/watermark/flicker…)",
    "WIDTH": "video width in px (int)",
    "HEIGHT": "video height in px (int)",
    "FRAMES": "total frame count (int) — duration × fps",
    "FPS": "frames per second (int)",
    "DURATION": "clip length in seconds (float)",
    "STEPS": "sampling steps (int, keep 15-25 on an 8GB card)",
    "CFG": "guidance scale (float)",
    "SHIFT": "ModelSamplingSD3 shift (float, Wan: 8.0)",
    "SEED": "rng seed (int, -1 = random per attempt)",
    "MOTION": "motion strength (float, 0..1)",
    "TEXT": "the scene's spoken Khmer line (for models that read text)",
    "MOOD": "mood tag from Stage 1 (e.g. sunrise-warm)",
    "SFX_PROMPT": "natural-ambience description for MMAudio",
    "VIDEO_PATH": "input video filename inside ComfyUI (uploaded/relative)",
    "AUDIO_PATH": "input audio filename inside ComfyUI",
    "OUT_PREFIX": "output filename prefix requested from ComfyUI",
    "START_IMAGE": "uploaded start-frame image name (TI2V only)",
    "SAMPLE_RATE": "audio sample rate (int)",
}


def search_dirs(cfg=None):
    dirs = []
    if cfg:
        from .config import data_root
        dirs.append(os.path.join(data_root(cfg), "workflows"))
    dirs.append(BUILTIN_DIR)
    return dirs


def resolve_workflow(name_or_path, cfg=None, default=None):
    """'wan2.1_t2v_1.3b_480p' | 'name.json' | absolute path → (path, workflow_dict).

    An empty ``name_or_path`` means "built-in default" (that's what the
    settings UI's blank dropdown option submits) — falls back to ``default``.
    """
    name_or_path = name_or_path or default
    cand = [name_or_path] if name_or_path and os.path.isabs(str(name_or_path)) else []
    base = str(name_or_path or "").strip()
    for d in search_dirs(cfg):
        for fn in (base, base + ".json"):
            cand.append(os.path.join(d, fn))
    for p in cand:
        if p and os.path.isfile(p):
            data = read_json(p, None)
            if isinstance(data, dict):
                return p, _api_format(data), p
    raise FileNotFoundError(
        f"workflow '{name_or_path}' not found. Looked in: {', '.join(search_dirs(cfg))}. "
        "Export your ComfyUI workflow in API format and drop it there.")


def _api_format(data):
    """Accept a raw UI export ({'nodes': [...]} with a 'workflow' key) or API format."""
    if isinstance(data.get("nodes"), list) and "prompt" not in data:
        # ComfyUI desktop exports sometimes wrap it; if it's a UI graph we can't
        # rebuild node ids reliably — tell the user instead of half-working.
        raise ValueError("This looks like a UI (not API-format) ComfyUI export. "
                         "In ComfyUI use 'Save (API Format)' / Settings → Enable Dev mode "
                         "→ 'Export (API)' and save that file.")
    if isinstance(data.get("prompt"), dict):
        return copy.deepcopy(data["prompt"])
    return copy.deepcopy(data)


def template_placeholders(workflow):
    found = set()

    def walk(node):
        if isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
        elif isinstance(node, str):
            for m in PLACEHOLDER.finditer(node):
                found.add(m.group(1))
    walk(workflow)
    return sorted(found)


def render(workflow, values):
    """Return (patched_workflow, {"used": [...], "unresolved": [...], "coerced": {...}})."""
    used, coerced = [], {}
    unresolved = set(PLACEHOLDER.findall(json.dumps(workflow, ensure_ascii=False)))
    if "seed" in values and values["seed"] is not None and "SEED" not in unresolved:
        values.pop("seed", None)      # template doesn't take a seed: don't claim we set one

    def coerce(v):
        if isinstance(v, bool):
            return v
        if isinstance(v, int):
            return int(v)
        if isinstance(v, float):
            return round(float(v), 4)
        return v

    def walk(node):
        if isinstance(node, dict):
            return {k: walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v) for v in node]
        if not isinstance(node, str):
            return node
        m = PLACEHOLDER.fullmatch(node.strip())
        if m and m.group(1) in values:
            key = m.group(1)
            used.append(key)
            unresolved.discard(key)
            val = values[key]
            coerced[key] = type(val).__name__
            return coerce(val)
        if "{{" not in node:
            return node

        def sub(match):
            key = match.group(1)
            if key in values:
                used.append(key)
                unresolved.discard(key)
                v = values[key]
                return str(round(float(v), 3)) if isinstance(v, float) else str(v)
            return match.group(0)
        return PLACEHOLDER.sub(sub, node)

    out = walk(workflow)
    return out, {"used": sorted(set(used)), "unresolved": sorted(unresolved), "coerced": coerced}


def missing_required(workflow, required=("PROMPT", "FRAMES")):
    """Sanity check before burning GPU minutes: does the template accept the essentials?"""
    have = set(template_placeholders(workflow))
    if "FRAMES" not in have and "DURATION" not in have:
        # some workflows set length via a `length` int node input; allow either
        return [r for r in required if r not in have and r != "FRAMES"] or ["FRAMES-or-DURATION"]
    return [r for r in required if r not in have]

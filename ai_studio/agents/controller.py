"""Stage 1 — Controller / Scene Breakdown.

The one job: turn a finished script into an ordered list of scenes, each with
`{text, visual_prompt, estimated_duration_sec, mood_tag, sfx_prompt}`.

Mode A guarantee, enforced in code rather than by asking nicely: the LLM is only
ever allowed to *annotate*. Whatever it returns as `text` is discarded and the
Director's original sentence is restored, then the scene texts are compared with
the input script (whitespace-insensitive). If the model dropped or invented
content the run says so and the deterministic segmentation wins. So "no agent may
rewrite the script" is a property of the system, not a prompt hoping.

With Ollama offline, `deterministic_breakdown()` (in pipeline.fallbacks) already
produced a good skeleton, and we simply keep it.
"""
import json

from .. import content as content_mod
from .. import khmer, style as style_mod
from ..llm import scenes_validator

_SYSTEM_BASE = (
    "You are the CONTROLLER (scene-breakdown director) of a local Khmer video studio.\n"
    + style_mod.STYLE_GUIDELINE
    + "\nTASK: you receive a FINAL script that is already split into numbered segments. "
    "You must NOT rewrite, paraphrase, shorten, translate or reorder any segment text — "
    "the text field must be echoed byte-for-byte. You only ADD production metadata:\n"
)

def controller_system(content_type="explainer"):
    """The controller system prompt for a specific content type."""
    ct = content_mod.normalize(content_type)
    return (
        _SYSTEM_BASE
        + "CONTENT TYPE INSTRUCTION:\n"
        + content_mod.instruction_block(ct)
        + "\n"
        "  visual_prompt: one English sentence describing a single calm shot for a text-to-video "
        "model (subject + environment + light + slow camera motion; no text, no faces close-up)\n"
        "  mood_tag: one lowercase slug from this list (or the closest): "
        + ", ".join(sorted(style_mod.MOOD_AMBIENCE.keys()))
        + "\n  sfx_prompt: a SHORT natural-ambience description (no music, no stingers)\n"
        "  side: for structured types (compare/word_nuance/choose/myth_vs_fact) which side "
        "this scene belongs to (e.g. 'A'/'B', 'myth'/'fact')\n"
        "  estimated_duration_sec: how long the segment takes to speak calmly (you are given an "
        "estimate; only adjust it if it is clearly wrong)\n"
        "Respond ONLY with valid JSON: "
        '{"scenes":[{"index":int,"text":string,"visual_prompt":string,"mood_tag":string,'
        '"sfx_prompt":string,"side":string,"estimated_duration_sec":number}]}'
    )


# Backward-compatible module constant for tools that import SYSTEM directly.
SYSTEM = controller_system("explainer")


async def break_down(llm, script, cfg, plan_scenes=None, content_type="explainer",
                     character_id="", project_id="", run_id=""):
    """Returns (scenes, meta). `scenes` is always renderable."""
    from ..pipeline.fallbacks import deterministic_breakdown, enforce_script_integrity

    skeleton = deterministic_breakdown(script, cfg, plan_scenes,
                                       content_type=content_type, character_id=character_id)
    meta = {"engine": "deterministic", "scenes": len(skeleton), "notes": []}
    if not skeleton:
        return [], meta

    if llm is None or not llm.enabled("controller"):
        meta["notes"].append("controller role off / no LLM — mechanical segmentation only")
        return skeleton, meta

    payload = {"script": khmer.clip_clusters(khmer.normalize(script), 9000),
               "content_type": content_mod.normalize(content_type),
               "segments": [{"index": i, "text": s["text"],
                             "side": (s.get("meta") or {}).get("side", ""),
                             "estimated_duration_sec": s["estimated_duration_sec"]}
                            for i, s in enumerate(skeleton)]}
    user = ("Break down this FINAL script. Echo each `text` exactly. JSON only.\n"
            + json.dumps(payload, ensure_ascii=False, indent=1))
    data, lmeta = await llm.ask_json("controller", "breakdown",
                                     controller_system(content_type), user,
                                     validate=scenes_validator(expected_count=len(skeleton)))
    if not data:
        meta["engine"] = "deterministic"
        meta["notes"].append(f"LLM breakdown unavailable ({lmeta.get('reason', 'no json')}) — "
                             "kept deterministic segmentation")
        return skeleton, meta

    tagged = data.get("scenes") or []
    merged = _merge_annotations(skeleton, tagged, meta)
    merged, integrity = enforce_script_integrity(merged, script, cfg)
    if not integrity["ok"]:
        meta["notes"].append("LLM altered the script wording — original text restored "
                             f"({integrity['detail']})")
        merged = skeleton
    meta.update({"engine": f"ollama:{lmeta.get('model', '')}", "model": lmeta.get("model"),
                 "latency_ms": lmeta.get("latency_ms"), "scenes": len(merged),
                 "integrity": integrity})
    return merged, meta


def _merge_annotations(skeleton, tagged, meta):
    """Adopt visual_prompt/mood/sfx from the model; ignore any text change."""
    by_index = {}
    for i, t in enumerate(tagged):
        if isinstance(t, dict):
            try:
                by_index[int(t.get("index", i))] = t
            except Exception:
                by_index[i] = t
    out = []
    for i, base in enumerate(skeleton):
        sc = dict(base)
        sc["meta"] = dict(base.get("meta") or {})     # Director's production flags live on
        ann = by_index.get(i) or {}
        vp = str(ann.get("visual_prompt") or "").strip()
        if vp and len(vp) > 8:
            sc["visual_prompt"] = vp[:600]
        mood = str(ann.get("mood_tag") or "").strip().lower().replace(" ", "-")
        if mood:
            sc["mood_tag"] = mood if mood in style_mod.MOOD_AMBIENCE else _nearest_mood(mood)
        sfx = str(ann.get("sfx_prompt") or "").strip()
        if sfx:
            sc["sfx_prompt"] = sfx[:300]
        side = str(ann.get("side") or "").strip()
        if side and len(side) <= 24:
            sc.setdefault("meta", {})["side"] = side
        try:
            est = float(ann.get("estimated_duration_sec") or 0)
            if 1.0 <= est <= 30.0 and abs(est - sc["estimated_duration_sec"]) > 0.6:
                sc["estimated_duration_sec"] = round(est, 2)   # model timing wins slightly
                sc.setdefault("meta", {})["timing_source"] = "controller"
        except Exception:
            pass
        out.append(sc)
    if len(by_index) < len(skeleton):
        meta["notes"].append(f"model returned {len(by_index)}/{len(skeleton)} annotations — "
                             "missing ones keep deterministic values")
    return out


def _nearest_mood(mood):
    """Keep any mood slug (the SFX mapper falls back to DEFAULT_AMBIENCE)."""
    slug = "".join(c for c in mood if c.isalnum() or c in "-_")[:28] or style_mod.DEFAULT_MOOD
    return slug

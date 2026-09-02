"""Stage 6 — QA Reviewer.

Two halves, always both attempted:

* **mechanical** (in :mod:`ai_studio.pipeline.fallbacks`): duration mismatch
  between the voice and the picture, silence gaps, clipping, missing assets,
  "this was rendered by a fallback engine" honesty flags;
* **editorial** (Ollama): is the line actually calm/warm/encouraging, does it
  shame anyone, does the visual prompt fight the words?

The result is stored as JSON per scene and drives the QA card in the UI. A
``fail`` does not kill the run: assembly still happens (a Director can watch the
draft), unless `pipeline.require_qa_pass` is on.
"""
import json

from .. import khmer, style as style_mod
from ..llm import qa_validator

SYSTEM = (
    "You are the QA REVIEWER of a local Khmer video studio.\n"
    + style_mod.STYLE_GUIDELINE
    + "\nReview ONE finished scene and report defects. Look for: wording that is harsh, "
    "shaming, preachy or alarming (not house voice); a visual prompt that contradicts the "
    "mood of the words; an absurd length for a single sentence; text that looks like notes/"
    "markdown instead of narration; unsafe claims (medical, financial, political).\n"
    "Be concise and only report real problems. Respond ONLY with valid JSON: "
    '{"approved": bool, "summary": string, "issues": [{"severity": "warn|fail", '
    '"issue": string}]}'
)


async def review_scene(llm, scene, asset_facts, cfg, scene_idx, run_id="", project_id=""):
    from ..pipeline.fallbacks import deterministic_qa

    mechanical = deterministic_qa(scene, asset_facts, cfg)
    out = {
        "scene_idx": int(scene_idx),
        "approved": mechanical["approved"],
        "issues": list(mechanical["issues"]),
        "checks": {k: v for k, v in (mechanical.get("checks") or {}).items()},
        "mechanical": {"issue_count": len(mechanical["issues"])},
        "facts": {k: v for k, v in asset_facts.items() if not k.startswith("_")},
        "engine": "deterministic",
    }
    if llm is None or not llm.enabled("qa"):
        out["notes"] = "QA role off / no LLM — mechanical checks only"
        return out

    payload = {
        "scene_index": int(scene_idx),
        "text": (scene.get("text") or "")[:1200],
        "language": "km",
        "visual_prompt": (scene.get("visual_prompt") or "")[:400],
        "mood_tag": scene.get("mood_tag") or "",
        "voice_duration_sec": asset_facts.get("voice", {}).get("duration") if isinstance(
            asset_facts.get("voice"), dict) else asset_facts.get("voice_duration"),
        "video_duration_sec": asset_facts.get("video_duration"),
        "ambience": asset_facts.get("ambient_layers") or asset_facts.get("ambient_engine"),
        "mechanical_flags": [i.get("issue") for i in mechanical["issues"]][:6],
    }
    data, meta = await llm.ask_json("qa", "qa", SYSTEM,
                                    "Review this scene. JSON only.\n"
                                    + json.dumps(payload, ensure_ascii=False, indent=1),
                                    scene_idx=int(scene_idx))
    if data:
        issues = data.get("issues") or []
        for it in issues:
            sev = str(it.get("severity") or "warn").lower()
            out["issues"].append({"severity": "fail" if sev == "fail" else "warn",
                                  "issue": str(it.get("issue") or "")[:280], "source": "llm"})
        if data.get("summary"):
            out["summary"] = str(data["summary"])[:400]
        if data.get("approved") is False:
            out["approved"] = False
        out["engine"] = f"ollama:{meta.get('model', '')}"
    else:
        out["notes"] = f"LLM review unavailable ({meta.get('reason', '')})"
    out["approved"] = out["approved"] and not any(i["severity"] == "fail" for i in out["issues"])
    return out

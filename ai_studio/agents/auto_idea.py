"""Stage 2 — Auto-Idea Generator (Mode B only).

Given a topic hint (sometimes just a mood word), produce a title + a complete
Khmer script in the house voice. Two hard rules:

1. The output must *be* Khmer: we check the script's Khmer ratio and reject
   English-heavy answers (small multilingual models love to slip into English),
   then retry once with a tightened instruction before falling back to the
   deterministic template writer.
2. The Director can approve / edit / regenerate before production: this agent
   never triggers the rest of the pipeline by itself.
"""
import json

from .. import content as content_mod, khmer, style as style_mod
from ..llm import script_validator

SYSTEM_BASE = (
    "You are the AUTO-IDEA writer of a local Khmer short-video studio.\n"
    + style_mod.STYLE_GUIDELINE
    + "\nWrite a COMPLETE narration script in Khmer for one short video. Requirements:\n"
    "- 5 to 8 lines, one sentence per line, separated by newlines (the editor splits "
    "scenes on these lines);\n"
    "- total spoken length close to the requested runtime (a calm Khmer line of 60-90 "
    "characters takes about 6 seconds);\n"
    "- the words must be Khmer script (អក្សរខ្មែរ). English only inside a technical term if "
    "unavoidable;\n"
    "- no headings, no bullet marks, no scene numbers, no camera notes, no emoji;\n"
    "Respond ONLY with valid JSON: {\"title\": string, \"logline\": string, \"script\": string}"
)


def _system(content_type):
    return SYSTEM_BASE + content_mod.content_type_prompt(content_type)


async def generate(llm, topic_hint, cfg, style_notes="", regenerate_note="", content_type="explainer"):
    """Returns {title, logline, script, engine, notes[]} — always usable."""
    notes = []
    topic = khmer.truncate_clusters(khmer.strip_emoji_and_marks(topic_hint or ""), 300)
    if not topic:
        topic = "ការមិនបោះបង់ចិត្ត ទោះថ្ងៃលំបាក"
        notes.append("no topic given — the controller picked 'not giving up'")
    if llm is None or not llm.enabled("auto_idea"):
        return _fallback(topic, cfg, "auto_idea role off / no LLM", content_type=content_type)

    payload = {
        "topic_hint": topic,
        "target_seconds": int(cfg.get("target_duration") or 30),
        "extra_style_notes": khmer.truncate_clusters(style_notes or "", 600),
        "director_note": khmer.truncate_clusters(regenerate_note or "", 400),
    }
    system = _system(content_type)
    user = ("Write the script now. JSON only.\n" + json.dumps(payload, ensure_ascii=False, indent=1))
    data, meta = await llm.ask("auto_idea", "script", system, user,
                              validate=script_validator(min_chars=30))
    if not data:
        return _fallback(topic, cfg, f"LLM unavailable ({meta.get('reason', 'no answer')})",
                         content_type=content_type)
    # normalize_block alone keeps whatever markdown the model added (it's the
    # same function Mode A uses for the Director's own pasted script, where
    # preserving exact formatting is the whole point — wrong tool here: this
    # is AI-authored text meant to be spoken/captioned, so **bold**, `code`,
    # #headings etc. must go. strip_emoji_and_marks does that but collapses
    # all whitespace including newlines, which would merge every scene line
    # into one — clean each line individually instead, to keep the one
    # line = one scene structure the rest of this function depends on.
    raw_script = khmer.normalize_block(data.get("script") or "")
    script = "\n".join(
        cleaned for ln in raw_script.split("\n")
        if (cleaned := khmer.strip_emoji_and_marks(ln))
    )
    ratio = _khmer_ratio(script)
    if ratio < 0.55:
        notes.append(f"first draft was only {int(ratio * 100)}% Khmer — retrying once")
        data2, meta2 = await llm.ask("auto_idea", "script",
                                     system + "\nIMPORTANT: write ONLY in Khmer script.", user,
                                     validate=script_validator(min_chars=30))
        cand = khmer.normalize_block((data2 or {}).get("script") or "")
        if cand and _khmer_ratio(cand) >= 0.55:
            script, data, meta = cand, data2, meta2
            notes.append("retry produced Khmer text")
        else:
            notes.append(f"retry still {int(_khmer_ratio(cand) * 100)}% Khmer — kept best effort")
            script = script or cand
    if not script:
        return _fallback(topic, cfg, "empty script from model", content_type=content_type)
    est = khmer.estimate_speech_seconds(script, calm=cfg.get("pipeline", {}).get("pace_calm", 1.15))
    want = float(cfg.get("target_duration") or 30)
    if want and est < want * 0.6:
        notes.append(f"script is short ({est:.0f}s vs {want:.0f}s requested) — "
                     "approve as is, or regenerate with a note like 'more scenes'")
    elif want and est > want * 1.6:
        notes.append(f"script runs long ({est:.0f}s vs {want:.0f}s requested)")
    return {
        "title": khmer.truncate_clusters(data.get("title") or khmer.title_from(script), 120),
        "logline": khmer.truncate_clusters(data.get("logline") or "", 300),
        "script": script,
        "engine": f"ollama:{meta.get('model', '')}",
        "model": meta.get("model"),
        "latency_ms": meta.get("latency_ms"),
        "notes": notes,
        "origin": "ai:ollama",
        "khmer_ratio": round(ratio, 3),
        "estimated_seconds": est,
        "content_type": content_type,
    }


def _fallback(topic, cfg, why, content_type="explainer"):
    from ..pipeline.fallbacks import template_script   # local: keeps module graph light
    out = template_script(topic, cfg, content_type=content_type)
    out["notes"] = [f"auto-writer fallback: {why} — deterministic template script used"]
    out["engine"] = "template"
    out["origin"] = "ai:template"
    out["khmer_ratio"] = round(_khmer_ratio(out["script"]), 3)
    out["estimated_seconds"] = khmer.estimate_speech_seconds(
        out["script"], calm=cfg.get("pipeline", {}).get("pace_calm", 1.15))
    return out


def _khmer_ratio(text):
    letters = [c for c in text or "" if not c.isspace()]
    if not letters:
        return 0.0
    km = sum(1 for c in letters if 0x1780 <= ord(c) <= 0x17FF)
    return km / len(letters)

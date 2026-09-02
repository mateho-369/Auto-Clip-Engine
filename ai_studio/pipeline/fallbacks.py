"""Deterministic brains of the studio — used whenever a model is unavailable.

Three things live here because they are *algorithms*, not opinions:

* `deterministic_breakdown` — Khmer-aware mechanical scene segmentation with a
  per-scene visual tag derived from the imagery table (Stage 1 with Ollama offline,
  and the skeleton the LLM is asked to annotate).
* `enforce_script_integrity` — the Mode-A no-rewrite proof.
* `template_script` — a house-style Khmer script generator used when Mode B has no
  LLM. It is formulaic on purpose: it must never fail, and the UI labels it.
"""
import hashlib
import random

from .. import khmer, style as style_mod


# --------------------------------------------------------------- Stage 1 core
def deterministic_breakdown(script, cfg, plan_scenes=None):
    """Split a finished script into scenes sized for calm narration.

    `plan_scenes` (when the Director edited the board) takes precedence: we keep
    their text/visual prompts and only fill the holes + re-estimate timing.
    """
    p = cfg.get("pipeline", {})
    target = float(p.get("scene_target_seconds", style_mod.SCENE_TARGET_SECONDS))
    lo = float(p.get("scene_min_seconds", style_mod.SCENE_MIN_SECONDS))
    hi = float(p.get("scene_max_seconds", style_mod.SCENE_MAX_SECONDS))
    max_chars = int(p.get("scene_max_chars", style_mod.SCENE_MAX_CHARS))
    calm = float(p.get("pace_calm", 1.0))
    limit = int(p.get("max_scenes", 12))

    if plan_scenes:
        scenes = []
        for s in plan_scenes:
            text = khmer.strip_emoji_and_marks(s.get("text") or "")
            if not text:
                continue
            est = float(s.get("estimated_duration_sec") or 0) or khmer.estimate_speech_seconds(text, calm=calm)
            visual = (s.get("visual_prompt") or "").strip()
            mood = (s.get("mood_tag") or "").strip()
            if not visual or not mood:
                v2, m2 = style_mod.imagery_for(text)
                visual = visual or v2
                mood = mood or m2
            scenes.append({"index": len(scenes), "text": text, "visual_prompt": visual,
                           "mood_tag": mood, "estimated_duration_sec": round(est, 2),
                           "sfx_prompt": s.get("sfx_prompt") or style_mod.ambience_for(mood, visual),
                           "source": s.get("source") or "director-board"})
        return scenes[:limit] if scenes else []

    sentences = khmer.split_sentences(script, max_chars=max_chars)
    sentences = [khmer.strip_emoji_and_marks(s) for s in sentences]
    sentences = [s for s in sentences if s]
    if not sentences:
        return []

    # Greedy packing: never break a sentence, keep scenes inside [lo, hi] seconds.
    scenes, cur, cur_dur = [], [], 0.0
    for s in sentences:
        d = khmer.estimate_speech_seconds(s, calm=calm)
        if cur and (cur_dur + d > target * 1.35 or cur_dur + d > hi):
            scenes.append(_make_scene(cur, cur_dur, cfg))
            cur, cur_dur = [], 0.0
        cur.append(s)
        cur_dur += d
        if cur_dur >= lo and cur_dur >= target * 0.9:
            scenes.append(_make_scene(cur, cur_dur, cfg))
            cur, cur_dur = [], 0.0
    if cur:
        if scenes and cur_dur < lo * 0.7 and scenes[-1]["estimated_duration_sec"] + cur_dur <= hi * 1.15:
            last = scenes[-1]                                   # avoid a 1.5s orphan tail
            last["text"] = khmer.join_sentences([last["text"]] + cur)
            last["estimated_duration_sec"] = round(last["estimated_duration_sec"] + cur_dur, 2)
            v, m = style_mod.imagery_for(last["text"])
            last["visual_prompt"], last["mood_tag"] = _blend_prompts(last["visual_prompt"], v), m
            last["sfx_prompt"] = style_mod.ambience_for(m, last["visual_prompt"])
        else:
            scenes.append(_make_scene(cur, cur_dur, cfg))

    if len(scenes) > limit:            # over budget: merge the smallest neighbours
        scenes = _merge_to_limit(scenes, limit, cfg)
    for i, sc in enumerate(scenes):
        sc["index"] = i
    return scenes


def _make_scene(sentences, dur, cfg):
    text = khmer.join_sentences(sentences)
    visual, mood = style_mod.imagery_for(text)
    return {
        "text": text,
        "visual_prompt": visual,
        "mood_tag": mood,
        "estimated_duration_sec": round(max(1.0, dur), 2),
        "sfx_prompt": style_mod.ambience_for(mood, visual),
        "sentence_count": len(sentences),
        "source": "mechanical",
    }


def _merge_to_limit(scenes, limit, cfg):
    while len(scenes) > limit and len(scenes) > 1:
        # merge the pair whose combined duration is smallest
        best, best_sum = 0, 1e9
        for i in range(len(scenes) - 1):
            s = scenes[i]["estimated_duration_sec"] + scenes[i + 1]["estimated_duration_sec"]
            if s < best_sum:
                best, best_sum = i, s
        a, b = scenes[best], scenes.pop(best + 1)
        a["text"] = khmer.join_sentences([a["text"], b["text"]])
        a["estimated_duration_sec"] = round(a["estimated_duration_sec"] + b["estimated_duration_sec"], 2)
        a["visual_prompt"] = _blend_prompts(a["visual_prompt"], b["visual_prompt"])
    return scenes


def _blend_prompts(a, b):
    a, b = (a or "").strip(), (b or "").strip()
    if not b or a == b:
        return a
    return f"{a}; then {b}" if a else b


# ------------------------------------------------- Mode A integrity contract
def enforce_script_integrity(scenes, script, cfg=None):
    """Prove no agent changed the words. Returns (scenes, report)."""
    joined = khmer.join_sentences([s.get("text", "") for s in scenes])
    ok = khmer.equal_text(joined, script)
    detail = ""
    if not ok:
        detail = (f"scene text has {khmer.char_len(joined)} chars vs script "
                  f"{khmer.char_len(script)} chars")
        # put the Director's words back, keeping the LLM's visual annotations
        orig = khmer.split_sentences(script, max_chars=None)
        flat = []
        for s in orig:
            flat.extend(khmer.split_sentences(s, max_chars=240) or [s])
        rebuilt = []
        cursor = 0
        for sc in scenes:
            take = max(1, len(khmer.split_sentences(sc.get("text", ""), max_chars=None) or [""]))
            chunk = flat[cursor:cursor + take]
            cursor += len(chunk)
            if chunk:
                sc = dict(sc)
                sc["text"] = khmer.join_sentences(chunk)
                sc["text_restored"] = True
            rebuilt.append(sc)
        if cursor < len(flat):                       # model dropped whole scenes: append leftovers
            for leftover in flat[cursor:]:
                v, m = style_mod.imagery_for(leftover)
                rebuilt.append({"text": leftover, "visual_prompt": v, "mood_tag": m,
                                "estimated_duration_sec": khmer.estimate_speech_seconds(leftover),
                                "sfx_prompt": style_mod.ambience_for(m, v), "text_restored": True})
        ok2 = khmer.equal_text(khmer.join_sentences([s["text"] for s in rebuilt]), script)
        return rebuilt, {"ok": ok2, "restored": True, "detail": detail,
                         "verified": ok2}
    return scenes, {"ok": True, "restored": False, "detail": "byte-identical wording"}


# ------------------------------------------------------ Mode B template script
OPENINGS = [
    "តោះ អង្គុយស្ងាត់ៗមួយស្របក់សិន។",
    "ថ្ងៃនេះ ខ្ញុំចង់និយាយអ្វីមួយតិចៗទៅកាន់អ្នក។",
    "បើអ្នកកំពុងពិបាកໃច្ចេះ សូមអានបន្តិចទៀត។",
    "មានរឿងមួយ ដែលអ្នកគួរនឹកឃើញវិញ។",
]
BODY_A = [
    "{topic} មិនមែនជាបញ្ហាទេ វាគ្រាន់តែជាដំណាក់កាលមួយ។",
    "យើងច្រើនតែវាស់តម្លៃខ្លួនឯង តាមលទ្ធផលនៃថ្ងៃតែមួយ។",
    "ការលំបាកនេះ មិនមែនជាសញ្ញាថាអ្នកខ្វះខាតទេ។",
    "គ្មាននរណា រីកចម្រើនដោយមិនធ្លាប់ដួលសักម្ដងទេ។",
]
BODY_B = [
    "គិតត្រឹមទឹកដែលហូរ វាមិនរត់ប៉ាន់នរណាទេ តែវាទាញផ្លូវរបស់វាឆ្លងកាត់ថ្ម។",
    "ដើមឈើមិនប្រកាន់ខ្លួននូវខ្យល់ប៉ុន្មានទេ តែវាបន្តលូតលាស់។",
    "ផ្កាមិនបើកព្រមគ្នាទេ តែវាបើកត្រឹមតែក្នុងរដូវរបស់វា។",
    "ព្រះអាទិត្យរះរាល់ព្រឹក ទោះយប់វែងប៉ុន្មានក៏ដោយ។",
]
STEP = [
    "ថ្ងៃនេះ សូមធ្វើមួយជំហានតូចប៉ុណ្ណោះ ជំហានតូចៗក៏ជាផ្លូវដែយ។",
    "អត់ទោសឱ្យខ្លួនឯងចំពោះកំហុសចាស់ រួចចាប់ផ្ដើមថ្មីដោយស្ងប់។",
    "ដកដង្ហើមវែងៗ រួចធ្វើអ្វីមួយ ទោះតិចតួចប៉ុណ្ណាក៏ដោយ។",
    "ដាក់ទូរស័ព្ទចុះ ធ្វើកិច្ចការមួយឱ្យចប់ រួចសរសើរខ្លួនឯង។",
]
CLOSING = [
    "អ្នកកំពុងធ្វើបានល្អជាងអ្វីដែលអ្នកគិត។",
    "សូមមេត្តាអត់ធ្មត់ជាមួយខ្លួនឯងបន្តិចទៀត។",
    "កុំបោះបង់។ ការព្យាយាមរបស់អ្នក មានន័យជានិច្ច។",
    "ថ្ងៃស្អែក នៅតែជាឱកាសមួយសម្រាប់អ្នក។",
]


def template_script(topic, cfg=None):
    """Formulaic but real Khmer — the 'never dead-end' Mode B writer."""
    topic = khmer.strip_emoji_and_marks(topic or "")[:80] or "ការមិនបោះបង់ចិត្ត"
    seed = int(hashlib.sha256((topic + str(len(topic))).encode("utf-8")).hexdigest()[:6], 16)
    rng = random.Random(seed)
    want_sec = float((cfg or {}).get("target_duration") or 30.0) or 30.0
    calm = float((cfg or {}).get("pipeline", {}).get("pace_calm", 1.15))
    lines = [rng.choice(OPENINGS), BODY_A[0].format(topic=topic), rng.choice(BODY_B),
             rng.choice(STEP), rng.choice(CLOSING)]
    script = "\n".join(lines)
    # length pass: add gentle middle beats until we approach the requested runtime
    extras = BODY_A[1:] + BODY_B + STEP
    tries = 0
    while (khmer.estimate_speech_seconds(script, calm=calm) < want_sec * 0.8
           and tries < 6):
        script = script.rstrip() + "\n" + rng.choice(extras)
        tries += 1
    title = topic[:70]
    return {"title": title, "script": script.strip(),
            "logline": f"សារថ្ងៃនេះ៖ {topic}", "engine": "template",
            "beat_count": len(lines) + tries}


# ------------------------------------------------------------------- QA core
def deterministic_qa(scene, assets, cfg):
    """Per-scene mechanical checks — these run even when the LLM reviewer is off."""
    issues = []
    tol = float(cfg.get("pipeline", {}).get("duration_tolerance_sec", 0.9))
    audio = assets.get("voice_final") or assets.get("voice")
    video = assets.get("video_fit") or assets.get("video")
    amb = assets.get("ambient")
    if not (scene.get("text") or "").strip():
        issues.append({"severity": "fail", "issue": "scene has no text to speak"})
    if audio is None:
        issues.append({"severity": "fail", "issue": "voice track missing"})
    if video is None:
        issues.append({"severity": "fail", "issue": "video clip missing"})
    if audio and video:
        drift = abs(float(audio.get("duration") or 0) - float(video.get("duration") or 0))
        if drift > tol:
            issues.append({"severity": "fail" if drift > tol * 2 else "warn",
                           "issue": f"voice/video length mismatch {drift:.2f}s "
                                     f"(voice {audio.get('duration'):.2f}s, "
                                     f"video {video.get('duration'):.2f}s)",
                           "kind": "duration_mismatch"})
    if audio:
        head, tail = assets.get("_head_silence", 0.0), assets.get("_tail_silence", 0.0)
        if head > 1.2:
            issues.append({"severity": "warn", "issue": f"{head:.1f}s of silence at the start",
                           "kind": "silence_gap"})
        if tail > 1.6:
            issues.append({"severity": "warn", "issue": f"{tail:.1f}s of silence at the end",
                           "kind": "silence_gap"})
        if float(audio.get("peak", 0)) > 0.985:
            issues.append({"severity": "warn", "issue": "voice is clipping (peak > -0.1 dBFS)"})
    if not assets.get("voice_engine_ok", True):
        issues.append({"severity": "warn",
                       "issue": f"voice came from the '{assets.get('voice_engine')}' placeholder "
                                "engine — install sherpa-onnx + vits-mms-khm for real Khmer speech",
                       "kind": "engine"})
    if assets.get("video_engine") == "previz":
        issues.append({"severity": "warn", "issue": "video is a CPU previz draft, not a Wan render",
                       "kind": "engine"})
    if amb and video:
        if float(amb.get("duration") or 0) + 0.5 < float(video.get("duration") or 0):
            issues.append({"severity": "warn", "issue": "ambience shorter than the picture"})
    est = float(scene.get("estimated_duration_sec") or 0)
    if est and audio:
        if float(audio.get("duration") or 0) < 0.6:
            issues.append({"severity": "fail", "issue": "voice clip is under 0.6s — nothing to watch"})
    return {"approved": not any(i["severity"] == "fail" for i in issues), "issues": issues}

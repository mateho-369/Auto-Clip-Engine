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

from .. import content as content_mod, khmer, style as style_mod


# --------------------------------------------------------------- Stage 1 core
def deterministic_breakdown(script, cfg, plan_scenes=None, content_type="explainer"):
    """Split a finished script into scenes sized for calm narration.

    `plan_scenes` (when the Director edited the board) takes precedence: we keep
    their text/visual prompts and only fill the holes + re-estimate timing.
    `content_type` shapes the per-scene metadata/side tags even when Ollama is off.
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
        return _content_tag_scenes(scenes[:limit], content_type) if scenes else []

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
    return _content_tag_scenes(scenes, content_type)


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


def _content_tag_scenes(scenes, content_type):
    """Attach content-type side/visual metadata to deterministic scenes.

    ``compare``/``word_nuance`` split on **sentence count** rather than pure scene
    index, so a script that genuinely has two halves keeps them balanced even if
    the mechanical packer merged or split differently.
    """
    ct = content_mod.normalize_content_type(content_type)
    total_sents = sum(int(s.get("sentence_count") or 1) for s in scenes or [])
    running = 0
    out = []
    for i, s in enumerate(scenes or []):
        sc = dict(s)
        meta = dict(sc.get("meta") or {})
        count = max(1, int(sc.get("sentence_count") or 1))
        running += count
        if ct == "compare":
            side = "A" if running <= total_sents / 2 else "B"
        elif ct == "word_nuance":
            side = "meaning-1" if running <= total_sents / 2 else "meaning-2"
        elif ct == "myth_vs_fact":
            side = "myth" if running <= total_sents / 2 else "fact"
        else:
            side = content_mod.scene_tag(ct, i, max(1, len(scenes or [])))
        meta["content_type"] = ct
        meta["content_side"] = side
        meta["scene_index"] = i
        meta["scene_count"] = len(scenes or [])
        sc["meta"] = meta
        tail = content_mod.visual_tail(ct)
        vp = (sc.get("visual_prompt") or "").strip()
        if tail and tail.lower() not in vp.lower():
            sc["visual_prompt"] = khmer.truncate_clusters(f"{vp}; {tail}" if vp else tail, 700)
        out.append(sc)
    return out


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


# Deterministic content-type script shapes (all Khmer, placeholder {topic}).
CT_SCRIPT_SHAPES = {
    "explainer": lambda t, r: [r.choice(OPENINGS), BODY_A[0].format(topic=t), r.choice(BODY_B),
                               r.choice(STEP), r.choice(CLOSING)],
    "what_if": lambda t, r: [
        f"ចុះបើយើងសាកម្តងផ្ទុយពីធម្មតា៖ {t}?",
        "រឿងនេះ មិនមែនជាការស្រមើស្រមៃទទេ វាគឺជាមធ្យោបាយមួយដើម្បីមើលឃើញបញ្ហាឡើងវិញ។",
        "បើយើងធ្វើដូចនោះសប្តាហ៍មួយ យើងអាចសង្កេតឃើញភាពខុសគ្នាដ៏ស្ងៀមស្ងាត់។",
        "ចុងក្រោយ សំណួរមិនមែនថាអ្វីត្រឹមត្រូវទេ ប៉ុន្តែថាតើការសង្កេតនោះបង្ហាញយើងនូវអ្វី។",
        r.choice(CLOSING),
    ],
    "compare": lambda t, r: [
        f"ថ្ងៃនេះ យើងប្រៀបធៀប {t} ជាពីរផ្នែក។",
        f"ផ្នែក A៖ {t} គឺសាមញ្ញ រហ័ស និងច្បាស់ភ្លាមៗ។",
        f"ផ្នែក B៖ {t} គឺយឺតជាង ប៉ុន្តែផ្តល់ភាពច្បាស់ក្នុងរយៈពេលយូរ។",
        "បើអ្នកត្រូវការលទ្ធផលភ្លាមៗ A សមជាង។ បើអ្នកចង់យល់ជាង A ជម្រើស B គឺសមជាង។",
        "ដូច្នេះ ជម្រើសមិនមែនជាសត្រូវទេ វាគ្រាន់តែជាការជ្រើសរើសតាមគោលដៅ។",
    ],
    "choose": lambda t, r: [
        f"ចង់សម្រេចចិត្តលើ {t}? យើងអាចមើលវាជាពីរជម្រើស។",
        "ជម្រើសទី១៖ លឿន ងាយ ប៉ុន្តែអាចមិនយូរអង្វែង។",
        "ជម្រើសទី២៖ ត្រូវការពេលច្រើន ប៉ុន្តែផ្តល់ភាពធូរស្រាលជាង។",
        "ប្រសិនបើអ្នកមានពេលតិច សូមជ្រើសជម្រើសទី១។ បើអ្នកមានពេលគ្រប់គ្រាន់ ជម្រើសទី២ គឺប្រសើរជាង។",
        "ភាគច្រើន ចម្លើយគឺអាស្រ័យលើពេលវេលា និងអាទិភាពរបស់អ្នក។",
    ],
    "word_nuance": lambda t, r: [
        f"ពាក្យ \"{t}\" អាចមានអត្ថន័យពីរផ្សេងគ្នា។",
        "អត្ថន័យទី១៖ វាមានន័យថា ... ឧទាហរណ៍៖ សព្វថ្ងៃនេះ ខ្ញុំយល់អ្វីមួយថ្មី។",
        "អត្ថន័យទី២៖ វាក៏អាចមានន័យថា ... ឧទាហរណ៍៖ អ្នកយល់ពីអារម្មណ៍របស់ខ្ញុំ។",
        "ដូច្នេះ មុននឹងប្រើពាក្យនេះ សូមមើលបរិបទឲ្យច្បាស់។",
        "នេះជាការខុសគ្នាតូច ប៉ុន្តែវាធ្វើឲ្យការសន្ទនារបស់អ្នកច្បាស់ជាង។",
    ],
    "myth_vs_fact": lambda t, r: [
        f"មានជំនឿថា {t} គឺពិបាកណាស់។",
        "មនុស្សច្រើនតែជឿដូចនេះ ព្រោះពួកគេឃើញតែការបរាជ័យខ្លះៗ។",
        "ប៉ុន្តែការពិតគឺ {t} គឺអាចធ្វើបាន ដោយចាប់ផ្តើមមួយជំហានតូច។",
        "ការបរាជ័យម្តង មិនមែនជាសញ្ញាថាវាមិនអាចទេ វាគ្រាន់តែជាផ្នែកនៃផ្លូវ។",
        "នៅពេលអ្នកដឹងការពិតនេះ អ្នកអាចធ្វើសកម្មភាពដោយស្ងប់ស្ងាត់ជាង។",
    ],
    "quick_tip": lambda t, r: [
        f"គន្លឹះរហ័សមួយ៖ {t} ។",
        "សូមចាប់ផ្តើមត្រឹមតែមួយជំហានតូច មុនពេលអ្នកធ្វើអ្វីផ្សេង។",
        "ធ្វើវានៅពេលនេះ ហើយមើលលទ្ធផលនៅថ្ងៃស្អែក។",
        r.choice(CLOSING),
    ],
}


def template_script(topic, cfg=None, content_type="explainer"):
    """Formulaic but real Khmer — the 'never dead-end' Mode B writer."""
    ct = content_mod.normalize_content_type(content_type)
    topic = khmer.truncate_clusters(khmer.strip_emoji_and_marks(topic or ""), 80) or "ការមិនបោះបង់ចិត្ត"
    seed = int(hashlib.sha256((topic + str(len(topic)) + ct).encode("utf-8")).hexdigest()[:6], 16)
    rng = random.Random(seed)
    want_sec = float((cfg or {}).get("target_duration") or content_mod.default_duration(ct)) \
        or content_mod.default_duration(ct)
    calm = float((cfg or {}).get("pipeline", {}).get("pace_calm", 1.15))
    lines = CT_SCRIPT_SHAPES.get(ct, CT_SCRIPT_SHAPES["explainer"])(topic, rng)
    script = "\n".join(lines)
    # length pass: add gentle middle beats until we approach the requested runtime
    extras = BODY_A[1:] + BODY_B + STEP
    tries = 0
    while (khmer.estimate_speech_seconds(script, calm=calm) < want_sec * 0.8
           and tries < 6):
        script = script.rstrip() + "\n" + rng.choice(extras)
        tries += 1
    title = khmer.truncate_clusters(topic, 70)
    return {"title": title, "script": script.strip(),
            "logline": f"សារថ្ងៃនេះ៖ {topic}", "engine": "template",
            "content_type": ct, "beat_count": len(lines) + tries}


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

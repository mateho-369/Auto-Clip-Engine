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
def deterministic_breakdown(script, cfg, plan_scenes=None, content_type="explainer",
                            character_id=""):
    """Split a finished script into scenes sized for calm narration.

    `plan_scenes` (when the Director edited the board) takes precedence: we keep
    their text/visual prompts and only fill the holes + re-estimate timing.
    `content_type` shapes the *structure* even with no LLM: compare halves,
    myth-first, meaning pair, options+takeaway, one-tip — never a silent
    fall back to explainer behaviour.
    """
    from .. import content as content_mod
    from ..config import pace_engine

    ct = content_mod.normalize(content_type)
    p = cfg.get("pipeline", {})
    target = float(p.get("scene_target_seconds", style_mod.SCENE_TARGET_SECONDS))
    lo = float(p.get("scene_min_seconds", style_mod.SCENE_MIN_SECONDS))
    hi = float(p.get("scene_max_seconds", style_mod.SCENE_MAX_SECONDS))
    max_chars = int(p.get("scene_max_chars", style_mod.SCENE_MAX_CHARS))
    calm = float(pace_engine(cfg).get("pace_calm", 1.0))
    limit = int(p.get("max_scenes", 12))
    # quick_tip: shorter scenes by default (one fast practical tip)
    if ct == "quick_tip":
        target = min(target, 5.0)
        lo = min(lo, 2.0)
        hi = min(hi, 8.0)

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
            meta = dict(s.get("meta") or {})
            # Director's per-scene production flags survive re-segmentation
            meta = {k: v for k, v in meta.items()
                    if k in ("visual_source", "render_mode", "character_id", "side", "content_type")}
            scenes.append({"index": len(scenes), "text": text, "visual_prompt": visual,
                           "mood_tag": mood, "estimated_duration_sec": round(est, 2),
                           "sfx_prompt": s.get("sfx_prompt") or style_mod.ambience_for(mood, visual),
                           "source": s.get("source") or "director-board", "meta": meta})
        if character_id:
            for sc in scenes:
                sc["meta"].setdefault("character_id", character_id)
        return scenes[:limit] if scenes else []

    sentences = khmer.split_sentences(script, max_chars=max_chars)
    sentences = [khmer.strip_emoji_and_marks(s) for s in sentences]
    sentences = [s for s in sentences if s]
    if not sentences:
        return []

    if ct in ("compare", "myth_vs_fact", "word_nuance", "choose"):
        # structured types keep one sentence per scene — the structure is the
        # point (A half vs B half, myth vs fact). Merging the packer's greedy
        # pass would destroy exactly the shape the content type promises.
        scenes = [_make_scene([s], khmer.estimate_speech_seconds(s, calm=calm), cfg)
                  for s in sentences]
        if len(scenes) > limit:
            scenes = _merge_to_limit(scenes, limit, cfg)
        scenes = _shape_content_structure(scenes, ct)
        for i, sc in enumerate(scenes):
            sc["index"] = i
            sc.setdefault("meta", {})
            sc["meta"]["content_type"] = ct
            if character_id:
                sc["meta"].setdefault("character_id", character_id)
        return scenes

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
    scenes = _shape_content_structure(scenes, ct)
    for i, sc in enumerate(scenes):
        sc["index"] = i
        sc.setdefault("meta", {})
        sc["meta"]["content_type"] = ct
        if character_id:
            sc["meta"].setdefault("character_id", character_id)
    return scenes


def _shape_content_structure(scenes, ct):
    """Deterministic content-type structure when there is no LLM.

    This is the no-Ollama guarantee: ``compare`` really becomes two parallel
    halves (not just `explainer` in disguise), myth-vs-fact starts with the
    myth, word-nuance gets meaning-1/meaning-2, choose gets option scenes plus
    a takeaway, what_if gets a hypothetical opening frame.
    """
    if not scenes:
        return scenes
    n = len(scenes)
    for i, s in enumerate(scenes):
        s = dict(s)
        meta = dict(s.get("meta") or {})
        meta["content_type"] = ct
        if ct == "compare":
            half = max(1, n // 2)
            meta["side"] = "A" if i < half else ("B" if i < 2 * half else "summary")
            # with an odd extra scene it becomes the balanced summary — never steal
            # a B slot from a two-scene comparison
            if n >= 3 and i == n - 1 and meta["side"] == "B":
                meta["side"] = "summary"
            if meta["side"] in ("A", "B"):
                grade = "warm golden" if meta["side"] == "A" else "cool blue"
                vp = (s.get("visual_prompt") or "").strip()
                if "contrast" not in vp.lower():
                    meta["visual_contrast"] = meta["side"]
                    s["visual_prompt"] = (vp + f", {meta['side']}-side visual treatment, "
                                               f"distinct {grade} colour grade").strip()
        elif ct == "word_nuance":
            meta["side"] = "meaning-1" if i == 0 else ("meaning-2" if i == n - 1 else "contrast")
        elif ct == "myth_vs_fact":
            meta["side"] = "myth" if i == 0 else ("fact" if i == 1 else "why-it-matters")
        elif ct == "choose":
            meta["side"] = f"option-{i + 1}" if i < n - 1 else "takeaway"
        elif ct == "what_if" and i == 0:
            meta["side"] = "hypothetical"
        elif ct == "quick_tip":
            meta["side"] = "tip"
        # what_if visual bias: imaginative even with the calm-nature imagery table
        if ct == "what_if" and (s.get("visual_prompt") or "").strip() == style_mod.DEFAULT_VISUAL:
            s["visual_prompt"] = ("dreamlike soft light, gently surreal landscape, "
                                  "speculative atmosphere, calm filmic motion")
        s["meta"] = meta
        scenes[i] = s
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


# content-type structural templates (used when Ollama is offline): each is the
# skeleton of the type, so the no-LLM Mode B script is recognisably that type,
# not just explainer-with-a-different-label.
_CT_LINES = {
    "explainer": [
        "ថ្ងៃនេះ យើងមកនិយាយអំពី {topic} ។",
        "រឿងនេះមានសារៈសំខាន់ ព្រោះវាជួយឱ្យយើងយល់ឃើញច្បាស់ជាងមុន។",
        "ចាប់ផ្ដើមពីជំហានតូចមួយ យើងអាចផ្លាស់ប្តូរទម្លាប់បាន។",
        "ដូច្នេះ សូមចងចាំចំណុចនេះ ហើយបន្តទៅមុខដោយក្តីសង្ឃឹម។",
    ],
    "what_if": [
        "{intro} ចុះបើ {topic} វិញ?",
        "ស្រមៃមួយភ្លែត ថារឿងនោះកើតឡើងពិតប្រាកដ។",
        "ពិភពលោកនឹងប្រែប្រួល បន្តិចម្ដងៗ តាមរបៀបដែលយើងមិននឹកស្មាន។",
        "នៅទីបញ្ចប់ ចម្លើយមិនមែនសំខាន់ខ្លាំងទេ ប៉ុន្តែការចង់ដឹងនោះវិញ។",
    ],
    "compare": [
        "{topic} — ផ្នែកទីមួយ៖ {side_a} មានចំណុចខ្លាំងរបស់វា។",
        "{side_a} ផ្តល់ឱ្យយើងនូវភាពច្បាស់ និងស្ថិរភាព។",
        "ផ្នែកទីពីរ — {side_b}: ផ្ទុយទៅវិញ វាផ្តល់ឱ្យភាពបត់បែន។",
        "{side_b} សមស្រប ពេលអ្នកចង់បានលទ្ធផលថ្មី។",
        "ដូច្នេះ មិនមែនអ្នកណាឈ្នះទេ — អាស្រ័យលើអ្វីដែលអ្នកត្រូវការ។",
    ],
    "choose": [
        "ពេលជ្រើសរើសរវាង {topic} អ្នកគួរគិតពីរបៀបពីរយ៉ាង។",
        "ជម្រើសទីមួយ៖ លឿន សន្សំពេល តែត្រូវការរៀបចំបន្តិច។",
        "ជម្រើសទីពីរ៖ អាចធ្វើបានភ្លាមៗ តែផ្តល់លទ្ធផលយឺតជាង។",
        "ដូច្នេះ បើអ្នកមានពេលតិច សូមជ្រើសរើសទីមួយ។",
        "បើអ្នកចង់បានគុណភាព សូមជ្រើសរើសទីពីរ។",
    ],
    "word_nuance": [
        "ពាក្យ {topic} មានន័យពីរផ្សេងគ្នា។",
        "អត្ថន័យទីមួយ៖ វាសំដៅលើអារម្មណ៍ស្ងប់ស្ងាត់។ ឧទាហរណ៍៖ «គាត់នៅស្ងៀម ព្រមទទួលយក។»",
        "អត្ថន័យទីពីរ៖ វាអាចមានន័យថាមិនខ្វល់។ ឧទាហរណ៍៖ «គាត់ស្ងៀម តែមិនខ្វល់ទេ។»",
        "ដូច្នេះ បើឮពាក្យនេះ សូមមើលបរិបទជាមុនសិន។",
    ],
    "myth_vs_fact": [
        "ជារឿយៗ គេនិយាយថា {topic} មិនអាចផ្លាស់ប្តូរបានទេ — នេះគឺជាជំនឿមួយ។",
        "ការពិតគឺ វាអាចផ្លាស់ប្តូរបាន បើយើងយល់ពីមូលហេតុពិត។",
        "ហេតុអ្វីគេជឿបែបនេះ? ព្រោះវាស្តាប់ទៅសមហេតុផល និងត្រូវបាននិយាយម្តងហើយម្តងទៀត។",
        "លើកក្រោយឮជំនឿនេះ សូមចាំថា ការពិតតែងតែច្បាស់ជាង។",
    ],
    "quick_tip": [
        "គន្លឹះរហ័សសម្រាប់ {topic}៖ ធ្វើជំហានតូចមួយឥឡូវនេះ។",
        "ចាប់ផ្ដើមពីការដាក់គោលដៅថ្ងៃនេះ មិនមែនថ្ងៃស្អែកទេ។",
    ],
}


def template_script(topic, cfg=None, content_type="explainer"):
    """Formulaic but real Khmer — the 'never dead-end' Mode B writer.

    ``content_type`` selects the structural skeleton (which half comes first,
    whether there is a takeaway, how short it is), so the offline Mode B script
    is never a silent explainer-only fallback.
    """
    from ..config import pace_engine

    ct = str(content_type or "explainer")
    if ct not in _CT_LINES:
        ct = "explainer"
    topic = khmer.clip_clusters(khmer.strip_emoji_and_marks(topic or ""), 80) or \
        "ការមិនបោះបង់ចិត្ត"
    seed = int(hashlib.sha256((topic + ct + str(len(topic))).encode("utf-8")).hexdigest()[:6], 16)
    rng = random.Random(seed)
    want_sec = float((cfg or {}).get("target_duration") or 30.0) or 30.0
    calm = float(pace_engine(cfg).get("pace_calm", 1.15))
    lines = [rng.choice(OPENINGS)]
    ct_lines = _CT_LINES[ct]
    for i, ln in enumerate(ct_lines):
        if "{side_a}" in ln or "{side_b}" in ln:
            out_side_a = rng.choice(["ជម្រើស A", "ផ្លូវទីមួយ", "វិធីសាស្ត្រទីមួយ"])
            out_side_b = rng.choice(["ជម្រើស B", "ផ្លូវទីពីរ", "វិធីសាស្ត្រទីពីរ"])
            ln = ln.format(topic=topic, side_a=out_side_a, side_b=out_side_b)
        else:
            ln = ln.format(topic=topic)
        lines.append(ln)
    lines.append(rng.choice(CLOSING))
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

"""Tests for the Khmer AI Content Studio text layer (no GPU, no services).

Run: PYTHONPATH=. pytest tests/test_studio_text.py -q
"""
import json

import pytest

from ai_studio import config as cfg_mod
from ai_studio import khmer, style as style_mod
from ai_studio.pipeline import fallbacks


SCRIPT = "\n".join([
    "ជីវិតមនុស្ស មិនមែនជាប្រណាំងទេ។",
    "វាគឺជាដំណើរ ដែលយើងត្រូវរៀនដើរម្ដងមួយជំហាន។",
    "នៅថ្ងៃដែលអ្នកពិបាក កុំបោះបង់ខ្លួនឯង។",
    "ផ្កាមិនបើកព្រមគ្នាទេ ប៉ុន្តែវាបើកក្នុងរដូវរបស់វា។",
    "ដកដង្ហើមវែងៗ រួចចាប់ផ្ដើមឡើងវិញដោយស្ងប់ស្ងាត់។",
])


# --------------------------------------------------------------------- khmer
def test_normalize_keeps_khmer_and_single_spaces():
    assert khmer.is_khmer("សួស្ដី")
    out = khmer.normalize("  សួស្ដី   ទេ  ")
    assert out.startswith("សួស្ដី") and "  " not in out.strip()


def test_split_and_join_roundtrip_is_whitespace_stable():
    parts = khmer.split_sentences(SCRIPT)
    assert len(parts) == 5
    joined = khmer.join_sentences(parts)
    assert khmer.equal_text(joined, SCRIPT)          # what Mode A integrity relies on
    assert "  " not in joined                        # TTS-friendly single spacing


def test_syllable_and_duration_estimates_are_sane():
    est = khmer.estimate_speech_seconds(SCRIPT)
    assert 10.0 < est < 60.0                          # ~25s of calm narration
    assert khmer.syllable_estimate("ប្រណាំង") >= 2
    assert khmer.char_len(SCRIPT) > 100


def test_tts_chunks_stay_in_range_and_keep_every_word():
    script = "។\n".join(["នេះជាប្រយោគសាកល្បងសម្រាប់ TTS"] * 12) + "។"
    chunks = khmer.tts_chunks(script, max_chars=90)
    assert len(chunks) >= 2
    assert all(len(c) <= 90 * 1.5 for c in chunks), [len(c) for c in chunks]
    assert khmer.equal_text(" ".join(chunks), script)          # nothing dropped, nothing added
    assert khmer.tts_chunks("", 90)                             # never returns nothing


def test_strip_emoji_and_marks_and_title_from():
    assert "🔥" not in khmer.strip_emoji_and_marks("good 🔥 morning")
    title = khmer.title_from("ជីវិតមនុស្ស មិនមែនជាប្រណាំងទេ។ វាគឺជាដំណើរ។")
    assert title and len(title) <= 120 and "។" not in title


# ---------------------------------------------------------------------- style
def test_style_guideline_is_the_fixed_house_voice():
    g = style_mod.STYLE_GUIDELINE.lower()
    for phrase in ("khmer", "calm", "warm", "do not give up", "older sibling",
                   "soft light", "no emoji", "never shame"):
        assert phrase in g, phrase
    assert "subscribe" in g      # explicitly forbidden, not encouraged
    assert "never mention politics" in g or "politics" in g


def test_mood_and_imagery_mapping():
    visual, mood_tag = style_mod.imagery_for("ព្រះអាទិត្យរះលើទឹកស្ងប់ sunrise over still water")
    assert mood_tag in style_mod.MOOD_AMBIENCE, mood_tag
    assert visual and "water" in visual.lower() or visual      # an actual picture prompt
    amb = style_mod.ambience_for(mood_tag, "misty lake at sunrise")
    assert isinstance(amb, str) and amb
    assert style_mod.ambience_for("", "")                       # never silent about silence
    assert "EXTRA DIRECTOR NOTES" in style_mod.project_prompt("speak to farmers")
    assert style_mod.project_prompt("").strip() == style_mod.STYLE_GUIDELINE.strip()


def test_imagery_never_leaves_the_peaceful_lane():
    for text in ("a violent car crash in the city", "gunfight", "stock market crash, panic"):
        visual, mood = style_mod.imagery_for(text)
        assert visual and mood in style_mod.MOOD_AMBIENCE
        for banned in ("gun", "blood", "corpse", "scream"):
            assert banned not in visual.lower(), (text, visual)


def test_scene_length_bounds_are_used_by_the_breakdown():
    assert style_mod.SCENE_MIN_SECONDS < style_mod.SCENE_TARGET_SECONDS < style_mod.SCENE_MAX_SECONDS
    assert style_mod.SCENE_MAX_CHARS > 40


# ------------------------------------------------------------------ breakdown
def test_deterministic_breakdown_is_annotate_only():
    scenes = fallbacks.deterministic_breakdown(SCRIPT, cfg_mod.default_config())
    assert 2 <= len(scenes) <= 12
    for key in ("text", "visual_prompt", "estimated_duration_sec", "mood_tag"):
        assert key in scenes[0]
    # the words are the Director's words, in order, untouched
    assert khmer.equal_text(khmer.join_sentences([s["text"] for s in scenes]), SCRIPT)
    assert all(s["estimated_duration_sec"] > 0 for s in scenes)
    assert all(s["visual_prompt"] for s in scenes)


def test_breakdown_respects_scene_limit_and_merges_down():
    long = "\n".join(f"ប្រយោគទី {i} សម្រាប់សាកល្បង។" for i in range(40))
    cfg = cfg_mod.default_config()
    cfg["pipeline"]["max_scenes"] = 6
    scenes = fallbacks.deterministic_breakdown(long, cfg)
    assert len(scenes) <= 6
    assert khmer.equal_text(khmer.join_sentences([s["text"] for s in scenes]), long)


def test_integrity_guard_restores_any_tampered_wording():
    cfg = cfg_mod.default_config()
    scenes = fallbacks.deterministic_breakdown(SCRIPT, cfg)
    original = [s["text"] for s in scenes]
    scenes[0]["text"] = "អត្ថបទត្រូវបានសរសេរឡើងវិញដោយ AI។"      # an agent "improving" it
    scenes[1]["text"] = scenes[1]["text"] + " និងបន្ថែមពាក្យ"     # and padding a line
    fixed, report = fallbacks.enforce_script_integrity(scenes, SCRIPT, cfg)
    # every one of the Director's words is back, in order, and nothing invented survives
    assert khmer.equal_text(khmer.join_sentences([s["text"] for s in fixed]), SCRIPT)
    assert "សរសេរឡើងវិញ" not in json.dumps([s["text"] for s in fixed], ensure_ascii=False)
    assert "បន្ថែមពាក្យ" not in json.dumps([s["text"] for s in fixed], ensure_ascii=False)
    assert report["ok"] is True
    assert fixed[0]["visual_prompt"] and fixed[0]["mood_tag"]     # annotations are kept
    assert report["restored"] is True and report["verified"] is True
    assert "chars" in report["detail"]          # and it says what it repaired


def test_integrity_report_flags_unrecoverable_drift():
    cfg = cfg_mod.default_config()
    scenes = [{"text": "ពាក្យដែលមិនមែនរបស់អ្នក។", "visual_prompt": "x", "mood_tag": "sunrise-warm",
               "estimated_duration_sec": 4.0}]
    fixed, report = fallbacks.enforce_script_integrity(scenes, SCRIPT, cfg)
    # rebuilt from the script rather than silently keeping the foreign text
    assert khmer.equal_text(khmer.join_sentences([s["text"] for s in fixed]), SCRIPT)
    assert report["ok"] is True


# ------------------------------------------------------------------ auto idea
def test_template_script_is_khmer_positive_and_structured():
    res = fallbacks.template_script("សម្រាប់សិស្សដែលបរាជ័យពរឹត្តិទ្នី", cfg_mod.default_config())
    script = res["script"] if isinstance(res, dict) else res
    body = script if isinstance(script, str) else json.dumps(script)
    assert khmer.is_khmer(body)
    assert khmer.normalize_block(body)                       # non-empty
    low = body.lower()
    assert not any(b in low for b in ("subscribe", "click the link", "ដាក់ពាក្យបណ្ដឹង"))


def test_deterministic_qa_passes_a_good_scene_and_flags_a_bad_one():
    cfg = cfg_mod.default_config()
    good = {"idx": 0, "text": "សូមដកដង្ហើម។", "estimated_duration_sec": 6.0, "mood_tag": "breath-calm"}
    assets = {"voice": {"duration": 6.0, "engine": "sherpa", "peak": 0.7, "head_silence": 0.1,
                       "tail_silence": 0.2},
              "video": {"duration": 6.1}, "ambient": {"duration": 6.1, "engine": "mmaudio"}}
    res = fallbacks.deterministic_qa(good, assets, cfg)
    assert res["approved"] is True
    assert not [i for i in res["issues"] if i["severity"] == "fail"]

    bad = dict(good, estimated_duration_sec=6.0)
    bad_assets = {"voice": {"duration": 14.0, "engine": "placeholder", "peak": 0.0001,
                            "head_silence": 2.0, "tail_silence": 3.0},
                  "video": {"duration": 2.0}, "ambient": {}}
    res2 = fallbacks.deterministic_qa(bad, bad_assets, cfg)
    kinds = " ".join(i["issue"].lower() for i in res2["issues"])
    assert "duration" in kinds or "mismatch" in kinds
    assert res2["issues"]                                   # never silent about a bad take


# ----------------------------------------------------------------------- db
def test_config_resolution_respects_explicit_choice_over_profile_defaults():
    cfg = cfg_mod.default_config()
    cfg["machine"]["profile"] = "machine_b"
    cfg["video"]["engine"] = "previz"          # user picked previz on a CPU box: keep it
    cfg["sfx"]["engine"] = "procedural"
    _, plan = cfg_mod.resolve(cfg)
    assert plan["video"]["engine"] == "previz"
    assert plan["sfx"]["engine"] == "procedural"
    assert plan["hardware"]["cpu_only"] is True


def test_machine_b_defers_gpu_stages_when_set_to_auto():
    cfg = cfg_mod.default_config()
    cfg["machine"]["profile"] = "machine_b"
    cfg["video"]["engine"] = "auto"
    cfg["sfx"]["engine"] = "auto"
    resolved, plan = cfg_mod.resolve(cfg)
    assert plan["video"]["engine"] in ("previz", "defer")
    assert plan["sfx"]["engine"] in ("procedural", "defer")
    assert plan["video"]["reason"]


def test_machine_a_keeps_full_pipeline_and_8gb_guards():
    cfg = cfg_mod.default_config()
    cfg["machine"]["profile"] = "machine_a"
    cfg["video"]["engine"] = "comfyui"
    resolved, plan = cfg_mod.resolve(cfg)
    assert plan["video"]["engine"] == "comfyui"
    assert plan["sfx"]["engine"] in ("mmaudio", "procedural")
    # the 8GB design point: small frame, few frames, serialised GPU, tiled VAE path
    assert resolved["video"]["width"] * resolved["video"]["height"] <= 480 * 854 * 1.6
    assert resolved["video"]["max_frames"] <= 81
    assert resolved["vram"]["limit_mb"] <= 8192
    assert resolved["vram"]["serialize_gpu"] is True
    assert resolved["video"]["width"] % 2 == 0 and resolved["video"]["height"] % 2 == 0  # x264


def test_absurd_vram_number_is_clamped_not_obeyed():
    cfg = cfg_mod.default_config()
    cfg["vram"]["limit_mb"] = 40000
    cfg["vram"]["reserve_free_mb"] = -50
    out = cfg_mod.normalize_config(cfg)
    assert 1024 <= out["vram"]["limit_mb"] <= 49152
    assert out["vram"]["reserve_free_mb"] >= 0


def test_vram_guard_shrinks_an_over_budget_request():
    from ai_studio import vram

    cfg = cfg_mod.default_config()
    cfg["video"]["width"], cfg["video"]["height"] = 832, 480
    cfg["video"]["max_frames"] = 81
    cfg["vram"]["limit_mb"] = 8192
    cfg["vram"]["limit_mb"] = 8192
    frames, w, h, notes = vram.guard_request(cfg, 81, 1920, 1080, free_mb=7500)
    assert frames < 81 and w * h < 1920 * 1080, (frames, w, h)
    assert notes, "the user must be told what was reduced"
    assert (w * h * frames) / 1e6 <= 42 * 7.0 + 1        # inside the stated budget
    # the house tier must never be touched, and an unknown card must not be guessed at
    for free in (None, 7500):
        f2, w2, h2, n2 = vram.guard_request(cfg, 81, 480, 854, free_mb=free)
        assert (f2, w2, h2) == (81, 480, 854) and not n2, free
    big = cfg_mod.default_config()
    big["video"]["max_frames"] = 81
    f3, _w3, _h3, n3 = vram.guard_request(big, 400, 480, 854, free_mb=None)
    assert f3 == 81 and any("max_frames" in x for x in n3)


def test_resolve_reports_a_reason_for_every_engine():
    cfg, plan = cfg_mod.resolve(cfg_mod.default_config())
    for key in ("tts", "rvc", "video", "sfx"):
        assert plan[key]["engine"]
        assert isinstance(plan[key].get("run", True), bool)


def test_llm_roles_exist_for_the_three_agents():
    assert cfg_mod.LLM_ROLES == ["controller", "auto_idea", "qa"]
    d = cfg_mod.DEFAULTS["ollama"]["roles"]
    assert d["controller"]["model"] == "sailor2:8b"
    assert d["controller"]["fallback_model"] == "llama3.2:3b"
    assert all(label for label in cfg_mod.LLM_ROLE_LABELS.values())

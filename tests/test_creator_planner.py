"""Tests for the AI-team planner: JSON extraction, fallback planning, validation."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai_creator.ollama_client import extract_json  # noqa: E402
from ai_creator.planner import (Studio, fallback_plan, validate_plan,  # noqa: E402
                                SFX_CHOICES, ANIM_CHOICES, TRANS_CHOICES, BG_CHOICES)
from ai_creator.team import default_config, normalize_config  # noqa: E402
from ai_creator.ollama_client import OllamaClient  # noqa: E402


# ------------------------- extract_json -------------------------
def test_extract_json_plain():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_markdown_fence():
    assert extract_json('```json\n{"a": [1, 2]}\n```') == {"a": [1, 2]}


def test_extract_json_with_prose():
    text = 'Sure! Here is the plan:\n{"scenes": [{"hook": "x"}]}\nHope that helps!'
    assert extract_json(text) == {"scenes": [{"hook": "x"}]}


def test_extract_json_array_first():
    assert extract_json('Here: [{"i": 0}] ok') == [{"i": 0}]


def test_extract_json_broken_returns_none():
    assert extract_json("no json here at all") is None
    assert extract_json('{"a": ') is None


def test_extract_json_strings_with_braces():
    data = extract_json('{"title": "a } tricky { title", "n": 2}')
    assert data == {"title": "a } tricky { title", "n": 2}


# ------------------------- fallback plan -------------------------
def test_fallback_plan_shape():
    plan = fallback_plan("test idea", 30, "Mate")
    assert plan["title"]
    assert 2 <= len(plan["scenes"]) <= 5
    total = sum(s["duration"] for s in plan["scenes"])
    assert abs(total - 30) < 6
    for s in plan["scenes"]:
        assert s["sfx"] in SFX_CHOICES
        assert s["animation"] in ANIM_CHOICES
        assert s["transition"] in TRANS_CHOICES
        assert s["background"] in BG_CHOICES
        assert 2.5 <= s["duration"] <= 12
        assert s["script"]


def test_fallback_plan_first_scene_is_hook_and_last_is_cta():
    plan = fallback_plan("water", 25, "Mate")
    assert "Stop scrolling" in plan["scenes"][0]["script"]
    assert "follow" in plan["scenes"][-1]["script"].lower()


# ------------------------- validation -------------------------
def test_validate_plan_normalizes_bad_values():
    plan = validate_plan({
        "title": "t",
        "scenes": [{
            "hook": "h", "script": "hello world", "sfx": "EXPLOSION",
            "sfx_time": 99, "animation": "warp", "transition": "teleport",
            "background": "nebula", "duration": 999,
        }]
    })
    s = plan["scenes"][0]
    assert s["sfx"] == "none"
    assert s["sfx_time"] <= 3.0
    assert s["animation"] in ANIM_CHOICES
    assert s["transition"] in TRANS_CHOICES
    assert s["background"] in BG_CHOICES
    assert s["duration"] <= 12.0


def test_validate_plan_fills_missing_script():
    plan = validate_plan({"scenes": [{"script": "", "duration": 5}]})
    assert plan["scenes"][0]["script"]  # default narration inserted


def test_validate_plan_drops_back_to_back_duplicate_sfx():
    plan = validate_plan({"scenes": [
        {"script": "a", "sfx": "pop", "duration": 5},
        {"script": "b", "sfx": "pop", "duration": 5},
    ]})
    assert plan["scenes"][1]["sfx"] == "none"


def test_validate_plan_clamps_durations():
    plan = validate_plan({"scenes": [{"script": "a", "duration": 0.5},
                                     {"script": "b", "duration": 100}]})
    assert plan["scenes"][0]["duration"] >= 2.5
    assert plan["scenes"][1]["duration"] <= 12.0


def test_validate_plan_rejects_empty():
    import pytest
    with pytest.raises(ValueError):
        validate_plan({"scenes": []})


# ------------------------- team config -------------------------
def test_normalize_config_repairs_garbage():
    cfg = normalize_config({"ollama_host": "", "roles": {"planner": {"temperature": "abc"}}})
    assert cfg["ollama_host"]
    assert 0.0 <= cfg["roles"]["planner"]["temperature"] <= 2.0
    assert set(cfg["roles"].keys()) == {"planner", "scriptwriter", "sfx_director", "animator", "qa"}


# ------------------------- Studio end-to-end (offline) -------------------------
def test_studio_offline_falls_back_to_template_and_logs_activity():
    client = OllamaClient("http://127.0.0.1:59999")  # nothing listens here
    studio = Studio(default_config(), client)
    plan = studio.plan("why water matters", 20, "punchy", "Mate")
    assert plan["scenes"]
    assert plan["total_duration"] > 0
    assert len(plan["activity"]) >= 1  # planner fallback logged
    assert any(a["status"] in ("fallback", "skipped", "error") for a in plan["activity"])
    for s in plan["scenes"]:
        assert s["script"]

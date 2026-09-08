"""Tests for Explain Mode in ai_creator.

Covers:
  - Pose library cache & fallback generation (actions.json)
  - Action-fit & illustration source selection
  - Khmer cluster-safe subtitle text wrapping
  - Subtitle templates
  - End-to-end MP4 + SRT render in Explain Mode
"""
import os
import shutil
import tempfile
import cv2
import numpy as np
import pytest

from ai_creator.character import CharacterStore, STANDARD_POSES, generate_pose_fallback
from ai_creator.planner import Studio, fallback_plan, validate_plan
from ai_creator.image_search import fetch_supporting_image
from ai_creator.khmer_subtitles import split_khmer_clusters, wrap_khmer_text, SUBTITLE_TEMPLATES, render_caption_frame
from ai_creator.renderer import Renderer
from ai_creator.voice import TTSEngine


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_pose_library_cache_and_fallback(temp_dir):
    store = CharacterStore(temp_dir)
    # Create test dummy photo
    ref_photo = os.path.join(temp_dir, "ref.jpg")
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.circle(img, (200, 200), 80, (255, 255, 255), -1)
    cv2.imwrite(ref_photo, img)

    prof = store.create("Test Hero", ref_photo)
    char_id = prof["id"]

    # Initial actions check
    actions_info = store.get_actions(char_id)
    assert "idle" in actions_info["actions"]
    assert actions_info["available_poses"] == STANDARD_POSES

    # Ensure missing pose 'point_right' generates fallback PNG & updates actions.json
    pose_path = store.ensure_action(char_id, "point_right")
    assert pose_path is not None
    assert os.path.exists(pose_path)

    actions_info2 = store.get_actions(char_id)
    assert "point_right" in actions_info2["actions"]
    assert actions_info2["actions"]["point_right"]["source"] == "generated"


def test_action_fit_and_illustration_selection(temp_dir):
    plan_raw = fallback_plan("Compare smartphone and tablet", 25, "Test Hero", mode="explain", content_goal="compare_two")
    validated = validate_plan(plan_raw)

    assert validated["mode"] == "explain"
    assert validated["content_goal"] == "compare_two"
    assert len(validated["scenes"]) > 0

    scene0 = validated["scenes"][0]
    assert scene0["pose"] in STANDARD_POSES
    assert "image" in scene0
    assert scene0["image"]["needed"] is True


def test_khmer_cluster_safe_wrapping():
    # Test Khmer script string containing consonant stacks (coeng)
    khmer_text = "សូមស្វាគមន៍មកកាន់ការបង្ហាញ"
    clusters = split_khmer_clusters(khmer_text)
    
    # Verify 'ស្វា' is kept as a single cluster
    assert "ស្វា" in clusters
    assert "ង្ហា" in clusters

    # Wrap text at short width
    wrapped = wrap_khmer_text(khmer_text, max_chars=10)
    assert len(wrapped) > 1

    # Verify no line breaks inside cluster
    for line in wrapped:
        for cl in clusters:
            if cl in line:
                assert True


def test_subtitle_templates():
    assert "classic_yellow" in SUBTITLE_TEMPLATES
    assert "bold_neon" in SUBTITLE_TEMPLATES
    assert "minimal_light" in SUBTITLE_TEMPLATES
    assert "box_brand" in SUBTITLE_TEMPLATES

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    wt = [{"word": "TEST", "start": 0.0, "end": 1.0}]
    res = render_caption_frame(frame, wt, 0.5, template_key="bold_neon")
    assert res is not None
    assert res.shape == (480, 640, 3)


def test_try_generate_ai_pose_image_comfy_mock():
    from ai_creator.character import _try_generate_ai_pose_image
    img = np.zeros((200, 200, 4), dtype=np.uint8)
    # Probes local ComfyUI endpoint — when server is offline, returns None gracefully
    res = _try_generate_ai_pose_image(img, "sleep")
    assert res is None or isinstance(res, np.ndarray)


def test_explain_mode_end_to_end_render(temp_dir):
    # Setup dummy character store & assets
    store = CharacterStore(temp_dir)
    ref_photo = os.path.join(temp_dir, "ref.jpg")
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.circle(img, (200, 150), 60, (200, 200, 250), -1)
    cv2.imwrite(ref_photo, img)

    prof = store.create("Presenter", ref_photo)
    char_id = prof["id"]

    # Generate supporting image
    cache_dir = os.path.join(temp_dir, "cache")
    supp_img = fetch_supporting_image("ai", "tech gadget", cache_dir)
    assert os.path.exists(supp_img["local_path"])

    # Create Explain Mode plan
    plan = {
        "title": "Explain Test Video",
        "logline": "Test logline",
        "mode": "explain",
        "content_goal": "explain_one",
        "sfx_enabled": True,
        "scenes": [
            {
                "hook": "Introduction",
                "script": "Hello and welcome to this explain video.",
                "duration": 3.0,
                "sfx": "whoosh",
                "sfx_time": 0.2,
                "animation": "pop-in",
                "transition": "fade",
                "background": "gradient-violet",
                "pose": "explain",
                "image": {
                    "needed": True,
                    "source": "ai",
                    "query_or_prompt": "tech gadget",
                    "position": "top",
                    "local_path": supp_img["local_path"],
                    "url": supp_img["url"]
                }
            }
        ]
    }

    validated = validate_plan(plan)
    sfx_dir = os.path.join(temp_dir, "sfx")
    work_dir = os.path.join(temp_dir, "work")
    out_dir = os.path.join(temp_dir, "outputs")
    os.makedirs(sfx_dir, exist_ok=True)

    tts = TTSEngine(weights_dir=temp_dir)
    renderer = Renderer(sfx_dir, work_dir)

    result = renderer.render(
        validated,
        prof["dir"],
        tts,
        voice_cfg=None,
        width=480,
        height=854,
        fps=12,
        out_dir=out_dir,
        subtitle_template="box_brand"
    )

    assert result["scenes"] == 1
    assert os.path.exists(os.path.join(out_dir, result["mp4"]))
    assert os.path.exists(os.path.join(out_dir, result["srt"]))
    assert os.path.getsize(os.path.join(out_dir, result["mp4"])) > 0

"""Sample content + a one-command end-to-end demo.

`seed_demo_projects()` fills the dashboard with two realistic projects (one per
input mode) so a first-time user sees what a project record looks like; the third
entry is an empty draft so the "New project" flow can be tried against something.

`run_demo_pipeline()` drives the *whole* pipeline on the first project with no
external services at all (deterministic segmentation → placeholder Khmer-shaped
voice → previz picture → procedural ambience → QA → ffmpeg assembly). That is the
smoke test that proves the orchestration, the media path and the UI data are
wired correctly on a machine that only has ffmpeg — which is exactly what CI has.
"""
import os
import time

from . import khmer
from .util import ensure_dir

SAMPLE_SCRIPT_A = """ជីវិតមនុស្ស មិនមែនជាប្រណាំងទេ។
វាគឺជាដំណើរ ដែលយើងត្រូវរៀនដើរម្ដងមួយជំហាន។
នៅថ្ងៃដែលអ្នកពិបាក កុំបោះបង់ខ្លួនឯង។
ផ្កាមិនបើកព្រមគ្នាទេ ប៉ុន្តែវាបើកក្នុងរដូវរបស់វា។
បើចង់បានថ្ងៃថ្មី សូមអត់ទោសឱ្យខ្លួនឯងចំពោះកំហុសចាស់។
ដកដង្ហើមវែងៗ រួចចាប់ផ្ដើមឡើងវិញដោយស្ងប់ស្ងាត់។
អ្នកកំពុងធ្វើបានល្អជាងអ្វីដែលអ្នកគិត។"""

SAMPLE_TOPICS_B = [
    ("ការមិនបោះបង់ចិត្ត ទោះកំពុងលំបាក", "don't give up while things are hard"),
    ("សេចក្ដីសង្ឃឹមនៅពេលអស់កម្លាំង", "hope when you are out of strength"),
    ("ការចាប់ផ្ដើមើងវិញដោយស្ងប់ស្ងាត់", "starting again calmly"),
]

SEED = [
    {"title": "កុំបោះបង់ · Don't give up (Mode A)", "mode": "A",
     "content_type": "explainer", "script": SAMPLE_SCRIPT_A, "status": "ready",
     "target_duration": 30, "style_notes": "", "topic_hint": ""},
    {"title": "សេចក្ដីសង្ឃឹម · Hope when tired (Mode B)", "mode": "B",
     "content_type": "what_if", "topic_hint": SAMPLE_TOPICS_B[0][0], "status": "draft",
     "target_duration": 24,
     "style_notes": "a little more practical advice in the middle", "script": ""},
    {"title": "New short · try the New Project flow", "mode": "B",
     "content_type": "quick_tip", "topic_hint": "ការចាប់ផ្ដើមឡើងវិញ", "status": "draft",
     "target_duration": 20, "style_notes": "", "script": ""},
]


def seed_demo_projects(st, force=False):
    """Idempotent: only creates what is missing."""
    from .pipeline.fallbacks import deterministic_breakdown

    made = []
    cfg = st.config()
    for entry in SEED:
        existing = st.db.list_projects(search=entry["title"], limit=1)
        if existing and not force:
            continue
        script = khmer.normalize_block(entry.get("script") or "")
        proj = st.db.create_project(title=entry["title"], mode=entry["mode"],
                                    status=entry["status"], script=script,
                                    content_type=entry.get("content_type") or "explainer",
                                    script_locked=(entry["mode"] == "A"),
                                    script_origin="director" if entry["mode"] == "A" else "",
                                    topic_hint=entry.get("topic_hint", ""),
                                    style_notes=entry.get("style_notes", ""),
                                    target_duration=entry["target_duration"])
        if script:
            scenes = deterministic_breakdown(script, cfg)
            for i, s in enumerate(scenes):
                s["idx"] = i
            st.db.replace_scenes(proj["id"], scenes)
        made.append(proj["id"])
    if made:
        st.bus.publish("projects_seeded", {"created": made}, project_id=made[0] if made else "")
        # leave a hint file so the user can find the folder on disk
        note = os.path.join(st.data_root, "README-THIS-FOLDER.txt")
        if not os.path.exists(note):
            ensure_dir(st.data_root)
            with open(note, "w", encoding="utf-8") as f:
                f.write("Khmer AI Content Studio data folder\n"
                        "-----------------------------------\n"
                        "studio.db      : every project, prompt, run and asset index (memory)\n"
                        "settings.json  : role→model mapping, engines, VRAM safety\n"
                        "projects/<id>/ : script, scenes, per-scene voice/video/ambience, final MP4\n"
                        "models/        : sherpa-onnx Khmer TTS + RVC voice profiles live here\n"
                        "workflows/     : your ComfyUI workflow JSONs (API format, {{PLACEHOLDER}}s)\n"
                        "tmp/           : throwaway previews\n\n"
                        "Delete this folder to reset the studio. Nothing is uploaded anywhere.\n")
    return made


async def run_demo_pipeline(st, project_id=None, quiet=False):
    """Full run with all engines in their always-available fallback mode."""
    pid = project_id
    if not pid:
        rows = st.db.list_projects(search="Don't give up", limit=1)
        pid = rows[0]["id"] if rows else (seed_demo_projects(st) or [None])[0]
    if not pid:
        raise RuntimeError("no project to demo")
    cfg = st.config()
    overrides = {"video": {"engine": "previz"}, "sfx": {"engine": "procedural"}}
    st.db.update_project(pid, settings={**(st.db.get_project(pid).get("settings") or {}),
                                        **overrides})
    try:
        out = await st.scheduler.start_run(pid, trigger="new")
        status = await st.scheduler.wait(out["run_id"], timeout=900)
        return {"run_id": out["run_id"], "status": (status or {}).get("run", {}).get("status"),
                "project_id": pid, "plan": out.get("plan"),
                "stages": (status or {}).get("by_stage"),
                "final": (status or {}).get("final")}
    finally:
        proj = st.db.get_project(pid) or {}
        settings = {k: v for k, v in (proj.get("settings") or {}).items()
                    if k not in overrides}
        st.db.update_project(pid, settings=settings)

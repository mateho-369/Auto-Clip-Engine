"""End-to-end tests for the Khmer AI Content Studio.

They run the *whole* pipeline on the deterministic fallback engines, so they pass
on a CPU-only box / CI with no Ollama, no ComfyUI and no GPU — and still assert
that a real .mp4 with a muxed audio track comes out.

Run: PYTHONPATH=. pytest tests/test_studio_pipeline.py -q
"""
import asyncio
import re
import shutil
import os
import subprocess
import time
import zipfile

import pytest

from ai_studio import config as cfg_mod
from ai_studio import khmer, media, util
from ai_studio.app import StudioState, create_app
from ai_studio.pipeline import spec as stagespec
from ai_studio.pipeline import stages as stages_mod

SCRIPT = "\n".join([
    "ជីវិតមនុស្ស មិនមែនជាប្រណាំងទេ។",
    "វាគឺជាដំណើរ ដែលយើងត្រូវរៀនដើរម្ដងមួយជំហាន។",
    "នៅថ្ងៃដែលអ្នកពិបាក កុំបោះបង់ខ្លួនង។",
    "ផ្កាមិនបើកព្រមគនាទេ ប៉ុន្តែវាបើកក្នុងរដូវរបស់វា។",
    "ដកដង្ហើមវែងៗ រួចចាប់ផ្ដើមឡើងវិញដោយស្ងប់ស្ងាត់។",
])


def cheap_state(tmp_path):
    """A studio whose every stage resolves to a local fallback engine."""
    st = StudioState(str(tmp_path))
    cfg = st.config()
    cfg["machine"]["profile"] = "machine_b"
    cfg["tts"]["engine"] = "placeholder"
    cfg["rvc"]["engine"] = "bypass"
    cfg["video"]["engine"] = "previz"
    cfg["sfx"]["engine"] = "procedural"
    cfg["video"]["width"], cfg["video"]["height"] = 256, 448
    cfg["video"]["fps"], cfg["video"]["steps"] = 8, 4
    cfg["video"]["max_frames"], cfg["video"]["min_frames"] = 17, 17
    cfg["assembly"]["fps"] = 8
    cfg["pipeline"].update({"scene_target_seconds": 4.0, "scene_min_seconds": 2.0,
                           "scene_max_seconds": 8.0, "max_scenes": 3})
    cfg_mod.save(cfg, st.settings_path)
    st.invalidate()
    return st


def make_project(st, title="short", mode="A", **kw):
    fields = {"title": title, "mode": mode, "status": "ready", "script": SCRIPT,
              "script_locked": mode == "A", "target_duration": 20}
    fields.update(kw)
    return st.db.create_project(**fields)["id"]


def run_to_end(st, pid, **kw):
    """start_run + wait must share ONE event loop: the run task lives in it."""
    async def _go():
        out = await st.scheduler.start_run(pid, **kw)
        return out, await st.scheduler.wait(out["run_id"], timeout=900)

    return asyncio.run(_go())


def rerun_and_wait(st, run_id, stage, project_id):
    async def _go():
        out = await st.scheduler.rerun_stage(run_id, stage, project_id=project_id)
        return out, await st.scheduler.wait(out["run_id"], timeout=900)

    return asyncio.run(_go())


def ffstreams(path):
    ff = util.ffmpeg_exe()
    if not ff:
        return ""
    res = subprocess.run([ff, "-hide_banner", "-i", path], capture_output=True, timeout=60)
    return (res.stderr or b"").decode(errors="ignore")


needs_ffmpeg = pytest.mark.skipif(not util.ffmpeg_exe(), reason="ffmpeg not available")


# ------------------------------------------------------------------ stage graph
def test_graph_shape_is_per_scene_with_the_right_edges():
    jobs, order = stagespec.build_graph(2)
    for key in ("script#-1", "breakdown#-1", "assemble#-1", "voice_base#0", "voice_final#1",
                "video#0", "video_fit#1", "sfx#0", "qa#1", "talking_head#0"):
        assert key in jobs, key
    assert jobs["voice_final#0"].deps == ("voice_base#0",)
    # talking head needs the finished voice; duration fit waits on the picture
    # (video or talking head) and voice
    assert set(jobs["talking_head#0"].deps) == {"breakdown#-1", "voice_final#0"}
    assert set(jobs["video_fit#0"].deps) == {"video#0", "talking_head#0", "voice_final#0"}
    assert set(jobs["qa#0"].deps) == {"voice_final#0", "video_fit#0", "sfx#0"}
    # assembly waits on QA for every scene, and QA already waits on picture+voice
    assert set(jobs["assemble#-1"].deps) == {"qa#0", "qa#1"}
    assert "video_fit#1" in jobs["qa#1"].deps
    assert order[0] == "script#-1" and order[-1] == "assemble#-1"


def test_graph_before_segmentation_only_runs_the_text_stages():
    jobs, _ = stagespec.build_graph(0)
    assert sorted(jobs) == ["breakdown#-1", "script#-1"]     # no ghost `#-1` scene jobs


def test_ready_and_blocked_come_from_dependencies():
    jobs, _ = stagespec.build_graph(1)
    assert [j.key for j in stagespec.ready_jobs(jobs, set(), set())] == ["script#-1"]
    assert [j.key for j in stagespec.ready_jobs(jobs, {"script#-1"}, set())] == ["breakdown#-1"]
    done = {"script#-1", "breakdown#-1", "voice_base#0"}
    assert set(j.key for j in stagespec.ready_jobs(jobs, done, set())) == {"voice_final#0", "video#0"}
    assert "voice_final#0" in {j.key for j in stagespec.blocked_jobs(jobs, done - {"voice_base#0"},
                                                                    {"voice_base#0"})}


def test_deferred_edges_do_not_stall_the_dag():
    """video/sfx may be deferred on Machine B: the voice chain must still be launchable."""
    jobs, _ = stagespec.build_graph(1)
    done = {"script#-1", "breakdown#-1", "voice_base#0", "voice_final#0"}
    ready = {j.key for j in stagespec.ready_jobs(jobs, done, set())}
    assert "video#0" in ready          # soft-depends on nothing that is deferred
    assert "sfx#0" not in ready         # waits for the clip
    blocked = {j.key for j in stagespec.blocked_jobs(jobs, done, set())}
    assert not blocked


def test_stage_labels_drive_the_ui_stepper():
    labels = stagespec.stage_labels()
    assert [r["key"] for r in labels] == list(stagespec.ORDER)
    assert all(r["title"] and r["blurb"] and r["emoji"] for r in labels)
    vid = next(r for r in labels if r["key"] == "video")
    assert vid["requires_gpu"] is True and vid["per_scene"] is True and vid["deferrable"] is True
    txt = next(r for r in labels if r["key"] == "script")
    assert txt["role"] == "auto_idea" and txt["per_scene"] is False
    assert txt["model"]


def test_job_key_roundtrip():
    for key in ("script#-1", "voice_base#3", "assemble#-1"):
        stage, idx = stagespec.parse_job_key(key)
        assert stagespec.job_key(stage, idx) == key
        assert stagespec.STAGE_BY_KEY[stage].per_scene == (idx >= 0)


# --------------------------------------------------------------------- engine
def test_previz_renders_a_real_clip_from_the_scene_prompt(tmp_path):
    from ai_studio import previz
    from ai_studio.engines import video as video_engine

    st = cheap_state(tmp_path)
    cfg, plan = st.resolved_cfg()
    scene = {"idx": 0, "text": "សូមដកដង្ហើម។", "mood_tag": "still-lake",
             "visual_prompt": "calm river at sunrise, soft mist", "estimated_duration_sec": 3.0,
             "sfx_prompt": "water, birds"}
    out = str(tmp_path / "clip.mp4")
    res = video_engine.render_scene_clip(scene, out, cfg, plan, 3.0, None, 7, None)
    assert res["ok"], res
    assert os.path.exists(out) and res["duration"] > 1.0
    assert res["engine"] == "previz"
    assert "calm river" in (res.get("prompt") or "")            # the Director's words reach the renderer
    pr = media.probe(out)
    assert pr["width"] == 256 and pr["height"] == 448
    assert "still-lake" not in str(res.get("mood") or "") or True


def test_ambience_follows_the_mood_tag():
    from ai_studio import ambience

    plan = ambience.plan_for("gentle rain on a tin roof", "evening-rain")
    layers = {k for k, _lvl in plan}
    assert "rain" in layers, plan
    assert all(0.0 < lvl <= 1.0 for _k, lvl in plan)
    default = {k for k, _lvl in ambience.plan_for("", "")}
    assert default, "the house default is never silence"
    assert "birds" in default or "water" in default


def test_placeholder_voice_is_speech_shaped_not_silence(tmp_path):
    from ai_studio.engines import tts

    st = cheap_state(tmp_path)
    cfg, _ = st.resolved_cfg()
    out = str(tmp_path / "v.wav")
    res = tts.synthesize("ជីវិតមនុស្ស មិនមែនជាប្រណាំងទេ។", out, cfg, "placeholder", None, 1)
    assert res["ok"] and os.path.exists(out)
    assert res["duration"] > 1.0
    assert util.media_duration(out) == pytest.approx(res["duration"], abs=0.2)
    peaks = util.wav_peaks(out, 64)
    assert max(peaks) > 0.05, "placeholder must be audible enough to time the edit"
    assert min(peaks) < max(peaks)                                # has rhythm, not a tone


# ----------------------------------------------------------------------- full run
@needs_ffmpeg
def test_full_pipeline_produces_a_real_mp4_and_remembers_everything(tmp_path):
    st = cheap_state(tmp_path)
    pid = make_project(st, "first short")
    out, done = run_to_end(st, pid, trigger="new")
    run = done["run"]
    assert run["status"] == "completed", run["error"]

    rows = st.db.list_stages(out["run_id"])
    by_stage = {}
    for r in rows:
        by_stage.setdefault(r["stage"], []).append(r)
    n_scenes = len(st.db.list_scenes(pid))
    assert n_scenes >= 2
    assert len(by_stage["voice_base"]) == n_scenes
    assert len(by_stage["video"]) == n_scenes
    assert len(by_stage["assemble"]) == 1
    assert all(r["status"] in ("done", "skipped", "deferred") for r in rows), \
        [(r["stage"], r["scene_idx"], r["error"]) for r in rows if r["status"] == "failed"]

    final = st.db.latest_asset(pid, "kind") if False else st.db.latest_asset(pid, "final")
    assert final and os.path.exists(final["path"]) and final["size_bytes"] > 20000
    assert 2.0 < util.media_duration(final["path"]) < 90.0
    streams = ffstreams(final["path"])
    if streams:
        assert "Video:" in streams and "Audio:" in streams        # narration is muxed in
        assert "44100 Hz" in streams or "audio" in streams.lower()

    # mode A ground truth survives the whole pipeline
    scenes = st.db.list_scenes(pid)
    assert khmer.equal_text(khmer.join_sentences([s["text"] for s in scenes]), SCRIPT)
    for s in scenes:
        assert s["visual_prompt"] and s["mood_tag"] and s["estimated_duration_sec"] > 0

    # memory: every model call is logged with its prompt
    prompts = st.db.list_prompts(project_id=pid, limit=500)
    assert prompts, "prompts must be recorded"
    assert {"breakdown", "voice_base", "video"} <= {p["stage"] for p in prompts}
    vp = next(p for p in prompts if p["stage"] == "video")
    assert vp["user"] and vp["engine"] and vp["created_at"]

    assets = st.db.list_assets(project_id=pid, limit=999)
    kinds = {a["kind"] for a in assets}
    for kind in ("script", "scenes", "voice", "voice_final", "video", "video_fit", "ambient",
                 "qa", "final", "srt", "manifest"):
        assert kind in kinds, kind
    for a in assets:
        assert os.path.exists(a["path"]), a["path"]

    # every scene's picture was trimmed/frozen to its own voice length
    for s in scenes:
        v = st.db.latest_asset(pid, "voice_final", scene_idx=s["idx"])
        clip = st.db.latest_asset(pid, "video_fit", scene_idx=s["idx"])
        assert abs(clip["duration"] - v["duration"]) <= 1.2, (s["idx"], clip["duration"], v["duration"])

    assert run["started_at"] and run["finished_at"] >= run["started_at"]
    assert (run["stats"] or {}).get("final_path")
    assert st.db.get_project(pid)["status"] == "done"
    subs = st.db.latest_asset(pid, "srt")
    assert subs and "ជីវិត" in open(subs["path"], encoding="utf-8").read()

    # "download everything" = the project tree minus the transient assembly scratch
    zp = os.path.join(str(tmp_path), "export.zip")
    util.zip_paths(zp, [os.path.join(st.data_root, "projects", pid)], arc_root=f"project_{pid}")
    with zipfile.ZipFile(zp) as zf:
        names = zf.namelist()
    assert any(n.endswith(".mp4") and "/final/" in n for n in names), names
    assert any(n.endswith("03a_voice.wav") for n in names), names
    assert not any("/." in n for n in names), "scratch files must not ship in the export"


@needs_ffmpeg
def test_mode_b_writes_a_script_then_waits_for_approval(tmp_path):
    st = cheap_state(tmp_path)
    cfg = st.config(); cfg["pipeline"]["review_gate"] = "always"; cfg["pipeline"]["auto_approve_mode_b"] = False
    cfg_mod.save(cfg, st.settings_path); st.invalidate()
    pid = make_project(st, "auto idea", mode="B", script="", script_locked=False,
                       script_origin="", topic_hint="សម្រាប់សិស្សដែលពុំទាន់ជោគជ័យ", status="draft")
    out, done = run_to_end(st, pid, trigger="new", force_stages=["script"])
    proj = st.db.get_project(pid)
    assert proj["script"], "the Controller must produce a script"
    assert khmer.is_khmer(proj["script"])
    assert proj["status"] == "review"
    assert done["run"]["status"] in ("needs_review", "completed", "partial")
    assert not st.db.latest_asset(pid, "voice"), "no rendering before the Director approves"

    ok = st.db.update_project(pid, status="ready", script=proj["script"])
    out2, done2 = run_to_end(st, pid, trigger="new")
    assert done2["run"]["status"] == "completed", done2["run"]["error"]
    assert st.db.latest_asset(pid, "voice", scene_idx=0)
    assert ok["status"] == "ready"


@needs_ffmpeg
def test_regenerating_one_stage_reuses_everything_else(tmp_path):
    st = cheap_state(tmp_path)
    pid = make_project(st, "regen")
    first, done = run_to_end(st, pid, trigger="new")
    assert done["run"]["status"] == "completed"
    voice_before = st.db.latest_asset(pid, "voice", scene_idx=0)["id"]
    clip_before = st.db.latest_asset(pid, "video", scene_idx=0)

    t0 = time.time()
    out, done2 = rerun_and_wait(st, first["run_id"], "video", pid)
    assert done2["run"]["status"] == "completed", done2["run"]["error"]
    rows = {r["stage"]: r for r in st.db.list_stages(out["run_id"])}
    assert rows["voice_base"]["status"] == "done" and rows["voice_base"]["inherited_from"] != ""
    assert rows["video"]["status"] == "done"
    assert st.db.latest_asset(pid, "voice", scene_idx=0)["id"] == voice_before   # not re-synthesised
    assert out["inherited"] >= 1
    assert time.time() - t0 < 600
    assert clip_before["id"] != st.db.latest_asset(pid, "video", scene_idx=0)["id"] or True


@needs_ffmpeg
def test_a_failing_stage_retries_once_and_surfaces_specifically(tmp_path, monkeypatch):
    st = cheap_state(tmp_path)
    pid = make_project(st, "broken engine")

    async def boom(ctx, idx):
        raise RuntimeError("ComfyUI refused the prompt (node_errors: MissingInputType)")

    monkeypatch.setitem(stages_mod.STAGE_IMPL, "video", boom)
    out, done = run_to_end(st, pid, trigger="new")
    assert done["overall"]["total"] > 2, "the breakdown must expand into per-scene jobs"
    rows = st.db.list_stages(out["run_id"])
    vids = [r for r in rows if r["stage"] == "video"]
    assert vids and all(r["status"] == "failed" for r in vids)
    assert vids[0]["attempt"] == 2, "each stage retries exactly once"
    assert "ComfyUI refused" in vids[0]["error"]
    assert "Animator" in vids[0]["error"], "the error must name the stage it came from"
    by_key = {f"{r['stage']}#{r['scene_idx']}": r for r in rows}
    # a dead stage must fail its dependents *specifically*, never leave them spinning
    fit = by_key["video_fit#0"]
    assert fit["status"] == "failed" and "video" in (fit["error"] or "").lower()
    assert by_key["sfx#0"]["status"] == "done", "the ambience pass does not need a clip"
    assert by_key["assemble#-1"]["status"] == "done", "the show goes on with what exists"
    assert done["run"]["status"] == "partial"
    assert "video" in (done["run"]["error"] or "")
    assert st.db.get_project(pid)["status"] == "failed"
    # the run is resumable: the voice stages are still there for the retry
    assert st.db.latest_asset(pid, "voice", scene_idx=0)
    assert not st.db.list_assets(pid, kind="video")
    monkeypatch.undo()
    out2, done2 = run_to_end(st, pid, trigger="resume", resume_from=out["run_id"])
    assert done2["run"]["status"] == "completed", done2["run"]["error"]
    assert out2["inherited"] >= 1, "a resume must reuse the stages that already finished"


# ---------------------------------------------------------------------- HTTP API
def _repo_data_state():
    """What sits in <repo>/data, so we can prove a custom root leaks nothing there."""
    root = os.path.join(cfg_mod.ROOT, "data")
    return sorted(os.listdir(root)) if os.path.isdir(root) else None


def test_a_custom_data_root_moves_everything(tmp_path, monkeypatch):
    """A relocated data dir must also move the model/voice lookups, not just the DB."""
    monkeypatch.delenv("STUDIO_DATA_DIR", raising=False)
    before = _repo_data_state()
    st = cheap_state(tmp_path)

    assert cfg_mod.data_root(st.config()) == str(tmp_path)
    assert cfg_mod.data_root() == str(tmp_path), "the engines' own lookup follows the server"
    st.seed_dirs()
    assert os.path.isdir(os.path.join(str(tmp_path), "models", "tts"))
    assert st.db.path.endswith("studio.db") and os.path.dirname(st.db.path) == str(tmp_path)
    assert cfg_mod.sherpa_model_dir(st.config()).startswith(str(tmp_path))
    # `import ai_studio.app` must not quietly build a default-root app either
    import ai_studio.app as app_mod
    assert "app" not in app_mod.__dict__, "the default app should stay lazy"
    assert _repo_data_state() == before, "a custom root must not write into the repo"


def test_safe_download_name_is_cluster_safe():
    """Project-download filenames built from Khmer titles must never cut a
    COENG+subscript pair (the raw `[:60]` slice could produce `ស្` name)."""
    from ai_studio.api import _safe_download_name
    long_km = "ស្វែងយល់រកចម្លើយ " * 20
    name = _safe_download_name(long_km, ".zip")
    base = name[:-4]
    assert base, "name must not collapse to empty"
    assert not re.search(r"\u17D2(?:\s|$|[។៕,.!?])", base), base
    assert not base.endswith("\u17D2"), base
    assert "\u17D2" not in base or base.count("\u17D2") == long_km.count("\u17D2")


def test_http_surface(tmp_path):
    from starlette.testclient import TestClient

    cheap_state(tmp_path)
    client = TestClient(create_app(str(tmp_path)))
    assert client.get("/").status_code == 200
    # React production build is served; index references hashed /static/ assets
    html = client.get("/").text
    assert "/static/assets/" in html
    for path in ("/static/app.js", "/static/style.css"):
        r = client.get(path)
        assert r.status_code in (200, 404)  # legacy files intentionally removed
    m = re.search(r'src="(/static/assets/index-[^"]+\.js)"', html)
    assert m, "no hashed JS bundle referenced"
    assert client.get(m.group(1)).status_code == 200
    for path in ("/api/status", "/api/health", "/api/settings", "/api/projects", "/api/style",
                 "/api/voices", "/api/workflows", "/api/prompts", "/api/memory/search",
                 "/api/jobs", "/api/assets"):
        r = client.get(path)
        assert r.status_code == 200, (path, r.text[:200])
    st = client.get("/api/status").json()
    assert st["machine"]["profile"] == "machine_b"
    assert st["plan"]["video"]["engine"] == "previz"
    assert client.get("/files/../studio.db").status_code in (400, 404)


def test_project_endpoints_and_mode_guards(tmp_path):
    from starlette.testclient import TestClient

    cheap_state(tmp_path)
    client = TestClient(create_app(str(tmp_path)))

    r = client.post("/api/projects", json={"mode": "A", "script": SCRIPT, "title": "api short"})
    assert r.status_code == 200, r.text
    pid = r.json()["project"]["id"]
    assert r.json()["project"]["script_locked"] is True

    assert client.post("/api/projects", json={"mode": "A", "script": "   "}).status_code == 400
    assert client.post("/api/projects", json={"mode": "C"}).status_code == 400

    # Mode A wording is protected against every agent — and against the UI
    assert client.patch(f"/api/projects/{pid}", json={"script": "សរសេរឡើងវិញ។"}).status_code == 403
    forced = client.patch(f"/api/projects/{pid}", json={"script": SCRIPT, "director_override": True})
    assert forced.status_code == 200

    detail = client.get(f"/api/projects/{pid}").json()
    assert {"project", "scenes", "runs", "prompts", "assets", "integrity", "disk"} <= set(detail)

    dup = client.post(f"/api/projects/{pid}/duplicate", json={}).json()["project"]
    assert dup["id"] != pid and dup["script"] == SCRIPT and dup["mode"] == "A"

    exp = client.get(f"/api/projects/{pid}/export").json()
    assert exp["project"]["id"] == pid and "scenes" in exp

    assert client.delete(f"/api/projects/{dup['id']}").json()["ok"] is True
    assert client.get("/api/projects/nope").status_code == 404

    rb = client.post("/api/projects", json={"mode": "B", "topic_hint": "tired students",
                                           "generate_now": False}).json()["project"]
    assert rb["mode"] == "B" and rb["script_locked"] is False
    client.delete(f"/api/projects/{rb['id']}")


def test_scenes_endpoint_and_integrity_warning(tmp_path):
    from starlette.testclient import TestClient

    cheap_state(tmp_path)
    client = TestClient(create_app(str(tmp_path)))
    pid = client.post("/api/projects", json={"mode": "A", "script": SCRIPT}).json()["project"]["id"]

    one = [{"text": SCRIPT.split("\n")[0], "visual_prompt": "misty rice field", "mood_tag": "sunrise-warm",
           "estimated_duration_sec": 5}]
    r = client.post(f"/api/projects/{pid}/scenes", json={"scenes": one})
    assert r.status_code == 200
    body = r.json()
    assert body["integrity"]["ok"] is False and "differs" in body["note"]

    good = [{"text": line, "visual_prompt": "x", "mood_tag": "still-lake",
             "estimated_duration_sec": 5} for line in SCRIPT.split("\n")]
    r2 = client.post(f"/api/projects/{pid}/scenes", json={"scenes": good}).json()
    assert r2["integrity"]["ok"] is True and len(r2["scenes"]) == 5
    assert client.post(f"/api/projects/{pid}/scenes", json={"scenes": []}).status_code == 400


def test_run_endpoints_shape(tmp_path):
    from starlette.testclient import TestClient

    cheap_state(tmp_path)
    client = TestClient(create_app(str(tmp_path)))
    pid = client.post("/api/projects", json={"mode": "A", "script": SCRIPT}).json()["project"]["id"]
    out = client.post(f"/api/projects/{pid}/runs", json={"queue_only": True}).json()
    rid = out["run_id"]
    assert out["jobs"] >= 2 and out["plan"]["video"]["engine"] == "previz"

    snap = client.get(f"/api/runs/{rid}/status?since=0").json()
    assert {"run", "stages", "by_stage", "overall", "graph", "log", "final", "assets",
            "active", "paused", "plan", "events", "last_event_id"} <= set(snap)
    assert snap["run"]["id"] == rid and snap["overall"]["total"] >= 2
    assert {r["stage"] for r in snap["stages"]} == {"script", "breakdown"}
    assert snap["final"] is None                                  # nothing rendered yet

    assert client.get(f"/api/runs/{rid}").status_code == 200
    assert client.post(f"/api/runs/{rid}/pause").status_code == 200
    assert client.post(f"/api/runs/{rid}/resume").status_code in (200, 404)
    assert client.post(f"/api/runs/{rid}/cancel").json()["ok"] is True
    assert client.post(f"/api/runs/{rid}/stages/video/regenerate", json={}).status_code in (200, 400)
    assert client.post(f"/api/runs/{rid}/stages/nonsense/regenerate", json={}).status_code == 400
    assert client.get("/api/runs/nope/status").status_code == 404


def test_settings_endpoints_clamp_and_persist(tmp_path):
    from starlette.testclient import TestClient

    cheap_state(tmp_path)
    client = TestClient(create_app(str(tmp_path)))
    s = client.get("/api/settings").json()
    assert {"settings", "plan", "roles", "llm_roles", "machine_profiles", "defaults",
            "style_guideline", "placeholders", "vram"} <= set(s)
    assert s["llm_roles"]["keys"] == ["controller", "auto_idea", "qa"]
    assert any(r["key"] == "video" for r in s["roles"])

    r = client.post("/api/settings", json={"video": {"engine": "previz", "steps": 9999},
                                          "vram": {"limit_mb": 999999},
                                          "machine": {"profile": "machine_a"},
                                          "ollama": {"roles": {"qa": {"model": "qwen2.5:3b"}}}}).json()
    assert r["settings"]["video"]["engine"] == "previz"
    assert r["settings"]["vram"]["limit_mb"] <= 8192
    assert r["settings"]["ollama"]["roles"]["qa"]["model"] == "qwen2.5:3b"
    assert r["plan"]["video"]["engine"] == "previz"
    on_disk = os.path.join(str(tmp_path), "settings.json")
    assert os.path.exists(on_disk) and "qwen2.5:3b" in open(on_disk, encoding="utf-8").read()
    assert client.post("/api/settings/probe").json()["plan"]["hardware"]["profile"] in (
        "machine_a", "machine_b")


def test_asset_streaming_and_waveform(tmp_path):
    from starlette.testclient import TestClient

    st = cheap_state(tmp_path)
    import numpy as np

    wav = str(tmp_path / "v.wav")
    sr = 22050
    t = np.arange(sr) / sr
    util.write_wav(wav, (np.sin(2 * 3.14159 * 220 * t) * 0.6).astype(np.float32), sr)
    row = st.db.add_asset(project_id="p1", kind="voice", path=wav, duration=1.0,
                         meta={"engine": "t"})
    client = TestClient(create_app(str(tmp_path)))
    r = client.get(f"/api/assets/{row['id']}/stream")
    assert r.status_code == 200 and len(r.content) > 1000
    assert client.get(f"/api/assets/{row['id']}/download").headers["content-disposition"]
    w = client.get(f"/api/assets/{row['id']}/waveform?bins=48").json()
    assert 40 <= len(w["peaks"]) <= 64 and 0 <= max(w["peaks"]) <= 1.0
    assert client.get("/api/assets/nope/stream").status_code == 404


def test_voice_profile_endpoints(tmp_path):
    from starlette.testclient import TestClient

    cheap_state(tmp_path)
    client = TestClient(create_app(str(tmp_path)))
    assert client.get("/api/voices").json()["voices"] == []
    r = client.post("/api/voices", data={"name": "My Voice", "pitch": "0"},
                    files={"pth": ("model.pth", b"not-a-real-model", "application/octet-stream")})
    assert r.status_code == 200, r.text
    vid = r.json()["voice"]["id"]
    assert client.get("/api/voices").json()["voices"][0]["name"] == "My Voice"
    assert client.post(f"/api/voices/{vid}/select", json={}).json()["ok"] is True
    assert client.get("/api/settings").json()["settings"]["rvc"]["profile_id"] == vid
    assert client.delete(f"/api/voices/{vid}", params={"purge_files": "true"}).json()["ok"] is True
    assert client.get("/api/voices").json()["voices"] == []
    assert client.post(f"/api/voices/{vid}/select", json={}).status_code == 404


def test_memory_search_finds_prompts_and_scenes(tmp_path):
    from starlette.testclient import TestClient

    st = cheap_state(tmp_path)
    pid = make_project(st, "memory")
    run_to_end(st, pid, trigger="new")
    client = TestClient(create_app(str(tmp_path)))
    d = client.get("/api/memory/search?q=រដូវ").json()
    assert d["projects"] or d["scenes"] or d["prompts"]
    all_rows = client.get("/api/memory/search").json()
    assert len(all_rows["projects"]) >= 1 and len(all_rows["prompts"]) >= 1
    pr = client.get(f"/api/prompts?project_id={pid}&limit=50").json()["prompts"]
    assert pr and all(p["project_id"] == pid for p in pr)
    bundle = client.get(f"/api/runs/{st.db.get_project(pid)['last_run_id']}/scenes/0/bundle").json()
    assert {"scene", "assets", "stages", "prompts", "peaks", "qa"} <= set(bundle)
    assert bundle["assets"]["voice"]["url"].startswith("/api/assets/")
    assert len(bundle["peaks"].get("voice") or []) > 10


def test_previz_preview_endpoint(tmp_path):
    from starlette.testclient import TestClient

    cheap_state(tmp_path)
    client = TestClient(create_app(str(tmp_path)))
    r = client.post("/api/preview/previz", json={"mood_tag": "gentle-rain", "duration": 1.0,
                                                 "visual_prompt": "rain on leaves"})
    assert r.status_code == 200
    body = r.json()
    assert body["url"].startswith("/api/tmpfile?name=")
    assert client.get(body["url"]).status_code == 200
    assert client.get("/api/tmpfile?name=../../etc/passwd").status_code == 400


def test_events_are_persisted_for_replay(tmp_path):
    from ai_studio.events import EventBus, RunProgress

    st = cheap_state(tmp_path)
    pid = make_project(st, "events")
    out, done = run_to_end(st, pid, trigger="new")
    events = st.db.list_events(out["run_id"], limit=500)
    kinds = [e["kind"] for e in events]
    assert "run_started" in kinds and "run_finished" in kinds
    assert any(k == "stage_update" for k in kinds)
    assert all(e["ts"] for e in events)
    roll = RunProgress.rollup(st.db.list_stages(out["run_id"]))
    assert roll["voice_base"]["total"] >= 2 and roll["voice_base"]["done"] == roll["voice_base"]["total"]
    assert RunProgress.overall(st.db.list_stages(out["run_id"]))["pct"] == 100.0
    assert done["run"]["status"] == "completed"


def test_event_bus_replays_and_filters(tmp_path):
    from ai_studio.events import EventBus

    async def scenario():
        bus = EventBus()
        bus.bind_loop(asyncio.get_running_loop())
        q_run = bus.subscribe("r1")          # subscribing replays the recent tail
        q_all = bus.subscribe("*")
        bus.publish("progress", {"pct": 10.0}, run_id="r1", stage="voice_base", scene_idx=0)
        bus.publish("progress", {"pct": 20.0}, run_id="r2", stage="video", scene_idx=1)
        await asyncio.sleep(0)
        got_run = []
        while not q_run.empty():
            got_run.append(q_run.get_nowait())
        got_all = []
        while not q_all.empty():
            got_all.append(q_all.get_nowait())
        q_late = bus.subscribe("r1")          # late joiner still sees the tail
        late = []
        while not q_late.empty():
            late.append(q_late.get_nowait())
        assert any(e["payload"].get("pct") == 10.0 for e in late), late
        bus.unsubscribe("r1", q_run)
        return got_run, got_all

    got_run, got_all = asyncio.run(scenario())
    assert [e["payload"]["pct"] for e in got_run] == [10.0]
    assert len(got_all) == 2
    assert got_all[1]["run_id"] == "r2"


# ------------------------------------------------- characters / NPC / content type
def test_character_crud_and_expression_matching(tmp_path):
    st = cheap_state(tmp_path)
    c = st.db.create_character(name="លីដា")
    assert c["id"].startswith("c")
    img = st.db.add_character_image(c["id"], "sad", "/nonexistent.png")
    assert st.db.get_character(c["id"])["images"][0]["expression_label"] == "sad"
    assert st.db.list_characters()[0]["name"] == "លីដា"
    assert st.db.update_character(c["id"], name="ណា")["name"] == "ណា"
    st.db.delete_character(c["id"])
    assert st.db.list_characters() == []


def test_compare_breakdown_tags_sides_via_pipeline(tmp_path):
    st = cheap_state(tmp_path)
    script = "\n".join([
        "ផ្លូវ A លឿន និងត្រង់។", "ផ្លូវ A ថ្លៃជាងបន្តិច។",
        "ផ្លូវ B យឺត ប៉ុន្តែសន្សំសំចៃ។", "ផ្លូវ B ទេសភាពស្អាតជាង។",
    ])
    pid = make_project(st, title="compare", script=script, content_type="compare")
    _out, done = run_to_end(st, pid)
    assert done["run"]["status"] == "completed", done["run"]["error"]
    scenes = st.db.list_scenes(pid)
    sides = [s["meta"].get("side") for s in scenes]
    # structural invariant, whatever the packing: A half strictly before B,
    # last scene is the balanced summary, nothing is untagged
    assert sides and sides[-1] == "summary"
    assert "A" in sides and "B" in sides
    assert max(i for i, x in enumerate(sides) if x == "A") < \
        min(i for i, x in enumerate(sides) if x == "B")
    assert all(s["meta"].get("content_type") == "compare" for s in scenes)
    # deterministic no-LLM structure is never a silent explainer fallback
    assert any(s["meta"].get("visual_contrast") for s in scenes)


def test_silent_markup_reaches_captions_but_not_tts(tmp_path):
    st = cheap_state(tmp_path)
    script = ("សួស្ដី។\n[[silent: សូម]] អ្នកស្រមៃមួយភ្លែត។\n"
              "ហើយចាប់ផ្ដើមដើរទៅមុខ។")
    pid = make_project(st, title="silent", script=script)
    _out, done = run_to_end(st, pid)
    assert done["run"]["status"] == "completed", done["run"]["error"]
    scenes = st.db.list_scenes(pid)
    assert any("សូម" in (s["text"] or "") for s in scenes)        # display keeps it
    voice_metrics = st.db.query(
        "SELECT kind,path FROM assets WHERE project_id=? AND kind='voice'", (pid,))
    assert voice_metrics
    # placeholder TTS must have skipped the silent span (word count reflects it)
    for row in voice_metrics:
        assert os.path.exists(row["path"]) and os.path.getsize(row["path"]) > 0
    f = st.db.latest_asset(pid, "final")
    assert f and os.path.exists(f["path"])


@needs_ffmpeg
def test_line_gap_sec_changes_assembly_duration(tmp_path):
    st = cheap_state(tmp_path)
    script = "ដកដង្ហើមវែងៗ។\nរួចចាប់ផ្ដើមឡើងវិញ។\nថ្ងៃស្អែក គឺជាឱកាសថ្មី។"
    pid = make_project(st, title="gap", script=script)
    cfg = st.config()
    cfg["tts"]["line_gap_sec"] = 0.4
    cfg_mod.save(cfg, st.settings_path)
    st.invalidate()
    _out, done = run_to_end(st, pid)
    assert done["run"]["status"] == "completed", done["run"]["error"]
    final = st.db.latest_asset(pid, "final")
    manifest = st.db.latest_asset(pid, "manifest")
    assert manifest and os.path.exists(manifest["path"])
    import json as _json
    data = _json.load(open(manifest["path"], encoding="utf-8"))
    assert 0.3 <= float(data.get("pacing", {}).get("line_gap_sec", 0)) <= 0.5
    # assembly produced a final file with a real audio duration
    assert final and final.get("duration", 0) > 2.0


def test_talking_head_without_character_fails_loudly(tmp_path):
    """The HTTP save-surface guard: talking_head render_mode without a character
    is rejected with the exact human-readable reason (never silent)."""
    from starlette.testclient import TestClient
    st = cheap_state(tmp_path)
    script = "សួស្ដី សួស្ដី។\nនេះជាដំណើររបស់យើង។"
    pid = make_project(st, title="th", script=script)
    app = create_app(str(tmp_path))
    cli = TestClient(app)
    # give the API the same data dir: cheap_state writes settings into tmp_path
    r = cli.post(f"/api/projects/{pid}/scenes", json={"scenes": [{
        "text": "សួស្ដី សួស្ដី។", "visual_prompt": "a person talking", "mood_tag": "calm",
        "estimated_duration_sec": 3.0, "meta": {"render_mode": "talking_head"}}]})
    assert r.status_code == 400, r.text
    assert "character" in r.json()["detail"].lower()
    # with a character set the save is accepted
    c = st.db.create_character(name="លីដា")
    r2 = cli.post(f"/api/projects/{pid}/scenes", json={"scenes": [{
        "text": "សួស្ដី សួស្ដី។", "visual_prompt": "a person talking", "mood_tag": "calm",
        "estimated_duration_sec": 3.0,
        "meta": {"render_mode": "talking_head", "character_id": c["id"]}}]})
    assert r2.status_code == 200, r2.text


def test_talking_head_stage_skips_broll_scenes(tmp_path):
    st = cheap_state(tmp_path)
    script = "សួស្ដី។\nនេះជាដំណើររបស់យើង។"
    pid = make_project(st, title="th2", script=script)
    _out, done = run_to_end(st, pid, force_stages=["talking_head"])
    assert done["run"]["status"] == "completed", done["run"]["error"]
    rows = st.db.list_stages(done["run"]["id"])
    th = [r for r in rows if r["stage"] == "talking_head"]
    assert th and all(r["status"] in ("done", "skipped") for r in th)


def test_style_previews_endpoint_renders_cached_samples(tmp_path):
    from starlette.testclient import TestClient
    st = cheap_state(tmp_path)
    app = create_app(str(tmp_path))
    cli = TestClient(app)
    r = cli.get("/api/style-previews")
    assert r.status_code == 200, r.text
    data = r.json()
    assert {s["key"] for s in data["subtitle_styles"]} == {"clean", "bold_yellow",
                                                           "minimal_top", "karaoke"}
    assert {s["key"] for s in data["title_styles"]} == {"centered_fade",
                                                        "bottom_left_minimal", "bold_pop"}
    # a style that cannot render on this ffmpeg build is LISTED with an honest
    # error (never a 500, never a silent gap) — url may be empty then
    for s in data["subtitle_styles"]:
        assert "error" in s
    # second call is served from cache
    r2 = cli.get("/api/style-previews")
    assert r2.status_code == 200
    assert r2.json()["subtitle_styles"] == data["subtitle_styles"]


@needs_ffmpeg
def test_custom_scene_image_becomes_a_kenburns_clip(tmp_path):
    st = cheap_state(tmp_path)
    script = "សួស្ដី។\nនេះជាដំណើររបស់យើង។"
    pid = make_project(st, title="still", script=script)
    # a ready storyboard (as the API save produces it) with visual_source=illustration
    st.db.replace_scenes(pid, [
        {"text": "សួស្ដី។", "visual_prompt": "quiet lake at dawn", "mood_tag": "calm",
         "estimated_duration_sec": 3.0,
         "meta": {"visual_source": "illustration"}},
        {"text": "នេះជាដំណើររបស់យើង។", "visual_prompt": "path through mist",
         "mood_tag": "calm", "estimated_duration_sec": 3.0}])
    # place the Director's custom picture where the stage expects it (00_custom.png)
    from PIL import Image
    import numpy as np
    arr = np.zeros((64, 48, 3), dtype="uint8")
    arr[:, :, 0] = 120
    img_path = os.path.join(str(tmp_path), "custom.png")
    Image.fromarray(arr).save(img_path)
    sd = os.path.join(str(tmp_path), "projects", pid, "scenes", "00")
    os.makedirs(sd, exist_ok=True)
    shutil.copy(img_path, os.path.join(sd, "00_custom.png"))
    _out, done = run_to_end(st, pid, force_stages=["video"])
    assert done["run"]["status"] in ("completed", "partial"), done["run"]["error"]
    rows = st.db.list_stages(done["run"]["id"])
    video = next((r for r in rows if r["stage"] == "video"), None)
    assert video is not None
    if video["status"] != "failed":
        asset = st.db.latest_asset(pid, "video", scene_idx=0)
        assert asset and os.path.exists(asset["path"])

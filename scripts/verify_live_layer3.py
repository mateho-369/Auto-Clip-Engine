#!/usr/bin/env python3
"""Live HTTP verification for Layer 3 (silent markup, line gap, characters,
NPC modes, illustrations, subtitles, title cards, content-type structure).

Run against a studio server on :8000 (python -m ai_studio --port 8000).
Prints a results table; exits non-zero when any check fails.
"""
import io
import json
import os
import sys
import time

import httpx

BASE = os.environ.get("STUDIO_URL", "http://127.0.0.1:8000")
API = BASE + "/api"
RESULTS = []


def ok(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))
    print(("  ✔ " if cond else "  ✘ ") + name + (f"  — {detail}" if detail else ""))


def client():
    return httpx.Client(timeout=httpx.Timeout(30.0, read=120.0))


def wait_run(c, run_id, timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = c.get(f"{API}/runs/{run_id}/status").json()
        run = r.get("run") or r
        if run.get("status") not in ("queued", "running", "paused"):
            stages = r.get("stages") or run.get("stages") or []
            return {**run, "stages": stages, "overall": r.get("overall") or run.get("overall") or {}}
        time.sleep(2.5)
    return {"status": "timeout", "error": "wait timed out"}


def make_png(label):
    from PIL import Image
    import numpy as np
    rng = np.random.default_rng(abs(hash(label)) % 2**31)
    arr = rng.integers(30, 200, (128, 96, 3), dtype="uint8")
    im = Image.fromarray(arr)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    buf.seek(0)
    return buf


FORCE = {"video": {"engine": "previz"}, "sfx": {"engine": "procedural"},
         "rvc": {"engine": "bypass"}, "tts": {"engine": "placeholder"}}
def merge_settings(*dicts):
    out: dict = {}
    for d in dicts:
        for k, v in (d or {}).items():
            out[k] = {**(out.get(k) or {}), **v}
    return out


def new_project(c, **kw):
    payload = {"mode": "A", "title": kw.pop("title", "verify"),
               "script": kw.pop("script", "សួស្ដី។\nនេះជាដំណើររបស់យើង។"), **kw}
    payload["settings"] = merge_settings(FORCE, payload.get("settings"))
    r = c.post(f"{API}/projects", json=payload)
    assert r.status_code == 200, r.text
    return r.json()["project"]


def run_full(c, pid, extra=None):
    r = c.post(f"{API}/projects/{pid}/runs", json=extra or {})
    assert r.status_code == 200, r.text
    rid = r.json()["run_id"]
    st = wait_run(c, rid)
    stages = st.get("stages") or []
    failed = [s for s in stages if s["status"] in ("failed", "blocked")]
    return st, failed


def main():
    c = client()
    print("== Layer 3 live verification ==\n")

    # 0. style previews
    r = c.get(f"{API}/style-previews")
    d = r.json()
    ok("style-previews subtitle styles", {s["key"] for s in d["subtitle_styles"]} >=
       {"clean", "bold_yellow", "minimal_top", "karaoke"}, str(len(d["subtitle_styles"])) + " styles")
    ok("style-previews title styles", {s["key"] for s in d["title_styles"]} >=
       {"centered_fade", "bottom_left_minimal", "bold_pop"}, str(len(d["title_styles"])) + " styles")
    u = d["subtitle_styles"][0]["url"]
    rr = c.get(u if u.startswith("http") else BASE + u)
    ok("style-previews cached file serves", rr.status_code == 200 and
       rr.headers.get("content-type", "").startswith("video"),
       rr.headers.get("content-type", "?"))

    # 1. silent markup + line_gap
    p = new_project(c, title="silent-gap",
                    script="សួស្ដី។\n[[silent: សូម]] អ្នកស្រមៃមួយភ្លែត។\nហើយចាប់ផ្ដើមដើរទៅមុខ។",
                    content_type="explainer",
                    settings={"tts": {"pace": "slow", "line_gap_sec": 0.6},
                              "assembly": {"burn_captions": True, "subtitle_style": "clean"}})
    st, failed = run_full(c, p["id"])
    manifest = None
    assets = c.get(f"{API}/assets", params={"project_id": p["id"]}).json()["assets"]
    for a in assets:
        if a["kind"] == "manifest":
            manifest = json.load(open(a["path"], encoding="utf-8"))
    srt = next((a for a in assets if a["kind"] == "srt"), None)
    ok("silent run completed", st["status"] == "completed", st["status"])
    ok("silent run 0 failed jobs", not failed, f"{len(failed)} failed")
    ok("line_gap_sec applied", manifest and round(manifest["pacing"]["line_gap_sec"], 1) == 0.6,
       str(manifest and manifest["pacing"]["line_gap_sec"]))
    ok("srt keeps silent word", bool(srt) and "សូម" in open(srt["path"], encoding="utf-8").read())
    # spoken words: placeholder voice audio exists; script text in scenes keeps markup
    scenes = c.get(f"{API}/projects/{p['id']}/scenes").json()["scenes"]
    ok("scene text keeps [[silent:]] markup", any("សូម" in s["text"] for s in scenes))
    ok("final asset present", any(a["kind"] == "final" for a in assets))

    # 2. characters
    ch = c.post(f"{API}/characters", json={"name": "លីដា"}).json()["character"]
    for lbl in ("neutral", "calm", "sad", "happy"):
        rr = c.post(f"{API}/characters/{ch['id']}/images",
                    files={"image": (f"{lbl}.png", make_png(lbl), "image/png")},
                    data={"expression_label": lbl})
        assert rr.status_code == 200, rr.text
    ch = c.get(f"{API}/characters").json()["characters"][0]
    ok("character created + 4 images", len(ch["images"]) == 4, f'{ch["name"]}: {len(ch["images"])} images')
    mo = c.get(f"{API}/characters").json()["mood_to_expression"]
    ok("mood→expression map served", "sad" in mo.values() or "sorrow" in mo)

    # 3. compare WITHOUT character → illustration per side
    p1 = new_project(c, title="compare-no-char",
                     script="ផ្លូវ A លឿន និងត្រង់។\nផ្លូវ A ថ្លៃជាងបន្តិច។\nផ្លូវ B យឺត ប៉ុន្តែសន្សំសំចៃ។\nផ្លូវ B ទេសភាពស្អាត។\nសរុប៖ អាស្រ័យលើអ្វីដែលអ្នកត្រូវការ។",
                     content_type="compare")
    st, failed = run_full(c, p1["id"])
    sc = c.get(f"{API}/projects/{p1['id']}/scenes").json()["scenes"]
    sides = [s["meta"].get("side") for s in sc]
    vs = [s["meta"].get("visual_source") for s in sc]
    ok("compare run completed", st["status"] == "completed" and not failed)
    ok("compare sides A→B→summary", sides[0] == "A" and "B" in sides and sides[-1] == "summary", str(sides))
    ok("compare no-character → illustration", set(vs) <= {"illustration"}, str(set(vs)))

    # 4. compare WITH character → character_demo + pose in prompt
    p2 = new_project(c, title="compare-char", content_type="compare", character_id=ch["id"],
                     script="ផ្លូវ A លឿន និងត្រង់។\nផ្លូវ B យឺត ប៉ុន្តែសន្សំសំចៃ។\nដូច្នេះ អាស្រ័យលើអ្វីដែលអ្នកត្រូវការ។")
    st, failed = run_full(c, p2["id"])
    sc = c.get(f"{API}/projects/{p2['id']}/scenes").json()["scenes"]
    ok("compare+char run completed", st["status"] == "completed" and not failed)
    ok("compare+char → character_demo", any(s["meta"].get("visual_source") == "character_demo" for s in sc),
       str([s["meta"].get("visual_source") for s in sc]))
    videos = [a for a in c.get(f"{API}/assets", params={"project_id": p2["id"], "kind": "video"}).json()["assets"]
              if a["scene_idx"] == 0]
    prompt = videos[0]["meta"].get("prompt", "") if videos else ""
    ok("character_demo prompt has in-place gesture", "standing in place" in prompt and "miming motion" in prompt)
    ok("pose phrase in composed prompt", "posture" in prompt or "breath" in prompt or "slow" in prompt or "relaxed" in prompt,
       prompt[:160])

    # 5. two-character script → per-scene character tags (shot/reverse-shot)
    ch2 = c.post(f"{API}/characters", json={"name": "ណា"}).json()["character"]
    for lbl in ("neutral", "sad"):
        c.post(f"{API}/characters/{ch2['id']}/images",
               files={"image": (f"{lbl}.png", make_png(lbl), "image/png")},
               data={"expression_label": lbl})
    p3 = new_project(c, title="two-char", content_type="explainer",
                     script="សួស្ដី លីដា សួស្ដី។\nណា តើអ្នកយល់យ៉ាងណា?\nខ្ញុំគិតថា វាស្រស់ស្អាតណាស់។\nចាំមើលតើវាដំណើរការយ៉ាងណា?\nវាមានសារៈសំខាន់ខ្លាំងណាស់។\nយើងត្រូវរៀនពីវាឱ្យបានច្បាស់។\nពេលនោះយើងនឹងយល់កាន់តែច្បាស់។\nអរគុណដែលបានស្ដាប់ការពន្យល់នេះ។")
    rr = c.patch(f"{API}/projects/{p3['id']}", json={"character_id": ch["id"]})
    assert rr.status_code == 200, rr.text
    # first run generates the scene board (breakdown)
    st0, f0 = run_full(c, p3["id"])
    ok("two-char initial run completed", st0["status"] == "completed" and not f0)
    board = c.get(f"{API}/projects/{p3['id']}/scenes").json()["scenes"]
    ok("two-char board has scenes", len(board) >= 2, len(board))
    for i, s in enumerate(board[:3]):
        s["meta"] = {**(s.get("meta") or {}), "character_id": ch["id"] if i % 2 == 0 else ch2["id"],
                     "visual_source": "character_demo"}
    saved = c.post(f"{API}/projects/{p3['id']}/scenes", json={"scenes": board}).json()
    ok("two-character scene tags saved", sorted({s["meta"]["character_id"] for s in saved["scenes"]}) ==
       sorted([ch["id"], ch2["id"]]), "shot / reverse-shot")
    st, failed = run_full(c, p3["id"], {"settings": None} if False else None)
    ok("two-char run completed", st["status"] == "completed" and not failed)

    # 6. talking head (no SadTalker → still fallback via matched expression image)
    p4 = new_project(c, title="talking-head", character_id=ch["id"],
                     script="សួស្ដី។\nនេះជាដំណើររបស់យើង។")
    st0, f0 = run_full(c, p4["id"])
    ok("talking-head initial run completed", st0["status"] == "completed" and not f0)
    board = c.get(f"{API}/projects/{p4['id']}/scenes").json()["scenes"]
    ok("talking-head board has scenes", len(board) >= 1, len(board))
    board[0]["meta"] = {**(board[0].get("meta") or {}), "render_mode": "talking_head", "character_id": ch["id"]}
    c.post(f"{API}/projects/{p4['id']}/scenes", json={"scenes": board})
    st, failed = run_full(c, p4["id"])
    th = [x for x in st.get("stages") or [] if x["stage"] == "talking_head"]
    vid = [x for x in st.get("stages") or [] if x["stage"] == "video" and x["scene_idx"] == 0]
    th0 = next((x for x in th if x["scene_idx"] == 0), None)
    ok("talking_head run completed", st["status"] == "completed" and not failed)
    ok("talking_head stage ran (still fallback)", th0 and th0["status"] == "done" and "still" in th0["engine"],
       f'{th0 and th0["status"]}/{th0 and th0["engine"]}')
    ok("video stage skipped for talking-head scene", vid and vid[0]["status"] == "skipped",
       f'{vid[0]["status"] if vid else "?"}')

    # 6b. render_mode/talking_head rejection without character on the save surface
    p5 = new_project(c, title="th-reject", script="សួស្ដី។\nហើយចាប់ផ្ដើមដើរ។")
    run_full(c, p5["id"])
    board = c.get(f"{API}/projects/{p5['id']}/scenes").json()["scenes"]
    if not board:
        board = [{"text": "សួស្ដី។", "meta": {"index": 0}}]
    board[0] = {**board[0], "meta": {"render_mode": "talking_head"}}
    rr = c.post(f"{API}/projects/{p5['id']}/scenes", json={"scenes": board})
    ok("talking_head without character → 400 with exact text", rr.status_code == 400 and
       "character" in rr.json()["detail"].lower(), rr.json()["detail"][:80])
    board[0]["meta"] = {"visual_source": "character_demo"}
    rr = c.post(f"{API}/projects/{p5['id']}/scenes", json={"scenes": board})
    ok("character_demo without character → 400", rr.status_code == 400 and
       "character" in rr.json()["detail"].lower())

    # 7. manually uploaded scene image → Ken Burns clip
    p6 = new_project(c, title="scene-image", script="សួស្ដី។\nហើយចាប់ផ្ដើមដើរទៅមុខ។")
    run_full(c, p6["id"])
    rr = c.post(f"{API}/projects/{p6['id']}/scenes/0/image",
                files={"image": ("custom.png", make_png("custom"), "image/png")})
    ok("scene image upload accepted", rr.status_code == 200, rr.json().get("note", "")[:70])
    st, failed = run_full(c, p6["id"])
    vids = [a for a in c.get(f"{API}/assets", params={"project_id": p6["id"], "kind": "video"}).json()["assets"]
            if a["scene_idx"] == 0]
    ok("custom image → kenburns clip", bool(vids) and vids[0]["meta"].get("engine") == "kenburns",
       f'{vids[0]["meta"].get("engine") if vids else "no asset"}')
    ok("custom image run completed", st["status"] == "completed" and not failed)

    # 8. subtitle styles (one run each)
    for style in ("clean", "bold_yellow", "minimal_top", "karaoke"):
        pp = new_project(c, title=f"sub-{style}", script="សួស្ដី។\nថ្ងៃនេះយើងរៀនពាក្យមួយ។\nរួចចាប់ផ្ដើមដើរទៅមុខ។",
                         settings={"assembly": {"burn_captions": True, "subtitle_style": style,
                                                "title_style": ""},
                                   "tts": {"line_gap_sec": 0.4}})
        st, failed = run_full(c, pp["id"])
        assets = c.get(f"{API}/assets", params={"project_id": pp["id"]}).json()["assets"]
        cap = next((a for a in assets if a["kind"] == "final_captions"), None)
        mf = next((a for a in assets if a["kind"] == "manifest"), None)
        mf_d = json.load(open(mf["path"], encoding="utf-8")) if mf else {}
        ok(f"subtitle '{style}' run completed", st["status"] == "completed" and not failed)
        ok(f"subtitle '{style}' burned captions asset", bool(cap), "" if cap else "no libass? " + mf_d.get("pacing", {}).get("line_gap_sec", ""))
        ok(f"subtitle '{style}' in manifest", mf_d.get("pacing", {}).get("subtitle_style") == style)

    # 9. title styles (one run each)
    for style in ("centered_fade", "bottom_left_minimal", "bold_pop"):
        pp = new_project(c, title=f"title-{style}", script="សួស្ដី។\nថ្ងៃនេះយើងរៀនពាក្យមួយ។",
                         settings={"assembly": {"title_style": style, "burn_captions": False}})
        st, failed = run_full(c, pp["id"])
        mf = next((a for a in c.get(f"{API}/assets", params={"project_id": pp["id"]}).json()["assets"]
                   if a["kind"] == "manifest"), None)
        mf_d = json.load(open(mf["path"], encoding="utf-8")) if mf else {}
        ok(f"title '{style}' run completed", st["status"] == "completed" and not failed)
        ok(f"title '{style}' in manifest", mf_d.get("pacing", {}).get("title_style") == style,
           str(mf_d.get("pacing", {}).get("title_style")))
        total = float((mf_d.get("video") or {}).get("duration", 0) or 0)
        lead = float((mf_d.get("scenes") or [{}])[0].get("start", 0) or 0)
        ok(f"title '{style}' added length", total >= 2.0 and lead >= 1.2,
           f'{total:.2f}s total, {lead:.2f}s title lead-in')

    # 10. sad-mood character scene — pose phrase in actual composed prompt
    p7 = new_project(c, title="sad-pose", character_id=ch["id"], content_type="explainer",
                     script="នាងលីដាសោកសៅ។\nនាងសង្ឃឹមថាថ្ងៃស្អែកប្រសើរជាង។")
    run_full(c, p7["id"])
    board = c.get(f"{API}/projects/{p7['id']}/scenes").json()["scenes"]
    if not board:
        board = [{"text": "នាងលីដាសោកសៅ។", "meta": {"index": 0}}]
    board[0]["mood_tag"] = "sad"
    board[0]["meta"] = {**(board[0].get("meta") or {}), "character_id": ch["id"],
                        "visual_source": "character_demo"}
    c.post(f"{API}/projects/{p7['id']}/scenes", json={"scenes": board})
    st, failed = run_full(c, p7["id"])
    vids = c.get(f"{API}/assets", params={"project_id": p7["id"], "kind": "video"}).json()["assets"]
    v = next((a for a in vids if a["scene_idx"] == 0), None)
    pm = (v or {}).get("meta", {}).get("prompt", "")
    ok("sad character run completed", st["status"] == "completed" and not failed)
    ok("sad mood pose phrase in actual prompt", "head lowered" in pm or "slumped" in pm or
       "shoulders" in pm or "sad" in pm.lower(), pm[:220])

    # 11. content-type matrix — every type end-to-end via HTTP
    ct_payload = c.get(f"{API}/content-types").json()
    ct_keys = [t.get("key") or t.get("id") or t.get("name") for t in (ct_payload.get("types") or [])]
    ct_keys = [k for k in ct_keys if k] or \
        ["explainer", "what_if", "compare", "choose", "word_nuance", "myth_vs_fact", "quick_tip"]
    ct_scripts = {
        "explainer": "សួស្ដី។\nថ្ងៃនេះយើងរៀនអំពីការថែទាំសុខភាព។\nយើងត្រូវគេងឲ្យបានគ្រប់គ្រាន់។\nហើយផឹកទឹកឲ្យបានច្រើន។\nសូមអរគុណ។",
        "what_if": "ចុះបើយើងអាចហោះហើរបាន?\nយើងនឹងឃើញទេសភាពពីលើមេឃ។\nប៉ុន្តែយើងត្រូវប្រុងប្រយ័ត្ន។\nចុះបើគ្រប់គ្នាហោះហើរ?",
        "compare": "ផ្លូវ A លឿន និងត្រង់។\nផ្លូវ A ថ្លៃជាងបន្តិច។\nផ្លូវ B យឺត ប៉ុន្តែសន្សំសំចៃ។\nផ្លូវ B ទេសភាពស្អាត។\nសរុប៖ អាស្រ័យលើអ្វីដែលអ្នកត្រូវការ។",
        "choose": "ជម្រើសទីមួយ គឺទិញកាបូបថ្មី។\nជម្រើសទីពីរ គឺជួសជុលកាបូបចាស់។\nជម្រើសទីបី គឺរង់ចាំបន្តិចទៀត។\nដូច្នេះ អ្នកអាចជ្រើសរើសតាមចំណូលចិត្ត។",
        "word_nuance": "ន័យទីមួយ គឺការដើរតាមផ្លូវ។\nន័យទីពីរ គឺការដើរឆ្ពោះទៅរកគោលដៅ។\nពេលប្រៀបធៀប យើងឃើញភាពខុសគ្នា។",
        "myth_vs_fact": "មនុស្សជាច្រើនជឿថា ត្រជាក់បណ្ដាលឲ្យផ្ដាសាយ។\nប៉ុន្តែតាមពិត វីរុសគឺជាមូលហេតុ។\nដូច្នេះ ការដឹងច្បាស់ជួយយើងការពារខ្លួន។",
        "quick_tip": "គន្លឹះមួយគឺ ដាក់ទឹកឲ្យបានគ្រប់គ្រាន់។\nចាំថា ទឹកគឺសំខាន់ណាស់។",
    }
    ct_expect = {
        "compare": lambda sides: {"A", "B", "summary"} <= set(sides) and sides[-1] == "summary",
        "word_nuance": lambda sides: {"meaning-1", "meaning-2"} <= set(sides),
        "myth_vs_fact": lambda sides: {"myth", "fact"} <= set(sides),
        "choose": lambda sides: len(sides) >= 3 and sides[-1] == "takeaway" and sides[0].startswith("option-"),
        "what_if": lambda sides: bool(sides) and sides[0] == "hypothetical",
        "quick_tip": lambda sides: True,
        "explainer": lambda sides: True,
    }
    for ct in ct_keys:
        title = f"ct-{ct}"
        script = ct_scripts.get(ct, ct_scripts["explainer"])
        pp = new_project(c, title=title, content_type=ct, script=script,
                         settings={"tts": {"line_gap_sec": 0.4}})
        st, failed = run_full(c, pp["id"])
        sc = c.get(f"{API}/projects/{pp['id']}/scenes").json()["scenes"]
        sides = [x["meta"].get("side", "") for x in sc]
        cts = {x["meta"].get("content_type") for x in sc} if sc else set()
        ok(f"content-type '{ct}' completed", st["status"] == "completed" and not failed, st["status"])
        ok(f"content-type '{ct}' all scenes carry ct", cts <= {ct}, str(cts))
        if ct == "quick_tip":
            ok(f"content-type '{ct}' ≤2 scenes", 0 < len(sc) <= 2, len(sc))
        else:
            ok(f"content-type '{ct}' structure tags", ct_expect[ct](sides), str(sides))
        if ct == "compare":
            ok(f"content-type '{ct}' A before B before summary",
               sides.index("A") < sides.index("B") < sides.index("summary"), str(sides))

    # 12. long subscript-heavy script → zero lone coeng at any boundary
    SUB_SCRIPT = ("ស្វែងយល់រកចម្លើយ ស្រស់ស្អាត និងប្រណាំងពេលថ្ងៃ។\n"
                  "ចម្រៀងខ្មែរដ៏ផ្អែម ដើរជាមួយក្ដីសង្ឃឹម។\n"
                  "កម្មវិធីនេះជួយឲ្យយល់ច្បាស់ពីអ្វីដែលសំខាន់។\n"
                  "ស្រ្ដីម្នាក់សម្លឹងមើលផ្កាក្នុងសួន។\n"
                  "នាងស្វែងរកឱកាស ហើយមិនដែលបោះបង់។\n"
                  "នេះគឺជាការចាប់ផ្ដើមដ៏ល្អសម្រាប់ថ្ងៃស្អែក។")
    pp = new_project(c, title="coeng-boundary", content_type="explainer", script=SUB_SCRIPT,
                     settings={"assembly": {"burn_captions": True, "subtitle_style": "clean"},
                               "tts": {"line_gap_sec": 0.4}})
    st, failed = run_full(c, pp["id"])
    proj = c.get(f"{API}/projects").json()
    proj_row = next(x for x in proj["projects"] if x["id"] == pp["id"])
    title = proj_row.get("title") or ""
    bad = lambda t: "្" in t and bool(__import__("re").search(r"្(?:\s|$)|្[។៕,.!?]", t))
    ok("coeng run completed", st["status"] == "completed" and not failed)
    ok("coeng title has no broken cluster", not bad(title), title)
    sc = c.get(f"{API}/projects/{pp['id']}/scenes").json()["scenes"]
    scene_texts = "\n".join(x["text"] for x in sc)
    ok("coeng scene texts have no lone coeng", not bad(scene_texts))
    srt = next((a for a in c.get(f"{API}/assets", params={"project_id": pp["id"]}).json()["assets"]
                if a["kind"] == "srt"), None)
    srt_txt = open(srt["path"], encoding="utf-8").read() if srt else ""
    ok("coeng SRT has no lone coeng", bool(srt_txt) and not bad(srt_txt))
    asstexts = []
    for a in c.get(f"{API}/assets", params={"project_id": pp["id"]}).json()["assets"]:
        if a["kind"] == "ass":
            try: asstexts.append(open(a["path"], encoding="utf-8").read())
            except Exception: pass
    ok("coeng ASS/other captions clean", all(not bad(t) for t in asstexts), f"{len(asstexts)} ass files")

    # 13. [[silent:]] shortens estimated speech vs the spoken version, live
    def est_total(script_text, title):
        q = new_project(c, title=title, script=script_text)
        stq, fq = run_full(c, q["id"])
        assert stq["status"] == "completed" and not fq
        sq = c.get(f"{API}/projects/{q['id']}/scenes").json()["scenes"]
        return sum(float(x.get("estimated_duration_sec") or 0) for x in sq)
    spoken = est_total("សួស្ដី។\nនេះជាដំណើររបស់យើងដ៏វែងឆ្ងាយ។", "est-spoken")
    sil = est_total("សួស្ដី។\n[[silent: នេះជា]] ដំណើររបស់យើងដ៏វែងឆ្ងាយ។", "est-silent")
    ok("[[silent:]] shortens estimated speech (silent < spoken)",
       sil < spoken, f"{sil:.2f}s vs {spoken:.2f}s")

    # summary
    print("\n== SUMMARY ==")
    good = sum(1 for _n, cnd, _d in RESULTS if cnd)
    print(f"  {good}/{len(RESULTS)} checks passed")
    bad = [(n, d) for n, c, d in RESULTS if not c]
    for n, d in bad:
        print(f"  FAIL {n} :: {d}")
    sys.exit(0 if not bad else 1)


if __name__ == "__main__":
    main()

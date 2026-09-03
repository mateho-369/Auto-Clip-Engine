# 🎬 Khmer AI Content Studio

**Director-led, multi-agent, 100 % local video pipeline.** You give it either a finished
Khmer script (*Mode A*) or just a topic (*Mode B*); it breaks the script into scenes,
speaks it in Khmer, turns your own trained voice into the timbre, animates 480p footage,
lays calm nature ambience under the narration, QA-checks every scene, and stitches a
vertical `.mp4` — with a live per-stage stepper you can inspect, retry, or re-run one
stage at a time.

No cloud API, no subscription, nothing leaves the machine. The whole studio — database,
media, prompts, settings — is one folder (`data/studio/`) you can copy to another PC.

```
python -m ai_studio --check     # what is installed, what is missing, and how to fix it
python -m ai_studio --demo      # http://localhost:8000 with three sample projects
```

> This is the "v4" of the repo's agent system: it reuses `ai_creator/`'s role→model
> mapping and JSON-extraction helpers, but replaces its linear script with a real async
> job graph. The older studio (`ai_creator/`, also port 8000) and this one can't run at
> the same time on the default port — give one of them `--port 8002`.

---

## 1 · The pipeline

| # | Stage (UI stepper) | Model / tool | Runs on | If it is missing |
|---|---|---|---|---|
| 2 | 📝 Script ready | Ollama `sailor2:8b` → fallback `llama3.2:3b` | CPU/GPU (LLM) | template script in the house style (Mode B only) |
| 1 | 🧩 Scene breakdown | same LLM, JSON-constrained | CPU/GPU (LLM) | deterministic sentence splitter |
| 3a | 🎙️ Khmer voice | `sherpa-onnx` + `vits-mms-khm` | CPU | syllable-timed placeholder tone (clearly flagged) |
| 3b | 🎚️ Your timbre | RVC WebUI / `RVC_CLI` (your trained model) | GPU (or CPU) | base voice passes through (bypass) |
| 4 | 🎞️ Animator | ComfyUI + **Wan2.1-T2V-1.3B** (or Wan2.2-TI2V-5B) | GPU, 8 GB-tamed | procedural *previz* clip (real animation, soft motion, mood-tinted) |
| 4b | ⏱️ Duration match | ffmpeg | CPU | — (always available) |
| 5 | 🔊 SFX Director | ComfyUI + **MMAudio** (small 8 GB variant) | GPU | procedural ambience layered from the mood tag |
| 6 | ✅ QA Reviewer | Ollama (`sailor2:8b`) | CPU/GPU (LLM) | heuristic checks (duration, pacing, missing media) |
| 7 | 🎬 Final assembly | ffmpeg (`libx264` + `aac`, loudness-normalised) | CPU | needs ffmpeg; a run without it reports the reason |

The dependency graph (each scene gets its own row of jobs):

```
script ──► breakdown ──► per scene i:
                          voice_base:i ──► voice_final:i ─┐
                          video:i ────────────────────────┼─► video_fit:i ─┐
                          video:i ──► sfx:i ─────────────────────────────┤
                                                          └──────────► qa:i ─┴─► assemble
```

`voice_base:i` and `video:i` are **deliberately parallel** — narration timing is what the
picture must match, so video starts as soon as the scene text exists and is trimmed/
looped to the *final* voice length in 4b. `qa:i` judges the whole scene, and `assemble`
only waits on QA, so one slow scene never blocks the others' QA.

**Retries, failures, resume.** Every stage gets one automatic retry (`pipeline.retry_limit`)
with the specific engine error recorded — e.g.

```
video#3 · [🎞️ 4 · Animator (Wan)] RuntimeError: ComfyUI refused the prompt (node_errors: MissingInputType)
```

A failed stage does **not** abort the run: its dependents either fail with their own clear
reason (`no video clip to fit`) or continue with what exists, and the run ends `partial`
with every failure listed. Then **Resume from last success** (or `POST /api/projects/{id}/runs
{"resume_from": "<run>"}`) reuses every already-finished stage — the DB rows for them are
copied over with `inherited_from`, so nothing is re-synthesised or re-rendered. Per-stage
regeneration is the same mechanism with one stage forced: `POST /api/runs/{id}/stages/{stage}/regenerate`.

---

## 2 · Two ways in

**Mode A — Director (script is ground truth).** You paste the Khmer script. It is stored
locked and *only* mechanically segmented: no agent may rewrite wording, tone, or order.
The Controller assigns each scene a visual prompt, a mood tag, and an estimated spoken
duration — never new sentences. If a stage's output drifts from your text (a dropped
particle, an added sentence), the integrity gate **restores your wording** and reports it;
`PATCH /api/projects/{id}` refuses script edits unless you send `director_override: true`.
The project page shows the integrity verdict (`ok / restored / verified`) next to the script.

**Mode B — Auto idea.** You give a topic hint (or nothing) and the Controller writes the
whole script in the house style. Depending on `pipeline.review_gate` (`auto|always|never`)
the run pauses after Stage 2 with **"waiting for the Director to approve the script"** —
approve, edit, or regenerate before any GPU time is spent. Tick *full autonomy*
(`auto_approve_mode_b`) to skip the gate.

The mode is stored on every project record and shown in the dashboard, because it changes
what "the AI changed my text" means.

---

## 3 · House style (fixed, not negotiable)

Every generated or tagged line is written under one guideline (`ai_studio/style.py`,
editable per project in the UI, visible any time at `GET /api/style`):

* calm, warm, gentle Khmer; a caring older sibling, never a lecturer;
* positive and life-affirming — "don't give up" — no shame, no fear-mongering;
* no emoji, no "subscribe/follow", no engagement bait, no politics/religion;
* visuals and SFX lean peaceful nature: soft light, water, birds, gentle motion,
  sunrise/dusk warmth (mood tags map to ambience recipes in `style.MOOD_MAP`).

The QA reviewer judges against the same guideline, so "off-tone" is a real, reportable
issue rather than a vibe.

---

## 4 · The two machines (the hard constraint)

| | Machine A | Machine B |
|---|---|---|
| CPU | Ryzen 9 (12C/24T) | Ryzen 5 6600H |
| GPU | **RTX 5070 — 8 GB VRAM** | AMD iGPU only (no CUDA) |
| RAM | 16 GB | 16 GB |
| Script / QA (LLM) | `sailor2:8b` @ Q4 via Ollama | `llama3.2:3b` on CPU |
| Khmer voice (3a) | ✅ sherpa-onnx (CPU, ~real-time) | ✅ same |
| RVC timbre (3b) | ✅ GPU | ⚠️ CPU (slow, works) or bypass |
| Video (4) | ✅ Wan 1.3B @ 480p / Wan2.2-5B with offloading | ❌ **deferred** |
| MMAudio (5) | ✅ small 44 k variant | ❌ **deferred** |
| Assembly (7) | ✅ ffmpeg | ✅ ffmpeg |

### What "8 GB" means here

Nothing may *need* more than 8 GB at inference, so the studio enforces three things:

1. **One GPU model resident at a time** (`vram.serialize_gpu`, default on): the RVC pass,
   the Wan render and the MMAudio pass are serialised through a single semaphore, and
   `keep_alive: "0"` unloads the LLM after each stage instead of squatting on VRAM.
2. **A megapixel-frame budget** (`ai_studio/vram.py`): `42 Mpx·f` per 8 GB
   (`~0.88 GB/GB`), so a request is scaled down *before* submission —
   `480×854×81 = 33.2 Mpx·f` is inside the house tier and untouched; asking for
   `1080p×121` gets you a smaller clip plus a note in the stage log
   (`vram: 1080×1920×121 → 480×854×81 (budget 42 Mpx·f/8GB)`), never a CUDA OOM.
   A hard floor of 34 frames keeps motion coherent.
3. **A free-VRAM reservation** (`vram.reserve_free_mb: 900`): if the card reports less
   free memory than that, the stage defers to its CPU fallback and says so.

`machine.profile` (`auto | machine_a | machine_b`) sets the policy; `machine_a` also caps
`vram.limit_mb` at 8192 so a mis-read "16 GB total" can never license an 16 GB allocation.

### Machine B: script + voice now, picture later

With `--machine machine_b` (or auto-detected) the video and SFX stages resolve to `defer`:
they stay in the graph, marked deferred, and **Stages 3a/3b, QA and assembly still run** so
you get a complete narrated draft (previz pictures, or none if you set `video.engine=off`).
Then, on Machine A:

1. copy the project folder — `data/studio/projects/<id>/` plus `data/studio/studio.db`
   (or `GET /api/projects/{id}/export` → zip, and `POST` the same zip shape on the other box);
2. open the project and press **Finish the deferred stages**, or
   `POST /api/projects/{id}/catchup` (no body) — it reads the last run's `deferred` rows and returns `{"run_id", "jobs", "inherited", "deferred_stages"}`;
3. download the final `.mp4` from Machine A and keep working there.

Same DB, same project, real render — because every prompt the CPU pass used is recorded
(see §7), Machine A's run reuses your exact wording and seeds unless you change them.

---

## 5 · Install

### 0 · One command

```bash
./setup-studio.sh              # venv + deps + folders + readiness report
./setup-studio.sh --with-tts   # …and convert the Khmer voice now
```
```powershell
.\setup-studio.ps1             # Windows / Machine A
.\setup-studio.ps1 -WithTts -WithTests
```

Both stop short of installing the big models on purpose — those are separate services
below, and you want to see exactly what you're pulling.

### 1 · Ollama (Stages 1, 2, 6)

```powershell
winget install Ollama.Ollama
ollama pull sailor2:8b        # 4.9 GB · Apache-2.0 · Qwen2.5-based · 15 languages incl. Khmer
ollama pull llama3.2:3b       # 2 GB · the CPU-only fallback Machine B runs by default
```

`sailor2` is the pick because Khmer is explicitly supported and its tokenizer handles
Khmer numerals/danda properly — the usual failure mode (an 8 B English model emitting
Khmer as tofu or as transliteration) is what kills these pipelines. Each role
(controller / auto-idea / QA) is independently assigned a model **and a fallback model** in
Settings, exactly like `ai_creator/team.py`; the fallback is used when the primary is not
pulled or the run is on a CPU-only profile.

### 2 · Khmer voice (Stage 3a) — `sherpa-onnx` + `vits-mms-khm`

Meta publishes `facebook/mms-tts/khm` weights, and k2-fsa documents how to convert them,
but there is **no pre-built `vits-mms-khm` download**. So this repo ships the conversion:

```bash
./scripts/setup_khmer_tts.sh          # Windows:  .\scripts\setup_khmer_tts.ps1
```

It installs `onnx scipy Cython` + a *CPU* torch wheel, downloads `G_100000.pth` /
`config.json` / `vocab.txt`, clones the MMS space, builds `monotonic_align` (needs a C
compiler — VS Build Tools "Desktop development with C++", or run it in WSL), exports
`model.onnx` + `tokens.txt`, drops them in `data/studio/models/tts/vits-mms-khm/`,
installs `sherpa-onnx`, and speaks one Khmer sentence as a smoke test
(`smoke-test.wav`). ~10 minutes, one time. Details in `scripts/vits-mms-export.py --help`.

The engine then uses the Python API (`OfflineTts` + `OfflineTtsVitsModelConfig`, CPU
provider). The MMS frontend is **grapheme-based** — no `espeak-ng-data`, no lexicon file.
Long scripts are chunked at Khmer sentence boundaries (`។`) to ~180 chars and cross-faded,
because one giant pass is what makes VITS models hallucinate a trailing whisper.

*If you skip this step, Stage 3a produces a clearly-labelled placeholder track (correct
duration, syllable-timed tone) so the rest of the pipeline stays testable end-to-end. The
UI, the stepper, the QA verdict and the final file all say "placeholder voice" — never
silent, never pretending.*

### 3 · Your own voice (Stage 3b) — RVC

Train an RVC model on **your own** 10–15 minutes of clean Khmer speech once, then every
video afterwards uses it:

```bash
# RVC-WebUI (https://github.com/RVC-WebUI/RVC-WebUI) — dataset 10–15 min, 40k, f0=rmvpe
python infer-web.py                     # web UI → train tab
# or Applio if you prefer its REST API
```

The studio finds voices by scanning `rvc.models_dir` (`data/studio/models/rvc/` by
default) for `<name>.pth` + `assets/<name>.index`; drop them there (or symlink your
RVC-WebUI `assets/weights`) and press *Rescan* in Settings → Voice. Pick one per project
or globally; the settings panel then shows: base voice → converted voice, with A/B players
for both and pitch/index-rate/RMS-mix controls that map to the RVC CLI flags.

Two ways to run inference, auto-detected and overridable (`rvc.engine`):

* `http` — RVC-WebUI/Applio inference API. Defaults: `api_base http://127.0.0.1:9513`,
  `api_endpoint /sync`, upload field `audio_file`, extra fields
  `rvc_model/pitch/index_rate/rms_mix_rate/f0_method`. If your server speaks a dialect
  (`/infer`, `audio_path`…) change those three keys in Settings; nothing else has to move.
* `cli` — `python infer_cli.py …` inside `rvc.webui_dir` (works with `RVC_CLI`'s
  `rvc.py infer` too, by editing `rvc.cli_template`).
* `bypass` / `off` — skip the timbre pass; 3a's base voice is used as final.

### 4 · ComfyUI (Stages 4 and 5) — Wan + MMAudio

```powershell
git clone https://github.com/comfyanonymous/ComfyUI ; cd ComfyUI
python -m pip install -r requirements.txt
git clone https://github.com/kijai/ComfyUI-MMAudio custom_nodes\ComfyUI-MMAudio
python main.py --listen 127.0.0.1 --port 8188
```

Model files (place exactly as shown — these are the names upstream publishes):

```
ComfyUI/models/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors     # or wan2.1_t2v_1.3b_bf16
ComfyUI/models/clip/umt5_xxl_fp8_e4m3fn_scaled.safetensors
ComfyUI/models/vae/wan2.2_vae.safetensors                           # 1.3B: wan_2.1_vae.safetensors
ComfyUI/models/mmaudio/mmaudio_small_08_44k_v2.safetensors         # sub-8GB pick (large_44k_v2 if you have room)
ComfyUI/models/mmaudio/mmaudio_vae_44k_fp16.safetensors
ComfyUI/models/mmaudio/mmaudio_synchformer_fp16.safetensors
ComfyUI/models/mmaudio/apple_DFN5B-CLIP-ViT-H-14-384_fp16.safetensors
ComfyUI/models/mmaudio/bigvgan_v2_44khz_128band_512x/              # vocoder folder
```

The studio's default render settings are the community's 8 GB-safe Wan2.2 recipe:
**480×854 (9:16), 16 fps, 20 steps, CFG 6, `euler/simple`, tiled VAE, ≤ 81 frames per
clip**, with longer scenes split into consecutive clips (`vram.max_scene_seconds_for_model`)
rather than pushed into one big latent. `video.workflow` selects which graph to inject
into: `wan2.1_t2v_1.3b_480p` (fast, the default — 1.3 B is the honest choice for 8 GB if you
want 49-frame clips without offloading), `wan2.2_ti2v_5b_480p` (better motion; enable
ComfyUI's native CPU offloading), or `mmaudio_small_480p` for Stage 5.

**Bring-your-own-workflow is first class.** The studio does not build node graphs in code;
it fills `{{PROMPT}}`, `{{NEGATIVE}}`, `{{WIDTH}}`, `{{FRAMES}}`, `{{SEED}}` … markers in a
ComfyUI *API-format* JSON. Export your own working workflow (ComfyUI → *Save (API
format)*), put it in `ai_studio/workflows/` or `data/studio/workflows/`, drop the markers
where values belong (`ai_studio/workflows/README.md` lists them all), and select it in
Settings. Missing markers are reported as `unresolved placeholders: …` instead of silently
rendering the wrong thing — which matters because MMAudio's custom-node class names are the
one part of the ecosystem this repo cannot verify without a GPU: if `kijai/ComfyUI-MMAudio`
renamed a node in your version, you export yours and change one filename, nothing else.

While ComfyUI is not running, Stage 4 falls back to the built-in **previz** renderer: a real
animated clip (numpy/PIL — parallax sky, water shimmer, leaves, dust motes, mood-matched
palette) rather than a grey frame, so you can validate pacing, subtitles and duration
matching on a machine with no GPU at all. It is labelled `engine=previz` in the stepper and
in QA, and it is a *draft*, not a substitute for Wan.

### 5 · ffmpeg (Stage 7)

`winget install Gyan.FFmpeg` (or rely on the `imageio-ffmpeg` wheel the studio installs).
`ffprobe` is not required — durations come from `ffmpeg -i` parsing.

---

## 6 · Running it

```bash
python -m ai_studio                      # http://localhost:8000
python -m ai_studio --port 8002 --demo   # + three sample projects (one Mode A with a script, one Mode B waiting for its idea, one empty draft to try the create flow)
python -m ai_studio --check              # readiness report only, no server
python -m ai_studio --seed-demo          # create the three sample projects and exit
python -m ai_studio --machine machine_b  # force the CPU-only policy
python -m ai_studio --data-dir D:\studio # move everything (DB, media, models) elsewhere
STUDIO_DATA_DIR=/mnt/fast/studio python -m ai_studio    # same thing via env
```

or `uvicorn ai_studio.app:app --host 0.0.0.0 --port 8000`. `--host 0.0.0.0` is the default
so your phone can watch the render on the LAN; the API is unauthenticated by design (it is
a local tool — put it behind Tailscale rather than a public port).

### The UI, in four sentences

**Dashboard** — searchable (`title`, topic, script text), sorted by updated/title/status,
with mode, scene count, last run state and duration on each card; click through to a
project, *Duplicate* it to reuse the exact voice + prompt settings, or open *Memory*.
**New project** — pick Mode A (paste script; you see the char/sentence counters and the
segmentation preview) or Mode B (topic hint + target duration + optional full autonomy).
**Run view** — the per-stage × per-scene grid: queued / running (with % and engine name) /
done / failed / blocked / deferred, live over WebSocket (`/api/runs/{id}/events`, polling
fallback), plus the overall bar, the run log, and Pause / Cancel / Resume / Regenerate-stage
buttons on every cell. **Inspect** — click a scene: play 3a voice, play 3b converted voice
(A/B), preview the silent clip, preview the ambience with a waveform, read the QA verdict
and the exact prompts that produced them, edit the visual/mood tags, and re-run only the
stage you are unhappy with. `Download final .mp4` sits on the project header, and
*all intermediates* is one zip next to it.

---

## 7 · What is remembered (and why you should care)

`data/studio/studio.db` (SQLite, WAL) keeps: projects (+ mode, script, lock, target
duration), scenes (text, visual prompt, mood tag, SFX prompt, estimated vs actual
duration, per-scene meta), runs and per-stage rows (status, attempt count, engine used,
progress, timings, error text, `inherited_from`), assets (kind, path, size, duration,
meta), **every prompt sent to every model with the raw response and the model name**, and
the event log. That is what makes "why does scene 3 look wrong?" answerable, and it is
browsable in the UI (`/api/prompts`, `/api/memory/search?q=…`) and reusable (duplicate a
project and the same visual prompts/mood tags/voice profile come along).

Nothing else is written outside `data/studio/` — media lives in
`projects/<id>/{audio,video,ambient,final}/`, so a backup is `cp -a`.

---

## 8 · HTTP surface (all of it, in one table)

| Area | Routes |
|---|---|
| health/config | `GET /api/status` · `GET /api/health` · `GET|POST /api/settings` · `POST /api/settings/probe` · `GET /api/style` · `GET /api/workflows` |
| projects | `GET|POST /api/projects` · `GET|PATCH|DELETE /api/projects/{id}` · `POST …/duplicate` · `POST …/scenes` · `POST …/approve-script` · `POST …/regenerate-script` · `POST …/generate-idea` · `POST …/catchup` · `GET …/export` · `GET …/download?kind=final|bundle|all` · `GET …/scene/{idx}/download` |
| runs | `POST /api/projects/{id}/runs` · `GET /api/runs/{id}` · `GET /api/runs/{id}/status?since=N` · `POST /api/runs/{id}/pause|resume|cancel|continue` · `POST /api/runs/{id}/stages/{stage}/regenerate` · `GET /api/runs/{id}/scenes/{idx}/bundle` · `WS /api/runs/{id}/events` · `SSE /api/runs/{id}/stream` |
| media | `GET /api/assets?project_id&kind` · `GET /api/assets/{id}/stream|/download|/waveform` · `GET /api/tmpfile?name=` · `/files/<relpath>` (static) |
| memory | `GET /api/prompts?project_id&run_id&stage` · `GET /api/memory/search?q=` · `GET /api/jobs` |
| voices | `GET|POST /api/voices` · `POST /api/voices/import-discovered` · `DELETE /api/voices/{id}` · `POST /api/voices/{id}/select|preview|train` · `GET /api/training/{job_id}` |
| preview | `POST /api/preview/previz` (render a 2 s mood draft without creating a project) |

`GET /docs` is the OpenAPI version of that list. Every route is covered by the FastAPI
tests in `tests/test_studio_pipeline.py`.

---

## 9 · Tests

```bash
PYTHONPATH=. pytest tests/test_studio_text.py tests/test_studio_pipeline.py -q
```

56 tests, no GPU and no network needed: Khmer text handling (segmentation, cluster-
safe truncation, danda, syllable→duration, chunking), style-guardrail invariants, the
settings clamps (including the 8 GB VRAM rules), the DAG (topology, ready/blocked sets,
`stage#idx` keys), integrity enforcement for Mode A, the VRAM guard's arithmetic, a full
7-stage run in both modes through the fallback engines (asserting the final `.mp4` exists
and has video+audio streams), content_type round-trips and offline breakdown for every
content type, resume inheritance, per-stage regeneration, review gating, a forced stage
failure (retry count, specific surfaced error, dependents' behaviour, resumability), the
event bus (WebSocket replay), settings persistence, and the HTTP surface with
`TestClient`. ffmpeg-dependent tests skip themselves if ffmpeg is missing.

---

## 10 · Troubleshooting

| Symptom | What is actually happening | Do this |
|---|---|---|
| Voice says "placeholder" everywhere | no `model.onnx` + `tokens.txt` in `data/studio/models/tts/vits-mms-khm` | `./scripts/setup_khmer_tts.sh`, then *Re-probe* in Settings |
| `unresolved placeholders: {{WIDTH}}` | your workflow JSON lost a marker (common after re-exporting) | re-add the marker, or pick the shipped workflow in Settings |
| `no video clip to fit` | Stage 4 failed upstream; 4b has nothing to trim | read the `video` cell's error — usually ComfyUI model/node names |
| ComfyUI: `node_errors: MissingInputType` / `I'm missing X` | a model file is not in the folder the *running* ComfyUI uses | check `--base-directory`, re-place files, `GET /api/status → capabilities.comfyui` |
| Every scene `deferred` on Machine A | profile resolved to CPU-only (no CUDA visible to Python) | `python -m ai_studio --machine machine_a`, or fix drivers; `--check` shows what it saw |
| Ollama slow / 429 | model cold or num_ctx too big | `ollama pull sailor2:8b` once, keep `num_ctx 4096`, or set role model to `llama3.2:3b` |
| "waiting for the Director to approve the script" | Mode B review gate, working as intended | Approve / edit / regenerate on the project page (or set `review_gate: never`) |
| Run stays 100 % GPU-bound for minutes | one long scene split into two clips | normal: 2 clips × 81 frames @ 8 GB; shorten the scene or lower `video.steps` |
| `audioop`/`gtts` import errors | legacy `ai_creator/` extras in a slim venv | not needed by the studio; `pip install -r requirements.txt` if you also use the old UI |
| CUDA OOM anyway | another app holding VRAM (browser, DAW, game overlay) | close it, or raise `vram.reserve_free_mb`, or set `video.max_frames` 49 |
| `database is locked` | you opened the same DB twice (two servers) | one server per data dir; the DB already uses WAL + 8 s busy timeout |
| Port 8000 busy | the legacy `ai_creator` app is up | `--port 8002` for one of them |

---

## 11 · Honest limits (v1)

* **Khmer MMS is intelligible, not broadcast-grade.** Meta's per-language VITS was trained
  on limited data; expect a slight accent and occasional rushed syllables. Your RVC model
  (3b) hides a lot of that. Quality-per-effort here beats any available local Khmer
  alternative, and Stage 3a's output is what 3b's A/B player lets you judge.
* **Previz is not Wan.** Without ComfyUI you get an animated mood draft, and QA labels the
  picture accordingly. The 80 % "rewatchable" bar in the brief assumes the Wan stage ran.
* **MMAudio node names** are the one part unverified on this repo's side (no GPU here) — the
  bring-your-own-workflow path is deliberate, not a dodge.
* **QA is a language model reading facts about the media**, not a pixel-differ. It catches
  tone drift, pacing, missing/short tracks, and off-style tags.
* **Vertical 480×854 only** is validated at the 8 GB tier; 720p/1080p needs a bigger card —
  raise `vram.limit_mb` *and* the profile cap, on purpose.
* The `--demo` projects exist so the pipeline is testable in 30 seconds; they are marked
  `demo` in the dashboard and can be deleted in one click.

## 13 · React/TypeScript frontend

The UI is a React + TypeScript SPA (`web/`). It is built against the existing FastAPI
HTTP/WebSocket surface — no backend business logic is required to run it. Two ways to
serve it:

**Production (recommended):** build into the Python static folder, then run the one
existing studio process.

```bash
cd web
npm install
npm run build          # writes hashed files into ../ai_studio/static
cd ..
python -m ai_studio --port 8000
```

The Vite config uses `base: '/static/'` and `outDir: '../ai_studio/static'`, so the
backend's existing `/` route and `/static` mount serve the bundle. `npm run build` is
what CI/a fresh checkout should run before starting the studio.

**Dev server:** while editing the UI, use Vite's proxy and keep FastAPI on 8000.

```bash
# terminal 1
cd web && npm run dev   # http://localhost:5173, proxies /api and /files to :8000
# terminal 2
python -m ai_studio --port 8000
```

The app reads `content_types`, roles/engines, workflows and fix commands from the
backend (`/api/settings`, `/api/workflows`, `/api/settings/probe`) rather than hardcoding
them. The project list, content-type cards, live pipeline graph, scene inspector,
services panel, team and plugins screens are all wired to those endpoints.

## 14 · Licences

`vits-mms-*` weights: **CC-BY-NC 4.0** (Meta) — fine for your own channel, verify before
monetising. `sailor2`: Apache-2.0. Wan 2.1/2.2: Apache-2.0. MMAudio: check the repo's licence
(non-commercial-leaning). RVC: MIT, but *your* voice model's rights follow your recordings.
Everything in this folder is MIT with the rest of the repo.

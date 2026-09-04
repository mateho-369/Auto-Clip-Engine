# 🎬 Khmer AI Content Studio

**Director-led, multi-agent, 100 % local video pipeline.** You give it either a finished
Khmer script (*Mode A*) or just a topic (*Mode B*); it breaks the script into scenes,
speaks it in Khmer, turns your own trained voice into the timbre, animates 480p footage,
lays calm nature ambience under the narration, QA-checks every scene, and stitches a
vertical `.mp4` — with a live per-stage stepper you can inspect, retry, or re-run one
stage at a time.

Content-type-aware by default (`explainer`, `what_if`, `compare`, `choose`, `word_nuance`,
`myth_vs_fact`, `quick_tip`) — the structure (parallel sides, myth then fact, meaning
pairs, options + takeaway, one fast tip) survives even when no LLM is online.

No cloud API, no subscription, nothing leaves the machine. The whole studio — database,
media, prompts, settings — is one folder (`data/studio/`) you can copy to another PC.

```bash
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
| 1 | 🧩 Scene breakdown | same LLM, JSON-constrained | CPU/GPU (LLM) | deterministic, **content-type-aware** sentence splitter |
| 3a | 🎙️ Khmer voice | `sherpa-onnx` + `vits-mms-khm` | CPU | syllable-timed placeholder tone (clearly flagged) |
| 3b | 🎚️ Your timbre | RVC WebUI / `RVC_CLI` (your trained model) | GPU (or CPU) | base voice passes through (bypass) |
| 3c | 🧑 Talking head (NPC) | SadTalker (`infer.py`, image + voice → lip-sync) | GPU, 8 GB-tamed | matched-expression **still** with Ken Burns (`engine=still`, honestly flagged) |
| 4 | 🎞️ Animator | ComfyUI + **Wan2.1-T2V-1.3B** (or Wan2.2-TI2V-5B) | GPU, 8 GB-tamed | procedural *previz* clip (real animation, soft motion, mood-tinted) |
| 4b | 🖼️ Illustration / character still | ComfyUI **FLUX.2 klein 4B (fp8)** → Wan I2V start-frame, or Ken Burns on the still | GPU, 8 GB-tamed | PIL gradient card + Ken Burns (labelled `gradient`) |
| 4c | ⏱️ Duration match | ffmpeg | CPU | — (always available) |
| 5 | 🔊 SFX Director | ComfyUI + **MMAudio** (small 8 GB variant) | GPU | procedural ambience layered from the mood tag |
| 6 | ✅ QA Reviewer | Ollama (`sailor2:8b`) | CPU/GPU (LLM) | heuristic checks (duration, pacing, missing media) |
| 7 | 🎬 Final assembly | ffmpeg (`libx264` + `aac`, loudness-normalised, subtitles/title card burned) | CPU | needs ffmpeg; a run without it reports the reason |

The dependency graph (each scene gets its own row of jobs):

```
script ──► breakdown ──► per scene i:
                          voice_base:i ──► voice_final:i ─┐
                          talking_head:i (NPC scenes only)│
                          video:i ────────────────────────┼─► video_fit:i ─┐
                          video:i ──► sfx:i ─────────────────────────────┤
                                                          └──────────► qa:i ─┴─► assemble
```

`voice_base:i` and `video:i` are **deliberately parallel** — narration timing is what the
picture must match, so video starts as soon as the scene text exists and is trimmed/
looped to the *final* voice length in 4c. `qa:i` judges the whole scene, and `assemble`
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

### 1a · Scene visual sources (`visual_source`)

Every scene carries one of:

| `visual_source` | What happens |
|---|---|
| `generated_video` | Wan T2V/TI2V from the composed prompt (default) |
| `character_demo` | Wan I2V start-frame = matched-expression character image; prompt = scene visuals + **in-place gesture tail** (`standing in place, <action> miming motion, minimal background movement, camera static`) + the character's **pose phrase** (`ai_studio/mood_poses.py`, editable) |
| `illustration` | FLUX.2 klein still → Ken Burns pan/zoom |
| (custom image) | `POST /api/projects/{id}/scenes/{idx}/image` — your still → Ken Burns, generation skipped |

NPC mode: `render_mode` `broll` (default) or `talking_head`. `talking_head` **requires a
character** — the save surface returns `400 render_mode 'talking_head' needs a character
on the project or this scene`, and the UI shows that exact text. `character_demo` likewise
rejects a scene with no character. Character broll (non-talking) uses the matched mood
expression image as the I2V start-frame, so the character is *consistent* across scenes.

Content-type defaults: **compare** → `character_demo` on every side when a character is
set, else `illustration` per side; **word_nuance** / **choose** → `illustration` per
meaning/option; everything else stays `generated_video`.

### 1b · `[[silent: …]]` markup

```khmer
សួស្ដី។ [[silent: សូម]] អ្នកស្រមៃមួយភ្លែត។
```

The bracketed phrase is **displayed** (scene text, captions, SRT) but **never sent to
TTS** — `spoken_text` removes it before synthesis, timings/syllable counts exclude it.
Exactly one deterministic reference to a silent beat exists: the old random
phrase-final pause is gone; `assembly.line_gap_sec` (default 1.0 s, clamped 0.3–3.0 s,
per project) adds silence between lines via `tpad` with the value recorded in the
manifest (`pacing.line_gap_sec`).

### 1c · Pace

`tts.pace` (per project): `slow` / `natural` / `brisk` → `pace_calm` 0.9 / 1.0 / 1.08 and
a per-syllable factor, so scene estimates, QA duration checks and the assembled video all
agree. The Advanced tab in the New-Project flow prefills `line_gap_sec`.

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

### 3a · Content types

New projects pick one on a **card picker** (icon + one-line description, never a bare
dropdown). The type rides through every role prompt (Controller, Auto-Idea, QA) and the
scene schema (`content_type` on project + scenes), and the **deterministic fallback
honours it structurally** even with Ollama offline:

| Type | Structure without an LLM | Visual default |
|---|---|---|
| `explainer` 🧠 (default) | greedy scene packing, ≥1 sentence | generated video |
| `what_if` ✨ | hypothetical opening frame + packed body | generated video |
| `compare` ⚖️ | one sentence per scene, A half → B half → summary | character_demo w/ character, else illustration per side |
| `choose` 🧭 | option scenes + takeaway | illustration |
| `word_nuance` 🔤 | meaning-1 → meaning-2 → contrast | illustration |
| `myth_vs_fact` ✅ | myth → fact → why | generated video |
| `quick_tip` ⚡ | 1–2 scenes, shorter caps | generated video |

### 3b · Characters

`characters` + `character_images` tables (local SQLite + files in
`data/studio/characters/<id>/`). Each character has named expression photos
(`neutral`, `calm`, `sad`, `happy`, … — any label). A mood tag is mapped to the **nearest
expression label** via `content.expression_for_mood` (synonym-first, unknown-mood guard);
the chosen image is either the talking_head still or the I2V start-frame. See §8 for the
CRUD routes; the UI has a Characters panel with per-expression upload and mood→label
mapping preview.

### 3c · Two-character scripts — current manual option, automated later

Two-character scripts are supported **today** by tagging scenes per character (shot /
reverse-shot): each scene carries its own `character_id` + `visual_source:
character_demo`, and the alternating single-character scenes are assembled in order.
The UI exposes character assignment per scene in the inspector.

**Not automatic yet:** rendering *two characters in the same frame* (dialogue coverage,
both on screen) and an automated speaker-role parser for a `A: … B: …` script. That needs
the larger image-to-video headroom of a **12 GB+ GPU** (two-character Wan I2V at the same
quality tier is beyond the 8 GB budget) and is a future manual option — the schema
(per-scene character_id) already supports it; the packer and the ComfyUI graphs are the
pieces that would change.

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
| Talking head (3c) | ✅ SadTalker, else matched still | ⚠️ still + Ken Burns always |
| Illustration 💡 | ✅ FLUX.2 klein 4B (fp8) @ 480×854 | ✅ PIL gradient (or via ComfyUI if shared) |
| Video (4) | ✅ Wan 1.3B @ 480p / Wan2.2-5B with offloading | ❌ **deferred** (previz allowed) |
| MMAudio (5) | ✅ small 44 k variant | ❌ **deferred** |
| Assembly (7) | ✅ ffmpeg | ✅ ffmpeg |

### What "8 GB" means here

Nothing may *need* more than 8 GB at inference, so the studio enforces four things:

1. **One GPU model resident at a time** (`vram.serialize_gpu`, default on): the RVC pass,
   the Wan render, the FLUX still, the SadTalker pass and the MMAudio pass are serialised
   through a single semaphore, and `keep_alive: "0"` unloads the LLM after each stage
   instead of squatting on VRAM. The talker/illustration planners prefer **sequential
   load/unload** when ComfyUI and the other engine would otherwise co-reside on 8 GB.
2. **A megapixel-frame budget** (`ai_studio/vram.py`): `42 Mpx·f` per 8 GB
   (`~0.88 GB/GB`), so a request is scaled down *before* submission —
   `480×854×81 = 33.2 Mpx·f` is inside the house tier and untouched; asking for
   `1080p×121` gets you a smaller clip plus a note in the stage log
   (`vram: 1080×1920×121 → 480×854×81 (budget 42 Mpx·f/8GB)`), never a CUDA OOM.
   A hard floor of 34 frames keeps motion coherent.
3. **A free-VRAM reservation** (`vram.reserve_free_mb: 900`): if the card reports less
   free memory than that, the stage defers to its CPU fallback and says so.
4. **Small-model picks**: Wan `1.3B` (default), FLUX.2 **klein 4B fp8** at 480p,
   MMAudio **small 44 k**, SadTalker with a 256px head crop, Ollama 8B Q4 — every heavy
   graph is the community-recognised 8 GB-safe variant.

`machine.profile` (`auto | machine_a | machine_b`) sets the policy; `machine_a` also caps
`vram.limit_mb` at 8192 so a mis-read "16 GB total" can never license an 16 GB allocation.

### VRAM measurements (recorded on first real run)

The studio's own reports (already live on Machine B / this sandbox):

| Source | What it tells you |
|---|---|
| `GET /api/status` | `machine.gpus` (name / total / free from `nvidia-smi`), `machine.vram_total_mb`, `machine.vram_free_mb`, per-engine `plan` with `engine`/`reason` |
| `GET /api/settings` | the active budget: `vram.limit_mb`, `reserve_free_mb`, `serialize_gpu`, `downscale_on_pressure`, `video.max_frames` |
| `POST /api/settings/probe` | per-engine `check_ok` + verbatim `--check` **fix command** (talking_head, illustration, video, tts, rvc, sfx) |
| stage logs | any actual guard hit: `vram: … → … (budget 42 Mpx·f/8GB)`, `free VRAM 512 MB < reserve 900` |

**Peak-residency table** — the per-stage numbers below are the **safe 8 GB budgets the
config enforces**; the *measured* peaks on the target RTX 5070 can only be read on Machine
A, so they are intentionally blank until the first real GPU run. On that first run, record
`nvidia-smi --query-gpu=memory.used --format=csv` (or *Services → status* in the UI)
during each stage and fill this in:

| Stage | Budget target (config) | Measured peak (Machine A) |
|---|---|---|
| Ollama 8B Q4 (LLM) | ≤ ~6 GB, `keep_alive: 0` unload | _to record_ |
| Wan 1.3B @ 480×854 ≤81 f | ≤ ~7.2 GB (33.2 Mpx·f + reserve) | _to record_ |
| FLUX.2 klein 4B fp8 @ 480p | ≤ ~6.5 GB | _to record_ |
| SadTalker (512 px face) | ≤ ~4–5 GB | _to record_ |
| MMAudio small 44 k | ≤ ~6 GB | _to record_ |
| RVC (GPU) | ≤ ~2 GB | _to record_ |

This sandbox's live verification ran on **Machine B (CPU-only, no CUDA at all)**, so no
GPU peak could be measured here — every GPU stage resolved to its honest fallback
(previz / FLUX-gradient / still) and *said so* in the logs, which is exactly what the
budget above is designed to catch on Machine A.

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

### 4 · ComfyUI (Stages 4, 4b, 5) — Wan + FLUX + MMAudio

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
# Stage 4b (illustrations / character start-frames):
ComfyUI/models/unet/flux2-klein-4b-fp8.safetensors
```

The studio's default render settings are the community's 8 GB-safe Wan2.2 recipe:
**480×854 (9:16), 16 fps, 20 steps, CFG 6, `euler/simple`, tiled VAE, ≤ 81 frames per
clip**, with longer scenes split into consecutive clips (`vram.max_scene_seconds_for_model`)
rather than pushed into one big latent. `video.workflow` selects which graph to inject
into: `wan2.1_t2v_1.3b_480p` (fast, the default — 1.3 B is the honest choice for 8 GB if you
want 49-frame clips without offloading), `wan2.2_ti2v_5b_480p` (better motion; enable
ComfyUI's native CPU offloading), `flux2_klein_t2i_480p` (Stage 4b stills), or
`mmaudio_small_480p` for Stage 5.

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
in QA, and it is a *draft*, not a substitute for Wan. Stage 4b without ComfyUI renders a
mood-tinted **PIL gradient card** (`engine=gradient`) so the Ken Burns path still works.

### 5 · Talking head (Stage 3c) — SadTalker

Clone SadTalker (its `infer.py` is the interface), point `talking_head.sadtalker_dir` at
it (or env `SADTALKER_DIR`), and the probe picks it up. Each talking-head scene is pushed
as: matched-expression character image + the scene's final voice WAV → lip-synced clip
(≤ scene duration). Without SadTalker the scene still renders — the matched expression
image becomes a **Ken Burns still** with `engine=still` and `real_talking_head: false` so
the difference is never hidden.

### 6 · ffmpeg (Stage 7)

`winget install Gyan.FFmpeg` (or rely on the `imageio-ffmpeg` wheel the studio installs).
`ffprobe` is not required — durations come from `ffmpeg -i` parsing. The studio probes the
bundled binary for filters at startup: `subtitles` (libass) is present → caption burn
works; `drawtext` may be absent → title-card text renders through the PIL fallback instead
(handled automatically, same visual result).

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

### 6a · The UI: React/TypeScript build vs dev server

The UI is a **React 18 + TypeScript + Vite** app whose source lives in
`ai_studio/frontend/`. Two ways to run it:

**Production build (what the backend serves by default).**

```bash
cd ai_studio/frontend
npm ci && npm run build      # Vite emits into ai_studio/static/
cd ../..
python -m ai_studio          # FastAPI serves / (index.html, no-cache) + /static/*
```

* **Why this is the shipped choice:** one deploy folder (`ai_studio/static/`), no extra
  process, no CDN/font/asset host, no CORS — the browser calls the same origin. Vite emits
  content-hashed `assets/index-*.js/.css`, which are immutable and cache-safe; only the
  2 KB `index.html` is served `no-cache` so a rebuild never serves stale markup. The build
  replaced the old hand-written vanilla `app.js`/`style.css` 1:1 (feature-inventory was
  ported, then Layers 2–3 panels were added on top).

**Dev server (frontend-only iteration).**

```bash
cd ai_studio/frontend
npm run dev                  # Vite on http://localhost:5173
# keep the backend on :8000 (python -m ai_studio)
```

Vite proxies `/api` and `/files` (and the WebSocket) to `http://127.0.0.1:8000`, and the
built assets are not touched — you edit `src/` and see HMR immediately. *Any change to the
UI must end with `npm run build` before delivery,* because FastAPI serves the built copy.

### 6b · The UI, panel by panel (Premiere-style, dark, dense)

* **Top bar** — project name, run controls (Run / Pause / Resume / Cancel / Regenerate),
  health dot (backend + Ollama + ComfyUI + RVC), global Search.
* **Left sidebar** — Projects (cards with poster, mode/content-type badges, duplicate,
  delete), **New project**, and Services / AI Team / Plugins / Characters / Voices /
  Settings / Memory. Keyboard-friendly, window-docked panels — no marketing hero.
* **Center** — the scene board (cards = scenes with icons, per-scene character/side
  badges) and the live pipeline DAG (10 stages × scenes in one grid, exact stage names,
  live over WebSocket/SSE with polling fallback).
* **Right inspector** — select a scene: its assets (voice A/B, clip, still, captions,
  QA JSON), `visual_source` three-way control, `render_mode` toggle (only with a
  character), per-scene character display, image upload, stage regenerate, edit & save.
  The **event log** below surfaces every error verbatim from the backend (API error →
  toast with `detail`; stage error → red badge with the exact engine message).
* **New project flow** (7 steps, wizard): ① Mode → ② Content type **card picker** →
  ③ Character (optional) → ④ Pace + line gap (Advanced prefilled) → ⑤ Subtitle on/off +
  **Style Gallery** (real cached previews) → ⑥ **Title style gallery** → ⑦ Script/topic +
  duration + voice + style notes → Create.
* **Style Gallery** — real pre-rendered 3-second samples (`/api/style-previews`,
  cached on disk) for every subtitle style *and* title style — never blind dropdown names.
* **Director script editor** — live preview under the textarea: `[[silent: …]]` spans
  shown greyed + struck-through (displayed, never spoken), plus a
  **“mark selection as not spoken”** helper that wraps your selection in the markup.
* **Scene-board groups** — `compare` scenes are grouped under **side A / side B / summary**
  header rows and `word_nuance` under **meaning-1 / meaning-2 / contrast**, matching the
  structural tags the backend assigns (every other type stays a flat board).
* **AI Team** — per-role model + temperature + enable, options fetched from the backend,
  saved instantly with a toast on every change.
* **Old-UI features preserved 1:1** — run control (pause/resume/cancel/continue), per-stage
  regenerate, scene-board edit/save, script save + `generate-idea`, **`regenerate-script`**
  (Mode B), **GPU catch-up** button (deferred-count badge), **audio waveform** drawn from
  `/assets/{id}/waveform` in the inspector, live-dot (ws/poll), project duplicate/delete,
  download final/bundle/json, voice A/B players, QA JSON, event log with exact errors.
* **Services** — Studio (8000) / Ollama (11434) / RVC (9513) / ComfyUI (8188) live status,
  click-to-open, and the exact `--check` fix command per engine shown verbatim.
* **History** — projects + runs tables (mode, content type, status, poster, duplicate).

### 6c · Subtitles & title cards

Per-project `assembly.burn_captions` (on/off). Styles (each with a cached preview):

| Subtitle | Look |
|---|---|
| `clean` | neutral white, dark soft box, bottom |
| `bold_yellow` | bold yellow, stronger box |
| `minimal_top` | small, top of frame |
| `karaoke` | word-by-word highlight via ASS `\k` timing — timings come from `khmer.syllable_estimate`; **approximation by design** (ASR-based word timing is future work) |

Title cards are optional: `assembly.title_style` is **nullable** on the project; presets
`centered_fade` / `bottom_left_minimal` / `bold_pop`. Same preview gallery. Every run's
manifest records `pacing.title_style`, `pacing.subtitle_style` and `pacing.line_gap_sec`.

---

## 7 · What is remembered (and why you should care)

`data/studio/studio.db` (SQLite, WAL) keeps: projects (+ mode, script, lock, target
duration, **content_type**, **character_id**, per-project settings), **characters +
expression images**, scenes (text, visual prompt, mood tag, SFX prompt, estimated vs actual
duration, per-scene meta with visual_source/render_mode/character_id/side), runs and
per-stage rows (status, attempt count, engine used, progress, timings, error text,
`inherited_from`), assets (kind, path, size, duration, meta), **every prompt sent to every
model with the raw response and the model name**, and the event log. That is what makes
"why does scene 3 look wrong?" answerable, and it is browsable in the UI (`/api/prompts`,
`/api/memory/search?q=…`) and reusable (duplicate a project and the same visual prompts /
mood tags / character / voice profile come along).

Nothing else is written outside `data/studio/` — media lives in
`projects/<id>/{audio,video,ambient,final}/`, character images in `characters/<id>/`.

---

## 8 · HTTP surface (all of it, in one table)

| Area | Routes |
|---|---|
| health/config | `GET /api/status` · `GET /api/health` · `GET|POST /api/settings` · `POST /api/settings/probe` · `GET /api/style` · `GET /api/workflows` · `GET /api/ollama/models` |
| content/characters | `GET /api/content-types` · `GET|POST /api/characters` · `GET|PATCH|DELETE /api/characters/{id}` · `POST /api/characters/{id}/images` · `GET /api/characters/{id}/images/{image_id}/file` · `DELETE /api/characters/{id}/images/{image_id}` |
| projects | `GET|POST /api/projects` · `GET|PATCH|DELETE /api/projects/{id}` · `POST …/duplicate` · `POST …/scenes` · `POST …/scenes/{idx}/image` (+ DELETE) · `POST …/approve-script` · `POST …/regenerate-script` · `POST …/generate-idea` · `POST …/catchup` · `GET …/export` · `GET …/download?kind=final|bundle|all` · `GET …/scene/{idx}/download` |
| runs | `POST /api/projects/{id}/runs` · `GET /api/runs` · `GET /api/runs/{id}` · `GET /api/runs/{id}/status?since=N` · `POST /api/runs/{id}/pause|resume|cancel|continue` · `POST /api/runs/{id}/stages/{stage}/regenerate` · `GET /api/runs/{id}/scenes/{idx}/bundle` · `WS /api/runs/{id}/events` · `SSE /api/runs/{id}/stream` |
| media | `GET /api/assets?project_id&kind` · `GET /api/assets/{id}/stream|/download|/waveform` · `GET /api/tmpfile?name=` · `/files/<relpath>` (static) |
| styles | `GET /api/style-previews` (cached per subtitle/title style; honest `error` field when a render fails) |
| memory | `GET /api/prompts?project_id&run_id&stage` · `GET /api/memory/search?q=` · `GET /api/jobs` |
| voices | `GET|POST /api/voices` · `POST /api/voices/import-discovered` · `DELETE /api/voices/{id}` · `POST /api/voices/{id}/select|preview|train` · `GET /api/training/{job_id}` |
| preview | `POST /api/preview/previz` (render a 2 s mood draft without creating a project) |

`GET /docs` is the OpenAPI version of that list. Every route is covered by the FastAPI
tests in `tests/test_studio_pipeline.py`.

---

## 9 · Tests

```bash
PYTHONPATH=. pytest tests/test_studio_text.py tests/test_studio_pipeline.py -q   # 66 passed
python -m pytest -q    # via `python -m` from the repo root: 143 passed (incl. legacy ai_creator suite)
```

No GPU and no network needed: Khmer text handling (cluster-safe segmentation, danda,
syllable→duration, chunking, silent-markup display vs speech), style-guardrail invariants,
the settings clamps (including the 8 GB VRAM rules), the DAG (topology, ready/blocked sets,
`stage#idx` keys), integrity enforcement for Mode A, the VRAM guard's arithmetic, a full
7-stage run in both modes through the fallback engines (asserting the final `.mp4` exists
and has video+audio streams), resume inheritance, per-stage regeneration, review gating, a
forced stage failure (retry count, specific surfaced error, dependents' behaviour,
resumability), the event bus (WebSocket replay), settings persistence, the HTTP surface with
`TestClient`, content-type structural fallback, character CRUD, talking-head guard (400
without character), custom scene-image → Ken Burns, and the style-previews gallery.
ffmpeg-dependent tests skip themselves if ffmpeg is missing.

### Live verification battery

`scripts/verify_live_layer3.py` is the mandatory end-to-end proof against a *running*
server (`python -m ai_studio --data-dir <fresh dir>`, then run the script). It drives real
HTTP calls end-to-end and asserts `status: completed`, **0 failed jobs**, plus inspecting
the output JSON/audio/asset files:

| Run | What it proves | Result |
|---|---|---|
| style previews | 4 subtitle + 3 title styles listed, cached MP4 actually serves | ✅ |
| GPU catch-up | deferred `video`/`sfx`/`video_fit` run → catchup starts a run, `completed`, 0 failed | ✅ |
| regenerate-script | Mode B `POST …/regenerate-script` returns fresh `ai:template` draft | ✅ |
| waveform | `GET /api/assets/{id}/waveform?bins=64` → 65 peaks + duration | ✅ |
| content types ×7 | each type end-to-end via HTTP: `completed`, 0 failed, all scenes carry `content_type`, structure tags (A/B/summary, meaning-1/2, myth/fact, option-N/takeaway, hypothetical, ≤2 scenes) | ✅ 21/21 |
| coeng boundary | long subscript-heavy script → title + scene texts + SRT contain **zero** lone `្` (no `U+17D2` before space/EOL/punct) | ✅ |
| silent estimate | same script with `[[silent: នេះជា]]` → estimated speech **shorter** than spoken version (3.34s vs 3.74s) | ✅ |
| silent-gap | `[[silent:]]` not spoken (manifest/pacing), present in SRT + scene text, `line_gap_sec=0.6` honoured, final asset exists | ✅ |
| characters | CRUD + 4 expression uploads + mood→expression map | ✅ |
| compare (no char) | A→B→summary ordering, per-side `illustration` | ✅ |
| compare (char) | `character_demo` per side, **in-place gesture tail** + pose phrase in the actual composed prompt | ✅ |
| two-character | 3 scenes saved with alternating `character_id` (shot/reverse-shot), run completes | ✅ |
| talking head | with character → stage `engine=still` fallback + video skipped; without character → `400` with exact text | ✅ |
| scene image upload | custom PNG → `engine=kenburns` clip | ✅ |
| subtitle styles ×4 | each `clean`/`bold_yellow`/`minimal_top`/`karaoke` run completes, captions burned + manifest key | ✅ |
| title styles ×3 | each preset run completes, manifest key + title lead-in added | ✅ |
| sad pose | mood `sad` → pose phrase + in-place tail in the real video prompt | ✅ |
| **Total** | | **83/83 checks passed** |

---

## 10 · Troubleshooting

| Symptom | What is actually happening | Do this |
|---|---|---|
| Voice says "placeholder" everywhere | no `model.onnx` + `tokens.txt` in `data/studio/models/tts/vits-mms-khm` | `./scripts/setup_khmer_tts.sh`, then *Re-probe* in Settings |
| `unresolved placeholders: {{WIDTH}}` | your workflow JSON lost a marker (common after re-exporting) | re-add the marker, or pick the shipped workflow in Settings |
| `no video clip to fit` | Stage 4 failed upstream; 4c has nothing to trim | read the `video` cell's error — usually ComfyUI model/node names |
| ComfyUI: `node_errors: MissingInputType` / `I'm missing X` | a model file is not in the folder the *running* ComfyUI uses | check `--base-directory`, re-place files, `GET /api/status → capabilities.comfyui` |
| `render_mode 'talking_head' needs a character` | NPC mode without a character — by design | pick/upload a character in Characters, or set `render_mode: broll` |
| Style preview shows an error card | that render failed on this machine (e.g. no libass for subtitle burn) | read the `error` field — it's displayed honestly, never a broken dropdown |
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
* **Karaoke timing is approximate.** Word highlights use syllable estimates, not ASR
  alignment; exact phoneme timing is future work, and the style label says so.
* **Two characters in one frame is not automatic yet** — see §3c (needs 12 GB+; alternation
  works today).
* **QA is a language model reading facts about the media**, not a pixel-differ. It catches
  tone drift, pacing, missing/short tracks, and off-style tags.
* **Vertical 480×854 only** is validated at the 8 GB tier; 720p/1080p needs a bigger card —
  raise `vram.limit_mb` *and* the profile cap, on purpose.
* **Peak VRAM per stage was not measurable in this environment** (no GPU). The budgets are
  enforced in code; the measured table in §4 needs one real run on Machine A.
* The `--demo` projects exist so the pipeline is testable in 30 seconds; they are marked
  `demo` in the dashboard and can be deleted in one click.

## 12 · Licences

`vits-mms-*` weights: **CC-BY-NC 4.0** (Meta) — fine for your own channel, verify before
monetising. `sailor2`: Apache-2.0. Wan 2.1/2.2: Apache-2.0. FLUX.2 klein: see
Black Forest Labs' licence (non-commercial for the open weights — verify before
monetising). MMAudio: check the repo's licence (non-commercial-leaning). SadTalker:
Apache-2.0. RVC: MIT, but *your* voice model's rights follow your recordings.
Everything in this folder is MIT with the rest of the repo.

# Architecture — how the studio is actually built

This explains *why* the code is shaped the way it is, so you can find your
way around without re-reading every file. For *running* the thing, see
[`HOW-TO-RUN.md`](HOW-TO-RUN.md). For *using* the app once it's running,
see [`USER-GUIDE.md`](USER-GUIDE.md).

## In one sentence

A single Python app (`ai_studio`) is the **orchestrator** — it doesn't do any
AI itself. It calls out to four separate tools over HTTP (Ollama for text,
sherpa-onnx for speech, ComfyUI for video/SFX, RVC-WebUI for voice cloning),
tracks every step in a local SQLite database, and always has a fallback so a
run never just dies.

```
                         ┌──────────────────────┐
   your browser  ─────▶  │   ai_studio (:8000)   │   the only thing YOU talk to
                         │   FastAPI + SQLite     │
                         └──────────┬─────────────┘
                                    │ HTTP, only when that stage is due
             ┌──────────────┬──────┴───────┬───────────────┐
             ▼              ▼              ▼               ▼
        Ollama (:11434) sherpa-onnx   ComfyUI (:8188)  RVC-WebUI (:9513)
        script + QA     Khmer TTS     Wan video + MMAudio  voice timbre
        (subprocess/HTTP) (in-process)  (real GPU jobs)    (HTTP)
```

Nothing here talks to the others directly — everything routes back through
`ai_studio`. That's deliberate: it's the one place that knows the VRAM
budget, the fallback rules, and what to do when a service is offline.

## The pipeline: one project → one run → seven stages

A **project** is a script (or a topic, in Mode B) plus settings. Each time
you press "Start production" it creates a **run**, which is broken into
**stages** — some run once per project, most run once per *scene*:

| Stage | What it does | Runs on |
|---|---|---|
| `script` | Locks the Director's pasted script (Mode A) or has the LLM write one from a topic hint (Mode B) | once |
| `breakdown` | LLM splits the script into scenes, each with a visual prompt + mood tag | once |
| `voice_base` | sherpa-onnx synthesises the Khmer line for that scene | per scene |
| `voice_final` | RVC converts that voice to your cloned timbre (or passes it through unchanged if no voice profile exists) | per scene |
| `video` | ComfyUI renders a silent Wan2.1 clip for that scene | per scene |
| `video_fit` | ffmpeg re-times the clip so it matches the voice's exact length | per scene |
| `sfx` | ComfyUI + MMAudio generates ambience matched to the video | per scene |
| `qa` | LLM reviews the scene for script drift / obvious issues | per scene |
| `assemble` | ffmpeg stitches every scene's video+voice+sfx into the final `.mp4` | once |

Voice (`voice_base`→`voice_final`) and video (`video`→`video_fit`) run
**concurrently** per scene, then `sfx` waits on the video, and `assemble`
waits on everything — this is why voice generation starts from Stage 1's
*estimate* of scene length, and `video_fit` exists purely to correct the
picture to the voice's real, finished length afterward.

Code map: `ai_studio/pipeline/spec.py` defines this exact graph (dependencies,
per-scene vs. once, which stages are GPU-bound); `ai_studio/pipeline/
scheduler.py` walks it; `ai_studio/pipeline/stages.py` has the actual stage
implementations, which mostly just call into `ai_studio/engines/*.py`.

## The fallback philosophy: it never just fails

Every stage has a cheap, always-available fallback, and the config resolver
picks between them automatically:

| Stage | Real engine | Fallback |
|---|---|---|
| voice | sherpa-onnx (Khmer TTS) | a placeholder tone, so downstream stages still have *something* to time against |
| voice timbre | RVC | bypass (real voice, un-timbred) |
| video | ComfyUI + Wan2.1 | `previz` — a cheap procedural placeholder clip |
| SFX | ComfyUI + MMAudio | `procedural` ambience |

This is why you'll see engine names like `previz` or `bypass` in run logs —
that's not a bug, it's the system honestly telling you which stage didn't
have its real backend available *this run*. `python -m ai_studio --check`
shows you the same picture ahead of time.

## The config resolver: `auto | explicit | defer | off`

Every stage (`tts`, `rvc`, `video`, `sfx`) has an `engine` setting in
`data/studio/settings.json`, one of:

- **`auto`** — probe what's actually available right now and pick the best
  real option, falling back per the table above. This is the default.
- an **explicit engine name** (`comfyui`, `mmaudio`, `previz`, …) — force it,
  even if `auto` would have picked something else.
- **`defer`** — skip for now (used on a CPU-only "Machine B" for GPU stages;
  the run finishes with placeholders, and you can later run a GPU
  "catch-up" pass on Machine A to fill them in for real).
- **`off`** — never run this stage at all.

`ai_studio/config.py`'s `resolve()` function does this once per run and
produces a `plan` dict — that's the thing both the UI and the scheduler
read, so "what will actually happen" is decided in exactly one place.

## VRAM is the hard constraint

Target hardware is an 8GB GPU. `ai_studio/vram.py` is the traffic cop:

- estimates frames/resolution it can afford from currently-free VRAM
  (`frames_for`, `guard_request`) and shrinks a request rather than let it
  OOM;
- serializes all GPU work through one semaphore — ComfyUI and Ollama never
  run concurrently on purpose;
- tells Ollama to unload itself after each call (`keep_alive: "0"`) so its
  VRAM is free again for the next ComfyUI job.

If you ever see a run auto-shrink resolution or frame count with a note in
the log, that's this system working as intended, not an error.

## ComfyUI integration: templates, not generated graphs

`ai_studio` does **not** build ComfyUI node graphs in Python. Instead:

1. `ai_studio/workflows/*.json` are real ComfyUI graphs exported in **API
   format**, with `{{PLACEHOLDER}}` markers dropped into the fields that
   change per scene (prompt, resolution, seed, …).
2. `ai_studio/workflows.py`'s `render()` does simple string/value
   substitution into those placeholders and reports back exactly which ones
   it couldn't resolve — so a broken template fails loudly with a specific
   missing-placeholder name, not silently.
3. `ai_studio/comfy.py` is a small hand-rolled WebSocket client that submits
   the filled-in graph, watches progress, and downloads the result.

This means upgrading Wan/MMAudio versions, or swapping in a different model,
is a JSON edit in `workflows/`, not a code change — as long as the new
graph still has the placeholders the engine code expects (documented in
`ai_studio/workflows/README.md`).

## Where things are stored

```
data/studio/
  settings.json         ← the config described above
  studio.db              ← SQLite: projects, runs, stages, prompts, assets
  projects/<id>/
    scenes/<n>/           per-scene intermediate assets (voice, video, sfx…)
    final/                 the assembled output .mp4 for each run
  voices/<id>/            uploaded/trained RVC voice profiles
  models/                 downloaded model files (sherpa TTS model, etc.)
```

Every asset produced by a stage is a real file on disk under a project's
folder, referenced from `studio.db` — nothing is held only in memory, so a
run's intermediate state survives a restart and can be inspected directly.

## The two "legacy" folders you'll also see in this repo

- **`src/`** — the original 2024 highlight-clipper app. Still deployed
  (`Dockerfile` currently points at it, not at `ai_studio`) and still has
  live code paths — not dead, just a different, older product living in the
  same repo.
- **`ai_creator/`** — a second, separate FastAPI app with its own routes and
  tests. Also live, also not part of the `ai_studio` pipeline described
  above.

If you're working on the Khmer content studio, everything you want is under
`ai_studio/`; the other two folders are unrelated products that happen to
share this repository.

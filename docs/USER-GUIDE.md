# User Guide — Khmer AI Content Studio

Everything in the app, page by page, button by button. If something in the
UI isn't explained here, that's a gap to report, not something you're
expected to guess at.

- To get the app itself running: [`HOW-TO-RUN.md`](HOW-TO-RUN.md)
- To understand how it's built internally: [`ARCHITECTURE.md`](ARCHITECTURE.md)
- This document: how to actually **use** it once it's running.

Open **http://localhost:8000** to start. The top bar has six tabs:
**Projects · New · Voices · Memory · Settings**, plus a machine/plan status
pill on the right.

---

## Top bar — always visible

- **Machine pill** — which hardware profile is active and how much VRAM the
  GPU has free right now.
- **Plan pill** — a one-glance summary of what each stage will *actually*
  use this run: 🗣️ voice engine · 🎙️ timbre engine · 🖼️ video engine · 🌿 SFX
  engine. If any of these says `previz`, `procedural`, `placeholder`, or
  `bypass`, that stage isn't using its real backend right now — the banner
  underneath explains why and what to do about it.
- **Warning banner** (only shows when relevant) — plain-language notes like
  "Ollama is offline" or "RVC timbre is bypassed — add a voice profile
  under Voices." Click it to dismiss for this session.

---

## Projects (`#/projects`) — the home page

A list of every project you've created, each showing its mode (A/B),
status, and last run. Click one to open it. This is also where you land
after creating a new project.

---

## New (`#/new`) — start a project

### 1 · Pick how much control you want

- **Mode A — Director script.** You paste a finished Khmer script. It's
  treated as **ground truth** — the studio only splits it into scenes for
  timing; no AI agent is allowed to rewrite, paraphrase, or "improve" a
  single word of it. Use this when you already know exactly what you want
  said.
- **Mode B — Auto idea.** You give a one-line topic; the Controller (LLM)
  writes the full Khmer script for you, in a fixed house style (calm, warm,
  "don't give up" tone). You then review, edit, or regenerate it before
  production actually starts — or check "skip the review gate" to let it
  run fully autonomously.

### 2 · Details

- **Title** (optional) — just a label.
- **Target length (seconds)** — how long you want the finished video.
- **Mode A**: paste your script, one sentence per line, in Khmer. A live
  counter shows line count, character count, and an estimated spoken
  duration — use it to gut-check pacing before you commit.
- **Mode B**: a topic hint, optional extra style notes ("address it to
  young farmers, mention morning rain"), and the auto-approve checkbox.
- **Voice profile** — pick a trained RVC voice (see [Voices](#voices-voices)
  below) to use for this project, or leave as "the base Khmer voice" to skip
  timbre conversion.
- **Engine overrides** — force a specific engine for voice/video/SFX on
  *this project only*, overriding whatever Settings has configured. Leave
  blank to just use the global Settings choice.
- **"fill with sample script"** — drops in a ready-made Khmer sample so you
  can try the whole pipeline without writing anything yourself.
- **"see the guideline"** link — shows the house style rules every script
  (Mode A and B alike) is checked against.

Press **Create project** (Mode A) or **Create project & generate idea**
(Mode B) to move on.

---

## A project page (`#/project/<id>`)

This is where you actually run things and watch them happen. It has
several stacked sections:

### Script
Shows the locked script (Mode A) or the AI-generated one with an
approve/edit/regenerate flow (Mode B) before it locks.

### Storyboard
The scene breakdown — each scene's visual prompt, mood tag, and estimated
duration, as decided by Stage 1.

### Pipeline · stage by stage
A row per stage (`script → breakdown → voice_base → voice_final → video →
video_fit → sfx → qa → assemble`), each showing live status
(queued/running/done/error), which engine actually handled it, and a
progress percentage. **This is the most important thing to watch during a
run** — if a stage shows an engine you didn't expect (e.g. `previz` instead
of `comfyui`), that's your first clue something's misconfigured, not a
silent success.

Click a stage's scene to open the **Inspector**:
- hear/see the actual asset produced (audio player, waveform, video player)
- read every prompt sent to every model for that scene, straight from the
  SQLite log — genuinely useful for understanding *why* a scene came out
  the way it did
- **re-run this stage** with a different visual prompt / mood / ambience
  instruction, without re-running the whole pipeline
- **↻ Regenerate** / **🎨 previz test** buttons for quick iteration

### Files & downloads
The final assembled `.mp4` (and intermediate files, if you want them)
for this run.

### Live log
A raw scrolling feed of everything happening during an active run —
useful when a stage errors and you want the exact message.

### Run history
Every run this project has ever had, with status, trigger, machine profile,
and job counts. Click **open** on an old run to inspect it the same way as
the current one — nothing gets thrown away.

**Start production** kicks off a new run. If a previous run is incomplete,
you can resume it instead of starting fresh.

---

## Voices (`#/voices`)

This is where voice cloning (Stage 3b — RVC) is managed. Read this section
fully before training — it explains a real, current limitation.

### How the two stages relate
- **Stage 3a** always speaks Khmer using sherpa-onnx (`vits-mms-khm`) — the
  right words, a generic voice.
- **Stage 3b** re-timbres that audio using **your** trained RVC model, so
  the words stay correct but the voice becomes yours. If no voice profile
  is selected (or none exists), 3b is skipped ("bypass") and you just get
  3a's voice unmodified — this is a normal, working state, not an error.

### Voice profiles panel
Each saved voice profile shows as a tile with:
- **use as default / this project** — pick which voice a run should use
- **🔊 preview** — synthesizes a short line through 3a and (if trained) 3b,
  so you can hear it without running a full pipeline
- **sample** — plays back the raw training sample you uploaded
- **🎯 train** — see the important note below
- **🗑** — remove the profile (files on disk are kept, not deleted)
- status chips: whether a `.pth`/`.index` file exists, how long the sample
  is, current pitch shift, and training status

### ＋ add a voice
Opens a form:
- **name**, **pitch shift**, **notes** — labels/metadata
- **RVC .pth weights** — only fill this in if you already trained a model
  elsewhere (e.g. directly in RVC-WebUI)
- **.index** (optional) — goes with the `.pth` above
- **10–15 min training sample** — upload your raw recording here. **Audio
  or video both work** — if you upload a video, only its audio track is
  used (extracted automatically); you don't need to convert it yourself
  first.

You need **either** a `.pth` **or** a sample — not both. If you don't have
a trained model yet, just upload your sample and leave the `.pth` field
empty.

### ⚠ Important: the "🎯 train" button, honestly

Pressing **Train** on a profile calls a command configured in
**Settings → Voice timbre → training command**. On a fresh install, that
setting is **empty**, so pressing Train will show `training: error` in the
UI — this is expected, not a bug, and it's exactly what happens if you
haven't configured it yet.

**Two ways to actually train:**

1. **Manual, always works** (recommended until you've set up automation):
   Start RVC-WebUI's own interface directly —
   ```
   cd Retrieval-based-Voice-Conversion-WebUI
   ./.venv/Scripts/python.exe webui.py
   ```
   open **http://127.0.0.1:7865**, go to its **Train** tab, point the
   dataset path at your uploaded sample's folder (shown in the voice
   profile's details), and run through preprocess → extract features →
   train → train index. When it finishes, come back here and either
   **import from RVC folder** or upload the resulting `.pth`/`.index`
   directly.

2. **In-app automation** (optional, needs one-time setup): go to
   **Settings → Voice timbre**, fill in **training command** with the exact
   command line RVC-WebUI needs on your machine (there's a placeholder
   example in the field), save, then the **Train** button here will launch
   it directly and stream the log into a popup instead of erroring.

Either way, **quality depends on your sample length** — RVC wants 10-15
minutes of clean speech (no music, no echo). A much shorter sample will
still train, but the cloned voice will sound noticeably rougher.

### "Discovered on disk"
If RVC-WebUI's own model folder already has trained voices (from using its
UI directly, outside the studio), they show up here. Press **⇥ import from
RVC folder** to register them as studio voice profiles without re-uploading
anything.

### "Hear it now"
A standalone box to type any Khmer text and synthesize it through the exact
same 3a+3b engines a real run would use — the fastest way to sanity-check a
voice before committing 10 minutes to a full render.

### "How to train the model"
A condensed reminder of the manual RVC-WebUI steps above, always visible on
this page.

---

## Memory (`#/memory`)

A searchable browser over the studio's own SQLite record of *everything*
every AI call has ever done: every prompt sent, every response received,
grouped by project and by scene. Use this to answer "why did the AI decide
X" for any past run, or to audit exactly what got sent to Ollama/ComfyUI.
Type in the search box to filter; leave it empty to see the latest entries.

---

## Settings (`#/settings`)

Every configuration knob, grouped into cards:

- **Roles → models** — which Ollama model each AI role (controller,
  auto_idea, qa) uses, plus a fallback model if the primary isn't pulled.
- **What each stage will actually run on this machine** — a live preview of
  the resolved plan, same information as the top-bar plan pill but with
  full reasons.
- **Engines** — the `auto | explicit | defer | off` choice per stage
  (voice, timbre, video, SFX). `auto` is almost always right; only override
  if you specifically want to force or disable something.
- **Video · Wan through ComfyUI** — resolution, fps, steps, CFG, which
  workflow template to use, ComfyUI host/port, timeout.
- **SFX director** — MMAudio settings: workflow template, ambience gain,
  how much the ambience ducks under narration.
- **VRAM safety** — the 8GB budget knobs; you generally don't need to touch
  these unless you're on different hardware.
- **Machine profile** — force "Machine A" (GPU) or "Machine B" (CPU-only,
  defers GPU stages) instead of auto-detecting.
- **Pipeline** — max scenes per run, retry limit, per-stage concurrency.
- **Final assembly** — output resolution/format options for the stitched
  `.mp4`.
- **Paths & services** — Ollama URL, ComfyUI host, RVC settings (`webui_dir`,
  `models_dir`, `api_base`, `train_command`, pitch, index rate, F0 method —
  this is where the Voices page's Train button setup lives, see above), plus
  a **Probe** button to force an immediate live re-check of every service
  (bypassing the normal 25-second cache).
- **House style** (read-only) — the fixed tone/content rules every script
  is checked against, regardless of mode.
- **ComfyUI workflow templates** — lists the `.json` template files the
  studio knows about and which placeholders each one resolves.

Settings save automatically as you change them (per-field), no separate
"Save" button for most sections.

---

## Reading the plan pill / engine names

You'll see these exact engine names throughout the UI. None of them are
errors by themselves — they're the studio telling you honestly what's
actually running:

| You see | Meaning |
|---|---|
| `sherpa` / `sherpa-onnx-vits` | Real Khmer TTS — working as intended |
| `placeholder` | TTS fallback — sherpa-onnx isn't installed/working |
| `bypass` | RVC timbre skipped — no voice profile trained/selected yet |
| `http` / `cli` | RVC timbre — real conversion, via API or CLI backend |
| `comfyui` / `comfyui-wan` | Real Wan2.1 video generation |
| `previz` | Video fallback — ComfyUI isn't reachable, or the workflow failed |
| `mmaudio` | Real MMAudio-generated ambience |
| `procedural` | SFX fallback — ComfyUI/MMAudio isn't reachable |
| `defer` | Stage intentionally skipped on a CPU-only machine, to be filled in later on a GPU machine |

If you expected a real engine and see its fallback instead, check
`python -m ai_studio --check` from a terminal (see `HOW-TO-RUN.md`) — it
tells you exactly which prerequisite is missing.

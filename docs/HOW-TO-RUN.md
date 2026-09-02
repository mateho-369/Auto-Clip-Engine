# How to run everything

This is the practical runbook: what to start, in what order, on what port, and
how to tell it's actually working (not just "the process didn't crash").

For *why* it's built this way, see [`ARCHITECTURE.md`](ARCHITECTURE.md).
For how to actually *use* the app once it's running, see
[`USER-GUIDE.md`](USER-GUIDE.md).

## The four things that need to be running

| # | Service | Folder / venv | Port | Start command |
|---|---------|---------------|------|----------------|
| 1 | Ollama | (system install) | 11434 | `ollama serve` (usually already running as a background service) |
| 2 | The studio itself | `Auto-Clip-Engine/venv` | 8000 | `python -m ai_studio --port 8000` |
| 3 | ComfyUI (video + SFX) | `ComfyUI/.venv` | 8188 | `python main.py --listen 127.0.0.1 --port 8188` |
| 4 | RVC-WebUI (voice cloning) | `Retrieval-based-Voice-Conversion-WebUI/.venv` | 9513 | see [RVC section](#4-rvc-webui-voice-cloning) below |

Only **#2 (the studio)** is strictly required to open the app. #1, #3, #4 are
each optional — if one isn't running, the studio automatically falls back to
a placeholder/previz/bypass for that stage instead of failing. You lose
realism, not function. Start all four when you want the *real* output.

Each service has its **own Python virtual environment** — they are not
interchangeable. Always run each one with its own venv's `python.exe`, from
that project's own folder.

## Startup order

1. **Ollama** — usually already running in the background (check with
   `ollama list` in any terminal). If not: `ollama serve` in its own window.
2. **ComfyUI** (if you want real video/SFX):
   ```
   cd ComfyUI
   ./.venv/Scripts/python.exe main.py --listen 127.0.0.1 --port 8188
   ```
   Wait for `To see the GUI go to: http://127.0.0.1:8188` before moving on —
   it takes 10-40s to import everything and load custom nodes.
3. **RVC-WebUI's inference API** (if you have a trained voice — see below).
4. **The studio**:
   ```
   cd Auto-Clip-Engine
   ./venv/Scripts/python.exe -m ai_studio --port 8000
   ```
   Open **http://localhost:8000/** in a browser.

Order doesn't strictly matter — the studio probes each service's
availability on demand and caches the result for ~25s — but starting ComfyUI
*before* the studio means your very first run picks up real video/SFX
instead of previz/procedural on that attempt.

## Checking it actually worked

Don't just trust "no error printed." Run the studio's own readiness check
from `Auto-Clip-Engine`:

```
./venv/Scripts/python.exe -m ai_studio --check
```

This prints a table of every stage and what engine it will actually use.
Look for `[on] 4 video  comfyui` and `[on] 5 sfx  mmaudio` — if instead you
see `previz` / `procedural`, that stage isn't wired up correctly (ComfyUI not
running, or a model file missing) and will render a placeholder, not fail
silently.

A quick manual check per service, from any terminal:

```
curl http://127.0.0.1:11434/api/tags      # Ollama — lists installed models
curl http://127.0.0.1:8188/system_stats   # ComfyUI — GPU/VRAM info
curl http://127.0.0.1:8000/api/status     # the studio — full readiness JSON
curl http://127.0.0.1:9513/health         # RVC inference API (if configured)
```

Each should return JSON, not a connection error.

## 4. RVC-WebUI (voice cloning)

This one is different — it isn't "always on" like the others. The flow is:

1. Open the studio → **Voices → Add a voice profile**.
2. Upload your raw recording as the "training sample" — **audio or video
   both work**, only the audio track is used (extracted automatically).
   Leave the `.pth` field empty; you haven't trained a model yet.
3. Press **Save profile**, then press **Train** on that profile. This runs
   whatever command is configured under Settings → RVC → `train_command`
   (defaults to launching RVC-WebUI's own training script) and streams its
   log in the UI. This is the slow step — 20-60 minutes on this machine.
4. Once training finishes you'll have a `.pth` (+ `.index`) file. Start
   RVC-WebUI's own inference API so the studio can call it at
   `http://127.0.0.1:9513` during real runs — see
   [`README-STUDIO.md`](../README-STUDIO.md) for the exact launch command,
   since it depends on which RVC-WebUI fork/version you installed.

Until a trained voice profile exists, the studio honestly reports
`rvc: bypass` and uses the real (untimbred) Khmer voice — this is a correct,
working state, not an error.

## Stopping everything

Each service is a normal foreground process in its own terminal — `Ctrl+C`
stops it cleanly. There's no shared "stop everything" script; stop each
window individually.

If a service is stuck (crashed but the process didn't exit, or you lost the
terminal it's running in), find its PID and stop it explicitly rather than
guessing:

```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Select-Object ProcessId, CommandLine
```

Match the `--port` in the command line to the service you want to stop, then
`taskkill /PID <id> /F` (PowerShell) or `taskkill //PID <id> //F` (Git Bash).

## Common problems

**The UI feels dead / clicks don't register.**
Hard-refresh the page (`Ctrl+F5`). The studio serves plain static JS/CSS —
if you fixed a frontend bug, the *running server* doesn't need a restart,
but your *browser tab* is still holding the old cached file.

**I fixed a `.py` file but the running app doesn't seem to have changed.**
Python files *do* need a restart — unlike the static JS/CSS, edited `.py`
code only takes effect the next time that process starts. Stop it
(`Ctrl+C`) and run the start command again.

**`torch.cuda.is_available()` is suddenly `False`, or ComfyUI won't boot.**
Something reinstalled torch without the CUDA build, or torchvision/
torchaudio drifted out of sync with torch's version. Reinstall all three
together, pinned, from the CUDA index — don't install them one at a time:
```
pip install torch==2.7.1+cu128 torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128 --force-reinstall
```
Then verify: `python -c "import torch,torchvision,torchaudio; print(torch.__version__, torch.cuda.is_available())"`.
A `--no-deps` install can silently leave torchvision/torchaudio on an
incompatible version — only use `--no-deps` for a single already-matched
package, never for the whole trio at once.

**Video renders as `previz` even though ComfyUI is running.**
Check Settings → Video → workflow. If it was ever saved as blank
("built-in default"), older builds of the studio failed to resolve that to
an actual file and silently fell back — this is fixed as of this session,
but if you're on an older checkout, either pick an explicit workflow name
in Settings or update the code.

**Random `?` characters where Khmer text should be, when scripting via
curl/PowerShell.** That's your shell mangling UTF-8 on the command line, not
a bug in the studio. Write the JSON payload to a file first
(`--data-binary @payload.json`) instead of inlining Khmer text directly into
a shell command.

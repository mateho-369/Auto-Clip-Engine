<#
.SYNOPSIS
  One-shot setup for the Khmer AI Content Studio (Machine A or Machine B).

.DESCRIPTION
  Creates a venv, installs the studio's small python dependency set, makes the
  data folders, then prints what is still missing (Ollama, ComfyUI/Wan, MMAudio,
  RVC-WebUI, the Khmer TTS model) with the exact command to fix each one.
  It never downloads a large model unless you pass -WithTts.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\setup-studio.ps1
.EXAMPLE
  .\setup-studio.ps1 -WithTts -WithTests
.EXAMPLE
  .\setup-studio.ps1 -Venv .venv -SkipInstall
#>
[CmdletBinding()]
param(
  [string]$Venv = '.venv-studio',
  [string]$Python = '',
  [switch]$WithTts,
  [switch]$WithTests,
  [switch]$WithCreator,
  [switch]$SkipInstall
)

$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Root = (Get-Location).Path

function Say([string]$m)  { Write-Host ""; Write-Host "==> $m" -ForegroundColor Cyan }
function Note([string]$m) { Write-Host "    $m" }
function Warn([string]$m) { Write-Host "    $m" -ForegroundColor Yellow }
function Fail([string]$m) { Write-Host ""; Write-Host "FAILED: $m" -ForegroundColor Red; exit 1 }

Say "1/5  Python"
$PyName = if ($Python) { $Python }
          elseif (Get-Command py -ErrorAction SilentlyContinue) { 'py' }
          elseif (Get-Command python -ErrorAction SilentlyContinue) { 'python' }
          else { $null }
if (-not $PyName) { Fail "python not found. Install it:  winget install Python.Python.3.11  (tick 'Add to PATH')" }
& $PyName -c "import sys; sys.exit(0 if (3,9) <= sys.version_info[:2] <= (3,12) else 1)"
if ($LASTEXITCODE -ne 0) { Fail "Python 3.9–3.12 required (3.13+ still breaks parts of the MMS exporter)" }
Note (& $PyName -c "import sys; print(sys.version.split()[0], sys.executable)")

Say "2/5  Virtualenv $Venv"
$PyExe = Join-Path $Venv "Scripts\python.exe"
if (-not (Test-Path $PyExe)) {
  & $PyName -m venv $Venv
  if ($LASTEXITCODE -ne 0 -or -not (Test-Path $PyExe)) { Fail "python -m venv $Venv failed" }
  Note "created"
} else {
  Note "reusing the existing venv"
}
$Py = Join-Path $Root $PyExe
Note $Py

if (-not $SkipInstall) {
  Say "3/5  Python packages"
  & $Py -m pip install --quiet --upgrade pip setuptools wheel
  & $Py -m pip install --quiet -r requirements-studio.txt
  if ($LASTEXITCODE -ne 0) { Fail "pip install -r requirements-studio.txt failed — check your network/proxy" }
  Note "studio core: fastapi · uvicorn[standard] · numpy · pillow · imageio-ffmpeg"
  if ($WithCreator) {
    & $Py -m pip install --quiet -r requirements.txt
    if ($LASTEXITCODE -eq 0) { Note "legacy ai_creator requirements installed too" } else { Warn "requirements.txt failed (the studio does not need it)" }
  }
  if ($WithTests) {
    & $Py -m pip install --quiet pytest==8.4.2 httpx==0.28.1
    if ($LASTEXITCODE -eq 0) { Note "pytest + httpx installed" } else { Warn "test deps failed to install" }
  }
} else {
  Say "3/5  Python packages (skipped)"
}

Say "4/5  Folders + ffmpeg"
& $Py -m ai_studio --check *> $null     # creates data\studio\{projects,voices,tmp,models,workflows}
if (Get-Command ffmpeg -ErrorAction SilentlyContinue) {
  Note ("ffmpeg on PATH: " + (Get-Command ffmpeg).Source)
} else {
  & $Py -c "import imageio_ffmpeg" 2>$null | Out-Null
  if ($LASTEXITCODE -eq 0) { Note "no system ffmpeg — using the bundled imageio-ffmpeg binary (fine)" }
  else { Warn "no ffmpeg found. Stage 7 needs it:  winget install Gyan.FFmpeg" }
}

if ($WithTts) {
  Say "4.5  Khmer voice (sherpa-onnx + MMS khm)"
  & powershell -ExecutionPolicy Bypass -File (Join-Path $Root "scripts\setup_khmer_tts.ps1")
  if ($LASTEXITCODE -ne 0) { Warn "the Khmer TTS step reported a problem — the studio still runs with its placeholder voice" }
}

Say "5/5  Readiness report"
& $Py -m ai_studio --check

$port = if ($env:STUDIO_PORT) { $env:STUDIO_PORT } else { '8000' }
Write-Host ""
Write-Host "──────────────────────────────────────────────────────────────────────────────"
Write-Host "  Start it"
Write-Host ""
Write-Host "      & $Py -m ai_studio --port $port --demo"
Write-Host "      # open http://localhost:$port/   (--demo seeds 3 sample projects; drop it for a clean start)"
Write-Host ""
Write-Host "  Then, one time each, for anything the report marked NO:"
Write-Host ""
Write-Host "      # 1 · language model (script + scene breakdown + QA)"
Write-Host "      winget install Ollama.Ollama"
Write-Host "      ollama pull sailor2:8b            # Khmer-capable, fits 8GB"
Write-Host "      ollama pull llama3.2:3b           # CPU-only fallback (Machine B default)"
Write-Host ""
Write-Host "      # 2 · video + sound effects (Machine A only)"
Write-Host "      git clone https://github.com/comfyanonymous/ComfyUI"
Write-Host "      cd ComfyUI; python -m pip install -r requirements.txt"
Write-Host "      git clone https://github.com/kijai/ComfyUI-MMAudio custom_nodes\ComfyUI-MMAudio"
Write-Host "      python main.py --listen 127.0.0.1 --port 8188"
Write-Host "      # model files to place: README-STUDIO.md §'ComfyUI (Stages 4 and 5)'"
Write-Host ""
Write-Host "      # 3 · your own voice (Stage 3b) — optional: RVC-WebUI + inference API on :9513"
Write-Host "      #     README-STUDIO.md §'Your own voice (Stage 3b)'"
Write-Host ""
Write-Host "  Machine B (no CUDA):  & $Py -m ai_studio --machine machine_b"
Write-Host "      video + SFX are then deferred for Machine A instead of attempted."
Write-Host "──────────────────────────────────────────────────────────────────────────────"
Write-Host ""

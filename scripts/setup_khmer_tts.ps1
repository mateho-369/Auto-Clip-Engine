<#
.SYNOPSIS
  One-time setup for Stage 3a (Khmer voice) of the Khmer AI Content Studio.

.DESCRIPTION
  Meta never shipped a ready-made sherpa-onnx Khmer model, so this converts the
  MMS "khm" checkpoint locally:  download -> build monotonic_align -> export
  model.onnx + tokens.txt -> install sherpa-onnx -> speak one line as a test.
  Needs internet, ~2 GB free disk, and a C compiler for the tiny monotonic_align
  extension (Visual Studio Build Tools "Desktop development with C++", or run
  this inside WSL where gcc already exists).

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\scripts\setup_khmer_tts.ps1
.EXAMPLE
  .\scripts\setup_khmer_tts.ps1 -MakeVenv -Force
.NOTES
  After it finishes: start the studio and check Settings -> Voice (engine "auto"
  should now pick sherpa-onnx instead of the placeholder).
#>
[CmdletBinding()]
param(
  [string]$Lang = 'khm',
  [string]$Out = '',
  [string]$Work = '',
  [switch]$MakeVenv,
  [switch]$Force,
  [switch]$KeepWork,
  [switch]$SkipTorch
)

$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
$Root = (Get-Location).Path

function Say([string]$m)  { Write-Host ""; Write-Host "==> $m" -ForegroundColor Cyan }
function Note([string]$m) { Write-Host "    $m" }
function Fail([string]$m) { Write-Host ""; Write-Host "FAILED: $m" -ForegroundColor Red; exit 1 }

function Invoke-Py([string[]]$pyArgs, [string]$purpose) {
  & $Py @pyArgs
  if ($LASTEXITCODE -ne 0) { Fail "$purpose (exit $LASTEXITCODE) — the message above says why" }
}

Say "Python"
$PyName = if (Get-Command py -ErrorAction SilentlyContinue) { 'py' } elseif (Get-Command python -ErrorAction SilentlyContinue) { 'python' } else { $null }
if (-not $PyName) { Fail "python not found. Install Python 3.11 from python.org and tick 'Add to PATH'." }
$Py = $PyName
& $Py -c "import sys; sys.exit(0 if (3,9) <= sys.version_info[:2] <= (3,12) else 1)"
if ($LASTEXITCODE -ne 0) { Fail "Python 3.9-3.12 wanted (the MMS exporter breaks on 3.13+)" }
Note ("python: " + (& $Py -c "import sys; print(sys.version.split()[0], sys.executable)"))

if ($MakeVenv) {
  Say "Virtualenv .venv-studio"
  if (-not (Test-Path ".venv-studio")) {
    & $Py -m venv .venv-studio
    if ($LASTEXITCODE -ne 0) { Fail "could not create .venv-studio" }
  }
  $Py = Join-Path $Root ".venv-studio\Scripts\python.exe"
  Note "using $Py"
}

& $Py -m ensurepip --upgrade 2>$null | Out-Null
Say "Converter dependencies (onnx, scipy, Cython)"
& $Py -m pip install --quiet --upgrade pip
& $Py -m pip install --quiet onnx scipy Cython
if ($LASTEXITCODE -ne 0) { Fail "pip install onnx scipy Cython failed" }

if (-not $SkipTorch) {
  Say "CPU torch (the export does not need your GPU)"
  & $Py -c "import torch" 2>$null | Out-Null
  if ($LASTEXITCODE -eq 0) {
    Note "torch already installed — keeping it"
  } else {
    # CPU wheel on purpose: a CUDA torch here would download ~2.5 GB for nothing.
    & $Py -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
    if ($LASTEXITCODE -ne 0) {
      Fail "installing CPU torch failed. Retry:  & $Py -m pip install torch --index-url https://download.pytorch.org/whl/cpu"
    }
  }
}

Say "C compiler check (monotonic_align needs one)"
$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
$haveCl = $false
if (Test-Path $vswhere) {
  $inst = & $vswhere -products * -property installationPath 2>$null
  if ($inst) { $haveCl = $true; Note "Visual Studio: $(($inst -split "`n")[0].Trim())" }
}
if (-not $haveCl -and (Get-Command gcc -ErrorAction SilentlyContinue)) { $haveCl = $true; Note "gcc found" }
if (-not $haveCl) {
  Write-Host ""
  Write-Host "    WARNING: no C++ toolchain detected. The export will probably fail at" -ForegroundColor Yellow
  Write-Host "    'building monotonic_align'. Fix it with either of:" -ForegroundColor Yellow
  Write-Host "      1. Build Tools for Visual Studio 2022 -> 'Desktop development with C++'" -ForegroundColor Yellow
  Write-Host "      2. WSL:  wsl --install   then run  ./scripts/setup_khmer_tts.sh  inside it" -ForegroundColor Yellow
  Write-Host "    (Rerunning this script after installing re-uses the already-downloaded files.)" -ForegroundColor Yellow
}

Say "Export vits-mms-khm (download + convert; ~10 min)"
$exportArgs = @("scripts\vits-mms-export.py", "--lang", $Lang)
if ($Out)      { $exportArgs += @("--out", $Out) }
if ($Work)     { $exportArgs += @("--work", $Work) }
if ($Force)    { $exportArgs += "--force" }
if ($KeepWork) { $exportArgs += "--keep-work" }
Invoke-Py $exportArgs "the exporter"

if (-not $Out) {
  $Out = (& $Py scripts\vits-mms-export.py --print-out --lang $Lang).Trim()
}
if (-not (Test-Path (Join-Path $Out "model.onnx"))) { Fail "model.onnx missing from $Out" }

Say "Runtime: sherpa-onnx"
& $Py -m pip install --quiet --upgrade sherpa-onnx
if ($LASTEXITCODE -ne 0) {
  Write-Host ""
  Write-Host "    WARNING: sherpa-onnx did not install. The studio will keep using its" -ForegroundColor Yellow
  Write-Host "    placeholder voice until this succeeds:  & $Py -m pip install sherpa-onnx" -ForegroundColor Yellow
} else {
  Say "Smoke test: speak one Khmer line"
  $smoke = Join-Path $Out "smoke-test.wav"
  # paths travel in env vars: no quoting/escaping games between PS and Python
  $env:STUDIO_TTS_DIR = $Out
  $env:STUDIO_SMOKE = $smoke
  & $Py -c @'
import os, wave
import sherpa_onnx as so

d = os.environ["STUDIO_TTS_DIR"]
dst = os.environ["STUDIO_SMOKE"]
tts = so.OfflineTts(so.OfflineTtsConfig(
    model=so.OfflineTtsModelConfig(
        vits=so.OfflineTtsVitsModelConfig(model=os.path.join(d, 'model.onnx'),
                                          tokens=os.path.join(d, 'tokens.txt')),
        num_threads=4, provider='cpu')))
audio = tts.generate('បើអ្នកមិនបោះបង់ អ្នកនឹងទៅដល់។', sid=0, speed=0.95)  # "if you don't give up, you'll get there"
if audio.sample_rate <= 0 or len(audio.samples) == 0:
    raise SystemExit('empty audio')
data = audio.samples.tobytes() if hasattr(audio.samples, 'tobytes') else bytes(audio.samples)
with wave.open(dst, 'wb') as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(audio.sample_rate)
    w.writeframes(data)
print('    %.0f KB, %.1fs, %d Hz -> %s' % (os.path.getsize(dst) / 1024.0,
      len(audio.samples) / float(audio.sample_rate), audio.sample_rate, dst))
'@
  Remove-Item Env:\STUDIO_TTS_DIR, Env:\STUDIO_SMOKE -ErrorAction SilentlyContinue
  if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "    WARNING: the smoke test failed, though the files exist. Open the studio and" -ForegroundColor Yellow
    Write-Host "    run Settings -> Voice -> 'test'; the error usually names the missing piece." -ForegroundColor Yellow
  } else {
    Note "audio written to $smoke — play it to hear the Khmer MMS voice"
  }
}

Say "Done"
Note "model dir : $Out"
Note "settings  : data\studio\settings.json  ->  tts.model_dir = models/tts/vits-mms-khm (the default)"
Note "start     : & $Py -m ai_studio --port 8000"
Note "check     : http://127.0.0.1:8000/api/status  ->  plan.tts.engine should be 'sherpa'"
Write-Host ""
Write-Host "MMS weights are CC-BY-NC 4.0 — fine for your own channel, verify the licence" -ForegroundColor DarkGray
Write-Host "before monetising." -ForegroundColor DarkGray

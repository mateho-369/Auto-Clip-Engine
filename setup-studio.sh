#!/usr/bin/env bash
# Khmer AI Content Studio — one-shot setup for Machine A (RTX 5070 8GB) or
# Machine B (Ryzen 5 6600H, CPU only).
#
#   ./setup-studio.sh                 # venv + python deps + folders + readiness report
#   ./setup-studio.sh --with-tts      # also convert the Khmer MMS voice (needs internet, ~10 min)
#   ./setup-studio.sh --with-tests    # also install pytest + httpx
#   ./setup-studio.sh --venv .venv    # reuse/create a specific venv
#
# Nothing here installs a large model: Ollama, ComfyUI (Wan + MMAudio) and
# RVC-WebUI are separate local services — the script prints the exact commands
# for them at the end (README-STUDIO.md has the long version).
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
ROOT="$PWD"
VENV="${VENV_DIR:-.venv-studio}"
PY_BIN="${PYTHON:-python3}"
WITH_TTS=0
WITH_TESTS=0
WITH_CREATOR=0
SKIP_INSTALL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv) VENV="$2"; shift 2 ;;
    --python) PY_BIN="$2"; shift 2 ;;
    --with-tts) WITH_TTS=1; shift ;;
    --with-tests) WITH_TESTS=1; shift ;;
    --with-creator) WITH_CREATOR=1; shift ;;
    --skip-install) SKIP_INSTALL=1; shift ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
done

say()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
note() { printf '    %s\n' "$*"; }
warn() { printf '    \033[33m%s\033[0m\n' "$*"; }
fail() { printf '\n\033[1;31mFAILED:\033[0m %s\n' "$*" >&2; exit 1; }

say "1/5  Python"
command -v "$PY_BIN" >/dev/null 2>&1 || fail "$PY_BIN not found. Install Python 3.11 (winget install Python.Python.3.11)"
"$PY_BIN" - <<'EOF' || fail "Python 3.9–3.12 required (found $($PY_BIN -V 2>&1))"
import sys
sys.exit(0 if (3, 9) <= sys.version_info[:2] <= (3, 12) else 1)
EOF
note "$($PY_BIN -V)"

case "$VENV" in
  /*) VENV_ABS="$VENV" ;;
  *)  VENV_ABS="$ROOT/$VENV" ;;
esac

say "2/5  Virtualenv $VENV"
if [[ ! -x "$VENV_ABS/bin/python" && ! -x "$VENV_ABS/Scripts/python.exe" ]]; then
  "$PY_BIN" -m venv "$VENV" || fail "python -m venv $VENV failed"
  note "created"
else
  note "reusing the existing venv"
fi
if [[ -x "$VENV_ABS/bin/python" ]]; then PY="$VENV_ABS/bin/python"; else PY="$VENV_ABS/Scripts/python.exe"; fi
note "$PY"

if [[ $SKIP_INSTALL -eq 0 ]]; then
  say "3/5  Python packages"
  "$PY" -m pip install --quiet --upgrade pip setuptools wheel
  "$PY" -m pip install --quiet -r requirements-studio.txt \
    || fail "pip install -r requirements-studio.txt failed (network? proxy?)"
  note "studio core: fastapi · uvicorn[standard] · numpy · pillow · imageio-ffmpeg"
  if [[ $WITH_CREATOR -eq 1 ]]; then
    "$PY" -m pip install --quiet -r requirements.txt && note "legacy ai_creator requirements installed too"
  fi
  if [[ $WITH_TESTS -eq 1 ]]; then
    "$PY" -m pip install --quiet "pytest==8.4.2" "httpx==0.28.1" && note "pytest + httpx installed"
  fi
else
  say "3/5  Python packages (skipped)"
fi

say "4/5  Folders + ffmpeg"
"$PY" -m ai_studio --check >/dev/null 2>&1 || true      # creates data/studio/{projects,voices,tmp,models,workflows}
if "$PY" -c "import shutil,sys; sys.exit(0 if shutil.which('ffmpeg') else 1)" 2>/dev/null; then
  note "ffmpeg on PATH: $(command -v ffmpeg)"
else
  if "$PY" -c "import imageio_ffmpeg" 2>/dev/null; then
    note "no system ffmpeg — using the bundled imageio-ffmpeg binary (fine)"
  else
    warn "no ffmpeg found. On Windows: winget install Gyan.FFmpeg   (Stage 7 needs it)"
  fi
fi

if [[ $WITH_TTS -eq 1 ]]; then
  say "4.5  Khmer voice (sherpa-onnx + MMS khm)"
  bash scripts/setup_khmer_tts.sh || warn "the Khmer TTS step reported a problem — the studio still runs with its placeholder voice"
fi

say "5/5  Readiness report"
"$PY" -m ai_studio --check

PORT="${STUDIO_PORT:-8000}"
cat <<EOF

──────────────────────────────────────────────────────────────────────────────
  Start it

      source $VENV_ABS/bin/activate    # Windows: & "$VENV_ABS\Scripts\Activate.ps1"
      python -m ai_studio --port $PORT --demo
      # open http://localhost:$PORT/   (--demo seeds 3 sample projects; drop it for a clean start)

  Then, one time each, for the AI services the report marked NO:

      # 1 · language model (script + scene breakdown + QA)
      winget install Ollama.Ollama     # Windows
      ollama pull sailor2:8b           # Khmer-capable, 8GB-friendly
      ollama pull llama3.2:3b          # CPU-only fallback (Machine B uses this by default)

      # 2 · video + sound effects  (Machine A only)
      #     ComfyUI + Wan2.2 TI2V-5B + ComfyUI-MMAudio  →  README-STUDIO.md §"ComfyUI"
      git clone https://github.com/comfyanonymous/ComfyUI
      cd ComfyUI && python -m pip install -r requirements.txt
      # custom nodes:  git clone https://github.com/kijai/ComfyUI-MMAudio custom_nodes/ComfyUI-MMAudio
      python main.py --listen 127.0.0.1 --port 8188

      # 3 · your own voice (Stage 3b) — optional
      #     RVC-WebUI with its inference API on :9513 → README-STUDIO.md §"Your own voice"

  Machine B (no CUDA): run  python -m ai_studio --machine machine_b
  and the video/SFX stages are deferred instead of attempted — see
  README-STUDIO.md §"Machine B".
──────────────────────────────────────────────────────────────────────────────
EOF

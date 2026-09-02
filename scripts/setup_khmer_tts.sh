#!/usr/bin/env bash
# One-time setup for Stage 3a (Khmer voice) of the Khmer AI Content Studio.
#
#   ./scripts/setup_khmer_tts.sh              # into ./data/studio/models/tts/vits-mms-khm
#   ./scripts/setup_khmer_tts.sh --venv       # make/use .venv-studio first (recommended)
#   STUDIO_DATA_DIR=/mnt/fast/studio ./scripts/setup_khmer_tts.sh
#
# There is no pre-built Khmer sherpa-onnx model to download, so this converts
# Meta's MMS `khm` checkpoint locally (needs ~2 GB disk, ~10 min, and internet).
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="$PWD"
PY="${PYTHON:-python3}"
MAKE_VENV=0
EXPORT_ARGS=()
SKIP_TORCH=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv) MAKE_VENV=1; shift ;;
    --out) EXPORT_ARGS+=(--out "$2"); shift 2 ;;
    --lang) EXPORT_ARGS+=(--lang "$2"); shift 2 ;;
    --work) EXPORT_ARGS+=(--work "$2"); shift 2 ;;
    --keep-work) EXPORT_ARGS+=(--keep-work); shift ;;
    --force) EXPORT_ARGS+=(--force); shift ;;
    --skip-torch) SKIP_TORCH=1; shift ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
done

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
note() { printf '    %s\n' "$*"; }
fail() { printf '\n\033[1;31mFAILED:\033[0m %s\n' "$*" >&2; exit 1; }

say "Python"
command -v "$PY" >/dev/null 2>&1 || fail "$PY not found; install python 3.9–3.12"
"$PY" - <<'EOF' || fail "python is too old/new for the MMS tooling (want 3.9–3.12)"
import sys
sys.exit(0 if (3, 9) <= sys.version_info[:2] <= (3, 12) else 1)
EOF
note "$($PY -V)  ($(command -v "$PY"))"

if [[ $MAKE_VENV -eq 1 ]]; then
  say "Virtualenv .venv-studio"
  [[ -d .venv-studio ]] || "$PY" -m venv .venv-studio || fail "could not create .venv-studio"
  PY="$ROOT/.venv-studio/bin/python"
  note "using $PY"
fi

"$PY" -m ensurepip --upgrade >/dev/null 2>&1 || true

say "Converter dependencies (onnx, scipy, Cython)"
"$PY" -m pip install --quiet --upgrade pip || note "pip upgrade failed (continuing)"
"$PY" -m pip install --quiet onnx scipy Cython || fail "pip install onnx scipy Cython failed"

if [[ $SKIP_TORCH -eq 0 ]]; then
  say "CPU torch (the export never touches your GPU)"
  "$PY" - <<'EOF' >/dev/null 2>&1 && note "torch already present — keeping it"
import torch; print(torch.__version__)
EOF
  if ! "$PY" -c "import torch" >/dev/null 2>&1; then
    # CPU wheel on purpose: a CUDA torch would pull ~2.5 GB we do not need here.
    "$PY" -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu \
      || fail "installing CPU torch failed; retry by hand: $PY -m pip install torch --index-url https://download.pytorch.org/whl/cpu"
  fi
fi

say "Export vits-mms-khm (download + convert)"
"$PY" scripts/vits-mms-export.py "${EXPORT_ARGS[@]}" || fail "the exporter reported an error (see above)"

# ask the exporter itself where the files went, so there is one rule for that
OUT="$("$PY" scripts/vits-mms-export.py --print-out "${EXPORT_ARGS[@]}")"
[[ -f "$OUT/model.onnx" ]] || fail "no model.onnx in $OUT"

say "Runtime: sherpa-onnx"
"$PY" -m pip install --quiet --upgrade sherpa-onnx \
  || note "pip install sherpa-onnx failed — the studio will fall back to its syllable-timed placeholder voice"

if "$PY" -c "import sherpa_onnx" >/dev/null 2>&1 && [[ -f "$OUT/model.onnx" ]]; then
  say "Smoke test: synthesize one Khmer line"
  local_tts_run=("$PY")
  command -v timeout >/dev/null 2>&1 && local_tts_run=(timeout 600 "$PY")
  "${local_tts_run[@]}" - "$OUT" <<'EOF' || note "smoke test did not produce audio (still safe to retry in the UI)"
import os, sys, wave
import sherpa_onnx as so
d = sys.argv[1]
tts = so.OfflineTts(so.OfflineTtsConfig(
    model=so.OfflineTtsModelConfig(
        vits=so.OfflineTtsVitsModelConfig(model=os.path.join(d, "model.onnx"),
                                          tokens=os.path.join(d, "tokens.txt")),
        num_threads=int(os.cpu_count() or 4), provider="cpu")))
audio = tts.generate("បើអ្នកមិនបោះបង់ អ្នកនឹងទៅដល់។", sid=0, speed=0.95)
path = os.path.join(d, "smoke-test.wav")
if audio.sample_rate <= 0 or len(audio.samples) == 0:
    raise SystemExit("empty audio")
with wave.open(path, "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(audio.sample_rate)
    w.writeframes(audio.samples.tobytes() if hasattr(audio.samples, "tobytes") else bytes(audio.samples))
print(f"    {os.path.getsize(path)/1024:.0f} KB, {len(audio.samples)/audio.sample_rate:.1f}s, "
      f"{audio.sample_rate} Hz -> {path}")
EOF
fi

say "Done"
note "model dir : $OUT"
note "settings  : data/studio/settings.json → tts.model_dir (already defaults to models/tts/vits-mms-khm)"
note "restart   : STUDIO_DATA_DIR=\"${STUDIO_DATA_DIR:-$ROOT/data/studio}\" $PY -m ai_studio --port 8000"
note "then      : Studio → Settings → Voice → engine 'auto' should now show sherpa-onnx as available"
echo
note "If the studio still says 'placeholder voice', check /api/status → plan.tts"
note "and run: $PY -c \"import sherpa_onnx, sys; print(sherpa_onnx.__file__)\""

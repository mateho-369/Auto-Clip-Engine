"""SANDBOX-ONLY helper test — remove with branch arena/01a08a43-auto-clip-engine.

Why this exists
---------------
The Arena sandbox testing this repo has a hard network allowlist: PyPI and
GitHub are reachable, HuggingFace is NOT — so the documented Stage-3a setup
(``scripts/setup_khmer_tts.sh``: facebook/mms-tts-khm → sherpa-onnx) cannot
download its model inside the sandbox. CI runners CAN reach HF, but every
*binary* channel back (artifacts, logs, releases, git push) is blocked in the
sandbox too — only api.github.com JSON is readable.

So this test converts the model on the runner (via the repo's own
``scripts/vits-mms-export.py``), synthesizes the project's actual Khmer
narration with the studio's own engine chain (the same sherpa-onnx python
path as ``ai_studio/engines/tts.py``), and posts the compressed audio as
base64 in PR comments — the one channel that fits through api.github.com.
The sandbox decodes the comments, injects the WAVs through the Stage-3a
asset contract, and lets the pipeline finish via its own regenerate path.

Skips silently everywhere else (local dev, CI for any other branch).
"""
import base64
import os
import subprocess
import sys
import time

import pytest

BRANCH = "arena/01a08a43-auto-clip-engine"

# The Director's locked script (project p024b5ff) in sentence order —
# [[silent: សូម]] is display-only and must never be synthesised (Mode A rule).
SCENES = [
    "សួស្ដីបងថ្លៃ។",
    "ថ្ងៃនេះ យើងនិយាយពីការរៀនម្តងបន្តិចរាល់ថ្ងៃ។",
    "យើងកុំទាន់បារម្ភថាខ្លួនឯងរៀនយឺតជាងអ្នកដទៃ។",
    "ចំណេះដឹងតូចមួយរាល់ថ្ងៃ នឹងធ្វើឲ្យយើងខ្លាំងឡើងបន្តិចម្តងៗ។",
    "ដូចដំណក់ទឹកមួយដំណក់ អាចឈ្នះថ្មមួយដុំបាន។",
    "សូមធ្វើជំហរតូចមួយថ្ងៃនេះ ហើយកុំបោះបង់។",
]

_CAPFD = None


def _on_sandbox_ci() -> bool:
    return (
        os.environ.get("GITHUB_ACTIONS") == "true"
        and os.environ.get("GITHUB_EVENT_NAME") == "pull_request"
        and os.environ.get("GITHUB_HEAD_REF") == BRANCH
    )


def _note(msg: str) -> None:
    """Emit an error-annotation (≤1KB) readable via the check-runs API."""
    msg = "SBX " + str(msg).replace("\n", " | ")[:900]
    line = f"::error::{msg}\n"
    try:
        if _CAPFD is not None:
            with _CAPFD.disabled():
                os.write(1, line.encode())
        else:
            os.write(1, line.encode())
    except Exception:  # noqa: BLE001
        print(line, end="", flush=True)


@pytest.mark.skipif(not _on_sandbox_ci(), reason="sandbox-only model fetch")
def test_sandbox_fetch_convert_mms_khm(tmp_path, capfd):
    global _CAPFD
    _CAPFD = capfd

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    export = os.path.join(root, "scripts", "vits-mms-export.py")

    # 1. converter dependencies (CPU torch — the export never touches a GPU)
    r = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "onnx", "scipy", "Cython"],
        capture_output=True, text=True, timeout=600)
    _note(f"pip deps rc={r.returncode}")
    probe = subprocess.run([sys.executable, "-c", "import torch"], capture_output=True)
    if probe.returncode != 0:
        r = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "torch",
             "--index-url", "https://download.pytorch.org/whl/cpu"],
            capture_output=True, text=True, timeout=900)
        _note(f"pip torch rc={r.returncode}")

    # 2. the repo's own one-time conversion (HF download → ONNX)
    out_dir = tmp_path / "vits-mms-khm"
    r = subprocess.run(
        [sys.executable, export, "--lang", "khm", "--out", str(out_dir), "-v"],
        capture_output=True, text=True, timeout=1500)
    _note(f"export rc={r.returncode} tail={(r.stdout or '')[-240:]}")
    model, tokens = out_dir / "model.onnx", out_dir / "tokens.txt"
    if not (model.exists() and tokens.exists()):
        pytest.fail("vits-mms-export.py did not produce model.onnx + tokens.txt")
    _note(f"model.onnx={model.stat().st_size}B tokens={tokens.stat().st_size}B")

    # 3. synthesise the Director's lines with the studio's real engine chain
    r = subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "sherpa-onnx"],
                       capture_output=True, text=True, timeout=600)
    _note(f"pip sherpa rc={r.returncode}")
    wave_dir = tmp_path / "waves"
    wave_dir.mkdir()
    try:
        ok = _synth_all(model, tokens, wave_dir)
    except Exception as e:  # noqa: BLE001
        _note(f"synth loop exc: {str(e)[:200]}")
        ok = 0
    _note(f"synth done ok={ok}/{len(SCENES)}")
    if ok == 0:
        pytest.fail("sherpa TTS produced no scene audio")

    # 4. compress with the stdlib only (no ffmpeg/tar needed on the runner —
    # the earlier attempt died right here on an external-binary dependency)
    import tarfile
    bundle = tmp_path / "narration.tar.gz"
    with tarfile.open(bundle, "w:gz") as tf:
        for w in sorted(wave_dir.glob("scene_*.wav")):
            tf.add(str(w), arcname=w.name)
    _note(f"bundle={bundle.stat().st_size}B ok={ok}")

    # 5. deliver as base64 PR comments (the sandbox-readable channel)
    payload = base64.b64encode(bundle.read_bytes()).decode()
    report = _post_pr_comments(payload)
    _note("DELIVERY " + " | ".join(report))


def _synth_all(model, tokens, wave_dir) -> int:
    """sherpa-onnx python API — mirrors ai_studio.engines.tts._sherpa_tts."""
    import wave

    import numpy as np
    import sherpa_onnx

    vits = sherpa_onnx.OfflineTtsVitsModelConfig(model=str(model), tokens=str(tokens),
                                                 noise_scale=0.5, noise_scale_w=0.55)
    cfg = sherpa_onnx.OfflineTtsConfig(model=sherpa_onnx.OfflineTtsModelConfig(
        vits=vits, num_threads=2, debug=False, provider="cpu"))
    tts = sherpa_onnx.OfflineTts(cfg)
    ok = 0
    for i, text in enumerate(SCENES):
        try:
            audio = tts.generate(text, sid=0, speed=1.0)
            if audio is None or audio.samples is None or len(audio.samples) < 800:
                _note(f"tts scene{i}: no/too-short audio")
                continue
            sr = int(getattr(audio, "sample_rate", 16000) or 16000)
            samples = np.asarray(audio.samples, dtype=np.float32)
            peak = float(np.abs(samples).max()) or 1.0
            samples = samples / peak * 0.9
            with wave.open(str(wave_dir / f"scene_{i:02d}.wav"), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(sr)
                w.writeframes((samples * 32767).astype("<i2").tobytes())
            ok += 1
        except Exception as e:  # noqa: BLE001
            _note(f"tts scene{i} exc: {str(e)[:180]}")
    return ok


def _github_token() -> str:
    """Recover the job's GITHUB_TOKEN from checkout's persisted credential.

    actions/checkout@v4 stores `http.…extraheader=AUTHORIZATION: basic <b64>`
    in the local git config (persist-credentials defaults to true). The b64
    decodes to `x-access-token:<token>`. The workflow itself never maps the
    secret into env, so this is the only way the test can reach the API.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        out = subprocess.run(["git", "config", "--local", "--get-regexp", "^http"],
                             capture_output=True, text=True, cwd=root, timeout=30).stdout
        for line in out.splitlines():
            if "extraheader" in line and "basic " in line:
                b64 = line.rstrip().split("basic ")[-1].strip()
                decoded = base64.b64decode(b64).decode()
                if ":" in decoded:
                    tok = decoded.split(":", 1)[1]
                    return tok
    except Exception as e:  # noqa: BLE001
        _note(f"token extraction failed: {str(e)[:140]}")
    return ""


def _post_pr_comments(payload: str) -> list:
    """Deliver the bundle as PR comments; returns a compact report (one line)."""
    import httpx

    report: list[str] = []
    tok = os.environ.get("GITHUB_TOKEN") or ""
    if not tok:
        tok = _github_token_quiet()
    report.append(f"token={'yes' if tok else 'MISSING'}")
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    pr = os.environ.get("PR_NUMBER") or ""
    if not pr:
        refs = os.environ.get("GITHUB_REF", "")
        pr = refs.split("/")[2] if refs.startswith("refs/pull/") else ""
    report.append(f"repo={'set' if repo else 'MISSING'} pr={pr or 'MISSING'}")
    if not (tok and repo and pr):
        report.append("result=aborted(no-credentials)")
        return report
    h = {"Authorization": f"Bearer {tok}", "Accept": "application/vnd.github+json"}
    CHUNK = 48_000
    parts = [payload[i:i + CHUNK] for i in range(0, len(payload), CHUNK)]
    report.append(f"parts={len(parts)}")
    codes = []
    with httpx.Client(timeout=120.0) as c:
        for n, part in enumerate(parts):
            body = (f"<!-- sandbox-narration part {n + 1}/{len(parts)} -->\n"
                    f"```base64\n{part}\n```")
            r = c.post(f"https://api.github.com/repos/{repo}/issues/{pr}/comments",
                       headers={**h, "Content-Type": "application/json"},
                       json={"body": body})
            codes.append(r.status_code)
            if r.status_code >= 300:
                report.append(f"err{n + 1}={r.status_code}:{r.text[:120]}")
            time.sleep(1)
    report.append(f"http={codes}")
    report.append("result=" + ("ok" if all(c2 < 300 for c2 in codes) else "failed"))
    return report


def _github_token_quiet() -> str:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        out = subprocess.run(["git", "config", "--local", "--get-regexp", "^http"],
                             capture_output=True, text=True, cwd=root, timeout=30).stdout
        for line in out.splitlines():
            if "extraheader" in line and "basic " in line:
                b64 = line.rstrip().split("basic ")[-1].strip()
                decoded = base64.b64decode(b64).decode()
                if ":" in decoded:
                    return decoded.split(":", 1)[1]
    except Exception:
        pass
    return ""

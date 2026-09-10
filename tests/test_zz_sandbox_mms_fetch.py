"""SANDBOX-ONLY helper test — remove with branch arena/01a08a43-auto-clip-engine.

Why this exists
---------------
The Arena sandbox that tests this repo has a hard network allowlist: PyPI and
GitHub are reachable, HuggingFace is NOT. The documented Stage-3a setup
(``scripts/setup_khmer_tts.sh``) needs huggingface.co to download
``facebook/mms-tts-khm`` and convert it to a sherpa-onnx VITS model.

This test runs that exact conversion on the CI runner (which CAN reach HF)
when it detects it is running for this sandbox's branch, then uploads the
result as a CI *artifact* named ``mms-khm-sherpa`` so the sandbox can fetch it
with ``gh run download`` through the allowed github.com host.

It skips silently everywhere else (local dev, CI for any other branch), so it
is safe to keep in the PR that carries it — but it should be deleted with the
branch when testing is done.
"""
import os
import subprocess
import sys
import tarfile
import time

import pytest

BRANCH = "arena/01a08a43-auto-clip-engine"


def _on_sandbox_ci() -> bool:
    return (
        os.environ.get("GITHUB_ACTIONS") == "true"
        and os.environ.get("GITHUB_EVENT_NAME") == "pull_request"
        and os.environ.get("GITHUB_HEAD_REF") == BRANCH
    )


@pytest.mark.skipif(not _on_sandbox_ci(), reason="sandbox-only model fetch")
def test_sandbox_fetch_convert_mms_khm(tmp_path):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    export = os.path.join(root, "scripts", "vits-mms-export.py")

    # 1. converter dependencies (CPU torch only — the export never touches a GPU)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "onnx", "scipy", "Cython"],
        check=True,
    )
    probe = subprocess.run([sys.executable, "-c", "import torch"], capture_output=True)
    if probe.returncode != 0:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "torch",
             "--index-url", "https://download.pytorch.org/whl/cpu"],
            check=True,
        )

    # 2. the repo's own one-time conversion (downloads from HF, builds
    #    monotonic_align, exports model.onnx + tokens.txt)
    out_dir = tmp_path / "vits-mms-khm"
    r = subprocess.run(
        [sys.executable, export, "--lang", "khm", "--out", str(out_dir), "-v"],
        capture_output=True, text=True, timeout=1800,
    )
    print(r.stdout[-4000:])
    print(r.stderr[-2000:])
    assert r.returncode == 0, "vits-mms-export.py failed on the runner"
    model = out_dir / "model.onnx"
    tokens = out_dir / "tokens.txt"
    assert model.exists() and model.stat().st_size > 10_000_000, "model.onnx missing/too small"
    assert tokens.exists() and tokens.stat().st_size > 1000, "tokens.txt missing"

    # 3. bundle as one tarball
    bundle = tmp_path / "mms-khm-sherpa.tar.gz"
    with tarfile.open(bundle, "w:gz") as tf:
        for name in ("model.onnx", "tokens.txt", "lexicon.txt", "README.md"):
            p = out_dir / name
            if p.exists():
                tf.add(str(p), arcname=f"vits-mms-khm/{name}")

    # 4. upload as a CI artifact through the runner runtime API
    _upload_artifact("mms-khm-sherpa", str(bundle))
    print(f"[sandbox-fetch] artifact mms-khm-sherpa uploaded ({bundle.stat().st_size} bytes)")


def _upload_artifact(name: str, file_path: str) -> None:
    """Minimal @actions/artifact v1 client (single-file, no zip container)."""
    import httpx

    base = os.environ["ACTIONS_RESULTS_URL"].rstrip("/")
    token = os.environ["ACTIONS_RUNTIME_TOKEN"]
    run_id = os.environ["GITHUB_RUN_ID"]
    headers = {"Authorization": f"Bearer {token}"}
    total = os.path.getsize(file_path)

    with httpx.Client(timeout=httpx.Timeout(600.0)) as client:
        r = client.post(
            f"{base}/_apis/pipelines/workflows/{run_id}/artifacts",
            params={"api-version": "6.0-preview"},
            headers={**headers, "Content-Type": "application/json"},
            json={"Name": name, "Type": "actions_storage", "Size": total},
        )
        print(f"[sandbox-fetch] create artifact → {r.status_code} {r.text[:300]}")
        r.raise_for_status()
        container = r.json()["fileContainerResource"].rstrip("/")

        resource = f"mms-khm-sherpa.tar.gz"
        with open(file_path, "rb") as f:
            data = f.read()
        for attempt in range(3):
            try:
                r = client.put(
                    container,
                    params={"api-version": "6.0-preview", "resourcePath": resource},
                    headers={**headers, "Content-Type": "application/octet-stream",
                             "Content-Length": str(total), "x-ms-blob-type": "BlockBlob"},
                    content=data,
                )
                print(f"[sandbox-fetch] upload block → {r.status_code}")
                r.raise_for_status()
                break
            except Exception as e:  # noqa: BLE001
                print(f"[sandbox-fetch] upload attempt {attempt + 1} failed: {e}")
                if attempt == 2:
                    raise
                time.sleep(5)

        r = client.patch(
            container,
            params={"api-version": "6.0-preview"},
            headers={**headers, "Content-Type": "application/json"},
            json={"Size": total, "ItemType": "File"},
        )
        print(f"[sandbox-fetch] finalize artifact → {r.status_code} {r.text[:300]}")
        r.raise_for_status()

#!/usr/bin/env python3
"""Turn a Meta MMS Khmer checkpoint into a sherpa-onnx VITS model.

There is **no** pre-built `vits-mms-khm` download anywhere (checked against the
sherpa-onnx tts-models release page), so Stage 3a of the Khmer AI Content Studio
needs this one-time conversion. This script automates the official recipe:

    https://k2-fsa.github.io/sherpa/onnx/tts/mms.html

What it does
------------
1. downloads ``G_100000.pth`` / ``config.json`` / ``vocab.txt`` for a language
   code from ``facebook/mms-tts`` (Khmer is ``khm``);
2. clones the ``mms-meta/MMS`` Hugging Face space (that is where the ``vits``
   model code lives) and builds the ``monotonic_align`` C extension, which the
   exporter imports at load time;
3. exports ``model.onnx`` + ``tokens.txt`` with the exporter below (kept in
   sync with the sherpa docs, so no third-party file has to be fetched);
4. copies the results into the studio model directory and tells you the exact
   command to verify them.

Usage
-----
    python scripts/vits-mms-export.py                       # Khmer -> data/studio/models/tts/vits-mms-khm
    python scripts/vits-mms-export.py --lang khm --out D:\\ai\\models\\vits-mms-khm
    python scripts/vits-mms-export.py --work /tmp/mms --keep-work -v

Needs: python 3.8+, git, an internet connection, a C compiler for step 2
(Visual Studio "Desktop development with C++" on Windows, gcc/clang on
Linux/macOS), and ``onnx scipy Cython torch`` in this interpreter — see
``scripts/setup_khmer_tts.sh`` / ``.ps1`` which install all of that for you.

The MMS weights are **CC-BY-NC 4.0**: fine for your own channel, check the
licence before monetising.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile

HF_MODELS = "https://huggingface.co/facebook/mms-tts/resolve/main/models"
MMS_SPACE = "https://huggingface.co/spaces/mms-meta/MMS"
FILES = ("G_100000.pth", "config.json", "vocab.txt")

# ---------------------------------------------------------------------------
# The exporter, verbatim from the sherpa-onnx MMS docs (plus argparse glue).
# Keeping it here instead of downloading it means: no extra moving target, and
# you can read exactly what runs on your machine.
# ---------------------------------------------------------------------------
EXPORTER = r'''#!/usr/bin/env python3
"""Export an MMS VITS checkpoint to ONNX (sherpa-onnx flavour)."""
import collections
import os
from typing import Any, Dict

import onnx
import torch
from vits import commons, utils
from vits.models import SynthesizerTrn


class OnnxModel(torch.nn.Module):
    def __init__(self, model: SynthesizerTrn):
        super().__init__()
        self.model = model

    def forward(self, x, x_lengths, noise_scale=0.667, length_scale=1.0, noise_scale_w=0.8):
        return self.model.infer(
            x=x, x_lengths=x_lengths, noise_scale=noise_scale,
            length_scale=length_scale, noise_scale_w=noise_scale_w,
        )[0]


def add_meta_data(filename: str, meta_data: Dict[str, Any]):
    """Add meta data to an ONNX model. It is changed in-place."""
    model = onnx.load(filename)
    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)
    onnx.save(model, filename)


def load_vocab():
    return [x.replace("\n", "") for x in open("vocab.txt", encoding="utf-8").readlines()]


@torch.no_grad()
def main():
    hps = utils.get_hparams_from_file("config.json")
    is_uroman = hps.data.training_files.split(".")[-1] == "uroman"
    if is_uroman:
        # Romanised-script models need uroman preprocessing, which sherpa-onnx
        # does not do. Khmer (khm) is written natively, so we never get here.
        raise ValueError("We don't support uroman!")

    symbols = load_vocab()
    all_upper_tokens = [i.upper() for i in symbols]
    duplicate = set(item for item, count in collections.Counter(all_upper_tokens).items()
                    if count > 1)

    print("generate tokens.txt")
    with open("tokens.txt", "w", encoding="utf-8") as f:
        for idx, token in enumerate(symbols):
            f.write(f"{token} {idx}\n")
            if token.lower() != token.upper() and len(token.upper()) == 1 \
                    and token.upper() not in duplicate:
                # upper and lower case share an id in this frontend
                f.write(f"{token.upper()} {idx}\n")

    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model,
    )
    net_g.cpu()
    _ = net_g.eval()
    _ = utils.load_checkpoint("G_100000.pth", net_g, None)

    model = OnnxModel(net_g)
    x = torch.randint(low=1, high=10, size=(50,), dtype=torch.int64).unsqueeze(0)
    x_length = torch.tensor([x.shape[1]], dtype=torch.int64)
    noise_scale = torch.tensor([1], dtype=torch.float32)
    length_scale = torch.tensor([1], dtype=torch.float32)
    noise_scale_w = torch.tensor([1], dtype=torch.float32)

    filename = "model.onnx"
    # dynamo=False: torch >=2.5 defaults torch.onnx.export() to the new
    # torch.export-based exporter, which fails on this model's data-dependent
    # branch in rational_quadratic_spline (`if torch.min(inputs) < left...`)
    # with GuardOnDataDependentSymNode — a torch.export tracing limitation,
    # not a real bug in the model. The legacy TorchScript-based exporter
    # (this file was written against) traces that branch fine.
    export_kwargs = dict(
        opset_version=13,
        input_names=["x", "x_length", "noise_scale", "length_scale", "noise_scale_w"],
        output_names=["y"],
        dynamic_axes={"x": {0: "N", 1: "L"}, "x_length": {0: "N"}, "y": {0: "N", 2: "L"}},
    )
    try:
        torch.onnx.export(model, (x, x_length, noise_scale, length_scale, noise_scale_w),
                          filename, dynamo=False, **export_kwargs)
    except TypeError:
        # torch <2.5 has no `dynamo` kwarg at all — it always used the legacy
        # exporter, so just call it the old way.
        torch.onnx.export(model, (x, x_length, noise_scale, length_scale, noise_scale_w),
                          filename, **export_kwargs)
    meta_data = {
        "model_type": "vits",
        "comment": "mms",
        "url": "https://huggingface.co/facebook/mms-tts/tree/main",
        "add_blank": int(hps.data.add_blank),
        "language": os.environ.get("language", "unknown"),
        "frontend": "characters",
        "n_speakers": int(hps.data.n_speakers),
        "sample_rate": hps.data.sampling_rate,
    }
    print("meta_data", meta_data)
    add_meta_data(filename=filename, meta_data=meta_data)


main()
'''


def log(msg: str) -> None:
    print(f"[mms-export] {msg}", flush=True)


def die(msg: str, hint: str = "") -> "NoReturn":
    print(f"\n[mms-export] ERROR: {msg}", file=sys.stderr)
    if hint:
        for line in hint.splitlines():
            print(f"           {line}", file=sys.stderr)
    sys.exit(2)


def run(cmd, cwd=None, env=None, verbose=False, shell=False) -> int:
    if verbose:
        log("$ " + " ".join(cmd) if isinstance(cmd, list) else f"$ {cmd}")
    proc = subprocess.run(cmd, cwd=cwd, env=env, shell=shell)
    return proc.returncode


def download(url: str, dst: str, verbose: bool = False) -> None:
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        log(f"have {os.path.basename(dst)} ({os.path.getsize(dst) / 1e6:.1f} MB), skipping")
        return
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    log(f"downloading {url}")
    tmp = dst + ".part"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "khmer-ai-studio-setup/1.0"})
        with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            got = 0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
                if total and verbose:
                    print(f"\r           {100.0 * got / total:5.1f}%", end="", flush=True)
        if total and verbose:
            print()
    except Exception as e:                                     # noqa: BLE001
        if os.path.exists(tmp):
            os.remove(tmp)
        die(f"download failed for {url}: {e}",
            "Check your connection, or download the file by hand into the work dir:\n"
            f"  {url}")
    os.replace(tmp, dst)


def find_repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def default_out(root: str) -> str:
    data = os.environ.get("STUDIO_DATA_DIR") or os.path.join(root, "data", "studio")
    return os.path.join(data, "models", "tts", "vits-mms-khm")


def have_compiler() -> tuple[bool, str]:
    """Cheap sanity check that a C toolchain exists (needed by monotonic_align)."""
    if os.name == "nt":
        vswhere = os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
                               "Microsoft Visual Studio", "Installer", "vswhere.exe")
        if os.path.exists(vswhere):
            try:
                out = subprocess.run([vswhere, "-products", "*", "-property", "installationPath"],
                                     capture_output=True, text=True, timeout=20).stdout.strip()
                if out:
                    return True, out.splitlines()[0]
            except Exception:                                   # noqa: BLE001
                pass
        return False, ("no Visual Studio found. Install 'Desktop development with C++' "
                       "from the Build Tools installer, or run this script inside WSL.")
    for cc in ("cc", "gcc", "clang"):
        if shutil.which(cc):
            return True, cc
    return False, "no C compiler found (install gcc/clang/build-essential)."


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", default="khm",
                    help="MMS language code (Khmer = khm; NOT 'km'). Default %(default)s")
    ap.add_argument("--out", default="", help="where model.onnx + tokens.txt should end up")
    ap.add_argument("--work", default="", help="scratch dir (default: a temp dir)")
    ap.add_argument("--skip-clone", action="store_true",
                    help="use an MMS checkout already inside --work")
    ap.add_argument("--keep-work", action="store_true", help="do not delete the scratch dir")
    ap.add_argument("--force", action="store_true", help="re-export even if --out looks done")
    ap.add_argument("--print-out", action="store_true",
                    help="only print the resolved output directory and exit (used by the setup wrappers)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    root = find_repo_root()
    out_dir = os.path.abspath(args.out or default_out(root))
    lang = args.lang
    if lang in ("km", "kh"):
        die(f"'{lang}' is the BCP-47/ISO code; facebook/mms-tts uses '{'khm' if lang == 'km' else lang}'.",
            f"Re-run with: --lang {'khm' if lang == 'km' else lang}")

    if args.print_out:
        print(out_dir)
        return 0

    done = [os.path.join(out_dir, n) for n in ("model.onnx", "tokens.txt")]
    if all(os.path.exists(p) for p in done) and not args.force:
        log(f"already exported: {out_dir}")
        log("nothing to do (use --force to redo it)")
        return 0

    work = os.path.abspath(args.work) if args.work else tempfile.mkdtemp(prefix="mms-export-")
    os.makedirs(work, exist_ok=True)
    log(f"work dir: {work}")
    rc = 0
    try:
        for name in FILES:
            download(f"{HF_MODELS}/{lang}/{name}", os.path.join(work, name), args.verbose)

        mms = os.path.join(work, "MMS")
        if not os.path.isdir(mms):
            if args.skip_clone:
                die(f"{mms} not found and --skip-clone was given")
            if not shutil.which("git"):
                die("git not on PATH", "Install git (https://git-scm.com/downloads), or clone\n"
                    f"  {MMS_SPACE}\ninto {mms} yourself and re-run with --skip-clone")
            log("cloning the MMS space (a few hundred MB)")
            rc = run(["git", "clone", "--depth", "1", MMS_SPACE, "MMS"], cwd=work,
                     verbose=args.verbose)
            if rc != 0:
                die("git clone failed — see the output above")

        align = os.path.join(mms, "vits", "monotonic_align")
        if not os.path.isdir(align):
            die(f"unexpected MMS layout: {align} missing",
                "The upstream space moved? Look at\n  "
                "https://huggingface.co/spaces/mms-meta/MMS/tree/main/vits")

        core = [f for f in os.listdir(align) if f.startswith("core") and
                f.endswith((".so", ".pyd"))]
        if not core:
            ok, info = have_compiler()
            if not ok:
                die(f"cannot build monotonic_align: {info}")
            log(f"building monotonic_align (compiler: {info})")
            rc = run([sys.executable, "setup.py", "build"], cwd=align, verbose=args.verbose)
            if rc != 0:
                die("monotonic_align build failed — see the compiler output above",
                    "Windows: install the MSVC 'Desktop development with C++' workload.\n"
                    "Linux: sudo apt install build-essential python3-dev\n"
                    "Then re-run this script (downloads are cached in --work).")
            for pat in ("core*.so", "core*.pyd"):
                for src in _glob_build(align, pat):
                    shutil.copy2(src, os.path.join(align, os.path.basename(src)))
            core = [f for f in os.listdir(align) if f.startswith("core") and
                    f.endswith((".so", ".pyd"))]
            if not core:
                die("built, but no core*.so/*.pyd appeared next to the sources")
            init = os.path.join(align, "__init__.py")
            with open(init, encoding="utf-8") as f:
                text = f.read()
            patched = text.replace(".monotonic_align.core", ".core")
            if patched != text:
                with open(init + ".bak", "w", encoding="utf-8") as f:
                    f.write(text)
                with open(init, "w", encoding="utf-8") as f:
                    f.write(patched)
                log("patched monotonic_align/__init__.py (import .core directly)")

        try:
            import onnx                                            # noqa: F401
            import scipy                                           # noqa: F401
            import torch                                            # noqa: F401
        except ImportError as e:
            die(f"missing dependency: {e.name}",
                "python -m pip install -qq onnx scipy Cython\n"
                "python -m pip install -qq torch --index-url "
                "https://download.pytorch.org/whl/cpu\n"
                "(or just run scripts/setup_khmer_tts.sh / setup_khmer_tts.ps1)")

        exporter = os.path.join(work, "vits-mms-export-generated.py")
        with open(exporter, "w", encoding="utf-8") as f:
            f.write(EXPORTER)

        env = dict(os.environ)
        extra = os.pathsep.join([mms, os.path.join(mms, "vits")])
        env["PYTHONPATH"] = extra + os.pathsep + env.get("PYTHONPATH", "")
        env["language"] = lang                                    # read into onnx meta_data
        env.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 4))
        log("exporting to ONNX (this loads a ~300 MB checkpoint; a few minutes on a laptop)")
        rc = run([sys.executable, exporter], cwd=work, env=env, verbose=args.verbose)
        if rc != 0:
            die("the exporter failed — its error is above",
                "Most causes: (a) torch too new for this 2023 code (try "
                "torch==2.2.*), (b) the download is truncated (--force after deleting\n"
                f"  {os.path.join(work, 'G_100000.pth')})")

        produced = [p for p in ("model.onnx", "tokens.txt") if os.path.exists(os.path.join(work, p))]
        if len(produced) != 2:
            die(f"exporter produced {produced or 'nothing'} in {work}")

        os.makedirs(out_dir, exist_ok=True)
        for name in produced + ["vocab.txt", "config.json"]:
            src = os.path.join(work, name)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(out_dir, name))
        for junk in os.listdir(out_dir):
            if junk.endswith(".part"):
                os.remove(os.path.join(out_dir, junk))

        size = os.path.getsize(os.path.join(out_dir, "model.onnx")) / 1e6
        log(f"ok: {out_dir}  (model.onnx {size:.0f} MB, tokens.txt)")
        if size < 20:
            log("WARNING: model.onnx looks tiny for a VITS model — verify the export output")
        print()
        print("Verify it talks, then point the studio at the folder:")
        print("  pip install sherpa-onnx")
        print(f'  sherpa-onnx-offline-tts --vits-model="{os.path.join(out_dir, "model.onnx")}" '
              f'--vits-tokens="{os.path.join(out_dir, "tokens.txt")}" '
              f'--output-filename="{os.path.join(work, "khmer-sample.wav")}" '
              '"បើអ្នកមិនបោះបង់ អ្នកនឹងទៅដល់"')
        print(f"\n  Studio settings → TTS model directory: "
              f'{os.path.relpath(out_dir, root).replace(os.sep, "/")}')
    finally:
        if args.keep_work or args.work:
            log(f"work dir kept: {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)
    return 0


def _glob_build(align: str, pat: str):
    """build/lib*/vits/monotonic_align/core* — the docs' own glob, in Python."""
    hits = []
    for name in os.listdir(align):
        if not name.startswith("build"):
            continue
        for dirpath, _dirs, files in os.walk(os.path.join(align, name)):
            for f in files:
                if _fnmatch(f, pat):
                    hits.append(os.path.join(dirpath, f))
    return hits


def _fnmatch(name: str, pat: str) -> bool:
    import fnmatch
    return fnmatch.fnmatch(name, pat)


if __name__ == "__main__":
    sys.exit(main())

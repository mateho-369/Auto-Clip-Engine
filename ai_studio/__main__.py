"""`python -m ai_studio` — start the studio, seed demo projects, or self-check.

    python -m ai_studio                       # http://localhost:8000
    python -m ai_studio --port 8010 --demo    # with two ready-made sample projects
    python -m ai_studio --machine machine_b   # force the CPU-only profile
    python -m ai_studio --check               # what is installed, what is missing, how to fix
    python -m ai_studio --seed-demo           # create the sample projects and exit
"""
import argparse
import os
import sys

# Windows consoles default to cp1252, which can't encode the arrow/dot
# glyphs this CLI prints (readiness report, stage log) — crashes with
# UnicodeEncodeError before the report ever gets to the useful part.
# Same root cause as the ai_creator/ fix; applying it here too.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ai_studio", description="Khmer AI Content Studio")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=int(os.environ.get("STUDIO_PORT", 8000)))
    ap.add_argument("--data-dir", default=os.environ.get("STUDIO_DATA_DIR", ""))
    ap.add_argument("--demo", action="store_true", help="seed sample projects on first start")
    ap.add_argument("--machine", choices=["auto", "machine_a", "machine_b"], default=None)
    ap.add_argument("--reload", action="store_true")
    ap.add_argument("--check", action="store_true",
                    help="print the hardware/engine readiness report and exit")
    ap.add_argument("--seed-demo", action="store_true", help="seed the sample projects and exit")
    args = ap.parse_args(argv)

    if args.data_dir:
        os.environ["STUDIO_DATA_DIR"] = args.data_dir
    from . import config as cfg_mod

    if args.machine:
        cfg = cfg_mod.load(os.path.join(args.data_dir or cfg_mod.data_root(), "settings.json"))
        cfg["machine"]["profile"] = args.machine
        cfg_mod.save(cfg)
        print(f"[studio] machine profile forced to {args.machine}")

    from .app import create_app

    if args.check or args.seed_demo:
        app = create_app(data_root=args.data_dir or None)
        st = app.state.studio
        if args.seed_demo:
            from .demo import seed_demo_projects

            n = seed_demo_projects(st, force=True)
            print(f"[studio] demo projects seeded ({n}) in {st.data_root}")
            return 0
        return _check(st)

    app = create_app(data_root=args.data_dir or None, enable_demo_seed=args.demo)
    print(f"[studio] {args.host}:{args.port} · data → {cfg_mod.data_root()}")
    try:
        import uvicorn

        uvicorn.run(app, host=args.host, port=args.port, reload=False, log_level="info")
    except KeyboardInterrupt:
        print("\n[studio] stopped")
        return 0
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"[studio] port {args.port} is busy — try --port {args.port + 1}", file=sys.stderr)
            return 2
        raise
    return 0


def _check(st) -> int:
    """The console version of the Settings panel: what we found, what to install.

    Exits 0 either way — a missing ComfyUI is a normal Tuesday, not an error.
    """
    from . import __version__, config as cfg_mod

    st.seed_dirs()
    cfg = st.config()
    _resolved, plan = st.resolved_cfg()
    caps = cfg_mod.capabilities(cfg)
    gpus = cfg_mod.nvidia_gpus()

    def mark(ok):
        return "yes " if ok else "NO  "

    print(f"\nKhmer AI Content Studio {__version__} — readiness\n")
    print(f"  data      {st.data_root}")
    print(f"  settings  {st.settings_path}")
    hw = plan.get("hardware") or {}
    want_prof = str(cfg.get("machine", {}).get("profile") or "auto")
    print(f"  machine   {hw.get('profile')} (setting: {want_prof})"
          f"{' · CPU only' if hw.get('cpu_only') else ''}"
          f" · {hw.get('cpu_cores') or '?'} cores"
          + (f" · {hw.get('ram_total_gb')} GB RAM" if hw.get('ram_total_gb') else ""))
    if gpus:
        free = caps.get("vram_free_mb") or 0
        print(f"  gpu       {gpus[0].get('name', 'gpu')} · {free} MB free")
    else:
        print("  gpu       none detected — video/SFX stages use the CPU previz draft")

    print("\n  what each stage will actually run")
    label = {"tts": "3a voice", "rvc": "3b timbre", "video": "4 video", "sfx": "5 sfx",
             "ollama": "1+2+6 language", "ffmpeg": "7 assembly"}
    for key in ("tts", "rvc", "video", "sfx", "ollama", "ffmpeg"):
        p = plan.get(key) or {}
        if key in ("ollama", "ffmpeg"):
            ok = bool(p.get("available"))
            engine = "online" if ok else "offline"
            why = p.get("reason") or ("local models only, no cloud" if ok else
                                      "deterministic fallbacks — the run still finishes")
            print(f"   [{'on ' if ok else 'off'}] {label.get(key, key):14s} {engine:12s} {why[:64]}")
            continue
        run = p.get("run", True)
        engine = p.get("engine") or "?"
        why = p.get("reason") or ""
        print(f"   [{'on ' if run else 'off'}] {label.get(key, key):14s} {str(engine):12s} {why[:64]}")

    print("\n  services and models")
    rows = [
        ("ffmpeg", caps.get("ffmpeg"), "winget install Gyan.FFmpeg   |   pip install imageio-ffmpeg"),
        ("ollama", caps.get("ollama"), "ollama serve   then   ollama pull sailor2:8b"),
        ("khmer tts", caps.get("sherpa_tts"), "./scripts/setup_khmer_tts.sh   (one-time MMS conversion)"),
        ("sherpa-onnx", caps.get("sherpa_python") or caps.get("sherpa_cli"), "pip install sherpa-onnx"),
        ("RVC api", caps.get("rvc_http"), "start RVC-WebUI's inference API (default http://127.0.0.1:9513)"),
        ("RVC cli", caps.get("rvc_cli"), "set rvc.webui_dir to your RVC-WebUI folder"),
        ("ComfyUI", caps.get("comfyui"), "python main.py --listen 127.0.0.1 --port 8188"),
    ]
    for name, ok, fix in rows:
        print(f"   {mark(ok)} {name:12s}" + ("" if ok else f" → {fix}"))
    prof = caps.get("rvc_profiles") or 0
    print(f"   ·   RVC voice profiles found: {prof}"
          + ("" if prof else "   → train one (README-STUDIO.md, 'Your own voice')"))

    if not caps.get("ffmpeg"):
        print("\n  BLOCKED: no ffmpeg, so Stage 7 cannot write the final .mp4.\n"
              "           winget install Gyan.FFmpeg  (or pip install imageio-ffmpeg),\n"
              "           then re-run --check.")
    elif not caps.get("sherpa_tts"):
        print("\n  Next: no Khmer voice yet, so Stage 3a uses the syllable-timed "
              "placeholder.\n        Run scripts/setup_khmer_tts.sh when you want real speech.")
    else:
        print("\n  Everything essential is present.")
    port = (cfg.get("server") or {}).get("port") or 8000
    print(f"\n  start:  python -m ai_studio --port {port}"
          f"   →   http://localhost:{port}/\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

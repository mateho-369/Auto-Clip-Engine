"""Stage 3b — timbre conversion: make the Khmer voice sound like *you*.

Stage 3a speaks with the MMS Khmer voice (right pronunciation, generic timbre).
Stage 3b runs Retrieval-based Voice Conversion so the narration carries the
Director's own trained voice model, keeping the pronunciation from 3a.

Three interchangeable back-ends, chosen by `rvc.engine` (auto probes them):

``http``  an already-running RVC WebUI / Applio inference API
          (default base `http://127.0.0.1:9513`, endpoint `/sync`, fields in
          `Settings → Voice timbre`; works with the jaaari/RVC_CLI server's
          `/infer` too — it's just a form POST, so point it at whatever you run)
``cli``   calls `infer_cli.py` in your RVC-WebUI install directly
          (argv template is editable, so fork-specific flags are a setting, not a
          code change)
``bypass`` no conversion: keep 3a's audio and (optionally) nudge pitch/formant
          with ffmpeg. The stage then reports `converted: false` — the UI shows a
          badge instead of silently pretending the voice is yours.

Voice **profiles** are first-class (the user may train several): a profile is a
`.pth` + optional `.index`, registered in the DB or discovered by scanning
`models/rvc/`. Training itself is RVC WebUI's job; the studio prepares the
10–15 minute sample (mono 40k wav + a suggested command) and can launch a
user-configured training command while streaming its log into the run feed.
"""
import json
import mimetypes
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

from ..util import ensure_dir, ffmpeg_exe, media_duration, run_ffmpeg


# ------------------------------------------------------------------- probing
def http_reachable(cfg, timeout=2.0):
    base = (cfg.get("rvc", {}).get("api_base") or "").rstrip("/")
    if not base:
        return False
    for path in ("/ping", "/", "/config"):
        try:
            with urllib.request.urlopen(base + path, timeout=timeout) as r:
                if r.status < 500:
                    return True
        except urllib.error.HTTPError as e:
            return e.code < 500
        except Exception:
            continue
    return False


def cli_available(cfg):
    from ..config import rvc_webui_dir

    return bool(rvc_webui_dir(cfg))


def discover_profiles(cfg):
    from ..config import data_root, scan_rvc_profiles

    root = data_root(cfg)
    found = []
    for p in scan_rvc_profiles(cfg, root):
        found.append({"name": p["name"], "pth_path": p["pth"], "index_path": p["index"] or "",
                      "engine": "rvc", "discovered": True,
                      "pth_size": os.path.getsize(p["pth"]) if p["pth"] else 0})
    return found


# ------------------------------------------------------------------ conversion
def convert(in_wav, out_wav, cfg, profile=None, progress=None):
    """Convert the base TTS wav to the user's timbre. Never raises."""
    r = cfg.get("rvc", {})
    if not r.get("enabled", True):
        return {"ok": True, "engine": "off", "converted": False,
                "reason": "timbre stage switched off in settings"}
    want = r.get("engine", "auto")
    attempts = []
    for choice in ([want] if want != "auto" else ["http", "cli", "bypass"]) + \
                  ([] if want in ("auto", "bypass") else ["bypass"]):
        if choice == "http":
            res = _http_convert(in_wav, out_wav, cfg, profile, progress, attempts)
        elif choice == "cli":
            res = _cli_convert(in_wav, out_wav, cfg, profile, progress, attempts)
        else:
            res = _bypass(in_wav, out_wav, cfg, profile, attempts)
        if res.get("ok"):
            return res
    return {"ok": False, "engine": "bypass", "converted": False,
            "reason": "no RVC back-end usable: " + ("; ".join(attempts) or "n/a"),
            "attempts": attempts}


def _fields(cfg, profile, in_wav, out_wav):
    r = cfg.get("rvc", {})
    prof = profile or {}
    model_name = os.path.splitext(os.path.basename(prof.get("pth_path") or ""))[0]
    webui_dir = r.get("webui_dir") or ""
    subs = {
        "model_name": model_name, "pth": prof.get("pth_path") or "",
        "index": prof.get("index_path") or "",
        "pitch": int(prof.get("pitch", r.get("pitch", 0)) or 0),
        "index_rate": float(prof.get("index_rate", r.get("index_rate", 0.75)) or 0.75),
        "rms_mix_rate": float(prof.get("rms_mix_rate", r.get("rms_mix_rate", 0.25)) or 0.25),
        "f0_method": prof.get("f0_method") or r.get("f0_method") or "rmvpe",
        "input": in_wav, "device": r.get("device") or ("cpu" if r.get("device") == "" else "cuda:0"),
        "threads": int(r.get("threads") or 4), "fp16": "True" if r.get("fp16") else "False",
        "clean": "True" if r.get("clean") else "False",
        "python": _python_exe(webui_dir), "webui_dir": webui_dir,
        "outdir": os.path.dirname(out_wav) or ".", "output": out_wav,
        "crepe_hop": int(r.get("crepe_hop_length") or 128),
        "sample_rate": str(40000),
    }
    return subs


def _http_convert(in_wav, out_wav, cfg, profile, progress, attempts):
    r = cfg.get("rvc", {})
    base = (r.get("api_base") or "").rstrip("/")
    if not base or not http_reachable(cfg):
        attempts.append("http: no answer from " + (base or "api_base"))
        return {"ok": False}
    if not (profile or {}).get("pth_path"):
        attempts.append("http: no voice profile selected")
        return {"ok": False}
    subs = _fields(cfg, profile, in_wav, out_wav)
    boundary = "----aiStudio" + uuid.uuid4().hex
    parts = []
    fn = os.path.basename(in_wav)
    ctype = mimetypes.guess_type(fn)[0] or "audio/wav"
    field = r.get("api_upload_field") or "audio_file"
    parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{field}\"; "
                 f"filename=\"{fn}\"\r\nContent-Type: {ctype}\r\n\r\n".encode())
    with open(in_wav, "rb") as f:
        parts.append(f.read())
    parts.append(b"\r\n")
    for k, tmpl in (r.get("api_fields") or {}).items():
        if isinstance(tmpl, str) and "{" in tmpl:
            try:
                v = tmpl.format(**subs)
            except Exception:
                v = tmpl
        else:
            v = tmpl
        if v is None or v == "":
            continue
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n"
                     f"{v}\r\n".encode())
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)
    url = base + (r.get("api_endpoint") or "/sync")
    if progress:
        progress(20, "RVC http conversion")
    try:
        req = urllib.request.Request(url, data=body, method="POST",
                                    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        with urllib.request.urlopen(req, timeout=int(r.get("timeout_sec") or 900)) as resp:
            raw = resp.read()
            ctype_out = resp.headers.get("Content-Type", "")
    except Exception as e:
        attempts.append(f"http request failed: {str(e)[:160]}")
        return {"ok": False}
    mode = r.get("api_result_mode") or "auto"
    looks_audio = not ctype_out.startswith("application/json") and raw[:4] in (b"RIFF", b"ID3\x03") \
        or raw[:3] == b"Ogg"
    if mode in ("auto", "audio") and (looks_audio or mode == "audio") and len(raw) > 100:
        ensure_dir(os.path.dirname(out_wav) or ".")
        if raw[:4] == b"RIFF":
            with open(out_wav, "wb") as f:
                f.write(raw)
        else:                                    # returned mp3/flac: rewrap to wav
            tmp = out_wav + ".src"
            with open(tmp, "wb") as f:
                f.write(raw)
            try:
                run_ffmpeg(["-i", tmp, "-ac", "1", "-ar", "44100", "-c:a", "pcm_s16le", out_wav])
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)
        if os.path.exists(out_wav) and media_duration(out_wav, 0) > 0.1:
            return {"ok": True, "engine": "rvc-http", "converted": True,
                    "profile": (profile or {}).get("name", ""), "duration": media_duration(out_wav, 0)}
    # JSON answer: find the produced file
    try:
        data = json.loads(raw.decode("utf-8", errors="ignore"))
    except Exception:
        attempts.append("http: response was neither audio nor JSON")
        return {"ok": False}
    cand = None
    for key in ("output_path", "output", "file_path", "file", "audio_path", "path"):
        v = data.get(key)
        if isinstance(v, str) and v:
            cand = v
            break
    if cand and os.path.exists(cand):
        shutil.copyfile(cand, out_wav)
        return {"ok": True, "engine": "rvc-http", "converted": True,
                "profile": (profile or {}).get("name", ""), "src": cand}
    if cand:      # remote path → fetch through the same server
        try:
            q = urllib.parse.urlencode({"filename": os.path.basename(cand),
                                        "filepath": os.path.dirname(cand)})
            with urllib.request.urlopen(f"{base}/sync?{q}", timeout=120) as resp:
                blob = resp.read()
            with open(out_wav, "wb") as f:
                f.write(blob)
            return {"ok": True, "engine": "rvc-http", "converted": True,
                    "profile": (profile or {}).get("name", "")}
        except Exception as e:
            attempts.append(f"http: could not fetch result file ({str(e)[:100]})")
            return {"ok": False}
    attempts.append("http: JSON had no readable output path")
    return {"ok": False}


def _cli_convert(in_wav, out_wav, cfg, profile, progress, attempts):
    from ..config import rvc_webui_dir

    r = cfg.get("rvc", {})
    webui = rvc_webui_dir(cfg)
    if not webui:
        attempts.append("cli: RVC-WebUI dir not found (set rvc.webui_dir)")
        return {"ok": False}
    if not (profile or {}).get("pth_path"):
        attempts.append("cli: no voice profile selected")
        return {"ok": False}
    outdir = ensure_dir(os.path.join(os.path.dirname(out_wav) or ".", ".rvc_out"))
    subs = _fields(cfg, profile, in_wav, out_wav)
    subs["outdir"] = outdir
    argv = []
    for tok in (r.get("cli_template") or []):
        if isinstance(tok, str) and "{" in tok:
            try:
                tok = tok.format(**subs)
            except Exception:
                pass
        argv.append(str(tok))
    if progress:
        progress(18, "RVC cli conversion")
    t0 = time.time()
    try:
        res = subprocess.run(argv, cwd=webui, capture_output=True,
                             timeout=int(r.get("timeout_sec") or 900))
    except Exception as e:
        attempts.append(f"cli: {str(e)[:160]}")
        return {"ok": False}
    produced = []
    for rootp, _d, files in os.walk(outdir):
        for fn in sorted(files):
            if fn.lower().endswith((".wav", ".mp3", ".flac", ".ogg")):
                produced.append(os.path.join(rootp, fn))
    if not produced and os.path.exists(out_wav):
        produced = [out_wav]
    if res.returncode != 0 and not produced:
        attempts.append("cli exit " + str(res.returncode) + ": "
                        + (res.stderr or b"").decode(errors="ignore")[-220:])
        return {"ok": False}
    if not produced:
        attempts.append("cli: ran but produced no audio in " + outdir)
        return {"ok": False}
    best = max(produced, key=os.path.getsize)
    if os.path.abspath(best) != os.path.abspath(out_wav):
        ensure_dir(os.path.dirname(out_wav) or ".")
        shutil.copyfile(best, out_wav)
    return {"ok": True, "engine": "rvc-cli", "converted": True,
            "profile": (profile or {}).get("name", ""), "seconds": round(time.time() - t0, 2),
            "duration": media_duration(out_wav, 0)}


def _bypass(in_wav, out_wav, cfg, profile, attempts):
    """No converter: copy the 3a audio, optionally nudge pitch/formant with ffmpeg.

    This keeps Machine B / untrained users moving — but it reports
    `converted: false` so nobody mistakes it for their own voice.
    """
    r = cfg.get("rvc", {})
    semis = float(r.get("pitch") or 0)
    formant = float(r.get("formant_shift") or 0)
    ensure_dir(os.path.dirname(out_wav) or ".")
    reason = ("; ".join(attempts) if attempts else
              "no RVC back-end available — using the MMS Khmer voice directly")
    if abs(semis) < 0.05 and abs(formant) < 0.01:
        shutil.copyfile(in_wav, out_wav)
        return {"ok": True, "engine": "bypass", "converted": False, "reason": reason,
                "duration": media_duration(out_wav, 0)}
    factor = 2 ** (semis / 12.0)
    ff = ffmpeg_exe()
    if not ff:
        shutil.copyfile(in_wav, out_wav)
        attempts.append("pitch nudge skipped: ffmpeg missing")
        return {"ok": True, "engine": "bypass", "converted": False,
                "reason": "copied (ffmpeg missing for pitch shift)"}
    filters = []
    if abs(semis) >= 0.05:
        filters.append(f"asetrate=44100*{factor:.4f}")
        filters.append("aresample=44100")
        filters.append(f"atempo={1.0 / factor:.4f}")
    if abs(formant) >= 0.01:
        filters.append(f"asetrate=44100*(1+{formant:.3f}),aresample=44100,"
                       f"atempo=1/(1+{formant:.3f})")
    try:
        run_ffmpeg(["-i", in_wav, "-af", ",".join(filters), "-ac", "1", "-ar", "44100",
                    "-c:a", "pcm_s16le", out_wav], timeout=600)
        return {"ok": True, "engine": "bypass+ffmpeg-pitch", "converted": False,
                "reason": f"pitch {semis:+.1f} st applied, timbre not converted",
                "duration": media_duration(out_wav, 0)}
    except Exception as e:
        shutil.copyfile(in_wav, out_wav)
        attempts.append(f"pitch filter failed: {str(e)[:120]}")
        return {"ok": True, "engine": "bypass", "converted": False,
                "reason": "copied after filter error", "duration": media_duration(out_wav, 0)}


def _python_exe(webui_dir=""):
    """The RVC-WebUI venv's own interpreter, not this process's (they're separate
    envs — sys.executable here is ai_studio's venv, which doesn't have RVC's deps)."""
    if webui_dir:
        for cand in ("./.venv/Scripts/python.exe", "./.venv/bin/python",
                     "./venv/Scripts/python.exe", "./venv/bin/python"):
            p = os.path.join(webui_dir, cand)
            if os.path.isfile(p):
                return os.path.abspath(p)
    import sys
    return sys.executable or "python"


# ------------------------------------------------------------------- samples
def prepare_sample(src_path, dst_dir, target_sr=40000, want_sec=(120, 1200)):
    """Normalise a Director's raw recording to mono wav for RVC training."""
    ensure_dir(dst_dir)
    dst = os.path.join(dst_dir, "training_sample.wav")
    run_ffmpeg(["-i", src_path, "-ac", "1", "-ar", str(target_sr), "-vn",
                "-af", "highpass=f=70,acompressor=threshold=0.10:ratio=3:attack=8:release=120",
                "-c:a", "pcm_s16le", dst], timeout=900)
    dur = media_duration(dst, 0.0)
    warn = []
    if dur < want_sec[0]:
        warn.append(f"{dur:.0f}s of audio — RVC wants 10-15 min for a good timbre")
    elif dur > want_sec[1]:
        warn.append(f"{dur/60:.1f} min — fine, but training will take longer")
    return {"path": dst, "seconds": round(dur, 1), "warnings": warn,
            "sample_rate": target_sr}


def _default_train_command(webui_dir):
    """The RVC-Project mainline (what README-STUDIO.md installs) has no
    single-command CLI for training — its own train/*.py scripts even break
    when run directly (see scripts/rvc_autotrain.py's docstring). Default to
    that wrapper instead of guessing at a fork-specific `infer-web.py` flag
    set that doesn't exist in what's actually installed."""
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    wrapper = os.path.join(here, "scripts", "rvc_autotrain.py")
    return (f'"{{python}}" "{wrapper}" --rvc-dir "{webui_dir or "{webui_dir}"}" '
            f'--exp "{{name}}" --dataset "{{sample}}"')


def _train_subs(profile, sample_path, cfg):
    r = cfg.get("rvc", {})
    webui_dir = r.get("webui_dir") or ""
    return {"sample": sample_path, "sample_dir": os.path.dirname(sample_path),
            "name": (profile or {}).get("name") or "my_voice",
            "exp": (profile or {}).get("name") or "my_voice",
            "python": _python_exe(webui_dir), "webui_dir": webui_dir}


def training_command(profile, sample_path, cfg):
    """The exact command we'd run — shown to the Director, copy-pasteable."""
    r = cfg.get("rvc", {})
    subs = _train_subs(profile, sample_path, cfg)
    cmd = r.get("train_command") or _default_train_command(subs["webui_dir"])
    cmd = " ".join(str(t) for t in cmd) if isinstance(cmd, list) else cmd
    try:
        return cmd.format(**subs)
    except Exception:
        return cmd


def run_training(profile, sample_path, cfg, log_line=None, timeout=21600):
    """Launch the (configured, or default wrapper-script) training command,
    with {sample}/{name}/{python}/{webui_dir}/... substituted in — same
    placeholder set Settings' own field hint advertises."""
    r = cfg.get("rvc", {})
    subs = _train_subs(profile, sample_path, cfg)
    if not subs["webui_dir"]:
        return {"ok": False, "reason": "set rvc.webui_dir in Settings first — "
                                       "training needs to know where RVC-WebUI is installed"}
    cmd = r.get("train_command") or _default_train_command(subs["webui_dir"])
    if isinstance(cmd, list):
        argv = [str(t).format(**subs) if isinstance(t, str) and "{" in t else t for t in cmd]
    else:
        try:
            argv = cmd.format(**subs)
        except Exception:
            argv = cmd
    try:
        proc = subprocess.Popen(argv, shell=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        lines = []
        t0 = time.time()
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            lines.append(line.rstrip())
            if log_line:
                log_line(line.rstrip()[:400])
            if time.time() - t0 > timeout:
                proc.kill()
                return {"ok": False, "reason": "training timed out", "lines": len(lines)}
        rc = proc.wait()
        return {"ok": rc == 0, "code": rc, "lines": len(lines), "seconds": round(time.time() - t0, 1)}
    except Exception as e:
        return {"ok": False, "reason": str(e)[:200]}


def probe(cfg):
    from ..config import rvc_webui_dir

    r = cfg.get("rvc", {})
    return {"enabled": bool(r.get("enabled", True)), "engine_pref": r.get("engine"),
            "http": http_reachable(cfg), "cli": cli_available(cfg),
            "webui_dir": rvc_webui_dir(cfg) or "", "api_base": r.get("api_base"),
            "profiles": discover_profiles(cfg)}

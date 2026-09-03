"""RunContext — everything a stage needs, in one object.

Owns: config + resolved engine plan, the DB, the event bus, the project/run rows,
the cancel flag, and the on-disk layout for a project:

    <data>/projects/<project_id>/
        01_script.txt              the Director's locked / approved script
        02_scenes.json             Stage 1 output (scenes + prompts + timings)
        scenes/00/03a_voice.wav    per-scene assets, numbered so a folder view
        scenes/00/03b_voice_final  reads like the pipeline
        scenes/00/04_video.mp4
        scenes/00/04b_video_fit.mp4
        scenes/00/05_ambient.wav
        scenes/00/06_qa.json
        final/<run_id>.mp4 .srt .poster.png .manifest.json
        prompts/                   (per-run prompt dumps, browsable without SQLite)
"""
import os

from ..util import (ensure_dir, jdump, media_duration, read_wav, rms, write_json)
from .. import khmer


class RunContext:
    def __init__(self, db, cfg, plan, project, run, bus=None, data_root="", cancel=None,
                 resume_from="", force_stages=()):
        self.db = db
        self.cfg = cfg
        self.plan = plan or {}
        self.project = project
        self.run = run
        self.bus = bus
        self.data_root = data_root
        self.cancel = cancel
        self.resume_from = resume_from
        self.force_stages = tuple(force_stages or ())
        self._scenes = None

    # ------------------------------------------------------------- identities
    @property
    def project_id(self):
        return self.project["id"]

    @property
    def run_id(self):
        return self.run["id"]

    @property
    def cancelled(self):
        return bool(self.cancel and getattr(self.cancel, "is_set", lambda: False)())

    # ------------------------------------------------------------------- paths
    def project_dir(self):
        return ensure_dir(os.path.join(self.data_root, "projects", self.project_id))

    def scenes_root(self):
        return ensure_dir(os.path.join(self.project_dir(), "scenes"))

    def scene_dir(self, idx):
        return ensure_dir(os.path.join(self.scenes_root(), f"{int(idx):02d}"))

    def final_dir(self):
        return ensure_dir(os.path.join(self.project_dir(), "final"))

    def asset_path(self, kind, idx, ext):
        names = {"script": "01_script.txt", "scenes": "02_scenes.json"}
        if kind in names:
            return os.path.join(self.project_dir(), names[kind])
        stem = {"voice": "03a_voice", "voice_final": "03b_voice_final", "video": "04_video",
                "video_fit": "04b_video_fit", "ambient": "05_ambient", "qa": "06_qa",
                "thumb": "00_thumb", "waveform": "03a_voice.waveform"}.get(kind, kind)
        return os.path.join(self.scene_dir(idx), f"{stem}{ext}")

    def relurl(self, path):
        """URL-safe relative path under the project root (served by the API)."""
        try:
            return "/" + os.path.relpath(path, self.data_root).replace(os.sep, "/")
        except Exception:
            return "/" + os.path.basename(path)

    # ------------------------------------------------------------------ assets
    def register_asset(self, kind, path, stage="", scene_idx=-1, meta=None, duration=None):
        if not path or not os.path.exists(path):
            return None
        if duration is None:
            duration = media_duration(path, 0.0) if str(path).endswith(
                (".wav", ".mp4", ".mp3", ".mov", ".webm", ".flac", ".ogg")) else 0.0
        return self.db.add_asset(self.project_id, kind, path, run_id=self.run_id, stage=stage,
                                 scene_idx=scene_idx, duration=duration,
                                 relpath=self.relurl(path), meta=meta or {},
                                 size_bytes=os.path.getsize(path))

    def latest_asset(self, kind, stage="", scene_idx=None):
        return self.db.latest_asset(self.project_id, kind, stage=stage, scene_idx=scene_idx)

    def asset_info(self, kind, idx):
        a = self.latest_asset(kind, scene_idx=idx)
        if not a:
            return None
        if a.get("duration"):
            return a
        a = dict(a)
        a["duration"] = media_duration(a["path"], 0.0)
        return a

    def write_text(self, path, text):
        ensure_dir(os.path.dirname(path) or ".")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text or "")
        return path

    # --------------------------------------------------------------- scenes
    @property
    def scenes(self):
        if self._scenes is None:
            self._scenes = self.db.list_scenes(self.project_id)
        return self._scenes

    def reload_scenes(self):
        self._scenes = None
        return self.scenes

    # ------------------------------------------------------------- reporting
    def report(self, stage, scene_idx, pct, message="", **extra):
        if self.bus is not None:
            payload = {"pct": round(float(pct or 0), 1), "message": message}
            payload.update(extra)
            self.bus.publish("progress", payload, run_id=self.run_id, project_id=self.project_id,
                             stage=stage, scene_idx=int(scene_idx))

    def event(self, kind, payload=None, stage="", scene_idx=-1):
        if self.bus is not None:
            self.bus.publish(kind, payload or {}, run_id=self.run_id, project_id=self.project_id,
                             stage=stage, scene_idx=int(scene_idx))

    def progress_cb(self, stage, scene_idx, lo=0.0, hi=100.0, prefix=""):
        def cb(pct, note=""):
            try:
                mapped = lo + (hi - lo) * max(0.0, min(100.0, float(pct))) / 100.0
            except Exception:
                mapped = lo
            self.report(stage, scene_idx, mapped, f"{prefix}{note}" if note else prefix)
        return cb

    # ------------------------------------------------------------------- LLM
    def llm(self):
        from ..llm import LLM

        return LLM(self.cfg, db=self.db, run_id=self.run_id, project_id=self.project_id)

    # --------------------------------------------------------------- helpers
    def log_engine_prompt(self, stage, scene_idx, engine, model="", system="", user="",
                          response="", ok=True, error="", latency_ms=0):
        """Ledger entry for a *non-LLM* call (TTS text, ComfyUI prompt, SFX prompt...).

        The spec asks for every prompt sent to every model, so the fallback
        engines are logged too — otherwise the Memory view is empty on a machine
        without Ollama and 'reuse these exact settings' has nothing to reuse.
        """
        try:
            self.db.log_prompt(project_id=self.project_id, run_id=self.run_id, stage=stage,
                               scene_idx=int(scene_idx), role=stage, model=model or engine or "",
                               engine=engine or "", system=system or "", user=user or "",
                               response=response or "", ok=bool(ok), error=error or "",
                               latency_ms=int(latency_ms or 0))
        except Exception:
            pass

    def voice_facts(self, path):
        """Durations/levels QA needs, in one numpy pass."""
        out = {"duration": media_duration(path, 0.0), "peak": 0.0, "head_silence": 0.0,
               "tail_silence": 0.0}
        try:
            x, sr = read_wav(path)
            import numpy as np
            out["peak"] = float(np.max(np.abs(x))) if x.size else 0.0
            out["rms_db"] = round(20 * (1e-9 + __import__("math").log10(max(1e-7, rms(x)))), 1)
            from ..util import leading_trailing_silence
            h, t = leading_trailing_silence(path)
            out["head_silence"], out["tail_silence"] = round(h, 2), round(t, 2)
        except Exception:
            pass
        return out

    def estimated_total(self):
        return round(sum(float(s.get("estimated_duration_sec") or 0) for s in self.scenes), 2)

    def actual_total(self):
        total = 0.0
        for s in self.scenes:
            total += float(s.get("audio_duration") or s.get("estimated_duration_sec") or 0)
        return round(total, 2)


def scene_dict_for_prompt(scene):
    """The compact scene view handed to models / stored in prompt logs."""
    return {"idx": scene.get("idx"), "text": khmer.truncate_clusters(scene.get("text") or "", 900),
            "visual_prompt": scene.get("visual_prompt") or "",
            "mood_tag": scene.get("mood_tag") or "",
            "estimated_duration_sec": scene.get("estimated_duration_sec"),
            "sfx_prompt": scene.get("sfx_prompt") or ""}

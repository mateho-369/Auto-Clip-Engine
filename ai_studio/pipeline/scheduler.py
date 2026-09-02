"""The async scheduler: a real task queue over the stage graph.

Why not "just await the nine stages in a row": voice (3a/3b) and video (4) only
depend on Stage 1, so on a 5070 they are worth overlapping — narration while the
diffusion model works — and per-scene rows must update independently. So:

* every (stage × scene) pair is a **job**; the graph comes from
  :mod:`ai_studio.pipeline.spec`;
* jobs launch as soon as their dependencies are satisfied, through **per-resource
  semaphores** (llm 1, tts 1, gpu 1, cpu 2, io 4) plus a single GPU lock, because
  8GB cannot host Wan + MMAudio + an 8B LLM at the same time;
* each job retries **once** automatically, then fails *that job* only — dependents
  go to `blocked` carrying the reason, other branches keep going, and the run
  finishes as `partial`. No silent stall, no whole-run crash;
* `deferred` (a GPU stage on the CPU machine) counts as satisfied, so a draft cut
  still completes; the later `gpu-catchup` run re-runs exactly those stages;
* resume = a new run inheriting the parent's completed stages; a single-stage
  regeneration is the same mechanism with `force_stages`, which gives the stale
  cascade for free.

State lives in SQLite (one `stage_runs` row per job) and is mirrored onto the
event bus, so the UI stepper is a projection of the database rather than a
fragile in-memory copy.
"""
import asyncio
import contextlib
import os
import time

from .. import config as cfg_mod
from ..events import EventBus, RunProgress
from ..util import now
from . import spec as stagespec
from .context import RunContext
from .stages import run_stage


class Scheduler:
    def __init__(self, db, data_root, bus=None, cfg=None):
        self.db = db
        self.data_root = data_root
        self.bus = bus or EventBus(db=db)
        self.cfg = cfg
        self.runs = {}                              # run_id → _ActiveRun

    # ------------------------------------------------------------------ config
    def current_config(self):
        base = self.cfg or cfg_mod.load(os.path.join(self.data_root, "settings.json"))
        return base

    def resolved(self, overrides=None):
        cfg = cfg_mod.normalize_config(_deep_merge(self.current_config(), overrides or {}))
        cfg, plan = cfg_mod.resolve(cfg)
        return cfg, plan

    # ------------------------------------------------------------------- start
    async def start_run(self, project_id, trigger="new", resume_from="", force_stages=None,
                        auto_start=True, scene_count_hint=None):
        project = self.db.get_project(project_id)
        if not project:
            raise KeyError(f"project {project_id} not found")
        cfg, plan = self.resolved(project.get("settings") or {})
        prev = None
        if resume_from:
            row = self.db.get_run(resume_from)
            prev = row if row and row["project_id"] == project_id else None
        elif trigger in ("resume", "regenerate", "gpu-catchup", "continue"):
            last = project.get("last_run_id") or ""
            prev = self.db.get_run(last) if last else None
        force = [s for s in (force_stages or []) if s in stagespec.STAGE_BY_KEY]
        inheriting = bool(prev) and (trigger != "new" or force)

        n_scenes = scene_count_hint if scene_count_hint is not None else len(
            self.db.list_scenes(project_id))
        full_jobs, _ = stagespec.build_graph(n_scenes, plan=plan, cfg=cfg)
        pending_jobs = (stagespec.build_graph(n_scenes, plan=plan, cfg=cfg, only=force)[0]
                        if force else full_jobs)
        if not inheriting:
            pending_jobs = full_jobs
        inherit = {}
        if inheriting and prev:
            inherit = {stagespec.job_key(r["stage"], r["scene_idx"]): r
                       for r in self.db.list_stages(prev["id"]) if r["status"] == "done"}
            if not force:
                # a plain resume must actually resume: everything that already
                # finished is inherited instead of being rendered a second time
                pending_jobs = {k: j for k, j in pending_jobs.items() if k not in inherit}
                if not pending_jobs:
                    pending_jobs = {}

        run = self.db.create_run(
            project_id, status="queued", trigger=trigger,
            resume_from=(prev["id"] if inheriting and prev else ""),
            machine_profile=plan.get("hardware", {}).get("profile", ""),
            gpu_policy=_plan_summary(plan), stage_filter=force,
            start_now=False)
        run_id = run["id"]
        for key, job in full_jobs.items():
            if key in pending_jobs:
                self.db.upsert_stage(run_id, project_id, job.stage, job.scene_idx, status="queued",
                                     message="")
            else:
                src = inherit.get(key) or {}
                self.db.upsert_stage(
                    run_id, project_id, job.stage, job.scene_idx,
                    status="done" if src else "pending", engine=src.get("engine", ""),
                    inherited_from=src.get("id", ""), progress=100.0 if src else 0.0,
                    message=(f"reused from run {prev['id']}" if src else "not required by this run"))

        st = _ActiveRun(run_id, project_id, pending_jobs, cfg, plan, project, run, self.bus,
                        self.db, self.data_root, resume_from=(prev["id"] if inheriting and prev else ""),
                        force_stages=force)
        self.runs[run_id] = st
        self.bus.publish("run_queued", {"trigger": trigger, "jobs": len(pending_jobs),
                                        "inherited": len(full_jobs) - len(pending_jobs),
                                        "force": force, "plan": _plan_summary(plan)},
                        run_id=run_id, project_id=project_id)
        if auto_start:
            st.task = asyncio.create_task(self._execute(st))
        return {"run_id": run_id, "jobs": len(pending_jobs),
                "inherited": len(full_jobs) - len(pending_jobs), "plan": _plan_summary(plan)}

    async def rerun_stage(self, run_id, stage, scene_idx=None, project_id=None):
        """'Regenerate just this stage' — forks a run limited to that stage + downstream."""
        run = self.db.get_run(run_id)
        if not run:
            raise KeyError("run not found")
        pid = project_id or run["project_id"]
        if stage not in stagespec.STAGE_BY_KEY:
            raise KeyError(f"unknown stage {stage}")
        return await self.start_run(pid, trigger="regenerate", resume_from=run_id,
                                    force_stages=[stage])

    # ------------------------------------------------------------- control plane
    async def cancel_run(self, run_id):
        st = self.runs.get(run_id)
        if not st:
            self.db.update_run(run_id, status="cancelled", finished_at=now())
            return {"ok": True, "note": "run is not active — marked cancelled"}
        st.cancel.set()
        st.paused.clear()
        return {"ok": True}

    def pause_run(self, run_id):
        st = self.runs.get(run_id)
        if not st:
            return {"ok": False, "reason": "run is not active"}
        st.paused.set()
        self.db.update_run(run_id, status="paused")
        self.bus.publish("run_paused", {}, run_id=run_id, project_id=st.project_id)
        return {"ok": True}

    def resume_paused(self, run_id):
        st = self.runs.get(run_id)
        if not st:
            return {"ok": False, "reason": "run is not active — start a continuation run instead"}
        st.paused.clear()
        self.db.update_run(run_id, status="running")
        self.bus.publish("run_resumed", {}, run_id=run_id, project_id=st.project_id)
        return {"ok": True}

    async def wait(self, run_id, timeout=None):
        st = self.runs.get(run_id)
        if not st or not st.task:
            return self.status(run_id)
        with contextlib.suppress(asyncio.TimeoutError, Exception):
            await asyncio.wait_for(asyncio.shield(st.task), timeout)
        return self.status(run_id)

    def is_active(self, run_id):
        st = self.runs.get(run_id)
        return bool(st and st.task and not st.task.done())

    async def shutdown(self):
        for st in list(self.runs.values()):
            st.cancel.set()
            st.paused.clear()
        await asyncio.sleep(0)

    # ------------------------------------------------------------------ execute
    async def _execute(self, st):
        run_id = st.run_id
        self.db.update_run(run_id, status="running", started_at=now())
        st.ctx.event("run_started", {"jobs": len(st.jobs)})
        scheduler_errors = []
        try:
            while True:
                if st.cancel.is_set():
                    break
                if st.needs_review:
                    # Mode B gate: the script stage asked for the Director's eye, so
                    # nothing downstream may consume GPU time before they approve.
                    break
                while st.paused.is_set() and not st.cancel.is_set():
                    await asyncio.sleep(0.35)
                # 1. harvest finished tasks
                if st.running:
                    for key in [k for k, t in st.running.items() if t.done()]:
                        task = st.running.pop(key)
                        err = None
                        with contextlib.suppress(asyncio.CancelledError):
                            err = task.exception()
                        if err:
                            scheduler_errors.append(f"{key}: {type(err).__name__}: {err}")
                if not st.pending:
                    if not st.running:
                        break
                    await _await_some(st.running)
                    continue
                # 2. launch whatever is ready now
                launched = 0
                for key in list(st.pending.keys()):
                    job = st.jobs[key]
                    if all(d in st.satisfied or d not in st.jobs for d in job.deps):
                        del st.pending[key]
                        st.running[key] = asyncio.create_task(self._guarded(st, job))
                        launched += 1
                # 3. nothing running, nothing launchable → resolve or give up
                if not st.running:
                    blocked = [k for k in st.pending if any(d in st.failed for d in st.jobs[k].deps)]
                    for key in blocked:
                        del st.pending[key]
                        await self._mark_blocked(st, key)
                    if st.pending and not launched:
                        # still stuck and nothing in flight: unresolvable, don't spin
                        for key in list(st.pending.keys()):
                            del st.pending[key]
                            await self._mark_blocked(st, key)
                    if not st.pending:
                        break
                    continue
                await _await_some(st.running)
        except Exception as e:                                  # never hang the UI on a bug
            scheduler_errors.append(f"scheduler error: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            for task in list(st.running.values()):
                task.cancel()
            for key, task in list(st.running.items()):
                with contextlib.suppress(Exception):
                    await task
            st.running.clear()
            hold = "queued" if st.needs_review else "cancelled"
            for key in list(st.pending.keys()):
                job = st.jobs[key]
                row = self.db.get_stage(run_id, job.stage, job.scene_idx)
                if row and row["status"] in ("running", "done", "failed"):
                    continue                                   # never rewrite a finished stage
                self.db.upsert_stage(run_id, st.project_id, job.stage, job.scene_idx,
                                     status=hold,
                                     message=("waiting for the Director to approve the script"
                                              if st.needs_review else
                                              "run stopped before this stage"))
            st.pending.clear()
            await self._finish(st, scheduler_errors)

    async def _mark_blocked(self, st, key):
        job = st.jobs[key]
        why = [d for d in job.deps if d in st.failed]
        msg = (f"blocked — depends on a failed stage ({', '.join(why[:3])})") if why else \
              "blocked — unmet dependencies"
        self.db.upsert_stage(st.run_id, st.project_id, job.stage, job.scene_idx, status="blocked",
                             message=msg, error=msg, finished_at=now())
        st.blocked.add(key)
        st.satisfied.add(key)          # let the run settle instead of spinning
        st.ctx.event("stage_update", {"status": "blocked", "message": msg}, stage=job.stage,
                     scene_idx=job.scene_idx)

    async def _guarded(self, st, job):
        sp = job.spec
        lim = int(st.cfg["pipeline"]["concurrency"].get(sp.resource, 2))
        sem = st.semaphores.get(sp.resource)
        if sem is None:
            sem = asyncio.Semaphore(max(1, lim))
            st.semaphores[sp.resource] = sem
        async with sem:
            gpu = st.gpu_lock if (sp.requires_gpu and st.cfg["vram"].get("serialize_gpu", True)) \
                else contextlib.nullcontext()
            async with gpu:
                await self._run_job(st, job)

    async def _run_job(self, st, job):
        sp = job.spec
        run_id, project_id = st.run_id, st.project_id
        attempts = max(1, 1 + int(st.cfg["pipeline"].get("retry_limit", 1)))
        backoff = float(st.cfg["pipeline"].get("retry_backoff_sec", 3))
        t0 = time.time()
        for attempt in range(1, attempts + 1):
            if st.cancel.is_set():
                return
            self.db.upsert_stage(run_id, project_id, sp.key, job.scene_idx, status="running",
                                 attempt=attempt, started_at=now(), progress=1.0,
                                 message="retry 2/2" if attempt > 1 else "running", error="")
            st.ctx.event("stage_update", {"status": "running", "attempt": attempt},
                         stage=sp.key, scene_idx=job.scene_idx)
            try:
                res = await run_stage(sp.key, st.ctx, job)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                err = f"{type(e).__name__}: {str(e)[:280]}"
                if attempt < attempts and sp.retryable:
                    await asyncio.sleep(backoff)
                    continue
                self._fail(st, job, err, t0, attempt)
                return
            status = res.get("status") or ("done" if res.get("ok", True) else "failed")
            ms = int((time.time() - t0) * 1000)
            if status == "failed" or not res.get("ok", True):
                err = str(res.get("error") or "stage reported failure")
                if attempt < attempts and sp.retryable:
                    st.ctx.event("log", {"level": "warn",
                                        "text": f"{sp.title}: {err[:150]} — retrying once"},
                                 stage=sp.key, scene_idx=job.scene_idx)
                    await asyncio.sleep(backoff)
                    continue
                self._fail(st, job, err, t0, attempt, engine=res.get("engine", ""))
                return
            # ------------------------------------------------------ success
            for a in res.get("assets") or []:
                st.ctx.register_asset(a["kind"], a["path"], stage=sp.key,
                                      scene_idx=int(a.get("scene_idx", job.scene_idx)),
                                      meta=a.get("meta") or {"engine": res.get("engine", "")},
                                      duration=a.get("duration"))
            upd = {k: v for k, v in (res.get("scene_update") or {}).items() if k != "meta"}
            if upd and job.scene_idx >= 0:
                self.db.update_scene(project_id, job.scene_idx, **upd)
                st.ctx.reload_scenes()
            if res.get("project_update"):
                self.db.update_project(project_id, **res["project_update"])
                st.project = self.db.get_project(project_id) or st.project
                st.ctx.project = st.project
            if res.get("run_update"):
                stats = dict((self.db.get_run(run_id) or {}).get("stats") or {})
                stats.update(res["run_update"])
                self.db.update_run(run_id, stats=stats)
            if res.get("requires_review"):
                st.needs_review = True
            self.db.upsert_stage(run_id, project_id, sp.key, job.scene_idx, status=status,
                                 engine=res.get("engine", ""), progress=res.get("progress", 100.0),
                                 message=str(res.get("message") or "")[:300], error="",
                                 finished_at=now(), duration_ms=ms, attempt=attempt)
            st.satisfied.add(job.key)
            st.results[job.key] = res
            for note in res.get("notes") or []:
                if note:
                    st.ctx.event("log", {"level": "info", "text": str(note)[:400]},
                                 stage=sp.key, scene_idx=job.scene_idx)
            st.ctx.event("stage_update", {"status": status, "pct": 100.0,
                                          "engine": res.get("engine", ""),
                                          "message": str(res.get("message") or "")[:220],
                                          "duration_ms": ms, "notes": res.get("notes") or []},
                         stage=sp.key, scene_idx=job.scene_idx)
            if sp.key == "breakdown":
                await self._reexpand(st)
            return

    def _fail(self, st, job, err, t0, attempt, engine=""):
        sp = job.spec
        self.db.upsert_stage(st.run_id, st.project_id, sp.key, job.scene_idx, status="failed",
                             error=_stage_error(sp, err), message="failed after retry"
                             if attempt > 1 else "failed", engine=engine, finished_at=now(),
                             duration_ms=int((time.time() - t0) * 1000), attempt=attempt)
        st.failed.add(job.key)
        st.satisfied.add(job.key)
        st.ctx.event("stage_failed", {"error": str(err)[:400], "attempts": attempt,
                                      "title": sp.title, "scene_idx": job.scene_idx},
                     stage=sp.key, scene_idx=job.scene_idx)

    async def _reexpand(self, st):
        """Stage 1 is what tells us how many scenes there are — expand the graph then."""
        scenes = self.db.list_scenes(st.project_id)
        jobs, _order = stagespec.build_graph(len(scenes), plan=st.plan, cfg=st.cfg)
        added = 0
        for key, job in jobs.items():
            st.jobs.setdefault(key, job)
            if key in st.satisfied or key in st.running:
                continue
            if self.db.get_stage(st.run_id, job.stage, job.scene_idx):
                continue
            self.db.upsert_stage(st.run_id, st.project_id, job.stage, job.scene_idx,
                                 status="queued")
            st.pending.setdefault(key, None)
            added += 1
        st.ctx.event("graph_ready", {"scenes": len(scenes), "added_jobs": added}, stage="breakdown")

    async def _finish(self, st, scheduler_errors):
        run_id = st.run_id
        rows = self.db.list_stages(run_id)
        finished_rows = [r for r in rows if r["status"] not in ("pending", "queued")]
        ok = [r for r in finished_rows if r["status"] in ("done", "skipped", "deferred")]
        failed_rows = [r for r in finished_rows if r["status"] == "failed"]
        if st.cancel.is_set():
            status = "cancelled"
        elif st.needs_review and not failed_rows:
            status = "needs_review"
        elif failed_rows or scheduler_errors:
            status = "partial" if ok else "failed"
        elif not finished_rows:
            status = "cancelled"
        else:
            status = "completed"
        summary = RunProgress.overall(rows)
        deferred = sorted({r["stage"] for r in rows if r["status"] == "deferred"})
        stats = dict((self.db.get_run(run_id) or {}).get("stats") or {})
        final = (st.results.get("assemble#-1") or {}).get("final_asset_path") or ""
        err = ""
        if failed_rows:
            err = " || ".join(f"{r['stage']}#{r['scene_idx']}: {(r['error'] or r['message'])[:150]}"
                              for r in failed_rows[:3])
        elif scheduler_errors:
            err = "; ".join(scheduler_errors)[:400]
        stats.update({"stages_seen": len(rows), "stages_done": len(ok), "stages_failed": len(failed_rows),
                      "deferred_stages": deferred, "overall_pct": summary["pct"],
                      "elapsed_sec": round(time.time() - (st.started_at or time.time()), 1),
                      "errors": ([err] if err else [])[:1]})
        if final:
            stats["final_path"] = final
        self.db.update_run(run_id, status=status, finished_at=now(), stats=stats, error=err[:900])
        if status == "completed":
            self.db.update_project(st.project_id, status="done", last_run_id=run_id)
        elif status in ("failed", "partial"):
            self.db.update_project(st.project_id, status="failed", last_run_id=run_id)
        elif status == "needs_review":
            self.db.update_project(st.project_id, status="review", last_run_id=run_id)
        else:
            self.db.update_project(st.project_id, last_run_id=run_id)
        st.ctx.event("run_finished", {"status": status, "deferred_stages": deferred,
                                      "error": err[:400], "stats": stats, "final": bool(final),
                                      "needs_review": bool(st.needs_review)})
        st.done = True
        return status

    # ----------------------------------------------------------------- status
    def status(self, run_id):
        run = self.db.get_run(run_id)
        if not run:
            return None
        rows = run["stages"]
        active = self.runs.get(run_id)
        return {"run": {k: v for k, v in run.items() if k not in ("stages", "assets")},
                "stages": rows, "by_stage": RunProgress.rollup(rows),
                "overall": RunProgress.overall(rows),
                "graph": [{"key": j.key, "stage": j.stage, "scene_idx": j.scene_idx,
                           "deps": list(j.deps)} for j in (active.jobs.values() if active else [])],
                "active": bool(active and not getattr(active, "done", False)),
                "paused": bool(active and active.paused.is_set()),
                "needs_review": bool(active and active.needs_review),
                "plan": _plan_summary(active.plan) if active else None,
                "log": self.db.list_events(run_id, limit=400)[-100:],
                "final": next((a for a in run["assets"] if a["kind"] == "final"), None),
                "assets": run["assets"]}


class _ActiveRun:
    """Mutable scheduling state for one run."""

    def __init__(self, run_id, project_id, jobs, cfg, plan, project, run, bus, db, data_root,
                 resume_from="", force_stages=()):
        self.run_id, self.project_id = run_id, project_id
        self.jobs = dict(jobs)
        self.cfg, self.plan, self.project, self.run = cfg, plan, project, run
        self.bus, self.db, self.data_root = bus, db, data_root
        self.resume_from, self.force_stages = resume_from, tuple(force_stages)
        self.pending = dict(self.jobs)
        self.satisfied, self.failed, self.blocked, self.results = set(), set(), set(), {}
        self.running, self.semaphores = {}, {}
        self.gpu_lock = asyncio.Lock()
        self.cancel, self.paused = asyncio.Event(), asyncio.Event()
        self.task, self.done, self.needs_review = None, False, False
        self.started_at = time.time()
        self.ctx = RunContext(db, cfg, plan, project, run, bus=bus, data_root=data_root,
                              cancel=self.cancel, resume_from=resume_from,
                              force_stages=force_stages)


async def _await_some(running):
    if not running:
        await asyncio.sleep(0.05)
        return
    await asyncio.wait(list(running.values()), return_when=asyncio.FIRST_COMPLETED)


def _stage_error(sp, err):
    """What the UI shows: which stage, why — not a bare traceback line."""
    return f"[{sp.emoji} {sp.title}] {str(err)[:420]}"


def _plan_summary(plan):
    out = {}
    for k in ("tts", "rvc", "video", "sfx"):
        if isinstance((plan or {}).get(k), dict):
            out[k] = {"engine": plan[k].get("engine"), "reason": plan[k].get("reason", "")}
    hw = (plan or {}).get("hardware") or {}
    out["hardware"] = {"profile": hw.get("profile"), "cpu_only": hw.get("cpu_only"),
                       "vram_total_mb": hw.get("vram_total_mb"),
                       "vram_free_mb": hw.get("vram_free_mb"),
                       "gpus": [g.get("name") for g in (hw.get("gpus") or [])]}
    out["ollama"] = (plan or {}).get("ollama") or {}
    out["pressure"] = (plan or {}).get("vram_pressure")
    return out


def _deep_merge(base, over):
    out = dict(base or {})
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        elif v is not None:
            out[k] = v
    return out

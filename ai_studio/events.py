"""Live status plumbing: an asyncio fan-out bus + a persisted run log.

The scheduler runs inside uvicorn's event loop and publishes one event per state
change ("voice scene 2 -> done", "video scene 3 -> 62%"). The UI consumes them
over a WebSocket, with SSE and plain polling as fallbacks (proxies sometimes eat
WS), and every event is also written to SQLite so a run's history survives a
restart and can be replayed in the "memory" view.
"""
import asyncio
import collections
import threading
import time

from .util import new_id


class EventBus:
    """Topic → set of asyncio.Queues. `publish` is safe from any thread."""

    def __init__(self, db=None, loop=None, max_queue=400, persist=True):
        self._subs = collections.defaultdict(set)
        self._lock = threading.Lock()
        self._loop = loop
        self.db = db
        self.persist = persist
        self.max_queue = max_queue
        self.seq = 0
        self.recent = collections.deque(maxlen=200)   # late-joining clients get a replay

    def bind_loop(self, loop):
        self._loop = loop

    def subscribe(self, topic="*"):
        q = asyncio.Queue(maxsize=self.max_queue)
        with self._lock:
            self._subs[topic].add(q)
        for ev in list(self.recent)[-40:]:
            if ev.get("topic") in (topic, "*") or topic == "*":
                try:
                    q.put_nowait(ev)
                except Exception:
                    break
        return q

    def unsubscribe(self, topic, q):
        with self._lock:
            self._subs.get(topic, set()).discard(q)

    def publish(self, kind, payload=None, run_id="", project_id="", stage="", scene_idx=-1):
        ev = {"id": self._next_id(), "ts": time.time(), "kind": kind, "run_id": run_id,
              "project_id": project_id, "stage": stage, "scene_idx": int(scene_idx),
              "topic": run_id or "*", "payload": payload or {}}
        self.seq += 1
        with self._lock:
            targets = set(self._subs.get(ev["topic"], set())) | set(self._subs.get("*", set()))
            self.recent.append(ev)
        for q in targets:
            self._push(q, ev)
        if self.db is not None and self.persist and run_id:
            try:
                self.db.log_event(run_id, project_id, kind, payload, stage=stage, scene_idx=scene_idx)
            except Exception:
                pass
        return ev

    def _push(self, q, ev):
        def put():
            try:
                if q.full():
                    try:
                        q.get_nowait()
                    except Exception:
                        pass
                q.put_nowait(ev)
            except Exception:
                pass
        loop = self._loop
        if loop is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
        if loop is not None:
            try:
                if loop.is_running():
                    loop.call_soon_threadsafe(put)
                    return
            except Exception:
                pass
        put()

    def _next_id(self):
        return int(time.time() * 1000) * 1000 + (self.seq % 1000)


class RunProgress:
    """Aggregates stage rows into the numbers the stepper/hero cards show."""

    STAGE_ORDER = ["script", "breakdown", "voice_base", "voice_final", "video", "video_fit",
                   "sfx", "qa", "assemble"]

    @staticmethod
    def rollup(stage_rows):
        by_stage = collections.defaultdict(list)
        for r in stage_rows:
            by_stage[r["stage"]].append(r)
        out = {}
        for stage, rows in by_stage.items():
            n = len(rows)
            done = sum(1 for r in rows if r["status"] in ("done", "skipped", "deferred"))
            failed = sum(1 for r in rows if r["status"] == "failed")
            running = sum(1 for r in rows if r["status"] == "running")
            prog = sum(float(r.get("progress") or 0) for r in rows if r["status"] != "done")
            pct = 100.0 if done == n else round((done * 100.0 + (prog / max(1, n - done))) / n, 1)
            out[stage] = {
                "stage": stage, "total": n, "done": done, "failed": failed, "running": running,
                "pct": min(99.9, pct) if (running or done < n) else 100.0,
                "status": ("failed" if failed else
                           "running" if running else
                           "done" if done == n else
                           "deferred" if all(r["status"] in ("deferred", "skipped") for r in rows) else
                           "queued" if any(r["status"] in ("queued", "running") for r in rows) else
                           "pending"),
                "engines": sorted({r.get("engine") or "" for r in rows if r.get("engine")}),
                "last_message": next((r.get("message") for r in sorted(
                    rows, key=lambda x: -(x.get("updated_at") or 0)) if r.get("message")), ""),
                "elapsed_ms": sum(int(r.get("duration_ms") or 0) for r in rows),
            }
        return out

    @staticmethod
    def overall(stage_rows):
        if not stage_rows:
            return {"pct": 0.0, "done": 0, "total": 0, "failed": 0}
        n = len(stage_rows)
        done = sum(1 for r in stage_rows if r["status"] in ("done", "skipped", "deferred"))
        failed = sum(1 for r in stage_rows if r["status"] == "failed")
        running = [r for r in stage_rows if r["status"] == "running"]
        partial = sum(float(r.get("progress") or 0) for r in running) / 100.0
        return {"pct": round(min(99.9, (done + partial) * 100.0 / n), 1) if done < n else 100.0,
                "done": done, "total": n, "failed": failed, "running": len(running)}

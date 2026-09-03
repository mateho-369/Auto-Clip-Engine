"""SQLite persistence layer — the studio's long-term memory.

Everything is stored: which input mode a project used, the Director's locked
script, every generated script, every scene, every *prompt sent to every model*
(the `prompts` table is an append-only audit log), every intermediate and final
asset with its engine/timings, and the run/stage state machine that makes
resuming and per-stage regeneration possible.

Connections are opened per operation (tiny cost, zero thread headaches) with WAL
on, so the asyncio scheduler, the API and CLI can all write at once.
"""
import os
import sqlite3
import threading

from .util import jdump, jload, new_id, now

SCHEMA_VERSION = 2

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(
  key TEXT PRIMARY KEY, value TEXT
);
CREATE TABLE IF NOT EXISTS projects(
  id TEXT PRIMARY KEY,
  title TEXT NOT NULL DEFAULT 'Untitled',
  mode TEXT NOT NULL DEFAULT 'A',                 -- 'A' = Director script, 'B' = auto idea
  status TEXT NOT NULL DEFAULT 'draft',           -- draft|review|ready|rendering|done|failed|archived
  content_type TEXT NOT NULL DEFAULT 'explainer', -- explainer|what_if|compare|choose|word_nuance|myth_vs_fact|quick_tip
  language TEXT NOT NULL DEFAULT 'km',
  script TEXT NOT NULL DEFAULT '',
  script_locked INTEGER NOT NULL DEFAULT 0,       -- Mode A: no agent may rewrite
  script_origin TEXT NOT NULL DEFAULT '',          -- 'director'|'ai:ollama'|'ai:template'
  topic_hint TEXT NOT NULL DEFAULT '',
  style_notes TEXT NOT NULL DEFAULT '',
  target_duration REAL NOT NULL DEFAULT 30,
  voice_profile_id TEXT NOT NULL DEFAULT '',
  settings_json TEXT NOT NULL DEFAULT '{}',        -- per-project overrides (reuse prompts/settings)
  parent_id TEXT NOT NULL DEFAULT '',
  last_run_id TEXT,
  created_at REAL, updated_at REAL
);
CREATE INDEX IF NOT EXISTS idx_projects_updated ON projects(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_projects_status ON projects(status);

CREATE TABLE IF NOT EXISTS scenes(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  project_id TEXT NOT NULL,
  idx INTEGER NOT NULL,
  text TEXT NOT NULL DEFAULT '',
  visual_prompt TEXT NOT NULL DEFAULT '',
  mood_tag TEXT NOT NULL DEFAULT '',
  est_duration REAL NOT NULL DEFAULT 0,
  audio_duration REAL NOT NULL DEFAULT 0,
  sfx_prompt TEXT NOT NULL DEFAULT '',
  meta_json TEXT NOT NULL DEFAULT '{}',
  UNIQUE(project_id, idx)
);

CREATE TABLE IF NOT EXISTS runs(
  id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'queued',          -- queued|running|paused|completed|failed|cancelled|needs_review
  trigger TEXT NOT NULL DEFAULT 'new',            -- new|resume|regenerate|gpu-catchup
  resume_from TEXT NOT NULL DEFAULT '',
  machine_profile TEXT NOT NULL DEFAULT '',
  gpu_policy_json TEXT NOT NULL DEFAULT '{}',
  stage_filter_json TEXT NOT NULL DEFAULT '[]',   -- regenerate: which stages were forced dirty
  error TEXT NOT NULL DEFAULT '',
  stats_json TEXT NOT NULL DEFAULT '{}',
  started_at REAL, finished_at REAL, created_at REAL
);
CREATE INDEX IF NOT EXISTS idx_runs_project ON runs(project_id, created_at DESC);

CREATE TABLE IF NOT EXISTS stage_runs(
  id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  project_id TEXT NOT NULL,
  stage TEXT NOT NULL,
  scene_idx INTEGER NOT NULL DEFAULT -1,          -- -1 = whole-run stage
  status TEXT NOT NULL DEFAULT 'pending',         -- pending|queued|running|done|failed|skipped|deferred|stale|blocked|cancelled
  attempt INTEGER NOT NULL DEFAULT 0,
  progress REAL NOT NULL DEFAULT 0,
  message TEXT NOT NULL DEFAULT '',
  error TEXT NOT NULL DEFAULT '',
  engine TEXT NOT NULL DEFAULT '',
  inherited_from TEXT NOT NULL DEFAULT '',        -- stage_run id copied on resume (output reused)
  started_at REAL, finished_at REAL, updated_at REAL, duration_ms INTEGER NOT NULL DEFAULT 0
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_stage_unique ON stage_runs(run_id, stage, scene_idx);
CREATE INDEX IF NOT EXISTS idx_stage_run ON stage_runs(run_id);

CREATE TABLE IF NOT EXISTS assets(
  id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL,
  run_id TEXT NOT NULL DEFAULT '',
  stage TEXT NOT NULL DEFAULT '',
  scene_idx INTEGER NOT NULL DEFAULT -1,
  kind TEXT NOT NULL,                             -- voice|voice_final|video|video_fit|ambient|qa|final|srt|poster|manifest|scenes|script
  path TEXT NOT NULL,
  relpath TEXT NOT NULL DEFAULT '',
  mime TEXT NOT NULL DEFAULT '',
  size_bytes INTEGER NOT NULL DEFAULT 0,
  duration REAL NOT NULL DEFAULT 0,
  meta_json TEXT NOT NULL DEFAULT '{}',
  created_at REAL
);
CREATE INDEX IF NOT EXISTS idx_assets_lookup ON assets(project_id, stage, scene_idx, kind, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_assets_run ON assets(run_id, kind);

CREATE TABLE IF NOT EXISTS prompts(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  project_id TEXT NOT NULL DEFAULT '',
  run_id TEXT NOT NULL DEFAULT '',
  stage TEXT NOT NULL DEFAULT '',
  scene_idx INTEGER NOT NULL DEFAULT -1,
  role TEXT NOT NULL DEFAULT '',
  model TEXT NOT NULL DEFAULT '',
  engine TEXT NOT NULL DEFAULT '',
  system TEXT NOT NULL DEFAULT '',
  user TEXT NOT NULL DEFAULT '',
  response TEXT NOT NULL DEFAULT '',
  ok INTEGER NOT NULL DEFAULT 1,
  error TEXT NOT NULL DEFAULT '',
  latency_ms INTEGER NOT NULL DEFAULT 0,
  created_at REAL
);
CREATE INDEX IF NOT EXISTS idx_prompts_lookup ON prompts(project_id, created_at DESC);

CREATE TABLE IF NOT EXISTS voice_profiles(
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  pth_path TEXT NOT NULL DEFAULT '',
  index_path TEXT NOT NULL DEFAULT '',
  sample_path TEXT NOT NULL DEFAULT '',
  sample_seconds REAL NOT NULL DEFAULT 0,
  engine TEXT NOT NULL DEFAULT 'rvc',
  pitch INTEGER NOT NULL DEFAULT 0,
  index_rate REAL NOT NULL DEFAULT 0.75,
  rms_mix_rate REAL NOT NULL DEFAULT 0.25,
  f0_method TEXT NOT NULL DEFAULT 'rmvpe',
  notes TEXT NOT NULL DEFAULT '',
  training_status TEXT NOT NULL DEFAULT '',
  created_at REAL
);

CREATE TABLE IF NOT EXISTS events(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL DEFAULT '',
  project_id TEXT NOT NULL DEFAULT '',
  ts REAL NOT NULL,
  kind TEXT NOT NULL,
  stage TEXT NOT NULL DEFAULT '',
  scene_idx INTEGER NOT NULL DEFAULT -1,
  payload_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id, id);
"""

PROJECT_COLS = ("id,title,mode,status,content_type,language,script,script_locked,script_origin,"
                "topic_hint,style_notes,target_duration,voice_profile_id,settings_json,parent_id,"
                "last_run_id,created_at,updated_at")


class Database:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self._lock = threading.Lock()
        self._local = threading.local()
        self._init()

    # ------------------------------------------------------------ plumbing
    def _conn(self):
        con = getattr(self._local, "con", None)
        if con is None:
            con = sqlite3.connect(self.path, timeout=30, isolation_level=None)
            con.row_factory = sqlite3.Row
            con.execute("PRAGMA journal_mode=WAL")
            con.execute("PRAGMA synchronous=NORMAL")
            con.execute("PRAGMA foreign_keys=ON")
            con.execute("PRAGMA busy_timeout=8000")
            self._local.con = con
        return con

    def _init(self):
        con = self._conn()
        with self._lock:
            con.executescript(SCHEMA)
            self._migrate(con)
            con.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('schema_version',?)",
                        (str(SCHEMA_VERSION),))

    def _migrate(self, con):
        """Add columns introduced after schema v1 to databases created earlier."""
        cols = {r["name"] for r in con.execute("PRAGMA table_info(projects)").fetchall()}
        if "content_type" not in cols:
            con.execute("ALTER TABLE projects ADD COLUMN content_type TEXT NOT NULL DEFAULT 'explainer'")
        rows = con.execute("PRAGMA table_info(scenes)").fetchall()
        _ = rows  # reserved for future scene-level migrations

    def close(self):
        con = getattr(self._local, "con", None)
        if con is not None:
            try:
                con.close()
            except Exception:
                pass
            self._local.con = None

    def execute(self, sql, params=()):
        return self._conn().execute(sql, params)

    def query(self, sql, params=()):
        return [dict(r) for r in self._conn().execute(sql, params).fetchall()]

    def one(self, sql, params=()):
        r = self._conn().execute(sql, params).fetchone()
        return dict(r) if r else None

    # ------------------------------------------------------------ projects
    def create_project(self, **kw):
        pid = kw.get("id") or f"p{new_id(7)}"
        ts = now()
        self.execute(
            f"INSERT INTO projects ({PROJECT_COLS}) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (pid, kw.get("title") or "Untitled", kw.get("mode") or "A", kw.get("status") or "draft",
             kw.get("content_type") or "explainer", kw.get("language") or "km",
             kw.get("script") or "", 1 if kw.get("script_locked") else 0,
             kw.get("script_origin") or "", kw.get("topic_hint") or "", kw.get("style_notes") or "",
             float(kw.get("target_duration") or 30), kw.get("voice_profile_id") or "",
             jdump(kw.get("settings") or {}), kw.get("parent_id") or "", kw.get("last_run_id"),
             ts, ts))
        return self.get_project(pid)

    def get_project(self, pid):
        row = self.one("SELECT * FROM projects WHERE id=?", (pid,))
        if not row:
            return None
        row["settings"] = jload(row.pop("settings_json", "{}"), {})
        row["script_locked"] = bool(row.get("script_locked"))
        row["scenes"] = self.list_scenes(pid)
        return row

    def update_project(self, pid, **kw):
        allowed = {"title", "mode", "status", "content_type", "language", "script", "script_locked",
                   "script_origin", "topic_hint", "style_notes", "target_duration",
                   "voice_profile_id", "settings_json", "parent_id", "last_run_id"}
        sets, params = [], []
        for k, v in kw.items():
            if k == "settings":
                k, v = "settings_json", jdump(v)
            elif k == "script_locked":
                v = 1 if v else 0
            if k in allowed:
                sets.append(f"{k}=?")
                params.append(v)
        if not sets:
            return self.get_project(pid)
        sets.append("updated_at=?")
        params.append(now())
        params.append(pid)
        self.execute(f"UPDATE projects SET {', '.join(sets)} WHERE id=?", params)
        return self.get_project(pid)

    def delete_project(self, pid):
        for table in ("projects", "scenes", "runs", "stage_runs", "assets", "prompts"):
            col = "id" if table == "projects" else "project_id"
            self.execute(f"DELETE FROM {table} WHERE {col}=?", (pid,))
        self.execute("DELETE FROM events WHERE project_id=?", (pid,))
        return True

    def list_projects(self, search="", status="", mode="", sort="updated_desc", limit=200, offset=0):
        where, params = [], []
        if search:
            where.append("(title LIKE ? OR script LIKE ? OR topic_hint LIKE ? OR id LIKE ?)")
            like = f"%{search}%"
            params += [like, like, like, like]
        if status:
            where.append("status=?")
            params.append(status)
        if mode:
            where.append("mode=?")
            params.append(mode)
        order = {"updated_desc": "updated_at DESC", "updated_asc": "updated_at ASC",
                 "title_asc": "title COLLATE NOCASE ASC", "title_desc": "title COLLATE NOCASE DESC",
                 "created_desc": "created_at DESC", "status_asc": "status ASC, updated_at DESC"}.get(
            sort, "updated_at DESC")
        sql = ("SELECT id,title,mode,status,content_type,language,target_duration,script_origin,"
               "voice_profile_id,parent_id,last_run_id,created_at,updated_at,"
               "substr(script,1,240) AS script_excerpt, length(script) AS script_chars,"
               "(SELECT COUNT(*) FROM scenes s WHERE s.project_id=projects.id) AS scene_count,"
               "(SELECT COUNT(*) FROM runs r WHERE r.project_id=projects.id) AS run_count,"
               "(SELECT r2.status FROM runs r2 WHERE r2.project_id=projects.id "
               " ORDER BY r2.created_at DESC LIMIT 1) AS last_run_status "
               "FROM projects")
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += f" ORDER BY {order} LIMIT ? OFFSET ?"
        params += [int(limit), int(offset)]
        return self.query(sql, params)

    def project_counts(self):
        rows = self.query("SELECT status, COUNT(*) n FROM projects GROUP BY status")
        return {r["status"]: r["n"] for r in rows}

    # ------------------------------------------------------------ scenes
    def replace_scenes(self, pid, scenes):
        """Scenes come out of Stage 1 (or an edit) as a full ordered list."""
        con = self._conn()
        con.execute("BEGIN")
        con.execute("DELETE FROM scenes WHERE project_id=?", (pid,))
        for i, s in enumerate(scenes):
            con.execute(
                "INSERT INTO scenes (project_id,idx,text,visual_prompt,mood_tag,est_duration,"
                "audio_duration,sfx_prompt,meta_json) VALUES (?,?,?,?,?,?,?,?,?)",
                (pid, i, s.get("text", ""), s.get("visual_prompt", ""), s.get("mood_tag", ""),
                 float(s.get("estimated_duration_sec") or s.get("est_duration") or 0),
                 float(s.get("audio_duration") or 0), s.get("sfx_prompt", ""),
                 jdump({k: v for k, v in s.items()
                        if k not in ("text", "visual_prompt", "mood_tag", "estimated_duration_sec",
                                     "audio_duration", "sfx_prompt")})))
        con.execute("COMMIT")
        return self.list_scenes(pid)

    def list_scenes(self, pid):
        rows = self.query("SELECT * FROM scenes WHERE project_id=? ORDER BY idx ASC", (pid,))
        for r in rows:
            meta = jload(r.pop("meta_json", "{}"), {})
            inner = meta.pop("meta", None)          # older boards nested scene meta
            if isinstance(inner, dict):
                for k, v in inner.items():
                    meta.setdefault(k, v)
            r["meta"] = meta
            r["estimated_duration_sec"] = r.get("est_duration") or 0
        return rows

    def get_scene(self, pid, idx):
        for s in self.list_scenes(pid):
            if s["idx"] == idx:
                return s
        return None

    def update_scene(self, pid, idx, **kw):
        allowed = {"text", "visual_prompt", "mood_tag", "est_duration", "audio_duration", "sfx_prompt"}
        sets, params = [], []
        for k, v in kw.items():
            if k == "estimated_duration_sec":
                k = "est_duration"
            if k in allowed:
                sets.append(f"{k}=?")
                params.append(v)
        if sets:
            params += [pid, idx]
            self.execute(f"UPDATE scenes SET {', '.join(sets)} WHERE project_id=? AND idx=?", params)
        return self.get_scene(pid, idx)

    # ------------------------------------------------------------ runs
    def create_run(self, project_id, **kw):
        rid = kw.get("id") or f"r{new_id(7)}"
        ts = now()
        self.execute(
            "INSERT INTO runs (id,project_id,status,trigger,resume_from,machine_profile,gpu_policy_json,"
            "stage_filter_json,error,stats_json,started_at,finished_at,created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (rid, project_id, kw.get("status") or "queued", kw.get("trigger") or "new",
             kw.get("resume_from") or "", kw.get("machine_profile") or "",
             jdump(kw.get("gpu_policy") or {}), jdump(kw.get("stage_filter") or []),
             kw.get("error") or "", jdump(kw.get("stats") or {}),
             ts if kw.get("start_now") else kw.get("started_at"), kw.get("finished_at"), ts))
        return self.get_run(rid)

    def get_run(self, rid):
        row = self.one("SELECT * FROM runs WHERE id=?", (rid,))
        if not row:
            return None
        row["gpu_policy"] = jload(row.pop("gpu_policy_json", "{}"), {})
        row["stage_filter"] = jload(row.pop("stage_filter_json", "[]"), [])
        row["stats"] = jload(row.pop("stats_json", "{}"), {})
        row["stages"] = self.list_stages(rid)
        row["assets"] = self.list_assets(run_id=rid)
        return row

    def update_run(self, rid, **kw):
        allowed = {"status", "error", "stats_json", "started_at", "finished_at", "last_stage",
                   "machine_profile", "gpu_policy_json", "stage_filter_json", "trigger", "resume_from"}
        sets, params = [], []
        for k, v in kw.items():
            if k == "stats":
                k, v = "stats_json", jdump(v)
            if k == "stage_filter":
                k, v = "stage_filter_json", jdump(v)
            if k == "gpu_policy":
                k, v = "gpu_policy_json", jdump(v)
            if k in allowed:
                sets.append(f"{k}=?")
                params.append(v)
        if sets:
            params.append(rid)
            self.execute(f"UPDATE runs SET {', '.join(sets)} WHERE id=?", params)
        return self.get_run(rid)

    def list_runs(self, pid=None, limit=100):
        if pid:
            rows = self.query("SELECT * FROM runs WHERE project_id=? ORDER BY created_at DESC LIMIT ?",
                              (pid, int(limit)))
        else:
            rows = self.query("SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (int(limit),))
        for r in rows:
            r["stats"] = jload(r.get("stats_json"), {})
            r.pop("stats_json", None)
        return rows

    def latest_run(self, pid):
        return self.one("SELECT * FROM runs WHERE project_id=? ORDER BY created_at DESC LIMIT 1", (pid,))

    def active_runs(self):
        return self.query("SELECT * FROM runs WHERE status IN ('queued','running','paused') "
                          "ORDER BY created_at DESC")

    # ------------------------------------------------------------ stage runs
    def upsert_stage(self, run_id, project_id, stage, scene_idx, **kw):
        key = f"{run_id}:{stage}:{scene_idx}"
        row = self.one("SELECT * FROM stage_runs WHERE run_id=? AND stage=? AND scene_idx=?",
                       (run_id, stage, int(scene_idx)))
        ts = now()
        if row is None:
            sid = kw.pop("id", None) or f"{new_id(6)}"
            self.execute(
                "INSERT INTO stage_runs (id,run_id,project_id,stage,scene_idx,status,attempt,progress,"
                "message,error,engine,inherited_from,started_at,finished_at,updated_at,duration_ms) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (sid, run_id, project_id, stage, int(scene_idx), kw.get("status") or "pending",
                 kw.get("attempt", 0), kw.get("progress", 0), kw.get("message", ""), kw.get("error", ""),
                 kw.get("engine", ""), kw.get("inherited_from", ""), kw.get("started_at"),
                 kw.get("finished_at"), ts, kw.get("duration_ms", 0)))
            return self.one("SELECT * FROM stage_runs WHERE id=?", (sid,))
        sets, params = [], []
        for k in ("status", "attempt", "progress", "message", "error", "engine", "started_at",
                  "finished_at", "duration_ms", "inherited_from"):
            if k in kw:
                sets.append(f"{k}=?")
                params.append(kw[k])
        if not sets:
            return row
        sets.append("updated_at=?")
        params.append(ts)
        params.append(row["id"])
        self.execute(f"UPDATE stage_runs SET {', '.join(sets)} WHERE id=?", params)
        out = self.one("SELECT * FROM stage_runs WHERE id=?", (row["id"],))
        out["key"] = key
        return out

    def list_stages(self, run_id):
        rows = self.query("SELECT * FROM stage_runs WHERE run_id=? "
                          "ORDER BY scene_idx ASC, id ASC", (run_id,))
        for r in rows:
            # same format as pipeline.spec.job_key, so "which job is this row?" has
            # one answer across the codebase (the resume path depends on it matching)
            r["key"] = f"{r['stage']}#{r['scene_idx']}"
        return rows

    def get_stage(self, run_id, stage, scene_idx=-1):
        return self.one("SELECT * FROM stage_runs WHERE run_id=? AND stage=? AND scene_idx=?",
                        (run_id, stage, int(scene_idx)))

    def set_stage_status(self, run_id, stage, scene_idx, status, **kw):
        kw["status"] = status
        return self.upsert_stage(run_id, kw.pop("project_id", "") or
                                 (self.one("SELECT project_id FROM runs WHERE id=?", (run_id,)) or {})
                                 .get("project_id", ""), stage, scene_idx, **kw)

    def delete_stages(self, run_id, stage, scene_indices=None):
        if scene_indices:
            marks = ",".join("?" * len(scene_indices))
            self.execute(f"DELETE FROM stage_runs WHERE run_id=? AND stage=? AND scene_idx IN ({marks})",
                         [run_id, stage] + [int(i) for i in scene_indices])
        else:
            self.execute("DELETE FROM stage_runs WHERE run_id=? AND stage=?", (run_id, stage))

    def stage_summary(self, run_id):
        rows = self.query("SELECT stage, status, COUNT(*) n FROM stage_runs WHERE run_id=? "
                          "GROUP BY stage, status", (run_id,))
        out = {}
        for r in rows:
            out.setdefault(r["stage"], {})[r["status"]] = r["n"]
        return out

    # ------------------------------------------------------------ assets
    def add_asset(self, project_id, kind, path, **kw):
        aid = kw.pop("id", None) or f"a{new_id(7)}"
        size = os.path.getsize(path) if path and os.path.exists(path) else 0
        self.execute(
            "INSERT INTO assets (id,project_id,run_id,stage,scene_idx,kind,path,relpath,mime,"
            "size_bytes,duration,meta_json,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (aid, project_id, kw.get("run_id") or "", kw.get("stage") or "",
             int(kw.get("scene_idx", -1)), kind, path, kw.get("relpath") or os.path.basename(path or ""),
             kw.get("mime") or _mime(path), size, float(kw.get("duration") or 0),
             jdump(kw.get("meta") or {}), now()))
        return self.one("SELECT * FROM assets WHERE id=?", (aid,))

    def list_assets(self, project_id=None, run_id=None, stage=None, scene_idx=None, kind=None, limit=500):
        where, params = [], []
        for col, val in (("project_id", project_id), ("run_id", run_id), ("stage", stage),
                         ("scene_idx", scene_idx), ("kind", kind)):
            if val is not None:
                where.append(f"{col}=?")
                params.append(val)
        sql = "SELECT * FROM assets"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += f" ORDER BY created_at DESC LIMIT {int(limit)}"
        rows = self.query(sql, params)
        for r in rows:
            r["meta"] = jload(r.pop("meta_json", "{}"), {})
        return rows

    def latest_asset(self, project_id, kind, stage="", scene_idx=None):
        """The current output for a (kind, stage, scene) slot — what the UI plays."""
        where = ["project_id=?", "kind=?"]
        params = [project_id, kind]
        if stage:
            where.append("stage=?")
            params.append(stage)
        if scene_idx is not None:
            where.append("scene_idx=?")
            params.append(int(scene_idx))
        row = self.one("SELECT * FROM assets WHERE " + " AND ".join(where) +
                       " ORDER BY created_at DESC LIMIT 1", params)
        if row:
            row["meta"] = jload(row.pop("meta_json", "{}"), {})
        return row

    def delete_asset(self, aid):
        self.execute("DELETE FROM assets WHERE id=?", (aid,))
        return True

    # ------------------------------------------------------------ prompts log
    def log_prompt(self, **kw):
        self.execute(
            "INSERT INTO prompts (project_id,run_id,stage,scene_idx,role,model,engine,system,user,"
            "response,ok,error,latency_ms,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (kw.get("project_id") or "", kw.get("run_id") or "", kw.get("stage") or "",
             int(kw.get("scene_idx", -1)), kw.get("role") or "", kw.get("model") or "",
             kw.get("engine") or "", kw.get("system") or "", kw.get("user") or "",
             kw.get("response") or "", 1 if kw.get("ok", True) else 0, kw.get("error") or "",
             int(kw.get("latency_ms") or 0), now()))
        return True

    def list_prompts(self, project_id=None, run_id=None, stage=None, limit=200):
        where, params = [], []
        for col, val in (("project_id", project_id), ("run_id", run_id), ("stage", stage)):
            if val:
                where.append(f"{col}=?")
                params.append(val)
        sql = "SELECT * FROM prompts"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += f" ORDER BY id DESC LIMIT {int(limit)}"
        return self.query(sql, params)

    # ------------------------------------------------------------ voice profiles
    def create_voice_profile(self, **kw):
        pid = kw.get("id") or f"v{new_id(6)}"
        self.execute(
            "INSERT INTO voice_profiles (id,name,pth_path,index_path,sample_path,sample_seconds,engine,"
            "pitch,index_rate,rms_mix_rate,f0_method,notes,training_status,created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (pid, kw.get("name") or "My Voice", kw.get("pth_path") or "", kw.get("index_path") or "",
             kw.get("sample_path") or "", float(kw.get("sample_seconds") or 0),
             kw.get("engine") or "rvc", int(kw.get("pitch") or 0),
             float(kw.get("index_rate") if kw.get("index_rate") is not None else 0.75),
             float(kw.get("rms_mix_rate") if kw.get("rms_mix_rate") is not None else 0.25),
             kw.get("f0_method") or "rmvpe", kw.get("notes") or "",
             kw.get("training_status") or "", now()))
        return self.one("SELECT * FROM voice_profiles WHERE id=?", (pid,))

    def list_voice_profiles(self):
        return self.query("SELECT * FROM voice_profiles ORDER BY created_at DESC")

    def get_voice_profile(self, pid):
        return self.one("SELECT * FROM voice_profiles WHERE id=?", (pid,))

    def update_voice_profile(self, pid, **kw):
        allowed = {"name", "pth_path", "index_path", "sample_path", "sample_seconds", "engine",
                   "pitch", "index_rate", "rms_mix_rate", "f0_method", "notes", "training_status"}
        sets, params = [], []
        for k, v in kw.items():
            if k in allowed:
                sets.append(f"{k}=?")
                params.append(v)
        if sets:
            params.append(pid)
            self.execute(f"UPDATE voice_profiles SET {', '.join(sets)} WHERE id=?", params)
        return self.get_voice_profile(pid)

    def delete_voice_profile(self, pid):
        self.execute("DELETE FROM voice_profiles WHERE id=?", (pid,))
        return True

    # ------------------------------------------------------------ event log
    def log_event(self, run_id, project_id, kind, payload=None, stage="", scene_idx=-1):
        self.execute("INSERT INTO events (run_id,project_id,ts,kind,stage,scene_idx,payload_json) "
                     "VALUES (?,?,?,?,?,?,?)",
                     (run_id or "", project_id or "", now(), kind, stage or "", int(scene_idx),
                      jdump(payload or {})))
        # keep the log bounded (studio runs for months on a home PC)
        self.execute("DELETE FROM events WHERE id < "
                     "(SELECT MAX(id) - 6000 FROM events)")
        return True

    def list_events(self, run_id=None, limit=400, after_id=0):
        if run_id:
            rows = self.query("SELECT * FROM events WHERE run_id=? AND id>? ORDER BY id ASC LIMIT ?",
                              (run_id, int(after_id), int(limit)))
        else:
            rows = self.query("SELECT * FROM events WHERE id>? ORDER BY id ASC LIMIT ?",
                              (int(after_id), int(limit)))
        for r in rows:
            r["payload"] = jload(r.pop("payload_json", "{}"), {})
        return rows

    # ------------------------------------------------------------ stats
    def stats(self):
        out = {}
        for table, key in (("projects", "projects"), ("runs", "runs"), ("assets", "assets"),
                           ("prompts", "prompts"), ("scenes", "scenes"), ("voice_profiles", "voices")):
            row = self.one(f"SELECT COUNT(*) n FROM {table}")
            out[key] = (row or {}).get("n", 0)
        row = self.one("SELECT COALESCE(SUM(size_bytes),0) b FROM assets")
        out["disk_bytes"] = (row or {}).get("b", 0)
        row = self.one("SELECT COUNT(*) n FROM runs WHERE status='completed'")
        out["completed_runs"] = (row or {}).get("n", 0)
        return out


def _mime(path):
    ext = (os.path.splitext(path or "")[1] or "").lower()
    return {
        ".wav": "audio/wav", ".mp3": "audio/mpeg", ".ogg": "audio/ogg", ".flac": "audio/flac",
        ".mp4": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm", ".mkv": "video/x-matroska",
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".json": "application/json",
        ".srt": "application/x-subrip", ".txt": "text/plain", ".zip": "application/zip",
    }.get(ext, "application/octet-stream")

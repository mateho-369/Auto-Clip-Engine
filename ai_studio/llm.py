"""The LLM access layer for the studio's three language agents.

Reuses ``ai_creator.ollama_client`` for connectivity + the battle-tested JSON
recovery, and adds what a *pipeline* needs:

* role → model resolution with a per-role **fallback model** (sailor2:8b on
  Machine A, llama3.2:3b on CPU-only Machine B) and an automatic switch to the
  fallback when the primary model isn't pulled;
* ``format=json`` + one schema-repair retry, because a scene list that isn't
  parseable is worse than no scene list;
* ``keep_alive=0`` unload after the stage (8GB VRAM: the LLM must not sit on the
  card while Wan or MMAudio loads);
* every system/user prompt, model, latency and error written to the `prompts`
  table — the browsable "memory" the brief asks for;
* a hard contract with the caller: if Ollama is offline or the JSON is junk we
  return ``None`` and the caller uses its deterministic fallback. The pipeline
  never dead-ends on an LLM.
"""
import asyncio
import json
import time

from ai_creator.ollama_client import extract_json   # reused on purpose

from .util import scrub

try:  # stdlib only, but keep the import defensive
    import urllib.error
    import urllib.request
except Exception:  # pragma: no cover
    urllib = None


def ollama_online(host, timeout=2.0):
    if urllib is None:
        return False
    try:
        with urllib.request.urlopen((host or "").rstrip("/") + "/api/tags", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


class LLMError(RuntimeError):
    pass


class LLM:
    """One LLM facade per run (holds config + prompt-log destination)."""

    def __init__(self, cfg, db=None, run_id="", project_id="", timeout=None):
        self.cfg = cfg
        self.db = db
        self.run_id = run_id
        self.project_id = project_id
        self.host = (cfg.get("ollama", {}).get("host") or "http://127.0.0.1:11434").rstrip("/")
        self.timeout = timeout or cfg.get("ollama", {}).get("request_timeout_sec", 300)
        self.notes = []

    # ------------------------------------------------------------- plumbing
    def role_cfg(self, role):
        roles = self.cfg.get("ollama", {}).get("roles", {})
        return dict(roles.get(role) or {})

    def model_for(self, role):
        rc = self.role_cfg(role)
        return (rc.get("model") or "sailor2:8b", rc.get("fallback_model") or "llama3.2:3b",
                float(rc.get("temperature", 0.6)))

    def enabled(self, role):
        rc = self.role_cfg(role)
        return bool(rc.get("enabled", True))

    def list_models(self):
        try:
            with urllib.request.urlopen(self.host + "/api/tags", timeout=4) as r:
                data = json.loads(r.read().decode("utf-8"))
            return [m.get("name", "") for m in data.get("models", []) if m.get("name")]
        except Exception:
            return []

    def has_model(self, name):
        names = self.list_models()
        if not names:
            return False
        base = (name or "").split(":")[0]
        return any(n == name or n.split(":")[0] == base for n in names)

    def _post(self, path, payload, timeout):
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self.host + path, data=data,
                                      headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def _unload(self):
        if not self.cfg.get("vram", {}).get("unload_llm_after_stage", True):
            return
        try:  # free the card for the next GPU stage (Wan / MMAudio)
            self._post("/api/generate", {"model": self._last_model or "", "prompt": "",
                                        "keep_alive": 0}, 8)
        except Exception:
            pass

    # --------------------------------------------------------------- calls
    def _call_sync(self, model, system, user, temperature, json_mode, timeout):
        payload = {
            "model": model,
            "stream": False,
            "keep_alive": self.cfg.get("ollama", {}).get("keep_alive", "0"),
            "options": {
                "temperature": temperature,
                "num_ctx": int(self.cfg.get("ollama", {}).get("num_ctx", 4096)),
                "num_predict": 2048,
            },
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
        }
        if json_mode:
            payload["format"] = "json"
        self._last_model = model
        d = self._post("/api/chat", payload, timeout)
        return (d.get("message") or {}).get("content", "") or ""

    def _log(self, role, stage, scene_idx, model, engine, system, user, response, ok, error, ms):
        if self.db is None:
            return
        try:
            self.db.log_prompt(project_id=self.project_id, run_id=self.run_id, stage=stage,
                              scene_idx=scene_idx, role=role, model=model, engine=engine,
                              system=scrub(system, 4000), user=scrub(user, 6000),
                              response=scrub(response, 8000), ok=ok, error=scrub(error, 800),
                              latency_ms=int(ms))
        except Exception:
            pass

    async def ask(self, role, stage, system, user, *, scene_idx=-1, json_mode=True,
                  validate=None, attempts=2, timeout=None):
        """Ask a role. Returns (parsed, meta); parsed=None when it must fall back.

        ``validate`` is a callable(data) -> (ok, cleaned) so a stage can enforce
        its own JSON contract without the LLM layer knowing about scenes.
        """
        primary, fallback, temp = self.model_for(role)
        if not self.enabled(role):
            return None, {"engine": "off", "reason": f"role '{role}' switched off in settings"}
        if urllib is None or not ollama_online(self.host):
            self.notes.append("ollama offline")
            return None, {"engine": "offline", "reason": "Ollama not reachable — deterministic fallback"}
        models = []
        for m in (primary, fallback):
            if m and m not in models:
                models.append(m)
        online = None
        last_err = ""
        t0 = time.time()
        for model in models:
            for k in range(max(1, attempts)):
                try:
                    raw = await asyncio.to_thread(self._call_sync, model, system, user, temp,
                                                  json_mode, timeout or self.timeout)
                except Exception as e:
                    last_err = f"{type(e).__name__}: {str(e)[:200]}"
                    if online is None:
                        online = set(self.list_models())
                    base = model.split(":")[0]
                    if online and not any(m.split(":")[0] == base for m in online):
                        last_err += f" (model '{model}' not pulled)"
                        break            # don't retry a missing model, try the fallback
                    if k + 1 >= max(1, attempts):
                        break
                    await asyncio.sleep(0.4)
                    continue
                data = extract_json(raw) if json_mode else {"text": raw.strip()}
                ok, cleaned = (True, data)
                if validate is not None:
                    try:
                        ok, cleaned = validate(data)
                    except Exception as e:
                        ok, cleaned = False, None
                        last_err = f"validate: {e}"
                elif data is None:
                    ok, cleaned = False, None
                    last_err = "no JSON in response"
                ms = int((time.time() - t0) * 1000)
                self._log(role, stage, scene_idx, model, "ollama", system, user, raw, 1 if ok else 0,
                          "" if ok else last_err, ms)
                if ok:
                    self._unload()
                    return cleaned, {"engine": "ollama", "model": model, "latency_ms": ms,
                                     "chars": len(raw or ""), "attempts": k + 1}
                last_err = last_err or "response failed validation"
                if k + 1 >= max(1, attempts):
                    break
                await asyncio.sleep(0.3)     # one repair retry with same model
        self._unload()
        return None, {"engine": "error", "model": primary, "reason": last_err or "llm failed"}

    async def ask_json(self, role, stage, system, user, **kw):
        data, meta = await self.ask(role, stage, system, user, json_mode=True, **kw)
        return data, meta


def scenes_validator(expected_count=None, require_text=True):
    """Stage-1 contract: {scenes:[{text, visual_prompt, estimated_duration_sec, mood_tag}]}"""
    def _v(data):
        if isinstance(data, list):
            data = {"scenes": data}
        if not isinstance(data, dict) or not isinstance(data.get("scenes"), list):
            return False, None
        out = []
        for raw in data["scenes"]:
            if isinstance(raw, str):
                raw = {"text": raw}
            if not isinstance(raw, dict):
                continue
            text = str(raw.get("text") or raw.get("script") or "").strip()
            if not text and require_text:
                continue
            try:
                dur = float(raw.get("estimated_duration_sec") or raw.get("duration") or 0)
            except Exception:
                dur = 0.0
            out.append({
                "text": text,
                "visual_prompt": str(raw.get("visual_prompt") or raw.get("visual") or "").strip(),
                "estimated_duration_sec": round(max(0.0, dur), 2),
                "mood_tag": str(raw.get("mood_tag") or raw.get("mood") or "").strip().lower(),
                "sfx_prompt": str(raw.get("sfx_prompt") or raw.get("ambience") or "").strip(),
                "hook": str(raw.get("hook") or "").strip()[:60],
            })
        if not out:
            return False, None
        if expected_count and abs(len(out) - expected_count) > max(1, expected_count // 2):
            return False, None        # the model dropped/hallucinated whole scenes
        return True, {"scenes": out}
    return _v


def qa_validator():
    def _v(data):
        if not isinstance(data, dict):
            return False, None
        issues = data.get("issues") or data.get("flags") or []
        if isinstance(issues, str):
            issues = [issues]
        clean = []
        for it in issues:
            if isinstance(it, dict):
                clean.append({"scene_idx": int(it.get("scene_idx", it.get("index", -1)) or -1),
                              "severity": str(it.get("severity") or "warn"),
                              "issue": str(it.get("issue") or it.get("message") or "")[:300]})
            elif it:
                clean.append({"scene_idx": -1, "severity": "warn", "issue": str(it)[:300]})
        approved = data.get("approved", data.get("pass"))
        if approved is None:
            approved = not any(c["severity"] == "fail" for c in clean)
        return True, {"approved": bool(approved), "issues": clean,
                      "summary": str(data.get("summary") or "")[:400]}
    return _v


def script_validator(min_chars=40):
    def _v(data):
        if isinstance(data, str):
            data = {"script": data}
        if not isinstance(data, dict):
            return False, None
        script = str(data.get("script") or data.get("text") or "").strip()
        if len(script) < min_chars:
            return False, None
        return True, {"title": str(data.get("title") or "").strip()[:120],
                      "script": script,
                      "logline": str(data.get("logline") or "").strip()[:300]}
    return _v

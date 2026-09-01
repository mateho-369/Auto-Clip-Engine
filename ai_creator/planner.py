"""The studio brain: the controller AI plans a video and DELEGATES the work.

Pipeline (each role runs with its own Ollama model, per the team config):

    user idea
       │
       ▼
   1. PLANNER (controller)  -> scene skeleton: hooks, scripts, sfx, anim, bg
       │
       ▼
   2. SCRIPTWRITER          -> polishes the narration per scene
       │
       ▼
   3. SFX DIRECTOR          -> re-picks sound effects + timing
       │
       ▼
   4. ANIMATOR              -> re-picks character animations + transitions
       │
       ▼
   5. QA (optional)         -> sanity review + fixes
       │
       ▼
   deterministic VALIDATION (always) -> safe, renderable plan JSON

Any role that is disabled, offline, or returns garbage keeps the previous
stage's values — the plan always stays renderable. With Ollama fully
offline, a template fallback planner produces a solid default plan, so the
studio never dead-ends.
"""
import json
import time
import uuid
import numpy as np

from .ollama_client import OllamaClient, extract_json
from .sfx import SFX_LIBRARY
from .animation import ENTRIES, EXITS
from .transitions import TRANSITIONS
from .team import ROLES, ROLE_LABELS

SFX_CHOICES = list(SFX_LIBRARY.keys()) + ["none"]
ANIM_CHOICES = list(ENTRIES)
TRANS_CHOICES = list(TRANSITIONS)
BG_CHOICES = [
    "gradient-violet", "gradient-blue", "gradient-sunset", "gradient-forest",
    "solid-dark", "solid-black", "solid-navy",
    "pattern-dots", "pattern-grid",
]

BG_COLORS = {
    "gradient-violet": ((88, 44, 160), (28, 12, 60)),      # BGR top/bottom
    "gradient-blue": ((150, 90, 30), (30, 18, 12)),
    "gradient-sunset": ((60, 120, 200), (20, 40, 120)),
    "gradient-forest": ((90, 120, 40), (12, 28, 16)),
    "solid-dark": ((28, 28, 32), (28, 28, 32)),
    "solid-black": ((10, 10, 12), (10, 10, 12)),
    "solid-navy": ((60, 32, 22), (14, 8, 4)),
    "pattern-dots": ((36, 32, 44), (18, 16, 24)),
    "pattern-grid": ((30, 34, 46), (14, 16, 24)),
}


class Studio:
    def __init__(self, team_config, client: OllamaClient):
        self.cfg = team_config
        self.client = client
        self.activity = []

    # ---------- activity feed (the UI's "AI team at work" log) ----------
    def _log(self, role, status, model, note=""):
        entry = {"role": role, "label": ROLE_LABELS.get(role, role),
                 "status": status, "model": model, "note": note,
                 "time": round(time.time(), 1)}
        self.activity.append(entry)
        icon = {"ai": "🧠", "fallback": "📋", "skipped": "⏭️", "error": "⚠️"}.get(status, "•")
        print(f"{icon} [{role}] {status} {model or ''} {note}")

    def _call_role(self, role, system, user):
        """Returns (ok, text). Honors the per-role model/temperature/switch."""
        rcfg = self.cfg["roles"].get(role, {})
        if not rcfg.get("enabled", False):
            self._log(role, "skipped", None, "role switched off in AI team settings")
            return False, ""
        model = rcfg.get("model") or self.cfg.get("controller", "llama3.2:3b")
        if not self.client.is_online():
            self._log(role, "fallback", model, "Ollama offline — using built-in fallback")
            return False, ""
        try:
            text = self.client.chat(model, system, user,
                                    temperature=rcfg.get("temperature", 0.7), timeout=240)
            if not text:
                raise ValueError("empty response")
            self._log(role, "ai", model, f"{len(text)} chars")
            return True, text
        except Exception as e:
            self._log(role, "error", model, f"{str(e)[:120]} — kept previous stage values")
            return False, ""

    # ------------------------------ 1. PLANNER ------------------------------
    PLANNER_SYSTEM = (
        "You are the DIRECTOR AI of a local video-creation studio. The user gives you an idea "
        "for a short vertical video narrated by their own character (a presenter avatar on screen). "
        "You are the CONTROLLER: you build the scene plan and your output is delegated to other "
        "specialist AIs (scriptwriter, SFX director, animator) who will refine it.\n"
        "Respond ONLY with valid JSON (no markdown, no commentary) shaped exactly as:\n"
        '{"title": string, "logline": string, "scenes": [{\n'
        '  "hook": string,               // 3-6 word scene label\n'
        '  "script": string,             // 1-3 spoken sentences, plain text, no emojis/stage directions\n'
        '  "sfx": one of [' + ", ".join(SFX_CHOICES) + '],\n'
        '  "sfx_time": number,           // seconds from scene start, 0..3\n'
        '  "animation": one of [' + ", ".join(ANIM_CHOICES) + '],\n'
        '  "transition": one of [' + ", ".join(TRANS_CHOICES) + '],\n'
        '  "background": one of [' + ", ".join(BG_CHOICES) + '],\n'
        '  "duration": number            // seconds, 3..12\n'
        '}]}\n'
        "Rules: total scene duration must be close to the requested length; scene 1 must be a "
        "scroll-stopping HOOK; the final scene must end with a call-to-action; scripts must sound "
        "natural when spoken aloud."
    )

    def _plan_skeleton(self, idea, target_dur, style, character_name):
        user = (
            f"Video idea: {idea}\n"
            f"Character name: {character_name or 'the creator'}\n"
            f"Target total length: {target_dur} seconds\n"
            f"Style: {style or 'punchy, energetic, short-form'}\n"
            f"Return the JSON plan now."
        )
        ok, text = self._call_role("planner", self.PLANNER_SYSTEM, user)
        if ok:
            data = extract_json(text)
            if isinstance(data, dict) and isinstance(data.get("scenes"), list) and data["scenes"]:
                self._log("planner", "ai", self.cfg["roles"]["planner"]["model"],
                          f"planned {len(data['scenes'])} scenes")
                return data
            self._log("planner", "error", None, "invalid JSON from model — using template plan")
        return None

    # ---------------------------- 2. SCRIPTWRITER ----------------------------
    SCRIPT_SYSTEM = (
        "You are the SCRIPTWRITER AI in a local video studio. You receive the scene skeleton of a "
        "short narrated video and return polished, natural spoken narration for each scene. "
        "Keep each script 1-3 sentences (roughly 12-42 words), plain text, no emojis, no stage "
        "directions, no quotation marks. Keep the scene hooks.\n"
        "Respond ONLY with valid JSON: {\"scenes\": [{\"index\": 0-based int, \"script\": string}]}"
    )

    def _refine_scripts(self, plan, character_name, idea):
        payload = {"idea": idea, "character": character_name,
                   "scenes": [{"index": i, "hook": s.get("hook", ""), "script": s.get("script", ""),
                               "duration": s.get("duration")} for i, s in enumerate(plan["scenes"])]}
        ok, text = self._call_role(
            "scriptwriter", self.SCRIPT_SYSTEM,
            f"Scene skeleton JSON:\n{json.dumps(payload, indent=1)}\nReturn the polished scripts JSON.")
        if not ok:
            return plan
        data = extract_json(text)
        if isinstance(data, dict) and isinstance(data.get("scenes"), list):
            by_idx = {int(s.get("index", -1)): str(s.get("script", "")).strip()
                      for s in data["scenes"] if isinstance(s, dict)}
            for i, s in enumerate(plan["scenes"]):
                if i in by_idx and by_idx[i]:
                    s["script"] = by_idx[i][:400]
        return plan

    # ----------------------------- 3. SFX DIRECTOR ----------------------------
    SFX_SYSTEM = (
        "You are the SFX DIRECTOR AI. You assign one sound effect per scene of a short video, at a "
        "precise moment, to make cuts and reveals pop. Available effects and their character:\n"
        + "\n".join(f"- {k}: {v['desc']}" for k, v in SFX_LIBRARY.items())
        + "\n- none: no sound effect\n"
        "Rules: vary the effects (do not repeat the same one in consecutive scenes); use riser or "
        "boom before the big reveal; applause or sparkle near the CTA; sfx_time between 0 and 2.5.\n"
        "Respond ONLY with valid JSON: {\"scenes\": [{\"index\": int, \"sfx\": string, \"sfx_time\": number}]}"
    )

    def _refine_sfx(self, plan):
        payload = [{"index": i, "hook": s.get("hook", ""), "script": s.get("script", "")[:120]}
                   for i, s in enumerate(plan["scenes"])]
        ok, text = self._call_role("sfx_director", self.SFX_SYSTEM,
                                   f"Scenes JSON:\n{json.dumps(payload, indent=1)}\nAssign the SFX JSON now.")
        if not ok:
            return plan
        data = extract_json(text)
        if isinstance(data, dict) and isinstance(data.get("scenes"), list):
            by_idx = {}
            for s in data["scenes"]:
                if isinstance(s, dict):
                    by_idx[int(s.get("index", -1))] = s
            for i, sc in enumerate(plan["scenes"]):
                if i in by_idx:
                    sc["sfx"] = str(by_idx[i].get("sfx", sc.get("sfx", "none"))).lower().strip()
                    try:
                        sc["sfx_time"] = float(by_idx[i].get("sfx_time", sc.get("sfx_time", 0.3)))
                    except Exception:
                        sc["sfx_time"] = float(sc.get("sfx_time", 0.3))
        return plan

    # ------------------------------- 4. ANIMATOR ------------------------------
    ANIM_SYSTEM = (
        "You are the ANIMATOR AI. You choose, per scene, how the on-screen character enters and how "
        "scenes transition. Options:\n"
        f"animations: {', '.join(ANIM_CHOICES)}\n"
        f"transitions: {', '.join(TRANS_CHOICES)}\n"
        "Rules: scene 1 should pop-in or bounce (energy!); keep transitions varied; use zoom or "
        "slide for energetic moments, fade for calm ones, cut for quick cuts.\n"
        "Respond ONLY with valid JSON: {\"scenes\": [{\"index\": int, \"animation\": string, \"transition\": string}]}"
    )

    def _refine_anim(self, plan):
        payload = [{"index": i, "hook": s.get("hook", ""), "script": s.get("script", "")[:100]}
                   for i, s in enumerate(plan["scenes"])]
        ok, text = self._call_role("animator", self.ANIM_SYSTEM,
                                   f"Scenes JSON:\n{json.dumps(payload, indent=1)}\nAssign animations now.")
        if not ok:
            return plan
        data = extract_json(text)
        if isinstance(data, dict) and isinstance(data.get("scenes"), list):
            by_idx = {}
            for s in data["scenes"]:
                if isinstance(s, dict):
                    by_idx[int(s.get("index", -1))] = s
            for i, sc in enumerate(plan["scenes"]):
                if i in by_idx:
                    sc["animation"] = str(by_idx[i].get("animation", sc.get("animation", "pop-in"))).lower().strip()
                    sc["transition"] = str(by_idx[i].get("transition", sc.get("transition", "fade"))).lower().strip()
        return plan

    # --------------------------------- 5. QA ----------------------------------
    QA_SYSTEM = (
        "You are the QA REVIEWER AI. Check this video plan for: empty or unnatural scripts, scenes "
        "shorter than 2.5s or longer than 12s, the same sound effect repeated back-to-back, or a "
        "missing call-to-action at the end. If everything is fine respond exactly: {\"approved\": true}. "
        "Otherwise respond ONLY with valid JSON: {\"approved\": false, \"scenes\": [{\"index\": int, "
        "\"script\": string, \"sfx\": string}] } containing only the corrected fields."
    )

    def _qa_pass(self, plan):
        ok, text = self._call_role("qa", self.QA_SYSTEM,
                                   f"Plan JSON:\n{json.dumps(plan, indent=1)}\nReview it.")
        if not ok:
            return plan
        data = extract_json(text)
        if isinstance(data, dict) and data.get("approved") is False and isinstance(data.get("scenes"), list):
            by_idx = {int(s.get("index", -1)): s for s in data["scenes"] if isinstance(s, dict)}
            for i, sc in enumerate(plan["scenes"]):
                if i in by_idx:
                    if by_idx[i].get("script"):
                        sc["script"] = str(by_idx[i]["script"]).strip()[:400]
                    if by_idx[i].get("sfx"):
                        sc["sfx"] = str(by_idx[i]["sfx"]).lower().strip()
        return plan

    # ------------------------------ main entry ------------------------------
    def plan(self, idea, target_dur=25, style="", character_name=""):
        t0 = time.time()
        idea = (idea or "").strip() or "a fun short video about my channel"
        target_dur = max(8, min(90, int(target_dur or 25)))

        skeleton = self._plan_skeleton(idea, target_dur, style, character_name)
        if skeleton is None:
            plan = fallback_plan(idea, target_dur, character_name)
            self._log("planner", "fallback", None, "template plan generated (no AI available)")
        else:
            plan = {
                "title": str(skeleton.get("title") or f"{character_name or 'Creator'} short"),
                "logline": str(skeleton.get("logline") or ""),
                "scenes": skeleton["scenes"],
            }

        plan = self._refine_scripts(plan, character_name, idea)
        plan = self._refine_sfx(plan)
        plan = self._refine_anim(plan)
        if self.cfg["roles"].get("qa", {}).get("enabled", False):
            plan = self._qa_pass(plan)

        plan = validate_plan(plan)
        plan["total_duration"] = round(sum(s["duration"] for s in plan["scenes"]), 2)
        plan["elapsed_sec"] = round(time.time() - t0, 2)
        plan["activity"] = self.activity
        plan["idea"] = idea
        return plan


def validate_plan(plan):
    """Deterministic guard: the result is ALWAYS renderable."""
    if not isinstance(plan, dict) or not isinstance(plan.get("scenes"), list) or not plan["scenes"]:
        raise ValueError("Plan has no scenes.")
    scenes = []
    for raw in plan["scenes"]:
        if not isinstance(raw, dict):
            continue
        s = dict(raw)
        try:
            dur = float(s.get("duration", 5))
        except Exception:
            dur = 5.0
        s["duration"] = round(max(2.5, min(12.0, dur)), 2)
        s["hook"] = str(s.get("hook") or "Scene").strip()[:60]
        s["script"] = str(s.get("script") or "").strip()[:500]
        s["sfx"] = str(s.get("sfx") or "none").lower().strip()
        if s["sfx"] not in SFX_CHOICES:
            s["sfx"] = "none"
        try:
            s["sfx_time"] = round(max(0.0, min(3.0, float(s.get("sfx_time", 0.3)))), 2)
        except Exception:
            s["sfx_time"] = 0.3
        s["animation"] = str(s.get("animation") or "pop-in").lower().strip()
        if s["animation"] not in ANIM_CHOICES:
            s["animation"] = "pop-in"
        s["transition"] = str(s.get("transition") or "fade").lower().strip()
        if s["transition"] not in TRANS_CHOICES:
            s["transition"] = "fade"
        s["background"] = str(s.get("background") or "gradient-violet").lower().strip()
        if s["background"] not in BG_CHOICES:
            s["background"] = "gradient-violet"
        scenes.append(s)
    if not scenes:
        raise ValueError("Plan has no valid scenes.")
    # back-to-back duplicate SFX -> drop the second
    for i in range(1, len(scenes)):
        if scenes[i]["sfx"] != "none" and scenes[i]["sfx"] == scenes[i - 1]["sfx"]:
            scenes[i]["sfx"] = "none"
    # missing scripts get defaults (never render a silent-but-captionless scene)
    defaults = [
        "Welcome back — this one is going to be great.",
        "Here is the part nobody talks about.",
        "And this is where it gets interesting.",
        "Let's break it down step by step.",
        "That's it — now go make something amazing!",
    ]
    for i, s in enumerate(scenes):
        if not s["script"]:
            s["script"] = defaults[i % len(defaults)]
    out = {
        "title": str(plan.get("title") or "Untitled short")[:120],
        "logline": str(plan.get("logline") or "")[:300],
        "scenes": scenes[:12],
    }
    return out


def fallback_plan(idea, target_dur, character_name):
    """Deterministic template plan — used when Ollama is offline or a model
    returns garbage. Still produces a real, well-structured video plan."""
    n = max(2, min(5, int(round(target_dur / 6))))
    dur_each = round(target_dur / n, 2)
    idea_l = idea[:90].strip()
    scenes = []
    bgs = ["gradient-violet", "gradient-blue", "gradient-sunset", "gradient-forest", "solid-navy"]
    anims = ["pop-in", "bounce", "slide-left", "zoom", "slide-right"]
    trans = ["fade", "slide", "zoom", "wipe", "cut"]
    sfxes = ["whoosh", "ding", "riser", "boom", "applause"]
    hooks = ["The Hook", "The Problem", "The Reveal", "The Proof", "The CTA"]
    scripts = [
        f"Stop scrolling — {idea_l} is about to make total sense in the next few seconds.",
        f"Most people get {idea_l} completely wrong, and it costs them time every single day.",
        f"Here is the part that changes everything about {idea_l} — pay close attention to this.",
        f"Watch how quickly this works once you know the trick — no excuses left.",
        f"That is the whole secret. If this helped, follow for more like this — see you in the next one!",
    ]
    for i in range(n):
        # the last scene always carries the call-to-action
        script = scripts[4] if i == n - 1 else scripts[i % len(scripts)]
        scenes.append({
            "hook": hooks[i % len(hooks)],
            "script": script,
            "sfx": sfxes[i % len(sfxes)],
            "sfx_time": 0.3,
            "animation": anims[i % len(anims)],
            "transition": trans[i % len(trans)],
            "background": bgs[i % len(bgs)],
            "duration": dur_each,
        })
    return {"title": idea_l.title()[:120] or "Creator short",
            "logline": f"Auto-generated template plan for: {idea_l}",
            "scenes": scenes}


def new_plan_id():
    return str(uuid.uuid4())[:8]

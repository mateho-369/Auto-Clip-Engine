"""The AI team configuration: which local Ollama model does which job.

The team is deliberately small (5 roles). The user picks, per role, which
installed Ollama model handles it — including the CONTROLLER (planner)
that receives the user's idea and delegates tasks to the other roles.
Roles can be switched off; a deterministic template fallback always runs,
so the studio works even with Ollama completely offline.
"""
import json
import os

ROLES = ["planner", "scriptwriter", "sfx_director", "animator", "qa"]

ROLE_LABELS = {
    "planner": "Planner / Director (Controller)",
    "scriptwriter": "Scriptwriter",
    "sfx_director": "SFX Director",
    "animator": "Animator",
    "qa": "QA Reviewer",
}

ROLE_DESCRIPTIONS = {
    "planner": "Receives your idea, breaks the video into scenes, and delegates the work to the other AI roles.",
    "scriptwriter": "Writes and polishes the narration each scene speaks — the character's lines.",
    "sfx_director": "Chooses which sound effect plays in each scene and exactly when.",
    "animator": "Picks the character animation and scene transition for every scene.",
    "qa": "Reviews the finished plan for empty scripts, bad timing or repeated elements and fixes them.",
}


def default_config():
    return {
        "ollama_host": "http://localhost:11434",
        "controller": "llama3.2:3b",
        "roles": {
            "planner": {"enabled": True, "model": "llama3.2:3b", "temperature": 0.7},
            "scriptwriter": {"enabled": True, "model": "llama3.2:3b", "temperature": 0.8},
            "sfx_director": {"enabled": True, "model": "llama3.2:3b", "temperature": 0.4},
            "animator": {"enabled": True, "model": "llama3.2:3b", "temperature": 0.6},
            "qa": {"enabled": False, "model": "llama3.2:3b", "temperature": 0.2},
        },
    }


def _norm_role(role_cfg):
    role_cfg = dict(role_cfg or {})
    role_cfg["enabled"] = bool(role_cfg.get("enabled", False))
    model = str(role_cfg.get("model") or "").strip()
    role_cfg["model"] = model or "llama3.2:3b"
    try:
        temp = float(role_cfg.get("temperature", 0.7))
    except Exception:
        temp = 0.7
    role_cfg["temperature"] = max(0.0, min(2.0, temp))
    return role_cfg


def normalize_config(cfg):
    """Validates/repairs a team config coming from the UI."""
    base = default_config()
    cfg = cfg or {}
    out = {
        "ollama_host": str(cfg.get("ollama_host") or base["ollama_host"]).strip() or base["ollama_host"],
        "controller": str(cfg.get("controller") or base["controller"]).strip() or base["controller"],
        "roles": {},
    }
    for role in ROLES:
        out["roles"][role] = _norm_role(cfg.get("roles", {}).get(role))
    return out


def load_config(path):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return normalize_config(json.load(f))
        except Exception as e:
            print(f"Error loading team config: {e}")
    return default_config()


def save_config(path, cfg):
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error saving team config: {e}")
        return False

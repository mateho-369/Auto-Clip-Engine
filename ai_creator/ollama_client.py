"""Local Ollama client (stdlib-only) + robust JSON extraction for LLM output."""
import json
import urllib.request
import urllib.error


class OllamaClient:
    """Minimal Ollama REST client. All methods degrade to empty results
    instead of raising, so the studio can fall back to templates when
    Ollama is offline, uninstalled, or a model is missing."""

    def __init__(self, host="http://localhost:11434"):
        self.host = (host or "http://localhost:11434").rstrip("/")

    def _get(self, path, timeout=3):
        req = urllib.request.Request(self.host + path)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def _post(self, path, payload, timeout):
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.host + path, data=data, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def is_online(self):
        try:
            self._get("/api/tags", 2)
            return True
        except Exception:
            return False

    def list_models(self):
        try:
            d = self._get("/api/tags", 3)
            return [m.get("name", "") for m in d.get("models", []) if m.get("name")]
        except Exception:
            return []

    def chat(self, model, system, user, temperature=0.7, timeout=240):
        """Single-turn chat. Raises on failure — callers must catch and degrade."""
        payload = {
            "model": model,
            "stream": False,
            "options": {"temperature": temperature},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        d = self._post("/api/chat", payload, timeout)
        return (d.get("message") or {}).get("content", "").strip()


def extract_json(text):
    """Extract the first balanced JSON object/array from an LLM response.

    Handles markdown code fences, leading prose, and trailing chatter.
    Returns None when no valid JSON is found.
    """
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t.strip("`")
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    candidates = [(t.find("{"), "{", "}"), (t.find("["), "[", "]")]
    candidates = [c for c in candidates if c[0] != -1]
    candidates.sort(key=lambda c: c[0])  # whichever JSON structure starts first wins
    for i, opener, closer in candidates:
        depth = 0
        in_str = False
        esc = False
        for j in range(i, len(t)):
            c = t[j]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == opener:
                    depth += 1
                elif c == closer:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(t[i:j + 1])
                        except Exception:
                            return None
    try:
        return json.loads(t)
    except Exception:
        return None

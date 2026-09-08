"""Supporting image acquisition for Explain Mode scenes.

Priority sources:
  1. Reuse cached character action asset
  2. Web image search with STRICT Creative Commons / Public Domain license filtering
  3. AI image generation (local procedural synthesis or prompt rendering)
"""
import hashlib
import json
import os
import urllib.request
import urllib.parse
import numpy as np
import cv2


def fetch_supporting_image(source, query_or_prompt, cache_dir, char_id=None, character_store=None, pose="idle", width=640, height=480):
    """Fetches or generates a supporting image for a scene.

    Returns dict: {"url": str, "local_path": str, "source": str}
    """
    os.makedirs(cache_dir, exist_ok=True)
    source = (source or "web").lower().strip()
    query = (query_or_prompt or "illustration concept").strip()

    # Source 1: Action pose asset
    if source == "action" and character_store and char_id:
        action_path = character_store.ensure_action(char_id, pose)
        if action_path and os.path.exists(action_path):
            rel_url = f"/assets/characters/{char_id}/actions/{pose}.png"
            return {"url": rel_url, "local_path": action_path, "source": "action"}

    # Unique cache filename based on query & source
    hash_key = hashlib.md5(f"{source}:{query}:{width}x{height}".encode("utf-8")).hexdigest()[:12]
    filename = f"img_{hash_key}.png"
    local_path = os.path.join(cache_dir, filename)
    rel_url = f"/assets/cache/images/{filename}"

    if os.path.exists(local_path):
        return {"url": rel_url, "local_path": local_path, "source": source}

    # Source 2: Web image search with strict CC license filter
    if source in ("web", "action"):
        web_img = _search_creative_commons_images(query)
        if web_img is not None:
            cv2.imwrite(local_path, web_img)
            return {"url": rel_url, "local_path": local_path, "source": "web"}

    # Source 3: AI / Fallback image generation
    ai_img = _procedural_ai_illustration(query, width, height)
    cv2.imwrite(local_path, ai_img)
    return {"url": rel_url, "local_path": local_path, "source": "ai"}


def _search_creative_commons_images(query):
    """Searches open web sources ONLY with Creative Commons / Public Domain license filtering."""
    # 1. Wikimedia Commons API (100% CC / Public Domain)
    try:
        url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode({
            "action": "query",
            "generator": "search",
            "gsrsearch": f"file:{query}",
            "gsrlimit": "3",
            "prop": "imageinfo",
            "iiprop": "url",
            "format": "json"
        })
        req = urllib.request.Request(url, headers={"User-Agent": "AutoClipEngine/3.0 (CreativeCommons/1.0)"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            pages = data.get("query", {}).get("pages", {})
            for pid, pinfo in pages.items():
                infos = pinfo.get("imageinfo", [])
                if infos and "url" in infos[0]:
                    img_url = infos[0]["url"]
                    if img_url.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                        img_req = urllib.request.Request(img_url, headers={"User-Agent": "AutoClipEngine/3.0"})
                        with urllib.request.urlopen(img_req, timeout=4) as img_resp:
                            img_array = np.asarray(bytearray(img_resp.read()), dtype=np.uint8)
                            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                            if img is not None:
                                return img
    except Exception:
        pass

    # 2. Openverse / Creative Commons Search API (Filtered for Commercial/CC-BY/PublicDomain)
    try:
        url = "https://api.openverse.org/v1/images/?" + urllib.parse.urlencode({
            "q": query,
            "license_type": "commercial,modification",  # Strict CC filter
            "page_size": "2"
        })
        req = urllib.request.Request(url, headers={"User-Agent": "AutoClipEngine/3.0"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            results = data.get("results", [])
            for r in results:
                img_url = r.get("url")
                if img_url:
                    img_req = urllib.request.Request(img_url, headers={"User-Agent": "AutoClipEngine/3.0"})
                    with urllib.request.urlopen(img_req, timeout=4) as img_resp:
                        img_array = np.asarray(bytearray(img_resp.read()), dtype=np.uint8)
                        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                        if img is not None:
                            return img
    except Exception:
        pass

    return None


def _procedural_ai_illustration(query, w=640, h=480):
    """Generates a clean vector-style procedural card image when offline."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Hash query for deterministic background gradient
    hval = int(hashlib.md5(query.encode("utf-8")).hexdigest()[:6], 16)
    r1, g1, b1 = (hval & 0xFF), ((hval >> 8) & 0xFF), ((hval >> 16) & 0xFF)
    c1 = np.array([max(30, b1 // 2), max(30, g1 // 2), max(40, r1 // 2)], dtype=np.float32)
    c2 = np.array([min(220, b1 + 50), min(220, g1 + 50), min(240, r1 + 60)], dtype=np.float32)

    for y in range(h):
        p = y / max(1, h - 1)
        img[y, :] = (c1 * (1 - p) + c2 * p).astype(np.uint8)

    # Decorative shapes/cards
    cv2.rectangle(img, (20, 20), (w - 20, h - 20), (255, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(img, (w // 2, h // 2 - 20), int(min(w, h) * 0.28), (255, 255, 255), 3, cv2.LINE_AA)

    # Render topic query label
    text = query.upper()[:28]
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, 0.8, 2)
    tx = int((w - tw) / 2)
    ty = int(h // 2 + 10)
    cv2.putText(img, text, (tx + 2, ty + 2), font, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (tx, ty), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    return img

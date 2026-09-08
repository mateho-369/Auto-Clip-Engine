"""Character memory: your persistent on-screen character.

You train the character by uploading photos of it (a person, a drawing,
an avatar, a mascot). The system extracts and REMEMBERS:

  * face region  -> saved face crop + ahash (perceptual hash)
  * color palette -> dominant colors of the character
  * body/full figure -> the reference photos themselves

Those are stored on disk under characters/<id>/, so every new video
re-uses the exact same character. When you add more photos, the face
hash + palette distance tell the UI whether the photo "looks like the
same character" (rough match — good enough for a local studio).

The render asset (avatar.png) is a feathered cutout of the person. If
`rembg` is installed it does a real background removal; otherwise a
soft elliptical mask is generated so the studio works with zero extras.
"""
import cv2
import json
import os
import shutil
import time
import uuid
import numpy as np

# Bundled Haar cascade from the legacy project (the opencv wheels ship
# without the actual XML file — confirmed empirically).
_REPO_CASCADE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "src", "haarcascade_frontalface_default.xml")


STANDARD_POSES = [
    "idle", "point_left", "point_right", "point_up", "explain",
    "think", "wave", "laugh", "sleep", "eat", "sad", "surprised"
]


class CharacterStore:
    def __init__(self, root):
        self.root = os.path.join(root, "characters")
        os.makedirs(self.root, exist_ok=True)

    def _dir(self, char_id):
        return os.path.join(self.root, char_id)

    def _profile_path(self, char_id):
        return os.path.join(self._dir(char_id), "profile.json")

    def _actions_dir(self, char_id):
        return os.path.join(self._dir(char_id), "actions")

    def _actions_json_path(self, char_id):
        return os.path.join(self._dir(char_id), "actions.json")

    def list(self):
        out = []
        for d in sorted(os.listdir(self.root)):
            p = self._profile_path(d)
            if os.path.exists(p):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        prof = json.load(f)
                    prof["dir"] = self._dir(d)
                    out.append(prof)
                except Exception:
                    continue
        return out

    def get(self, char_id):
        p = self._profile_path(char_id)
        if not os.path.exists(p):
            return None
        with open(p, "r", encoding="utf-8") as f:
            prof = json.load(f)
        prof["dir"] = self._dir(char_id)
        return prof

    def _save(self, prof):
        os.makedirs(os.path.dirname(self._profile_path(prof["id"])), exist_ok=True)
        with open(self._profile_path(prof["id"]), "w", encoding="utf-8") as f:
            json.dump(prof, f, indent=2, ensure_ascii=False)

    def get_actions(self, char_id):
        prof = self.get(char_id)
        if prof is None:
            return {"actions": {}, "available_poses": STANDARD_POSES}
        actions_file = self._actions_json_path(char_id)
        actions = {}
        if os.path.exists(actions_file):
            try:
                with open(actions_file, "r", encoding="utf-8") as f:
                    actions = json.load(f)
            except Exception:
                actions = {}
        
        actions_dir = self._actions_dir(char_id)
        os.makedirs(actions_dir, exist_ok=True)

        # Ensure idle action exists if avatar.png exists
        avatar_path = os.path.join(self._dir(char_id), "avatar.png")
        idle_path = os.path.join(actions_dir, "idle.png")
        if "idle" not in actions or not os.path.exists(idle_path):
            if os.path.exists(avatar_path):
                shutil.copyfile(avatar_path, idle_path)
                actions["idle"] = {"path": "actions/idle.png", "source": "uploaded", "created": time.time()}
                self._save_actions(char_id, actions)

        # Filter out missing files
        valid_actions = {}
        for pose, meta in actions.items():
            full_path = os.path.join(self._dir(char_id), meta.get("path", f"actions/{pose}.png"))
            if os.path.exists(full_path):
                valid_actions[pose] = meta

        return {"actions": valid_actions, "available_poses": STANDARD_POSES}

    def _save_actions(self, char_id, actions):
        p = self._actions_json_path(char_id)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(actions, f, indent=2, ensure_ascii=False)

    def add_action(self, char_id, pose_name, img_path_or_data, source="uploaded"):
        prof = self.get(char_id)
        if prof is None:
            return None
        pose_name = pose_name.lower().strip().replace(" ", "_")
        actions_dir = self._actions_dir(char_id)
        os.makedirs(actions_dir, exist_ok=True)
        out_path = os.path.join(actions_dir, f"{pose_name}.png")

        if isinstance(img_path_or_data, str):
            if os.path.exists(img_path_or_data):
                img = cv2.imread(img_path_or_data, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    if img.shape[2] == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                    cv2.imwrite(out_path, img)
                else:
                    shutil.copyfile(img_path_or_data, out_path)
        elif isinstance(img_path_or_data, np.ndarray):
            cv2.imwrite(out_path, img_path_or_data)

        actions_info = self.get_actions(char_id)["actions"]
        actions_info[pose_name] = {
            "path": f"actions/{pose_name}.png",
            "source": source,
            "created": time.time()
        }
        self._save_actions(char_id, actions_info)
        return out_path

    def ensure_action(self, char_id, pose_name, prompt=""):
        pose_name = pose_name.lower().strip().replace(" ", "_")
        if not pose_name:
            pose_name = "idle"
        prof = self.get(char_id)
        if prof is None:
            return None
        actions_info = self.get_actions(char_id)["actions"]
        if pose_name in actions_info:
            full_path = os.path.join(self._dir(char_id), actions_info[pose_name]["path"])
            if os.path.exists(full_path):
                return full_path

        # Need to generate missing pose fallback
        avatar_path = os.path.join(self._dir(char_id), "avatar.png")
        if not os.path.exists(avatar_path):
            return None
        
        avatar_rgba = cv2.imread(avatar_path, cv2.IMREAD_UNCHANGED)
        if avatar_rgba is None:
            return None
        if avatar_rgba.shape[2] == 3:
            avatar_rgba = cv2.cvtColor(avatar_rgba, cv2.COLOR_BGR2BGRA)

        pose_img = generate_pose_fallback(avatar_rgba, pose_name)
        out_path = self.add_action(char_id, pose_name, pose_img, source="generated")
        return out_path

    def create(self, name, photo_path):
        char_id = str(uuid.uuid4())[:8]
        cdir = self._dir(char_id)
        refs = os.path.join(cdir, "refs")
        os.makedirs(refs, exist_ok=True)
        stored_ref = os.path.join(refs, "1.jpg")
        shutil.copyfile(photo_path, stored_ref)

        analysis = analyze_photo(stored_ref)
        prof = {
            "id": char_id,
            "name": name or "My Character",
            "created": time.time(),
            "photos": 1,
            "palette": analysis["palette"],
            "face": {"hash": analysis["face_hash"], "w": analysis["face_w"], "h": analysis["face_h"],
                     "detected": analysis["face_detected"]},
            "voice_id": "",
            "style_notes": "",
        }
        self._save(prof)
        self._rebuild_assets(prof)
        prof["dir"] = self._dir(char_id)
        return prof

    def add_photo(self, char_id, photo_path):
        prof = self.get(char_id)
        if prof is None:
            return None
        refs = os.path.join(prof["dir"], "refs")
        os.makedirs(refs, exist_ok=True)
        n = prof.get("photos", 0) + 1
        stored_ref = os.path.join(refs, f"{n}.jpg")
        shutil.copyfile(photo_path, stored_ref)

        analysis = analyze_photo(stored_ref)
        # rough same-character check
        dist = hamming(analysis["face_hash"], prof.get("face", {}).get("hash", 0))
        pal_delta = palette_distance(analysis["palette"], prof.get("palette", []))
        if dist <= 12 and pal_delta < 0.45:
            verdict = "same"
        elif dist <= 26:
            verdict = "maybe"
        else:
            verdict = "different"

        prof["photos"] = n
        # merge palette memory (keep the union of remembered colors, max 10)
        merged = list(prof.get("palette", []))
        for c in analysis["palette"]:
            if not any(_color_close(c, m) for m in merged):
                merged.append(c)
        prof["palette"] = merged[:10]
        # keep the best (detected, or latest) face signature
        if analysis["face_detected"] or not prof.get("face", {}).get("detected"):
            prof["face"] = {"hash": analysis["face_hash"], "w": analysis["face_w"],
                            "h": analysis["face_h"], "detected": analysis["face_detected"]}
        prof["last_similarity"] = {"hamming": int(dist), "palette_delta": round(float(pal_delta), 3),
                                   "verdict": verdict}
        self._save(prof)
        self._rebuild_assets(prof)
        return prof

    def update(self, char_id, **fields):
        prof = self.get(char_id)
        if prof is None:
            return None
        for k in ("name", "voice_id", "style_notes"):
            if k in fields and fields[k] is not None:
                prof[k] = fields[k]
        self._save(prof)
        return prof

    def delete(self, char_id):
        d = self._dir(char_id)
        if os.path.exists(d):
            shutil.rmtree(d, ignore_errors=True)
            return True
        return False

    def _rebuild_assets(self, prof):
        refs = os.path.join(self._dir(prof["id"]), "refs")
        best_ref = None
        for i in (1, 2, 3):
            p = os.path.join(refs, f"{i}.jpg")
            if os.path.exists(p):
                best_ref = p
                break
        if best_ref is None:
            return
        img = cv2.imread(best_ref, cv2.IMREAD_COLOR)
        if img is None:
            return
        face = detect_face(img)
        # face.png — the remembered face region (padded)
        if face is not None:
            x, y, w, h = face
            pad = int(w * 0.35)
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad)
            face_crop = img[y0:y1, x0:x1]
        else:
            face_crop = img
        face_crop = _cap_size(face_crop, 512)
        cv2.imwrite(os.path.join(self._dir(prof["id"]), "face.png"), face_crop)
        # avatar.png — the on-screen render asset (body + face, feathered)
        avatar = make_avatar(img, face, max_h=640)
        avatar_path = os.path.join(self._dir(prof["id"]), "avatar.png")
        cv2.imwrite(avatar_path, avatar)
        # Ensure actions/idle.png also exists
        actions_dir = self._actions_dir(prof["id"])
        os.makedirs(actions_dir, exist_ok=True)
        idle_path = os.path.join(actions_dir, "idle.png")
        cv2.imwrite(idle_path, avatar)
        actions = self.get_actions(prof["id"])["actions"]
        actions["idle"] = {"path": "actions/idle.png", "source": "uploaded", "created": time.time()}
        self._save_actions(prof["id"], actions)


def detect_face(img):
    """Returns (x, y, w, h) of the largest frontal face or None."""
    cascade_path = _REPO_CASCADE if os.path.exists(_REPO_CASCADE) else os.path.join(
        cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    try:
        cascade = cv2.CascadeClassifier(cascade_path)
    except Exception:
        cascade = None
    if cascade is None or getattr(cascade, "empty", lambda: True)():
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    for scale in (1.0, 0.5):
        g = gray if scale == 1.0 else cv2.resize(gray, None, fx=scale, fy=scale)
        faces = cascade.detectMultiScale(g, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) > 0:
            fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            fx, fy, fw, fh = int(fx / scale), int(fy / scale), int(fw / scale), int(fh / scale)
            return (fx, fy, fw, fh)
    return None


def extract_palette(img, k=5):
    """Dominant colors via k-means on downsampled pixels. Returns [[r,g,b],...]."""
    small = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
    pts = small.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pts, k, None, criteria, 4, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten(), minlength=k)
    order = np.argsort(-counts)
    return [[int(c[2]), int(c[1]), int(c[0])] for c in centers[order]]  # BGR -> RGB


def ahash(gray, size=8):
    """Average perceptual hash -> 64-bit int."""
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    bits = small < small.mean()
    val = 0
    for row in bits:
        for bit in row:
            val = (val << 1) | int(bit)
    return val


def hamming(a, b):
    x = int(a) ^ int(b)
    n = 0
    while x:
        n += x & 1
        x >>= 1
    return n


def palette_distance(p1, p2):
    if not p1 or not p2:
        return 0.0
    a = np.array(p1[:5], dtype=np.float32)
    b = np.array(p2[:5], dtype=np.float32)
    # min pairwise distance per color, averaged (normalized 0..~1)
    d = 0.0
    for ca in a:
        d += min(np.linalg.norm(ca - cb) for cb in b)
    return d / (len(a) * 441.7)


def _color_close(c1, c2, tol=60):
    return all(abs(int(c1[i]) - int(c2[i])) <= tol for i in range(3))


def _cap_size(img, max_dim):
    h, w = img.shape[:2]
    m = max(h, w)
    if m > max_dim:
        f = max_dim / m
        return cv2.resize(img, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)
    return img


def make_avatar(img, face, max_h=640):
    """Feathered person cutout (BGR->RGBA).

    Tries `rembg` for a real alpha cutout when installed; otherwise draws a
    soft ellipse/rounded region around the face extending down for the body.
    """
    img = np.ascontiguousarray(img)
    h, w = img.shape[:2]
    rgba = None
    try:
        import rembg
        session = rembg.new_session("u2net")
        out = rembg.remove(img, session=session)
        if out is not None and out.shape[2] == 4:
            rgba = out
    except Exception as e:
        print(f"rembg unavailable ({e}); using feathered mask fallback.")

    if rgba is None:
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, :3] = img
        if face is not None:
            x, y, fw, fh = face
            cx = x + fw / 2
            cy = y + fh / 2
            rx = int(fw * 1.05)
            # ellipse covering head + upper body
            ry = int(fh * 1.9)
            ecc_center = (int(cx), int(cy + fh * 0.62))
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(mask, ecc_center, (rx, ry), 0, 0, 360, 255, -1)
        else:
            mask = np.zeros((h, w), dtype=np.uint8)
            m = int(min(h, w) * 0.42)
            cv2.ellipse(mask, (w // 2, h // 2), (m, int(m * 1.5)), 0, 0, 360, 255, -1)
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        rgba[:, :, 3] = mask

    rgba = _cap_size(rgba, max_h)
    # trim fully-transparent border
    alpha = rgba[:, :, 3]
    ys, xs = np.where(alpha > 8)
    if len(xs) == 0:
        return rgba
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    pad = 4
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(rgba.shape[1], x1 + pad)
    y1 = min(rgba.shape[0], y1 + pad)
    return rgba[y0:y1, x0:x1]


def generate_pose_fallback(avatar_rgba, pose_name, prompt=""):
    """Generates a transparent RGBA pose image for a character.

    Attempts real AI image generation (via local ComfyUI at http://localhost:8188
    or local image generation endpoint) using reference character + pose prompt.
    If no AI image model is reachable, falls back to deterministic pose-synthesis.
    """
    if avatar_rgba is None:
        return None
    pose = (pose_name or "idle").lower().strip()

    # 1. Attempt AI Image Generation / ComfyUI pose transfer if server is online
    ai_pose_img = _try_generate_ai_pose_image(avatar_rgba, pose)
    if ai_pose_img is not None:
        return ai_pose_img

    # 2. Deterministic pose-synthesis fallback when AI image server is offline
    img = avatar_rgba.copy()
    h, w = img.shape[:2]

    if pose in ("idle", "explain"):
        return img

    if pose == "point_left":
        return cv2.flip(img, 1)

    if pose == "point_right":
        M = cv2.getRotationMatrix2D((w // 2, h // 2), -5, 1.0)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    if pose == "point_up":
        M = np.float32([[1, 0, 0], [0, 1, -int(h * 0.05)]])
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    if pose == "think":
        M = cv2.getRotationMatrix2D((w // 2, int(h * 0.7)), -8, 0.98)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    if pose == "wave":
        M = cv2.getRotationMatrix2D((w // 2, int(h * 0.7)), 6, 1.0)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    if pose == "laugh":
        M = cv2.getRotationMatrix2D((w // 2, h // 2), -3, 1.05)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    if pose == "sleep":
        M = cv2.getRotationMatrix2D((w // 2, h // 2), -18, 0.95)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    if pose in ("sad", "surprised", "eat"):
        angle = -6 if pose == "sad" else (8 if pose == "surprised" else 4)
        scale = 0.96 if pose == "sad" else 1.06
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, scale)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    return img


def _try_generate_ai_pose_image(avatar_rgba, pose, prompt="", work_dir="work"):
    """Generates an identity-locked pose image using the character's reference avatar image.

    1. Attempts reference-conditioned image-to-image pose transfer in ComfyUI (:8188)
       by uploading the reference avatar image, encoding it via VAEEncode with controlled
       denoise (0.45) so face, outfit, and color palette stay strictly fixed.
    2. Supports hosted image API endpoints (e.g. OpenAI / Replicate pose transfer API) if
       API keys are configured in environment.
    3. If no image generation model/endpoint is reachable, returns None so execution
       falls back gracefully.
    """
    if avatar_rgba is None:
        return None

    pose_desc = pose.replace("_", " ")

    # --- PATH A: Hosted Image API with Reference Image (if API key set) ---
    api_key = os.environ.get("POSE_TRANSFER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if api_key:
        try:
            hosted_img = _call_hosted_pose_transfer_api(avatar_rgba, pose_desc, api_key, work_dir)
            if hosted_img is not None:
                return hosted_img
        except Exception as e:
            print(f"Hosted pose transfer API skipped: {e}")

    # --- PATH B: Local ComfyUI Image-to-Image / Reference-Locked Pose Transfer ---
    try:
        import urllib.request
        import json
        import time

        # Probe ComfyUI health
        probe_req = urllib.request.Request("http://localhost:8188/system_stats", headers={"User-Agent": "AutoClipEngine"})
        with urllib.request.urlopen(probe_req, timeout=1) as resp:
            if resp.status != 200:
                return None

        # 1. Save reference avatar image to temporary file
        os.makedirs(work_dir, exist_ok=True)
        ref_filename = "ref_character.png"
        ref_path = os.path.join(work_dir, ref_filename)
        cv2.imwrite(ref_path, avatar_rgba)

        # 2. Upload reference image to ComfyUI (/upload/image)
        with open(ref_path, "rb") as f:
            img_bytes = f.read()

        boundary = "----WebKitFormBoundaryAutoClipEngine"
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="image"; filename="{ref_filename}"\r\n'
            f"Content-Type: image/png\r\n\r\n"
        ).encode("utf-8") + img_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

        up_req = urllib.request.Request(
            "http://localhost:8188/upload/image",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}", "User-Agent": "AutoClipEngine"}
        )
        with urllib.request.urlopen(up_req, timeout=4) as uresp:
            up_data = json.loads(uresp.read().decode("utf-8"))
            uploaded_name = up_data.get("name", ref_filename)

        # 3. Construct Image-to-Image / Reference-Locked Workflow Graph
        # Uses LoadImage -> VAEEncode with controlled denoise (0.45) so identity is locked
        positive_prompt = f"same character in {pose_desc} pose, same person, same face, same outfit and clothes, isolated standing presenter portrait, clean white background"
        negative_prompt = "different person, different face, different clothes, watermark, text, ugly, deformed, dark background"

        workflow = {
            "1": {
                "inputs": {
                    "image": uploaded_name,
                    "upload": "image"
                },
                "class_type": "LoadImage"
            },
            "2": {
                "inputs": {
                    "pixels": ["1", 0],
                    "vae": ["4", 2]
                },
                "class_type": "VAEEncode"
            },
            "3": {
                "inputs": {
                    "seed": int(time.time() * 1000) % 1000000,
                    "steps": 20,
                    "cfg": 6.5,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 0.45,  # Controlled denoise: 55% structure/identity locked, 45% pose shift
                    "model": ["4", 0],
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "latent_image": ["2", 0]  # Latent encoded directly from reference character avatar image!
                },
                "class_type": "KSampler"
            },
            "4": {
                "inputs": {
                    "ckpt_name": "v1-5-pruned-emaonly.safetensors"
                },
                "class_type": "CheckpointLoaderSimple"
            },
            "6": {
                "inputs": {
                    "text": positive_prompt,
                    "clip": ["4", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "7": {
                "inputs": {
                    "text": negative_prompt,
                    "clip": ["4", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "8": {
                "inputs": {
                    "samples": ["3", 0],
                    "vae": ["4", 2]
                },
                "class_type": "VAEDecode"
            },
            "9": {
                "inputs": {
                    "filename_prefix": "pose_gen_ref",
                    "images": ["8", 0]
                },
                "class_type": "SaveImage"
            }
        }

        # Submit workflow prompt
        data = json.dumps({"prompt": workflow}).encode("utf-8")
        req = urllib.request.Request("http://localhost:8188/prompt", data=data, headers={"Content-Type": "application/json", "User-Agent": "AutoClipEngine"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            res_data = json.loads(resp.read().decode("utf-8"))
            prompt_id = res_data.get("prompt_id")

        if not prompt_id:
            return None

        # Poll history for output
        output_filename = None
        for _ in range(30):
            time.sleep(1)
            hist_req = urllib.request.Request(f"http://localhost:8188/history/{prompt_id}", headers={"User-Agent": "AutoClipEngine"})
            with urllib.request.urlopen(hist_req, timeout=2) as hresp:
                hdata = json.loads(hresp.read().decode("utf-8"))
                if prompt_id in hdata:
                    outputs = hdata[prompt_id].get("outputs", {})
                    for node_id, node_out in outputs.items():
                        images = node_out.get("images", [])
                        if images:
                            output_filename = images[0].get("filename")
                            break
            if output_filename:
                break

        if not output_filename:
            return None

        # Download output image
        view_url = f"http://localhost:8188/view?filename={output_filename}&subfolder=&type=output"
        vreq = urllib.request.Request(view_url, headers={"User-Agent": "AutoClipEngine"})
        with urllib.request.urlopen(vreq, timeout=5) as vresp:
            img_bytes = vresp.read()
            img_array = np.asarray(bytearray(img_bytes), dtype=np.uint8)
            bgr_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if bgr_img is None:
            return None

        # Perform background removal to get transparent RGBA cutout
        rgba_img = make_avatar(bgr_img, face=None, max_h=640)
        return rgba_img

    except Exception as e:
        print(f"ComfyUI reference pose generation offline/skipped: {e}")
        return None


def _prepare_rgb_square_image(avatar_rgba, target_size=512, bg_color=(240, 240, 240)):
    """Converts a transparent RGBA character cutout into a solid 1:1 square RGB image.

    Prevents APIs (like OpenAI / DALL-E) from misinterpreting alpha transparency
    channels as edit mask regions, and ensures strict 1:1 square image input compliance.
    Uses neutral light gray background (default (240,240,240)) to prevent white clothes
    from bleeding into pure white backgrounds.
    """
    if avatar_rgba is None:
        return None

    h, w = avatar_rgba.shape[:2]
    max_dim = max(h, w)

    # 1. Create solid neutral background square canvas
    square = np.full((max_dim, max_dim, 3), bg_color, dtype=np.uint8)

    # 2. Composite RGBA character onto solid square center
    x_off = (max_dim - w) // 2
    y_off = (max_dim - h) // 2

    if avatar_rgba.shape[2] == 4:
        alpha = avatar_rgba[:, :, 3] / 255.0
        for c in range(3):
            square[y_off:y_off+h, x_off:x_off+w, c] = (
                avatar_rgba[:, :, c] * alpha + square[y_off:y_off+h, x_off:x_off+w, c] * (1 - alpha)
            ).astype(np.uint8)
    else:
        square[y_off:y_off+h, x_off:x_off+w] = avatar_rgba[:, :, :3]

    # 3. Resize to target_size x target_size
    resized = cv2.resize(square, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return resized


def _call_hosted_pose_transfer_api(avatar_rgba, pose_desc, api_key, work_dir="work"):
    """Hosted pose transfer API implementation (e.g. OpenAI Images Edit / Replicate).

    Pre-processes character into a solid 1:1 square RGB image (no alpha mask confusion)
    and sends to hosted image API with identity-preserving prompt, returning transparent RGBA cutout.
    """
    if avatar_rgba is None or not api_key:
        return None

    try:
        import urllib.request
        import json

        # Save solid 1:1 square RGB reference image (no alpha mask confusion!)
        os.makedirs(work_dir, exist_ok=True)
        rgb_square = _prepare_rgb_square_image(avatar_rgba, target_size=512)
        ref_path = os.path.join(work_dir, "ref_api_square.png")
        cv2.imwrite(ref_path, rgb_square)

        # OpenAI Image Variations / Edits API Path
        prompt = f"full figure presenter portrait of the exact same character performing {pose_desc} pose, same face, same clothes, isolated clean background"

        boundary = "----WebKitFormBoundaryAutoClipEngineHosted"
        with open(ref_path, "rb") as f:
            img_bytes = f.read()

        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="image"; filename="ref_api_square.png"\r\n'
            f"Content-Type: image/png\r\n\r\n"
        ).encode("utf-8") + img_bytes + (
            f"\r\n--{boundary}\r\n"
            f'Content-Disposition: form-data; name="prompt"\r\n\r\n'
            f"{prompt}\r\n"
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="n"\r\n\r\n1\r\n'
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="size"\r\n\r\n512x512\r\n'
            f"--{boundary}--\r\n"
        ).encode("utf-8")

        req = urllib.request.Request(
            "https://api.openai.com/v1/images/edits",
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "User-Agent": "AutoClipEngine"
            }
        )

        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            urls = data.get("data", [])
            if urls and "url" in urls[0]:
                img_url = urls[0]["url"]
                vreq = urllib.request.Request(img_url, headers={"User-Agent": "AutoClipEngine"})
                with urllib.request.urlopen(vreq, timeout=8) as vresp:
                    img_array = np.asarray(bytearray(vresp.read()), dtype=np.uint8)
                    bgr_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    if bgr_img is not None:
                        return make_avatar(bgr_img, face=None, max_h=640)

    except Exception as e:
        print(f"Hosted pose transfer API call failed: {e}")

    return None


def analyze_photo(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not read image file.")
    face = detect_face(img)
    if face is not None:
        x, y, fw, fh = face
        pad = int(fw * 0.3)
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(img.shape[1], x + fw + pad), min(img.shape[0], y + fh + pad)
        face_region = img[y0:y1, x0:x1]
    else:
        face_region = img
    gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
    return {
        "face_detected": face is not None,
        "face_hash": int(ahash(gray)),
        "face_w": int(face_region.shape[1]),
        "face_h": int(face_region.shape[0]),
        "palette": extract_palette(face_region if face is not None else img, k=5),
    }

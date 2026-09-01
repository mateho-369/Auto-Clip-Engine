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


class CharacterStore:
    def __init__(self, root):
        self.root = os.path.join(root, "characters")
        os.makedirs(self.root, exist_ok=True)

    def _dir(self, char_id):
        return os.path.join(self.root, char_id)

    def _profile_path(self, char_id):
        return os.path.join(self._dir(char_id), "profile.json")

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
        cv2.imwrite(os.path.join(self._dir(prof["id"]), "avatar.png"), avatar)


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

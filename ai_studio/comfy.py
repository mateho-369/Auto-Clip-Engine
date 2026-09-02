"""ComfyUI client: queue a workflow, watch progress, pull the file back.

Used by Stage 4 (Wan text/image-to-video) and Stage 5 (MMAudio video-to-audio).
Design notes for a 8GB card:

* only **one** GPU job can be in flight — the scheduler serialises submissions
  behind the GPU lock, and we ask ComfyUI to free VRAM after each job;
* progress comes from ComfyUI's WebSocket (`/ws`) — implemented here with a tiny
  stdlib RFC6455 client so the studio adds *no* new dependency and the UI can
  show "video scene 3 · 62%" instead of a spinner. If the WS can't be opened we
  transparently fall back to polling `/history` + `/queue`;
* outputs are fetched with `/view?filename=&subfolder=&type=output` and copied
  into the project folder, so a run's assets are browsable on disk too.
"""
import base64
import json
import mimetypes
import os
import socket
import ssl
import struct
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

from .util import ensure_dir


class ComfyError(RuntimeError):
    pass


class ComfyUIClient:
    def __init__(self, host="http://127.0.0.1:8188", timeout=8, client_id=None):
        self.host = (host or "http://127.0.0.1:8188").rstrip("/")
        self.timeout = timeout
        self.client_id = client_id or uuid.uuid4().hex[:12]

    # ------------------------------------------------------------- http bits
    def _req(self, method, path, payload=None, timeout=None, ctype="application/json"):
        url = self.host + path
        data = None
        headers = {}
        if payload is not None:
            if isinstance(payload, (bytes, bytearray)):
                data = bytes(payload)
                headers["Content-Type"] = ctype
            else:
                data = json.dumps(payload).encode("utf-8")
                headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, method=method, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
            raw = r.read()
        if path.startswith("/view") or "/download" in path:
            return raw
        try:
            return json.loads(raw.decode("utf-8")) if raw else {}
        except Exception:
            return {"raw": raw.decode("utf-8", errors="ignore")}

    def is_online(self):
        try:
            self._req("GET", "/system_stats", timeout=3)
            return True
        except Exception:
            return False

    def system_stats(self):
        try:
            return self._req("GET", "/system_stats", timeout=6) or {}
        except Exception:
            return {}

    def free_vram_mb(self):
        """Free VRAM as ComfyUI sees it (authoritative when ComfyUI owns the card)."""
        stats = self.system_stats()
        best = None
        for dev in (stats.get("devices") or []):
            total = dev.get("vram_total") or 0
            free = dev.get("vram_free") or 0
            if total:
                mb = int(free / (1024 * 1024))
                best = mb if best is None else min(best, mb)
        return best

    def object_info(self, class_type=None):
        path = "/object_info" + (f"/{class_type}" if class_type else "")
        try:
            return self._req("GET", path, timeout=10) or {}
        except Exception:
            return {}

    def has_node(self, class_type):
        return bool(self.object_info(class_type).get(class_type))

    def free_memory(self, unload_models=True):
        try:
            self._req("POST", "/free", {"unload_models": unload_models, "free_memory": True}, timeout=6)
            return True
        except Exception:
            return False

    def interrupt(self, prompt_id=None):
        try:
            self._req("POST", "/interrupt", ({"prompt_id": prompt_id} if prompt_id else {}), timeout=6)
            return True
        except Exception:
            return False

    def upload_image(self, path, subfolder="ai_studio"):
        """POST /upload/image (used by Wan2.2 TI2V start frames)."""
        with open(path, "rb") as f:
            blob = f.read()
        boundary = "----aiStudio" + uuid.uuid4().hex
        fn = os.path.basename(path)
        body = b"".join([
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; "
            f"filename=\"{fn}\"\r\nContent-Type: {mimetypes.guess_type(fn)[0] or 'image/png'}"
            f"\r\n\r\n".encode(),
            blob,
            f"\r\n--{boundary}--\r\n".encode(),
        ])
        q = urllib.parse.urlencode({"type": "input", "subfolder": subfolder, "overwrite": "true"})
        res = self._req("POST", f"/upload/image?{q}", body, timeout=60,
                        ctype=f"multipart/form-data; boundary={boundary}")
        if isinstance(res, dict) and res.get("name"):
            return {"name": res["name"], "subfolder": res.get("subfolder", subfolder),
                    "type": res.get("type", "input")}
        raise ComfyError(f"upload rejected: {str(res)[:200]}")

    # --------------------------------------------------------------- job flow
    def queue_prompt(self, workflow, front=None):
        payload = {"prompt": workflow, "client_id": self.client_id}
        if front:
            payload["front"] = front
        try:
            res = self._req("POST", "/prompt", payload, timeout=25)
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", errors="ignore")[:1200]
            except Exception:
                pass
            raise ComfyError(f"ComfyUI rejected the workflow ({e.code}): {detail or e.reason}") from None
        except Exception as e:
            raise ComfyError(f"could not submit to ComfyUI at {self.host}: {e}") from None
        pid = (res or {}).get("prompt_id")
        if not pid:
            raise ComfyError(f"ComfyUI returned no prompt_id: {str(res)[:300]}")
        node_errors = (res or {}).get("node_errors") or {}
        if node_errors and any(node_errors.values()):
            raise ComfyError("workflow node errors: " + json.dumps(node_errors)[:900])
        return pid

    def history(self, prompt_id):
        try:
            return (self._req("GET", f"/history/{urllib.parse.quote(prompt_id)}", timeout=15) or {}).get(prompt_id)
        except Exception:
            return None

    def queue_position(self, prompt_id):
        try:
            q = self._req("GET", "/queue", timeout=8) or {}
        except Exception:
            return None
        for i, item in enumerate(q.get("queue_running") or []):
            if len(item) > 1 and item[1] == prompt_id:
                return 0
        for i, item in enumerate(q.get("queue_pending") or []):
            if len(item) > 1 and item[1] == prompt_id:
                return i + 1
        return 0 if self.history(prompt_id) else None

    def wait(self, prompt_id, timeout=1800, on_progress=None, poll=1.5, cancel=None):
        """Block (in a thread) until the job finishes. Returns history outputs.

        Progress: prefers the WS feed, falls back to node-count polling so the
        UI percentage is real either way.
        """
        deadline = time.time() + timeout
        state = {"pct": 0.0, "node": "", "step": 0, "max": 0}
        last_reported = [None]
        stop = threading.Event()
        ws_thread = None
        try:
            ws_thread = threading.Thread(target=self._ws_progress, args=(prompt_id, state, stop),
                                         daemon=True)
            ws_thread.start()
        except Exception:
            ws_thread = None
        seen_done = False
        try:
            while time.time() < deadline:
                if cancel is not None and cancel.is_set():
                    self.interrupt(prompt_id)
                    raise ComfyError("cancelled by user")
                hist = self.history(prompt_id)
                if hist:
                    seen_done = True
                    outputs = (hist or {}).get("outputs") or {}
                    status = ((hist or {}).get("status") or {})
                    if status.get("status_str") == "error" or status.get("completed") is False:
                        errs = status.get("messages") or []
                        txt = "; ".join(str(e) for e in errs)[:600] or "node execution error"
                        raise ComfyError(f"ComfyUI job failed: {txt}")
                    self._report(on_progress, state, last_reported)
                    return outputs
                pos = self.queue_position(prompt_id)
                if pos is not None and pos > 0:
                    state["node"] = f"queued · position {pos}"
                self._report(on_progress, state, last_reported)
                time.sleep(poll)
            self.interrupt(prompt_id)
            raise ComfyError(f"ComfyUI job timed out after {int(timeout)}s")
        finally:
            stop.set()
            if ws_thread:
                ws_thread.join(timeout=2)
            if not seen_done:
                self.free_memory()

    @staticmethod
    def _report(on_progress, state, last_reported=None):
        if on_progress is None:
            return
        pct = min(99.5, max(1.0, state["pct"]))
        node = state.get("node") or ""
        if last_reported is not None:
            key = (round(pct, 1), node)
            if last_reported[0] == key:
                return  # unchanged since the last tick — e.g. still queued at
                        # the same position — don't spam the run's live log
            last_reported[0] = key
        try:
            on_progress(pct, node)
        except Exception:
            pass

    # ------------------------------------------------------ websocket progress
    def _ws_progress(self, prompt_id, state, stop):
        sock = None
        try:
            u = urllib.parse.urlparse(self.host)
            host, port = u.hostname, u.port or (443 if u.scheme == "https" else 80)
            sock = socket.create_connection((host, port), timeout=8)
            if u.scheme == "https":
                sock = ssl.create_default_context().wrap_socket(sock, server_hostname=host)
            key = base64.b64encode(os.urandom(16)).decode()
            path = "/ws?clientId=" + urllib.parse.quote(self.client_id)
            req = (f"GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nUpgrade: websocket\r\n"
                   f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n")
            sock.sendall(req.encode())
            buf = b""
            while b"\r\n\r\n" not in buf:
                chunk = sock.recv(4096)
                if not chunk:
                    return
                buf += chunk
            if b"101" not in buf.split(b"\r\n", 1)[0]:
                return
            rbuf = buf.split(b"\r\n\r\n", 1)[1]
            sock.settimeout(4)
            while not stop.is_set():
                msg, rbuf = _ws_read_frame(sock, rbuf)
                if msg is None:
                    if stop.is_set():
                        return
                    continue
                try:
                    data = json.loads(msg)
                except Exception:
                    continue
                t = data.get("type")
                d = data.get("data") or {}
                if d.get("prompt_id") and d.get("prompt_id") != prompt_id and t != "status":
                    continue
                if t == "progress":
                    mx = float(d.get("max") or 0)
                    v = float(d.get("value") or 0)
                    state["max"], state["step"] = mx, v
                    state["pct"] = 100.0 * v / mx if mx else state["pct"]
                    node = str(d.get("node") or "")
                    state["node"] = f"{node} {int(v)}/{int(mx)}" if node else state["node"]
                elif t == "execution_update":
                    if d.get("node"):
                        state["node"] = str(d.get("node"))
                elif t == "executing" and d.get("node") is None:
                    state["pct"] = 99.0
                elif t == "execution_error":
                    state["node"] = "error"
        except Exception:
            return
        finally:
            try:
                if sock:
                    sock.close()
            except Exception:
                pass

    # ------------------------------------------------------------- artefacts
    @staticmethod
    def output_refs(outputs, exts=(".mp4", ".webm", ".mov", ".wav", ".mp3", ".png", ".jpg")):
        refs = []
        for node_out in (outputs or {}).values():
            for key in ("gifs", "videos", "images", "audio"):
                for item in (node_out or {}).get(key, []) or []:
                    fn = item.get("filename", "")
                    if not fn or not fn.lower().endswith(exts):
                        continue
                    refs.append({"filename": fn, "subfolder": item.get("subfolder", ""),
                                 "type": item.get("type", "output"), "kind": key})
        return refs

    def download(self, ref, dest_path):
        q = urllib.parse.urlencode({"filename": ref["filename"], "subfolder": ref.get("subfolder", ""),
                                    "type": ref.get("type", "output")})
        raw = self._req("GET", f"/view?{q}", timeout=180)
        if isinstance(raw, str):
            raw = raw.encode("utf-8", errors="ignore")
        ensure_dir(os.path.dirname(dest_path) or ".")
        with open(dest_path, "wb") as f:
            f.write(raw)
        return dest_path


# ------------------------------------------------------- minimal RFC6455 read
def _ws_read_frame(sock, buf, mask_out=False):
    """Return (text_or_None, leftover_bytes). Handles ping/pong/close."""
    while True:
        while len(buf) < 2:
            chunk = sock.recv(4096)
            if not chunk:
                return None, buf
            buf += chunk
        b0, b1 = buf[0], buf[1]
        opcode = b0 & 0x0F
        length = b1 & 0x7F
        ext = 2
        if length == 126:
            while len(buf) < 4:
                buf += sock.recv(4096)
            length = struct.unpack(">H", buf[2:4])[0]
            ext = 4
        elif length == 127:
            while len(buf) < 10:
                buf += sock.recv(4096)
            length = struct.unpack(">Q", buf[2:10])[0]
            ext = 10
        masked = bool(b1 & 0x80)
        need = ext + (4 if masked else 0) + length
        while len(buf) < need:
            chunk = sock.recv(max(4096, need - len(buf)))
            if not chunk:
                return None, buf
            buf += chunk
        payload = buf[ext:ext + length]
        buf = buf[ext + length:]
        if masked:
            key = buf[:0]  # not used: servers don't mask
        if opcode == 0x9:            # ping -> pong
            try:
                sock.sendall(_ws_frame(0xA, payload))
            except Exception:
                return None, buf
            continue
        if opcode == 0x8:            # close
            return None, buf
        if opcode in (0x1, 0x2, 0x0):
            try:
                return payload.decode("utf-8", errors="ignore"), buf
            except Exception:
                return None, buf
        # control/other frames: ignore and keep reading


def _ws_frame(opcode, payload):
    header = bytes([0x80 | opcode])
    n = len(payload)
    mask = os.urandom(4)
    if n < 126:
        header += bytes([0x80 | n])
    elif n < 65536:
        header += bytes([0x80 | 126]) + struct.pack(">H", n)
    else:
        header += bytes([0x80 | 127]) + struct.pack(">Q", n)
    return header + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(payload))


def wait_for_job(client, prompt_id, timeout, on_progress=None, cancel=None):
    """Thread helper so the scheduler can `asyncio.to_thread` the blocking wait."""
    return client.wait(prompt_id, timeout=timeout, on_progress=on_progress, cancel=cancel)

// Thin typed client for the studio API. Central error surface: every failed
// call raises ApiError carrying the backend's EXACT human-readable message
// (detail / error / message / validation array), and `ToastHub` shows it.
export class ApiError extends Error {
  status: number;
  data: any;
  constructor(msg: string, status: number, data?: any) {
    super(msg);
    this.status = status;
    this.data = data;
  }
}

export function extractDetail(data: any, status: number, fallback?: string): string {
  if (!data) return fallback || `HTTP ${status}`;
  if (typeof data === "string" && data) return data.slice(0, 400);
  if (data.detail) {
    const d = data.detail;
    if (typeof d === "string") return d;
    if (Array.isArray(d)) {
      return d.map((x: any) => x.msg || x.loc?.join(".") || JSON.stringify(x)).join("; ").slice(0, 600);
    }
    return JSON.stringify(d);
  }
  if (data.error) return String(data.error).slice(0, 400);
  if (data.message) return String(data.message).slice(0, 400);
  if (Array.isArray(data)) return data.map((x: any) => x.msg || JSON.stringify(x)).join("; ");
  return JSON.stringify(data).slice(0, 400);
}

export async function api<T = any>(path: string, opts: {
  method?: string; json?: any; form?: FormData; query?: Record<string, any>;
} = {}): Promise<T> {
  const init: RequestInit = { method: opts.method || "GET", headers: {} };
  let url = "/api" + path;
  if (opts.query) {
    const qs = new URLSearchParams();
    for (const [k, v] of Object.entries(opts.query)) {
      if (v !== undefined && v !== null && v !== "") qs.set(k, String(v));
    }
    const s = qs.toString();
    if (s) url += (url.includes("?") ? "&" : "?") + s;
  }
  if (opts.json !== undefined) {
    (init.headers as any)["Content-Type"] = "application/json";
    init.body = JSON.stringify(opts.json);
  }
  if (opts.form) init.body = opts.form;
  let r: Response;
  try {
    r = await fetch(url, init);
  } catch (e: any) {
    throw new ApiError(`network error reaching the studio: ${e?.message || e}`, 0);
  }
  const ct = r.headers.get("content-type") || "";
  let data: any = null;
  if (ct.includes("json")) {
    try { data = await r.json(); } catch { data = null; }
  } else {
    data = await r.text();
  }
  if (!r.ok) {
    throw new ApiError(extractDetail(data, r.status), r.status, data);
  }
  return data as T;
}

export const downloadUrl = (path: string) => path.startsWith("/") ? path : "/api/" + path;

// ---------------------------------------------------------------- types
export interface ProjectRow {
  id: string; title: string; mode: string; status: string; language: string;
  target_duration: number; script_origin: string; voice_profile_id: string;
  content_type: string; character_id: string; parent_id: string; last_run_id: string;
  created_at: number; updated_at: number; script_excerpt?: string; script_chars?: number;
  scene_count: number; run_count: number; last_run_status?: string;
}

export interface Scene {
  idx: number; text: string; visual_prompt: string; mood_tag: string;
  estimated_duration_sec: number; audio_duration: number; sfx_prompt: string;
  meta: Record<string, any>;
}

export interface Project {
  id: string; title: string; mode: string; status: string; language: string;
  script: string; script_locked: boolean; script_origin: string; topic_hint: string;
  style_notes: string; target_duration: number; voice_profile_id: string;
  content_type: string; character_id: string; settings: Record<string, any>;
  parent_id: string; last_run_id?: string; created_at: number; updated_at: number;
  scenes?: Scene[];
}

export interface StageRow {
  id: string; run_id: string; stage: string; scene_idx: number; status: string;
  attempt: number; progress: number; message: string; error: string; engine: string;
  inherited_from: string; started_at?: number; finished_at?: number; duration_ms: number;
  key: string;
}

export interface Run {
  id: string; project_id: string; status: string; trigger: string; resume_from: string;
  machine_profile: string; gpu_policy: Record<string, any>; stage_filter: string[];
  error: string; stats: Record<string, any>; stages?: StageRow[]; assets?: Asset[];
  started_at?: number; finished_at?: number; created_at: number;
}

export interface Asset {
  id: string; project_id: string; run_id: string; stage: string; scene_idx: number;
  kind: string; path: string; relpath: string; mime: string; size_bytes: number;
  duration: number; meta: Record<string, any>; created_at: number;
}

export interface StageSpec {
  key: string; title: string; emoji: string; role: string; blurb: string; model: string;
  per_scene: boolean; requires_gpu: boolean; resource: string; depends: string[];
  deferrable: boolean;
}

export interface StatusPayload {
  studio: string; version: string; data_dir: string; db: Record<string, any>;
  machine: any; plan: Record<string, any>; capabilities: Record<string, any>;
  vram: any; active_runs: string[]; ffmpeg: string;
}

export interface ContentTypeMeta {
  key: string; label: string; one_liner: string; emoji: string;
  default_visual_source: string; default_render_mode: string; target_duration_factor: number;
}

export interface CharacterImage {
  id: string; character_id: string; expression_label: string; image_path: string;
  created_at: number; exists?: boolean; url?: string;
}

export interface Character {
  id: string; name: string; notes: string; created_at: number; images: CharacterImage[];
}

export interface StylePreview {
  key: string; label: string; url: string; desc?: string; error?: string; font_size?: number;
}

export interface VoiceProfile {
  id: string; name: string; pth_path: string; index_path: string; sample_path: string;
  sample_seconds: number; engine: string; pitch: number; index_rate: number; rms_mix_rate: number;
  f0_method: string; notes: string; training_status: string; created_at: number;
  pth_exists?: boolean; index_exists?: boolean; sample_url?: string;
}

export const EMPTY_PROJECT = {} as Project;

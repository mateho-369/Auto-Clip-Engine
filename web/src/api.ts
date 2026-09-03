export interface Toast {
  id: number;
  kind: 'info' | 'ok' | 'err';
  text: string;
}

export const api = {
  async request<T = any>(path: string, init: RequestInit = {}): Promise<T> {
    const res = await fetch(path, {
      headers: { 'Content-Type': 'application/json', ...(init.headers || {}) },
      ...init,
    });
    const text = await res.text();
    let data: any = null;
    try {
      data = text ? JSON.parse(text) : null;
    } catch {
      data = text;
    }
    if (!res.ok) {
      const detail =
        (data && (data.detail || data.error)) ||
        `HTTP ${res.status} ${res.statusText}`;
      throw new Error(detail);
    }
    return data as T;
  },
  get: <T = any>(path: string) => api.request<T>(path),
  post: <T = any>(path: string, body?: any) =>
    api.request<T>(path, { method: 'POST', body: JSON.stringify(body ?? {}) }),
  patch: <T = any>(path: string, body?: any) =>
    api.request<T>(path, { method: 'PATCH', body: JSON.stringify(body ?? {}) }),
  del: <T = any>(path: string, qs = '') =>
    api.request<T>(path + (qs ? `?${qs}` : ''), { method: 'DELETE' }),
};

export function wsUrl(path: string): string {
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${proto}//${window.location.host}${path}`;
}

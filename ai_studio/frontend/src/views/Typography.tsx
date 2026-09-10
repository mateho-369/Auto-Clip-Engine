import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  api, ApiError, CaptionBootstrap, CaptionFont, CaptionPreset, CaptionRenderResult,
  CaptionStyle, ProjectCaptions, captionPreview, fontFaceCss,
} from "../api";
import { useToast, errText } from "../main";
import { Badge, Panel, Spinner } from "../ui";

/**
 * Typography & Captions inspector.
 *
 * Two storage layers, kept deliberately separate in the UI *and* on disk:
 *   • Caption style  → what is burned into the exported MP4 (this panel).
 *   • App theme      → the studio's own interface (never touched by these
 *     controls; a caption colour cannot recolour the app).
 *
 * Style precedence, shown in the header: built-in default → studio settings →
 * this project. "Save for this project" writes projects.settings_json.
 * caption_style; "Save as studio default" writes settings.json. Older projects
 * have no override and simply inherit — nothing is discarded.
 *
 * The preview is rendered by the backend with the *exporter's own* renderer
 * (ffmpeg/libass + the bundled font), so the picture here is the picture that
 * ends up in the MP4 — not a CSS mock-up.
 */

const SAMPLE = "ជីវិតមនុស្ស មិនមែនជាការប្រណាំងទេ។\nយើងម្នាក់ៗ មានផ្លូវដើររៀងៗខ្លួន។";
const SAMPLE_SHORT = "យើងម្នាក់ៗ មានផ្លូវដើររៀងៗខ្លួន។";

const CANVASES: { key: string; label: string; w: number; h: number }[] = [
  { key: "9:16", label: "9:16 · 1080×1920", w: 1080, h: 1920 },
  { key: "16:9", label: "16:9 · 1920×1080", w: 1920, h: 1080 },
  { key: "9:16-720", label: "9:16 · 720×1280", w: 720, h: 1280 },
];

const KH = {
  title: "អក្សរ និងចំណងជើង",
  font: "ពុម្ពអក្សរ",
  size: "ទំហំអក្សរ",
  color: "ពណ៌អក្សរ",
  outline: "ពណ៌/កម្រាស់ស៊ុម",
  shadow: "ស្រមោល",
  background: "ផ្ទៃខាងក្រោយ",
  position: "ទីតាំង",
  spacing: "គម្លាតបន្ទាត់",
  preview: "មើលជាមុន",
};
const FOCUS_RING = { outline: "2px solid var(--acc)", outlineOffset: 1 } as React.CSSProperties;

type Scope = "project" | "global";

export function TypographyInspector({ projectId, onChanged }: {
  projectId: string;
  onChanged?: () => void;
}) {
  const toast = useToast();
  const [meta, setMeta] = useState<ProjectCaptions | null>(null);
  const [draft, setDraft] = useState<CaptionStyle | null>(null);
  const [scope, setScope] = useState<Scope>("project");
  const [canvas, setCanvas] = useState(CANVASES[0]);
  const [sample, setSample] = useState(SAMPLE_SHORT);
  const [burn, setBurn] = useState(false);
  const [emitSrt, setEmitSrt] = useState(true);
  const [karaoke, setKaraoke] = useState<{ enabled: boolean; color: string }>({ enabled: false, color: "#FFD84D" });
  const [saving, setSaving] = useState("");
  const [savedAt, setSavedAt] = useState<number | null>(null);
  const [dirty, setDirty] = useState(false);
  const [err, setErr] = useState("");
  const [previewUrl, setPreviewUrl] = useState("");
  const [previewInfo, setPreviewInfo] = useState<any>(null);
  const [previewBusy, setPreviewBusy] = useState(false);
  const [renderBusy, setRenderBusy] = useState(false);
  const [result, setResult] = useState<CaptionRenderResult | null>(null);
  const [thumbnails, setThumbnails] = useState<Record<string, string>>({});
  const seq = useRef(0);
  const toastRef = useRef(toast);
  toastRef.current = toast;

  const load = useCallback(async () => {
    try {
      const d = await api<ProjectCaptions>(`/projects/${projectId}/caption-style`);
      setMeta(d);
      setDraft(d.effective_style);
      setKaraoke(d.karaoke?.enabled !== undefined
        ? { enabled: !!d.karaoke.enabled, color: d.karaoke.color || "#FFD84D" }
        : { enabled: false, color: "#FFD84D" });
      setErr("");
      const asm = await api<CaptionBootstrap>("/caption-style");
      setBurn(!!asm.assembly?.burn_captions);
      setEmitSrt(asm.assembly?.emit_srt !== false);
      // real previews per preset × font, rendered by the exporter
      const pv = await api<{ items: { key: string; url: string }[] }>("/caption-style/previews");
      const map: Record<string, string> = {};
      (pv.items || []).forEach((i) => { if (i.url) map[i.key] = i.url; });
      setThumbnails(map);
    } catch (e) { setErr(errText(e)); }
  }, [projectId]);

  useEffect(() => { load(); }, [load]);

  // ---------------------------------------------------------------- preview
  const requestPreview = useCallback(async (style: CaptionStyle, text: string) => {
    const my = ++seq.current;
    setPreviewBusy(true);
    try {
      const r = await captionPreview({
        text, style, project_id: projectId, width: canvas.w, height: canvas.h,
        karaoke: karaoke.enabled ? karaoke : undefined,
      });
      if (my !== seq.current) { URL.revokeObjectURL(r.url); return; }   // stale → drop
      setPreviewUrl((old) => { if (old) URL.revokeObjectURL(old); return r.url; });
      setPreviewInfo({ ...(r.font || {}), warnings: r.warnings || [], lines: r.lines });
    } catch (e) {
      if (my === seq.current) {
        setPreviewInfo({ error: errText(e) });
      }
    } finally {
      if (my === seq.current) setPreviewBusy(false);
    }
  }, [projectId, canvas, karaoke]);

  useEffect(() => {
    if (!draft) return;
    const t = setTimeout(() => { requestPreview(draft, sample); }, 380);   // debounce
    return () => clearTimeout(t);
  }, [draft, sample, requestPreview]);

  const set = <K extends keyof CaptionStyle>(k: K, v: CaptionStyle[K]) => {
    setDraft((d) => (d ? { ...d, [k]: v, preset: k === "preset" ? (v as string) : "custom" } : d));
    setDirty(true);
  };
  const setNumber = (k: keyof CaptionStyle) => (e: React.ChangeEvent<HTMLInputElement>) => {
    const v = e.target.value === "" ? 0 : Number(e.target.value);
    if (Number.isFinite(v)) set(k, v as any);
  };

  const applyPreset = (p: CaptionPreset) => {
    setDraft({ ...p.style, preset: p.key });
    setDirty(true);
  };

  const save = async (target: Scope) => {
    if (!draft) return;
    setSaving(target);
    try {
      if (target === "project") {
        await api(`/projects/${projectId}/caption-style`, {
          method: "PUT",
          json: { style: draft, burn_captions: burn },
        });
        toast("ស្ទីលបានរក្សាទុកសម្រាប់គម្រោងនេះ · saved for this project", "ok");
      } else {
        await api("/caption-style", {
          method: "POST",
          json: { style: draft, burn_captions: burn, emit_srt: emitSrt, karaoke },
        });
        toast("បានរក្សាទុកជាលំនាំដើម · saved as studio default", "ok");
      }
      setDirty(false);
      setSavedAt(Date.now());
      await load();
      onChanged?.();
    } catch (e) {
      const msg = e instanceof ApiError ? e.message : errText(e);
      setErr(msg);
      toast(msg, "err");
    } finally { setSaving(""); }
  };

  const clearOverride = async () => {
    setSaving("clear");
    try {
      const d = await api<ProjectCaptions>(`/projects/${projectId}/caption-style`, {
        method: "PUT", json: { clear: true },
      });
      setMeta(d); setDraft(d.effective_style); setDirty(false);
      toast("project override cleared — back to the studio default", "ok");
      onChanged?.();
    } catch (e) { toast(errText(e), "err"); } finally { setSaving(""); }
  };

  const resetToPreset = () => {
    if (!meta) return;
    const p = meta.presets.find((x) => x.key === (draft?.preset !== "custom" ? draft?.preset : "clean"))
      || meta.presets[0];
    if (p) applyPreset(p);
  };

  const renderCaptions = async () => {
    setRenderBusy(true); setResult(null);
    try {
      const r = await api<CaptionRenderResult>(`/projects/${projectId}/render-captions`, {
        method: "POST",
        json: { style: draft, karaoke: karaoke.enabled ? karaoke : { enabled: false } },
      });
      setResult(r);
      toast(`បានដុតអក្សរទៅវីដេអូ · captions re-rendered (${r.cues} cues)`, "ok");
      onChanged?.();
    } catch (e) { toast(errText(e), "err"); } finally { setRenderBusy(false); }
  };

  const downloadSrt = () => {
    const a = document.createElement("a");
    a.href = `/api/projects/${projectId}/download?kind=srt`;
    a.download = "";
    a.click();
  };

  const family: CaptionFont | undefined = useMemo(
    () => meta?.fonts.find((f) => f.id === draft?.font), [meta, draft?.font]);

  const presetChanged = useMemo(() => {
    if (!meta || !draft) return [];
    const base = meta.presets.find((p) => p.key === draft.preset);
    if (!base) return ["custom"];
    return (Object.keys(draft) as (keyof CaptionStyle)[])
      .filter((k) => k !== "preset" && JSON.stringify((draft as any)[k]) !== JSON.stringify((base.style as any)[k]));
  }, [meta, draft]);

  const effPx = draft ? Math.max(1, Math.round((draft.font_size_px * canvas.h) / 1920)) : 0;

  if (!meta || !draft) {
    return (
      <Panel title={`${KH.title} · Typography & Captions`}>
        <div className="panel-b">{err ? <div className="errbar">⚠ {err}</div> : <Spinner />}</div>
      </Panel>
    );
  }

  const caps = meta.capabilities;

  return (
    <Panel
      title={`${KH.title} · Typography & Captions`}
      right={
        <div className="row" style={{ gap: 4 }}>
          {dirty && <Badge kind="warn">● unsaved</Badge>}
          {savedAt && !dirty && <Badge kind="ok">saved</Badge>}
          {meta.is_default ? <Badge>studio default</Badge> : <Badge kind="blue">project override</Badge>}
        </div>
      }
    >
      <div className="panel-b typography">
        {err && <div className="errbar" role="alert">⚠ {err}</div>}
        {!caps.ok && (
          <div className="errbar" role="alert">
            ⚠ Captions cannot be rendered correctly on this machine:
            <ul>{caps.problems.map((p, i) => <li key={i}>{p}</li>)}</ul>
          </div>
        )}

        {/* ---------------- live preview ---------------- */}
        <div className="cap-grid">
          <div className="cap-preview">
            <div className="row" style={{ justifyContent: "space-between" }}>
              <span className="hint">{KH.preview} · {canvas.label}</span>
              <div className="row" style={{ gap: 4 }}>
                {CANVASES.map((c) => (
                  <button key={c.key} className={`btn tiny ${c.key === canvas.key ? "primary" : ""}`}
                    onClick={() => setCanvas(c)}>{c.key}</button>
                ))}
              </div>
            </div>
            <div className="cap-stage" style={{ aspectRatio: `${canvas.w} / ${canvas.h}` }}>
              {previewUrl && <img src={previewUrl} alt="caption preview" />}
              {previewBusy && <span className="cap-busy"><Spinner /></span>}
            </div>
            <div className="hint">
              {previewInfo?.error
                ? <span style={{ color: "var(--red)" }}>⚠ {previewInfo.error}</span>
                : <>
                  {previewInfo?.family ? <>ពុម្ពអក្សរ <b>{previewInfo.family}</b> {previewInfo.weight} · {previewInfo.file}</> : "rendered by the exporter"}
                  {" · "}{effPx}px on {canvas.w}×{canvas.h}
                </>}
            </div>
            {(previewInfo?.warnings || []).map((w: string, i: number) => (
              <div key={i} className="hint warnline">⚠ {w}</div>
            ))}
            <label className="fld" style={{ marginTop: 8 }}>
              <span>test text · អត្ថបទសាកល្បង</span>
              <textarea rows={2} value={sample} onChange={(e) => setSample(e.target.value)} />
            </label>
          </div>

          {/* ---------------- controls ---------------- */}
          <div className="cap-controls">
            <fieldset>
              <legend>Preset</legend>
              <div className="preset-row">
                {meta.presets.map((p) => (
                  <button key={p.key}
                    className={`preset ${draft.preset === p.key ? "on" : ""}`}
                    title={p.desc}
                    onClick={() => applyPreset(p)}>
                    <img src={thumbnails[`${p.key}:${draft.font}`] || ""} alt=""
                      onError={(e) => { (e.target as HTMLImageElement).style.visibility = "hidden"; }} />
                    <span>{p.label}</span>
                  </button>
                ))}
              </div>
              <div className="row" style={{ marginTop: 6 }}>
                <button className="btn tiny" onClick={resetToPreset}>↺ reset to preset</button>
                <span className="hint">{meta.presets.find((p) => p.key === draft.preset)?.desc || "custom style"}</span>
              </div>
              {presetChanged.length > 0 && presetChanged[0] !== "custom" && (
                <div className="hint warnline">custom: {presetChanged.join(", ")}</div>
              )}
            </fieldset>

            <fieldset>
              <legend>{KH.font} · font</legend>
              <select value={draft.font} onChange={(e) => {
                const f = meta.fonts.find((x) => x.id === e.target.value)!;
                setDraft((d) => d && ({
                  ...d, font: f.id, preset: "custom",
                  weight: f.weights.includes(d.weight) ? d.weight : f.weights[0],
                }));
                setDirty(true);
              }}>
                {meta.fonts.map((f) => (
                  <option key={f.id} value={f.id}>{f.label} — {f.blurb.slice(0, 46)}</option>
                ))}
              </select>
              <div className="khmer-sample" style={{ fontFamily: `'${family?.family}', sans-serif` }}>
                {family?.sample || SAMPLE_SHORT}
              </div>
              <div className="row" style={{ marginTop: 6 }}>
                <span className="hint">weight</span>
                {(family?.weights || [400]).map((w) => (
                  <button key={w} className={`btn tiny ${draft.weight === w ? "primary" : ""}`}
                    onClick={() => set("weight", w)}>{w}{w === 400 ? " · regular" : w === 700 ? " · bold" : w === 300 ? " · light" : ""}</button>
                ))}
                <span className="hint">{(family as any)?.license || "OFL-1.1"} licensed, bundled</span>
              </div>
            </fieldset>

            <fieldset>
              <legend>{KH.size} · size &amp; colour</legend>
              <label className="fld">
                <span>font size — {draft.font_size_px}px on 1080×1920 (= {effPx}px here)</span>
                <input type="range" min={24} max={160} step={1} value={draft.font_size_px}
                  onChange={setNumber("font_size_px")} />
              </label>
              <div className="row tight">
                <ColorField label={KH.color} value={draft.color} onChange={(v) => set("color", v)} />
                <ColorField label={KH.outline} value={draft.outline_color}
                  onChange={(v) => set("outline_color", v)} />
              </div>
              <label className="fld">
                <span>outline width — {draft.outline_width}px
                  (bold outlines can fill Khmer counters: check the preview)</span>
                <input type="range" min={0} max={10} step={0.5} value={draft.outline_width}
                  onChange={setNumber("outline_width")} />
              </label>
            </fieldset>

            <fieldset>
              <legend>{KH.shadow} · shadow</legend>
              <div className="row">
                <label className="switch">
                  <input type="checkbox" checked={draft.shadow > 0}
                    onChange={(e) => set("shadow", e.target.checked ? 1.6 : 0)} />
                  <span>on</span>
                </label>
                <ColorField label="colour" value={draft.shadow_color}
                  onChange={(v) => set("shadow_color", v)} />
              </div>
              <label className="fld">
                <span>strength — {draft.shadow}px</span>
                <input type="range" min={0} max={8} step={0.2} value={draft.shadow}
                  disabled={draft.shadow <= 0} onChange={setNumber("shadow")} />
              </label>
              <label className="fld">
                <span>opacity — {Math.round(draft.shadow_opacity * 100)}%</span>
                <input type="range" min={0} max={1} step={0.05} value={draft.shadow_opacity}
                  disabled={draft.shadow <= 0} onChange={setNumber("shadow_opacity")} />
              </label>
            </fieldset>

            <fieldset>
              <legend>{KH.background} · background panel</legend>
              <div className="row">
                <label className="switch">
                  <input type="checkbox" checked={draft.background}
                    onChange={(e) => set("background", e.target.checked)} />
                  <span>panel</span>
                </label>
                <ColorField label="colour" value={draft.background_color}
                  onChange={(v) => set("background_color", v)} />
              </div>
              <label className="fld">
                <span>opacity — {Math.round(draft.background_opacity * 100)}%</span>
                <input type="range" min={0} max={1} step={0.05} value={draft.background_opacity}
                  disabled={!draft.background} onChange={setNumber("background_opacity")} />
              </label>
              <div className="row tight">
                <label className="fld grow">
                  <span>padding — {draft.background_padding}px</span>
                  <input type="range" min={0} max={90} step={1} value={draft.background_padding}
                    disabled={!draft.background} onChange={setNumber("background_padding")} />
                </label>
                <label className="fld grow">
                  <span>radius — {draft.background_radius}px</span>
                  <input type="range" min={0} max={60} step={1} value={draft.background_radius}
                    disabled={!draft.background} onChange={setNumber("background_radius")} />
                </label>
              </div>
            </fieldset>

            <fieldset>
              <legend>{KH.position} · layout</legend>
              <div className="row">
                <span className="hint">vertical</span>
                {(["bottom", "center", "top"] as const).map((p) => (
                  <button key={p} className={`btn tiny ${draft.position === p ? "primary" : ""}`}
                    onClick={() => set("position", p)}>
                    {p === "bottom" ? "⬇ bottom" : p === "center" ? "⏺ center" : "⬆ top"}
                  </button>
                ))}
              </div>
              <div className="row" style={{ marginTop: 6 }}>
                <span className="hint">horizontal</span>
                {(["left", "center", "right"] as const).map((a) => (
                  <button key={a} className={`btn tiny ${draft.alignment === a ? "primary" : ""}`}
                    onClick={() => set("alignment", a)}>{a}</button>
                ))}
              </div>
              <div className="row tight">
                <label className="fld grow">
                  <span>safe margin H — {draft.margin_h}%</span>
                  <input type="range" min={0} max={25} step={0.5} value={draft.margin_h}
                    onChange={setNumber("margin_h")} />
                </label>
                <label className="fld grow">
                  <span>safe margin V — {draft.margin_v}%</span>
                  <input type="range" min={0} max={35} step={0.5} value={draft.margin_v}
                    onChange={setNumber("margin_v")} />
                </label>
              </div>
              <div className="row tight">
                <label className="fld grow">
                  <span>{KH.spacing} — {draft.line_spacing}×</span>
                  <input type="range" min={0.9} max={2.2} step={0.05} value={draft.line_spacing}
                    onChange={setNumber("line_spacing")} />
                </label>
                <label className="fld grow">
                  <span>max lines — {draft.max_lines}</span>
                  <input type="range" min={1} max={5} step={1} value={draft.max_lines}
                    onChange={setNumber("max_lines")} />
                </label>
              </div>
              <label className="fld">
                <span>max caption width — {draft.max_width_pct}% of the frame</span>
                <input type="range" min={40} max={100} step={1} value={draft.max_width_pct}
                  onChange={setNumber("max_width_pct")} />
              </label>
            </fieldset>

            <fieldset>
              <legend>Karaoke &amp; export</legend>
              <div className="row">
                <label className="switch">
                  <input type="checkbox" checked={karaoke.enabled}
                    onChange={(e) => { setKaraoke({ ...karaoke, enabled: e.target.checked }); setDirty(true); }} />
                  <span>word highlight</span>
                </label>
                <ColorField label="highlight" value={karaoke.color}
                  onChange={(v) => { setKaraoke({ ...karaoke, color: v }); setDirty(true); }} />
              </div>
              <div className="hint">
                Karaoke timing is a proportional estimate from each scene's audio window,
                not forced alignment. The line's own text is never split — only the
                highlight moves.
              </div>
              <div className="row" style={{ marginTop: 6 }}>
                <label className="switch">
                  <input type="checkbox" checked={burn} onChange={(e) => { setBurn(e.target.checked); setDirty(true); }} />
                  <span>burn captions into the exported video</span>
                </label>
              </div>
              <div className="row" style={{ marginTop: 6 }}>
                <label className="switch">
                  <input type="checkbox" checked={emitSrt} onChange={(e) => { setEmitSrt(e.target.checked); setDirty(true); }} />
                  <span>also write an .srt sidecar</span>
                </label>
                <button className="btn tiny" onClick={downloadSrt}
                  title="download the subtitle file from the last run">⬇ SRT</button>
                <a className="btn tiny" href={`/api/projects/${projectId}/download?kind=final`}>⬇ final video</a>
              </div>
              {!burn && (
                <div className="hint warnline">
                  Burn-in is off: the export will have no captions (SRT only). Captions are never
                  added or dropped silently.
                </div>
              )}
            </fieldset>

            <div className="save-bar">
              <div className="row">
                <button className="btn primary" disabled={!!saving} onClick={() => save("project")}>
                  {saving === "project" ? <Spinner /> : "💾"} save for this project
                </button>
                <button className="btn" disabled={!!saving} onClick={() => save("global")}>
                  {saving === "global" ? <Spinner /> : "🌐"} save as studio default
                </button>
                {!meta.is_default && (
                  <button className="btn" disabled={!!saving} onClick={clearOverride}>clear override</button>
                )}
              </div>
              <div className="hint">
                precedence: {meta.precedence.join(" → ")}. This project currently uses{" "}
                {meta.is_default ? "the studio default" : "its own override"}.
              </div>
            </div>

            <fieldset>
              <legend>Render captions on the finished cut</legend>
              <div className="hint">
                Applies the style to the already-rendered video — no pipeline re-run. The
                uncaptioned master is burned again from scratch, so overlays never stack.
              </div>
              <div className="row" style={{ marginTop: 6 }}>
                <button className="btn primary" disabled={renderBusy} onClick={renderCaptions}>
                  {renderBusy ? <Spinner /> : "🎬"} render captions now
                </button>
              </div>
              {result && (
                <div className="cap-result">
                  <video controls preload="metadata" src={result.url} />
                  <div className="row" style={{ marginTop: 6 }}>
                    <a className="btn tiny primary" href={result.download}>⬇ captioned MP4</a>
                    <a className="btn tiny" href={result.url} target="_blank" rel="noreferrer">open</a>
                    <Badge kind="ok">{result.font.family} {result.font.weight}</Badge>
                    <Badge>{result.cues} cues</Badge>
                  </div>
                  <div className="hint">re-rendered from {result.source} · {result.karaoke?.timing}</div>
                  {(result.warnings || []).map((w, i) => <div key={i} className="hint warnline">⚠ {w}</div>)}
                  {(result.bounds_warnings || []).map((w, i) => (
                    <div key={i} className="hint warnline">⚠ {w}</div>
                  ))}
                </div>
              )}
            </fieldset>

            <div className="hint">
              App interface vs video: this panel only affects <b>exported captions</b>. The
              studio's own theme (colours, UI font) is stored separately and is not changed here.
            </div>
          </div>
        </div>
      </div>
    </Panel>
  );
}

function ColorField({ label, value, onChange }: {
  label: string; value: string; onChange: (v: string) => void;
}) {
  const [text, setText] = useState(value);
  const [bad, setBad] = useState(false);
  useEffect(() => { setText(value); setBad(false); }, [value]);
  const commit = (v: string) => {
    const ok = /^#?([0-9a-fA-F]{6})$/.test(v.trim());
    if (!ok) { setBad(true); return; }
    setBad(false);
    onChange(v.trim().startsWith("#") ? v.trim().toUpperCase() : "#" + v.trim().toUpperCase());
  };
  return (
    <label className="fld colorfield">
      <span>{label}</span>
      <div className="row" style={{ gap: 4 }}>
        <input type="color" value={value} aria-label={label}
          onChange={(e) => onChange(e.target.value.toUpperCase())} style={FOCUS_RING} />
        <input className="hex" value={text} spellCheck={false} aria-label={`${label} hex`}
          style={bad ? { borderColor: "var(--red)" } : undefined}
          onChange={(e) => setText(e.target.value)} onBlur={() => commit(text)}
          onKeyDown={(e) => { if (e.key === "Enter") commit(text); }} />
      </div>
      {bad && <span className="hint" style={{ color: "var(--red)" }}>use #RRGGBB</span>}
    </label>
  );
}

/** Inject the bundled Khmer @font-face rules once (UI samples + Khmer labels). */
export function useStudioFonts(fonts?: CaptionFont[]) {
  useEffect(() => {
    if (!fonts?.length) return;
    const id = "studio-khmer-fonts";
    let el = document.getElementById(id) as HTMLStyleElement | null;
    if (!el) {
      el = document.createElement("style");
      el.id = id;
      document.head.appendChild(el);
    }
    el.textContent = fontFaceCss(fonts);
  }, [fonts]);
}

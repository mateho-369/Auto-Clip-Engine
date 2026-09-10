import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, CaptionFont, CaptionSchema } from "../api";
import { useToast, errText } from "../main";

// The Typography & Captions inspector. Everything here drives the REAL
// exporter: the preview image comes from the backend's ASS+libass renderer
// (the exact burn path of the final MP4), and Save persists the validated
// style on the project (settings.captions, deep-merged server-side).
//
// Scope guard: these controls affect EXPORTED VIDEO CAPTIONS only — they do
// not change the app's UI theme.

const SAMPLE_DEFAULT = "យើងម្នាក់ៗ មានផ្លូវដើររៀងៗខ្លួន។";

interface Style {
  [k: string]: any;
}

export function CaptionStudio({ projectId, initial, onChanged }: {
  projectId: string;
  initial: Style | null;          // effective style from GET /projects/{id}
  onChanged: () => void;
}) {
  const toast = useToast();
  const [schema, setSchema] = useState<CaptionSchema | null>(null);
  const [style, setStyle] = useState<Style>(initial || {});
  const [savedStyle, setSavedStyle] = useState<Style>(initial || {});
  const [sample, setSample] = useState(SAMPLE_DEFAULT);
  const [bg, setBg] = useState("videoish");
  const [fmt, setFmt] = useState<"9:16" | "16:9">("9:16");
  const [previewUrl, setPreviewUrl] = useState("");
  const [previewBusy, setPreviewBusy] = useState(false);
  const [issues, setIssues] = useState<string[]>([]);
  const [dirty, setDirty] = useState(false);
  const [saving, setSaving] = useState(false);
  const [open, setOpen] = useState(true);
  const seq = useRef(0);

  useEffect(() => {
    api<CaptionSchema>("/captions/schema", { query: { project_id: projectId } })
      .then((s) => { setSchema(s); })
      .catch((e) => toast(errText(e), "err"));
  }, [projectId]);

  useEffect(() => { setStyle(initial || {}); setSavedStyle(initial || {}); setDirty(false); },
    [initial]);

  // debounced + stale-guarded preview: only the newest request may paint
  const refreshPreview = useCallback(async (st: Style) => {
    const my = ++seq.current;
    setPreviewBusy(true);
    try {
      const landscape = fmt === "16:9";
      const r = await api<{ url: string; issues: string[] }>("/captions/preview", {
        method: "POST",
        json: { project_id: projectId, style: st, text: sample,
                width: landscape ? 854 : 480, height: landscape ? 480 : 854,
                background: bg },
      });
      if (seq.current !== my) return;              // a newer request superseded us
      setPreviewUrl(r.url);
      setIssues(r.issues || []);
    } catch (e) {
      if (seq.current === my) { setIssues([errText(e)]); }
    } finally {
      if (seq.current === my) setPreviewBusy(false);
    }
  }, [projectId, sample, bg, fmt]);

  useEffect(() => {
    const t = setTimeout(() => { refreshPreview(style); }, 350);
    return () => clearTimeout(t);
  }, [style, sample, bg, fmt, refreshPreview]);

  const set = (k: string, v: any) => {
    setStyle((s) => ({ ...s, [k]: v }));
    setDirty(true);
  };
  const setPanel = (k: string, v: any) => {
    setStyle((s) => ({ ...s, panel: { ...(s.panel || {}), [k]: v } }));
    setDirty(true);
  };

  const applyPreset = (id: string) => {
    // preset semantics: selecting one re-derives the whole look server-side
    // (validated); any field the user then changes is a visible modification.
    setStyle((s) => ({ ...s, preset: id }));
    setDirty(true);
  };

  const save = async () => {
    setSaving(true);
    try {
      await api(`/projects/${projectId}`, {
        method: "PATCH",
        json: { settings: { captions: style } },
      });
      setSavedStyle(style);
      setDirty(false);
      toast("caption style saved — applies to the next render", "ok");
      onChanged();
    } catch (e) {
      toast(errText(e), "err");
    } finally {
      setSaving(false);
    }
  };

  const resetPreset = () => {
    if (!schema) return;
    const p = style.preset || "clean";
    const def = schema.defaults;
    setStyle({ ...def, preset: p, karaoke: style.karaoke || false });
    setDirty(true);
  };

  if (!schema) return <div className="panel-b hint">loading typography…</div>;

  const eff: Style = style;
  const modified: string[] = [];
  const base = { ...schema.defaults, ...(schema.presets.find((p) => p.id === (eff.preset || "clean")) ? {} : {}) };
  for (const k of Object.keys(eff)) {
    if (k === "version" || k === "preset") continue;
    if (JSON.stringify(eff[k]) !== JSON.stringify((base as any)[k]) && schema.defaults[k] !== undefined) {
      // vs preset-of-record would be nicer; schema only carries defaults — the
      // server's `modified` list is authoritative for the badge below.
    }
  }
  const isModified = JSON.stringify(style) !== JSON.stringify(savedStyle) || dirty;
  const presetModifiedHint = (eff.preset && eff.preset !== "custom")
    ? "custom tweaks to this preset are kept" : "";

  const fonts = schema.fonts;
  const curFont = fonts.find((f) => f.id === eff.font);
  const weights = curFont?.weights || ["regular"];
  const landscape = fmt === "16:9";

  return (
    <div className="panel cap-studio">
      <div className="panel-h" onClick={() => setOpen((o) => !o)} style={{ cursor: "pointer" }}>
        <h3>🔤 Typography &amp; Captions <span className="hint" style={{ fontWeight: 400 }}>· burned into the exported MP4</span></h3>
        {isModified && <span className="badge warn">unsaved</span>}
        {!isModified && eff.preset && eff.preset !== "custom" && <span className="badge">{eff.preset}</span>}
        <span className="spacer" />
        <button className="btn tiny" onClick={(e) => { e.stopPropagation(); setOpen((o) => !o); }}>
          {open ? "▾" : "▸"}
        </button>
      </div>
      {open && (
        <div className="panel-b cap-grid">
          {/* ---------- preview column ---------- */}
          <div className="cap-preview-col">
            <div className="row" style={{ marginBottom: 8 }}>
              <div className="tabs mini-tabs">
                <button className={fmt === "9:16" ? "on" : ""} onClick={() => setFmt("9:16")}>9:16</button>
                <button className={fmt === "16:9" ? "on" : ""} onClick={() => setFmt("16:9")}>16:9</button>
              </div>
              <select className="inline-sel" value={bg} onChange={(e) => setBg(e.target.value)} title="preview background">
                <option value="videoish">bg: video-like</option>
                <option value="dark">bg: dark</option>
                <option value="light">bg: light</option>
                <option value="busy">bg: busy</option>
              </select>
              {previewBusy && <span className="spin" />}
            </div>
            <div className={`cap-frame ${landscape ? "land" : ""}`}>
              {previewUrl && <img src={previewUrl} alt="caption preview" className="cap-preview-img" />}
            </div>
            <textarea className="cap-sample" rows={2} value={sample}
              onChange={(e) => setSample(e.target.value)}
              placeholder="preview text (Khmer)…" />
            {issues.length > 0 && (
              <div className="hint cap-warn">
                {issues.slice(0, 3).map((w, i) => <div key={i}>⚠ {w}</div>)}
              </div>
            )}
            <div className="hint">
              preview = the exporter's real libass render (same font file, same layout).
              {schema.timing_note}
            </div>
          </div>

          {/* ---------- controls column ---------- */}
          <div className="cap-controls-col">
            <label className="fld"><span>Preset</span>
              <div className="preset-row">
                {schema.presets.map((p) => (
                  <button key={p.id}
                    className={`preset-chip ${((eff.preset || "clean") === p.id) ? "on" : ""}`}
                    onClick={() => applyPreset(p.id)}>{p.id.replace("_", " ")}</button>
                ))}
                <button className="btn tiny" onClick={resetPreset} title="reset every field to the selected preset">reset</button>
              </div>
              <span className="hint">{presetModifiedHint}</span>
            </label>

            <div className="cap-two">
              <label className="fld"><span>Font family</span>
                <select value={eff.font || "noto_sans_khmer"}
                  onChange={(e) => {
                    const f = fonts.find((x) => x.id === e.target.value);
                    const w = f && !f.weights.includes(eff.weight) ? "regular" : eff.weight;
                    set("font", e.target.value); if (w !== eff.weight) set("weight", w);
                  }}>
                  {fonts.map((f) => <option key={f.id} value={f.id}
                    disabled={!f.available}>{f.label}{f.available ? "" : " (missing files)"}</option>)}
                </select>
                <span className="hint">{curFont?.note}</span>
              </label>
              <label className="fld"><span>Weight</span>
                <select value={eff.weight || "regular"} onChange={(e) => set("weight", e.target.value)}>
                  {weights.map((w) => <option key={w} value={w}>{w}</option>)}
                </select>
              </label>
            </div>

            <label className="fld"><span>Font size · <b>{Number(eff.size_pct).toFixed(1)}%</b> of output height ≈ {Math.round(eff.size_pct / 100 * 854)} px at 854p</span>
              <input type="range" min={schema.ranges.size_pct[0]} max={schema.ranges.size_pct[1]}
                step={0.1} value={eff.size_pct} onChange={(e) => set("size_pct", Number(e.target.value))} />
            </label>

            <div className="cap-two">
              <ColorField label="Text color" value={eff.text_color} onChange={(v) => set("text_color", v)} />
              <ColorField label="Outline color" value={eff.outline_color} onChange={(v) => set("outline_color", v)} />
            </div>

            <label className="fld"><span>Outline thickness · {Number(eff.outline_px).toFixed(1)} px</span>
              <input type="range" min={0} max={schema.ranges.outline_px[1]} step={0.5}
                value={eff.outline_px} onChange={(e) => set("outline_px", Number(e.target.value))} />
            </label>

            <div className="cap-two">
              <label className="fld"><span>Shadow</span>
                <select value={eff.shadow ? "on" : "off"} onChange={(e) => set("shadow", e.target.value === "on")}>
                  <option value="off">off</option><option value="on">on</option>
                </select>
              </label>
              <label className="fld"><span>Shadow strength · {Math.round(Number(eff.shadow_strength) * 100)}%</span>
                <input type="range" min={0} max={1} step={0.05} value={eff.shadow_strength}
                  disabled={!eff.shadow}
                  onChange={(e) => set("shadow_strength", Number(e.target.value))} />
              </label>
            </div>

            <div className="cap-panel-box">
              <div className="row" style={{ marginBottom: 6 }}>
                <b style={{ fontSize: 12 }}>Background panel</b>
                <select className="inline-sel" value={eff.panel?.enabled ? "on" : "off"}
                  onChange={(e) => setPanel("enabled", e.target.value === "on")}>
                  <option value="off">off</option><option value="on">on</option>
                </select>
                <span className="hint">libass draws a rectangle (rounded corners appear in web mockups only)</span>
              </div>
              <div className="cap-two">
                <ColorField label="Panel color" value={eff.panel?.color || "#0e1116"}
                  onChange={(v) => setPanel("color", v)} />
                <label className="fld"><span>Opacity · {Math.round(Number(eff.panel?.opacity ?? 0.62) * 100)}%</span>
                  <input type="range" min={0} max={1} step={0.05} value={eff.panel?.opacity ?? 0.62}
                    disabled={!eff.panel?.enabled}
                    onChange={(e) => setPanel("opacity", Number(e.target.value))} />
                </label>
              </div>
              <div className="cap-two">
                <label className="fld"><span>Padding · {eff.panel?.padding_px ?? 12} px</span>
                  <input type="range" min={0} max={28} step={1} value={eff.panel?.padding_px ?? 12}
                    disabled={!eff.panel?.enabled}
                    onChange={(e) => setPanel("padding_px", Number(e.target.value))} />
                </label>
                <label className="fld"><span>Corner radius · {eff.panel?.radius_px ?? 10} px <i className="hint">(web preview only)</i></span>
                  <input type="range" min={0} max={28} step={1} value={eff.panel?.radius_px ?? 10}
                    disabled={!eff.panel?.enabled}
                    onChange={(e) => setPanel("radius_px", Number(e.target.value))} />
                </label>
              </div>
            </div>

            <div className="cap-two">
              <label className="fld"><span>Position</span>
                <select value={eff.position || "bottom"} onChange={(e) => set("position", e.target.value)}>
                  <option value="bottom">bottom</option><option value="center">center</option><option value="top">top</option>
                </select>
              </label>
              <label className="fld"><span>Alignment</span>
                <select value={eff.align || "center"} onChange={(e) => set("align", e.target.value)}>
                  <option value="left">left</option><option value="center">center</option><option value="right">right</option>
                </select>
              </label>
            </div>

            <div className="cap-two">
              <label className="fld"><span>Side margin · {Number(eff.margin_h_pct).toFixed(1)}%</span>
                <input type="range" min={schema.ranges.margin_h_pct[0]} max={schema.ranges.margin_h_pct[1]}
                  step={0.5} value={eff.margin_h_pct} onChange={(e) => set("margin_h_pct", Number(e.target.value))} />
              </label>
              <label className="fld"><span>Vertical margin · {Number(eff.margin_v_pct).toFixed(1)}%</span>
                <input type="range" min={schema.ranges.margin_v_pct[0]} max={schema.ranges.margin_v_pct[1]}
                  step={0.5} value={eff.margin_v_pct} onChange={(e) => set("margin_v_pct", Number(e.target.value))} />
              </label>
            </div>

            <div className="cap-two">
              <label className="fld"><span>Line spacing · {Number(eff.line_spacing).toFixed(2)}×</span>
                <input type="range" min={schema.ranges.line_spacing[0]} max={schema.ranges.line_spacing[1]}
                  step={0.05} value={eff.line_spacing} onChange={(e) => set("line_spacing", Number(e.target.value))} />
              </label>
              <label className="fld"><span>Max caption width · {Number(eff.max_line_width_pct).toFixed(0)}%</span>
                <input type="range" min={schema.ranges.max_line_width_pct[0]} max={schema.ranges.max_line_width_pct[1]}
                  step={1} value={eff.max_line_width_pct} onChange={(e) => set("max_line_width_pct", Number(e.target.value))} />
              </label>
            </div>

            <div className="cap-two">
              <label className="fld"><span>Max lines</span>
                <select value={eff.max_lines} onChange={(e) => set("max_lines", Number(e.target.value))}>
                  {[1, 2, 3].map((n) => <option key={n} value={n}>{n}</option>)}
                </select>
              </label>
              <label className="fld"><span>Karaoke word highlight</span>
                <select value={eff.karaoke ? "on" : "off"} onChange={(e) => set("karaoke", e.target.value === "on")}>
                  <option value="off">off — sentence captions (frame-accurate)</option>
                  <option value="on">on — word sweep (estimated timing)</option>
                </select>
              </label>
            </div>
            {eff.karaoke ? <div className="hint cap-warn">⚠ karaoke word timing is a proportional
              estimate (no word-level timestamps from the TTS engine) — sentence captions are exact.</div>
              : null}

            <div className="row" style={{ marginTop: 10 }}>
              <button className="btn primary" onClick={save} disabled={saving || !isModified}>
                {saving ? "saving…" : "Save style"}
              </button>
              <button className="btn" onClick={() => { setStyle(savedStyle); setDirty(false); }}
                disabled={!isModified}>discard changes</button>
              <span className="hint">{isModified ? "changes apply after save + next render" : "saved"}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function ColorField({ label, value, onChange }: { label: string; value: string; onChange: (v: string) => void }) {
  const [txt, setTxt] = useState(value);
  useEffect(() => setTxt(value), [value]);
  const ok = /^#[0-9a-fA-F]{6}$/.test(txt);
  return (
    <label className="fld"><span>{label}</span>
      <div className="row" style={{ flexWrap: "nowrap" }}>
        <input type="color" value={ok ? txt : "#ffffff"} style={{ width: 42, padding: 2, flex: "0 0 42px" }}
          onChange={(e) => onChange(e.target.value)} />
        <input value={txt} className={!ok ? "invalid" : ""} onChange={(e) => {
          setTxt(e.target.value);
          if (/^#[0-9a-fA-F]{6}$/.test(e.target.value)) onChange(e.target.value);
        }} placeholder="#rrggbb" />
      </div>
    </label>
  );
}

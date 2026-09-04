import React, { useEffect, useState } from "react";
import { api, Character, ContentTypeMeta, Project, StylePreview, VoiceProfile } from "../api";
import { useToast, errText } from "../main";
import { Modal, Badge } from "../ui";

// New-project flow:
//  Mode → Content type (cards) → Character (optional) → [Advanced: Pace + line gap +
//  subtitles on/off + Style Gallery + title style gallery] → script/topic → create.
const STEPS = ["Mode", "Content", "Character", "Pacing", "Subtitles", "Title", "Script"];
const PACES = [
  { key: "slow", label: "Slow", desc: "0.9× speech · 1.4s gap" },
  { key: "natural", label: "Natural", desc: "1.0× speech · 1.15s gap (default)" },
  { key: "brisk", label: "Brisk", desc: "1.08× speech · 0.95s gap" },
];

export function Wizard({ onClose, onCreated }: { onClose: () => void; onCreated: (id: string) => void }) {
  const [step, setStep] = useState(0);
  const [mode, setMode] = useState<"A" | "B">("A");
  const [ct, setCt] = useState("explainer");
  const [cts, setCts] = useState<ContentTypeMeta[]>([]);
  const [chars, setChars] = useState<Character[]>([]);
  const [characterId, setCharacterId] = useState("");
  const [pace, setPace] = useState("natural");
  const [lineGap, setLineGap] = useState(1.15);
  const [advanced, setAdvanced] = useState(false);
  const [burn, setBurn] = useState(true);
  const [subStyle, setSubStyle] = useState("clean");
  const [titleStyle, setTitleStyle] = useState("");
  const [previews, setPreviews] = useState<{ sub: StylePreview[]; title: StylePreview[] }>({ sub: [], title: [] });
  const [script, setScript] = useState("");
  const [topic, setTopic] = useState("");
  const [duration, setDuration] = useState(30);
  const [styleNotes, setStyleNotes] = useState("");
  const [voices, setVoices] = useState<VoiceProfile[]>([]);
  const [voiceId, setVoiceId] = useState("");
  const [busy, setBusy] = useState(false);
  const toast = useToast();

  useEffect(() => {
    api<{ types: ContentTypeMeta[] }>("/content-types").then((r) => setCts(r.types || [])).catch(() => {});
    api<{ characters: Character[] }>("/characters").then((r) => setChars(r.characters || [])).catch(() => {});
    api<{ voices: VoiceProfile[] }>("/voices").then((r) => setVoices(r.voices || [])).catch(() => {});
    api<{ subtitle_styles: StylePreview[]; title_styles: StylePreview[] }>("/style-previews")
      .then((r) => setPreviews({ sub: r.subtitle_styles || [], title: r.title_styles || [] }))
      .catch((e) => toast("style previews unavailable: " + errText(e), "warn"));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const create = async () => {
    setBusy(true);
    try {
      const payload: any = {
        mode, content_type: ct, character_id: characterId,
        language: "km", target_duration: Number(duration) || 30,
        style_notes: styleNotes, voice_profile_id: voiceId,
        settings: {
          tts: { pace, line_gap_sec: Number(lineGap) || 1.15 },
          assembly: { burn_captions: burn, subtitle_style: subStyle, title_style: titleStyle },
        },
      };
      if (mode === "A") {
        if (!script.trim()) throw new Error("Mode A needs the finished script pasted in.");
        payload.script = script;
      } else {
        payload.topic_hint = topic;
      }
      const r = await api<{ project: Project }>("/projects", { method: "POST", json: payload });
      toast(`project "${r.project.title}" created`, "ok");
      onCreated(r.project.id);
    } catch (e) {
      toast(errText(e), "err");
      setBusy(false);
    }
  };

  const next = () => { if (step < STEPS.length - 1) setStep(step + 1); else create(); };
  const back = () => setStep(Math.max(0, step - 1));
  const canNext = (() => {
    if (step === 0) return true;
    if (step === 1) return !!ct;
    if (step === 6) return mode === "A" ? script.trim().length >= 12 : topic.trim().length > 0;
    return true;
  })();

  return (
    <Modal title="New project" onClose={onClose}>
      <div className="tabs">
        {STEPS.map((s, i) => (
          <button key={s} className={i === step ? "on" : ""} onClick={() => i < step || setStep(i)}>{`${i + 1}. ${s}`}</button>
        ))}
      </div>
      <div style={{ minHeight: 260 }}>
        {step === 0 && (
          <div className="cards" style={{ gridTemplateColumns: "1fr 1fr" }}>
            {[["A", "Director mode", "You paste the finished script. No agent may rewrite it — the studio only segments and produces."],
              ["B", "Auto mode", "Topic in, writer out: the Controller writes a draft, you approve it, then production runs."]].map(([k, t, d]) => (
              <div key={k} className={`ct-card ${mode === k ? "on" : ""}`} onClick={() => setMode(k as any)}>
                <b>{t}</b><div className="hint" style={{ marginTop: 4 }}>{d}</div>
              </div>
            ))}
          </div>
        )}
        {step === 1 && (
          <>
            <p className="hint" style={{ marginBottom: 10 }}>The creative framing shapes the script, scene structure, visuals and QA.</p>
            <div className="cards">
              {cts.map((c) => (
                <div key={c.key} className={`ct-card ${ct === c.key ? "on" : ""}`} onClick={() => setCt(c.key)}>
                  <span className="ct-emoji">{c.emoji}</span> <b>{c.label}</b>
                  <div className="hint" style={{ marginTop: 4 }}>{c.one_liner}</div>
                </div>
              ))}
            </div>
          </>
        )}
        {step === 2 && (
          <div>
            <p className="hint" style={{ marginBottom: 8 }}>Optional NPC: attach a character so scenes can use it (b-roll I2V, gesture demos, or a talking head).</p>
            <select value={characterId} onChange={(e) => setCharacterId(e.target.value)}>
              <option value="">— no character —</option>
              {chars.map((c) => <option key={c.id} value={c.id}>{c.name} ({c.images.length} expressions)</option>)}
            </select>
            <div className="hint" style={{ marginTop: 8 }}>Manage characters in the Characters tab once the project exists.</div>
          </div>
        )}
        {step === 3 && (
          <div>
            <p className="hint" style={{ marginBottom: 10 }}>
              Pace maps to TTS speed + the deterministic silence between lines (<code>tts.line_gap_sec</code>, clamped 0.3–3.0s).
            </p>
            <div className="cards" style={{ gridTemplateColumns: "1fr 1fr 1fr" }}>
              {PACES.map((p) => (
                <div key={p.key} className={`ct-card ${pace === p.key ? "on" : ""}`} onClick={() => { setPace(p.key); if (p.key === "slow") setLineGap(1.4); if (p.key === "natural") setLineGap(1.15); if (p.key === "brisk") setLineGap(0.95); }}>
                  <b>{p.label}</b><div className="hint">{p.desc}</div>
                </div>
              ))}
            </div>
            <label className="fld" style={{ marginTop: 12 }}>
              <span>Line gap (seconds between lines, silence inserted during assembly)</span>
              <input type="number" min={0.3} max={3.0} step={0.05} value={lineGap} onChange={(e) => setLineGap(Number(e.target.value))} />
            </label>
          </div>
        )}
        {step === 4 && (
          <div>
            <div className="spread" style={{ marginBottom: 10 }}>
              <b>Burn captions into the final MP4?</b>
              <input type="checkbox" checked={burn} onChange={(e) => setBurn(e.target.checked)} style={{ width: "auto" }} />
            </div>
            <p className="hint" style={{ marginBottom: 8 }}>Real pre-rendered samples (cached by the backend, ~3s each):</p>
            <div className="gallery">
              {previews.sub.map((s) => (
                <div key={s.key} className={`g-item ${subStyle === s.key ? "on" : ""}`} onClick={() => { setSubStyle(s.key); setBurn(true); }}>
                  {s.url ? <video src={s.url} muted loop autoPlay playsInline /> : <div className="hint" style={{ padding: 20 }}>render unavailable{s.error ? " — " + s.error.slice(0, 60) : ""}</div>}
                  <div className="g-cap"><b>{s.label}</b>{s.error ? <Badge kind="err">broken</Badge> : null}</div>
                </div>
              ))}
            </div>
          </div>
        )}
        {step === 5 && (
          <div>
            <p className="hint" style={{ marginBottom: 8 }}>Optional rendered title card at the start of the video. Pick "none" to skip.</p>
            <div className="gallery">
              <div className={`g-item ${titleStyle === "" ? "on" : ""}`} onClick={() => setTitleStyle("")}>
                <div style={{ height: 130, display: "flex", alignItems: "center", justifyContent: "center" }}>— none —</div>
                <div className="g-cap"><b>No title card</b></div>
              </div>
              {previews.title.map((s) => (
                <div key={s.key} className={`g-item ${titleStyle === s.key ? "on" : ""}`} onClick={() => setTitleStyle(s.key)}>
                  {s.url ? <video src={s.url} muted loop autoPlay playsInline /> : <div className="hint" style={{ padding: 20 }}>render unavailable</div>}
                  <div className="g-cap"><b>{s.label}</b></div>
                </div>
              ))}
            </div>
          </div>
        )}
        {step === 6 && (
          <div>
            {mode === "A" ? (
              <label className="fld">
                <span>Finished Khmer script (one sentence per line — line = scene)</span>
                <textarea rows={7} value={script} onChange={(e) => setScript(e.target.value)}
                  placeholder={"ជីវិតមនុស្ស មិនមែនជាប្រណាំងទេ។\nវាគឺជាដំណើរ ដែលយើងត្រូវរៀនដើរម្ដងមួយជំហាន។"} />
              </label>
            ) : (
              <label className="fld">
                <span>Topic hint (Khmer)</span>
                <input value={topic} onChange={(e) => setTopic(e.target.value)} placeholder="ការមិនបោះបង់ចិត្ត ទោះថ្ងៃលំបាក" />
              </label>
            )}
            <div className="row">
              <label className="fld grow"><span>Target duration (seconds)</span>
                <input type="number" min={10} max={300} value={duration} onChange={(e) => setDuration(Number(e.target.value))} />
              </label>
              <label className="fld grow"><span>Voice profile (optional)</span>
                <select value={voiceId} onChange={(e) => setVoiceId(e.target.value)}>
                  <option value="">house voice (khmer VITS)</option>
                  {voices.map((v) => <option key={v.id} value={v.id}>{v.name}</option>)}
                </select>
              </label>
            </div>
            <label className="fld"><span>Style notes (optional)</span>
              <input value={styleNotes} onChange={(e) => setStyleNotes(e.target.value)} placeholder="kept minimal; slow calm; khmer only" />
            </label>
            <label className="fld"><span>Advanced</span>
              <input type="checkbox" checked={advanced} onChange={(e) => setAdvanced(e.target.checked)} style={{ width: "auto" }} />
              <span className="hint" style={{ marginLeft: 6 }}>opens per-project settings after creation</span>
            </label>
          </div>
        )}
      </div>
      <div className="spread" style={{ marginTop: 14 }}>
        <div className="hint">Step {step + 1}/{STEPS.length}</div>
        <div className="row">
          {step > 0 && <button className="btn" onClick={back}>← back</button>}
          <button className="btn primary" disabled={!canNext || busy} onClick={next}>
            {busy ? <span className="spin" /> : step === STEPS.length - 1 ? "Create project" : "continue →"}
          </button>
        </div>
      </div>
    </Modal>
  );
}

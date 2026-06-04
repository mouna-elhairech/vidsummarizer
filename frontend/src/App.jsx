import { useState, useRef, useEffect } from "react";
import "./App.css";

const API = "http://127.0.0.1:8000";

const STAGES = [
  { key: "extracting", label: "Frames" },
  { key: "captioning", label: "Captions" },
  { key: "transcribing", label: "Audio" },
  { key: "summarizing", label: "Summary" },
];

const stageOrder = ["uploading", "extracting", "captioning", "transcribing", "summarizing", "done"];

function App() {
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [videoUrl, setVideoUrl] = useState("");
  const [language, setLanguage] = useState("en");
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");
  const [result, setResult] = useState(null);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const intervalRef = useRef(null);

  useEffect(() => {
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
      window.speechSynthesis.cancel();
    };
  }, []);

  const handleVideoChange = (event) => {
    const file = event.target.files[0];
    setSelectedVideo(file);
    setResult(null);
    setErrorMsg("");
    if (file) setVideoUrl(URL.createObjectURL(file));
  };

  const handleAnalyze = async () => {
    if (!selectedVideo) {
      setErrorMsg("Please select a video first.");
      return;
    }

    window.speechSynthesis.cancel();
    setIsSpeaking(false);
    setLoading(true);
    setErrorMsg("");
    setResult(null);

    const formData = new FormData();
    formData.append("video", selectedVideo);
    formData.append("language", language);

    try {
      const startResponse = await fetch(`${API}/analyze-video`, {
        method: "POST",
        body: formData,
      });
      const startData = await startResponse.json();
      const jobId = startData.job_id;

      intervalRef.current = setInterval(async () => {
        const statusResponse = await fetch(`${API}/job-status/${jobId}`);
        const statusData = await statusResponse.json();

        if (statusData.result) setResult(statusData.result);

        if (statusData.status === "completed") {
          clearInterval(intervalRef.current);
          setLoading(false);
        }
        if (statusData.status === "failed") {
          clearInterval(intervalRef.current);
          setErrorMsg(statusData.error || "Analysis failed.");
          setLoading(false);
        }
      }, 1000);
    } catch (error) {
      console.error(error);
      setErrorMsg("Error while analyzing video.");
      setLoading(false);
    }
  };

  const handleReadSummary = () => {
    if (!result?.final_summary) return;
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(result.final_summary);
    const lang = result?.language || language;
    u.lang = lang === "fr" ? "fr-FR" : "en-US";
    u.onstart = () => setIsSpeaking(true);
    u.onend = () => setIsSpeaking(false);
    u.onerror = () => setIsSpeaking(false);
    window.speechSynthesis.speak(u);
  };

  const handleStopReading = () => {
    window.speechSynthesis.cancel();
    setIsSpeaking(false);
  };

  const stageIndex = result ? stageOrder.indexOf(result.stage) : -1;
  const started = !!result;

  return (
    <div className="vs-shell">
      <style>{`
        .vs-shell {
          min-height: 100vh;
          background: radial-gradient(1200px 600px at 50% -10%, #1e293b 0%, #0f172a 60%, #0b1120 100%);
          color: #e2e8f0;
          font-family: Inter, system-ui, -apple-system, sans-serif;
          padding: 32px 20px 80px;
          box-sizing: border-box;
        }
        .vs-container { max-width: 1000px; margin: 0 auto; }
        .vs-title { text-align: center; font-size: 30px; font-weight: 800; letter-spacing: -0.02em;
          background: linear-gradient(135deg,#60a5fa,#a78bfa); -webkit-background-clip: text;
          background-clip: text; color: transparent; margin: 6px 0 4px; }
        .vs-subtitle { text-align: center; color: #94a3b8; margin: 0 0 28px; font-size: 14px; }

        .vs-panel { background: rgba(30,41,59,0.6); border: 1px solid rgba(148,163,184,0.12);
          border-radius: 18px; padding: 22px; backdrop-filter: blur(8px); margin-bottom: 18px; }

        .vs-drop { border: 2px dashed rgba(148,163,184,0.3); border-radius: 14px; padding: 26px;
          text-align: center; transition: border-color .2s; }
        .vs-drop:hover { border-color: #60a5fa; }
        .vs-file-input { width: 100%; cursor: pointer; color: #cbd5e1; }

        .vs-lang-row { display:flex; align-items:center; gap:10px; margin: 18px 0; flex-wrap: wrap; }
        .vs-chip { padding: 8px 18px; border-radius: 999px; border: 1px solid rgba(148,163,184,0.25);
          background: transparent; color: #cbd5e1; cursor: pointer; font-weight: 600; transition: all .15s; }
        .vs-chip.active { background: linear-gradient(135deg,#3b82f6,#6366f1); color: white; border-color: transparent; }
        .vs-chip:disabled { opacity: .5; cursor: not-allowed; }

        .vs-btn { padding: 12px 28px; border-radius: 12px; border: none; cursor: pointer;
          font-weight: 700; font-size: 15px; background: linear-gradient(135deg,#3b82f6,#6366f1);
          color: white; transition: transform .1s, box-shadow .2s; box-shadow: 0 8px 24px rgba(59,130,246,0.3); }
        .vs-btn:hover:not(:disabled) { transform: translateY(-1px); }
        .vs-btn:disabled { opacity: .6; cursor: not-allowed; box-shadow: none; }
        .vs-btn.secondary { background: linear-gradient(135deg,#475569,#64748b); box-shadow:none; }

        .vs-video { width: 100%; border-radius: 12px; margin-top: 14px; max-height: 320px; background:#000; }

        .vs-stepper { display:flex; gap: 8px; margin: 4px 0 22px; }
        .vs-step { flex:1; text-align:center; padding: 10px 6px; border-radius: 10px; font-size: 13px;
          font-weight:600; background: rgba(148,163,184,0.08); color:#64748b; border:1px solid transparent; }
        .vs-step.done { color:#34d399; background: rgba(52,211,153,0.1); }
        .vs-step.active { color:#60a5fa; background: rgba(96,165,250,0.12); border-color: rgba(96,165,250,0.4);
          animation: vsPulse 1.2s ease-in-out infinite; }

        .vs-section-title { font-size: 15px; font-weight: 700; color:#cbd5e1; margin: 0 0 14px;
          display:flex; align-items:center; gap:8px; }
        .vs-count { font-size: 12px; font-weight:600; color:#64748b; }

        .vs-frame-grid { display:grid; grid-template-columns: repeat(auto-fill, minmax(190px,1fr)); gap: 14px; }
        .vs-frame-card { background: rgba(15,23,42,0.6); border:1px solid rgba(148,163,184,0.12);
          border-radius: 12px; overflow:hidden; animation: vsFadeUp .45s ease-out; }
        .vs-frame-img { width:100%; height: 120px; object-fit: cover; display:block; background:#1e293b; }
        .vs-frame-body { padding: 10px 12px; }
        .vs-frame-ts { font-size: 11px; font-weight:700; color:#60a5fa; }
        .vs-frame-cap { font-size: 13px; color:#cbd5e1; margin: 4px 0 0; line-height: 1.4; }

        .vs-transcript-line { display:flex; gap:12px; padding: 8px 0; border-bottom:1px solid rgba(148,163,184,0.08);
          animation: vsFadeUp .4s ease-out; }
        .vs-ts-time { flex-shrink:0; font-size: 12px; font-weight:700; color:#a78bfa; min-width: 64px; }
        .vs-ts-text { font-size: 14px; color:#e2e8f0; }

        .vs-summary { font-size: 16px; line-height: 1.7; color:#f1f5f9; }
        .vs-cursor { display:inline-block; width:8px; background:#60a5fa; margin-left:2px;
          animation: vsBlink 1s step-start infinite; }

        .vs-meta { display:flex; gap:16px; flex-wrap:wrap; margin-bottom:14px; font-size:13px; color:#94a3b8; }
        .vs-error { color:#f87171; text-align:center; margin-top: 12px; font-weight:600; }
        .vs-muted { color:#64748b; font-size: 14px; }

        @keyframes vsFadeUp { from {opacity:0; transform:translateY(14px);} to {opacity:1; transform:translateY(0);} }
        @keyframes vsPulse { 0%,100%{opacity:1;} 50%{opacity:.55;} }
        @keyframes vsBlink { 50% { opacity:0; } }
      `}</style>

      <div className="vs-container">
        <h1 className="vs-title">AI Video Summarizer</h1>
        <p className="vs-subtitle">Local · Frames · Captions · Audio · Summary</p>

        {/* ---------- Upload panel (always visible) ---------- */}
        <div className="vs-panel">
          <div className="vs-drop">
            <input
              className="vs-file-input"
              type="file"
              accept="video/*"
              onChange={handleVideoChange}
              disabled={loading}
            />
            {!selectedVideo && (
              <p className="vs-muted" style={{ marginBottom: 0 }}>
                Choose a video to begin.
              </p>
            )}
          </div>

          {videoUrl && <video src={videoUrl} controls className="vs-video" />}

          <div className="vs-lang-row">
            <span className="vs-muted">Summary language:</span>
            <button
              className={`vs-chip ${language === "en" ? "active" : ""}`}
              onClick={() => setLanguage("en")}
              disabled={loading}
            >
              English
            </button>
            <button
              className={`vs-chip ${language === "fr" ? "active" : ""}`}
              onClick={() => setLanguage("fr")}
              disabled={loading}
            >
              Français
            </button>
          </div>

          <button className="vs-btn" onClick={handleAnalyze} disabled={loading}>
            {loading ? "Analyzing..." : "Analyze Video"}
          </button>

          {errorMsg && <p className="vs-error">{errorMsg}</p>}
        </div>

        {/* ---------- Live stepper ---------- */}
        {started && (
          <div className="vs-panel">
            <div className="vs-stepper">
              {STAGES.map((s) => {
                const idx = stageOrder.indexOf(s.key);
                const cls =
                  stageIndex > idx ? "done" : stageIndex === idx ? "active" : "";
                return (
                  <div key={s.key} className={`vs-step ${cls}`}>
                    {stageIndex > idx ? "✓ " : ""}
                    {s.label}
                  </div>
                );
              })}
            </div>
            {result?.extraction_result && (
              <div className="vs-meta">
                <span>Duration: {result.extraction_result.duration_sec}s</span>
                <span>Scenes: {result.extraction_result.scenes_detected}</span>
                <span>Kept frames: {result.extraction_result.kept_frames}</span>
                <span>Ignored: {result.extraction_result.ignored_frames}</span>
              </div>
            )}
          </div>
        )}

        {/* ---------- Frames + captions (progressive) ---------- */}
        {started && result.captions.length > 0 && (
          <div className="vs-panel">
            <p className="vs-section-title">
              Frames &amp; Captions
              <span className="vs-count">
                {result.captions.length}/{result.frames.length || result.captions.length}
              </span>
            </p>
            <div className="vs-frame-grid">
              {result.captions.map((c) => (
                <div key={c.index} className="vs-frame-card">
                  <img
                    className="vs-frame-img"
                    src={`${API}/frames/${c.frame}`}
                    alt={c.frame}
                    loading="lazy"
                  />
                  <div className="vs-frame-body">
                    {c.timestamp_sec != null && (
                      <div className="vs-frame-ts">{c.timestamp_sec.toFixed(2)}s</div>
                    )}
                    <p className="vs-frame-cap">{c.caption}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ---------- Transcript (progressive) ---------- */}
        {started && result.transcript.segments.length > 0 && (
          <div className="vs-panel">
            <p className="vs-section-title">
              Audio Transcription
              {result.transcript.language && (
                <span className="vs-count">
                  {result.transcript.language.toUpperCase()}
                  {result.transcript.language_probability != null &&
                    ` · ${(result.transcript.language_probability * 100).toFixed(0)}%`}
                </span>
              )}
            </p>
            <div>
              {result.transcript.segments.map((seg, i) => (
                <div key={i} className="vs-transcript-line">
                  <span className="vs-ts-time">
                    {seg.start.toFixed(2)}s
                  </span>
                  <span className="vs-ts-text">{seg.text}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ---------- Final summary (typewriter) ---------- */}
        {started && (result.stage === "summarizing" || result.stage === "done") && (
          <div className="vs-panel">
            <p className="vs-section-title">
              Final Summary
              {result.language && (
                <span className="vs-count">{result.language.toUpperCase()}</span>
              )}
            </p>
            <p className="vs-summary">
              {result.final_summary || (
                <span className="vs-muted">Generating...</span>
              )}
              {!result.final_summary_done && result.final_summary && (
                <span className="vs-cursor">&nbsp;</span>
              )}
            </p>

            {result.final_summary_done && (
              <div className="vs-lang-row" style={{ marginTop: 10 }}>
                <button className="vs-btn" onClick={handleReadSummary}>
                  Read Summary
                </button>
                <button className="vs-btn secondary" onClick={handleStopReading}>
                  Stop
                </button>
                {isSpeaking && <span className="vs-muted">Reading...</span>}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
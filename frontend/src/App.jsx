import { useState, useRef, useEffect } from "react";
import "./App.css";

const API = "http://127.0.0.1:8000";

const STAGES = [
  { key: "extracting",   label: "Frames"    },
  { key: "captioning",   label: "Captions"  },
  { key: "ocr",          label: "OCR"       },
  { key: "transcribing", label: "Audio"     },
  { key: "summarizing",  label: "Summary"   },
];

const stageOrder = [
  "uploading", "extracting", "captioning",
  "ocr", "transcribing", "summarizing", "done",
];

export default function App() {
  const [inputMode,     setInputMode]     = useState("upload");
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [videoUrl,      setVideoUrl]      = useState("");
  const [youtubeUrl,    setYoutubeUrl]    = useState("");
  const [language,      setLanguage]      = useState("en");
  const [loading,       setLoading]       = useState(false);
  const [errorMsg,      setErrorMsg]      = useState("");
  const [result,        setResult]        = useState(null);
  const [isSpeaking,    setIsSpeaking]    = useState(false);
  const videoRef  = useRef(null);
  const intervalRef = useRef(null);

  useEffect(() => () => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    window.speechSynthesis.cancel();
  }, []);

  const handleVideoChange = (e) => {
    const file = e.target.files[0];
    setSelectedVideo(file);
    setResult(null);
    setErrorMsg("");
    if (file) setVideoUrl(URL.createObjectURL(file));
  };

  const startPolling = (jobId) => {
    intervalRef.current = setInterval(async () => {
      const res  = await fetch(`${API}/job-status/${jobId}`);
      const data = await res.json();
      if (data.result) setResult({ ...data.result });
      if (data.status === "completed") { clearInterval(intervalRef.current); setLoading(false); }
      if (data.status === "failed")    {
        clearInterval(intervalRef.current);
        setErrorMsg(data.error || "Analysis failed.");
        setLoading(false);
      }
    }, 1000);
  };

  const handleAnalyze = async () => {
    if (inputMode === "upload"  && !selectedVideo)        { setErrorMsg("Please select a video."); return; }
    if (inputMode === "youtube" && !youtubeUrl.trim())    { setErrorMsg("Please paste a YouTube URL."); return; }

    window.speechSynthesis.cancel();
    setIsSpeaking(false);
    setLoading(true);
    setErrorMsg("");
    setResult(null);

    try {
      const fd = new FormData();
      fd.append("language", language);
      let endpoint;
      if (inputMode === "upload") {
        fd.append("video", selectedVideo);
        endpoint = `${API}/analyze-video`;
      } else {
        fd.append("url", youtubeUrl.trim());
        endpoint = `${API}/analyze-youtube`;
      }
      const res  = await fetch(endpoint, { method: "POST", body: fd });
      const data = await res.json();
      startPolling(data.job_id);
    } catch (err) {
      console.error(err);
      setErrorMsg("Error while starting analysis.");
      setLoading(false);
    }
  };

  const jumpToTime = (sec) => {
    if (videoRef.current && sec != null) {
      videoRef.current.currentTime = sec;
      videoRef.current.play().catch(() => {});
    }
  };

  const handleReadSummary = () => {
    if (!result?.final_summary) return;
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(result.final_summary);
    u.lang = (result?.language || language) === "fr" ? "fr-FR" : "en-US";
    u.onstart = () => setIsSpeaking(true);
    u.onend   = () => setIsSpeaking(false);
    u.onerror = () => setIsSpeaking(false);
    window.speechSynthesis.speak(u);
  };

  const stageIndex = result ? stageOrder.indexOf(result.stage) : -1;
  const started    = !!result;

  return (
    <div className="vs-shell">
      <style>{`
        *{box-sizing:border-box;}
        .vs-shell{min-height:100vh;
          background:radial-gradient(1200px 600px at 50% -10%,#1e293b 0%,#0f172a 60%,#0b1120 100%);
          color:#e2e8f0;font-family:Inter,system-ui,-apple-system,sans-serif;
          padding:32px 20px 80px;}
        .vs-container{max-width:1000px;margin:0 auto;}
        .vs-title{text-align:center;font-size:30px;font-weight:800;letter-spacing:-.02em;
          background:linear-gradient(135deg,#60a5fa,#a78bfa);-webkit-background-clip:text;
          background-clip:text;color:transparent;margin:6px 0 4px;}
        .vs-subtitle{text-align:center;color:#94a3b8;margin:0 0 28px;font-size:14px;}
        .vs-panel{background:rgba(30,41,59,.6);border:1px solid rgba(148,163,184,.12);
          border-radius:18px;padding:22px;backdrop-filter:blur(8px);margin-bottom:18px;}
        .vs-tabs{display:flex;gap:8px;margin-bottom:16px;}
        .vs-tab{flex:1;padding:10px;border-radius:10px;border:1px solid rgba(148,163,184,.2);
          background:transparent;color:#94a3b8;cursor:pointer;font-weight:600;transition:all .15s;}
        .vs-tab.active{background:linear-gradient(135deg,#3b82f6,#6366f1);color:#fff;border-color:transparent;}
        .vs-tab:disabled{opacity:.5;cursor:not-allowed;}
        .vs-drop{border:2px dashed rgba(148,163,184,.3);border-radius:14px;padding:26px;text-align:center;}
        .vs-file-input{width:100%;cursor:pointer;color:#cbd5e1;}
        .vs-url-input{width:100%;padding:14px 16px;border-radius:12px;
          border:1px solid rgba(148,163,184,.25);background:rgba(15,23,42,.6);
          color:#e2e8f0;font-size:14px;outline:none;}
        .vs-url-input:focus{border-color:#60a5fa;}
        .vs-lang-row{display:flex;align-items:center;gap:10px;margin:18px 0;flex-wrap:wrap;}
        .vs-chip{padding:8px 18px;border-radius:999px;border:1px solid rgba(148,163,184,.25);
          background:transparent;color:#cbd5e1;cursor:pointer;font-weight:600;transition:all .15s;}
        .vs-chip.active{background:linear-gradient(135deg,#3b82f6,#6366f1);color:#fff;border-color:transparent;}
        .vs-chip:disabled{opacity:.5;cursor:not-allowed;}
        .vs-btn{padding:12px 28px;border-radius:12px;border:none;cursor:pointer;font-weight:700;
          font-size:15px;background:linear-gradient(135deg,#3b82f6,#6366f1);color:#fff;
          box-shadow:0 8px 24px rgba(59,130,246,.3);transition:transform .1s;}
        .vs-btn:hover:not(:disabled){transform:translateY(-1px);}
        .vs-btn:disabled{opacity:.6;cursor:not-allowed;box-shadow:none;}
        .vs-btn.secondary{background:linear-gradient(135deg,#475569,#64748b);box-shadow:none;}
        .vs-video{width:100%;border-radius:12px;margin-top:14px;max-height:320px;background:#000;}
        .vs-stepper{display:flex;gap:8px;margin:4px 0 16px;flex-wrap:wrap;}
        .vs-step{flex:1;min-width:80px;text-align:center;padding:10px 6px;border-radius:10px;
          font-size:13px;font-weight:600;background:rgba(148,163,184,.08);
          color:#64748b;border:1px solid transparent;}
        .vs-step.done{color:#34d399;background:rgba(52,211,153,.1);}
        .vs-step.active{color:#60a5fa;background:rgba(96,165,250,.12);
          border-color:rgba(96,165,250,.4);animation:vsPulse 1.2s ease-in-out infinite;}
        .vs-meta{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:14px;font-size:13px;color:#94a3b8;}
        .vs-section-title{font-size:15px;font-weight:700;color:#cbd5e1;margin:0 0 14px;
          display:flex;align-items:center;gap:8px;}
        .vs-count{font-size:12px;font-weight:600;color:#64748b;}
        .vs-frame-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:14px;}
        .vs-frame-card{background:rgba(15,23,42,.6);border:1px solid rgba(148,163,184,.12);
          border-radius:12px;overflow:hidden;animation:vsFadeUp .45s ease-out;
          cursor:pointer;transition:border-color .15s;}
        .vs-frame-card:hover{border-color:#60a5fa;}
        .vs-frame-img{width:100%;height:120px;object-fit:cover;display:block;background:#1e293b;}
        .vs-frame-body{padding:10px 12px;}
        .vs-frame-ts{font-size:11px;font-weight:700;color:#60a5fa;}
        .vs-frame-cap{font-size:13px;color:#cbd5e1;margin:4px 0 0;line-height:1.4;}
        .vs-ocr-badge{display:inline-block;background:rgba(167,139,250,.15);
          border:1px solid rgba(167,139,250,.3);color:#a78bfa;
          font-size:11px;padding:2px 8px;border-radius:999px;margin-top:6px;}
        .vs-ocr-text{font-size:12px;color:#94a3b8;margin:4px 0 0;font-style:italic;line-height:1.4;}
        .vs-ocr-panel{background:rgba(167,139,250,.06);border:1px solid rgba(167,139,250,.15);
          border-radius:12px;padding:14px 16px;margin-top:12px;}
        .vs-ocr-panel-title{font-size:12px;font-weight:700;color:#a78bfa;margin:0 0 8px;}
        .vs-ocr-panel-text{font-size:14px;color:#e2e8f0;line-height:1.6;}
        .vs-transcript-line{display:flex;gap:12px;padding:8px 0;
          border-bottom:1px solid rgba(148,163,184,.08);animation:vsFadeUp .4s ease-out;}
        .vs-ts-time{flex-shrink:0;font-size:12px;font-weight:700;color:#a78bfa;min-width:64px;}
        .vs-ts-text{font-size:14px;color:#e2e8f0;}
        .vs-summary{font-size:16px;line-height:1.7;color:#f1f5f9;}
        .vs-cursor{display:inline-block;width:8px;background:#60a5fa;margin-left:2px;
          animation:vsBlink 1s step-start infinite;}
        .vs-error{color:#f87171;text-align:center;margin-top:12px;font-weight:600;}
        .vs-muted{color:#64748b;font-size:14px;}
        @keyframes vsFadeUp{from{opacity:0;transform:translateY(14px);}to{opacity:1;transform:translateY(0);}}
        @keyframes vsPulse{0%,100%{opacity:1;}50%{opacity:.55;}}
        @keyframes vsBlink{50%{opacity:0;}}
      `}</style>

      <div className="vs-container">
        <h1 className="vs-title">AI Video Summarizer</h1>
        <p className="vs-subtitle">Local · Frames · Captions · OCR · Audio · Summary</p>

        {/* Upload panel */}
        <div className="vs-panel">
          <div className="vs-tabs">
            <button className={`vs-tab ${inputMode==="upload"?"active":""}`}
              onClick={()=>{setInputMode("upload");setErrorMsg("");}} disabled={loading}>
              📁 Upload file
            </button>
            <button className={`vs-tab ${inputMode==="youtube"?"active":""}`}
              onClick={()=>{setInputMode("youtube");setErrorMsg("");}} disabled={loading}>
              ▶️ YouTube link
            </button>
          </div>

          {inputMode === "upload" ? (
            <>
              <div className="vs-drop">
                <input className="vs-file-input" type="file" accept="video/*"
                  onChange={handleVideoChange} disabled={loading} />
                {!selectedVideo && <p className="vs-muted" style={{marginBottom:0}}>Choose a video to begin.</p>}
              </div>
              {videoUrl && (
                <video ref={videoRef} src={videoUrl} controls className="vs-video" />
              )}
            </>
          ) : (
            <input className="vs-url-input" type="text"
              placeholder="Paste a YouTube URL (max 15 min)..."
              value={youtubeUrl} onChange={e=>setYoutubeUrl(e.target.value)} disabled={loading} />
          )}

          <div className="vs-lang-row">
            <span className="vs-muted">Summary language:</span>
            <button className={`vs-chip ${language==="en"?"active":""}`}
              onClick={()=>setLanguage("en")} disabled={loading}>English</button>
            <button className={`vs-chip ${language==="fr"?"active":""}`}
              onClick={()=>setLanguage("fr")} disabled={loading}>Français</button>
          </div>

          <button className="vs-btn" onClick={handleAnalyze} disabled={loading}>
            {loading ? "Analyzing..." : "Analyze Video"}
          </button>
          {errorMsg && <p className="vs-error">{errorMsg}</p>}
        </div>

        {/* Stepper */}
        {started && (
          <div className="vs-panel">
            <div className="vs-stepper">
              {STAGES.map(s => {
                const idx = stageOrder.indexOf(s.key);
                const cls = stageIndex > idx ? "done" : stageIndex === idx ? "active" : "";
                return (
                  <div key={s.key} className={`vs-step ${cls}`}>
                    {stageIndex > idx ? "✓ " : ""}{s.label}
                  </div>
                );
              })}
            </div>
            {result?.video_filename && (
              <div className="vs-meta"><span>📹 {result.video_filename}</span></div>
            )}
            {result?.extraction_result && (
              <div className="vs-meta">
                <span>Duration: {result.extraction_result.duration_sec}s</span>
                <span>Scenes: {result.extraction_result.scenes_detected}</span>
                <span>Kept: {result.extraction_result.kept_frames} frames</span>
                <span>Ignored: {result.extraction_result.ignored_frames}</span>
              </div>
            )}
          </div>
        )}

        {/* Frames + Captions + OCR badges */}
        {started && result.captions.length > 0 && (
          <div className="vs-panel">
            <p className="vs-section-title">
              Frames &amp; Captions
              <span className="vs-count">
                {result.captions.length}/{result.frames.length || result.captions.length}
              </span>
              {result.ocr_results?.length > 0 && (
                <span className="vs-count" style={{color:"#a78bfa"}}>
                  · {result.ocr_results.length} with text
                </span>
              )}
            </p>
            <div className="vs-frame-grid">
              {result.captions.map(c => {
                const ocrEntry = result.ocr_results?.find(o => o.frame === c.frame);
                return (
                  <div key={c.index} className="vs-frame-card"
                    onClick={() => jumpToTime(c.timestamp_sec)}
                    title="Click to jump to this moment">
                    <img className="vs-frame-img"
                      src={`${API}/frames/${c.frame}`} alt={c.frame} loading="lazy" />
                    <div className="vs-frame-body">
                      {c.timestamp_sec != null && (
                        <div className="vs-frame-ts">{c.timestamp_sec.toFixed(2)}s ▶</div>
                      )}
                      <p className="vs-frame-cap">{c.caption}</p>
                      {ocrEntry && (
                        <>
                          <span className="vs-ocr-badge">📝 Text detected</span>
                          <p className="vs-ocr-text">"{ocrEntry.text}"</p>
                        </>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Combined OCR panel */}
            {result.ocr_combined && (
              <div className="vs-ocr-panel">
                <p className="vs-ocr-panel-title">📝 All on-screen text</p>
                <p className="vs-ocr-panel-text">{result.ocr_combined}</p>
              </div>
            )}
          </div>
        )}

        {/* Audio Transcript */}
        {started && result.transcript.segments.length > 0 && (
          <div className="vs-panel">
            <p className="vs-section-title">
              Audio Transcription
              {result.transcript.language && (
                <span className="vs-count">
                  {result.transcript.language.toUpperCase()}
                  {result.transcript.language_probability != null &&
                    ` · ${(result.transcript.language_probability*100).toFixed(0)}%`}
                </span>
              )}
            </p>
            <div>
              {result.transcript.segments.map((seg,i) => (
                <div key={i} className="vs-transcript-line"
                  style={{cursor:"pointer"}} onClick={()=>jumpToTime(seg.start)}
                  title="Click to jump to this moment">
                  <span className="vs-ts-time">{seg.start.toFixed(2)}s ▶</span>
                  <span className="vs-ts-text">{seg.text}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Final Summary */}
        {started && (result.stage === "summarizing" || result.stage === "done") && (
          <div className="vs-panel">
            <p className="vs-section-title">
              Final Summary
              {result.language && <span className="vs-count">{result.language.toUpperCase()}</span>}
            </p>
            <p className="vs-summary">
              {result.final_summary || (
                <span className="vs-muted">{result.summary_progress || "Generating..."}</span>
              )}
              {!result.final_summary_done && result.final_summary && (
                <span className="vs-cursor">&nbsp;</span>
              )}
            </p>
            {result.final_summary_done && (
              <div className="vs-lang-row" style={{marginTop:10}}>
                <button className="vs-btn" onClick={handleReadSummary}>Read Summary</button>
                <button className="vs-btn secondary" onClick={()=>{window.speechSynthesis.cancel();setIsSpeaking(false);}}>Stop</button>
                {isSpeaking && <span className="vs-muted">Reading...</span>}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
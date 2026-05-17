import { useState } from "react";
import "./App.css";

function App() {
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [videoUrl, setVideoUrl] = useState("");
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [progressSteps, setProgressSteps] = useState([]);
  const [isSpeaking, setIsSpeaking] = useState(false);

  const handleVideoChange = (event) => {
    const file = event.target.files[0];
    setSelectedVideo(file);
    setResult(null);
    setMessage("");

    if (file) {
      const url = URL.createObjectURL(file);
      setVideoUrl(url);
      setProgressSteps(["Video selected"]);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedVideo) {
      setMessage("Please select a video first.");
      return;
    }

    window.speechSynthesis.cancel();
    setIsSpeaking(false);

    setLoading(true);
    setMessage("Analysis in progress...");
    setResult(null);
    setProgressSteps(["Video selected", "Uploading video..."]);

    const formData = new FormData();
    formData.append("video", selectedVideo);

    try {
      const startResponse = await fetch("http://127.0.0.1:8000/analyze-video", {
        method: "POST",
        body: formData,
      });

      const startData = await startResponse.json();
      const jobId = startData.job_id;

      const interval = setInterval(async () => {
        const statusResponse = await fetch(`http://127.0.0.1:8000/job-status/${jobId}`);
        const statusData = await statusResponse.json();

        if (statusData.steps) {
          setProgressSteps(
            statusData.steps.map((step) =>
              step.replace("✅ ", "").replace("⏳ ", "").replace("❌ ", "")
            )
          );
        }

        if (statusData.status === "completed") {
          clearInterval(interval);
          setResult(statusData.result);
          setMessage("Analysis completed successfully.");
          setLoading(false);
        }

        if (statusData.status === "failed") {
          clearInterval(interval);
          setMessage("Analysis failed.");
          setLoading(false);
        }
      }, 5000);
    } catch (error) {
      console.error(error);
      setMessage("Error while analyzing video.");
      setProgressSteps((prev) => [...prev, "Analysis failed"]);
      setLoading(false);
    }
  };

  const handleReadSummary = () => {
    if (!result?.final_summary) return;

    window.speechSynthesis.cancel();

    const utterance = new SpeechSynthesisUtterance(result.final_summary);
    utterance.lang = "en-US";
    utterance.rate = 1;
    utterance.pitch = 1;

    utterance.onstart = () => setIsSpeaking(true);
    utterance.onend = () => setIsSpeaking(false);
    utterance.onerror = () => setIsSpeaking(false);

    window.speechSynthesis.speak(utterance);
  };

  const handleStopReading = () => {
    window.speechSynthesis.cancel();
    setIsSpeaking(false);
  };

  return (
    <div className="app-shell">
      <header className="hero">
        <div>
          <p className="eyebrow">Local AI Video Understanding</p>
          <h1>Video Summarizer Dashboard</h1>
          <p className="hero-text">
            Upload a video, analyze key visual moments, review extracted captions,
            and read a final coherent summary generated from the visual pipeline.
          </p>
        </div>

        <div className="hero-badges">
          <span className="badge">FastAPI Backend</span>
          <span className="badge">React Frontend</span>
          <span className="badge">Local Models</span>
        </div>
      </header>

      <main className="dashboard-grid">
        <section className="left-column">
          <div className="card">
            <div className="card-header">
              <div>
                <p className="card-kicker">Step 1</p>
                <h2>Video Upload</h2>
              </div>
            </div>

            <div className="upload-area">
              <input
                className="file-input"
                type="file"
                accept="video/*"
                onChange={handleVideoChange}
              />

              {selectedVideo ? (
                <div className="file-info">
                  <span className="file-label">Selected file</span>
                  <strong>{selectedVideo.name}</strong>
                </div>
              ) : (
                <p className="muted">Choose a video to start the visual analysis.</p>
              )}
            </div>

            {videoUrl && (
              <div className="video-wrapper">
                <video src={videoUrl} controls className="video-player" />
              </div>
            )}

            <div className="actions-row">
              <button
                className="analyze-button"
                onClick={handleAnalyze}
                disabled={loading}
              >
                {loading ? "Analyzing..." : "Analyze Video"}
              </button>

              {message && <span className="status-message">{message}</span>}
            </div>
          </div>

          <div className="card">
            <div className="card-header">
              <div>
                <p className="card-kicker">Step 2</p>
                <h2>Analysis Progress</h2>
              </div>
            </div>

            {progressSteps.length > 0 ? (
              <div className="progress-list">
                {progressSteps.map((step, index) => (
                  <div key={index} className="progress-item">
                    <div className="progress-dot" />
                    <span>{step}</span>
                  </div>
                ))}
              </div>
            ) : (
              <p className="muted">Progress updates will appear here during analysis.</p>
            )}
          </div>

          <div className="card">
            <div className="card-header">
              <div>
                <p className="card-kicker">Step 3</p>
                <h2>Extraction Stats</h2>
              </div>
            </div>

            {result?.extraction_result ? (
              <div className="stats-grid">
                <div className="stat-box">
                  <span className="stat-label">Comparisons</span>
                  <strong>{result.extraction_result.total_comparisons}</strong>
                </div>

                <div className="stat-box">
                  <span className="stat-label">Kept Frames</span>
                  <strong>{result.extraction_result.kept_frames}</strong>
                </div>

                <div className="stat-box">
                  <span className="stat-label">Ignored Frames</span>
                  <strong>{result.extraction_result.ignored_frames}</strong>
                </div>

                <div className="stat-box">
                  <span className="stat-label">Frames Folder</span>
                  <strong>{result.extraction_result.frames_folder}</strong>
                </div>
              </div>
            ) : (
              <p className="muted">Extraction statistics will appear after analysis.</p>
            )}
          </div>
        </section>

        <section className="right-column">
          <div className="card">
            <div className="card-header">
              <div>
                <p className="card-kicker">Step 4</p>
                <h2>Generated Captions</h2>
              </div>
            </div>

            {result?.caption_result ? (
              <>
                <div className="caption-meta">
                  <span>Frames captioned: {result.caption_result.frames_captioned}</span>
                  <span>Frames skipped: {result.caption_result.frames_skipped}</span>
                  <span>Output file: {result.caption_result.output_file}</span>
                </div>

                <div className="captions-list">
                  {result.caption_result.captions.map((item, index) => (
                    <div key={index} className="caption-card">
                      <div className="caption-frame">{item.frame}</div>
                      <p className="caption-text">{item.caption}</p>
                    </div>
                  ))}
                </div>
              </>
            ) : (
              <p className="muted">Generated captions will appear here.</p>
            )}
          </div>

          <div className="card summary-card">
            <div className="card-header">
              <div>
                <p className="card-kicker">Step 5</p>
                <h2>Final Summary</h2>
              </div>
            </div>

            {result?.final_summary ? (
              <>
                <p className="summary-text">{result.final_summary}</p>

                <div className="actions-row" style={{ marginTop: "18px" }}>
                  <button className="analyze-button" onClick={handleReadSummary}>
                    Read Summary
                  </button>

                  <button
                    className="analyze-button"
                    onClick={handleStopReading}
                    style={{ background: "linear-gradient(135deg, #475569 0%, #64748b 100%)" }}
                  >
                    Stop Reading
                  </button>

                  {isSpeaking && (
                    <span className="status-message">Reading summary...</span>
                  )}
                </div>
              </>
            ) : (
              <p className="muted">The final visual summary will appear here.</p>
            )}
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
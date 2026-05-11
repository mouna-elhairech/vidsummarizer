import { useState } from "react";

function App() {
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [videoUrl, setVideoUrl] = useState("");
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const handleVideoChange = (event) => {
    const file = event.target.files[0];
    setSelectedVideo(file);
    setResult(null);
    setMessage("");

    if (file) {
      const url = URL.createObjectURL(file);
      setVideoUrl(url);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedVideo) {
      setMessage("Please select a video first.");
      return;
    }

    setLoading(true);
    setMessage("Analysis in progress...");
    setResult(null);

    const formData = new FormData();
    formData.append("video", selectedVideo);

    try {
      const response = await fetch("http://127.0.0.1:8000/analyze-video", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      setResult(data);
      setMessage("Analysis completed successfully.");
    } catch (error) {
      console.error(error);
      setMessage("Error while analyzing video.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: "40px", fontFamily: "Arial, sans-serif" }}>
      <h1>Video Summarizer</h1>
      <p>Visual analysis prototype for video summarization.</p>

      <section style={{ marginTop: "30px" }}>
        <h2>1. Video Upload</h2>
        <input type="file" accept="video/*" onChange={handleVideoChange} />

        {selectedVideo && (
          <p style={{ marginTop: "10px" }}>
            Selected video: {selectedVideo.name}
          </p>
        )}

        {videoUrl && (
          <video
            src={videoUrl}
            controls
            width="500"
            style={{ display: "block", marginTop: "20px", borderRadius: "8px" }}
          />
        )}

        <button
          onClick={handleAnalyze}
          disabled={loading}
          style={{
            marginTop: "20px",
            padding: "10px 20px",
            cursor: "pointer",
            border: "none",
            borderRadius: "6px",
          }}
        >
          {loading ? "Analyzing..." : "Analyze Video"}
        </button>

        {message && <p style={{ marginTop: "15px" }}>{message}</p>}
      </section>

      <section style={{ marginTop: "30px" }}>
        <h2>2. Extraction Result</h2>
        {result?.extraction_result ? (
          <div>
            <p>Total comparisons: {result.extraction_result.total_comparisons}</p>
            <p>Kept frames: {result.extraction_result.kept_frames}</p>
            <p>Ignored frames: {result.extraction_result.ignored_frames}</p>
            <p>Frames folder: {result.extraction_result.frames_folder}</p>
          </div>
        ) : (
          <p>Extraction results will appear here.</p>
        )}
      </section>

      <section style={{ marginTop: "30px" }}>
        <h2>3. Captions Result</h2>
        {result?.caption_result ? (
          <div>
            <p>Frames captioned: {result.caption_result.frames_captioned}</p>
            <p>Frames skipped: {result.caption_result.frames_skipped}</p>
            <p>Output file: {result.caption_result.output_file}</p>

            <div style={{ marginTop: "20px" }}>
              <h3>Generated Captions</h3>
              {result.caption_result.captions.map((item, index) => (
                <div
                  key={index}
                  style={{
                    padding: "10px",
                    marginBottom: "10px",
                    border: "1px solid #ddd",
                    borderRadius: "8px",
                  }}
                >
                  <strong>{item.frame}</strong>
                  <p style={{ marginTop: "8px" }}>{item.caption}</p>
                </div>
              ))}
            </div>
          </div>
        ) : (
          <p>Captions will appear here.</p>
        )}
      </section>
    </div>
  );
}

export default App;
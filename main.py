from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import shutil
import uuid
import threading

from extract_frames import extract_frames
from caption_frames import caption_frames
from whisper_transcribe import transcribe_audio_streaming
from gemma_summary import summarize_video_streaming

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve extracted frames as static files so the frontend can display them.
os.makedirs("frames", exist_ok=True)
app.mount("/frames", StaticFiles(directory="frames"), name="frames")

jobs = {}


@app.get("/")
def read_root():
    return {"message": "Backend is working"}


def _empty_partial():
    """Initial shape of partial_result, every field present so the frontend
    can render placeholders without crashing on undefined."""
    return {
        "stage": "uploading",          # uploading | extracting | captioning | transcribing | summarizing | done
        "language": "en",
        "video_filename": None,
        "extraction_result": None,
        "frames": [],                  # [{file, frame, timestamp_sec}]
        "captions": [],                # streamed: [{frame, caption, timestamp_sec, index}]
        "transcript": {
            "language": None,
            "language_probability": None,
            "duration_sec": None,
            "segments": [],            # streamed
        },
        "final_summary": "",           # grows char by char during streaming
        "final_summary_done": False,
    }


def process_video(job_id, file_path, filename, language):
    state = jobs[job_id]
    state["result"] = _empty_partial()
    state["result"]["language"] = language
    state["result"]["video_filename"] = filename

    try:
        # ---- 1. Extract frames ---------------------------------------
        state["result"]["stage"] = "extracting"
        state["steps"] = ["✅ Video uploaded", "⏳ Extracting frames..."]
        ex = extract_frames(file_path)

        state["result"]["extraction_result"] = {
            "frames_folder": ex.get("frames_folder", "frames"),
            "kept_frames": ex.get("kept_frames", 0),
            "ignored_frames": (
                ex.get("skipped_blurry", 0)
                + ex.get("skipped_brightness", 0)
                + ex.get("skipped_duplicates", 0)
            ),
            "total_comparisons": ex.get("ranges_processed", 0),
            "duration_sec": ex.get("duration_sec", 0),
            "scenes_detected": ex.get("scenes_detected", 0),
            "fps": ex.get("fps", 0),
        }
        # Pre-populate frames list (captions will arrive later)
        state["result"]["frames"] = [
            {
                "file": f["file"],
                "frame": os.path.basename(f["file"]),
                "timestamp_sec": f.get("timestamp_sec"),
            }
            for f in ex.get("frames", [])
        ]

        # ---- 2. Visual captions (streamed) ---------------------------
        state["result"]["stage"] = "captioning"
        state["steps"] = [
            "✅ Video uploaded",
            "✅ Frames extracted",
            "⏳ Generating captions...",
        ]

        def on_caption(entry):
            if not entry.get("ok"):
                return
            state["result"]["captions"].append({
                "index": entry["index"],
                "frame": os.path.basename(entry["file"]),
                "caption": entry["caption"],
                "timestamp_sec": entry.get("timestamp_sec"),
            })

        caption_frames(
            frames_metadata=ex.get("frames"),
            frames_folder=ex.get("frames_folder", "frames"),
            on_caption=on_caption,
        )

        # ---- 3. Audio transcription (streamed) -----------------------
        state["result"]["stage"] = "transcribing"
        state["steps"] = [
            "✅ Video uploaded",
            "✅ Frames extracted",
            "✅ Captions generated",
            "⏳ Transcribing audio...",
        ]

        def on_info(info):
            state["result"]["transcript"]["language"] = info["language"]
            state["result"]["transcript"]["language_probability"] = info["language_probability"]
            state["result"]["transcript"]["duration_sec"] = info["duration_sec"]

        def on_segment(seg):
            state["result"]["transcript"]["segments"].append(seg)

        try:
            tr_result = transcribe_audio_streaming(
                file_path,
                on_info=on_info,
                on_segment=on_segment,
            )
            # Persist a transcript.txt for Gemma consumption
            with open("transcript.txt", "w", encoding="utf-8") as f:
                for seg in tr_result["segments"]:
                    f.write(f"[{seg['start']:.2f}s] {seg['text']}\n")
        except Exception as e:
            state["result"]["transcript"]["error"] = str(e)

        # ---- 4. Final summary (streamed token by token) --------------
        state["result"]["stage"] = "summarizing"
        state["steps"] = [
            "✅ Video uploaded",
            "✅ Frames extracted",
            "✅ Captions generated",
            "✅ Audio transcribed",
            "⏳ Generating final summary...",
        ]

        captions_text = ""
        if os.path.exists("captions.txt"):
            with open("captions.txt", "r", encoding="utf-8") as f:
                captions_text = f.read()

        transcript_text = ""
        if os.path.exists("transcript.txt"):
            with open("transcript.txt", "r", encoding="utf-8") as f:
                transcript_text = f.read()

        def on_token(text_so_far):
            state["result"]["final_summary"] = text_so_far

        summarize_video_streaming(
            captions_text=captions_text,
            transcript_text=transcript_text,
            language=language,
            on_token=on_token,
        )
        state["result"]["final_summary_done"] = True
        state["result"]["stage"] = "done"

        state["steps"] = [
            "✅ Video uploaded",
            "✅ Frames extracted",
            "✅ Captions generated",
            "✅ Audio transcribed",
            "✅ Final summary generated",
            "✅ Analysis complete",
        ]
        state["status"] = "completed"

    except Exception as e:
        state["status"] = "failed"
        state["steps"].append("❌ Analysis failed")
        state["error"] = str(e)


@app.post("/analyze-video")
async def analyze_video(
    video: UploadFile = File(...),
    language: str = Form("en"),
):
    if language not in ("en", "fr"):
        language = "en"

    upload_folder = "uploads"
    os.makedirs(upload_folder, exist_ok=True)
    file_path = os.path.join(upload_folder, video.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "processing",
        "steps": ["⏳ Uploading video..."],
        "result": _empty_partial(),
        "error": None,
    }
    jobs[job_id]["result"]["language"] = language
    jobs[job_id]["result"]["video_filename"] = video.filename

    thread = threading.Thread(
        target=process_video,
        args=(job_id, file_path, video.filename, language),
    )
    thread.start()

    return {"job_id": job_id, "language": language}


@app.get("/job-status/{job_id}")
def get_job_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return {"error": "Job not found"}
    return job
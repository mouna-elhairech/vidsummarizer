from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import shutil
import uuid
import threading

import cv2

from extract_frames import extract_frames
from caption_frames import caption_frames
from ocr_frames import ocr_frames
from whisper_transcribe import transcribe_audio_streaming
from gemma_summary import summarize_video_streaming
from youtube_download import download_youtube

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("frames", exist_ok=True)
app.mount("/frames", StaticFiles(directory="frames"), name="frames")

jobs = {}
MAX_DURATION_SEC = 15 * 60


@app.get("/")
def read_root():
    return {"message": "Backend is working"}


def get_video_duration(path):
    try:
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        cap.release()
        if fps > 0:
            return frames / fps
    except Exception:
        pass
    return 0


def _empty_partial():
    return {
        "stage": "uploading",
        "language": "en",
        "video_filename": None,
        "extraction_result": None,
        "frames": [],
        "captions": [],
        "ocr_results": [],           # NEW
        "ocr_combined": "",          # NEW
        "transcript": {
            "language": None,
            "language_probability": None,
            "duration_sec": None,
            "segments": [],
        },
        "final_summary": "",
        "final_summary_done": False,
        "summary_progress": "",
    }


def process_video(job_id, file_path, filename, language):
    state = jobs[job_id]
    state["result"] = _empty_partial()
    state["result"]["language"] = language
    state["result"]["video_filename"] = filename

    try:
        # ---- Duration guard ------------------------------------------
        duration = get_video_duration(file_path)
        if duration and duration > MAX_DURATION_SEC:
            state["status"] = "failed"
            state["error"] = (
                f"Video is {int(duration // 60)} min, which exceeds the "
                f"{MAX_DURATION_SEC // 60} min limit."
            )
            state["steps"] = ["❌ Video too long"]
            return

        # ---- 1. Extract frames ---------------------------------------
        state["result"]["stage"] = "extracting"
        state["steps"] = ["✅ Video ready", "⏳ Extracting frames..."]
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
        state["steps"] = ["✅ Video ready", "✅ Frames extracted",
                          "⏳ Generating captions..."]

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

        # ---- 3. OCR (streamed) ---------------------------------------
        state["result"]["stage"] = "ocr"
        state["steps"] = ["✅ Video ready", "✅ Frames extracted",
                          "✅ Captions generated", "⏳ Reading on-screen text..."]

        def on_ocr(entry):
            state["result"]["ocr_results"].append({
                "frame": entry["frame"],
                "timestamp_sec": entry.get("timestamp_sec"),
                "text": entry["text"],
            })
            state["result"]["ocr_combined"] = " ".join(
                r["text"] for r in state["result"]["ocr_results"]
            )

        ocr_result = ocr_frames(
            frames_metadata=ex.get("frames"),
            frames_folder=ex.get("frames_folder", "frames"),
            on_ocr=on_ocr,
        )

        # ---- 4. Audio transcription (streamed) -----------------------
        state["result"]["stage"] = "transcribing"
        state["steps"] = ["✅ Video ready", "✅ Frames extracted",
                          "✅ Captions generated", "✅ Text read",
                          "⏳ Transcribing audio..."]

        def on_info(info):
            state["result"]["transcript"]["language"] = info["language"]
            state["result"]["transcript"]["language_probability"] = info["language_probability"]
            state["result"]["transcript"]["duration_sec"] = info["duration_sec"]

        def on_segment(seg):
            state["result"]["transcript"]["segments"].append(seg)

        try:
            tr_result = transcribe_audio_streaming(
                file_path, on_info=on_info, on_segment=on_segment,
            )
            with open("transcript.txt", "w", encoding="utf-8") as f:
                for seg in tr_result["segments"]:
                    f.write(f"[{seg['start']:.2f}s] {seg['text']}\n")
        except Exception as e:
            state["result"]["transcript"]["error"] = str(e)

        # ---- 5. Final summary (streamed) -----------------------------
        state["result"]["stage"] = "summarizing"
        state["steps"] = ["✅ Video ready", "✅ Frames extracted",
                          "✅ Captions generated", "✅ Text read",
                          "✅ Audio transcribed", "⏳ Generating final summary..."]

        captions_text = ""
        if os.path.exists("captions.txt"):
            with open("captions.txt", "r", encoding="utf-8") as f:
                captions_text = f.read()

        transcript_text = ""
        if os.path.exists("transcript.txt"):
            with open("transcript.txt", "r", encoding="utf-8") as f:
                transcript_text = f.read()

        # Inject OCR text into captions so Gemma benefits from screen text
        ocr_combined = ocr_result.get("combined_text", "").strip()
        if ocr_combined:
            captions_text += f"\n\nON-SCREEN TEXT:\n{ocr_combined}"

        def on_token(text_so_far):
            state["result"]["final_summary"] = text_so_far

        def on_progress(msg):
            state["result"]["summary_progress"] = msg

        summarize_video_streaming(
            captions_text=captions_text,
            transcript_text=transcript_text,
            language=language,
            on_token=on_token,
            on_progress=on_progress,
        )

        state["result"]["final_summary_done"] = True
        state["result"]["stage"] = "done"
        state["steps"] = [
            "✅ Video ready", "✅ Frames extracted",
            "✅ Captions generated", "✅ Text read",
            "✅ Audio transcribed", "✅ Final summary generated",
            "✅ Analysis complete",
        ]
        state["status"] = "completed"

    except Exception as e:
        state["status"] = "failed"
        state["steps"].append("❌ Analysis failed")
        state["error"] = str(e)


def process_youtube(job_id, url, language):
    state = jobs[job_id]
    try:
        state["steps"] = ["⏳ Fetching YouTube video..."]
        info = download_youtube(url, output_dir="uploads",
                                max_duration_sec=MAX_DURATION_SEC)
        file_path = info["file_path"]
        title = info.get("title") or info["filename"]
    except ValueError as e:
        state["status"] = "failed"
        state["error"] = str(e)
        state["steps"] = ["❌ Video too long"]
        return
    except Exception as e:
        state["status"] = "failed"
        state["error"] = f"YouTube download failed: {e}"
        state["steps"] = ["❌ YouTube download failed"]
        return
    process_video(job_id, file_path, title, language)


@app.post("/analyze-video")
async def analyze_video(
    video: UploadFile = File(...),
    language: str = Form("en"),
):
    if language not in ("en", "fr"):
        language = "en"
    os.makedirs("uploads", exist_ok=True)
    file_path = os.path.join("uploads", video.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "processing", "steps": ["⏳ Uploading video..."],
                    "result": _empty_partial(), "error": None}
    jobs[job_id]["result"]["language"] = language
    jobs[job_id]["result"]["video_filename"] = video.filename

    threading.Thread(target=process_video,
                     args=(job_id, file_path, video.filename, language)).start()
    return {"job_id": job_id, "language": language}


@app.post("/analyze-youtube")
async def analyze_youtube(
    url: str = Form(...),
    language: str = Form("en"),
):
    if language not in ("en", "fr"):
        language = "en"
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "processing", "steps": ["⏳ Fetching YouTube video..."],
                    "result": _empty_partial(), "error": None}
    jobs[job_id]["result"]["language"] = language

    threading.Thread(target=process_youtube,
                     args=(job_id, url, language)).start()
    return {"job_id": job_id, "language": language}


@app.get("/job-status/{job_id}")
def get_job_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return {"error": "Job not found"}
    return job
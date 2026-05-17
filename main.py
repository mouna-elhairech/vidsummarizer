from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import uuid
import threading

from extract_frames import extract_frames
from caption_frames import caption_frames
from gemma_summary import summarize_captions_with_gemma

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

jobs = {}

@app.get("/")
def read_root():
    return {"message": "Backend is working"}

def process_video(job_id, file_path, filename):
    try:
        jobs[job_id]["steps"] = [
            "✅ Video selected",
            "✅ Video uploaded",
            "⏳ Extracting frames..."
        ]

        extraction_result = extract_frames(file_path)

        jobs[job_id]["steps"] = [
            "✅ Video selected",
            "✅ Video uploaded",
            "✅ Frames extracted",
            "⏳ Generating captions..."
        ]

        caption_result = caption_frames("frames")

        jobs[job_id]["steps"] = [
            "✅ Video selected",
            "✅ Video uploaded",
            "✅ Frames extracted",
            "✅ Captions generated",
            "⏳ Generating final summary..."
        ]

        with open("captions.txt", "r", encoding="utf-8") as f:
            captions_text = f.read()

        final_summary = summarize_captions_with_gemma(captions_text)

        jobs[job_id]["steps"] = [
            "✅ Video selected",
            "✅ Video uploaded",
            "✅ Frames extracted",
            "✅ Captions generated",
            "✅ Final summary generated",
            "✅ Analysis complete"
        ]

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["result"] = {
            "message": "Video analyzed successfully",
            "filename": filename,
            "saved_path": file_path,
            "extraction_result": extraction_result,
            "caption_result": caption_result,
            "final_summary": final_summary
        }

    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["steps"].append("❌ Analysis failed")
        jobs[job_id]["error"] = str(e)

@app.post("/analyze-video")
async def analyze_video(video: UploadFile = File(...)):
    upload_folder = "uploads"
    os.makedirs(upload_folder, exist_ok=True)

    file_path = os.path.join(upload_folder, video.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    job_id = str(uuid.uuid4())

    jobs[job_id] = {
        "status": "processing",
        "steps": [
            "✅ Video selected",
            "⏳ Uploading video..."
        ],
        "result": None,
        "error": None
    }

    thread = threading.Thread(target=process_video, args=(job_id, file_path, video.filename))
    thread.start()

    return {
        "job_id": job_id,
        "message": "Analysis started"
    }

@app.get("/job-status/{job_id}")
def get_job_status(job_id: str):
    job = jobs.get(job_id)

    if not job:
        return {"error": "Job not found"}

    return job
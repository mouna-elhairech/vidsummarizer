from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
from extract_frames import extract_frames
from caption_frames import caption_frames

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Backend is working"}

@app.post("/analyze-video")
async def analyze_video(video: UploadFile = File(...)):
    upload_folder = "uploads"
    os.makedirs(upload_folder, exist_ok=True)

    file_path = os.path.join(upload_folder, video.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    extraction_result = extract_frames(file_path)
    caption_result = caption_frames("frames")

    return {
        "message": "Video uploaded, frames extracted, and captions generated successfully",
        "filename": video.filename,
        "saved_path": file_path,
        "extraction_result": extraction_result,
        "caption_result": caption_result
    }
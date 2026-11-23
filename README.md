# VidSummarizer.ai

AI video summarizer: give it a video (file upload or YouTube link) and it writes a summary by combining what is **seen**, **read** and **heard**. Everything runs locally, with no cloud API.

## How it works

1. **Key frames**: scene detection with OpenCV and PySceneDetect picks the important frames.
2. **Visual description**: Florence-2 describes each frame.
3. **On-screen text**: Tesseract OCR reads the text shown in the video.
4. **Audio**: Faster-Whisper transcribes the speech.
5. **Summary**: Gemma 3 4B, served by Ollama, merges captions, text and transcript into one summary.

## Tech stack

- **Backend**: Python, FastAPI, Uvicorn
- **Vision**: OpenCV, PySceneDetect, Florence-2 (Hugging Face Transformers, PyTorch), Tesseract OCR
- **Audio**: Faster-Whisper
- **LLM**: Gemma 3 4B via Ollama
- **Video input**: yt-dlp for YouTube links
- **Frontend**: React, Vite

## Getting started

Prerequisites: Python 3.12, Node.js, [Ollama](https://ollama.com) and [Tesseract OCR](https://github.com/tesseract-ocr/tesseract).

Backend:

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ollama pull gemma3:4b
    uvicorn main:app --port 8000

Frontend, in a second terminal:

    cd frontend
    npm install
    npm run dev

Then open http://localhost:5173 in your browser.

## Project structure

    main.py                FastAPI app and API routes
    extract_frames.py      Scene detection and key frame extraction
    caption_frames.py      Frame captioning pipeline
    florence_caption.py    Florence-2 image captioning
    ocr_frames.py          On-screen text extraction
    whisper_transcribe.py  Speech transcription
    gemma_summary.py       Summary generation with Ollama
    youtube_download.py    YouTube video download
    frontend/              React and Vite interface
    experiments/           Model comparisons (BLIP-2, Moondream, Qwen, Gemma) and tests

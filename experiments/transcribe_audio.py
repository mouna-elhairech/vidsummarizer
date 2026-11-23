"""
Run faster-whisper on a video file and persist the transcript.

Outputs two files:
  - transcript.txt   : timestamped, one segment per line (input for Gemma)
  - transcript.json  : structured transcript + metadata (for frontend / debug)
"""
import os
import json
import time

from whisper_transcribe import transcribe_audio


def transcribe_video(
    video_path,
    output_txt="transcript.txt",
    output_json="transcript.json",
    language=None,
    verbose=True,
):
    if not os.path.exists(video_path):
        return {
            "success": False,
            "error": f"Video not found: {video_path}",
        }

    if verbose:
        print(f"[audio] Transcribing {video_path}...")

    start = time.time()
    try:
        result = transcribe_audio(video_path, language=language)
    except Exception as e:
        return {
            "success": False,
            "error": f"Transcription failed: {e}",
        }
    elapsed = time.time() - start

    # Write timestamped txt for summarization
    with open(output_txt, "w", encoding="utf-8") as f:
        for seg in result["segments"]:
            f.write(f"[{seg['start']:.2f}s] {seg['text']}\n")

    # Write structured json for the frontend
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    seg_count = len(result["segments"])

    if verbose:
        print(f"[audio] Detected {result['language']} "
              f"({result['language_probability']*100:.1f}%), "
              f"{seg_count} segments in {elapsed:.1f}s")

    return {
        "success": True,
        "elapsed_sec": round(elapsed, 2),
        "language": result["language"],
        "language_probability": result["language_probability"],
        "duration_sec": result["duration_sec"],
        "text": result["text"],
        "segments": result["segments"],
        "segments_count": seg_count,
        "output_txt": output_txt,
        "output_json": output_json,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python transcribe_audio.py <video_path>")
        sys.exit(1)

    result = transcribe_video(sys.argv[1])
    print("\nResult:")
    for k, v in result.items():
        if k == "segments":
            print(f"  {k}: ({len(v)} items)")
        elif k == "text" and isinstance(v, str) and len(v) > 100:
            print(f"  {k}: {v[:100]}...")
        else:
            print(f"  {k}: {v}")
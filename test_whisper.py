"""
Quick benchmark for faster-whisper base on a video file.
Tests speed + quality + language detection.
"""
import sys
import time

from faster_whisper import WhisperModel


MODEL_NAME = "base"  # 74M params, multilingual, good speed/quality balance
COMPUTE_TYPE = "int8"  # quantized — much faster on CPU


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_whisper.py <video_or_audio_path>")
        print("Example: python test_whisper.py uploads/butterflyy.mp4")
        sys.exit(1)

    video_path = sys.argv[1]

    print(f"Loading faster-whisper '{MODEL_NAME}'...")
    load_start = time.time()
    model = WhisperModel(MODEL_NAME, device="cpu", compute_type=COMPUTE_TYPE)
    print(f"Loaded in {time.time() - load_start:.1f}s")
    print(f"(first run downloads ~150MB to ~/.cache/huggingface/, then cached)\n")

    print(f"Transcribing {video_path}...")
    print("(VAD filter ON, beam_size=5 — quality settings)\n")

    start = time.time()
    segments_gen, info = model.transcribe(
        video_path,
        vad_filter=True,                  # strip silences -> no hallucinations on quiet parts
        condition_on_previous_text=False, # avoid error propagation between segments
        beam_size=5,                      # better quality decoding
    )

    segments = []
    for seg in segments_gen:
        segments.append({
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": seg.text.strip(),
        })
    elapsed = time.time() - start

    print(f"Detected language: {info.language} (confidence {info.language_probability:.2%})")
    print(f"Duration of audio: {info.duration:.1f}s")
    print(f"Transcription time: {elapsed:.1f}s "
          f"(speed: {info.duration / max(elapsed, 0.01):.1f}x realtime)\n")

    print("--- Transcript ---")
    if not segments:
        print("(no speech detected)")
    for seg in segments:
        print(f"[{seg['start']:.2f}s -> {seg['end']:.2f}s] {seg['text']}")
    print("------------------")


if __name__ == "__main__":
    main()
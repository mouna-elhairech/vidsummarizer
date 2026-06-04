"""
faster-whisper transcription module with streaming support.

Two entry points:
  transcribe_audio(media_path, language)            -> blocking, returns all at once
  transcribe_audio_streaming(media_path, language,  -> streams segments as they come
                             on_info, on_segment)      via callbacks
"""
from faster_whisper import WhisperModel


MODEL_NAME = "base"
COMPUTE_TYPE = "int8"

_model = None


def load_model():
    global _model
    if _model is not None:
        return _model
    _model = WhisperModel(MODEL_NAME, device="cpu", compute_type=COMPUTE_TYPE)
    return _model


def _segments_and_info(media_path, language):
    model = load_model()
    return model.transcribe(
        media_path,
        language=language,
        vad_filter=True,
        condition_on_previous_text=False,
        beam_size=5,
    )


def transcribe_audio(media_path, language=None):
    """Blocking transcription, returns everything at once."""
    segments_gen, info = _segments_and_info(media_path, language)

    segments = []
    text_parts = []
    for seg in segments_gen:
        text = seg.text.strip()
        segments.append({
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": text,
        })
        text_parts.append(text)

    return {
        "language": info.language,
        "language_probability": round(info.language_probability, 3),
        "duration_sec": round(info.duration, 2),
        "text": " ".join(text_parts).strip(),
        "segments": segments,
    }


def transcribe_audio_streaming(media_path, language=None, on_info=None, on_segment=None):
    """
    Streaming transcription: fires callbacks as info and segments become available.

    Args:
        on_info(info_dict): called once after language detection.
            info_dict = {language, language_probability, duration_sec}
        on_segment(segment_dict): called for each segment as it's produced.
            segment_dict = {start, end, text}

    Returns the same dict shape as transcribe_audio() at the end.
    """
    segments_gen, info = _segments_and_info(media_path, language)

    info_dict = {
        "language": info.language,
        "language_probability": round(info.language_probability, 3),
        "duration_sec": round(info.duration, 2),
    }
    if on_info is not None:
        try:
            on_info(info_dict)
        except Exception:
            pass

    segments = []
    text_parts = []
    for seg in segments_gen:
        text = seg.text.strip()
        seg_dict = {
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": text,
        }
        segments.append(seg_dict)
        text_parts.append(text)
        if on_segment is not None:
            try:
                on_segment(seg_dict)
            except Exception:
                pass

    return {
        **info_dict,
        "text": " ".join(text_parts).strip(),
        "segments": segments,
    }


if __name__ == "__main__":
    import sys
    import time
    if len(sys.argv) < 2:
        print("Usage: python whisper_transcribe.py <video_or_audio_path>")
        sys.exit(1)
    start = time.time()
    result = transcribe_audio(sys.argv[1])
    print(f"Language: {result['language']} ({result['language_probability']*100:.1f}%)")
    print(f"Duration: {result['duration_sec']}s, transcribed in {time.time()-start:.1f}s")
    for seg in result["segments"]:
        print(f"[{seg['start']:.2f}s] {seg['text']}")
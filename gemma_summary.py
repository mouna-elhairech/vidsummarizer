"""
Gemma 3 4B summarization via Ollama, with streaming and map-reduce
for long audio transcripts.

Strategy:
  - Strip timestamps from the transcript (Gemma doesn't need them; the
    captions already carry temporal order). Saves ~15-20% of tokens.
  - If the cleaned transcript is long, condense it in chunks first
    (map step), then summarize the condensed digest + captions (reduce step).
    This keeps every Gemma call's input small -> fast prefill on CPU,
    and never overflows the context window on talky videos.
"""
import re
import ollama

MODEL = "gemma3:4b"

LANGUAGE_NAMES = {"en": "English", "fr": "French"}

# Above this many words, the transcript is condensed via map-reduce.
TRANSCRIPT_WORD_THRESHOLD = 800
# Words per chunk in the map step.
CHUNK_WORDS = 700


def _strip_timestamps(text):
    """Remove leading [X.XXs] markers and collapse to plain prose."""
    cleaned = []
    for line in text.splitlines():
        line = re.sub(r"^\s*\[\d+(?:\.\d+)?s\]\s*", "", line).strip()
        if line:
            cleaned.append(line)
    return " ".join(cleaned)


def _chunk_words(text, max_words=CHUNK_WORDS):
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


def _build_system_prompt(language):
    lang = LANGUAGE_NAMES.get(language, "English")
    return (
        "You are a video summarizer. "
        "You will receive a list of timestamped captions extracted from frames "
        "of a video, in chronological order. "
        f"Write a clear, natural, factual summary of the video in {lang}. "
        "Use 2 to 4 sentences in flowing prose, not a list. "
        "Avoid meta-language like 'the video shows'. "
        "Do not invent details that are not present in the captions. "
        f"The output MUST be written in {lang} only."
    )


def _build_combined_system_prompt(language):
    lang = LANGUAGE_NAMES.get(language, "English")
    return (
        "You are a video summarizer. "
        "You will receive two sources extracted from a video: "
        "(1) VISUAL CAPTIONS — descriptions of frames (what is seen), "
        "(2) AUDIO — what is said in the video. "
        f"Write a clear, natural, factual summary of the video in {lang}. "
        "Synthesize visual and audio information into one coherent description; "
        "prefer the audio for spoken content and the visuals for actions. "
        "Use 3 to 6 sentences in flowing prose, not a list. "
        "Avoid meta-language like 'the video shows'. "
        "Do not invent details that are not present in the sources. "
        f"The output MUST be written in {lang} only."
    )


def _condense_chunk(chunk_text, language):
    """Map step: condense one transcript chunk into 1-2 sentences."""
    lang = LANGUAGE_NAMES.get(language, "English")
    system = (
        "You are condensing part of a video's audio transcript. "
        f"Summarize the following excerpt into 1 to 2 concise sentences in {lang}. "
        "Keep only the key spoken content. Do not add commentary or meta-language."
    )
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": chunk_text},
        ],
        options={"num_ctx": 2048, "num_predict": 120, "temperature": 0.2},
    )
    return response["message"]["content"].strip()


def summarize_video(captions_text, transcript_text="", language="en", temperature=0.3):
    """Blocking version (no streaming, no progress)."""
    return _run(captions_text, transcript_text, language, temperature, None, None)


def summarize_video_streaming(
    captions_text,
    transcript_text="",
    language="en",
    temperature=0.3,
    on_token=None,       # callback(text_so_far) during final generation
    on_progress=None,    # callback(message) during the map step
):
    """Streaming version with map-reduce + progress callbacks."""
    return _run(captions_text, transcript_text, language, temperature, on_token, on_progress)


def _run(captions_text, transcript_text, language, temperature, on_token, on_progress):
    cleaned = _strip_timestamps(transcript_text) if transcript_text else ""
    has_audio = bool(cleaned.strip())

    # ---- Map step: condense long transcripts -------------------------
    if has_audio and len(cleaned.split()) > TRANSCRIPT_WORD_THRESHOLD:
        chunks = _chunk_words(cleaned)
        digests = []
        for i, chunk in enumerate(chunks, start=1):
            if on_progress:
                on_progress(f"Condensing audio {i}/{len(chunks)}")
            digests.append(_condense_chunk(chunk, language))
        cleaned = " ".join(digests)
        if on_progress:
            on_progress("Writing final summary")

    # ---- Reduce step: final summary ----------------------------------
    if has_audio:
        system = _build_combined_system_prompt(language)
        user = (
            f"VISUAL CAPTIONS:\n{captions_text.strip()}\n\n"
            f"AUDIO:\n{cleaned.strip()}"
        )
    else:
        system = _build_system_prompt(language)
        user = captions_text

    options = {"num_ctx": 2048, "num_predict": 300, "temperature": temperature}

    # No streaming requested -> single blocking call
    if on_token is None:
        response = ollama.chat(
            model=MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            options=options,
        )
        return response["message"]["content"].strip()

    # Streaming
    full_text = ""
    for chunk in ollama.chat(
        model=MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        options=options,
        stream=True,
    ):
        delta = chunk.get("message", {}).get("content", "")
        if delta:
            full_text += delta
            try:
                on_token(full_text)
            except Exception:
                pass
    return full_text.strip()


# Backwards-compat alias
summarize_text = summarize_video

if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2:
        with open(sys.argv[1], encoding="utf-8") as f:
            text = f.read()
    else:
        text = sys.stdin.read()
    lang = sys.argv[2] if len(sys.argv) >= 3 else "en"
    print(summarize_video(text, language=lang))
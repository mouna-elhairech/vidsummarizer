"""
Gemma 3 4B summarization via Ollama, with optional streaming.
"""
import ollama

MODEL = "gemma3:4b"

LANGUAGE_NAMES = {
    "en": "English",
    "fr": "French",
}


def _build_system_prompt(language):
    lang_name = LANGUAGE_NAMES.get(language, "English")
    return (
        "You are a video summarizer. "
        "You will receive a list of timestamped captions extracted from frames "
        "of a video, in chronological order. "
        f"Write a clear, natural, factual summary of the video in {lang_name}. "
        "Use 2 to 4 sentences in flowing prose, not a list. "
        "Avoid meta-language like 'the video shows' or 'this clip depicts'. "
        "Do not invent details that are not present in the captions. "
        f"The output MUST be written in {lang_name} only."
    )


def _build_combined_system_prompt(language):
    lang_name = LANGUAGE_NAMES.get(language, "English")
    return (
        "You are a video summarizer. "
        "You will receive two timestamped sources extracted from a video, "
        "both in chronological order: "
        "(1) VISUAL CAPTIONS — descriptions of frames (what is seen), "
        "(2) AUDIO TRANSCRIPT — what is said in the video. "
        f"Write a clear, natural, factual summary of the video in {lang_name}. "
        "Synthesize visual and audio information into a single coherent description; "
        "prefer the audio for spoken content and the visuals for actions. "
        "Use 3 to 6 sentences in flowing prose, not a list. "
        "Avoid meta-language like 'the video shows' or 'this clip depicts'. "
        "Do not invent details that are not present in the sources. "
        f"The output MUST be written in {lang_name} only."
    )


def _build_messages(captions_text, transcript_text, language):
    has_audio = bool(transcript_text and transcript_text.strip())
    if has_audio:
        system = _build_combined_system_prompt(language)
        user = (
            "VISUAL CAPTIONS:\n"
            f"{captions_text.strip()}\n\n"
            "AUDIO TRANSCRIPT:\n"
            f"{transcript_text.strip()}"
        )
        num_ctx = 4096
        num_predict = 300
    else:
        system = _build_system_prompt(language)
        user = captions_text
        num_ctx = 2048
        num_predict = 200
    return system, user, num_ctx, num_predict


def summarize_video(captions_text, transcript_text="", language="en", temperature=0.3):
    """Blocking summary."""
    system, user, num_ctx, num_predict = _build_messages(captions_text, transcript_text, language)
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        options={
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "temperature": temperature,
        },
    )
    return response["message"]["content"].strip()


def summarize_video_streaming(
    captions_text,
    transcript_text="",
    language="en",
    temperature=0.3,
    on_token=None,
):
    """
    Streaming summary. on_token(text_so_far) is called repeatedly as more
    tokens arrive. Returns the final full text.
    """
    system, user, num_ctx, num_predict = _build_messages(captions_text, transcript_text, language)

    full_text = ""
    stream = ollama.chat(
        model=MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        options={
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "temperature": temperature,
        },
        stream=True,
    )

    for chunk in stream:
        delta = chunk.get("message", {}).get("content", "")
        if not delta:
            continue
        full_text += delta
        if on_token is not None:
            try:
                on_token(full_text)
            except Exception:
                pass

    return full_text.strip()


# Backwards compatibility
summarize_text = summarize_video  # alias

if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2:
        with open(sys.argv[1], encoding="utf-8") as f:
            text = f.read()
    else:
        text = sys.stdin.read()
    lang = sys.argv[2] if len(sys.argv) >= 3 else "en"
    print(summarize_video(text, language=lang))
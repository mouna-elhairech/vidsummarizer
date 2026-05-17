import requests
import re
from collections import Counter

STOP_WORDS = {
    "a", "an", "the", "in", "on", "at", "of", "with", "and", "is", "are",
    "this", "that", "it", "to", "for", "from", "by", "as", "into", "over",
    "under", "through", "around", "multiple", "one", "two"
}

def extract_keywords(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    words = text.split()
    return [word for word in words if word not in STOP_WORDS and len(word) > 2]

def filter_stable_captions(captions_text):
    lines = [line.strip() for line in captions_text.splitlines() if line.strip()]

    all_keywords = []
    line_keywords = []

    for line in lines:
        if " : " in line:
            caption = line.split(" : ", 1)[1].strip()
        else:
            caption = line

        keywords = extract_keywords(caption)
        line_keywords.append((line, keywords))
        all_keywords.extend(keywords)

    keyword_counts = Counter(all_keywords)

    filtered_lines = []

    for line, keywords in line_keywords:
        stable_keywords = [word for word in keywords if keyword_counts[word] >= 2]

        if stable_keywords:
            filtered_lines.append(line)

    if not filtered_lines:
        filtered_lines = lines

    return "\n".join(filtered_lines)

def summarize_captions_with_gemma(captions_text):
    filtered_captions = filter_stable_captions(captions_text)

    prompt = f"""
You are helping summarize the visual content of a video.

Below are descriptions extracted from important frames of the same video:

{filtered_captions}

Write one short, coherent, natural paragraph describing what the video shows.
Do not mention frame names.
Do not list disconnected phrases.
Merge the information into one smooth paragraph.
Ignore small inconsistent details if they appear only once.
"""

    payload = {
        "model": "gemma3:4b",
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(
        "http://127.0.0.1:11434/api/generate",
        json=payload,
        timeout=300
    )

    result = response.json()
    return result["response"].strip()
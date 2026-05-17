import os
import re
from qwen_caption import describe_image_with_qwen

def normalize_caption(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    words = text.split()
    stop_words = {"a", "an", "the", "in", "on", "at", "of", "with", "and"}
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

def similarity_score(caption1, caption2):
    words1 = set(normalize_caption(caption1).split())
    words2 = set(normalize_caption(caption2).split())

    if not words1 or not words2:
        return 0

    intersection = words1.intersection(words2)
    union = words1.union(words2)

    return len(intersection) / len(union)

def caption_frames(frames_folder="frames", output_file="captions.txt", similarity_threshold=0.7):
    files = os.listdir(frames_folder)
    files.sort()

    kept_count = 0
    skipped_count = 0
    captions_data = []
    kept_captions = []

    with open(output_file, "w", encoding="utf-8") as f:
        for file_name in files:
            frame_path = os.path.join(frames_folder, file_name)

            caption = describe_image_with_qwen(frame_path)

            is_duplicate = False
            for existing_caption in kept_captions:
                score = similarity_score(caption, existing_caption)
                if score >= similarity_threshold:
                    is_duplicate = True
                    break

            if is_duplicate:
                skipped_count += 1
                continue

            line = f"{file_name} : {caption}\n"
            f.write(line)

            captions_data.append({
                "frame": file_name,
                "caption": caption
            })

            kept_captions.append(caption)
            kept_count += 1

    return {
        "frames_captioned": kept_count,
        "frames_skipped": skipped_count,
        "output_file": output_file,
        "captions": captions_data
    }
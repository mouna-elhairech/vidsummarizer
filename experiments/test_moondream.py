import os
import io
import time
import base64
import ollama
from PIL import Image

FRAMES_FOLDER = "frames"
MODEL = "moondream"

MAX_EDGE = 768

# Moondream needs short, direct prompts.
# Long multi-sentence prompts cause it to return empty strings.
PROMPT = "Describe this image in one short sentence."


def resize_image_b64(frame_path, max_edge=MAX_EDGE):
    img = Image.open(frame_path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_edge:
        if w >= h:
            new_w = max_edge
            new_h = round(h * max_edge / w)
        else:
            new_h = max_edge
            new_w = round(w * max_edge / h)
        img = img.resize((new_w, new_h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii"), img.size


def caption_one(frame_path):
    image_b64, resized_size = resize_image_b64(frame_path)
    start = time.time()
    response = ollama.chat(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": PROMPT,
            "images": [image_b64],
        }],
        options={"num_ctx": 2048},
    )
    elapsed = time.time() - start
    caption = response["message"]["content"].strip()
    return caption, elapsed, resized_size


def main():
    frames = sorted(
        os.path.join(FRAMES_FOLDER, f)
        for f in os.listdir(FRAMES_FOLDER)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if not frames:
        print("No frames found. Run extract_frames.py first.")
        return

    print(f"Model:     {MODEL}")
    print(f"Prompt:    {PROMPT!r}")
    print(f"Frames:    {len(frames)}\n")

    total = 0
    for index, frame_path in enumerate(frames, start=1):
        print(f"[{index}/{len(frames)}] {os.path.basename(frame_path)}...")
        caption, elapsed, size = caption_one(frame_path)
        total += elapsed
        label = "(cold)" if index == 1 else "(warm)"
        print(f"  Sent at: {size[0]}x{size[1]}")
        print(f"  Time:    {elapsed:.1f}s {label}")
        print(f"  Caption: {caption!r}")
        print()

    print(f"Total: {total:.1f}s for {len(frames)} frames")


if __name__ == "__main__":
    main()
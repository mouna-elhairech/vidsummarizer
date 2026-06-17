"""
OCR on-screen text from extracted video frames using Tesseract.

Most video frames contain no readable text, so results are aggressively
filtered by confidence and word length to avoid feeding noise into the
summary. Supports an on_ocr callback for streaming partial results.
"""
import os
from PIL import Image

try:
    import pytesseract
    _HAS_TESSERACT = True
except ImportError:
    _HAS_TESSERACT = False


def ocr_image(path, lang="eng+fra", min_confidence=60, min_word_len=2):
    """
    Run OCR on a single image, keeping only confident, meaningful words.

    Falls back to English-only if the requested language pack is missing.
    """
    if not _HAS_TESSERACT:
        raise RuntimeError("pytesseract is not installed (pip install pytesseract)")

    img = Image.open(path).convert("RGB")

    try:
        data = pytesseract.image_to_data(
            img, lang=lang, output_type=pytesseract.Output.DICT
        )
    except pytesseract.TesseractError:
        # language pack probably missing -> retry English only
        data = pytesseract.image_to_data(
            img, lang="eng", output_type=pytesseract.Output.DICT
        )

    words = []
    for i in range(len(data["text"])):
        word = (data["text"][i] or "").strip()
        try:
            conf = float(data["conf"][i])
        except (ValueError, TypeError):
            conf = -1
        if word and conf >= min_confidence and len(word) >= min_word_len:
            words.append(word)

    return " ".join(words)


def ocr_frames(
    frames_metadata=None,
    frames_folder="frames",
    lang="eng+fra",
    min_text_len=4,    # a frame's text must be at least this long to count
    on_ocr=None,       # callback(entry) for frames that have text
    verbose=True,
):
    # Build the work list (same pattern as caption_frames) -----------
    if frames_metadata:
        frames = list(frames_metadata)
    else:
        if not os.path.exists(frames_folder):
            return {"success": False, "error": "frames folder not found",
                    "results": [], "combined_text": "", "frames_with_text": 0}
        files = sorted(
            f for f in os.listdir(frames_folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
        )
        frames = [{"file": os.path.join(frames_folder, n)} for n in files]

    results = []
    for index, frame in enumerate(frames, start=1):
        path = frame["file"]
        if verbose:
            print(f"[ocr {index}/{len(frames)}] {os.path.basename(path)}...")

        try:
            text = ocr_image(path, lang=lang)
        except Exception as e:
            if verbose:
                print(f"           [FAILED] {e}")
            text = ""

        keep = len(text) >= min_text_len
        entry = {
            "index": index,
            "frame": os.path.basename(path),
            "timestamp_sec": frame.get("timestamp_sec"),
            "text": text if keep else "",
            "has_text": keep,
        }
        results.append(entry)

        if verbose and keep:
            print(f"           {text}")

        if on_ocr is not None and keep:
            try:
                on_ocr(entry)
            except Exception:
                pass

    combined = " ".join(e["text"] for e in results if e["has_text"])

    return {
        "success": True,
        "results": results,
        "combined_text": combined,
        "frames_with_text": sum(1 for e in results if e["has_text"]),
        "total_frames": len(results),
    }


if __name__ == "__main__":
    import sys
    frames_metadata = None
    if len(sys.argv) >= 2:
        from extract_frames import extract_frames
        ex = extract_frames(sys.argv[1])
        frames_metadata = ex.get("frames")
    result = ocr_frames(frames_metadata=frames_metadata)
    print("\nOCR result:")
    print(f"  frames with text: {result['frames_with_text']}/{result['total_frames']}")
    print(f"  combined: {result['combined_text']}")
"""
Caption a set of video frames using Florence-2.

Supports an optional on_caption callback that fires after each frame is
captioned — useful for streaming partial results to a UI.
"""
import os
import json

from florence_caption import caption_image


def caption_frames(
    frames_metadata=None,
    frames_folder="frames",
    output_txt="captions.txt",
    output_json="captions.json",
    prompt_type="<CAPTION>",
    verbose=True,
    on_caption=None,    # callback(entry_dict) called after each successful caption
):
    if frames_metadata:
        frames = list(frames_metadata)
    else:
        if not os.path.exists(frames_folder):
            return {
                "success": False,
                "error": f"Frames folder not found: {frames_folder}",
                "frames_captioned": 0,
                "captions": [],
            }
        files = sorted(
            f for f in os.listdir(frames_folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
        )
        frames = [
            {
                "file": os.path.join(frames_folder, name),
                "timestamp_sec": None,
                "frame_num": None,
            }
            for name in files
        ]

    if not frames:
        return {
            "success": False,
            "error": "No frames to caption",
            "frames_captioned": 0,
            "captions": [],
        }

    captions = []
    failures = 0

    for index, frame in enumerate(frames, start=1):
        path = frame["file"]
        if verbose:
            print(f"[caption {index}/{len(frames)}] {os.path.basename(path)}...")

        try:
            text = caption_image(path, prompt_type=prompt_type)
            entry = {
                "index": index,
                "file": path,
                "timestamp_sec": frame.get("timestamp_sec"),
                "frame_num": frame.get("frame_num"),
                "caption": text,
                "ok": True,
            }
            if verbose:
                print(f"           {text}")
        except Exception as e:
            failures += 1
            entry = {
                "index": index,
                "file": path,
                "timestamp_sec": frame.get("timestamp_sec"),
                "frame_num": frame.get("frame_num"),
                "caption": "",
                "error": str(e),
                "ok": False,
            }
            if verbose:
                print(f"           [FAILED] {e}")

        captions.append(entry)

        if on_caption is not None:
            try:
                on_caption(entry)
            except Exception:
                pass  # callback should never break the pipeline

    with open(output_txt, "w", encoding="utf-8") as f:
        for c in captions:
            if not c["ok"]:
                continue
            ts = c["timestamp_sec"]
            if ts is not None:
                f.write(f"[{ts:.2f}s] {c['caption']}\n")
            else:
                f.write(f"{c['caption']}\n")

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(captions, f, indent=2, ensure_ascii=False)

    return {
        "success": True,
        "frames_captioned": sum(1 for c in captions if c["ok"]),
        "frames_failed": failures,
        "total": len(captions),
        "output_txt": output_txt,
        "output_json": output_json,
        "captions": captions,
    }


if __name__ == "__main__":
    frames_metadata = None
    try:
        from extract_frames import extract_frames
        import sys
        if len(sys.argv) >= 2:
            print(f"Extracting frames from {sys.argv[1]}...")
            ex = extract_frames(sys.argv[1])
            if ex.get("success"):
                frames_metadata = ex.get("frames")
                print(f"  kept {len(frames_metadata)} frames\n")
    except ImportError:
        pass

    result = caption_frames(frames_metadata=frames_metadata)
    print("\nCaption result:")
    for k, v in result.items():
        if k == "captions":
            continue
        print(f"  {k}: {v}")
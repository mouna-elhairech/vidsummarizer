"""
Read captions.txt produced by caption_frames.py and write a coherent
video summary to summary.txt using Gemma 3 4B.
"""
import os
import time

from gemma_summary import summarize_text


def summarize_captions(
    captions_file="captions.txt",
    output_file="summary.txt",
    verbose=True,
):
    """
    Generate a final video summary from a timestamped captions file.

    Args:
        captions_file: path to captions.txt (one timestamped caption per line).
        output_file: path where the summary will be saved.
        verbose: print progress.

    Returns:
        Result dict with success, summary, output_file, elapsed_sec.
    """
    if not os.path.exists(captions_file):
        return {
            "success": False,
            "error": f"Captions file not found: {captions_file}",
            "summary": "",
        }

    with open(captions_file, "r", encoding="utf-8") as f:
        captions_text = f.read().strip()

    if not captions_text:
        return {
            "success": False,
            "error": "Captions file is empty",
            "summary": "",
        }

    line_count = len(captions_text.splitlines())

    if verbose:
        print(f"[summary] Reading {captions_file} ({line_count} captions)")
        print("[summary] Generating with Gemma 3 4B...")

    start = time.time()
    try:
        summary = summarize_text(captions_text)
    except Exception as e:
        return {
            "success": False,
            "error": f"Gemma call failed: {e}",
            "summary": "",
        }
    elapsed = time.time() - start

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(summary + "\n")

    if verbose:
        print(f"[summary] Done in {elapsed:.1f}s\n")
        print("--- Summary ---")
        print(summary)
        print("---------------")

    return {
        "success": True,
        "captions_file": captions_file,
        "output_file": output_file,
        "elapsed_sec": round(elapsed, 2),
        "line_count": line_count,
        "summary": summary,
    }


if __name__ == "__main__":
    result = summarize_captions()
    print()
    if result["success"]:
        print(f"Summary saved to {result['output_file']}")
    else:
        print(f"Error: {result['error']}")
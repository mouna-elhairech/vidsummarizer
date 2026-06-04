import os
import shutil
import cv2
import numpy as np
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector


# ---------------------------------------------------------------------
# Image scoring functions
# ---------------------------------------------------------------------

def compute_frame_hash(frame, hash_size=16):
    """Perceptual hash for similarity comparison (dHash)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(
        gray,
        (hash_size + 1, hash_size),
        interpolation=cv2.INTER_AREA,
    )
    diff = resized[:, 1:] > resized[:, :-1]
    return diff.flatten()


def hamming_distance(hash1, hash2):
    """Number of differing bits between two hashes."""
    return np.count_nonzero(hash1 != hash2)


def compute_blur_score(frame):
    """Laplacian variance. Higher = sharper."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def compute_brightness_score(frame):
    """Mean brightness 0..255."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def is_bad_brightness(brightness, min_brightness=30, max_brightness=225):
    return brightness < min_brightness or brightness > max_brightness


# ---------------------------------------------------------------------
# Range helpers
# ---------------------------------------------------------------------

def merge_overlapping_ranges(ranges):
    """
    Merge ranges that overlap or are adjacent.
    Avoids processing the same part of the video twice when
    scene ranges and fallback ranges intersect.
    """
    if not ranges:
        return []

    sorted_ranges = sorted(ranges, key=lambda r: r[0])
    merged = [sorted_ranges[0]]

    for start, end in sorted_ranges[1:]:
        last_start, last_end = merged[-1]
        # Only merge actual overlaps (start inside previous range).
        # Adjacent ranges from fallback sampling must stay separate,
        # otherwise we collapse the whole video into one range.
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))

    return merged


def detect_scene_ranges(video_path, fps, threshold=27.0, min_scene_len_sec=1.0):
    """Detect scene ranges using PySceneDetect."""
    ranges = []
    scene_list = []

    try:
        video = open_video(video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(
            ContentDetector(
                threshold=threshold,
                min_scene_len=int(min_scene_len_sec * fps),
            )
        )
        scene_manager.detect_scenes(video)
        scene_list = scene_manager.get_scene_list()

        for scene in scene_list:
            start = scene[0].get_frames()
            end = scene[1].get_frames() - 1
            if end > start:
                ranges.append((start, end))

    except Exception as e:
        print(f"[WARNING] Scene detection failed: {e}")

    return ranges, scene_list


def build_fallback_ranges(total_frames, fps, fallback_interval_sec=2.5):
    """Regular time-based ranges, used when scenes are too few."""
    ranges = []
    step = max(1, int(fallback_interval_sec * fps))
    for start in range(0, total_frames, step):
        end = min(start + step - 1, total_frames - 1)
        if end > start:
            ranges.append((start, end))
    return ranges


# ---------------------------------------------------------------------
# Candidate picking
# ---------------------------------------------------------------------

def pick_best_frame_in_range(cap, start_frame, end_frame, sample_count=5):
    """Sample a few frames in a range and return the sharpest one."""
    if end_frame <= start_frame:
        return None

    scene_length = end_frame - start_frame + 1
    margin = max(1, scene_length // 10)

    sample_start = start_frame + margin
    sample_end = end_frame - margin
    if sample_end <= sample_start:
        sample_start = start_frame
        sample_end = end_frame

    candidates = np.linspace(
        sample_start,
        sample_end,
        num=min(sample_count, max(1, sample_end - sample_start + 1)),
        dtype=int,
    )

    best = None

    for frame_num in candidates:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_num))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        candidate = {
            "frame_num": int(frame_num),
            "frame": frame,
            "blur_score": float(compute_blur_score(frame)),
            "brightness": float(compute_brightness_score(frame)),
        }

        if best is None or candidate["blur_score"] > best["blur_score"]:
            best = candidate

    return best


# ---------------------------------------------------------------------
# Adaptive filtering
# ---------------------------------------------------------------------

def compute_adaptive_blur_threshold(candidates, ratio=0.4, hard_floor=20.0):
    """
    Resolution-aware blur threshold.

    Laplacian variance scales with resolution: a sharp 480p frame can
    score ~60 while a sharp 1080p frame scores ~300. A fixed threshold
    over-rejects low-resolution videos.

    Strategy: take the median blur of all raw candidates and use a
    fraction of it. The hard_floor catches genuinely blurry frames
    even when the entire video is soft.
    """
    if not candidates:
        return hard_floor
    scores = [c["blur_score"] for c in candidates]
    median = float(np.median(scores))
    return max(hard_floor, median * ratio)


def remove_duplicate_candidates(candidates, similarity_threshold=18):
    """Remove visually similar frames using perceptual hash."""
    skipped = 0

    candidates = sorted(candidates, key=lambda c: c["blur_score"], reverse=True)
    kept_hashes = []
    final = []

    for candidate in candidates:
        frame_hash = compute_frame_hash(candidate["frame"])
        is_dup = any(
            hamming_distance(frame_hash, old) < similarity_threshold
            for old in kept_hashes
        )
        if is_dup:
            skipped += 1
            continue
        kept_hashes.append(frame_hash)
        final.append(candidate)

    final = sorted(final, key=lambda c: c["frame_num"])
    return final, skipped


def select_balanced_final_frames(candidates, total_frames, max_frames=6):
    """Pick max_frames spread across the video timeline."""
    if len(candidates) <= max_frames:
        return sorted(candidates, key=lambda c: c["frame_num"])

    selected = []
    zone_edges = np.linspace(0, total_frames, num=max_frames + 1, dtype=int)

    for i in range(max_frames):
        zone_start = zone_edges[i]
        zone_end = zone_edges[i + 1]
        zone_candidates = [
            c for c in candidates
            if zone_start <= c["frame_num"] < zone_end
        ]
        if zone_candidates:
            best_in_zone = max(zone_candidates, key=lambda c: c["blur_score"])
            selected.append(best_in_zone)

    # Fill empty zones with best unused candidates
    if len(selected) < max_frames:
        used = {c["frame_num"] for c in selected}
        remaining = sorted(
            [c for c in candidates if c["frame_num"] not in used],
            key=lambda c: c["blur_score"],
            reverse=True,
        )
        for c in remaining:
            if len(selected) >= max_frames:
                break
            selected.append(c)

    return sorted(selected, key=lambda c: c["frame_num"])[:max_frames]


# ---------------------------------------------------------------------
# Main extraction pipeline
# ---------------------------------------------------------------------

def extract_frames(
    video_path,
    output_folder="frames",
    threshold=27.0,
    min_scene_len_sec=1.0,
    fallback_interval_sec=2.5,
    similarity_threshold=18,
    blur_ratio=0.4,
    blur_hard_floor=20.0,
    min_brightness=30,
    max_brightness=225,
    max_frames=6,
    min_frames=4,
    sample_count=5,
    verbose=False,
):
    """
    Pipeline:
      1. Detect scene ranges (PySceneDetect).
      2. Add time-based fallback ranges if scenes are too few.
      3. Merge overlapping ranges.
      4. For each range, pick the sharpest frame (raw candidate).
      5. Adaptive blur filter (resolution-aware).
      6. Brightness filter.
      7. Perceptual-hash dedup.
      8. Balance final frames across timeline.
      9. Safety fallback: never return zero frames.
     10. Save final frames for Moondream captioning.
    """

    if not os.path.exists(video_path):
        return {
            "success": False,
            "error": f"Video not found: {video_path}",
            "frames_folder": output_folder,
        }

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            "success": False,
            "error": "Could not open video",
            "frames_folder": output_folder,
        }

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration_sec = total_frames / fps if fps else 0

    if total_frames <= 0:
        cap.release()
        return {
            "success": False,
            "error": "Video has no readable frames",
            "frames_folder": output_folder,
        }

    # 1. Scene detection
    scene_ranges, scene_list = detect_scene_ranges(
        video_path=video_path,
        fps=fps,
        threshold=threshold,
        min_scene_len_sec=min_scene_len_sec,
    )
    ranges = list(scene_ranges)

    # 2. Fallback ranges if needed
    if len(ranges) < min_frames:
        ranges.extend(
            build_fallback_ranges(
                total_frames=total_frames,
                fps=fps,
                fallback_interval_sec=fallback_interval_sec,
            )
        )

    # 3. Merge overlapping ranges
    ranges = merge_overlapping_ranges(ranges)

    if verbose:
        print(f"[extract] {len(scene_list)} scenes -> {len(ranges)} merged ranges")

    # 4. Raw candidates (no filtering yet, so blur threshold can be adaptive)
    raw_candidates = []
    for start_frame, end_frame in ranges:
        candidate = pick_best_frame_in_range(
            cap=cap,
            start_frame=start_frame,
            end_frame=end_frame,
            sample_count=sample_count,
        )
        if candidate is None:
            continue
        candidate["timestamp_sec"] = round(candidate["frame_num"] / fps, 2)
        raw_candidates.append(candidate)

    cap.release()

    if verbose:
        print(f"[extract] {len(raw_candidates)} raw candidates")

    # 5. Adaptive blur threshold + filtering
    adaptive_blur = compute_adaptive_blur_threshold(
        raw_candidates,
        ratio=blur_ratio,
        hard_floor=blur_hard_floor,
    )

    skipped_blurry = 0
    skipped_brightness = 0
    filtered = []

    for c in raw_candidates:
        if c["blur_score"] < adaptive_blur:
            skipped_blurry += 1
            continue
        if is_bad_brightness(c["brightness"], min_brightness, max_brightness):
            skipped_brightness += 1
            continue
        filtered.append(c)

    if verbose:
        print(
            f"[extract] adaptive_blur={adaptive_blur:.1f} | "
            f"blurry={skipped_blurry} brightness={skipped_brightness} "
            f"kept={len(filtered)}"
        )

    # 6. Dedup
    unique_candidates, skipped_duplicates = remove_duplicate_candidates(
        candidates=filtered,
        similarity_threshold=similarity_threshold,
    )

    # 7. Balanced selection
    final_frames = select_balanced_final_frames(
        candidates=unique_candidates,
        total_frames=total_frames,
        max_frames=max_frames,
    )

    # 8. Safety fallback: never return zero frames
    used_fallback = False
    if not final_frames and raw_candidates:
        used_fallback = True
        fallback = sorted(raw_candidates, key=lambda c: c["blur_score"], reverse=True)
        fallback = fallback[: max(min_frames, 1)]
        final_frames = sorted(fallback, key=lambda c: c["frame_num"])
        if verbose:
            print("[extract] all filters rejected everything -> raw fallback")

    # 9. Save frames for Moondream
    saved_files = []
    frame_meta = []

    for index, candidate in enumerate(final_frames, start=1):
        filename = f"frame_{index:03d}.jpg"
        frame_path = os.path.join(output_folder, filename)
        cv2.imwrite(frame_path, candidate["frame"], [cv2.IMWRITE_JPEG_QUALITY, 90])
        saved_files.append(frame_path)
        frame_meta.append({
            "file": frame_path,
            "frame_num": candidate["frame_num"],
            "timestamp_sec": candidate["timestamp_sec"],
            "blur_score": round(candidate["blur_score"], 2),
            "brightness": round(candidate["brightness"], 2),
        })

    return {
        "success": True,
        "video": video_path,
        "total_frames": total_frames,
        "fps": round(fps, 2),
        "duration_sec": round(duration_sec, 2),
        "scenes_detected": len(scene_list),
        "ranges_processed": len(ranges),
        "raw_candidates": len(raw_candidates),
        "adaptive_blur_threshold": round(adaptive_blur, 2),
        "skipped_blurry": skipped_blurry,
        "skipped_brightness": skipped_brightness,
        "skipped_duplicates": skipped_duplicates,
        "unique_candidates": len(unique_candidates),
        "kept_frames": len(final_frames),
        "used_safety_fallback": used_fallback,
        "frames_folder": output_folder,
        "saved_files": saved_files,
        "frames": frame_meta,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python extract_frames.py uploads/butterflyy.mp4")
        sys.exit(1)

    video_path = sys.argv[1]
    result = extract_frames(video_path, verbose=True)

    print("\nExtraction result:")
    for key, value in result.items():
        if key == "frames":
            print(f"{key}:")
            for f in value:
                print(f"  - {f}")
        else:
            print(f"{key}: {value}")
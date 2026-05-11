import os
import shutil
import cv2
import numpy as np
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector


def compute_frame_hash(frame, hash_size=16):
    """Compute a perceptual hash (dHash) for similarity comparison."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    return diff.flatten()


def hamming_distance(hash1, hash2):
    """Count differing bits between two perceptual hashes."""
    return np.count_nonzero(hash1 != hash2)


def compute_blur_score(frame):
    """Higher = sharper. Uses variance of Laplacian."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def pick_best_frame_in_range(cap, start_frame, end_frame, sample_count=5):
    """
    Sample a few frames in [start, end] and return the sharpest one.
    Avoids picking blurry transition frames.
    """
    if end_frame <= start_frame:
        return start_frame, None

    # Sample evenly across the scene, skip the very edges (often transitions/motion blur)
    margin = max(1, (end_frame - start_frame) // 10)
    s = start_frame + margin
    e = end_frame - margin
    if e <= s:
        s, e = start_frame, end_frame

    candidates = np.linspace(s, e, num=min(sample_count, e - s + 1), dtype=int)

    best_frame_num = candidates[0]
    best_score = -1
    best_image = None

    for fn in candidates:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fn))
        ok, frame = cap.read()
        if not ok:
            continue
        score = compute_blur_score(frame)
        if score > best_score:
            best_score = score
            best_frame_num = int(fn)
            best_image = frame

    return best_frame_num, best_image


def extract_frames(
    video_path,
    output_folder="frames",
    threshold=27.0,
    min_scene_len_sec=1.0,
    fallback_interval_sec=2.0,
    similarity_threshold=10,
    min_blur_score=50.0,
):
    """
    Extract one representative frame per scene.

    Args:
        threshold: Scene-detection sensitivity (higher = fewer scenes). 27 is a good default.
        min_scene_len_sec: Ignore scenes shorter than this (filters out flickers/cuts).
        fallback_interval_sec: If no scenes detected, sample every N seconds.
        similarity_threshold: Max Hamming distance (out of 256) below which frames are considered duplicates.
        min_blur_score: Reject frames blurrier than this (Laplacian variance).
    """
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    if total_frames <= 0:
        cap.release()
        return {"scenes_detected": 0, "kept_frames": 0, "skipped_duplicates": 0,
                "skipped_blurry": 0, "frames_folder": output_folder}

    # Detect scenes with a minimum length to avoid micro-cuts
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(threshold=threshold, min_scene_len=int(min_scene_len_sec * fps))
    )
    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()

    # Build list of (start, end) ranges to pick from
    ranges = []
    if scene_list:
        for scene in scene_list:
            s = scene[0].get_frames()
            e = scene[1].get_frames() - 1
            ranges.append((s, e))
    else:
        # Fallback: chunk the video into intervals
        step = int(fallback_interval_sec * fps)
        for s in range(0, total_frames, step):
            e = min(s + step - 1, total_frames - 1)
            ranges.append((s, e))

    kept_hashes = []
    kept_count = 0
    skipped_duplicates = 0
    skipped_blurry = 0

    for (s, e) in ranges:
        frame_num, frame = pick_best_frame_in_range(cap, s, e)
        if frame is None:
            continue

        # Reject blurry frames
        if compute_blur_score(frame) < min_blur_score:
            skipped_blurry += 1
            continue

        # Reject near-duplicates of already-kept frames
        h = compute_frame_hash(frame)
        is_duplicate = any(hamming_distance(h, kh) < similarity_threshold for kh in kept_hashes)
        if is_duplicate:
            skipped_duplicates += 1
            continue

        kept_hashes.append(h)
        frame_path = os.path.join(output_folder, f"frame_{frame_num:07d}.jpg")
        cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        kept_count += 1

    cap.release()

    return {
        "scenes_detected": len(scene_list),
        "kept_frames": kept_count,
        "skipped_duplicates": skipped_duplicates,
        "skipped_blurry": skipped_blurry,
        "frames_folder": output_folder,
    }
"""
Download a YouTube (or other yt-dlp supported) video for local processing.

Only the download touches the network; all AI inference stays local.
"""
import os
import yt_dlp


def get_video_info(url):
    """Fetch metadata without downloading (title, duration in seconds)."""
    ydl_opts = {"quiet": True, "no_warnings": True, "skip_download": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
    return {
        "id": info.get("id"),
        "title": info.get("title"),
        "duration_sec": info.get("duration") or 0,
    }


def download_youtube(url, output_dir="uploads", max_duration_sec=None):
    """
    Download a video to output_dir.

    Args:
        url: video URL.
        output_dir: destination folder.
        max_duration_sec: if set and the video is longer, raise ValueError
                          BEFORE downloading (saves bandwidth + time).

    Returns:
        dict with file_path, filename, title, duration_sec.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Check duration first (cheap, no download)
    meta = get_video_info(url)
    duration = meta["duration_sec"]
    if max_duration_sec and duration and duration > max_duration_sec:
        raise ValueError(
            f"Video is {int(duration // 60)} min "
            f"(limit is {int(max_duration_sec // 60)} min). Use a shorter video."
        )

    # 2. Download — cap at 720p (plenty for captioning, faster to fetch/process)
    ydl_opts = {
        "format": "best[ext=mp4][height<=720]/best[height<=720]/best[ext=mp4]/best",
        "outtmpl": os.path.join(output_dir, "yt_%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "merge_output_format": "mp4",
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        file_path = ydl.prepare_filename(info)

    # Robust fallback if the extension changed after a merge
    if not os.path.exists(file_path):
        vid_id = info.get("id", "")
        for f in os.listdir(output_dir):
            if f.startswith(f"yt_{vid_id}"):
                file_path = os.path.join(output_dir, f)
                break

    return {
        "file_path": file_path,
        "filename": os.path.basename(file_path),
        "title": info.get("title"),
        "duration_sec": info.get("duration") or duration,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python youtube_download.py <url>")
        sys.exit(1)
    result = download_youtube(sys.argv[1], max_duration_sec=15 * 60)
    print(result)
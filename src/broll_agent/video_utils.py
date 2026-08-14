"""Video discovery, filename slugging, and ffprobe helpers."""

from __future__ import annotations

import hashlib
import re
import subprocess
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}


def slugify(text: str) -> str:
    """Lowercase, spaces/underscores to hyphens, strip non-url-safe chars."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.strip().lower())
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return slug or "video"


def assign_video_slugs(video_paths: Sequence[Path]) -> dict[Path, str]:
    """Slugify each filename stem; append a short hash when two collide."""
    base = {path: slugify(path.stem) for path in video_paths}
    counts = Counter(base.values())
    slugs: dict[Path, str] = {}
    for path in video_paths:
        slug = base[path]
        if counts[slug] > 1:
            digest = hashlib.sha256(path.name.encode("utf-8")).hexdigest()[:8]
            slug = f"{slug}-{digest}"
        slugs[path] = slug
    return slugs


def find_videos(input_dir: Path) -> list[Path]:
    """Top-level scan of input_dir for supported video files (not recursive)."""
    if not input_dir.is_dir():
        return []
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def probe_duration(video_path: Path) -> float:
    """Video duration in seconds via ffprobe (ffmpeg on PATH)."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return float(result.stdout.strip())


def format_timestamp(seconds: float) -> str:
    """Seconds -> mm:ss (or h:mm:ss for long videos)."""
    total = max(0, int(seconds))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def count_files_with_prefix(folder: Path, prefix: str) -> int:
    """Number of files in folder whose name starts with prefix."""
    if not folder.is_dir():
        return 0
    return sum(1 for path in folder.iterdir() if path.is_file() and path.name.startswith(prefix))

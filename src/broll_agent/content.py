"""Normalize transcript JSON, SRT, and plain text into the Transcript model.

Videos are still transcribed by faster-whisper (transcribe.py); this module
only handles inputs that are already transcripts.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from broll_agent.models import Transcript, TranscriptSegment

_SRT_TIME = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*"
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{3})"
)


def _to_seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_json_segments(text: str) -> Transcript:
    payload = json.loads(text)
    wrapper_duration: float | None = None
    source_name: str | None = None
    source_language: str | None = None
    if isinstance(payload, dict):
        wrapper_duration = payload.get("duration_seconds")
        source_name = payload.get("video_file")
        source_language = payload.get("language")
        payload = payload.get("segments") or []
    if not isinstance(payload, list):
        raise ValueError("expected a JSON array of segments")
    segments = [TranscriptSegment.model_validate(item) for item in payload]
    if not segments:
        raise ValueError("no segments provided")
    duration = (
        float(wrapper_duration)
        if wrapper_duration is not None
        else max(s.end or 0.0 for s in segments)
    )
    return Transcript(
        video_file=source_name or "",
        language=source_language,
        duration_seconds=duration,
        segments=segments,
    )


def parse_srt(text: str) -> Transcript:
    segments: list[TranscriptSegment] = []
    blocks = re.split(r"\n\s*\n", text.strip())
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        time_line = next((ln for ln in lines if "-->" in ln), None)
        if time_line is None:
            continue
        match = _SRT_TIME.search(time_line)
        if match is None:
            continue
        start = _to_seconds(*match.group(1, 2, 3, 4))
        end = _to_seconds(*match.group(5, 6, 7, 8))
        idx = lines.index(time_line)
        caption = " ".join(lines[idx + 1 :]).strip()
        if caption:
            segments.append(TranscriptSegment(start=start, end=end, text=caption))
    if not segments:
        raise ValueError("no SRT cues found")
    return Transcript(
        video_file="",
        duration_seconds=max(seg.end or 0.0 for seg in segments),
        segments=segments,
    )


def load_content(path: Path) -> Transcript:
    """Dispatch on file extension: .json/.srt are transcripts; else article text."""
    text = path.read_text(encoding="utf-8-sig")
    suffix = path.suffix.lower()
    if suffix == ".json":
        return parse_json_segments(text)
    if suffix == ".srt":
        return parse_srt(text)
    return Transcript(
        video_file=path.name,
        duration_seconds=None,
        segments=[],
        article_text=text.strip(),
    )

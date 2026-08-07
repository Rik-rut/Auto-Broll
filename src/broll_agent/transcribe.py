"""Video -> Transcript via faster-whisper (decodes audio straight via PyAV)."""

from __future__ import annotations

import json
import logging
import os
import site
import sys
from pathlib import Path

from broll_agent.config import Settings
from broll_agent.models import Transcript, TranscriptSegment
from broll_agent.video_utils import probe_duration

logger = logging.getLogger(__name__)

TRANSCRIPT_FILENAME = "transcript.json"


def _register_cuda_dll_directories() -> None:
    """Windows: point the DLL loader at cuBLAS/cuDNN from the nvidia-* pip packages.

    ctranslate2 loads cublas64_12.dll / cudnn64_9.dll lazily at first GPU use and
    does not search site-packages/nvidia on Windows, so register those bin dirs
    before any faster-whisper CUDA call.
    """
    if sys.platform != "win32":
        return
    for root in site.getsitepackages():
        nvidia_dir = Path(root) / "nvidia"
        if not nvidia_dir.is_dir():
            continue
        for bin_dir in sorted(nvidia_dir.glob("*/bin")):
            os.add_dll_directory(str(bin_dir))


def transcribe_video(video_path: Path, cfg: Settings) -> Transcript:
    _register_cuda_dll_directories()
    from faster_whisper import WhisperModel

    logger.info(
        "transcribing %s (whisper=%s, device=%s, compute=%s)",
        video_path.name,
        cfg.whisper_model_size,
        cfg.whisper_device,
        cfg.whisper_compute_type,
    )
    model = WhisperModel(
        cfg.whisper_model_size,
        device=cfg.whisper_device,
        compute_type=cfg.whisper_compute_type,
    )
    segments_iter, info = model.transcribe(str(video_path), language=cfg.whisper_language)
    segments = [
        TranscriptSegment(start=float(seg.start), end=float(seg.end), text=seg.text.strip())
        for seg in segments_iter
    ]
    duration = (
        float(info.duration) if getattr(info, "duration", None) else probe_duration(video_path)
    )
    transcript = Transcript(
        video_file=video_path.name,
        language=getattr(info, "language", None),
        duration_seconds=duration,
        segments=segments,
    )
    logger.info(
        "transcribed %s: %d segments, %.1fs, language=%s",
        video_path.name,
        len(segments),
        duration,
        transcript.language,
    )
    return transcript


def save_transcript(transcript: Transcript, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(transcript.model_dump_json(indent=2), encoding="utf-8")


def load_transcript(path: Path) -> Transcript:
    return Transcript.model_validate(json.loads(path.read_text(encoding="utf-8-sig")))


def load_or_transcribe(
    video_path: Path, transcript_path: Path, cfg: Settings, force: bool = False
) -> Transcript:
    """Reuse an existing transcript.json unless --force, else transcribe + save."""
    if transcript_path.exists() and not force:
        logger.info("reusing existing transcript %s", transcript_path)
        return load_transcript(transcript_path)
    transcript = transcribe_video(video_path, cfg)
    save_transcript(transcript, transcript_path)
    return transcript

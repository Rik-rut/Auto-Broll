from __future__ import annotations

from pathlib import Path

import pytest

from broll_agent.content import load_content, parse_json_segments, parse_srt

SEGMENT_ARRAY = """[
  {"start": 0.0, "end": 6.5, "text": "Welcome back."},
  {"start": 6.5, "end": 12.0, "text": "First tip about puppies."}
]"""

WRAPPER = """{
  "duration_seconds": 12.0,
  "segments": [
    {"start": 0.0, "end": 6.5, "text": "Welcome back."},
    {"start": 6.5, "end": 12.0, "text": "First tip about puppies."}
  ]
}"""

FULL_SHAPE = """{
  "video_file": "how_to_train_a_puppy.mp4",
  "language": "en",
  "duration_seconds": 12.0,
  "segments": [
    {"start": 0.0, "end": 6.5, "text": "Welcome back."}
  ]
}"""

SRT_TEXT = """1
00:00:00,000 --> 00:00:06,500
Welcome back.

2
00:00:06,500 --> 00:00:12,000
First tip about puppies,
second line of the same cue.
"""

SRT_DOTS = """1
00:00:00.000 --> 00:00:06.500
Welcome back.
"""


def test_parse_bare_segment_array() -> None:
    transcript = parse_json_segments(SEGMENT_ARRAY)
    assert len(transcript.segments) == 2
    assert transcript.segments[0].start == 0.0
    assert transcript.duration_seconds == 12.0


def test_parse_wrapper_with_duration() -> None:
    transcript = parse_json_segments(WRAPPER)
    assert len(transcript.segments) == 2
    assert transcript.duration_seconds == 12.0


def test_parse_full_transcript_shape() -> None:
    transcript = parse_json_segments(FULL_SHAPE)
    assert transcript.video_file == "how_to_train_a_puppy.mp4"
    assert transcript.language == "en"
    assert transcript.duration_seconds == 12.0
    assert len(transcript.segments) == 1


def test_parse_json_rejects_empty_segments() -> None:
    with pytest.raises(ValueError, match="no segments"):
        parse_json_segments("[]")


def test_parse_json_rejects_malformed() -> None:
    with pytest.raises(ValueError):
        parse_json_segments("not json at all")


def test_parse_srt_joins_multiline_cues() -> None:
    transcript = parse_srt(SRT_TEXT)
    assert len(transcript.segments) == 2
    assert transcript.segments[0].start == pytest.approx(0.0)
    assert transcript.segments[0].end == pytest.approx(6.5)
    assert "second line" in transcript.segments[1].text
    assert transcript.duration_seconds == pytest.approx(12.0)


def test_parse_srt_accepts_dot_milliseconds() -> None:
    transcript = parse_srt(SRT_DOTS)
    assert len(transcript.segments) == 1
    assert transcript.segments[0].end == pytest.approx(6.5)


def test_parse_srt_rejects_empty() -> None:
    with pytest.raises(ValueError, match="no SRT cues"):
        parse_srt("just some text without a time line")


def test_load_content_dispatches_on_suffix(tmp_path: Path) -> None:
    json_path = tmp_path / "t.json"
    json_path.write_text(WRAPPER, encoding="utf-8")
    srt_path = tmp_path / "t.srt"
    srt_path.write_text(SRT_TEXT, encoding="utf-8")
    txt_path = tmp_path / "t.txt"
    txt_path.write_text("Plain narration text, no timestamps.", encoding="utf-8")

    assert len(load_content(json_path).segments) == 2
    assert len(load_content(srt_path).segments) == 2

    article = load_content(txt_path)
    assert article.segments == []
    assert article.duration_seconds is None
    assert article.article_text == "Plain narration text, no timestamps."
    assert article.video_file == "t.txt"

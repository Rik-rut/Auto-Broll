from __future__ import annotations

from pathlib import Path

from broll_agent.video_utils import (
    TRANSCRIPT_EXTENSIONS,
    assign_video_slugs,
    find_transcripts,
    find_videos,
)


def test_find_transcripts_scans_top_level(tmp_path: Path) -> None:
    (tmp_path / "talk.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n", encoding="utf-8")
    (tmp_path / "notes.json").write_text("[]", encoding="utf-8")
    (tmp_path / "story.txt").write_text("text", encoding="utf-8")
    (tmp_path / "movie.mp4").write_bytes(b"not really a video")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "nested.srt").write_text("x", encoding="utf-8")

    found = find_transcripts(tmp_path)
    assert [p.name for p in found] == ["notes.json", "story.txt", "talk.srt"]
    assert find_videos(tmp_path) == [tmp_path / "movie.mp4"]


def test_find_transcripts_missing_dir(tmp_path: Path) -> None:
    assert find_transcripts(tmp_path / "nope") == []


def test_slugs_assigned_over_combined_inputs(tmp_path: Path) -> None:
    video = tmp_path / "talk.mp4"
    transcript = tmp_path / "talk.json"
    video.touch()
    transcript.touch()
    slugs = assign_video_slugs([video, transcript])
    assert len(set(slugs.values())) == 2
    assert slugs[video] != slugs[transcript]
    assert slugs[video].startswith("talk")


def test_transcript_extensions_set() -> None:
    assert TRANSCRIPT_EXTENSIONS == {".json", ".srt", ".txt"}

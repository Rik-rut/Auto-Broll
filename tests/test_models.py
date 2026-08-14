from __future__ import annotations

from broll_agent.models import BrollOpportunity, Transcript, TranscriptSegment


def test_transcript_allows_article_text_without_timestamps() -> None:
    transcript = Transcript(
        video_file="article.txt",
        duration_seconds=None,
        segments=[],
        article_text="Some narration with no timestamps.",
    )
    assert transcript.article_text == "Some narration with no timestamps."
    assert transcript.duration_seconds is None


def test_segment_timestamps_optional() -> None:
    seg = TranscriptSegment(text="only text")
    assert seg.start is None
    assert seg.end is None


def test_opportunity_timestamps_optional() -> None:
    opp = BrollOpportunity(
        id="001",
        slug="casino-betting",
        context_text="ctx",
        reasoning="why",
        image_prompt="a casino table",
        searxng_query="casino table",
    )
    assert opp.timestamp_start is None
    assert opp.timestamp_end is None


def test_existing_timestamped_models_still_parse() -> None:
    transcript = Transcript.model_validate(
        {
            "video_file": "x.mp4",
            "duration_seconds": 10.0,
            "segments": [{"start": 0.0, "end": 10.0, "text": "hi"}],
        }
    )
    assert transcript.segments[0].start == 0.0
    opp = BrollOpportunity.model_validate(
        {
            "id": "001",
            "slug": "s",
            "timestamp_start": 1.0,
            "timestamp_end": 2.0,
            "context_text": "c",
            "reasoning": "r",
            "image_prompt": "p",
            "searxng_query": "q",
        }
    )
    assert opp.timestamp_start == 1.0

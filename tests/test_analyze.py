from __future__ import annotations

import pytest

from broll_agent.analyze import analyze_transcript, parse_opportunities
from broll_agent.models import BrollOpportunity

VALID_RESPONSE = """[
  {
    "id": "001",
    "slug": "puppy-chewing-a-shoe",
    "timestamp_start": 41.2,
    "timestamp_end": 47.8,
    "context_text": "...and this is where he chewed straight through my shoe.",
    "reasoning": "Concrete, visual anecdote with no on-screen demo.",
    "image_prompt": "a golden retriever puppy chewing a leather shoe on a hardwood floor",
    "searxng_query": "puppy chewing shoe hardwood floor"
  },
  {
    "id": "002",
    "slug": "leash-training-in-a-park",
    "timestamp_start": 60.0,
    "timestamp_end": 125.0,
    "context_text": "So grab your leash, and let's head to the park.",
    "reasoning": "Location change with a clear visual subject.",
    "image_prompt": "a puppy on a leash walking through a sunny park",
    "searxng_query": "puppy leash park"
  }
]"""


def test_parse_maps_raw_json_to_opportunities() -> None:
    opportunities = parse_opportunities(VALID_RESPONSE)
    assert len(opportunities) == 2
    assert all(isinstance(opp, BrollOpportunity) for opp in opportunities)
    first = opportunities[0]
    assert first.id == "001"
    assert first.slug == "puppy-chewing-a-shoe"
    assert first.timestamp_start == pytest.approx(41.2)
    assert first.timestamp_end == pytest.approx(47.8)
    assert "shoe" in first.context_text
    assert first.searxng_query == "puppy chewing shoe hardwood floor"


def test_parse_tolerates_markdown_fences() -> None:
    opportunities = parse_opportunities(f"```json\n{VALID_RESPONSE}\n```")
    assert len(opportunities) == 2


def test_parse_assigns_ids_and_slugs_when_missing() -> None:
    response = """[
      {
        "timestamp_start": 10.0,
        "timestamp_end": 15.0,
        "context_text": "ctx",
        "reasoning": "why",
        "image_prompt": "a sleepy puppy curled up on a wool blanket near a window",
        "searxng_query": "sleepy puppy blanket"
      }
    ]"""
    opportunities = parse_opportunities(response)
    assert opportunities[0].id == "001"
    assert opportunities[0].slug == "a-sleepy-puppy-curled-up-on"


def test_parse_rejects_non_array() -> None:
    with pytest.raises(ValueError, match="no JSON array"):
        parse_opportunities('{"id": "001"}')


def test_analyze_retries_once_on_malformed_json(transcript, fake_llm_factory) -> None:
    llm = fake_llm_factory(["sorry, I cannot output JSON", VALID_RESPONSE])
    opportunities = analyze_transcript(transcript, llm)
    assert llm.calls == 2
    assert [opp.id for opp in opportunities] == ["001", "002"]


def test_analyze_raises_after_retry_exhausted(transcript, fake_llm_factory) -> None:
    llm = fake_llm_factory(["not json", "still not json"])
    with pytest.raises(RuntimeError, match="parseable JSON"):
        analyze_transcript(transcript, llm)
    assert llm.calls == 2


def test_analyze_succeeds_without_retry(transcript, fake_llm_factory) -> None:
    llm = fake_llm_factory([VALID_RESPONSE])
    opportunities = analyze_transcript(transcript, llm)
    assert llm.calls == 1
    assert len(opportunities) == 2


NULL_TIMESTAMP_RESPONSE = """[
  {
    "id": "001",
    "slug": "casino-floor-betting",
    "timestamp_start": null,
    "timestamp_end": null,
    "context_text": "The casino floor is where the house always wins.",
    "reasoning": "Concrete, visual subject with no time reference.",
    "image_prompt": "a busy casino floor with roulette tables and slot machines",
    "searxng_query": "casino floor roulette"
  }
]"""


def test_parse_accepts_null_timestamps() -> None:
    opportunities = parse_opportunities(NULL_TIMESTAMP_RESPONSE)
    assert len(opportunities) == 1
    assert opportunities[0].timestamp_start is None
    assert opportunities[0].timestamp_end is None


def test_analyze_article_transcript_passes_article_text(fake_llm_factory) -> None:
    from broll_agent.models import Transcript

    article = Transcript(
        video_file="casino.txt",
        duration_seconds=None,
        segments=[],
        article_text="The casino floor is where the house always wins.",
    )
    llm = fake_llm_factory([NULL_TIMESTAMP_RESPONSE])
    opportunities = analyze_transcript(article, llm)
    assert llm.calls == 1
    assert llm.prompts[0][1] == "The casino floor is where the house always wins."
    assert len(opportunities) == 1

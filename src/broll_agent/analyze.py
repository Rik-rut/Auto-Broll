"""Transcript -> list[BrollOpportunity] via one strict-JSON LLM call."""

from __future__ import annotations

import json
import logging
import re

from pydantic import BaseModel, ValidationError

from broll_agent.llm_client import LLMCompleter
from broll_agent.models import BrollOpportunity, Transcript
from broll_agent.video_utils import format_timestamp, slugify

logger = logging.getLogger(__name__)

ANALYZE_SYSTEM_PROMPT = """\
You are a video editor's assistant scouting broll cutaway moments in a narration video.

You will receive a transcript with timestamps. Find the moments where cutting away to a
broll image would improve the video, and for each one write an image-generation prompt
and a short search query.

Rules:
- Return ONLY a JSON array — no markdown fences, no commentary.
- Each item must be an object with exactly these fields:
  "id": string, "001", "002", ... in order of occurrence,
  "slug": short kebab-case label of the broll idea,
  "timestamp_start": float seconds,
  "timestamp_end": float seconds (must be > timestamp_start),
  "context_text": the narration lines this moment covers,
  "reasoning": one sentence on why a cutaway works here,
  "image_prompt": a fully self-contained visual description — subject, setting, action,
    lighting. No meta text like "an image of", and no style words (style is appended later).
  "searxng_query": 3-6 keywords for an image search engine, not a rehash of the prompt.
- Don't over-suggest: a handful of well-chosen moments beats one per sentence. Base density
  on the video's length and how visual/concrete each moment is, not a fixed interval.
- Skip moments that are abstract, that reference the speaker directly, or where a cutaway
  would break continuity.
- Timestamps must fall within the transcript's time range.
"""


class _RawOpportunity(BaseModel):
    id: str | None = None
    slug: str | None = None
    timestamp_start: float
    timestamp_end: float
    context_text: str
    reasoning: str
    image_prompt: str
    searxng_query: str


def format_transcript_for_llm(transcript: Transcript) -> str:
    return "\n".join(
        f"[{format_timestamp(seg.start)} - {format_timestamp(seg.end)}] {seg.text}"
        for seg in transcript.segments
    )


def extract_json_array(text: str) -> str:
    """Pull the JSON array out of an LLM reply, tolerating code fences."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9]*\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    start = stripped.find("[")
    end = stripped.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON array found in LLM response")
    return stripped[start : end + 1]


def parse_opportunities(raw_response: str) -> list[BrollOpportunity]:
    payload = json.loads(extract_json_array(raw_response))
    if not isinstance(payload, list):
        raise ValueError("LLM response is not a JSON array")
    raw_items = [_RawOpportunity.model_validate(item) for item in payload]
    opportunities: list[BrollOpportunity] = []
    for index, raw in enumerate(raw_items, start=1):
        slug = raw.slug or slugify(" ".join(raw.image_prompt.split()[:6]))
        opportunities.append(
            BrollOpportunity(
                id=f"{index:03d}",
                slug=slugify(slug),
                timestamp_start=raw.timestamp_start,
                timestamp_end=raw.timestamp_end,
                context_text=raw.context_text,
                reasoning=raw.reasoning,
                image_prompt=raw.image_prompt,
                searxng_query=raw.searxng_query,
            )
        )
    return opportunities


def analyze_transcript(transcript: Transcript, client: LLMCompleter) -> list[BrollOpportunity]:
    """One LLM call; retries exactly once on a parse failure."""
    user_content = format_transcript_for_llm(transcript)
    prompt = user_content
    last_error: Exception | None = None
    for attempt in (1, 2):
        raw_response = client.complete(ANALYZE_SYSTEM_PROMPT, prompt)
        try:
            opportunities = parse_opportunities(raw_response)
        except (ValueError, ValidationError) as error:
            last_error = error
            logger.warning("analyze attempt %d returned unparsable JSON: %s", attempt, error)
            prompt = (
                f"{user_content}\n\n"
                "Your previous reply was not valid JSON. Reply with ONLY the raw JSON array."
            )
            continue
        logger.info("analyzed transcript: %d broll opportunities found", len(opportunities))
        return opportunities
    raise RuntimeError(f"LLM did not return parseable JSON after retry: {last_error}")

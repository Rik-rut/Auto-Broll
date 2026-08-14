"""Pydantic data contracts shared across the pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class TranscriptSegment(BaseModel):
    start: float | None = None
    end: float | None = None
    text: str


class Transcript(BaseModel):
    video_file: str
    language: str | None = None
    duration_seconds: float | None = None
    segments: list[TranscriptSegment] = Field(default_factory=list)
    article_text: str | None = None


class BrollOpportunity(BaseModel):
    id: str
    slug: str
    timestamp_start: float | None = None
    timestamp_end: float | None = None
    context_text: str
    reasoning: str
    image_prompt: str
    searxng_query: str

    @model_validator(mode="before")
    @classmethod
    def _accept_zimage_prompt(cls, data: object) -> object:
        if isinstance(data, dict) and "zimage_prompt" in data and "image_prompt" not in data:
            data = {**data, "image_prompt": data.pop("zimage_prompt")}
        return data


class BrollPlan(BaseModel):
    video_file: str
    video_slug: str
    analyzed_at: str
    llm_model: str
    broll_opportunities: list[BrollOpportunity] = Field(default_factory=list)

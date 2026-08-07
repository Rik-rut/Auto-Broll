from __future__ import annotations

import json
from pathlib import Path

import pytest

from broll_agent.config import Settings
from broll_agent.models import Transcript

FIXTURES = Path(__file__).parent / "fixtures"


class FakeLLM:
    """Duck-typed stand-in for LLMClient: replays canned responses in order."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls = 0
        self.prompts: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        self.prompts.append((system, user))
        return self.responses.pop(0)


@pytest.fixture()
def fake_llm_factory():
    return FakeLLM


@pytest.fixture()
def settings() -> Settings:
    return Settings(_env_file=None)


@pytest.fixture()
def transcript() -> Transcript:
    payload = json.loads((FIXTURES / "transcript.json").read_text(encoding="utf-8"))
    return Transcript.model_validate(payload)


@pytest.fixture()
def searxng_payload() -> dict:
    return json.loads((FIXTURES / "searxng_response.json").read_text(encoding="utf-8"))

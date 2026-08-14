"""Thin provider-agnostic LLM wrapper backed by litellm."""

from __future__ import annotations

import logging
import time
from typing import Protocol

from broll_agent.config import Settings

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Raised for unrecoverable LLM configuration or call failures."""


class LLMCompleter(Protocol):
    """Anything that can answer a system+user prompt with text."""

    def complete(self, system: str, user: str) -> str: ...


class LLMClient:
    def __init__(self, cfg: Settings) -> None:
        self.cfg = cfg

    def _litellm_model(self) -> str:
        provider = self.cfg.llm_provider.strip().lower()
        model = self.cfg.llm_model
        if provider == "anthropic":
            return model
        if provider == "openai":
            return model
        if provider == "ollama":
            return f"ollama_chat/{model}"
        if provider == "openai_compatible":
            return f"openai/{model}"
        raise LLMError(f"unsupported LLM_PROVIDER: {provider!r}")

    def complete(self, system: str, user: str) -> str:
        """One chat completion; litellm handles retries/backoff internally."""
        import litellm

        kwargs: dict[str, object] = {
            "model": self._litellm_model(),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "timeout": self.cfg.llm_timeout_seconds,
            "max_retries": self.cfg.llm_max_retries,
        }
        if self.cfg.llm_api_key:
            kwargs["api_key"] = self.cfg.llm_api_key
        if self.cfg.llm_base_url:
            kwargs["api_base"] = self.cfg.llm_base_url

        start = time.perf_counter()
        response = litellm.completion(**kwargs)
        text = response.choices[0].message.content
        if not text:
            raise LLMError(f"empty response from {self._litellm_model()}")
        logger.debug("llm completion ok (%.1fs, %d chars)", time.perf_counter() - start, len(text))
        return text

from __future__ import annotations

from pathlib import Path

from broll_agent.config import Settings


def test_blank_seed_parses_as_none() -> None:
    settings = Settings(_env_file=None, zimage_seed="")
    assert settings.zimage_seed is None


def test_seed_parses_as_int() -> None:
    settings = Settings(_env_file=None, zimage_seed="42")
    assert settings.zimage_seed == 42


def test_blank_and_comment_values_become_none() -> None:
    settings = Settings(
        _env_file=None,
        whisper_language="",
        llm_base_url="   ",
        negative_prompt="# blank = none",
    )
    assert settings.whisper_language is None
    assert settings.llm_base_url is None
    assert settings.negative_prompt is None


def test_variations_file_default() -> None:
    cfg = Settings(_env_file=None)
    assert cfg.variations_file == Path("./config/image_variations.json")
    assert not hasattr(cfg, "prompt_variations")

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

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


def test_image_engine_defaults_to_zimage() -> None:
    cfg = Settings(_env_file=None)
    assert cfg.image_engine == "zimage"


def test_invalid_image_engine_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, image_engine="bogus")


def test_qwen_defaults() -> None:
    cfg = Settings(_env_file=None)
    assert cfg.qwen_model_dir == Path("./qwen model")
    assert cfg.qwen_device == "cuda"
    assert cfg.qwen_images_per_prompt == 3
    assert cfg.qwen_num_inference_steps == 6
    assert cfg.qwen_guidance_scale == 1.0
    assert cfg.qwen_seed is None
    assert cfg.qwen_quanto_int8_kernel is True
    assert cfg.qwen_mmgp_profile == 5
    assert cfg.qwen_attention_backend == "auto"
    assert cfg.qwen_style_prompt is None


def test_blank_qwen_seed_parses_as_none() -> None:
    settings = Settings(_env_file=None, qwen_seed="")
    assert settings.qwen_seed is None


def test_blank_qwen_style_prompt_parses_as_none() -> None:
    settings = Settings(_env_file=None, qwen_style_prompt="")
    assert settings.qwen_style_prompt is None


def test_qwen_style_prompt_parses_as_string() -> None:
    settings = Settings(_env_file=None, qwen_style_prompt="grainy 35mm film")
    assert settings.qwen_style_prompt == "grainy 35mm film"

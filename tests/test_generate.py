from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image as PILImage

from broll_agent.config import Settings
from broll_agent.generate_lightning import (
    GENERATED_PREFIX,
    build_prompt,
    generate_images_for_opportunity,
    load_variations,
)
from broll_agent.models import BrollOpportunity
from broll_agent.video_utils import count_files_with_prefix

VARIANTS = [
    "wide environmental composition, low-angle camera, bright natural daylight",
    "close detail framing, high-angle camera, warm directional backlight",
    "overhead composition, high camera, broad soft daylight",
]


def _opportunity() -> BrollOpportunity:
    return BrollOpportunity(
        id="001",
        slug="puppy-chewing-a-shoe",
        timestamp_start=41.2,
        timestamp_end=47.8,
        context_text="...and this is where he chewed straight through my shoe.",
        reasoning="Concrete, visual anecdote.",
        image_prompt="a golden retriever puppy chewing a leather shoe",
        searxng_query="puppy chewing shoe",
    )


class FakePipeline:
    """Stands in for ZImagePipeline: records call kwargs, returns blanks."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, **kwargs) -> SimpleNamespace:
        self.calls.append(kwargs)
        image = PILImage.new("RGB", (kwargs["width"], kwargs["height"]), color=(12, 34, 56))
        return SimpleNamespace(images=[image])


def _generate_settings() -> Settings:
    return Settings(
        _env_file=None,
        zimage_device="cpu",
        zimage_images_per_prompt=2,
        image_width=64,
        image_height=96,
        style_prompt="cinematic photo, 35mm film grain",
    )


def test_generate_numbers_files_and_saves_pngs(tmp_path: Path) -> None:
    cfg = _generate_settings()
    pipe = FakePipeline()
    paths = generate_images_for_opportunity(pipe, _opportunity(), cfg, tmp_path, VARIANTS)
    assert [p.name for p in paths] == ["generated_01.png", "generated_02.png"]
    assert all(path.exists() for path in paths)
    assert count_files_with_prefix(tmp_path, GENERATED_PREFIX) == 2


def test_generate_appends_variation_and_style(tmp_path: Path) -> None:
    cfg = _generate_settings()
    pipe = FakePipeline()
    generate_images_for_opportunity(pipe, _opportunity(), cfg, tmp_path, VARIANTS)
    call = pipe.calls[0]
    assert VARIANTS[0] in call["prompt"]
    assert "cinematic photo, 35mm film grain" in call["prompt"]
    assert "a golden retriever puppy chewing a leather shoe" in call["prompt"]
    assert call["guidance_scale"] == 0.0
    assert call["num_inference_steps"] == cfg.zimage_num_inference_steps
    assert (call["width"], call["height"]) == (64, 96)


def test_generate_variations_differ_and_rotate(tmp_path: Path) -> None:
    cfg = _generate_settings()
    cfg.zimage_images_per_prompt = 4
    pipe = FakePipeline()
    generate_images_for_opportunity(pipe, _opportunity(), cfg, tmp_path, VARIANTS)
    prompts = [c["prompt"] for c in pipe.calls]
    assert prompts[0] != prompts[1]
    assert VARIANTS[2] in prompts[2]
    assert VARIANTS[0] in prompts[3]


def test_build_prompt_without_style() -> None:
    cfg = _generate_settings()
    cfg.negative_prompt = None
    cfg.style_prompt = ""
    result = build_prompt(_opportunity(), cfg, "test variation")
    assert result == "a golden retriever puppy chewing a leather shoe, test variation"


def test_generate_uses_configured_seed(tmp_path: Path) -> None:
    cfg = _generate_settings()
    cfg.zimage_seed = 1234
    pipe = FakePipeline()
    generate_images_for_opportunity(pipe, _opportunity(), cfg, tmp_path, VARIANTS)
    generators = [call["generator"] for call in pipe.calls]
    assert all(generator.initial_seed() == 1234 for generator in generators)


def test_count_files_with_prefix_missing_dir(tmp_path: Path) -> None:
    assert count_files_with_prefix(tmp_path / "nope", GENERATED_PREFIX) == 0


def test_load_variations_reads_json_file(tmp_path: Path) -> None:
    path = tmp_path / "image_variations.json"
    path.write_text(json.dumps({"variations": VARIANTS}), encoding="utf-8")
    assert load_variations(path) == VARIANTS


def test_load_variations_rejects_empty(tmp_path: Path) -> None:
    path = tmp_path / "image_variations.json"
    path.write_text(json.dumps({"variations": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="no variations"):
        load_variations(path)


def test_load_variations_rejects_missing(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        load_variations(tmp_path / "nope.json")

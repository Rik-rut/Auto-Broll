from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image as PILImage

from broll_agent.config import Settings
from broll_agent.engines import get_engine
from broll_agent.generation import (
    build_prompt,
    generate_images_for_opportunity,
    generated_prefix,
    images_per_prompt,
    load_variations,
    migrate_legacy_generated,
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
    assert [p.name for p in paths] == ["generated_zimage_01.png", "generated_zimage_02.png"]
    assert all(path.exists() for path in paths)
    assert count_files_with_prefix(tmp_path, generated_prefix("zimage")) == 2


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
    assert count_files_with_prefix(tmp_path / "nope", generated_prefix("zimage")) == 0


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


def test_migrate_legacy_generated_renames_untagged_files(tmp_path: Path) -> None:
    (tmp_path / "generated_01.png").write_bytes(b"x")
    (tmp_path / "generated_02.png").write_bytes(b"x")
    migrate_legacy_generated(tmp_path)
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "generated_zimage_01.png",
        "generated_zimage_02.png",
    ]


def test_migrate_legacy_generated_skips_existing_target(tmp_path: Path) -> None:
    (tmp_path / "generated_01.png").write_bytes(b"legacy")
    (tmp_path / "generated_zimage_01.png").write_bytes(b"tagged")
    migrate_legacy_generated(tmp_path)
    assert (tmp_path / "generated_01.png").read_bytes() == b"legacy"
    assert (tmp_path / "generated_zimage_01.png").read_bytes() == b"tagged"


def test_migrate_legacy_generated_ignores_tagged_files(tmp_path: Path) -> None:
    (tmp_path / "generated_qwen_image21_01.png").write_bytes(b"x")
    migrate_legacy_generated(tmp_path)
    assert (tmp_path / "generated_qwen_image21_01.png").read_bytes() == b"x"


def test_migrate_legacy_generated_missing_dir(tmp_path: Path) -> None:
    migrate_legacy_generated(tmp_path / "nope")


def test_get_engine_zimage() -> None:
    assert get_engine("zimage").name == "zimage"


def test_get_engine_unknown_raises() -> None:
    with pytest.raises(ValueError, match="unknown image engine"):
        get_engine("nope")


def test_images_per_prompt_dispatch_zimage() -> None:
    cfg = _generate_settings()
    assert images_per_prompt(cfg) == 2


class FakeQwenPipe:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return PILImage.new("RGB", (kwargs["width"], kwargs["height"]), color=(1, 2, 3))


def test_qwen_engine_dispatch_and_prefix() -> None:
    engine = get_engine("qwen_image21")
    assert engine.name == "qwen_image21"
    assert generated_prefix(engine.name) == "generated_qwen_image21_"


def test_qwen_engine_uses_qwen_seed() -> None:
    cfg = Settings(
        _env_file=None,
        image_engine="qwen_image21",
        qwen_seed=99,
        image_width=64,
        image_height=96,
    )
    assert get_engine("qwen_image21").configured_seed(cfg) == 99


def test_qwen_engine_rounds_dimensions_and_passes_settings() -> None:
    cfg = Settings(
        _env_file=None,
        image_engine="qwen_image21",
        image_width=720,
        image_height=1280,
    )
    pipe = FakeQwenPipe()
    image = get_engine("qwen_image21").generate_one(pipe, "prompt", cfg, seed=7)
    assert image.size == (704, 1280)
    assert pipe.calls[0] == {
        "prompt": "prompt",
        "seed": 7,
        "width": 704,
        "height": 1280,
        "num_inference_steps": 6,
        "guidance_scale": 1.0,
        "device": "cuda",
    }


def test_qwen_generate_numbers_tagged_files(tmp_path: Path) -> None:
    cfg = Settings(
        _env_file=None,
        image_engine="qwen_image21",
        qwen_images_per_prompt=2,
        image_width=64,
        image_height=96,
        style_prompt="cinematic photo",
    )
    pipe = FakeQwenPipe()
    paths = generate_images_for_opportunity(pipe, _opportunity(), cfg, tmp_path, VARIANTS)
    assert [path.name for path in paths] == [
        "generated_qwen_image21_01.png",
        "generated_qwen_image21_02.png",
    ]
    assert images_per_prompt(cfg) == 2

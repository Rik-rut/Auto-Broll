"""Application settings — the single place `.env` is read.

Every other module receives a `Settings` instance as a parameter instead of
re-reading the environment.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM (transcript analysis, prompt writing, download selection)
    llm_provider: str = "anthropic"
    llm_model: str = "claude-sonnet-4-6"
    llm_api_key: str | None = None
    llm_base_url: str | None = None
    llm_max_retries: int = 3
    llm_timeout_seconds: int = 120

    # Transcription (faster-whisper)
    whisper_model_size: str = "medium"
    whisper_device: str = "cuda"
    whisper_compute_type: str = "float16"
    whisper_language: str | None = None

    # Pipeline toggles — both may be true at the same time
    generate_broll: bool = True
    download_broll: bool = True

    # Z-Image Turbo (local generation via mmgp int8 pipeline)
    zimage_model_dir: Path = Path("./z-image model")
    zimage_device: str = "cuda"
    zimage_images_per_prompt: int = 3
    zimage_num_inference_steps: int = 4
    zimage_guidance_scale: float = 0.0
    zimage_seed: int | None = None
    zimage_quanto_int8_kernel: bool = True
    zimage_mmgp_profile: int = 5
    zimage_perc_reserved_mem_max: float = 0.9
    zimage_vram_safety_coefficient: float = 0.1
    zimage_attention_backend: str = "auto"
    prompt_variations: str | None = None

    # Image output — governs generation size and download filtering
    aspect_ratio: str = "9:16"
    image_width: int = 1080
    image_height: int = 1920
    style_prompt: str = (
        "cinematic photo, natural lighting, shallow depth of field, photorealistic, 35mm film grain"
    )
    negative_prompt: str | None = None

    # SearXNG (broll image search + download)
    searxng_base_url: str = "http://127.0.0.1:8080"
    searxng_timeout_seconds: int = 15
    download_images_per_broll: int = 3
    download_candidate_pool_size: int = 15
    download_min_width: int = 640
    download_min_height: int = 640
    download_timeout_seconds: int = 20

    # Paths
    input_dir: Path = Path("./input")
    output_dir: Path = Path("./output")

    # Logging
    log_level: str = "INFO"

    @field_validator(
        "llm_api_key",
        "llm_base_url",
        "whisper_language",
        "negative_prompt",
        "zimage_seed",
        "prompt_variations",
        mode="before",
    )
    @classmethod
    def _blank_or_comment_to_none(cls, value: object) -> object:
        # Guards against inline .env comments on blank values (e.g. "KEY=  # note"),
        # which pydantic-settings does not strip.
        if isinstance(value, str) and (not value.strip() or value.strip().startswith("#")):
            return None
        return value

@lru_cache
def load_settings() -> Settings:
    """Load settings exactly once per process."""
    return Settings()

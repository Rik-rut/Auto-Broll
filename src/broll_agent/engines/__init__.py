"""Image engine registry — maps IMAGE_ENGINE names to adapters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from PIL import Image

    from broll_agent.config import Settings


class ImageEngine(Protocol):
    """Adapter that loads one local image model and renders single images."""

    name: str

    def load(self, cfg: Settings) -> object: ...

    def images_per_prompt(self, cfg: Settings) -> int: ...

    def configured_seed(self, cfg: Settings) -> int | None: ...

    def generate_one(self, pipe: object, prompt: str, cfg: Settings, seed: int) -> Image.Image: ...


def get_engine(name: str) -> ImageEngine:
    """Return the adapter for a configured engine name."""
    if name == "zimage":
        from broll_agent.engines.zimage import ZImageEngine

        return ZImageEngine()
    if name == "qwen_image21":
        from broll_agent.engines.qwen21 import QwenImage21Engine

        return QwenImage21Engine()
    raise ValueError(f"unknown image engine: {name!r}")

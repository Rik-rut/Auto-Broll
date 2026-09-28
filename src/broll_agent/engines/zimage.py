"""Z-Image Turbo adapter (mmgp int8 pipeline)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from broll_agent.shared.alignment import round_to_alignment

if TYPE_CHECKING:
    from PIL import Image

    from broll_agent.config import Settings

logger = logging.getLogger(__name__)


class ZImageEngine:
    """Wraps the vendored Z-Image pipeline behind the ImageEngine protocol."""

    name = "zimage"
    alignment = 16

    def load(self, cfg: Settings) -> object:
        from broll_agent.zimage.model_loader import load_zimage_pipeline

        return load_zimage_pipeline(cfg)

    def images_per_prompt(self, cfg: Settings) -> int:
        return cfg.zimage_images_per_prompt

    def configured_seed(self, cfg: Settings) -> int | None:
        return cfg.zimage_seed

    def style_prompt(self, cfg: Settings) -> str:
        return cfg.style_prompt or ""

    def generate_one(self, pipe: object, prompt: str, cfg: Settings, seed: int) -> Image.Image:
        import torch

        width = round_to_alignment(cfg.image_width, self.alignment)
        height = round_to_alignment(cfg.image_height, self.alignment)
        if (width, height) != (cfg.image_width, cfg.image_height):
            logger.debug(
                "zimage aligned %sx%s -> %sx%s",
                cfg.image_width,
                cfg.image_height,
                width,
                height,
            )
        generator = torch.Generator(cfg.zimage_device).manual_seed(seed)
        return pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=cfg.zimage_num_inference_steps,
            guidance_scale=cfg.zimage_guidance_scale,
            generator=generator,
        ).images[0]

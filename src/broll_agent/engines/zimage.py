"""Z-Image Turbo adapter (mmgp int8 pipeline)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image

    from broll_agent.config import Settings


class ZImageEngine:
    """Wraps the vendored Z-Image pipeline behind the ImageEngine protocol."""

    name = "zimage"

    def load(self, cfg: Settings) -> object:
        from broll_agent.zimage.model_loader import load_zimage_pipeline

        return load_zimage_pipeline(cfg)

    def images_per_prompt(self, cfg: Settings) -> int:
        return cfg.zimage_images_per_prompt

    def configured_seed(self, cfg: Settings) -> int | None:
        return cfg.zimage_seed

    def generate_one(self, pipe: object, prompt: str, cfg: Settings, seed: int) -> Image.Image:
        import torch

        generator = torch.Generator(cfg.zimage_device).manual_seed(seed)
        return pipe(
            prompt=prompt,
            height=cfg.image_height,
            width=cfg.image_width,
            num_inference_steps=cfg.zimage_num_inference_steps,
            guidance_scale=cfg.zimage_guidance_scale,
            generator=generator,
        ).images[0]

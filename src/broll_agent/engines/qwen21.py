"""Qwen-Image 2.1 turbo adapter (mmgp int8 + Viggle LoRA)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from broll_agent.shared.alignment import round_to_alignment

if TYPE_CHECKING:
    from PIL import Image

    from broll_agent.config import Settings


class QwenImage21Engine:
    """Wraps the qwen21 pipeline behind the ImageEngine protocol."""

    name = "qwen_image21"
    alignment = 32

    def load(self, cfg: Settings) -> object:
        from broll_agent.qwen21.model_loader import load_qwen21_pipeline

        return load_qwen21_pipeline(cfg)

    def images_per_prompt(self, cfg: Settings) -> int:
        return cfg.qwen_images_per_prompt

    def configured_seed(self, cfg: Settings) -> int | None:
        return cfg.qwen_seed

    def style_prompt(self, cfg: Settings) -> str:
        # Blank by default: the Viggle turbo LoRA is distilled without CFG and
        # degrades visibly when a style suffix is appended (detail-suppressing
        # tokens like "shallow depth of field" and "film grain" wash the image
        # out at 6 steps). The bare image_prompt gives the sharpest result.
        return cfg.qwen_style_prompt or ""

    def generate_one(self, pipe: object, prompt: str, cfg: Settings, seed: int) -> Image.Image:
        return pipe.generate(
            prompt=prompt,
            seed=seed,
            width=round_to_alignment(cfg.image_width, self.alignment),
            height=round_to_alignment(cfg.image_height, self.alignment),
            num_inference_steps=cfg.qwen_num_inference_steps,
            guidance_scale=cfg.qwen_guidance_scale,
            device=cfg.qwen_device,
        )

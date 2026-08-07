"""BrollOpportunity -> generated_NN.png via local Z-Image Turbo.

Z-Image Turbo is a Flux-derived distilled model that runs in 4 steps
with guidance_scale=0.0. Weights are pre-quantized to int8 via mmgp
for fast inference with reduced disk and VRAM usage.
"""

from __future__ import annotations

import logging
from pathlib import Path

from broll_agent.config import Settings
from broll_agent.models import BrollOpportunity

logger = logging.getLogger(__name__)

GENERATED_PREFIX = "generated_"

PROMPT_VARIATIONS = [
    "wide angle shot, golden hour lighting, warm tones",
    "medium close-up, overcast sky, soft diffused light",
    "low angle shot, dramatic sunset, orange and purple sky",
    "high angle shot, misty morning, ethereal atmosphere",
    "over-the-shoulder shot, blue hour twilight, cool tones",
    "close-up shot, harsh midday sun, strong shadows",
    "wide establishing shot, rainy weather, wet reflections",
    "medium shot, autumn foliage, warm amber lighting",
    "low angle shot, winter scene, crisp cold atmosphere",
    "eye-level shot, spring morning, fresh vibrant colors",
    "dutch angle, stormy sky, dramatic tension",
    "tracking shot perspective, motion blur background, dynamic energy",
    "static wide shot, foggy dawn, mysterious ambiance",
    "medium shot, summer afternoon, bright cheerful lighting",
    "close-up detail shot, candlelight glow, intimate warm atmosphere",
]


def load_pipeline(cfg: Settings):
    """Load Z-Image Turbo once per run via mmgp with pre-quantized int8 weights."""
    from broll_agent.zimage.model_loader import load_zimage_pipeline

    logger.info("loading Z-Image Turbo via mmgp (model_dir=%s)", cfg.zimage_model_dir)
    return load_zimage_pipeline(cfg)


def build_prompt(opportunity: BrollOpportunity, cfg: Settings, variation: str) -> str:
    """STYLE_PROMPT is appended at generation time, never baked into the plan."""
    parts = [opportunity.image_prompt]
    if variation:
        parts.append(variation)
    if cfg.style_prompt:
        parts.append(cfg.style_prompt)
    return ", ".join(parts)


def generate_images_for_opportunity(
    pipe: object, opportunity: BrollOpportunity, cfg: Settings, dest_dir: Path
) -> list[Path]:
    import torch

    dest_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index in range(cfg.zimage_images_per_prompt):
        variation = PROMPT_VARIATIONS[index % len(PROMPT_VARIATIONS)]
        prompt = build_prompt(opportunity, cfg, variation)
        seed = (
            cfg.zimage_seed
            if cfg.zimage_seed is not None
            else torch.randint(0, 2**32 - 1, (1,)).item()
        )
        image = pipe(
            prompt=prompt,
            height=cfg.image_height,
            width=cfg.image_width,
            num_inference_steps=cfg.zimage_num_inference_steps,
            guidance_scale=cfg.zimage_guidance_scale,
            generator=torch.Generator(cfg.zimage_device).manual_seed(seed),
        ).images[0]
        out_path = dest_dir / f"{GENERATED_PREFIX}{index + 1:02d}.png"
        image.save(out_path)
        paths.append(out_path)
    return paths

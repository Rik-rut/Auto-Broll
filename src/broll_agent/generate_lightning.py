"""BrollOpportunity -> generated_NN.png via local Z-Image Turbo.

Z-Image Turbo is a Flux-derived distilled model that runs in 4 steps
with guidance_scale=0.0. The transformer is quantized to int8 at load
time via torchao for ~2-3x faster inference without extra disk overhead.
"""

from __future__ import annotations

import logging
from pathlib import Path

from broll_agent.config import Settings
from broll_agent.models import BrollOpportunity

logger = logging.getLogger(__name__)

GENERATED_PREFIX = "generated_"

PROMPT_VARIATIONS = [
    "wide angle shot, full scene, establishing view",
    "medium close-up, shallow depth of field, cinematic lighting",
    "low angle shot, dramatic perspective, dynamic composition",
    "over-the-shoulder shot, intimate point of view",
    "high angle shot, bird's eye perspective",
]


def load_pipeline(cfg: Settings):
    """Load Z-Image Turbo once per run with torchao int8 quantization."""
    import torch
    from diffusers import ZImagePipeline
    from torchao.quantization import Int8WeightOnlyConfig, quantize_
    from torchao.quantization.granularity import PerGroup

    model_dir = Path(cfg.zimage_model_dir)

    logger.info(
        "loading Z-Image Turbo (model_dir=%s, device=%s, dtype=%s, offload=%s)",
        model_dir,
        cfg.zimage_device,
        cfg.zimage_dtype,
        cfg.zimage_enable_cpu_offload,
    )

    dtype = getattr(torch, cfg.zimage_dtype)

    pipe = ZImagePipeline.from_pretrained(
        str(model_dir),
        torch_dtype=dtype,
        local_files_only=True,
    )

    logger.info("quantizing transformer to int8 (torchao)")
    quantize_(pipe.transformer, Int8WeightOnlyConfig(granularity=PerGroup(64)))

    if cfg.zimage_enable_cpu_offload:
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.to(cfg.zimage_device)

    return pipe


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

"""BrollOpportunity -> generated_<engine>_NN.png via the configured engine.

Engine-specific model loading and inference live in ``broll_agent.engines``;
this module owns prompt assembly, seed handling, file numbering, legacy
filename migration, and idempotency helpers shared by every engine.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from broll_agent.config import Settings
from broll_agent.engines import get_engine
from broll_agent.models import BrollOpportunity

logger = logging.getLogger(__name__)

LEGACY_GENERATED_PATTERN = re.compile(r"^generated_(\d+)\.(png|jpg|jpeg|webp)$", re.IGNORECASE)


def load_variations(path: Path) -> list[str]:
    """Load prompt variations from a JSON data file ({'variations': [...]})."""
    data = json.loads(path.read_text(encoding="utf-8"))
    variations = data.get("variations") or []
    if not variations:
        raise ValueError(f"no variations configured in {path}")
    return variations


def build_prompt(opportunity: BrollOpportunity, cfg: Settings, variation: str) -> str:
    """Assemble the final generation prompt.

    The style suffix is resolved by the active engine, not read from ``cfg``
    directly: engines differ in how much prompt augmentation they tolerate
    (see ``ImageEngine.style_prompt``). It is never baked into the plan.
    Trailing punctuation is stripped off the base prompt so joining does not
    produce a stray "., " mid-sentence.
    """
    parts = [opportunity.image_prompt.rstrip(" .,")]
    if variation:
        parts.append(variation)
    style = get_engine(cfg.image_engine).style_prompt(cfg)
    if style:
        parts.append(style)
    return ", ".join(parts)


def generated_prefix(engine_name: str) -> str:
    """Filename prefix for generated images, tagged by engine."""
    return f"generated_{engine_name}_"


def images_per_prompt(cfg: Settings) -> int:
    """Configured image count for the active engine."""
    return get_engine(cfg.image_engine).images_per_prompt(cfg)


def migrate_legacy_generated(scene_dir: Path) -> None:
    """Rename untagged generated_NN files from the Z-Image-only era.

    Idempotent and safe to call on every execute: already-tagged files and
    scenes with an existing target name are left untouched.
    """
    if not scene_dir.is_dir():
        return
    for path in sorted(scene_dir.iterdir()):
        if not path.is_file():
            continue
        match = LEGACY_GENERATED_PATTERN.match(path.name)
        if match is None:
            continue
        target = scene_dir / f"generated_zimage_{int(match.group(1)):02d}{path.suffix.lower()}"
        if target.exists():
            continue
        path.rename(target)
        logger.info("migrated %s -> %s", path.name, target.name)


def load_pipeline(cfg: Settings) -> object:
    """Load the configured engine once per run."""
    engine = get_engine(cfg.image_engine)
    logger.info("loading %s image engine", engine.name)
    return engine.load(cfg)


def generate_images_for_opportunity(
    pipe: object,
    opportunity: BrollOpportunity,
    cfg: Settings,
    dest_dir: Path,
    variations: list[str],
) -> list[Path]:
    import torch

    engine = get_engine(cfg.image_engine)
    prefix = generated_prefix(engine.name)
    dest_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index in range(engine.images_per_prompt(cfg)):
        variation = variations[index % len(variations)]
        prompt = build_prompt(opportunity, cfg, variation)
        seed = engine.configured_seed(cfg)
        if seed is None:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
        image = engine.generate_one(pipe, prompt, cfg, seed)
        out_path = dest_dir / f"{prefix}{index + 1:02d}.png"
        image.save(out_path)
        paths.append(out_path)
    return paths


__all__ = [
    "generated_prefix",
    "generate_images_for_opportunity",
    "images_per_prompt",
    "load_pipeline",
    "load_variations",
    "migrate_legacy_generated",
]

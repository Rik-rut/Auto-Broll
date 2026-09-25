"""Load the Qwen-Image 2.1 turbo pipeline via mmgp with pre-quantized int8 weights.

Artifacts and layout mirror Wan2GP's ``models/qwen21/qwen21_handler.py`` at the
pinned revisions below; the Viggle v0.2.1 LoRA is applied at runtime
(``offload.load_loras_into_model``) and never merged into the weights.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import NamedTuple

import torch
from accelerate import init_empty_weights
from mmgp import offload

from broll_agent.config import Settings
from broll_agent.qwen21.pipeline import Qwen21Pipeline
from broll_agent.qwen21.processor import load_processor
from broll_agent.qwen21.text_encoder import Qwen3VLTextEncoder, encoder_state_dict
from broll_agent.qwen21.transformer import QwenImage21Transformer2DModel
from broll_agent.qwen21.vae import AutoencoderKLQwenImage21
from broll_agent.shared.attention import get_default_attention_mode, resolve_attention_mode

logger = logging.getLogger(__name__)

_REPO_ID = "DeepBeepMeep/Qwen_image_2"
_REPO_REVISION = "8a136b04af66b7b20f689032f8d5a80f28add0cc"
_ENCODER_REPO_ID = "DeepBeepMeep/Ideogram4"
_ENCODER_REVISION = "a531af16b9cb570479356dae9e3f4a721adc3b7f"
_ENCODER_FOLDER = "Qwen3-VL-8B-Instruct"

_TRANSFORMER_WEIGHTS = "qwen_image_21_7B_int8_convrot.safetensors"
_TRANSFORMER_CONFIG = Path(__file__).resolve().parent / "configs" / "qwen_image_21_7B.json"
_VAE_WEIGHTS = "qwen_image_21/qwen_image_21_vae.safetensors"
_VAE_CONFIG = "qwen_image_21/vae_config.json"
_SCHEDULER_CONFIG = "qwen_image_21/scheduler_config.json"
_LORA_WEIGHTS = "loras/Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors"
_ENCODER_WEIGHTS = f"{_ENCODER_FOLDER}/Qwen3-VL-8B-Instruct_int8_convrot.safetensors"
_ENCODER_CONFIG = f"{_ENCODER_FOLDER}/config_legacy.json"
_PROCESSOR_FILES = (
    "tokenizer_legacy.json",
    "tokenizer_config_legacy.json",
    "preprocessor_config.json",
    "chat_template.jinja",
    "merges.txt",
    "vocab.json",
    "added_tokens.json",
    "special_tokens_map.json",
    "video_preprocessor_config.json",
)


class _ModelFile(NamedTuple):
    repo_id: str
    revision: str
    hf_path: str
    local_path: str


def _encoder_file(name: str) -> _ModelFile:
    return _ModelFile(
        _ENCODER_REPO_ID,
        _ENCODER_REVISION,
        f"{_ENCODER_FOLDER}/{name}",
        f"{_ENCODER_FOLDER}/{name}",
    )


_HF_FILES: tuple[_ModelFile, ...] = (
    _ModelFile(_REPO_ID, _REPO_REVISION, _TRANSFORMER_WEIGHTS, _TRANSFORMER_WEIGHTS),
    _ModelFile(_REPO_ID, _REPO_REVISION, _VAE_WEIGHTS, _VAE_WEIGHTS),
    _ModelFile(_REPO_ID, _REPO_REVISION, _VAE_CONFIG, _VAE_CONFIG),
    _ModelFile(_REPO_ID, _REPO_REVISION, _SCHEDULER_CONFIG, _SCHEDULER_CONFIG),
    _ModelFile(_REPO_ID, _REPO_REVISION, _LORA_WEIGHTS, _LORA_WEIGHTS),
    _ModelFile(_ENCODER_REPO_ID, _ENCODER_REVISION, _ENCODER_WEIGHTS, _ENCODER_WEIGHTS),
    _ModelFile(_ENCODER_REPO_ID, _ENCODER_REVISION, _ENCODER_CONFIG, _ENCODER_CONFIG),
    *(_encoder_file(name) for name in _PROCESSOR_FILES),
)


def ensure_model_files(model_dir: Path) -> None:
    """Download missing model files from HuggingFace (~18 GB on first use)."""
    from huggingface_hub import hf_hub_download

    missing = [
        entry for entry in _HF_FILES if not (model_dir / entry.local_path).exists()
    ]
    if not missing:
        logger.info("all model files present in %s", model_dir)
        return

    logger.info(
        "downloading %d missing Qwen-Image 2.1 file(s) (~18 GB total) into %s",
        len(missing),
        model_dir,
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    for entry in missing:
        logger.info("downloading %s from %s", entry.hf_path, entry.repo_id)
        hf_hub_download(
            repo_id=entry.repo_id,
            filename=entry.hf_path,
            revision=entry.revision,
            local_dir=str(model_dir),
        )
    logger.info("all model files ready")


def _lora_preprocessor(transformer: object):
    preprocessor = getattr(transformer, "preprocess_loras", None)
    if preprocessor is None:
        return None

    def wrapper(state_dict):
        return preprocessor(None, state_dict)

    return wrapper


def _setup_attention(cfg: Settings) -> None:
    if "_attention" in offload.shared_state:
        return
    mode = cfg.qwen_attention_backend
    offload.shared_state["_attention"] = (
        get_default_attention_mode() if mode == "auto" else resolve_attention_mode(mode)
    )


def _setup_triton_kernels(cfg: Settings) -> None:
    if not cfg.qwen_quanto_int8_kernel:
        logger.info("Triton int8 kernels disabled via config")
        return
    try:
        os.environ.setdefault("WAN2GP_QUANTO_INT8_KERNEL", "1")
        os.environ.setdefault("WAN2GP_QUANTO_INT8_ALLOW_RUNTIME_FALLBACK", "1")
        from broll_agent.shared.kernels.quanto_int8_inject import (
            maybe_enable_quanto_int8_kernel,
        )

        if maybe_enable_quanto_int8_kernel():
            logger.info("Triton int8 kernels enabled successfully")
        else:
            logger.warning("Triton int8 kernels not available — using standard quanto")
    except Exception as exc:
        logger.warning(
            "Triton int8 kernel injection failed (%s) — using standard quanto", exc
        )


def _setup_vram_management(pipeline: Qwen21Pipeline, cfg: Settings) -> None:
    pipe_dict = {
        "transformer": pipeline.transformer,
        "text_encoder": pipeline.text_encoder,
        "vae": pipeline.vae,
    }
    offload.profile(
        pipe_dict,
        profile_no=cfg.qwen_mmgp_profile,
        perc_reserved_mem_max=cfg.qwen_perc_reserved_mem_max,
        vram_safety_coefficient=cfg.qwen_vram_safety_coefficient,
        loras=["transformer"],
    )


def _setup_lora(pipeline: Qwen21Pipeline, model_dir: Path) -> None:
    lora_path = model_dir / _LORA_WEIGHTS
    transformer = pipeline.transformer
    offload.load_loras_into_model(
        transformer,
        str(lora_path),
        lora_multi=[1.0],
        activate_all_loras=True,
        preprocess_sd=_lora_preprocessor(transformer),
        split_linear_modules_map=getattr(transformer, "split_linear_modules_map", None),
    )
    errors = getattr(transformer, "_loras_errors", None) or []
    if errors:
        raise RuntimeError(
            "failed to load the Viggle turbo LoRA: "
            + "; ".join(message for _, message in errors)
        )
    logger.info("loaded Viggle turbo LoRA %s", lora_path.name)


def load_qwen21_pipeline(cfg: Settings) -> Qwen21Pipeline:
    model_dir = Path(cfg.qwen_model_dir)
    ensure_model_files(model_dir)

    logger.info("loading Qwen-Image 2.1 turbo via mmgp (model_dir=%s)", model_dir)

    transformer = offload.fast_load_transformers_model(
        str(model_dir / _TRANSFORMER_WEIGHTS),
        writable_tensors=False,
        modelClass=QwenImage21Transformer2DModel,
        defaultConfigPath=str(_TRANSFORMER_CONFIG),
    )

    with open(model_dir / _ENCODER_CONFIG, encoding="utf-8") as reader:
        encoder_config = json.load(reader)
    with torch.device("cpu"), init_empty_weights():
        text_encoder = Qwen3VLTextEncoder(encoder_config)
    text_encoder._config = encoder_config
    offload.load_model_data(
        text_encoder,
        str(model_dir / _ENCODER_WEIGHTS),
        writable_tensors=False,
        preprocess_sd=encoder_state_dict,
    )
    text_encoder.eval().requires_grad_(False)

    vae = offload.fast_load_transformers_model(
        str(model_dir / _VAE_WEIGHTS),
        writable_tensors=False,
        modelClass=AutoencoderKLQwenImage21,
        defaultConfigPath=str(model_dir / _VAE_CONFIG),
        default_dtype=torch.float32,
    )

    processor = load_processor(
        str(model_dir / _ENCODER_FOLDER),
        str(model_dir / _ENCODER_FOLDER / "preprocessor_config.json"),
        tokenizer_path=str(model_dir / _ENCODER_FOLDER / "tokenizer_legacy.json"),
        tokenizer_config_path=str(
            model_dir / _ENCODER_FOLDER / "tokenizer_config_legacy.json"
        ),
    )

    pipeline = Qwen21Pipeline(
        transformer,
        text_encoder,
        vae,
        processor,
        model_dir / _SCHEDULER_CONFIG,
    )
    _setup_vram_management(pipeline, cfg)
    _setup_lora(pipeline, model_dir)
    _setup_attention(cfg)
    _setup_triton_kernels(cfg)

    logger.info("Qwen-Image 2.1 turbo pipeline loaded successfully")
    return pipeline

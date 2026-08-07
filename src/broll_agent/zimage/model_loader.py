"""Load Z-Image Turbo pipeline via mmgp with pre-quantized int8 weights."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import torch
from accelerate import init_empty_weights
from diffusers import FlowMatchEulerDiscreteScheduler
from mmgp import offload
from transformers import AutoTokenizer, Qwen3ForCausalLM

from broll_agent.config import Settings
from broll_agent.zimage.pipeline import ZImagePipeline
from broll_agent.zimage.transformer import ZImageTransformer2DModel
from broll_agent.zimage.vae import AutoencoderKL

logger = logging.getLogger(__name__)

_TRANSFORMER_WEIGHTS_NAME = "ZImageTurbo_quanto_bf16_int8.safetensors"
_TEXT_ENCODER_WEIGHTS_NAME = "qwen3_quanto_bf16_int8.safetensors"
_VAE_WEIGHTS_NAME = "ZImageTurbo_VAE_bf16.safetensors"
_VAE_CONFIG_NAME = "ZImageTurbo_VAE_bf16_config.json"


def conv_state_dict(sd: dict) -> dict:
    if "x_embedder.weight" not in sd and "model.diffusion_model.x_embedder.weight" not in sd:
        return sd

    inverse_replace = {
        "final_layer.": "all_final_layer.2-1.",
        "x_embedder.": "all_x_embedder.2-1.",
        ".attention.out.bias": ".attention.to_out.0.bias",
        ".attention.k_norm.weight": ".attention.norm_k.weight",
        ".attention.q_norm.weight": ".attention.norm_q.weight",
        ".attention.out.weight": ".attention.to_out.0.weight",
    }

    out_sd: dict[str, torch.Tensor] = {}
    for key, tensor in sd.items():
        key = key.replace("model.diffusion_model.", "")
        new_key = key
        for ori_sub, orig_sub in inverse_replace.items():
            new_key = new_key.replace(ori_sub, orig_sub)
        out_sd[new_key] = tensor
    return out_sd


_FUSED_SPLIT_MAP = {
    "attention.to_qkv": {"mapped_modules": ("attention.to_q", "attention.to_k", "attention.to_v")},
    "attention.qkv": {"mapped_modules": ("attention.to_q", "attention.to_k", "attention.to_v")},
    "feed_forward.net.0.proj": {"mapped_modules": ("feed_forward.w3", "feed_forward.w1")},
    "feed_forward.net.2": {"mapped_modules": ("feed_forward.w2",)},
}


def load_zimage_pipeline(cfg: Settings) -> ZImagePipeline:
    model_dir = Path(cfg.zimage_model_dir)

    logger.info("loading Z-Image Turbo via mmgp (model_dir=%s)", model_dir)

    transformer_config_path = model_dir / "transformer" / "config.json"
    with open(transformer_config_path) as f:
        config = json.load(f)
    config.pop("_class_name", None)
    config.pop("_diffusers_version", None)

    def preprocess_sd(state_dict):
        return conv_state_dict(state_dict)

    with init_empty_weights():
        transformer = ZImageTransformer2DModel(**config)

    transformer_filename = model_dir / _TRANSFORMER_WEIGHTS_NAME
    kwargs_light = {
        "writable_tensors": False,
        "preprocess_sd": preprocess_sd,
        "fused_split_map": _FUSED_SPLIT_MAP,
    }
    offload.load_model_data(transformer, str(transformer_filename), **kwargs_light)
    transformer.to(torch.bfloat16)

    text_encoder_filename = model_dir / _TEXT_ENCODER_WEIGHTS_NAME
    text_encoder = offload.fast_load_transformers_model(
        str(text_encoder_filename),
        writable_tensors=True,
        modelClass=Qwen3ForCausalLM,
    )

    tokenizer_path = model_dir / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=True)

    vae_filename = model_dir / _VAE_WEIGHTS_NAME
    vae_config_path = model_dir / _VAE_CONFIG_NAME
    vae = offload.fast_load_transformers_model(
        str(vae_filename),
        writable_tensors=True,
        modelClass=AutoencoderKL,
        defaultConfigPath=str(vae_config_path),
        default_dtype=torch.float32,
    )

    scheduler_config_path = model_dir / "scheduler" / "scheduler_config.json"
    with open(scheduler_config_path, encoding="utf-8") as f:
        scheduler_config = json.load(f)
    scheduler = FlowMatchEulerDiscreteScheduler(**scheduler_config)

    pipeline = ZImagePipeline(
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
    )

    _setup_vram_management(pipeline, cfg)
    _setup_triton_kernels(cfg)

    logger.info("Z-Image Turbo pipeline loaded successfully")
    return pipeline


def _setup_vram_management(pipeline: ZImagePipeline, cfg: Settings) -> None:
    pipe_dict = {
        "text_encoder": pipeline.text_encoder,
        "transformer": pipeline.transformer,
        "vae": pipeline.vae,
    }
    offload.profile(
        pipe_dict,
        profile_no=cfg.zimage_mmgp_profile,
        perc_reserved_mem_max=cfg.zimage_perc_reserved_mem_max,
        vram_safety_coefficient=cfg.zimage_vram_safety_coefficient,
    )


def _setup_triton_kernels(cfg: Settings) -> None:
    if not cfg.zimage_quanto_int8_kernel:
        logger.info("Triton int8 kernels disabled via config")
        return
    try:
        os.environ.setdefault("WAN2GP_QUANTO_INT8_KERNEL", "1")
        os.environ.setdefault("WAN2GP_QUANTO_INT8_ALLOW_RUNTIME_FALLBACK", "1")
        from broll_agent.shared.kernels.quanto_int8_inject import maybe_enable_quanto_int8_kernel
        success = maybe_enable_quanto_int8_kernel()
        if success:
            logger.info("Triton int8 kernels enabled successfully")
        else:
            logger.warning("Triton int8 kernels not available — falling back to standard quanto")
    except Exception as e:
        logger.warning("Triton int8 kernel injection failed (%s) — falling back to standard quanto", e)

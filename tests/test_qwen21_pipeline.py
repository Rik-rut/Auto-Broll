from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from broll_agent.qwen21 import pipeline as pipeline_module
from broll_agent.qwen21.pipeline import (
    VIGGLE_SIGMAS,
    Qwen21Pipeline,
    build_viggle_scheduler,
    round_to_alignment,
)

SCHEDULER_CONFIG = {
    "base_image_seq_len": 256,
    "base_shift": 0.5,
    "max_image_seq_len": 8192,
    "max_shift": 0.9,
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": 0.02,
    "use_dynamic_shifting": True,
}


def test_viggle_sigma_tables() -> None:
    assert VIGGLE_SIGMAS[4] == (1.0, 0.75, 0.5, 0.25)
    assert VIGGLE_SIGMAS[5] == (1.0, 0.875, 0.75, 0.5, 0.25)
    assert VIGGLE_SIGMAS[6] == (1.0, 0.9375, 0.875, 0.75, 0.5, 0.25)


def test_build_viggle_scheduler_disables_terminal_shift() -> None:
    scheduler = build_viggle_scheduler(SCHEDULER_CONFIG, 6, token_count=3520, device="cpu")
    assert scheduler.config.shift_terminal is None
    assert len(scheduler.timesteps) == 6
    assert float(scheduler.sigmas[0]) == 1.0


def test_build_viggle_scheduler_rejects_unknown_steps() -> None:
    with pytest.raises(ValueError, match="Viggle turbo supports"):
        build_viggle_scheduler(SCHEDULER_CONFIG, 7, token_count=3520, device="cpu")


def test_round_to_alignment() -> None:
    assert round_to_alignment(720) == 704
    assert round_to_alignment(1280) == 1280
    assert round_to_alignment(32) == 32
    with pytest.raises(ValueError):
        round_to_alignment(16)


def test_generate_rejects_guidance_and_steps() -> None:
    pipe = Qwen21Pipeline.__new__(Qwen21Pipeline)
    with pytest.raises(ValueError, match="guidance_scale=1.0"):
        pipe.generate("x", width=64, height=64, guidance_scale=4.0)
    with pytest.raises(ValueError, match="Viggle turbo supports"):
        pipe.generate("x", width=64, height=64, num_inference_steps=7)
    with pytest.raises(ValueError, match="multiples of 32"):
        pipe.generate("x", width=48, height=64)


class _FakeVAE:
    dtype = torch.float32

    def __init__(self) -> None:
        self.config = SimpleNamespace(
            latents_mean=[0.0] * 64,
            latents_std=[1.0] * 64,
        )
        self.use_slicing = False

    def configure_tiling(self, tile_setting, width, height, batch_size=1, reference_pixels=0):
        return None

    def clear_cache(self) -> None:
        return None

    def decode_to_cpu_uint8(self, latents, output_channels=None):
        return torch.zeros((1, 3, 1, 96, 64), dtype=torch.uint8)


class _FakeTransformer:
    def __init__(self, token_count: int) -> None:
        self.token_count = token_count
        self.pos_embed = SimpleNamespace(freqs=[])
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return (torch.zeros(1, self.token_count, 64),)


def _fake_pipeline(token_count: int = 24) -> tuple[Qwen21Pipeline, _FakeTransformer]:
    pipe = Qwen21Pipeline.__new__(Qwen21Pipeline)
    transformer = _FakeTransformer(token_count)
    pipe.transformer = transformer
    pipe.text_encoder = SimpleNamespace()
    pipe.vae = _FakeVAE()
    pipe.text_encoder_cache = pipeline_module.TextEncoderCache()
    pipe.scheduler_config = SCHEDULER_CONFIG
    pipe.prompt_template_t2i = "{}"
    pipe._drop_idx = 0
    pipe._img_token_id = 0
    pipe.processor = SimpleNamespace(config={})
    pipe.tokenizer = SimpleNamespace()
    pipe.vae_scale_factor = 16

    def fake_embeds(prompt, device):
        return (
            torch.zeros(1, 5, 64),
            torch.ones(1, 5, dtype=torch.long),
            torch.zeros(1, 5, dtype=torch.bool),
        )

    pipe._get_qwen_prompt_embeds = fake_embeds
    return pipe, transformer


def test_generate_smoke_produces_expected_size() -> None:
    pipe, transformer = _fake_pipeline()
    image = pipe.generate(
        "a puppy on a wooden floor",
        seed=1,
        width=64,
        height=96,
        num_inference_steps=6,
        device="cpu",
    )
    assert image.size == (64, 96)
    assert len(transformer.calls) == 6
    assert transformer.calls[0]["timestep"].shape == (1,)

"""Text-to-image Qwen-Image 2.1 pipeline with the Viggle turbo schedule.

Adapted from Wan2GP ``models/qwen21/prompt.py`` and ``models/qwen21/pipeline.py``
@ 2345ae148f82740f66e82c41292dbbdd592e713d. Editing, inpainting, outpainting,
RGBA output, LanPaint, KV-cache and application progress plumbing are removed:
this port generates b-roll text-to-image only.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from mmgp import offload
from PIL import Image

from broll_agent.shared.text_encoder_cache import TextEncoderCache

from .compat import check_abort

logger = logging.getLogger(__name__)

ALIGNMENT = 32
IMG_TOKENS_PER_SLOT = 4
VIGGLE_SIGMAS: dict[int, tuple[float, ...]] = {
    4: (1.0, 0.75, 0.5, 0.25),
    5: (1.0, 0.875, 0.75, 0.5, 0.25),
    6: (1.0, 0.9375, 0.875, 0.75, 0.5, 0.25),
}


def round_to_alignment(value: int, alignment: int = ALIGNMENT) -> int:
    """Round a requested dimension down to the model's required alignment."""
    rounded = int(value) // alignment * alignment
    if rounded < alignment:
        raise ValueError(f"{value} is smaller than the {alignment}px minimum")
    return rounded


def build_viggle_scheduler(
    scheduler_config: dict,
    num_inference_steps: int,
    token_count: int,
    device: torch.device | str = "cuda",
) -> FlowMatchEulerDiscreteScheduler:
    """Build the Viggle turbo scheduler: raw sigma nodes + terminal shift off."""
    if num_inference_steps not in VIGGLE_SIGMAS:
        raise ValueError(
            f"Viggle turbo supports {sorted(VIGGLE_SIGMAS)} inference steps, "
            f"got {num_inference_steps}"
        )
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(
        scheduler_config, shift_terminal=None
    )
    config = scheduler.config
    sigmas = VIGGLE_SIGMAS[num_inference_steps]
    if config.use_dynamic_shifting:
        slope = (config.max_shift - config.base_shift) / (
            config.max_image_seq_len - config.base_image_seq_len
        )
        mu = token_count * slope + config.base_shift - slope * config.base_image_seq_len
        scheduler.set_timesteps(num_inference_steps, device=device, sigmas=sigmas, mu=mu)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, sigmas=sigmas)
    logger.debug(
        "viggle raw sigmas=%s applied=%s",
        list(sigmas),
        [round(value, 6) for value in scheduler.sigmas.tolist()],
    )
    return scheduler


class Qwen21Pipeline:
    """Qwen3-VL conditioning -> Viggle turbo denoise -> VAE decode to PIL."""

    def __init__(self, transformer, text_encoder, vae, processor, scheduler_config: str | Path):
        self.transformer, self.text_encoder, self.vae = transformer, text_encoder, vae
        self.processor, self.tokenizer = processor, processor.tokenizer
        self.text_encoder_cache = TextEncoderCache()
        self.vae_scale_factor = 16
        with open(scheduler_config, encoding="utf-8") as reader:
            self.scheduler_config = json.load(reader)
        self.sys_prompt = "Comprehend and analyze the provided prompt."
        system = f"<|im_start|>system\n{self.sys_prompt}<|im_end|>\n"
        self.prompt_template_t2i = (
            system + "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        )
        self._drop_idx = len(self.tokenizer.encode(system))
        self._img_token_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")

    def _vae_stats(self, latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = latents.new_tensor(self.vae.config.latents_mean).view(1, -1, 1, 1, 1)
        std = latents.new_tensor(self.vae.config.latents_std).view(1, -1, 1, 1, 1)
        return mean, std

    def _extract_masked_hidden(
        self, hidden_states: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        return torch.split(selected, valid_lengths.tolist(), dim=0)

    def _get_qwen_prompt_embeds(
        self, prompt: list[str], device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prompts = [self.prompt_template_t2i.format(text) for text in prompt]
        model_inputs = self.processor(
            text=prompts, padding=True, padding_side="left", return_tensors="pt"
        ).to(device)
        forward_kwargs = {
            "input_ids": model_inputs.input_ids,
            "attention_mask": model_inputs.attention_mask,
            "output_hidden_states": True,
            "use_cache": False,
            "logits_to_keep": 1,
        }
        if hasattr(model_inputs, "mm_token_type_ids"):
            forward_kwargs["mm_token_type_ids"] = model_inputs.mm_token_type_ids
        text_model = getattr(self.text_encoder.model, "language_model", self.text_encoder.model)
        # Qwen Image 2.1 consumes the final layer before the final RMSNorm.
        handle = text_model.norm.register_forward_hook(
            lambda module, args, output: args[0]
        )
        try:
            outputs = self.text_encoder(**forward_kwargs)
        finally:
            handle.remove()
        hidden_states = outputs.hidden_states[-1]
        split_hidden_states = list(
            self._extract_masked_hidden(hidden_states, model_inputs.attention_mask)
        )
        split_hidden_states = [embedding[self._drop_idx:] for embedding in split_hidden_states]
        image_pad_mask = [
            sample_ids[sample_mask.bool()] == self._img_token_id
            for sample_ids, sample_mask in zip(
                model_inputs.input_ids, model_inputs.attention_mask
            )
        ]
        image_pad_mask = [mask[self._drop_idx:] for mask in image_pad_mask]
        attn_mask_list = [
            torch.ones(embedding.size(0), dtype=torch.long, device=embedding.device)
            for embedding in split_hidden_states
        ]
        max_seq_len = max(embedding.size(0) for embedding in split_hidden_states)
        prompt_embeds = torch.stack(
            [
                torch.cat([item, item.new_zeros(max_seq_len - item.size(0), item.size(1))])
                for item in split_hidden_states
            ]
        )
        encoder_attention_mask = torch.stack(
            [
                torch.cat([item, item.new_zeros(max_seq_len - item.size(0))])
                for item in attn_mask_list
            ]
        )
        image_pad_mask = torch.stack(
            [
                torch.cat([item, item.new_zeros(max_seq_len - item.size(0))])
                for item in image_pad_mask
            ]
        )
        return prompt_embeds, encoder_attention_mask, image_pad_mask

    def encode_prompt(
        self, prompt: str | list[str], device: torch.device | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = device or torch.device("cuda")
        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [" " if not text else text for text in prompt]
        return self._get_qwen_prompt_embeds(prompt, device)

    def _encode_conditioning(
        self, prompts: list[str], device: torch.device
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        context = (
            id(self.text_encoder),
            getattr(self.text_encoder, "_model_dtype", None),
            id(self.tokenizer),
            self.prompt_template_t2i,
            self._drop_idx,
            self._img_token_id,
            json.dumps(self.processor.config, sort_keys=True),
        )

        def encode(prompts_to_encode: list[str]):
            check_abort()
            result = self.encode_prompt(prompt=prompts_to_encode[0], device=device)
            check_abort()
            return [result]

        return self.text_encoder_cache.encode(
            encode, prompts, device=device, cache_keys=[(prompt, context) for prompt in prompts]
        )

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        seed: int = 1,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 6,
        guidance_scale: float = 1.0,
        device: torch.device | str | None = None,
    ) -> Image.Image:
        if guidance_scale != 1.0:
            raise ValueError(
                "qwen21 supports guidance_scale=1.0 only; the Viggle turbo schedule has no CFG"
            )
        if num_inference_steps not in VIGGLE_SIGMAS:
            raise ValueError(
                f"Viggle turbo supports {sorted(VIGGLE_SIGMAS)} inference steps, "
                f"got {num_inference_steps}"
            )
        if width % ALIGNMENT or height % ALIGNMENT:
            raise ValueError("Qwen-Image 2.1 dimensions must be multiples of 32")

        device = torch.device(device or "cuda")
        batch_size = 1
        generator = torch.Generator(device=device).manual_seed(seed)
        self.vae.configure_tiling(None, width, height, batch_size, reference_pixels=0)
        self.vae.use_slicing = True

        branches = [
            {
                "encoder_hidden_states": embeds,
                "encoder_hidden_states_mask": mask,
                "img_mask": slots,
            }
            for embeds, mask, slots in self._encode_conditioning([prompt], device)
        ]
        dtype = branches[0]["encoder_hidden_states"].dtype
        latents = (
            torch.randn(
                (batch_size, 64, height // 16, width // 16),
                device=device,
                dtype=dtype,
                generator=generator,
            )
            .flatten(2)
            .transpose(1, 2)
        )
        shapes = [(1, height // 16, width // 16)]
        for branch in branches:
            slots = branch["img_mask"]
            branch["img_mask"] = torch.cat(
                [slots, slots.new_ones(slots.shape[0], latents.shape[1] // IMG_TOKENS_PER_SLOT)],
                dim=1,
            )
            branch["img_shapes"] = [shapes] * batch_size

        scheduler = build_viggle_scheduler(
            self.scheduler_config, num_inference_steps, latents.shape[1], device=device
        )
        try:
            for step, timestep in enumerate(scheduler.timesteps):
                check_abort()
                offload.set_step_no_for_lora(self.transformer, step)
                item = {
                    **branches[0],
                    "hidden_states": latents,
                    "timestep": timestep.expand(batch_size).to(dtype) / 1000,
                    "kv_cache_mode": None,
                    "return_dict": False,
                }
                prediction = self.transformer(**item)[0][:, -latents.shape[1]:]
                latents = scheduler.step(prediction, timestep, latents, return_dict=False)[0]

            latents = latents.transpose(1, 2).reshape(
                batch_size, 64, 1, height // 16, width // 16
            ).to(self.vae.dtype)
            mean, std = self._vae_stats(latents)
            latents = latents * std + mean
            output = self.vae.decode_to_cpu_uint8([latents], output_channels=3)
            pixels = output[0, :, 0].permute(1, 2, 0).numpy()
            return Image.fromarray(pixels)
        finally:
            self.vae.clear_cache()
            self.transformer.pos_embed.freqs = [
                freq.cpu() for freq in self.transformer.pos_embed.freqs
            ]

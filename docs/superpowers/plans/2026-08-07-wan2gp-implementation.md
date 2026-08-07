# Wan2GP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace torchao-based int8 quantization with Wan2GP's native int8 inference pipeline using mmgp, including pre-quantized transformer + text encoder, custom Triton kernels, and multi-backend attention.

**Architecture:** Port 4 core model files from Wan2GP's `models/z_image/` into a new `src/broll_agent/zimage/` package, plus shared infrastructure (attention, Triton kernels, text encoder cache). Strip ControlNet/LoRA/NAG/Unified Sampler. Rewrite `generate_lightning.py` to use the new model loader. Remove torchao dependency.

**Tech Stack:** mmgp 3.7.10, optimum-quanto, triton, diffusers >=0.36, accelerate, transformers >=4.44, torch >=2.4

## Global Constraints

- Package manager: `uv`, exclusively. No `pip install`.
- All tunables in `.env` via `config.py` Settings — nothing hardcoded.
- Model files stay in `z-image model/` directory.
- Public interface of `generate_lightning.py` unchanged (`load_pipeline`, `generate_images_for_opportunity`, `build_prompt`, `GENERATED_PREFIX`).
- `cli.py` and `test_generate.py` must not need changes beyond config field removals.
- Type-hint everything. Run `uv run ruff check .` after each task.
- No secrets in prompts or logs.

---

### Task 1: Update Dependencies

**Files:**
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: nothing
- Produces: updated dependency list for `uv sync`

- [ ] **Step 1: Remove torchao, add mmgp + optimum-quanto + triton**

In `pyproject.toml`, remove `"torchao>=0.18.0"` from dependencies and add:

```toml
    "mmgp==3.7.10",
    "optimum-quanto>=0.2.4",
    "triton; sys_platform == 'linux'",
    "sageattention>=2.0",
```

Note: triton is Linux-only for now. The kernel injection code has graceful fallback if triton is unavailable.

- [ ] **Step 2: Run uv sync to verify dependencies resolve**

Run: `uv sync`
Expected: resolves without errors. mmgp 3.7.10 should install from PyPI.

- [ ] **Step 3: Run ruff check**

Run: `uv run ruff check .`
Expected: no new errors (existing code unchanged).

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: replace torchao with mmgp + optimum-quanto + triton dependencies"
```

---

### Task 2: Port Text Encoder Cache

**Files:**
- Create: `src/broll_agent/shared/__init__.py`
- Create: `src/broll_agent/shared/text_encoder_cache.py`

**Interfaces:**
- Consumes: nothing
- Produces: `TextEncoderCache` class used by `pipeline.py` (Task 5)

- [ ] **Step 1: Create shared package init**

Create `src/broll_agent/shared/__init__.py` as an empty file.

- [ ] **Step 2: Copy TextEncoderCache from Wan2GP**

Copy `C:\Users\Biboy\Desktop\Wan2GP\shared\utils\text_encoder_cache.py` verbatim to `src/broll_agent/shared/text_encoder_cache.py`. No changes needed — it's self-contained (only depends on `torch` and stdlib).

- [ ] **Step 3: Run ruff check**

Run: `uv run ruff check .`
Expected: passes.

- [ ] **Step 4: Commit**

```bash
git add src/broll_agent/shared/__init__.py src/broll_agent/shared/text_encoder_cache.py
git commit -m "feat: port TextEncoderCache from Wan2GP"
```

---

### Task 3: Port Attention Module

**Files:**
- Create: `src/broll_agent/shared/attention.py`

**Interfaces:**
- Consumes: `mmgp.offload.shared_state`
- Produces: `pay_attention()`, `resolve_attention_mode()`, `attention_config_shared_state()` — used by `transformer.py` (Task 4)

- [ ] **Step 1: Copy attention.py from Wan2GP with adaptations**

Copy `C:\Users\Biboy\Desktop\Wan2GP\shared\attention.py` to `src/broll_agent/shared/attention.py` with these changes:

1. **Remove the `sage2_core` relative import** (lines 72-83). Replace the entire `try: from .sage2_core import ...` block with:
```python
sageattn2 = None
sage2_supported = False
sageattn_attention_mask_support_reason = lambda *args, **kwargs: "SageAttention 2 is unavailable"
```

2. **Remove sage3 imports and wrapper** (lines 124-165). Replace with:
```python
sageattn3 = None
```

3. **Remove `block_sparse_sage2_attn_cuda` / `radial` import** (lines 61-69). Replace with:
```python
block_sparse_sage2_attn_cuda = None
```

4. **Remove `chipmunk` branch** from `pay_attention()` (lines 394-396):
```python
    if attn == "chipmunk":
        from src.chipmunk.modules import SparseDiffMlp, SparseDiffAttn
        from src.chipmunk.util import LayerCounter, GLOBAL_CONFIG
```

5. **Simplify `get_attention_modes()`** — remove `sage3` and `radial` checks. The function should only check for: sdpa (always), flash (if flash_attn installed), xformers (if installed), sage (if sageattn_varlen installed), sage2 (if sageattn2 installed + sageattention v2).

6. **Simplify `get_supported_attention_modes()`** — remove `sage3` handling.

The core `pay_attention()` function stays intact — it's the main entry point called by the transformer's `Attention.forward()`.

- [ ] **Step 2: Run ruff check**

Run: `uv run ruff check .`
Expected: passes (fix any import issues).

- [ ] **Step 3: Commit**

```bash
git add src/broll_agent/shared/attention.py
git commit -m "feat: port attention module from Wan2GP with multi-backend support"
```

---

### Task 4: Port Transformer

**Files:**
- Create: `src/broll_agent/zimage/__init__.py`
- Create: `src/broll_agent/zimage/transformer.py`

**Interfaces:**
- Consumes: `shared.attention.pay_attention`, `diffusers.models.normalization.RMSNorm`
- Produces: `ZImageTransformer2DModel` class used by `model_loader.py` (Task 6) and `pipeline.py` (Task 5)

- [ ] **Step 1: Create zimage package init**

Create `src/broll_agent/zimage/__init__.py` as an empty file.

- [ ] **Step 2: Copy transformer from Wan2GP with stripping**

Copy `C:\Users\Biboy\Desktop\Wan2GP\models\z_image\z_image_transformer2d.py` to `src/broll_agent/zimage/transformer.py` with these changes:

1. **Fix imports** — change:
```python
from shared.attention import pay_attention
```
to:
```python
from broll_agent.shared.attention import pay_attention
```

2. **Remove `ModuleWrapper` class** (lines 18-23) — only used for ControlNet.

3. **Remove NAG handling from `Attention.forward()`** (lines 141-172). Replace the entire `if NAG is not None:` block with just the `else` branch:
```python
    def forward(self, h_list: list, freqs_cis: torch.Tensor, NAG=None) -> torch.Tensor:
        h = h_list.pop()
        query = self.to_q(h)
        key = self.to_k(h)
        value = self.to_v(h); del h
        query = query.unflatten(-1, (self.n_heads, -1))
        key = key.unflatten(-1, (self.n_heads, -1))
        value = value.unflatten(-1, (self.n_heads, -1))
        if self.norm_q is not None:
            query = self.norm_q(query)
        if self.norm_k is not None:
            key = self.norm_k(key)
        if freqs_cis is not None:
            q_list = [query]; del query
            query = apply_rotary_emb_inplace(q_list, freqs_cis)
            k_list = [key]; del key
            key = apply_rotary_emb_inplace(k_list, freqs_cis)
        dtype = query.dtype
        x_list = [query, key, value]
        del query, key, value
        out = pay_attention(x_list).flatten(2, 3).to(dtype)
        return self.to_out(out)
```

4. **Remove `ZImageControlTransformerBlock`** (lines 257-292) — ControlNet only.

5. **Remove `BaseZImageTransformerBlock`** (lines 296-319) — ControlNet only.

6. **Simplify `ZImageTransformer2DModel.__init__()`** — remove all control-related parameters and logic:
   - Remove: `control_layers_places`, `control_refiner_layers_places`, `control_in_dim`, `add_control_noise_refiner`, `enable_control`, `use_separate_control_refiner` parameters
   - Remove: all `self.control_*` attributes
   - Remove: `self._control_noise_uses_dedicated_layers`
   - Remove: `self.control_layers_mapping`, `self.control_refiner_layers_mapping`
   - Remove: the `if enable_control:` branches for noise_refiner and layers (keep the `else` branches)
   - Remove: `adapt_control_model()` method
   - Remove: `prepare_forward_control_1_0()`, `prepare_forward_control_2_0_refiner()`, `prepare_forward_control_2_0_layers()` methods
   - Remove: `has_control` property

7. **Simplify `forward()`** — remove all control processing:
   - Remove `control_context_list`, `control_context_scale` parameters
   - Remove all control-related processing blocks (control_V2, refiner_hints, etc.)
   - Remove NAG processing (the `NAG_index`, `neg_cap_embedded`, etc.)
   - Keep: list-based processing, noise_refiner, context_refiner, main layers, final layer, unpatchify
   - The simplified forward signature:
   ```python
   def forward(
       self,
       x_list: List[torch.Tensor],
       t,
       cap_feats_list: List[torch.Tensor],
       patch_size=2,
       f_patch_size=1,
       callback=None,
       pipeline=None,
   ):
   ```

8. **Remove `preprocess_loras()`** method (lines 389-404) — LoRA only.

9. **Keep all VRAM optimizations intact:**
   - In-place operations (`.add_()`, `.mul_()`, `.tanh_()`)
   - Chunked FFN (`_apply_ffn_chunked`)
   - List-based tensor passing (`.pop()` to free memory early)
   - RoPE with precomputed freqs_cis
   - `apply_rotary_emb_inplace`

- [ ] **Step 3: Run ruff check**

Run: `uv run ruff check .`
Expected: passes.

- [ ] **Step 4: Commit**

```bash
git add src/broll_agent/zimage/__init__.py src/broll_agent/zimage/transformer.py
git commit -m "feat: port ZImageTransformer2DModel from Wan2GP with ControlNet/LoRA/NAG stripped"
```

---

### Task 5: Port VAE and Pipeline

**Files:**
- Create: `src/broll_agent/zimage/vae.py`
- Create: `src/broll_agent/zimage/pipeline.py`
- Create: `src/broll_agent/zimage/pipeline_output.py`

**Interfaces:**
- Consumes: `transformer.ZImageTransformer2DModel`, `shared.text_encoder_cache.TextEncoderCache`, `diffusers` components
- Produces: `ZImagePipeline` class used by `model_loader.py` (Task 6)

- [ ] **Step 1: Copy VAE from Wan2GP**

Copy `C:\Users\Biboy\Desktop\Wan2GP\models\z_image\autoencoder_kl.py` verbatim to `src/broll_agent/zimage/vae.py`. No changes needed — it's a self-contained diffusers model with tiling/slicing support.

- [ ] **Step 2: Copy pipeline_output.py**

Copy `C:\Users\Biboy\Desktop\Wan2GP\models\z_image\pipeline_output.py` verbatim to `src/broll_agent/zimage/pipeline_output.py`. No changes needed.

- [ ] **Step 3: Copy pipeline from Wan2GP with stripping**

Copy `C:\Users\Biboy\Desktop\Wan2GP\models\z_image\pipeline_z_image.py` to `src/broll_agent/zimage/pipeline.py` with these changes:

1. **Fix imports** — replace:
```python
from mmgp import offload
from .z_image_transformer2d import ZImageTransformer2DModel
from .unified_sampler import UnifiedSampler
from .pipeline_output import ZImagePipelineOutput
from shared.utils.utils import get_outpainting_frame_location, resize_lanczos, calculate_new_dimensions, convert_image_to_tensor, fit_image_into_canvas
from shared.utils.loras_mutipliers import update_loras_slists
from shared.utils.text_encoder_cache import TextEncoderCache
```
with:
```python
from mmgp import offload
from .transformer import ZImageTransformer2DModel
from .pipeline_output import ZImagePipelineOutput
from broll_agent.shared.text_encoder_cache import TextEncoderCache
```

2. **Remove unified solver code** — delete:
   - `_UNIFIED_SOLVERS`, `_UNIFIED_PRESET_GAP` constants
   - `_UnifiedSamplerInterrupted` class
   - `_is_unified_solver()`, `_resolve_unified_sampler_config()` functions
   - The entire `if use_unified:` branch in `__call__()` (lines 678-883)

3. **Remove ControlNet support from `__call__()`** — remove these parameters:
   - `control_image`, `inpaint_mask`, `control_context_scale`, `input_ref_images`
   - Remove the entire control latent processing block (lines 619-676)
   - Remove `control_latent_input` references from transformer calls

4. **Remove LoRA support** — remove:
   - `loras_slists` parameter
   - `self.transformer.loras_slists = loras_slists` assignment
   - `offload.set_step_no_for_lora()` calls in the denoising loop
   - `update_loras_slists()` calls

5. **Remove NAG** — remove:
   - `NAG_scale`, `NAG_tau`, `NAG_alpha` parameters
   - NAG dict construction and `NAG["neg_feats"]` logic
   - `NAG=kwargs["NAG"]` from transformer calls

6. **Remove VAE upsampler** — remove:
   - `vae_upsampler`, `vae_upsampler_seed`, `vae_upsampler_progress_callback` parameters
   - The `if vae_upsampler is not None:` block in decode

7. **Remove `callback` progress reporting** — the `callback(-1, None, True, ...)` and `callback(i, latents_preview[0], False)` calls. Keep `callback_on_step_end` (standard diffusers API).

8. **Simplify `__call__()` signature** to:
```python
def __call__(
    self,
    prompt: Union[str, List[str]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    sigmas: Optional[List[float]] = None,
    guidance_scale: float = 0.0,
    cfg_normalization: bool = False,
    cfg_truncation: float = 1.0,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[List[torch.FloatTensor]] = None,
    negative_prompt_embeds: Optional[List[torch.FloatTensor]] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 512,
):
```

9. **Simplify the denoising loop** — the `else` branch (default solver, lines 884-972) becomes the only path. Remove:
   - `callback` parameter usage (use `callback_on_step_end` instead)
   - `pipeline=self` from transformer calls
   - `**kwargs` from transformer calls (no NAG passed)
   - `control_context_list` from transformer calls

10. **Keep `model_cpu_offload_seq = "text_encoder->transformer->vae"`** — mmgp uses this for sequential offloading.

- [ ] **Step 4: Run ruff check**

Run: `uv run ruff check .`
Expected: passes.

- [ ] **Step 5: Commit**

```bash
git add src/broll_agent/zimage/vae.py src/broll_agent/zimage/pipeline.py src/broll_agent/zimage/pipeline_output.py
git commit -m "feat: port ZImagePipeline and VAE from Wan2GP with ControlNet/LoRA/NAG/Unified stripped"
```

---

### Task 6: Port Triton Kernel Injection

**Files:**
- Create: `src/broll_agent/shared/kernels/__init__.py`
- Create: `src/broll_agent/shared/kernels/quanto_int8_inject.py`
- Create: `src/broll_agent/shared/kernels/quanto_int8_triton.py`

**Interfaces:**
- Consumes: `optimum.quanto`, `triton` (optional)
- Produces: `inject_quanto_int8_kernels()` function called by `model_loader.py` (Task 7)

- [ ] **Step 1: Create kernels package init**

Create `src/broll_agent/shared/kernels/__init__.py` as an empty file.

- [ ] **Step 2: Copy quanto_int8_triton.py verbatim**

Copy `C:\Users\Biboy\Desktop\Wan2GP\shared\kernels\quanto_int8_triton.py` verbatim to `src/broll_agent/shared/kernels/quanto_int8_triton.py`. No changes needed — it's self-contained (only depends on `torch` and `triton`, with graceful fallback if triton unavailable).

- [ ] **Step 3: Copy quanto_int8_inject.py verbatim**

Copy `C:\Users\Biboy\Desktop\Wan2GP\shared\kernels\quanto_int8_inject.py` verbatim to `src/broll_agent/shared/kernels/quanto_int8_inject.py`. No changes needed — it uses `WAN2GP_QUANTO_INT8_KERNEL` env var which we'll keep for compatibility. It imports from `quanto_int8_triton` using a relative import that needs to work within our package structure.

Verify the import at the top of `quanto_int8_inject.py` — if it uses `from .quanto_int8_triton import ...` it will work as-is. If it uses a different import path, fix it to `from broll_agent.shared.kernels.quanto_int8_triton import ...`.

- [ ] **Step 4: Run ruff check**

Run: `uv run ruff check .`
Expected: passes.

- [ ] **Step 5: Commit**

```bash
git add src/broll_agent/shared/kernels/
git commit -m "feat: port Triton kernel injection for int8 matmul acceleration from Wan2GP"
```

---

### Task 7: Build Model Loader

**Files:**
- Create: `src/broll_agent/zimage/model_loader.py`

**Interfaces:**
- Consumes: `transformer.ZImageTransformer2DModel`, `pipeline.ZImagePipeline`, `vae.AutoencoderKL`, `shared.kernels.quanto_int8_inject`, `mmgp.offload`, `config.Settings`
- Produces: `load_zimage_pipeline(cfg: Settings) -> ZImagePipeline` — the main entry point used by `generate_lightning.py` (Task 8)

- [ ] **Step 1: Write the model loader**

Create `src/broll_agent/zimage/model_loader.py` based on Wan2GP's `z_image_main.py` `model_factory.__init__()`, adapted for our Settings:

```python
"""Load Z-Image Turbo pipeline via mmgp with pre-quantized int8 weights."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import accelerate
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

_TRANSFORMER_CONFIG_NAME = "z_image_turbo_transformer_config.json"
_TRANSFORMER_WEIGHTS_NAME = "ZImageTurbo_quanto_bf16_int8.safetensors"
_TEXT_ENCODER_WEIGHTS_NAME = "qwen3_quanto_bf16_int8.safetensors"
_VAE_WEIGHTS_NAME = "ZImageTurbo_VAE_bf16.safetensors"
_VAE_CONFIG_NAME = "ZImageTurbo_VAE_bf16_config.json"
_SCHEDULER_CONFIG_NAME = "ZImageTurbo_scheduler_config.json"


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
    with open(transformer_config_path, "r") as f:
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
    with open(scheduler_config_path, "r", encoding="utf-8") as f:
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
```

- [ ] **Step 2: Run ruff check**

Run: `uv run ruff check .`
Expected: passes.

- [ ] **Step 3: Commit**

```bash
git add src/broll_agent/zimage/model_loader.py
git commit -m "feat: add model loader with mmgp int8 loading and VRAM management"
```

---

### Task 8: Update Config

**Files:**
- Modify: `src/broll_agent/config.py`

**Interfaces:**
- Consumes: nothing
- Produces: updated `Settings` class with new mmgp fields, removed torchao fields

- [ ] **Step 1: Update Settings class**

In `src/broll_agent/config.py`, make these changes:

1. **Remove** these fields:
   - `zimage_dtype: str = "bfloat16"`
   - `zimage_enable_cpu_offload: bool = False`

2. **Add** these fields after `zimage_seed`:
```python
    zimage_quanto_int8_kernel: bool = True
    zimage_mmgp_profile: int = 0
    zimage_perc_reserved_mem_max: float = 0.9
    zimage_vram_safety_coefficient: float = 0.1
    zimage_attention_backend: str = "auto"
```

- [ ] **Step 2: Run ruff check**

Run: `uv run ruff check .`
Expected: passes. Fix any references to removed fields in other files.

- [ ] **Step 3: Commit**

```bash
git add src/broll_agent/config.py
git commit -m "feat: update config with mmgp settings, remove torchao fields"
```

---

### Task 9: Rewrite generate_lightning.py

**Files:**
- Modify: `src/broll_agent/generate_lightning.py`

**Interfaces:**
- Consumes: `zimage.model_loader.load_zimage_pipeline`, `config.Settings`, `models.BrollOpportunity`
- Produces: same public API (`load_pipeline`, `generate_images_for_opportunity`, `build_prompt`, `GENERATED_PREFIX`)

- [ ] **Step 1: Rewrite load_pipeline to use model_loader**

Replace the `load_pipeline` function in `src/broll_agent/generate_lightning.py`:

```python
def load_pipeline(cfg: Settings):
    """Load Z-Image Turbo once per run via mmgp with pre-quantized int8 weights."""
    from broll_agent.zimage.model_loader import load_zimage_pipeline

    logger.info("loading Z-Image Turbo via mmgp (model_dir=%s)", cfg.zimage_model_dir)
    return load_zimage_pipeline(cfg)
```

The `build_prompt()` and `generate_images_for_opportunity()` functions stay unchanged — they call `pipe(...)` which works identically with the new pipeline.

- [ ] **Step 2: Update module docstring**

Change the module docstring from:
```python
"""BrollOpportunity -> generated_NN.png via local Z-Image Turbo.

Z-Image Turbo is a Flux-derived distilled model that runs in 4 steps
with guidance_scale=0.0. The transformer is quantized to int8 at load
time via torchao for ~2-3x faster inference without extra disk overhead.
"""
```
to:
```python
"""BrollOpportunity -> generated_NN.png via local Z-Image Turbo.

Z-Image Turbo is a Flux-derived distilled model that runs in 4 steps
with guidance_scale=0.0. Weights are pre-quantized to int8 via mmgp
for fast inference with reduced disk and VRAM usage.
"""
```

- [ ] **Step 3: Run ruff check**

Run: `uv run ruff check .`
Expected: passes.

- [ ] **Step 4: Run existing tests**

Run: `uv run pytest tests/test_generate.py -v`
Expected: all tests pass (they use `FakePipeline` mock, so the loader change doesn't affect them).

- [ ] **Step 5: Commit**

```bash
git add src/broll_agent/generate_lightning.py
git commit -m "feat: rewrite generate_lightning.py to use mmgp model loader"
```

---

### Task 10: Update .env.example

**Files:**
- Modify: `.env.example`

**Interfaces:**
- Consumes: nothing
- Produces: updated env reference for users

- [ ] **Step 1: Update the Z-Image Turbo section**

Replace the Z-Image Turbo section in `.env.example`:

```env
# ============================================================
# Z-Image Turbo (local generation — mmgp int8 pipeline)
# ============================================================

# Path to local model directory containing quantized transformer + text encoder + VAE
ZIMAGE_MODEL_DIR=./z-image model
# cuda | cpu
ZIMAGE_DEVICE=cuda
ZIMAGE_IMAGES_PER_PROMPT=3
# 4 steps is optimal for Z-Image Turbo (distilled, no CFG)
ZIMAGE_NUM_INFERENCE_STEPS=4
# must stay 0.0 — Z-Image Turbo is distilled without CFG
ZIMAGE_GUIDANCE_SCALE=0.0
# blank = random seed per image
ZIMAGE_SEED=

# Triton int8 kernel acceleration (falls back gracefully if unavailable)
ZIMAGE_QUANTO_INT8_KERNEL=true
# mmgp VRAM allocation profile index
ZIMAGE_MMGP_PROFILE=0
# max GPU memory reservation percentage (0.0-1.0)
ZIMAGE_PERC_RESERVED_MEM_MAX=0.9
# VRAM safety margin
ZIMAGE_VRAM_SAFETY_COEFFICIENT=0.1
# auto | sdpa | flash | sage2
ZIMAGE_ATTENTION_BACKEND=auto
```

- [ ] **Step 2: Commit**

```bash
git add .env.example
git commit -m "docs: update .env.example with mmgp settings"
```

---

### Task 11: Add Model Loader Tests

**Files:**
- Create: `tests/test_model_loader.py`

**Interfaces:**
- Consumes: `zimage.model_loader.conv_state_dict`, `zimage.model_loader._FUSED_SPLIT_MAP`
- Produces: unit tests for key mapping and loader structure

- [ ] **Step 1: Write tests for conv_state_dict and loader structure**

```python
from __future__ import annotations

import torch

from broll_agent.zimage.model_loader import conv_state_dict, _FUSED_SPLIT_MAP


def test_conv_state_dict_maps_final_layer():
    sd = {"final_layer.linear.weight": torch.zeros(1)}
    result = conv_state_dict(sd)
    assert "all_final_layer.2-1.linear.weight" in result


def test_conv_state_dict_maps_x_embedder():
    sd = {"x_embedder.weight": torch.zeros(1)}
    result = conv_state_dict(sd)
    assert "all_x_embedder.2-1.weight" in result


def test_conv_state_dict_maps_attention_keys():
    sd = {
        "layers.0.attention.out.bias": torch.zeros(1),
        "layers.0.attention.k_norm.weight": torch.zeros(1),
        "layers.0.attention.q_norm.weight": torch.zeros(1),
        "layers.0.attention.out.weight": torch.zeros(1),
    }
    result = conv_state_dict(sd)
    assert "layers.0.attention.to_out.0.bias" in result
    assert "layers.0.attention.norm_k.weight" in result
    assert "layers.0.attention.norm_q.weight" in result
    assert "layers.0.attention.to_out.0.weight" in result


def test_conv_state_dict_strips_model_prefix():
    sd = {"model.diffusion_model.x_embedder.weight": torch.zeros(1)}
    result = conv_state_dict(sd)
    assert "all_x_embedder.2-1.weight" in result


def test_conv_state_dict_passthrough_when_no_matching_keys():
    sd = {"some.other.key": torch.zeros(1)}
    result = conv_state_dict(sd)
    assert result == sd


def test_fused_split_map_has_qkv():
    assert "attention.to_qkv" in _FUSED_SPLIT_MAP
    assert _FUSED_SPLIT_MAP["attention.to_qkv"]["mapped_modules"] == (
        "attention.to_q", "attention.to_k", "attention.to_v"
    )
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_model_loader.py -v`
Expected: all tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_model_loader.py
git commit -m "test: add unit tests for conv_state_dict key mapping"
```

---

### Task 12: Run Full Test Suite and Lint

**Files:**
- All source files

**Interfaces:**
- Consumes: everything
- Produces: verified working state

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest -v`
Expected: all tests pass, including `test_generate.py` (FakePipeline mock) and `test_model_loader.py`.

- [ ] **Step 2: Run ruff check**

Run: `uv run ruff check .`
Expected: no errors.

- [ ] **Step 3: Run ruff format check**

Run: `uv run ruff format --check .`
Expected: no formatting issues. If there are, run `uv run ruff format .` to fix.

- [ ] **Step 4: Verify no torchao imports remain**

Run: `uv run python -c "import broll_agent.generate_lightning; print('ok')"`
Expected: prints "ok" without importing torchao.

- [ ] **Step 5: Final commit if any fixes needed**

```bash
git add -A
git commit -m "chore: fix lint/format issues after Wan2GP migration"
```

---

### Task 13: Download Pre-Quantized Model Files

**Files:**
- Modify: `z-image model/` directory contents

**Interfaces:**
- Consumes: HuggingFace or Wan2GP checkpoint sources
- Produces: pre-quantized model files in `z-image model/`

- [ ] **Step 1: Identify required model files**

The loader expects these files in `z-image model/`:
- `transformer/config.json` (already exists)
- `transformer/ZImageTurbo_quanto_bf16_int8.safetensors` (6.4 GB — needs download)
- `text_encoder/qwen3_quanto_bf16_int8.safetensors` (~3.5 GB — needs download)
- `tokenizer/` (already exists)
- `vae/` config and weights (already exists, but may need Wan2GP's VAE format)
- `scheduler/scheduler_config.json` (already exists)

Note: The exact filenames and locations depend on how Wan2GP distributes its checkpoints. Check Wan2GP's `z_image_handler.py` for download URLs and file resolution logic.

- [ ] **Step 2: Download or locate the pre-quantized transformer weights**

The 6.4 GB `ZImageTurbo_quanto_bf16_int8.safetensors` file needs to be placed in `z-image model/transformer/`. Check Wan2GP's model download mechanism or HuggingFace for the source.

- [ ] **Step 3: Download or locate the pre-quantized text encoder weights**

The ~3.5 GB `qwen3_quanto_bf16_int8.safetensors` file needs to be placed in `z-image model/text_encoder/`.

- [ ] **Step 4: Verify VAE compatibility**

Check if the existing VAE weights in `z-image model/vae/` are compatible with mmgp's `fast_load_transformers_model()`. If not, download Wan2GP's VAE format (`ZImageTurbo_VAE_bf16.safetensors` + config).

- [ ] **Step 5: Commit any config changes**

```bash
git add "z-image model/"
git commit -m "chore: add pre-quantized model files for mmgp pipeline"
```

Note: Large model files should likely be gitignored and documented in README instead.

---

### Task 14: Update AGENTS.md and README.md

**Files:**
- Modify: `AGENTS.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: nothing
- Produces: updated documentation reflecting the new pipeline

- [ ] **Step 1: Update AGENTS.md tech stack and generation section**

Update the tech stack table in AGENTS.md:
- Change "Image generation" row from `torchao` to `mmgp + optimum-quanto + triton`
- Update the notes to mention pre-quantized int8 weights

Update §7 Step 3a to reflect the new loading flow (mmgp instead of torchao).

Update `.env.example` section to reflect new variables.

- [ ] **Step 2: Update README.md**

Update the README to mention:
- The new mmgp-based pipeline
- Required model files (pre-quantized transformer + text encoder)
- Disk usage (~10 GB vs previous ~30 GB)
- Triton kernel acceleration (optional, Linux-only)

- [ ] **Step 3: Commit**

```bash
git add AGENTS.md README.md
git commit -m "docs: update documentation for mmgp pipeline migration"
```

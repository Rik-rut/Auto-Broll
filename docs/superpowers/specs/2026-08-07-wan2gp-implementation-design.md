# Wan2GP Implementation Design

**Date**: 2026-08-07  
**Branch**: `wan2gp-implementation`  
**Status**: Approved for implementation

## Overview

Replace the current torchao-based int8 quantization approach with Wan2GP's native int8 inference pipeline using the `mmgp` library. This achieves:
- **Disk savings**: 30.6 GB → ~14 GB (pre-quantized weights vs bf16 + runtime quantization)
- **Speed improvement**: ~2 min/image → ~1 min/image (custom Triton kernels + optimized transformer)
- **Better VRAM management**: Budget-based allocation vs module-by-module offload

## Approach

**Approach A: Direct port of Wan2GP's model files**

Port 4 core files from Wan2GP's `models/z_image/` into Auto-Broll's `src/broll_agent/zimage/` package, plus shared infrastructure (attention, Triton kernels). Strip out Gradio UI references, ControlNet/LoRA/NAG/Unified Sampler code paths, and control image support. Keep full transformer optimizations (in-place ops, chunked FFN, list-based tensor passing) and Triton kernel injection.

**Decision rationale**: User chose "full Wan2GP pipeline" and "replace torchao completely" — this approach gets there most directly with maximum fidelity to Wan2GP's proven code.

## File Structure

```
src/broll_agent/
├── zimage/
│   ├── __init__.py
│   ├── pipeline.py              # from Wan2GP's pipeline_z_image.py
│   ├── transformer.py           # from Wan2GP's z_image_transformer2d.py
│   ├── model_loader.py          # from Wan2GP's z_image_main.py (model_factory)
│   ├── vae.py                   # from Wan2GP's autoencoder_kl.py
│   └── configs/
│       └── z_image_turbo.json   # scheduler + pipeline config
├── shared/
│   ├── attention.py             # from Wan2GP's shared/attention.py
│   └── kernels/
│       ├── quanto_int8_inject.py
│       └── quanto_int8_triton.py
├── generate_lightning.py        # REWRITTEN — uses new zimage/ loader
├── config.py                    # UPDATED — new mmgp settings
└── ...
```

**What we strip from Wan2GP ports:**
- All Gradio UI references
- `z_image_handler.py` (file resolution, download URLs)
- ControlNet / LoRA / NAG / Unified Sampler code paths
- Control image support (v1/v2 control latents)
- VAE upsampler support

**What stays:**
- Full transformer with in-place ops, chunked FFN, list-based tensor passing
- Custom Triton kernel injection for int8 matmul acceleration
- Multi-backend attention (SageAttention2 > Flash > SDPA auto-detection)
- Complete `ZImagePipeline` denoising loop (flow matching with mu shift)

## Model Loading Flow

**Current flow (torchao):**
```python
ZImagePipeline.from_pretrained(model_dir)  # loads bf16 weights
torchao.quantize_(transformer, Int8WeightOnlyConfig)  # quantizes at load time
pipe.to(device) or pipe.enable_sequential_cpu_offload()
```

**New flow (mmgp):**
```python
1. Create ZImageTransformer2DModel under accelerate.init_empty_weights()
2. offload.load_model_data(transformer, "ZImageTurbo_quanto_bf16_int8.safetensors")
   # loads pre-quantized int8 weights directly
3. Load text encoder via offload.fast_load_transformers_model(Qwen3ForCausalLM)
4. Load VAE via offload.fast_load_transformers_model(AutoencoderKL)
5. Load tokenizer via AutoTokenizer.from_pretrained()
6. Create scheduler from JSON config
7. Assemble ZImagePipeline(scheduler, vae, text_encoder, tokenizer, transformer)
8. offload.profile(pipe, profile_no=...)  # budget-based VRAM management
9. Inject Triton kernels (if ZIMAGE_QUANTO_INT8_KERNEL=true)
```

**Key differences:**
- Weights are pre-quantized (6.4 GB vs 23 GB on disk)
- No dequantization to bf16 during inference — stays int8 throughout
- VRAM managed by mmgp's budget system, not diffusers' module-by-module offload
- Text encoder and VAE loaded via mmgp's `fast_load_transformers_model` (optimized for speed)

**Model files location:**
All model files stay in `z-image model/` directory. The loader looks for:
- `ZImageTurbo_quanto_bf16_int8.safetensors` (6.4 GB) — pre-quantized transformer
- `qwen3_quanto_bf16_int8.safetensors` (~3.5 GB) — pre-quantized text encoder
- Existing VAE, tokenizer, scheduler configs (already present)

**Text encoder quantization:**
Quantizing the text encoder provides moderate speed gains (10-15% per image) and significant VRAM savings (~3.5 GB), which reduces offloading pressure on the transformer. The text encoder is less sensitive to quantization than the transformer, and Wan2GP's int8 format is proven stable.

## Configuration Changes

**New `.env` variables:**

| Variable | Type | Default | Purpose |
|---|---|---|---|
| `ZIMAGE_QUANTO_INT8_KERNEL` | `bool` | `true` | Enable custom Triton kernels for int8 matmul |
| `ZIMAGE_MMGP_PROFILE` | `int` | `0` | mmgp VRAM allocation profile index |
| `ZIMAGE_PERC_RESERVED_MEM_MAX` | `float` | `0.9` | Max GPU memory reservation % |
| `ZIMAGE_VRAM_SAFETY_COEFFICIENT` | `float` | `0.1` | VRAM safety margin |
| `ZIMAGE_ATTENTION_BACKEND` | `str` | `auto` | `auto` / `sdpa` / `flash` / `sage2` |

**Removed variables:**
- `zimage_dtype` — no longer needed, mmgp handles dtype internally
- `zimage_enable_cpu_offload` — replaced by mmgp's profile-based offload

**Changed behavior:**
- `zimage_model_dir` still points to `./z-image model` — model files stay there
- The loader looks for `ZImageTurbo_quanto_bf16_int8.safetensors` in that directory instead of the sharded bf16 files

## Dependencies

**Add:**
- `mmgp==3.7.10`
- `optimum-quanto` (for int8 weight format compatibility)
- `triton` (for custom kernels — Windows support TBD, fallback if unavailable)
- `sageattention` (optional, for SageAttention2 backend)

**Remove:**
- `torchao` — fully replaced

## generate_lightning.py Rewrite

The public interface stays the same:
- `load_pipeline(cfg)` — returns a pipeline object
- `generate_images_for_opportunity(pipe, opportunity, cfg, dest_dir)` — generates images

Internally, `load_pipeline()` calls the new `zimage/model_loader.py` instead of `ZImagePipeline.from_pretrained()`. The `build_prompt()` and `generate_images_for_opportunity()` functions stay structurally identical — they just call the new pipeline's `__call__` method.

**Impact on other files:**
- `cli.py` — no changes needed (uses the same public interface)
- `test_generate.py` — no changes needed (uses `FakePipeline` mock)

## Testing Strategy

- Port existing `test_generate.py` tests — they use a `FakePipeline` so they'll work with the new pipeline interface unchanged
- Add a test for the model loader that verifies key mapping (`conv_state_dict`) with mocked weights
- Add a test for attention backend auto-detection
- Triton kernel injection tested via integration (skip if Triton not available on CI)

## Windows / Triton Risk Mitigation

- Triton kernel injection is gated behind `ZIMAGE_QUANTO_INT8_KERNEL` env var — defaults to `true` but gracefully falls back to quanto's default int8 matmul if Triton fails to compile or run
- Log a clear warning if Triton kernels fail to initialize, continue with standard path
- `WAN2GP_QUANTO_INT8_ALLOW_RUNTIME_FALLBACK=1` is always set — if a kernel config fails at runtime, it tries alternatives before falling back

## Risks and Open Questions

1. **Triton on Windows**: Custom Triton kernels may not compile on Windows. Mitigation: graceful fallback to standard quanto int8 matmul.
2. **mmgp documentation**: mmgp is poorly documented as a standalone library. Mitigation: we're porting the code directly from Wan2GP where it's known to work.
3. **Key mapping**: The `conv_state_dict()` function maps between Flux-like and Z-Image naming conventions. Needs verification with actual weights.
4. **mmgp hidden dependencies**: mmgp might have hidden dependencies on Wan2GP's directory structure or config system. Mitigation: test early, isolate issues quickly.

## Success Criteria

- [ ] `uv sync` installs mmgp, optimum-quanto, triton successfully
- [ ] Model loader loads pre-quantized int8 weights from `z-image model/`
- [ ] Pipeline generates images at ~1 min/image (vs current ~2 min)
- [ ] Disk usage is ~10 GB (vs current ~30 GB)
- [ ] All existing tests pass
- [ ] Graceful fallback if Triton kernels fail to initialize
- [ ] No changes needed to cli.py or test_generate.py

## Out of Scope (v1)

- ControlNet support
- LoRA support
- NAG (Normalized Attention Guidance)
- Unified Sampler (UCGM-S)
- Multiple quantization formats (fp8, nvfp4, gguf, nunchaku)
- VAE upsampler support

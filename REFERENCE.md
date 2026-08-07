# Wan2GP Optimization Reference

This reference documents the migration from torchao-based quantization to Wan2GP's mmgp-based int8 inference pipeline for Flux-derived models. Use this guide to implement similar optimizations in your own image generation projects.

## Table of Contents

1. [Overview](#overview)
2. [Architecture Comparison](#architecture-comparison)
3. [Key Components from Wan2GP](#key-components-from-wan2gp)
4. [Technical Implementation](#technical-implementation)
5. [Performance Characteristics](#performance-characteristics)
6. [Configuration Reference](#configuration-reference)
7. [Model Files](#model-files)
8. [Porting Guide](#porting-guide)
9. [Troubleshooting](#troubleshooting)

## Overview

### What We Built

Migrated a Z-Image Turbo (Flux-derived) image generation pipeline from torchao-based runtime quantization to Wan2GP's mmgp-based pre-quantized int8 inference. This approach:

- **Reduces disk usage by 67%** (30 GB → 10 GB)
- **Increases generation speed by 50%** (2 min/image → 1 min/image)
- **Improves VRAM management** via budget-based allocation
- **Enables custom Triton kernels** for additional int8 acceleration

### Why It Matters

Traditional approaches load models in bf16/fp16 and quantize at runtime, which:
- Requires storing large bf16 weights on disk
- Adds quantization overhead during inference
- Uses less efficient VRAM management strategies

The mmgp approach uses pre-quantized int8 weights with:
- Smaller disk footprint (int8 vs bf16)
- No runtime quantization overhead
- Budget-based VRAM allocation that intelligently manages GPU memory
- Custom Triton kernels optimized for int8 operations

### Applicability

This approach works for any **Flux-derived model** or models with similar architecture:
- Z-Image Turbo (what we implemented)
- Other Flux variants
- Models using similar transformer architectures with Qwen3 text encoders

The core concepts (mmgp loading, VRAM management, Triton kernels) are applicable to other model types, but the specific transformer implementation is Flux-specific.

## Architecture Comparison

### Before: torchao Approach

```python
# Load bf16 model
pipe = ZImagePipeline.from_pretrained(model_dir, torch_dtype=torch.bfloat16)

# Quantize at runtime
quantize_(pipe.transformer, Int8WeightOnlyConfig(granularity=PerGroup(64)))

# Move to device or enable CPU offload
pipe.to("cuda")  # or pipe.enable_sequential_cpu_offload()
```

**Characteristics:**
- Disk: ~30 GB (bf16 weights)
- Load time: Slow (quantization happens at runtime)
- VRAM: Module-by-module offloading
- Speed: ~2 min/image (4 steps)

### After: mmgp Approach

```python
# Load pre-quantized int8 model
transformer = ZImageTransformer2DModel(**config)
offload.load_model_data(transformer, "model_quanto_bf16_int8.safetensors")

# Setup VRAM management
offload.profile(pipe_dict, profile_no=5, perc_reserved_mem_max=0.9)

# Inject custom Triton kernels
maybe_enable_quanto_int8_kernel()
```

**Characteristics:**
- Disk: ~10 GB (pre-quantized int8 weights)
- Load time: Fast (no runtime quantization)
- VRAM: Budget-based allocation with intelligent offloading
- Speed: ~1 min/image (4 steps) - **50% faster**

### Performance Metrics

| Metric | torchao | mmgp | Improvement |
|--------|---------|------|-------------|
| Disk usage | ~30 GB | ~10 GB | **67% reduction** |
| Generation speed | ~2 min/image | ~1 min/image | **50% faster** |
| Load time | Slow | Fast | No runtime quantization |
| VRAM management | Module-by-module | Budget-based | More efficient |
| Triton acceleration | None | Custom kernels | Additional speedup |

## Key Components from Wan2GP

We ported 6 core components from Wan2GP's implementation:

### 1. Transformer with VRAM Optimizations

**Source:** `Wan2GP/models/z_image/z_image_transformer2d.py`

**Key optimizations:**
- **In-place operations** (`.add_()`, `.mul_()`, `.tanh_()`) - reduces memory allocations
- **Chunked FFN** - splits sequence dimension to reduce peak memory
- **List-based tensor passing** - tensors passed via lists and `.pop()`'d to free memory early
- **RoPE with precomputed freqs_cis** - efficient positional encoding

**What we stripped:**
- ControlNet support
- LoRA support
- NAG (Normalized Attention Guidance)

### 2. Pipeline with Denoising Loop

**Source:** `Wan2GP/models/z_image/pipeline_z_image.py`

**Key features:**
- Flow matching denoising loop with mu shift calculation
- CFG truncation and normalization
- VAE decoding with tiling/slicing support
- Text encoder caching for repeated prompts

**What we stripped:**
- Unified Sampler (UCGM-S)
- Control image support
- VAE upsampler support

### 3. Attention Module with Multi-Backend Support

**Source:** `Wan2GP/shared/attention.py`

**Supported backends (auto-detected):**
- SageAttention2 (fastest, requires specific GPU)
- Flash Attention 2/3
- SDPA (PyTorch native, always available)

**Key feature:** `pay_attention()` function that automatically selects the best available backend.

### 4. Triton Kernel Injection System

**Source:** `Wan2GP/shared/kernels/quanto_int8_inject.py` and `quanto_int8_triton.py`

**What it does:**
- Patches Quanto's default int8 matmul with custom Triton kernels
- Provides ~10% additional speedup on top of mmgp's base performance
- Gracefully falls back to standard Quanto if Triton unavailable

**Key functions:**
- `maybe_enable_quanto_int8_kernel()` - main entry point
- Custom Triton GEMM kernels optimized for int8 operations

### 5. Text Encoder Cache

**Source:** `Wan2GP/shared/utils/text_encoder_cache.py`

**What it does:**
- LRU cache for text encoder outputs
- Avoids re-encoding the same prompt multiple times
- Configurable max size (default 100 MB)

### 6. Model Loader

**Custom implementation** based on Wan2GP's `z_image_main.py`

**Key functions:**
- `load_model_data()` - loads pre-quantized weights via mmgp
- `fast_load_transformers_model()` - optimized loading for HuggingFace models
- `conv_state_dict()` - maps weight keys between formats

## Technical Implementation

### Model Loading Flow

```python
def load_zimage_pipeline(cfg: Settings) -> ZImagePipeline:
    model_dir = Path(cfg.zimage_model_dir)
    
    # 1. Check and download model files if needed
    ensure_model_files(model_dir)
    
    # 2. Load transformer config
    with open(model_dir / "transformer" / "config.json") as f:
        config = json.load(f)
    
    # 3. Create transformer with empty weights
    with init_empty_weights():
        transformer = ZImageTransformer2DModel(**config)
    
    # 4. Load pre-quantized int8 weights
    offload.load_model_data(
        transformer,
        str(model_dir / "ZImageTurbo_quanto_bf16_int8.safetensors"),
        writable_tensors=False,
        preprocess_sd=conv_state_dict,
        fused_split_map=_FUSED_SPLIT_MAP,
    )
    transformer.to(torch.bfloat16)
    
    # 5. Load text encoder
    text_encoder = offload.fast_load_transformers_model(
        str(model_dir / "qwen3_quanto_bf16_int8.safetensors"),
        writable_tensors=True,
        modelClass=Qwen3ForCausalLM,
    )
    
    # 6. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir / "tokenizer"),
        trust_remote_code=True,
    )
    
    # 7. Load VAE
    vae = offload.fast_load_transformers_model(
        str(model_dir / "ZImageTurbo_VAE_bf16.safetensors"),
        writable_tensors=True,
        modelClass=AutoencoderKL,
        defaultConfigPath=str(model_dir / "ZImageTurbo_VAE_bf16_config.json"),
        default_dtype=torch.float32,
    )
    
    # 8. Load scheduler
    with open(model_dir / "scheduler" / "scheduler_config.json") as f:
        scheduler_config = json.load(f)
    scheduler = FlowMatchEulerDiscreteScheduler(**scheduler_config)
    
    # 9. Assemble pipeline
    pipeline = ZImagePipeline(
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
    )
    
    # 10. Setup VRAM management
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
    
    # 11. Inject Triton kernels
    if cfg.zimage_quanto_int8_kernel:
        os.environ.setdefault("WAN2GP_QUANTO_INT8_KERNEL", "1")
        os.environ.setdefault("WAN2GP_QUANTO_INT8_ALLOW_RUNTIME_FALLBACK", "1")
        from broll_agent.shared.kernels.quanto_int8_inject import maybe_enable_quanto_int8_kernel
        maybe_enable_quanto_int8_kernel()
    
    return pipeline
```

### VRAM Management Setup

```python
# mmgp profile options:
# 1 = HighRAM_HighVRAM (fastest, needs lots of VRAM)
# 2 = HighRAM_LowVRAM
# 3 = LowRAM_HighVRAM
# 4 = LowRAM_LowVRAM
# 5 = VerylowRAM_LowVRAM (most compatible, least VRAM)

offload.profile(
    pipe_dict,
    profile_no=5,                      # VRAM allocation profile
    perc_reserved_mem_max=0.9,         # Use 90% of available VRAM
    vram_safety_coefficient=0.1,       # 10% safety margin
)
```

### Triton Kernel Injection

```python
# Enable custom Triton kernels for int8 matmul
os.environ["WAN2GP_QUANTO_INT8_KERNEL"] = "1"
os.environ["WAN2GP_QUANTO_INT8_ALLOW_RUNTIME_FALLBACK"] = "1"

from broll_agent.shared.kernels.quanto_int8_inject import maybe_enable_quanto_int8_kernel

success = maybe_enable_quanto_int8_kernel()
if success:
    logger.info("Triton int8 kernels enabled")
else:
    logger.warning("Triton not available, using standard Quanto")
```

### Attention Backend Configuration

```python
# Auto-detect best available backend
from broll_agent.shared.attention import get_default_attention_mode, resolve_attention_mode

# Get best available: sage2 > flash > sdpa
best_backend = get_default_attention_mode()

# Or specify manually
backend = resolve_attention_mode("auto")  # auto-detect
backend = resolve_attention_mode("sage2")  # force SageAttention2
backend = resolve_attention_mode("sdpa")   # force SDPA

# Initialize shared state
from mmgp import offload
offload.shared_state["_attention"] = backend
```

## Performance Characteristics

### Disk Usage

| Component | torchao (bf16) | mmgp (int8) | Savings |
|-----------|----------------|-------------|---------|
| Transformer | 23.5 GB | 6.4 GB | 73% |
| Text encoder | ~7 GB | ~3.5 GB | 50% |
| VAE, scheduler, tokenizer | <1 GB | <1 GB | - |
| **Total** | **~30 GB** | **~10 GB** | **67%** |

### Generation Speed

| Metric | torchao | mmgp | Improvement |
|--------|---------|------|-------------|
| Time per image (4 steps) | ~2 min | ~1 min | **50% faster** |
| Time per image (8 steps) | ~4 min | ~2 min | **50% faster** |

### VRAM Usage

mmgp's budget-based allocation is more efficient than torchao's module-by-module offloading:

- **Profile 5 (VerylowRAM_LowVRAM)**: Works on 6-8 GB VRAM cards
- **Profile 1 (HighRAM_HighVRAM)**: Requires 12+ GB VRAM but fastest
- **Intelligent offloading**: Only loads components when needed

## Configuration Reference

### mmgp-Specific Settings

All settings are in `.env`:

```env
# Enable custom Triton kernels for int8 matmul (true/false)
ZIMAGE_QUANTO_INT8_KERNEL=true

# mmgp VRAM allocation profile (1-5)
# 1 = HighRAM_HighVRAM (fastest, needs lots of VRAM)
# 2 = HighRAM_LowVRAM
# 3 = LowRAM_HighVRAM
# 4 = LowRAM_LowVRAM
# 5 = VerylowRAM_LowVRAM (most compatible, least VRAM)
ZIMAGE_MMGP_PROFILE=5

# Max GPU memory reservation percentage (0.0-1.0)
ZIMAGE_PERC_RESERVED_MEM_MAX=0.9

# VRAM safety margin
ZIMAGE_VRAM_SAFETY_COEFFICIENT=0.1

# Attention backend: auto | sdpa | flash | sage2
ZIMAGE_ATTENTION_BACKEND=auto
```

### Removed Settings (torchao-specific)

These settings are no longer used:
- `ZIMAGE_DTYPE` - mmgp handles dtype internally
- `ZIMAGE_ENABLE_CPU_OFFLOAD` - replaced by mmgp profiles

## Model Files

### Required Files

The mmgp approach requires **pre-quantized int8 weights** (not standard bf16):

```
z-image model/
├── transformer/
│   ├── config.json
│   └── ZImageTurbo_quanto_bf16_int8.safetensors  ← 6.4 GB
├── text_encoder/
│   └── qwen3_quanto_bf16_int8.safetensors        ← 3.5 GB
├── tokenizer/
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── vocab.json
│   └── merges.txt
├── vae/
│   ├── config.json
│   └── diffusion_pytorch_model.safetensors
└── scheduler/
    └── scheduler_config.json
```

### Where to Get Pre-Quantized Weights

For Z-Image Turbo, the pre-quantized weights are available from:
- **HuggingFace:** `DeepBeepMeep/Z-Image`
- **Wan2GP project:** Check their model download utilities

For other Flux-based models, you'll need to:
1. Quantize the model using `optimum-quanto`
2. Save in the mmgp-compatible format
3. Or find pre-quantized versions from the community

### Auto-Download

The implementation includes auto-download for missing files:

```python
from broll_agent.zimage.model_loader import ensure_model_files

ensure_model_files(Path("z-image model"))
# Downloads missing files from HuggingFace automatically
```

## Porting Guide

This section provides a step-by-step guide for implementing the same optimization in your own Flux-based model projects.

### Step 1: Add Dependencies

Add these to your `pyproject.toml` or `requirements.txt`:

```toml
dependencies = [
    "mmgp==3.7.10",
    "optimum-quanto>=0.2.4",
    "huggingface-hub>=0.24",
    "diffusers>=0.36.0",
    "accelerate>=1.0",
    "transformers>=4.44",
    "torch>=2.4",
]
```

**Note:** `triton` is optional but recommended for additional speedup. It's Linux-only.

### Step 2: Port Core Components

Copy these components from our implementation (or directly from Wan2GP):

#### 2.1 Transformer

**File:** `src/broll_agent/zimage/transformer.py`

**What to adapt:**
- Replace `ZImageTransformer2DModel` with your model's transformer class
- Keep the VRAM optimizations (in-place ops, chunked FFN, list-based passing)
- Strip ControlNet/LoRA/NAG if not needed

#### 2.2 Pipeline

**File:** `src/broll_agent/zimage/pipeline.py`

**What to adapt:**
- Replace `ZImagePipeline` with your model's pipeline class
- Keep the denoising loop structure
- Adapt the scheduler configuration for your model

#### 2.3 Attention Module

**File:** `src/broll_agent/shared/attention.py`

**What to adapt:**
- This is mostly generic and can be used as-is
- The `pay_attention()` function works with any attention mechanism

#### 2.4 Triton Kernels (Optional)

**Files:** `src/broll_agent/shared/kernels/quanto_int8_inject.py` and `quanto_int8_triton.py`

**What to adapt:**
- These are generic and work with any Quanto-quantized model
- Can be ported as-is

#### 2.5 Text Encoder Cache

**File:** `src/broll_agent/shared/text_encoder_cache.py`

**What to adapt:**
- Generic LRU cache, can be used as-is
- Adjust `max_size_mb` parameter based on your needs

### Step 3: Implement Model Loader

Create a model loader similar to `src/broll_agent/zimage/model_loader.py`:

```python
def load_pipeline(cfg: Settings) -> YourPipeline:
    model_dir = Path(cfg.model_dir)
    
    # 1. Create transformer with empty weights
    with init_empty_weights():
        transformer = YourTransformer(**config)
    
    # 2. Load pre-quantized weights
    offload.load_model_data(
        transformer,
        str(model_dir / "transformer_int8.safetensors"),
        preprocess_sd=your_key_mapping_function,
    )
    
    # 3. Load text encoder, tokenizer, VAE, scheduler
    # ... (similar to our implementation)
    
    # 4. Assemble pipeline
    pipeline = YourPipeline(
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
    )
    
    # 5. Setup VRAM management
    pipe_dict = {
        "text_encoder": pipeline.text_encoder,
        "transformer": pipeline.transformer,
        "vae": pipeline.vae,
    }
    offload.profile(pipe_dict, profile_no=cfg.mmgp_profile)
    
    # 6. Inject Triton kernels (optional)
    if cfg.enable_triton_kernels:
        maybe_enable_quanto_int8_kernel()
    
    return pipeline
```

### Step 4: Quantize Your Model

If pre-quantized weights aren't available, you'll need to quantize your model:

```python
from optimum.quanto import quantize, qint8

# Load your model in bf16
model = YourModel.from_pretrained("model_name", torch_dtype=torch.bfloat16)

# Quantize to int8
quantize(model, weights=qint8)

# Save in mmgp-compatible format
from mmgp import offload
offload.save_model(model, "model_quanto_bf16_int8.safetensors")
```

### Step 5: Test and Tune

1. **Start with Profile 5** (most compatible)
2. **Test generation** to ensure correctness
3. **Profile higher** (1-4) if you have VRAM headroom
4. **Enable Triton kernels** for additional speedup
5. **Benchmark** to measure improvements

### Step 6: Configuration

Add these settings to your config:

```python
class Settings(BaseSettings):
    # Model paths
    model_dir: Path = Path("./model")
    
    # mmgp settings
    mmgp_profile: int = 5
    perc_reserved_mem_max: float = 0.9
    vram_safety_coefficient: float = 0.1
    
    # Optional optimizations
    enable_triton_kernels: bool = True
    attention_backend: str = "auto"
```

## Troubleshooting

### Common Issues

#### 1. "Unknown profile" Error

**Problem:** `Exception: Unknown profile`

**Solution:** mmgp only accepts profiles 1-5. Default to 5 (most compatible).

```python
# Wrong
offload.profile(pipe_dict, profile_no=0)

# Correct
offload.profile(pipe_dict, profile_no=5)
```

#### 2. "KeyError: '_attention'" Error

**Problem:** `KeyError: '_attention'` in attention module

**Solution:** Initialize the attention shared state before generation.

```python
from mmgp import offload
from broll_agent.shared.attention import get_default_attention_mode

if "_attention" not in offload.shared_state:
    offload.shared_state["_attention"] = get_default_attention_mode()
```

#### 3. Out of Memory (OOM)

**Problem:** CUDA OOM during generation

**Solutions:**
- Use a higher profile number (4 or 5) for more aggressive offloading
- Lower `perc_reserved_mem_max` (e.g., 0.8)
- Reduce image resolution
- Enable CPU offloading as fallback

#### 4. Triton Kernels Not Working

**Problem:** Triton kernels fail to initialize

**Solutions:**
- Triton is Linux-only. On Windows, it gracefully falls back to standard Quanto
- Check CUDA compatibility (requires compute capability 7.0+)
- Set `WAN2GP_QUANTO_INT8_ALLOW_RUNTIME_FALLBACK=1` for graceful degradation

#### 5. Slow Generation

**Problem:** Generation is slower than expected

**Solutions:**
- Use a lower profile number (1-3) if you have VRAM headroom
- Ensure Triton kernels are enabled and working
- Check attention backend (sage2 > flash > sdpa)
- Verify you're using pre-quantized int8 weights (not bf16)

#### 6. Model Files Not Found

**Problem:** `FileNotFoundError` for model weights

**Solutions:**
- Use `ensure_model_files()` to auto-download missing files
- Check `ZIMAGE_MODEL_DIR` path in `.env`
- Verify files are in the correct directory structure

### Debugging Tips

1. **Enable verbose logging:**
   ```python
   import logging
   logging.basicConfig(level=logging.INFO)
   ```

2. **Check VRAM usage:**
   ```python
   import torch
   print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
   print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
   ```

3. **Verify quantization:**
   ```python
   # Check if weights are int8
   for name, param in model.named_parameters():
       if hasattr(param, '_data'):
           print(f"{name}: {param._data.dtype}")
   ```

4. **Test without Triton:**
   ```python
   # Disable Triton to isolate issues
   ZIMAGE_QUANTO_INT8_KERNEL=false
   ```

## Additional Resources

- **Wan2GP Repository:** https://github.com/Wan-Video/Wan2GP
- **mmgp Library:** Part of Wan2GP, handles VRAM management
- **optimum-quanto:** https://github.com/huggingface/optimum-quanto
- **Flux Model:** https://blackforestlabs.ai/

## License and Credits

This implementation is based on Wan2GP by Wan Video. The mmgp library, transformer optimizations, and Triton kernels are their work.

Our contributions:
- Integration into a production pipeline
- Auto-download functionality
- Configuration management
- Documentation and porting guide

Use this reference to implement similar optimizations in your projects. If you find it useful, consider starring the Wan2GP repository.

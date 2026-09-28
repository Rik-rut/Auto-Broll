# Auto-Broll

Drop a talking-head video into the `input/` folder and Auto-Broll finds moments
where a cutaway image would help. For each moment it produces actual images so
you can pick the ones you like and drop them into your edit.

## Before you start

You need:

- **Python 3.11 or 3.12** from [python.org](https://www.python.org/downloads/)
- **uv** (the package manager, runs everything for you):
  - Windows: `irm https://astral.sh/uv/install.ps1 | iex`
  - Mac/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **ffmpeg** installed and on your PATH from [ffmpeg.org](https://ffmpeg.org/download.html)
  (test it: open a terminal and type `ffmpeg -version`)
- An **API key** for an LLM provider like Anthropic, OpenAI, or DeepSeek
  (or a local [Ollama](https://ollama.com) install)
- **NVIDIA GPU** recommended for local image generation (6GB+ VRAM minimum, 8GB+ recommended)
  - Works on CPU but much slower
  - Model files auto-download on first run (~10 GB for Z-Image, ~18 GB for Qwen-Image 2.1)

## One-time setup

1. Open a terminal in this folder.
2. Install everything: `uv sync`
3. Copy the config template: `copy .env.example .env` (Windows) or `cp .env.example .env` (Mac/Linux)
4. Open `.env` and fill in:
   - `LLM_API_KEY` with your LLM provider key
   - If using Ollama, set `LLM_PROVIDER=ollama` and `LLM_BASE_URL=http://127.0.0.1:8080`
5. Everything else has sensible defaults, leave them for now.

## Choose your image engine

Two local image models are built in, and you pick one at a time with
`IMAGE_ENGINE` in `.env`:

- **`zimage`** (default) — Z-Image Turbo. Fast and light: ~10 GB of model files,
  about 1 minute per image on a 6 GB GPU. Good default.
- **`qwen_image21`** — Qwen-Image 2.1 with the Viggle turbo LoRA. Noticeably
  better at photorealistic people, products and scenes, but bigger and slower:
  ~18 GB of model files and roughly 3 minutes per image on a 6 GB GPU.

Both engines write into the same scene folders using tagged filenames
(`generated_zimage_01.png` vs `generated_qwen_image21_01.png`), so you can run
one engine, look at the results, switch `IMAGE_ENGINE`, run again, and compare
side by side without losing anything. Images you generated before this feature
existed are renamed automatically on the next run.

To try Qwen-Image 2.1: set `IMAGE_ENGINE=qwen_image21` in `.env` and run again.
To go back, set `IMAGE_ENGINE=zimage`.

**License note:** the Qwen-Image 2.1 model is released under the Qwen Research
License and is **for non-commercial use only**. Check the license before using
Qwen-generated images in paid work; Z-Image Turbo is Apache 2.0.

## Model files (for local generation)

If you want to generate images locally (`GENERATE_BROLL=true`), you need the
pre-quantized Z-Image Turbo model files. The good news: **they auto-download on first run!**

The system will automatically download these files (~10 GB total) from HuggingFace to `z-image model/`:

- `transformer/ZImageTurbo_quanto_bf16_int8.safetensors` — pre-quantized transformer (~6.4 GB)
- `text_encoder/qwen3_quanto_bf16_int8.safetensors` — pre-quantized text encoder (~3.5 GB)
- `vae/ZImageTurbo_VAE_bf16.safetensors` — VAE model
- `scheduler/scheduler_config.json` — scheduler config
- `tokenizer/` — Qwen2 tokenizer files

These files use the mmgp int8 format (not the standard HuggingFace format).
Total disk usage is ~10 GB (vs ~30 GB for the standard format).

If you choose `IMAGE_ENGINE=qwen_image21`, its files (~18 GB) auto-download to
`qwen model/` on the first Qwen run instead. You only pay for the engine you
actually use.

**First run will take time** due to downloading. Subsequent runs are fast.

## Running it

Put your videos (.mp4, .mov, .mkv) in the `input/` folder, then:

```
uv run broll-agent run
```

You can also drop a transcript file into `input/` instead of a video —
`*.json`, `*.srt`, or `*.txt` (plain narration text). It will skip the
transcription step entirely and go straight to planning. A `.json` can be a
bare list of segments (`[{"start": 12.4, "end": 15.9, "text": "..."}]`), a
wrapper with `segments` + `duration_seconds`, or a full `transcript.json` as
written by this tool.

It transcribes your video, finds broll moments, and generates images. Expect a
few minutes for transcription plus roughly 1 minute per generated image with
Z-Image or 3 minutes with Qwen-Image 2.1 on a 6 GB GPU.

To process one video only: `uv run broll-agent run --video input/my_video.mp4`

Running again is safe. It skips anything already finished. Add `--force` to redo
everything from scratch.

## Where to find your results

Everything is in `output/<your-video-name>/`:

- **`broll_script.md`** is your starting point. Read it like a script. It shows
  your full narration with each broll moment marked inline, telling you where
  each image goes and why.
- **`broll_plan.json`** is the same data for the computer. You can edit the
  prompts inside by hand, then re-run to use your edited versions.
- **`scenes/`** has one folder per broll moment, in video order. Inside each you
  find the actual image files and a `prompt.json` describing the moment.

## Settings worth changing

All in `.env`:

- **Which model generates images:** `IMAGE_ENGINE` — `zimage` (default, fast)
  or `qwen_image21` (better quality, slower, non-commercial license).
- **How many images per moment:** `ZIMAGE_IMAGES_PER_PROMPT` for Z-Image,
  `QWEN_IMAGES_PER_PROMPT` for Qwen; `DOWNLOAD_IMAGES_PER_BROLL` for downloads.
- **Turn generation on or off:** `GENERATE_BROLL=true/false` and
  `DOWNLOAD_BROLL=true/false`. Both can be on at once.
- **Image size:** `IMAGE_WIDTH` and `IMAGE_HEIGHT`. For vertical video use
  720x1280. For horizontal use 1920x1080. Each engine automatically rounds the
  size down to what its model needs (multiples of 16 for Z-Image, 32 for Qwen),
  so 1080x1920 becomes 1072x1920 or 1056x1920 depending on the engine.
- **The visual style:** each image engine has its own style setting, both added
  to every generated image. `STYLE_PROMPT` applies to Z-Image — change it to
  match your channel's look. `QWEN_STYLE_PROMPT` is for Qwen-Image 2.1 and is
  **blank on purpose**: that model is a 6-step "turbo" model that produces
  noticeably softer, flatter images when you add a style phrase to the
  description, so it gets the description on its own. If you want to try a
  style with Qwen, put it in `QWEN_STYLE_PROMPT` and compare the results.
- **How each b-roll image is shot:** `VARIATIONS_FILE` points at
  `config/image_variations.json` — edit that file to change how each b-roll
  image is framed (wide shot, close-up, overhead, and so on) without touching
  code. Keep these entries to framing and camera angle only: adding lighting or
  style words here tends to flatten the image.
- **VRAM management:** `ZIMAGE_MMGP_PROFILE` controls memory usage (1-5, default 5).
  Lower numbers use more VRAM but are faster. If you run out of memory, keep it at 5.
  If you have lots of VRAM (12GB+), try profile 1-3 for faster generation.
- **Triton acceleration:** `ZIMAGE_QUANTO_INT8_KERNEL=true/false` enables custom
  kernels for ~10% speedup. Requires Linux and compatible GPU. Falls back gracefully
  if unavailable.

## If something goes wrong

- **"ffmpeg not found"** Install ffmpeg and make sure it is on your PATH.
- **Model download fails** Check your internet connection. The first run downloads
  ~10 GB for Z-Image (from `DeepBeepMeep/Z-Image`) or ~18 GB for Qwen-Image 2.1
  (from `DeepBeepMeep/Qwen_image_2` and `DeepBeepMeep/Ideogram4`).
- **Qwen-Image 2.1 generation is slow** About 3 minutes per image on a 6 GB GPU
  is expected. For speed, set `IMAGE_ENGINE=zimage`; for lower memory, keep
  `QWEN_MMGP_PROFILE=5`.
- **Generation is very slow or crashes with memory errors** Try a different VRAM profile:
  - `ZIMAGE_MMGP_PROFILE=5` (default, most compatible)
  - `ZIMAGE_MMGP_PROFILE=1` (fastest, needs 12GB+ VRAM)
  - Or set `ZIMAGE_DEVICE=cpu` to run without a GPU (much slower but works)
- **Triton kernels not working** The log should say "Triton int8 kernels enabled
  successfully". If it says "not available", run `uv sync` (Triton ships with the
  project on Windows via `triton-windows` and on Linux via torch), and make sure
  your GPU has tensor cores (NVIDIA compute capability 8.0+).
- **Download errors** Check your SearXNG address in `SEARXNG_BASE_URL` or set
  `DOWNLOAD_BROLL=false`.
- **LLM errors about keys** Check `LLM_API_KEY` in `.env`.
- **It did nothing on re-run** That means everything is already done. Add
  `--force` to redo it.
- **All generated images look the same** Each image gets different environmental
  variations (lighting, weather, time of day, seasons) for visual diversity.

## A note on downloaded images

Downloaded images come from the web and may have usage restrictions or
watermarks. Check them before using one commercially.

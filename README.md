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
  - Model files (~10 GB) auto-download on first run

## One-time setup

1. Open a terminal in this folder.
2. Install everything: `uv sync`
3. Copy the config template: `copy .env.example .env` (Windows) or `cp .env.example .env` (Mac/Linux)
4. Open `.env` and fill in:
   - `LLM_API_KEY` with your LLM provider key
   - If using Ollama, set `LLM_PROVIDER=ollama` and `LLM_BASE_URL=http://127.0.0.1:8080`
5. Everything else has sensible defaults, leave them for now.

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
few minutes for transcription plus about 1 minute per generated image.

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

- **How many images per moment:** `ZIMAGE_IMAGES_PER_PROMPT` (generated) and
  `DOWNLOAD_IMAGES_PER_BROLL` (downloaded)
- **Turn generation on or off:** `GENERATE_BROLL=true/false` and
  `DOWNLOAD_BROLL=true/false`. Both can be on at once.
- **Image size:** `IMAGE_WIDTH` and `IMAGE_HEIGHT`. For vertical video use
  720x1280. For horizontal use 1920x1080.
- **The visual style:** `STYLE_PROMPT` is added to every generated image.
  Change it to match your channel's look.
- **How each b-roll image is shot:** `VARIATIONS_FILE` points at
  `config/image_variations.json` — edit that file to change how each b-roll
  image is shot (framing + camera + lighting) without touching code.
- **VRAM management:** `ZIMAGE_MMGP_PROFILE` controls memory usage (1-5, default 5).
  Lower numbers use more VRAM but are faster. If you run out of memory, keep it at 5.
  If you have lots of VRAM (12GB+), try profile 1-3 for faster generation.
- **Triton acceleration:** `ZIMAGE_QUANTO_INT8_KERNEL=true/false` enables custom
  kernels for ~10% speedup. Requires Linux and compatible GPU. Falls back gracefully
  if unavailable.

## If something goes wrong

- **"ffmpeg not found"** Install ffmpeg and make sure it is on your PATH.
- **Model download fails** Check your internet connection. The first run downloads ~10 GB.
  You can also manually download from HuggingFace: `DeepBeepMeep/Z-Image`
- **Generation is very slow or crashes with memory errors** Try a different VRAM profile:
  - `ZIMAGE_MMGP_PROFILE=5` (default, most compatible)
  - `ZIMAGE_MMGP_PROFILE=1` (fastest, needs 12GB+ VRAM)
  - Or set `ZIMAGE_DEVICE=cpu` to run without a GPU (much slower but works)
- **Triton kernels not working** This is normal on Windows. The system falls back to
  standard quantization. For Triton support, use Linux with a compatible GPU.
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

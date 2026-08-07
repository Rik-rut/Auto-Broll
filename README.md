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
- Optional: an **NVIDIA GPU** if you want local image generation
  (it works without one, just slower)

## One-time setup

1. Open a terminal in this folder.
2. Install everything: `uv sync`
3. Copy the config template: `copy .env.example .env` (Windows) or `cp .env.example .env` (Mac/Linux)
4. Open `.env` and fill in:
   - `LLM_API_KEY` with your LLM provider key
   - If using Ollama, set `LLM_PROVIDER=ollama` and `LLM_BASE_URL=http://127.0.0.1:8080`
5. Everything else has sensible defaults, leave them for now.

## Running it

Put your videos (.mp4, .mov, .mkv) in the `input/` folder, then:

```
uv run broll-agent run
```

It transcribes your video, finds broll moments, and generates images. Expect a
few minutes for transcription plus about 2 minutes per generated image.

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
- **Slow or out of memory:** Set `ZIMAGE_ENABLE_CPU_OFFLOAD=true` if your GPU
  has less than 8GB of VRAM. Set `ZIMAGE_DEVICE=cpu` to run without a GPU at
  all (much slower but works).

## If something goes wrong

- **"ffmpeg not found"** Install ffmpeg and make sure it is on your PATH.
- **Generation is very slow or crashes with memory errors** Set
  `ZIMAGE_ENABLE_CPU_OFFLOAD=true` or `ZIMAGE_DEVICE=cpu` in `.env`.
- **Download errors** Check your SearXNG address in `SEARXNG_BASE_URL` or set
  `DOWNLOAD_BROLL=false`.
- **LLM errors about keys** Check `LLM_API_KEY` in `.env`.
- **It did nothing on re-run** That means everything is already done. Add
  `--force` to redo it.
- **All generated images look the same** This was fixed in the latest version.
  Each image now gets a different camera angle prompt for more variety.

## A note on downloaded images

Downloaded images come from the web and may have usage restrictions or
watermarks. Check them before using one commercially.

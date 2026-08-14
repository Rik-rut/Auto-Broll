"""CLI entry point: plan / execute / run."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

if TYPE_CHECKING:
    import httpx

import typer

from broll_agent.analyze import analyze_transcript
from broll_agent.config import Settings, load_settings
from broll_agent.content import load_content
from broll_agent.download_searxng import (
    DOWNLOADED_PREFIX,
    USER_AGENT,
    download_for_opportunity,
)
from broll_agent.generate_lightning import (
    GENERATED_PREFIX,
    generate_images_for_opportunity,
    load_pipeline,
    load_variations,
)
from broll_agent.llm_client import LLMClient
from broll_agent.logging_setup import setup_logging
from broll_agent.models import BrollPlan
from broll_agent.planner import PLAN_FILENAME, load_plan, scene_folder_name, write_plan_and_script
from broll_agent.transcribe import (
    TRANSCRIPT_FILENAME,
    load_or_transcribe,
    save_transcript,
)
from broll_agent.video_utils import (
    VIDEO_EXTENSIONS,
    assign_video_slugs,
    count_files_with_prefix,
    find_transcripts,
    find_videos,
)

logger = logging.getLogger(__name__)

app = typer.Typer(
    help="Watch input/ videos and produce broll image suggestions.",
    no_args_is_help=True,
)


def _select_inputs(cfg: Settings, video: Path | None) -> list[Path]:
    all_inputs = sorted(find_videos(cfg.input_dir) + find_transcripts(cfg.input_dir))
    if video is None:
        if not all_inputs:
            logger.error("no videos or transcripts found in %s", cfg.input_dir)
            raise typer.Exit(code=1)
        return all_inputs
    target = Path(video).resolve()
    matches = [path for path in all_inputs if path.resolve() == target]
    if not matches:
        logger.error("%s not found in %s", video, cfg.input_dir)
        raise typer.Exit(code=1)
    return matches


def _plan_video(video_path: Path, slug: str, cfg: Settings, llm: LLMClient, force: bool) -> None:
    out_dir = cfg.output_dir / slug
    transcript = load_or_transcribe(video_path, out_dir / TRANSCRIPT_FILENAME, cfg, force=force)
    opportunities = analyze_transcript(transcript, llm)
    plan = BrollPlan(
        video_file=video_path.name,
        video_slug=slug,
        analyzed_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        llm_model=cfg.llm_model,
        broll_opportunities=opportunities,
    )
    write_plan_and_script(plan, transcript, out_dir)
    logger.info("planned %d broll opportunities for %s", len(opportunities), slug)


def _plan_transcript(
    path: Path, slug: str, cfg: Settings, llm: LLMClient, force: bool
) -> None:
    out_dir = cfg.output_dir / slug
    transcript_path = out_dir / TRANSCRIPT_FILENAME
    transcript = load_content(path)
    save_transcript(transcript, transcript_path)
    opportunities = analyze_transcript(transcript, llm)
    plan = BrollPlan(
        video_file=path.name,
        video_slug=slug,
        analyzed_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        llm_model=cfg.llm_model,
        broll_opportunities=opportunities,
    )
    write_plan_and_script(plan, transcript, out_dir)
    logger.info("planned %d broll opportunities for %s", len(opportunities), slug)


def _run_plan(cfg: Settings, video: Path | None, force: bool) -> None:
    inputs = _select_inputs(cfg, video)
    slugs = assign_video_slugs(inputs)
    llm: LLMClient | None = None
    for path in inputs:
        slug = slugs[path]
        plan_path = cfg.output_dir / slug / PLAN_FILENAME
        if plan_path.exists() and not force:
            logger.info("plan exists for %s — skipped (use --force to redo)", slug)
            continue
        if llm is None:
            llm = LLMClient(cfg)
        try:
            if path.suffix.lower() in VIDEO_EXTENSIONS:
                _plan_video(path, slug, cfg, llm, force)
            else:
                _plan_transcript(path, slug, cfg, llm, force)
        except Exception:
            logger.exception("planning failed for %s", path.name)


def _scene_dir(cfg: Settings, plan: BrollPlan, opportunity_index: int) -> Path:
    opportunity = plan.broll_opportunities[opportunity_index]
    return cfg.output_dir / plan.video_slug / "scenes" / scene_folder_name(opportunity)


def _scene_missing(scene_dir: Path, prefix: str, expected: int, force: bool) -> bool:
    return force or count_files_with_prefix(scene_dir, prefix) < expected


def _execute_plan(
    plan: BrollPlan,
    cfg: Settings,
    pipe: object | None,
    llm: LLMClient | None,
    http: httpx.Client | None = None,
    force: bool = False,
    variations: list[str] | None = None,
) -> None:
    for opportunity in plan.broll_opportunities:
        scene_dir = cfg.output_dir / plan.video_slug / "scenes" / scene_folder_name(opportunity)
        scene_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = scene_dir / "prompt.json"
        if force or not prompt_path.exists():
            prompt_path.write_text(opportunity.model_dump_json(indent=2), encoding="utf-8")

        if cfg.generate_broll and pipe is not None:
            if _scene_missing(
                scene_dir, GENERATED_PREFIX,                 cfg.zimage_images_per_prompt, force
            ):
                try:
                    paths = generate_images_for_opportunity(
                        pipe, opportunity, cfg, scene_dir, variations or []
                    )
                    logger.info("generated %d images for %s", len(paths), scene_dir.name)
                except Exception:
                    logger.exception("generation failed for %s", scene_dir.name)
            else:
                logger.info("generated images exist for %s — skipped", scene_dir.name)

        if cfg.download_broll and llm is not None and http is not None:
            if _scene_missing(scene_dir, DOWNLOADED_PREFIX, cfg.download_images_per_broll, force):
                try:
                    paths = download_for_opportunity(opportunity, cfg, llm, scene_dir, http=http)
                    logger.info("downloaded %d images for %s", len(paths), scene_dir.name)
                except Exception:
                    logger.exception("download failed for %s", scene_dir.name)
            else:
                logger.info("downloaded images exist for %s — skipped", scene_dir.name)


def _run_execute(cfg: Settings, video: Path | None, force: bool) -> None:
    import httpx

    variations = load_variations(cfg.variations_file) if cfg.generate_broll else []
    inputs = _select_inputs(cfg, video)
    slugs = assign_video_slugs(inputs)
    plans: list[BrollPlan] = []
    for path in inputs:
        slug = slugs[path]
        plan_path = cfg.output_dir / slug / PLAN_FILENAME
        if not plan_path.exists():
            logger.warning("no plan for %s — run `broll-agent plan` first", slug)
            continue
        plans.append(load_plan(plan_path))
    if not plans:
        logger.error("no plans found — run `broll-agent plan` first")
        raise typer.Exit(code=1)

    pipe = None
    if cfg.generate_broll:
        needs_generate = any(
            _scene_missing(
                _scene_dir(cfg, plan, index),
                GENERATED_PREFIX,
                cfg.zimage_images_per_prompt,
                force,
            )
            for plan in plans
            for index in range(len(plan.broll_opportunities))
        )
        if needs_generate:
            pipe = load_pipeline(cfg)
        else:
            logger.info("all generated images already exist — skipping pipeline load")

    llm: LLMClient | None = LLMClient(cfg) if cfg.download_broll else None
    http: httpx.Client | None = (
        httpx.Client(follow_redirects=True, headers={"User-Agent": USER_AGENT})
        if cfg.download_broll
        else None
    )
    try:
        for plan in plans:
            try:
                _execute_plan(plan, cfg, pipe, llm, http, force, variations)
            except Exception:
                logger.exception("execute failed for %s", plan.video_slug)
    finally:
        if http is not None:
            http.close()


VideoOption = Annotated[
    Path | None,
    typer.Option(
        "--video", help="Process a single input file (video or transcript) from input/."
    ),
]


@app.command()
def plan(
    video: VideoOption = None,
    force: Annotated[
        bool, typer.Option("--force", help="Redo work even if outputs exist.")
    ] = False,
) -> None:
    """Transcribe + analyze videos and write broll_plan.json (no images yet)."""
    cfg = load_settings()
    setup_logging(cfg.log_level)
    _run_plan(cfg, video, force)


@app.command()
def execute(
    video: VideoOption = None,
    force: Annotated[
        bool, typer.Option("--force", help="Ignore existing images, redo everything.")
    ] = False,
) -> None:
    """Read existing plans and generate/download images per .env toggles."""
    cfg = load_settings()
    setup_logging(cfg.log_level)
    _run_execute(cfg, video, force)


@app.command("run")
def run_all(
    video: VideoOption = None,
    force: Annotated[
        bool, typer.Option("--force", help="Ignore existing outputs, redo everything.")
    ] = False,
) -> None:
    """Plan + execute in one pass (typical usage)."""
    cfg = load_settings()
    setup_logging(cfg.log_level)
    _run_plan(cfg, video, force)
    _run_execute(cfg, video, force)


if __name__ == "__main__":
    app()

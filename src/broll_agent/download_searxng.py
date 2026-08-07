"""BrollOpportunity -> downloaded_NN.jpg via SearXNG search + LLM selection."""

from __future__ import annotations

import io
import json
import logging
import re
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import urlparse

import httpx
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ValidationError

from broll_agent.analyze import extract_json_array
from broll_agent.config import Settings
from broll_agent.llm_client import LLMCompleter
from broll_agent.models import BrollOpportunity

logger = logging.getLogger(__name__)

DOWNLOADED_PREFIX = "downloaded_"
USER_AGENT = "Mozilla/5.0 (compatible; broll-agent/0.1)"

CONTENT_TYPE_EXTENSIONS = {
    "image/jpeg": "jpg",
    "image/png": "png",
    "image/webp": "webp",
}

SELECTION_SYSTEM_PROMPT = """\
You are a picture editor choosing reference images for a broll cutaway in a video.
Favor genuine photo/video-still sources over stock-preview pages with obvious watermarks,
and prefer larger reported resolutions. Reply with ONLY a JSON array of objects shaped
like {"img_src": "...", "reason": "..."} — no markdown fences, no commentary.
"""


class ImageCandidate(BaseModel):
    title: str = ""
    url: str = ""
    img_src: str
    thumbnail_src: str = ""
    width: int | None = None
    height: int | None = None


class _LLMSelection(BaseModel):
    img_src: str
    reason: str = ""


def _parse_resolution(value: object) -> tuple[int | None, int | None]:
    if not value:
        return None, None
    match = re.match(r"\s*(\d+)\s*[x×]\s*(\d+)\s*", str(value), re.IGNORECASE)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def search_images(
    query: str, cfg: Settings, client: httpx.Client | None = None
) -> list[ImageCandidate]:
    """Stage 1: pull a candidate pool from the SearXNG JSON API."""
    url = cfg.searxng_base_url.rstrip("/") + "/search"
    params = {"q": query, "format": "json", "categories": "images"}
    owns_client = client is None
    http = client or httpx.Client(follow_redirects=True)
    try:
        response = http.get(url, params=params, timeout=cfg.searxng_timeout_seconds)
        response.raise_for_status()
        payload = response.json()
    finally:
        if owns_client:
            http.close()

    candidates: list[ImageCandidate] = []
    for result in payload.get("results", [])[: cfg.download_candidate_pool_size]:
        img_src = result.get("img_src")
        if not img_src:
            continue
        width, height = _parse_resolution(result.get("resolution"))
        candidates.append(
            ImageCandidate(
                title=result.get("title") or "",
                url=result.get("url") or "",
                img_src=img_src,
                thumbnail_src=result.get("thumbnail_src") or "",
                width=width,
                height=height,
            )
        )
    logger.info("searxng returned %d usable candidates for %r", len(candidates), query)
    return candidates


def _format_candidates_for_llm(candidates: Sequence[ImageCandidate]) -> str:
    lines = []
    for index, cand in enumerate(candidates, start=1):
        resolution = f"{cand.width}x{cand.height}" if cand.width and cand.height else "unknown size"
        domain = urlparse(cand.img_src).netloc
        title = cand.title[:80]
        lines.append(f"{index}. img_src={cand.img_src} | {resolution} | domain={domain} | {title}")
    return "\n".join(lines)


def select_candidates_with_llm(
    opportunity: BrollOpportunity,
    candidates: Sequence[ImageCandidate],
    cfg: Settings,
    client: LLMCompleter,
) -> list[ImageCandidate]:
    """Stage 2: the LLM picks the best candidates; falls back to pool order."""
    by_src = {cand.img_src: cand for cand in candidates}
    user_content = (
        f"Broll moment: {opportunity.slug}\n"
        f"Narration context: {opportunity.context_text}\n"
        f"Intended look (generation prompt): {opportunity.image_prompt}\n\n"
        f"Candidates:\n{_format_candidates_for_llm(candidates)}\n\n"
        f"Pick the {cfg.download_images_per_broll} best candidates."
    )
    prompt = user_content
    for attempt in (1, 2):
        raw_response = client.complete(SELECTION_SYSTEM_PROMPT, prompt)
        try:
            items = json.loads(extract_json_array(raw_response))
            selections = [_LLMSelection.model_validate(item) for item in items]
        except (ValueError, ValidationError) as error:
            logger.warning("selection attempt %d unparsable: %s", attempt, error)
            prompt = (
                f"{user_content}\n\n"
                "Your previous reply was not valid JSON. Reply with ONLY the raw JSON array."
            )
            continue
        chosen: list[ImageCandidate] = []
        seen: set[str] = set()
        for selection in selections:
            candidate = by_src.get(selection.img_src)
            if candidate is not None and selection.img_src not in seen:
                chosen.append(candidate)
                seen.add(selection.img_src)
        if chosen:
            return chosen[: cfg.download_images_per_broll]
        logger.warning("LLM selected no valid candidates on attempt %d", attempt)
    logger.warning(
        "falling back to first %d pool candidates for %r",
        cfg.download_images_per_broll,
        opportunity.slug,
    )
    return list(candidates[: cfg.download_images_per_broll])


def _validate_image(content: bytes, content_type: str, cfg: Settings) -> str | None:
    """Return a file extension if the body is a real image clearing the min size."""
    extension = CONTENT_TYPE_EXTENSIONS.get(content_type)
    if extension is None:
        logger.info("rejected download: unsupported content-type %r", content_type)
        return None
    try:
        with Image.open(io.BytesIO(content)) as image:
            width, height = image.size
    except (UnidentifiedImageError, OSError):
        logger.info("rejected download: body is not a decodable image")
        return None
    if width < cfg.download_min_width or height < cfg.download_min_height:
        logger.info(
            "rejected download: %dx%d below minimum %dx%d",
            width,
            height,
            cfg.download_min_width,
            cfg.download_min_height,
        )
        return None
    return extension


def download_images(
    urls: Sequence[str], cfg: Settings, dest_dir: Path, client: httpx.Client | None = None
) -> list[Path]:
    """Stage 3: fetch chosen img_src values, validate, backfill from the pool."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    owns_client = client is None
    http = client or httpx.Client(follow_redirects=True, headers={"User-Agent": USER_AGENT})
    saved: list[Path] = []
    try:
        for url in urls:
            if len(saved) >= cfg.download_images_per_broll:
                break
            try:
                response = http.get(
                    url,
                    timeout=cfg.download_timeout_seconds,
                    headers={"User-Agent": USER_AGENT},
                )
                response.raise_for_status()
            except httpx.HTTPError as error:
                logger.warning("download failed for %s: %s", url, error)
                continue
            content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
            extension = _validate_image(response.content, content_type, cfg)
            if extension is None:
                continue
            out_path = dest_dir / f"{DOWNLOADED_PREFIX}{len(saved) + 1:02d}.{extension}"
            out_path.write_bytes(response.content)
            saved.append(out_path)
    finally:
        if owns_client:
            http.close()
    return saved


def download_for_opportunity(
    opportunity: BrollOpportunity,
    cfg: Settings,
    client: LLMCompleter,
    dest_dir: Path,
    http: httpx.Client | None = None,
) -> list[Path]:
    """Full two-stage flow for one opportunity: search -> LLM pick -> download."""
    candidates = search_images(opportunity.searxng_query, cfg, client=http)
    if not candidates:
        logger.warning("no SearXNG image results for %r", opportunity.searxng_query)
        return []
    chosen = select_candidates_with_llm(opportunity, candidates, cfg, client)
    chosen_srcs = [cand.img_src for cand in chosen]
    backfill = [cand.img_src for cand in candidates if cand.img_src not in set(chosen_srcs)]
    return download_images([*chosen_srcs, *backfill], cfg, dest_dir, client=http)

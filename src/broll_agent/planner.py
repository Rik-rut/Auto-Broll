"""Read/write broll_plan.json and render the human-readable broll_script.md."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from broll_agent.models import BrollOpportunity, BrollPlan, Transcript
from broll_agent.video_utils import format_timestamp

logger = logging.getLogger(__name__)

PLAN_FILENAME = "broll_plan.json"
SCRIPT_FILENAME = "broll_script.md"

_SMALL_WORDS = frozenset(
    {"a", "an", "the", "and", "or", "but", "to", "of", "in", "on", "for", "with"}
)


def _title_case(text: str) -> str:
    """Title-case that lowercases articles, prepositions, and conjunctions."""
    words = text.split()
    if not words:
        return text
    result = [words[0].capitalize()]
    for w in words[1:]:
        result.append(w.lower() if w.lower() in _SMALL_WORDS else w.capitalize())
    return " ".join(result)


def scene_folder_name(opportunity: BrollOpportunity) -> str:
    return f"{opportunity.id}_{opportunity.slug}"


def write_plan(plan: BrollPlan, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(plan.model_dump_json(indent=2), encoding="utf-8")


def load_plan(path: Path) -> BrollPlan:
    # utf-8-sig tolerates a BOM from hand-edits in Windows editors
    return BrollPlan.model_validate(json.loads(path.read_text(encoding="utf-8-sig")))


def render_script(plan: BrollPlan, transcript: Transcript) -> str:
    """Markdown script: full narration broken at each broll opportunity."""
    title = _title_case(plan.video_slug.replace("-", " "))
    lines = [f"# Broll Script — {title}"]
    if transcript.duration_seconds is not None:
        lines.append(
            f"Source: {plan.video_file} · Duration: {format_timestamp(transcript.duration_seconds)}"
        )
    else:
        lines.append(f"Source: {plan.video_file}")
    lines.append("")

    opportunities = sorted(
        plan.broll_opportunities,
        key=lambda opp: (opp.timestamp_start is None, opp.timestamp_start or 0.0),
    )
    has_timestamps = any(opp.timestamp_start is not None for opp in plan.broll_opportunities)

    if has_timestamps and transcript.segments:
        cursor = 0.0
        for opp in opportunities:
            if opp.timestamp_start is None:
                _append_callout(lines, opp)
                continue
            block = [
                seg
                for seg in transcript.segments
                if seg.start is not None
                and seg.start < opp.timestamp_start
                and (seg.end or seg.start) > cursor
            ]
            if block:
                lines.append(
                    f"{format_timestamp(cursor)} – {format_timestamp(opp.timestamp_start)}"
                )
                lines.append(f'"{" ".join(seg.text for seg in block)}"')
                lines.append("")
            _append_callout(lines, opp)
            cursor = max(cursor, opp.timestamp_end or 0.0)
        tail = [seg for seg in transcript.segments if (seg.end or 0.0) > cursor]
        if tail:
            lines.append(
                f"{format_timestamp(cursor)} – "
                f"{format_timestamp(transcript.duration_seconds or cursor)}"
            )
            lines.append(f'"{" ".join(seg.text for seg in tail)}"')
            lines.append("")
    else:
        if transcript.segments:
            narration = " ".join(seg.text for seg in transcript.segments)
        else:
            narration = transcript.article_text or ""
        if narration:
            lines.append(f'"{narration}"')
            lines.append("")
        for opp in opportunities:
            _append_callout(lines, opp)
    return "\n".join(lines).rstrip() + "\n"


def _append_callout(lines: list[str], opp: BrollOpportunity) -> None:
    opp_title = opp.slug.replace("-", " ").title()
    if opp.timestamp_start is not None and opp.timestamp_end is not None:
        start = format_timestamp(opp.timestamp_start)
        end = format_timestamp(opp.timestamp_end)
        lines.append(f"> 🎬 **[{start}–{end}] {opp_title}**")
    else:
        lines.append(f"> 🎬 **{opp_title}**")
    lines.append(f"> {opp.reasoning}")
    lines.append(f"> Images: `scenes/{scene_folder_name(opp)}/`")
    lines.append("")


def write_plan_and_script(
    plan: BrollPlan, transcript: Transcript, out_dir: Path
) -> tuple[Path, Path]:
    """Write broll_plan.json and broll_script.md together — they stay in sync."""
    out_dir.mkdir(parents=True, exist_ok=True)
    plan_path = out_dir / PLAN_FILENAME
    script_path = out_dir / SCRIPT_FILENAME
    write_plan(plan, plan_path)
    script_path.write_text(render_script(plan, transcript), encoding="utf-8")
    logger.info("wrote %s and %s", plan_path.name, script_path.name)
    return plan_path, script_path

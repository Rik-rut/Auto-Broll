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
    title = plan.video_slug.replace("-", " ").title()
    lines = [
        f"# Broll Script — {title}",
        f"Source: {plan.video_file} · Duration: {format_timestamp(transcript.duration_seconds)}",
        "",
    ]
    opportunities = sorted(plan.broll_opportunities, key=lambda opp: opp.timestamp_start)
    cursor = 0.0
    for opp in opportunities:
        block = [
            seg
            for seg in transcript.segments
            if seg.start < opp.timestamp_start and seg.end > cursor
        ]
        if block:
            lines.append(f"{format_timestamp(cursor)} – {format_timestamp(opp.timestamp_start)}")
            lines.append(f'"{" ".join(seg.text for seg in block)}"')
            lines.append("")
        opp_title = opp.slug.replace("-", " ").title()
        start = format_timestamp(opp.timestamp_start)
        end = format_timestamp(opp.timestamp_end)
        lines.append(f"> 🎬 **[{start}–{end}] {opp_title}**")
        lines.append(f"> {opp.reasoning}")
        lines.append(f"> Images: `scenes/{scene_folder_name(opp)}/`")
        lines.append("")
        cursor = max(cursor, opp.timestamp_end)
    tail = [seg for seg in transcript.segments if seg.end > cursor]
    if tail:
        lines.append(
            f"{format_timestamp(cursor)} – {format_timestamp(transcript.duration_seconds)}"
        )
        lines.append(f'"{" ".join(seg.text for seg in tail)}"')
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


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

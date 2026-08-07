from __future__ import annotations

from pathlib import Path

from broll_agent.models import BrollOpportunity, BrollPlan
from broll_agent.planner import (
    PLAN_FILENAME,
    SCRIPT_FILENAME,
    load_plan,
    render_script,
    scene_folder_name,
    write_plan,
    write_plan_and_script,
)


def _sample_plan() -> BrollPlan:
    return BrollPlan(
        video_file="how_to_train_a_puppy.mp4",
        video_slug="how-to-train-a-puppy",
        analyzed_at="2026-08-06T10:00:00Z",
        llm_model="claude-sonnet-4-6",
        broll_opportunities=[
            BrollOpportunity(
                id="001",
                slug="puppy-chewing-a-shoe",
                timestamp_start=41.2,
                timestamp_end=47.8,
                context_text="...and this is where he chewed straight through my shoe.",
                reasoning="Concrete, visual anecdote with no on-screen demo.",
                image_prompt="a golden retriever puppy chewing a leather shoe",
                searxng_query="puppy chewing shoe",
            )
        ],
    )


def test_plan_round_trip(tmp_path: Path) -> None:
    plan = _sample_plan()
    dest = tmp_path / "out" / PLAN_FILENAME
    write_plan(plan, dest)
    loaded = load_plan(dest)
    assert loaded == plan
    assert loaded.broll_opportunities[0].timestamp_start == 41.2


def test_scene_folder_name() -> None:
    plan = _sample_plan()
    assert scene_folder_name(plan.broll_opportunities[0]) == "001_puppy-chewing-a-shoe"


def test_write_plan_and_script_creates_both(tmp_path: Path, transcript) -> None:
    plan = _sample_plan()
    plan_path, script_path = write_plan_and_script(plan, transcript, tmp_path)
    assert plan_path.exists() and plan_path.name == PLAN_FILENAME
    assert script_path.exists() and script_path.name == SCRIPT_FILENAME
    assert load_plan(plan_path) == plan


def test_render_script_contains_narration_and_callout(transcript) -> None:
    script = render_script(_sample_plan(), transcript)
    assert "# Broll Script — How To Train A Puppy" in script
    assert "Source: how_to_train_a_puppy.mp4 · Duration: 02:05" in script
    assert "00:00 – 00:41" in script
    assert "🎬 **[00:41–00:47] Puppy Chewing A Shoe**" in script
    assert "Concrete, visual anecdote" in script
    assert "Images: `scenes/001_puppy-chewing-a-shoe/`" in script
    # narration after the opportunity is still present
    assert "grab your leash" in script


def test_render_script_handles_no_opportunities(transcript) -> None:
    plan = _sample_plan()
    plan.broll_opportunities = []
    script = render_script(plan, transcript)
    assert "🎬" not in script
    assert "Welcome back" in script

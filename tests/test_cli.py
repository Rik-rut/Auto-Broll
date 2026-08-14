from __future__ import annotations

from pathlib import Path

import pytest

from broll_agent import cli
from broll_agent.config import Settings
from broll_agent.planner import PLAN_FILENAME
from broll_agent.transcribe import TRANSCRIPT_FILENAME

VALID_RESPONSE = """[
  {
    "id": "001",
    "slug": "casino-floor-betting",
    "timestamp_start": 0.0,
    "timestamp_end": 3.0,
    "context_text": "The casino floor is where the house always wins.",
    "reasoning": "Concrete, visual subject.",
    "image_prompt": "a busy casino floor with roulette tables",
    "searxng_query": "casino floor roulette"
  }
]"""


class FakeLLM:
    def __init__(self, cfg) -> None:
        self.calls = 0

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        return VALID_RESPONSE


@pytest.fixture()
def input_dir_with_srt(tmp_path: Path) -> Path:
    inp = tmp_path / "input"
    inp.mkdir()
    (inp / "casino.srt").write_text(
        "1\n00:00:00,000 --> 00:00:03,000\nThe casino floor is where the house always wins.\n",
        encoding="utf-8",
    )
    return inp


def test_select_inputs_finds_videos_and_transcripts(
    tmp_path: Path, input_dir_with_srt: Path
) -> None:
    (input_dir_with_srt / "movie.mp4").write_bytes(b"x")
    cfg = Settings(_env_file=None, input_dir=input_dir_with_srt)
    found = cli._select_inputs(cfg, None)
    assert [p.name for p in found] == ["casino.srt", "movie.mp4"]


def test_run_plan_processes_transcript_input(
    tmp_path: Path, input_dir_with_srt: Path, monkeypatch
) -> None:
    out = tmp_path / "output"
    cfg = Settings(_env_file=None, input_dir=input_dir_with_srt, output_dir=out)
    monkeypatch.setattr(cli, "LLMClient", FakeLLM)
    cli._run_plan(cfg, None, False)
    assert (out / "casino" / TRANSCRIPT_FILENAME).exists()
    assert (out / "casino" / PLAN_FILENAME).exists()


def test_run_plan_skips_existing_plan(
    tmp_path: Path, input_dir_with_srt: Path, monkeypatch
) -> None:
    instances: list[FakeLLM] = []

    def factory(cfg):
        llm = FakeLLM(cfg)
        instances.append(llm)
        return llm

    out = tmp_path / "output"
    cfg = Settings(_env_file=None, input_dir=input_dir_with_srt, output_dir=out)
    monkeypatch.setattr(cli, "LLMClient", factory)
    cli._run_plan(cfg, None, False)
    cli._run_plan(cfg, None, False)
    assert out.joinpath("casino", PLAN_FILENAME).exists()
    assert sum(llm.calls for llm in instances) == 1


def test_run_plan_completes_interrupted_transcript_job(
    tmp_path: Path, input_dir_with_srt: Path, monkeypatch
) -> None:
    out = tmp_path / "output"
    cfg = Settings(_env_file=None, input_dir=input_dir_with_srt, output_dir=out)
    monkeypatch.setattr(cli, "LLMClient", FakeLLM)
    stale_dir = out / "casino"
    stale_dir.mkdir(parents=True)
    (stale_dir / TRANSCRIPT_FILENAME).write_text(
        '{"video_file": "casino.srt", "segments": [{"start": 0.0, "end": 3.0, "text": "stale"}]}',
        encoding="utf-8",
    )
    cli._run_plan(cfg, None, False)
    assert (out / "casino" / PLAN_FILENAME).exists()

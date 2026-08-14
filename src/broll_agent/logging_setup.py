"""Structured logging setup — one line per opportunity per stage."""

from __future__ import annotations

import logging


def setup_logging(level: str = "INFO") -> None:
    upper = level.upper()
    if not hasattr(logging, upper):
        raise ValueError(f"invalid log level: {level!r}")
    logging.basicConfig(
        level=getattr(logging, upper),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

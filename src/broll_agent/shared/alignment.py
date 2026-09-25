"""Dimension alignment shared by image engines."""

from __future__ import annotations


def round_to_alignment(value: int, alignment: int) -> int:
    """Round a requested dimension down to an engine's required alignment."""
    rounded = int(value) // alignment * alignment
    if rounded < alignment:
        raise ValueError(f"{value} is smaller than the {alignment}px minimum")
    return rounded

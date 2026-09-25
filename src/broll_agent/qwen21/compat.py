"""Shims replacing Wan2GP application plumbing."""

from __future__ import annotations


def check_abort() -> None:
    """No-op: the CLI has no interactive abort channel."""

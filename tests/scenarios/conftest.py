"""Pytest fixtures for scenario replay tests."""

from __future__ import annotations

from pathlib import Path

import pytest

SCENARIOS_DIR = Path(__file__).parent / "definitions"


@pytest.fixture
def scenarios_dir() -> Path:
    """Return the absolute path to the scenarios definition directory."""
    return SCENARIOS_DIR

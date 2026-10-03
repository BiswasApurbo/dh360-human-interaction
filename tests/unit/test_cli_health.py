"""Unit tests for 'dh360-hi health' command."""

from __future__ import annotations

import json

import pytest

from dh360_hi.cli.main import main


def test_cli_health_table_output(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify 'dh360-hi health' outputs human-readable status table."""
    exit_code = main(["health", "--preset", "testing"])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Service Health Snapshot" in captured.out
    assert "runtime" in captured.out
    assert "database" in captured.out
    assert "HEALTHY" in captured.out


def test_cli_health_json_output(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify 'dh360-hi health --json' outputs machine-readable JSON."""
    exit_code = main(["health", "--preset", "testing", "--json"])
    assert exit_code == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["status"] in ("healthy", "degraded")
    assert "subsystems" in data
    assert "database" in data["subsystems"]

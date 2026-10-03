"""Unit tests for 'dh360-hi config' command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dh360_hi.cli.main import main


def test_cli_config_presets(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify 'dh360-hi config presets' lists all built-in presets."""
    exit_code = main(["config", "presets"])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "development" in captured.out
    assert "testing" in captured.out
    assert "production" in captured.out


def test_cli_config_show_formatted(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify 'dh360-hi config show' outputs formatted sections."""
    exit_code = main(["config", "show", "--preset", "testing"])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Active Configuration" in captured.out
    assert "db_path" in captured.out
    assert "Runtime" in captured.out


def test_cli_config_show_json(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify 'dh360-hi config show --json' outputs valid JSON."""
    exit_code = main(["config", "show", "--preset", "production", "--json"])
    assert exit_code == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["db_path"] == "dh360_hi.db"
    assert data["runtime"]["target_fps"] == 30.0
    assert data["require_active_for_gallery"] is True


def test_cli_config_validate_success(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify 'dh360-hi config validate' on valid configuration."""
    toml_path = tmp_path / "valid.toml"
    toml_path.write_text("db_path = ':memory:'\n[runtime]\ntarget_fps = 25.0\n")

    exit_code = main(["config", "validate", "--config-file", str(toml_path)])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "successful" in captured.out


def test_cli_config_validate_failure(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify 'dh360-hi config validate' on invalid configuration returns 1 with error on stderr."""
    toml_path = tmp_path / "invalid.toml"
    toml_path.write_text("[runtime]\ntarget_fps = -10.0\n")

    exit_code = main(["config", "validate", "--config-file", str(toml_path)])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "target_fps must be positive" in captured.err

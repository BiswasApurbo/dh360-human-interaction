"""Unit tests for root CLI entry point, dispatching, and error handling."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from dh360_hi.cli.main import create_parser, main
from dh360_hi.contracts.errors import EntityNotFoundError


def test_create_parser_structure() -> None:
    """Verify root argument parser registers all required subcommands."""
    parser = create_parser()
    assert parser.prog == "dh360-hi"

    # Test subcommand parsing
    args_run = parser.parse_args(["run", "--source", "synthetic"])
    assert args_run.command == "run"
    assert args_run.source == "synthetic"

    args_health = parser.parse_args(["health", "--json"])
    assert args_health.command == "health"
    assert args_health.json is True

    args_person = parser.parse_args(["person", "list"])
    assert args_person.command == "person"
    assert args_person.person_action == "list"

    args_session = parser.parse_args(["session", "list"])
    assert args_session.command == "session"
    assert args_session.session_action == "list"

    args_config = parser.parse_args(["config", "presets"])
    assert args_config.command == "config"
    assert args_config.config_action == "presets"


def test_main_help_exits_cleanly(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify root --help prints usage and exits cleanly with 0."""
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "Digital Health 360 Human Interaction" in captured.out


def test_main_no_args_exits_with_error() -> None:
    """Verify missing required subcommand exits with error code."""
    with pytest.raises(SystemExit):
        main([])


def test_main_catches_application_error(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify ApplicationError is caught at CLI boundary with formatted message on stderr."""
    with patch("dh360_hi.cli.main.handle_person") as mock_handler:
        mock_handler.side_effect = EntityNotFoundError(
            entity_type="person",
            entity_id="12345",
            message="Target person not found: 12345",
        )
        code = main(["person", "get", "12345"])
        assert code == 1

        captured = capsys.readouterr()
        assert "Error [ENTITY_NOT_FOUND]: Target person not found: 12345" in captured.err


def test_main_catches_keyboard_interrupt(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify KeyboardInterrupt returns exit code 130."""
    with patch("dh360_hi.cli.main.handle_run") as mock_handler:
        mock_handler.side_effect = KeyboardInterrupt()
        code = main(["run"])
        assert code == 130
        captured = capsys.readouterr()
        assert "interrupted" in captured.err


def test_main_handles_unexpected_error(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify unexpected non-ApplicationError errors exit with 2."""
    with patch("dh360_hi.cli.main.handle_health") as mock_handler:
        mock_handler.side_effect = RuntimeError("Fatal hardware failure")
        code = main(["health"])
        assert code == 2
        captured = capsys.readouterr()
        assert "Unexpected operational error: Fatal hardware failure" in captured.err

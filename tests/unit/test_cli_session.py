"""Unit tests for 'dh360-hi session' command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dh360_hi.application.config import HumanInteractionConfig
from dh360_hi.application.factory import create_interaction_service
from dh360_hi.backend.models import SessionRecord
from dh360_hi.cli.main import main


def _seed_sample_session(
    db_path: str, session_id: str = "00000000-0000-0000-0000-000000000001"
) -> None:
    """Seed a test session in SQLite."""
    config = HumanInteractionConfig(db_path=db_path)
    service = create_interaction_service(config=config, enable_perception=False)
    repo = service.session_repository
    assert repo is not None
    rec = SessionRecord(
        session_id=session_id,
        person_id=None,
        display_name=None,
        started_at=100.0,
        ended_at=125.0,
        duration_seconds=25.0,
        identity_was_known=False,
        identity_transitions_count=0,
        final_state="ENGAGED",
    )
    repo.save_session(rec)
    service.stop(timeout=1.0)


def test_cli_session_empty_list(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify 'session list' on empty database."""
    db_file = str(tmp_path / "sess_empty.db")
    exit_code = main(["session", "list", "--db-path", db_file])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "No historical interaction sessions found" in captured.out


def test_cli_session_list_and_get(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify 'session list' and 'session get' with seeded session."""
    db_file = str(tmp_path / "sess_test.db")
    target_id = "00000000-0000-0000-0000-000000000001"
    _seed_sample_session(db_file, target_id)

    # List
    exit_code = main(["session", "list", "--db-path", db_file, "--json"])
    assert exit_code == 0
    captured = capsys.readouterr()
    sessions = json.loads(captured.out)
    assert len(sessions) == 1
    assert sessions[0]["session_id"] == target_id

    # Get
    exit_code = main(["session", "get", target_id, "--db-path", db_file])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert target_id in captured.out
    assert "ENGAGED" in captured.out


def test_cli_session_query_and_analytics(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify 'session query' and 'session analytics'."""
    db_file = str(tmp_path / "sess_analytics.db")
    target_id = "00000000-0000-0000-0000-000000000002"
    _seed_sample_session(db_file, target_id)

    # Query
    exit_code = main(["session", "query", "--db-path", db_file, "--is-anonymous", "--json"])
    assert exit_code == 0
    captured = capsys.readouterr()
    res = json.loads(captured.out)
    assert res["total_matching"] == 1
    assert res["sessions"][0]["session_id"] == target_id

    # Analytics
    exit_code = main(["session", "analytics", "--db-path", db_file, "--json"])
    assert exit_code == 0
    captured = capsys.readouterr()
    analytics = json.loads(captured.out)
    assert analytics["total_sessions"] == 1
    assert analytics["total_duration_seconds"] == 25.0
    assert analytics["anonymous_sessions_count"] == 1


def test_cli_session_get_not_found(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify error on querying non-existent session."""
    db_file = str(tmp_path / "sess_missing.db")
    exit_code = main(
        ["session", "get", "00000000-0000-0000-0000-000000000099", "--db-path", db_file]
    )
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "ENTITY_NOT_FOUND" in captured.err

"""Unit tests for 'dh360-hi person' command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dh360_hi.cli.main import main


def test_cli_person_crud_flow(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify full CRUD lifecycle via 'dh360-hi person'."""
    db_file = str(tmp_path / "person_test.db")

    # 1. Initially empty list
    exit_code = main(["person", "list", "--db-path", db_file])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "No registered person profiles found" in captured.out

    # 2. Create person
    exit_code = main(
        [
            "person",
            "create",
            "--db-path",
            db_file,
            "--display-name",
            "Dr. Alice Smith",
            "--preferred-name",
            "Alice",
            "--status",
            "active",
            "--consent-status",
            "granted",
            "--json",
        ]
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    created_data = json.loads(captured.out)
    person_id = created_data["person_id"]
    assert created_data["display_name"] == "Dr. Alice Smith"
    assert created_data["preferred_name"] == "Alice"
    assert created_data["consent_status"] == "granted"

    # 3. Get person
    exit_code = main(["person", "get", person_id, "--db-path", db_file])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Dr. Alice Smith" in captured.out
    assert person_id in captured.out

    # 4. List persons with JSON
    exit_code = main(["person", "list", "--db-path", db_file, "--json"])
    assert exit_code == 0
    captured = capsys.readouterr()
    profiles = json.loads(captured.out)
    assert len(profiles) == 1
    assert profiles[0]["person_id"] == person_id

    # 5. Update person
    exit_code = main(
        [
            "person",
            "update",
            person_id,
            "--db-path",
            db_file,
            "--preferred-name",
            "Ali",
            "--json",
        ]
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    updated_data = json.loads(captured.out)
    assert updated_data["preferred_name"] == "Ali"

    # 6. Reset biometrics
    exit_code = main(["person", "reset-biometrics", person_id, "--db-path", db_file])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "0 embedding vector(s) deleted" in captured.out

    # 7. Delete person
    exit_code = main(["person", "delete", person_id, "--db-path", db_file])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Successfully deleted person profile" in captured.out

    # 8. Get deleted person fails with 1
    exit_code = main(["person", "get", person_id, "--db-path", db_file])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "ENTITY_NOT_FOUND" in captured.err

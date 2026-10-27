"""Regression and lifecycle tests verifying registered person persistence across restarts.

Verifies:
1. Person can be registered and retrieved.
2. Database closed and reopened, person exists.
3. Profile details and Q&A pairs survive restart.
4. Normal server shutdown does not delete registered people.
5. Startup uses intended persistent database path (defaulting to dh360_hi.db for CLI).
6. Temporary test database cannot accidentally replace development database.
7. Authorization session state ends on shutdown while person data remains after restart.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from starlette.testclient import TestClient

from dh360_hi.api.app import create_app
from dh360_hi.application.config import DEFAULT_DB_PATH, HumanInteractionConfig
from dh360_hi.application.factory import create_interaction_service
from dh360_hi.application.loader import load_config, resolve_cli_db_path
from dh360_hi.backend.database import DatabaseManager
from dh360_hi.backend.models import (
    ConsentStatus,
    PersonStatus,
)
from dh360_hi.backend.person_service import PersonService, QAPair
from dh360_hi.backend.repository import SQLitePersonRepository
from dh360_hi.contracts.authorization import AuthorizationStatus
from dh360_hi.contracts.dto import CreatePersonCommand
from dh360_hi.edge.authorization_controller import AuthorizationController
from dh360_hi.edge.session_manager import SessionManager
from dh360_hi.vision.embedder import normalize_l2


def test_person_can_be_registered_and_retrieved(tmp_path: Path) -> None:
    """Requirement 1: A person can be registered and retrieved with all fields intact."""
    db_file = tmp_path / "test_person.db"
    db = DatabaseManager(str(db_file), auto_init=True)
    repo = SQLitePersonRepository(db)
    service = PersonService(repo)

    qa = [
        QAPair(question="What is my department?", answer="Machine Learning"),
        QAPair(question="What is my room?", answer="Lab 402"),
    ]

    created = service.create_person(
        display_name="Dr. Jane Doe",
        preferred_name="Jane",
        role="Lead ML Scientist",
        department="Robotech ML",
        office_room="Lab 402",
        working_hours="09:00 - 18:00",
        profile_description="Specializes in biomolecular modeling and interactive AI.",
        qa_pairs=qa,
        notes="Internal staff",
    )

    assert created.person_id is not None
    assert created.display_name == "Dr. Jane Doe"
    assert created.preferred_name == "Jane"
    assert created.status == PersonStatus.ACTIVE
    assert created.consent_status == ConsentStatus.GRANTED
    assert created.notes == "Internal staff"

    # Retrieve by ID
    retrieved = service.get_person(created.person_id)
    assert retrieved is not None
    assert retrieved.person_id == created.person_id
    assert retrieved.display_name == "Dr. Jane Doe"
    assert retrieved.preferred_name == "Jane"
    assert retrieved.status == PersonStatus.ACTIVE
    assert retrieved.consent_status == ConsentStatus.GRANTED

    # Retrieve context
    ctx = service.get_person_context(created.person_id)
    assert ctx is not None
    assert ctx.role == "Lead ML Scientist"
    assert ctx.department == "Robotech ML"
    assert ctx.office_room == "Lab 402"
    assert ctx.working_hours == "09:00 - 18:00"
    assert ctx.profile_description == "Specializes in biomolecular modeling and interactive AI."
    assert len(ctx.qa_pairs) == 2
    assert ctx.qa_pairs[0].question == "What is my department?"
    assert ctx.qa_pairs[0].answer == "Machine Learning"

    db.close()


def test_database_closed_and_reopened_person_exists(tmp_path: Path) -> None:
    """Requirement 2: Database can be closed and reopened; person and embeddings remain."""
    db_file = tmp_path / "reopened.db"

    # 1. First run: Start service, register person, add face embeddings
    cfg1 = HumanInteractionConfig(db_path=str(db_file))
    service1 = create_interaction_service(config=cfg1, enable_perception=False)

    person = service1.create_person(
        CreatePersonCommand(
            display_name="Alice Smith",
            preferred_name="Alice",
            role="Perception Engineer",
            department="Vision Lab",
        )
    )
    person_id = person.person_id

    # Add 5 dummy embeddings
    embeddings = [normalize_l2(np.full(128, float(i + 1), dtype=np.float32)) for i in range(5)]
    added = service1.person_service.add_person_embeddings(
        person_id=person_id,
        embeddings=embeddings,
    )
    assert len(added) == 5

    profile1 = service1.person_service.get_person(person_id)
    assert profile1.embedding_count == 5

    # Shut down service 1 cleanly
    service1.stop()

    # 2. Second run: Start a completely fresh service instance against same database path
    cfg2 = HumanInteractionConfig(db_path=str(db_file))
    service2 = create_interaction_service(config=cfg2, enable_perception=False)

    try:
        # Internal person service profile must retain embedding count
        profile2 = service2.person_service.get_person(person_id)
        assert profile2 is not None
        assert profile2.person_id == person_id
        assert profile2.display_name == "Alice Smith"
        assert profile2.preferred_name == "Alice"
        assert profile2.embedding_count == 5
        assert profile2.status == PersonStatus.ACTIVE
        assert profile2.consent_status == ConsentStatus.GRANTED

        # Public service DTO must match
        dto = service2.get_person(person_id)
        assert dto is not None
        assert dto.display_name == "Alice Smith"

        # Context must exist
        ctx2 = service2.person_service.get_person_context(person_id)
        assert ctx2 is not None
        assert ctx2.role == "Perception Engineer"
        assert ctx2.department == "Vision Lab"

        # List must include person
        all_persons = service2.list_persons()
        assert any(p.person_id == person_id for p in all_persons)
    finally:
        service2.stop()


def test_profile_details_and_qa_pairs_survive_restart(tmp_path: Path) -> None:
    """Requirement 3: Profile details, role, department, office room, and Q&A survive."""
    db_file = tmp_path / "qa_survive.db"

    # Run 1: Create person with full context
    cfg1 = HumanInteractionConfig(db_path=str(db_file))
    service1 = create_interaction_service(config=cfg1, enable_perception=False)

    qa_data = [
        QAPair(
            question="What projects am I leading?",
            answer="DH360 Human Interaction edge runtime",
        ),
        QAPair(question="Where is my desk?", answer="Building 2, 3rd Floor, Desk 12"),
        QAPair(question="What are my core hours?", answer="10:00 to 17:00 weekdays"),
    ]

    p = service1.create_person(
        CreatePersonCommand(
            display_name="Robert Taylor",
            preferred_name="Bob",
            role="Principal Edge Architect",
            department="Hardware & Edge Systems",
            office_room="B2-312",
            working_hours="10:00 - 17:00",
            profile_description=(
                "Leads embedded edge runtime and biometric interaction architectures."
            ),
            qa_pairs=qa_data,
            notes="Key architect",
        )
    )
    pid = p.person_id
    service1.stop()

    # Run 2: Reopen from cold state
    cfg2 = HumanInteractionConfig(db_path=str(db_file))
    service2 = create_interaction_service(config=cfg2, enable_perception=False)

    try:
        ctx = service2.person_service.get_person_context(pid)
        assert ctx is not None
        assert ctx.person_id == pid
        assert ctx.display_name == "Robert Taylor"
        assert ctx.preferred_name == "Bob"
        assert ctx.role == "Principal Edge Architect"
        assert ctx.department == "Hardware & Edge Systems"
        assert ctx.office_room == "B2-312"
        assert ctx.working_hours == "10:00 - 17:00"
        assert ctx.profile_description == (
            "Leads embedded edge runtime and biometric interaction architectures."
        )
        assert len(ctx.qa_pairs) == 3
        assert ctx.qa_pairs[0].question == "What projects am I leading?"
        assert ctx.qa_pairs[0].answer == "DH360 Human Interaction edge runtime"
        assert ctx.qa_pairs[1].question == "Where is my desk?"
        assert ctx.qa_pairs[1].answer == "Building 2, 3rd Floor, Desk 12"
        assert ctx.qa_pairs[2].question == "What are my core hours?"
        assert ctx.qa_pairs[2].answer == "10:00 to 17:00 weekdays"
    finally:
        service2.stop()


def test_normal_server_shutdown_does_not_delete_registered_people(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Requirement 4: FastAPI server shutdown does not delete registered people."""
    token = "test-secret-operator-token-12345678"
    monkeypatch.setenv("DH360_HI_API__OPERATOR_TOKEN", token)
    monkeypatch.setenv("DH360_HI_API__OPERATOR_SCOPES", "persons:read,persons:create")

    db_file = tmp_path / "server_lifecycle.db"

    # Start service and register user
    cfg = HumanInteractionConfig(db_path=str(db_file))
    service = create_interaction_service(config=cfg, enable_perception=False)
    p = service.create_person(
        CreatePersonCommand(
            display_name="Sarah Connor",
            preferred_name="Sarah",
            role="Security Lead",
        )
    )
    pid = p.person_id

    app = create_app(service=service)

    # Simulate full API server lifecycle using Starlette TestClient with lifespan context
    with TestClient(app) as client:
        # Access endpoints as operator
        resp = client.get(
            "/persons",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert any(item["person_id"] == pid for item in data)

    # Server has exited lifespan context (shutdown complete)
    # Verify directly against database file using raw SQLite
    conn = sqlite3.connect(str(db_file))
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT person_id, display_name, status FROM persons WHERE person_id = ?",
            (pid,),
        )
        row = cur.fetchone()
        assert row is not None
        assert row[0] == pid
        assert row[1] == "Sarah Connor"
        assert row[2] == "active"
    finally:
        conn.close()


def test_startup_uses_intended_persistent_database_path() -> None:
    """Requirement 5: CLI startup defaults to dh360_hi.db and respects CLI / env overrides."""
    # 1. No CLI args and no environment variable -> resolves to DEFAULT_DB_PATH
    with patch.dict(os.environ, {}, clear=True):
        args_none = argparse.Namespace(db_path=None, preset=None, config_file=None)
        assert resolve_cli_db_path(args_none) == DEFAULT_DB_PATH
        assert resolve_cli_db_path(None) == DEFAULT_DB_PATH

    # 2. Explicit CLI --db-path overrides everything
    args_custom = argparse.Namespace(db_path="/var/data/custom.db", preset=None, config_file=None)
    assert resolve_cli_db_path(args_custom) == "/var/data/custom.db"

    # 3. Environment variable DH360_HI_DB_PATH is respected (returns None to defer to loader)
    with patch.dict(os.environ, {"DH360_HI_DB_PATH": "/custom/env.db"}, clear=True):
        assert resolve_cli_db_path(args_none) is None
        cfg = load_config()
        assert cfg.db_path == "/custom/env.db"

    # 4. Preset testing is respected (returns None, config sets :memory:)
    args_preset = argparse.Namespace(db_path=None, preset="testing", config_file=None)
    assert resolve_cli_db_path(args_preset) is None
    cfg_test = load_config(preset="testing")
    assert cfg_test.db_path == ":memory:"

    # 5. Preset production is respected (returns None, config sets dh360_hi.db)
    args_prod = argparse.Namespace(db_path=None, preset="production", config_file=None)
    assert resolve_cli_db_path(args_prod) is None
    cfg_prod = load_config(preset="production")
    assert cfg_prod.db_path == DEFAULT_DB_PATH


def test_temporary_test_database_cannot_accidentally_replace_development_database(
    tmp_path: Path,
) -> None:
    """Requirement 6: Isolated test databases do not overwrite development database."""
    # Create a mock "development" database
    dev_db_file = tmp_path / "dh360_hi.db"
    dev_db = DatabaseManager(str(dev_db_file), auto_init=True)
    dev_repo = SQLitePersonRepository(dev_db)
    dev_service = PersonService(dev_repo)
    dev_p = dev_service.create_person("Existing Dev User", preferred_name="DevUser")
    dev_db.close()

    # Now run an isolated test against an ephemeral or separate temp database
    test_db_file = tmp_path / "isolated_test.db"
    test_db = DatabaseManager(str(test_db_file), auto_init=True)
    test_repo = SQLitePersonRepository(test_db)
    test_service = PersonService(test_repo)
    test_service.create_person("Test Only User", preferred_name="Tester")
    test_db.close()

    # Verify development database was completely unaffected
    check_db = DatabaseManager(str(dev_db_file), auto_init=False)
    check_repo = SQLitePersonRepository(check_db)
    check_service = PersonService(check_repo)
    try:
        dev_profile = check_service.get_person(dev_p.person_id)
        assert dev_profile is not None
        assert dev_profile.display_name == "Existing Dev User"

        # Test user must NOT exist in dev database
        all_dev_persons = check_service.list_people()
        assert len(all_dev_persons) == 1
        assert all_dev_persons[0].person_id == dev_p.person_id
    finally:
        check_db.close()


def test_authorization_can_end_on_shutdown_while_person_data_remains_after_restart(
    tmp_path: Path,
) -> None:
    """Requirement 7: Authorization sessions end cleanly on shutdown while person data persists."""
    db_file = tmp_path / "auth_persistence.db"

    # Step 1: Create service and register person
    cfg = HumanInteractionConfig(db_path=str(db_file))
    service1 = create_interaction_service(config=cfg, enable_perception=False)
    person = service1.create_person(
        CreatePersonCommand(
            display_name="Authorized Operative",
            preferred_name="Operative",
            role="Security Inspector",
        )
    )
    pid = person.person_id

    # Construct session manager and authorization controller
    sm1 = SessionManager()
    auth_ctrl1 = AuthorizationController(session_manager=sm1, timeout_seconds=30.0)

    # Start authorization session
    session1 = auth_ctrl1.start()
    assert session1.status == AuthorizationStatus.ACTIVE
    assert auth_ctrl1.get_current() is not None

    # Shutdown process (simulates Ctrl+C or app closing)
    auth_ctrl1.close()
    service1.stop()

    # Step 2: Restart from persistent storage
    service2 = create_interaction_service(config=cfg, enable_perception=False)
    sm2 = SessionManager()
    auth_ctrl2 = AuthorizationController(session_manager=sm2, timeout_seconds=30.0)

    try:
        # Ephemeral authorization session from dead process is cleanly gone (no active session)
        assert auth_ctrl2.get_current() is None

        # Registered person profile remains fully intact in persistent database
        reopened_person = service2.get_person(pid)
        assert reopened_person is not None
        assert reopened_person.person_id == pid
        assert reopened_person.display_name == "Authorized Operative"
        assert reopened_person.preferred_name == "Operative"
        assert reopened_person.status == PersonStatus.ACTIVE

        # Can start a fresh authorization session cleanly
        new_session = auth_ctrl2.start()
        assert new_session.status == AuthorizationStatus.ACTIVE
        assert auth_ctrl2.get_current() is not None
    finally:
        auth_ctrl2.close()
        service2.stop()

"""Unit tests for SQLite Schema v3 migrations, person profile columns, and AI context fields."""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

import dh360_hi.contracts  # noqa: F401
from dh360_hi.backend.database import DatabaseManager
from dh360_hi.backend.models import PersonRecord, utc_now_iso
from dh360_hi.backend.schema import CURRENT_SCHEMA_VERSION, SCHEMA_V2_SQL

if TYPE_CHECKING:
    from pathlib import Path


def test_fresh_database_creates_all_v3_columns() -> None:
    """Verify that fresh database creation at schema v3 includes all six new person columns."""
    db = DatabaseManager(":memory:")
    try:
        assert db.get_schema_version() == 3
        assert CURRENT_SCHEMA_VERSION == 3

        with db.readonly_cursor() as cur:
            cur.execute("PRAGMA table_info(persons);")
            columns = {row["name"]: row for row in cur.fetchall()}

            # Baseline columns
            assert "person_id" in columns
            assert "display_name" in columns
            assert "preferred_name" in columns
            assert "status" in columns
            assert "consent_status" in columns
            assert "notes" in columns

            # Hospital profile columns
            assert "role" in columns
            assert "department" in columns
            assert "office_room" in columns
            assert "working_hours" in columns

            # AI context columns
            assert "profile_description" in columns
            assert "ai_qa_pairs" in columns

            # Default value check
            assert columns["ai_qa_pairs"]["dflt_value"] == "'[]'"
    finally:
        db.close()


def test_v2_to_v3_migration_preserves_existing_person_data(tmp_path: Path) -> None:
    """Verify that a v2 database migrates to v3 preserving records and defaults."""
    db_file = tmp_path / "v2_legacy.db"

    # Step 1: Manually set up a pure v2 database
    conn = sqlite3.connect(str(db_file))
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL,
            description TEXT NOT NULL
        );
        """
    )
    # v1 tables before v3 columns: create persons and embeddings like in original v1
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS persons (
            person_id TEXT PRIMARY KEY,
            display_name TEXT NOT NULL,
            preferred_name TEXT,
            status TEXT NOT NULL DEFAULT 'active',
            consent_status TEXT NOT NULL DEFAULT 'granted',
            consent_timestamp TEXT,
            notes TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS face_embeddings (
            embedding_id TEXT PRIMARY KEY,
            person_id TEXT NOT NULL,
            embedding BLOB NOT NULL,
            dimension INTEGER NOT NULL,
            model_name TEXT NOT NULL,
            model_version TEXT NOT NULL,
            quality_score REAL,
            enrollment_batch_id TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (person_id) REFERENCES persons(person_id) ON DELETE CASCADE
        );
        """
    )
    # apply v2 schema
    conn.executescript(SCHEMA_V2_SQL)
    now_iso = utc_now_iso()
    conn.execute(
        "INSERT INTO schema_version (version, applied_at, description) VALUES (1, ?, 'v1');",
        (now_iso,),
    )
    conn.execute(
        "INSERT INTO schema_version (version, applied_at, description) VALUES (2, ?, 'v2');",
        (now_iso,),
    )

    # Insert legacy person record
    pid = "11111111-2222-3333-4444-555555555555"
    conn.execute(
        """
        INSERT INTO persons (
            person_id, display_name, preferred_name, status, consent_status,
            notes, created_at, updated_at
        ) VALUES (?, 'Dr. Jane Doe', 'Jane', 'active', 'granted',
                  'Chief Cardiologist', '2026-10-01T00:00:00Z', '2026-10-01T00:00:00Z');
        """,
        (pid,),
    )
    conn.commit()
    conn.close()

    # Step 2: Open with DatabaseManager, triggering automatic migration to v3
    db = DatabaseManager(db_file)
    try:
        assert db.get_schema_version() == 3

        with db.readonly_cursor() as cur:
            cur.execute("SELECT * FROM persons WHERE person_id = ?;", (pid,))
            row = cur.fetchone()
            assert row is not None
            assert row["display_name"] == "Dr. Jane Doe"
            assert row["preferred_name"] == "Jane"
            assert row["notes"] == "Chief Cardiologist"
            # New columns should have NULL / default values
            assert row["role"] is None
            assert row["department"] is None
            assert row["office_room"] is None
            assert row["working_hours"] is None
            assert row["profile_description"] is None
            assert row["ai_qa_pairs"] == "[]"
    finally:
        db.close()


def test_person_record_model_defaults() -> None:
    """Verify PersonRecord model dataclass carries the new fields with correct defaults."""
    pid = "12345678-1234-5678-1234-567812345678"
    record = PersonRecord(
        person_id=pid,
        display_name="Dr. Gregory House",
    )
    assert record.role is None
    assert record.department is None
    assert record.office_room is None
    assert record.working_hours is None
    assert record.profile_description is None
    assert record.ai_qa_pairs_json == "[]"

    # With explicit values
    custom_record = PersonRecord(
        person_id=pid,
        display_name="Dr. Gregory House",
        role="Head of Diagnostic Medicine",
        department="Diagnostics",
        office_room="Room 404",
        working_hours="10:00 - 16:00",
        profile_description="Specialist in infectious diseases and nephrology.",
        ai_qa_pairs_json=('[{"question":"Where is House?","answer":"In clinic."}]'),
    )
    assert custom_record.role == "Head of Diagnostic Medicine"
    assert custom_record.department == "Diagnostics"
    assert custom_record.office_room == "Room 404"
    assert custom_record.working_hours == "10:00 - 16:00"
    assert custom_record.profile_description == (
        "Specialist in infectious diseases and nephrology."
    )
    assert custom_record.ai_qa_pairs_json == (
        '[{"question":"Where is House?","answer":"In clinic."}]'
    )

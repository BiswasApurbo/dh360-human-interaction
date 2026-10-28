"""Unit tests for SQLite Schema v2 migrations, tables, and indices."""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

import pytest

from dh360_hi.backend.database import DatabaseManager
from dh360_hi.backend.exceptions import SchemaVersionError
from dh360_hi.backend.models import SessionRecord, utc_now_iso
from dh360_hi.backend.schema import CURRENT_SCHEMA_VERSION, SCHEMA_V1_SQL
from dh360_hi.backend.session_repository import SQLiteSessionRepository

if TYPE_CHECKING:
    from pathlib import Path


def test_fresh_database_initializes_at_v2() -> None:
    """Verify that a newly initialized database starts at current schema version with all tables."""
    db = DatabaseManager(":memory:")
    try:
        assert db.get_schema_version() == CURRENT_SCHEMA_VERSION
        assert CURRENT_SCHEMA_VERSION >= 2

        with db.readonly_cursor() as cur:
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
            tables = [row["name"] for row in cur.fetchall()]
            assert "schema_version" in tables
            assert "persons" in tables
            assert "face_embeddings" in tables
            assert "interaction_sessions" in tables

            # Verify indices on interaction_sessions
            cur.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='index' AND tbl_name='interaction_sessions';"
            )
            indices = [row["name"] for row in cur.fetchall()]
            assert "idx_interaction_sessions_person_id" in indices
            assert "idx_interaction_sessions_ended_at" in indices
            assert "idx_interaction_sessions_started_at" in indices
            assert "idx_interaction_sessions_created_at" in indices
    finally:
        db.close()


def test_v1_database_migrates_to_v2_preserving_data(tmp_path: Path) -> None:
    """Verify that an existing v1 database automatically migrates to v2 without data loss."""
    db_file = tmp_path / "v1_legacy.db"

    # Step 1: Manually initialize a pure v1 database
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
    conn.executescript(SCHEMA_V1_SQL)
    conn.execute(
        "INSERT INTO schema_version (version, applied_at, description) "
        "VALUES (1, ?, 'Initial v1');",
        (utc_now_iso(),),
    )

    # Insert v1 data: person and face embedding
    pid = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    eid = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
    conn.execute(
        """
        INSERT INTO persons (
            person_id, display_name, status, consent_status, created_at, updated_at
        )
        VALUES (?, 'Legacy User', 'active', 'granted',
                '2026-10-01T00:00:00Z', '2026-10-01T00:00:00Z');
        """,
        (pid,),
    )
    conn.execute(
        """
        INSERT INTO face_embeddings (
            embedding_id, person_id, embedding, dimension, model_name, model_version, created_at
        ) VALUES (?, ?, X'00000000', 128, 'SFace', '2021dec', '2026-10-01T00:00:00Z');
        """,
        (eid, pid),
    )

    conn.commit()
    conn.close()

    # Step 2: Open with DatabaseManager (triggers automatic migration to current schema)
    db = DatabaseManager(db_file)
    try:
        assert db.get_schema_version() == CURRENT_SCHEMA_VERSION

        with db.readonly_cursor() as cur:
            # Check persons survived
            cur.execute("SELECT display_name FROM persons WHERE person_id = ?;", (pid,))
            p_row = cur.fetchone()
            assert p_row is not None
            assert p_row["display_name"] == "Legacy User"

            # Check face_embeddings survived
            cur.execute("SELECT model_name FROM face_embeddings WHERE embedding_id = ?;", (eid,))
            e_row = cur.fetchone()
            assert e_row is not None
            assert e_row["model_name"] == "SFace"

            # Check interaction_sessions table is created and operational
            cur.execute("SELECT COUNT(*) FROM interaction_sessions;")
            assert cur.fetchone()[0] == 0

        # Step 3: Insert session in migrated DB via repository
        repo = SQLiteSessionRepository(db)
        sid = "cccccccc-cccc-cccc-cccc-cccccccccccc"
        record = SessionRecord(
            session_id=sid,
            person_id=pid,
            display_name="Legacy User",
            started_at=100.0,
            ended_at=120.0,
            duration_seconds=20.0,
            final_state="engaged",
            identity_was_known=True,
            identity_transitions_count=1,
        )
        repo.save_session(record)

        queried = repo.get_session(sid)
        assert queried is not None
        assert queried.session_id == sid
        assert queried.person_id == pid
        assert queried.duration_seconds == 20.0
    finally:
        db.close()


def test_schema_migration_is_idempotent(tmp_path: Path) -> None:
    """Verify that multiple initialize() calls on an existing database are idempotent."""
    db_file = tmp_path / "idempotent_test.db"
    db = DatabaseManager(db_file)
    try:
        assert db.get_schema_version() == CURRENT_SCHEMA_VERSION
        # Call initialize repeatedly
        assert db.initialize() == CURRENT_SCHEMA_VERSION
        assert db.initialize() == CURRENT_SCHEMA_VERSION
        assert db.get_schema_version() == CURRENT_SCHEMA_VERSION
    finally:
        db.close()


def test_unsupported_future_version_raises_schema_error(tmp_path: Path) -> None:
    """Verify that opening a database with schema version > 2 raises SchemaVersionError."""
    db_file = tmp_path / "future_ver.db"
    db = DatabaseManager(db_file)
    try:
        with db.transaction() as cur:
            sql = (
                "INSERT INTO schema_version (version, applied_at, description) "
                "VALUES (99, ?, 'Future v99');"
            )
            cur.execute(sql, (utc_now_iso(),))
        with pytest.raises(SchemaVersionError, match="higher than supported version"):
            db.initialize()
    finally:
        db.close()

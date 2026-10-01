"""Unit tests for SQLite DatabaseManager, connection settings, transactions, and migrations."""

from __future__ import annotations

import sqlite3
import threading
from typing import TYPE_CHECKING

import pytest

from dh360_hi.backend.database import DatabaseManager
from dh360_hi.backend.exceptions import SchemaVersionError
from dh360_hi.backend.schema import CURRENT_SCHEMA_VERSION

if TYPE_CHECKING:
    from pathlib import Path


def test_in_memory_database_initialization() -> None:
    """Test initializing an ephemeral in-memory database with current schema version."""
    db = DatabaseManager(":memory:")
    try:
        assert db.is_memory is True
        assert db.get_schema_version() == CURRENT_SCHEMA_VERSION
        with db.readonly_cursor() as cur:
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
            tables = [row["name"] for row in cur.fetchall()]
            assert "persons" in tables
            assert "face_embeddings" in tables
            assert "interaction_sessions" in tables
            assert "schema_version" in tables
    finally:
        db.close()


def test_on_disk_database_creation_and_recovery(tmp_path: Path) -> None:
    """Test on-disk database creation, closing, and reopening data recovery."""
    db_file = tmp_path / "test_dh360.db"
    db = DatabaseManager(db_file)
    pid = "11111111-1111-1111-1111-111111111111"
    try:
        assert db.is_memory is False
        assert db.get_schema_version() == CURRENT_SCHEMA_VERSION

        # Insert test row directly
        with db.transaction() as cur:
            cur.execute(
                """
                INSERT INTO persons (
                    person_id, display_name, status, consent_status,
                    created_at, updated_at
                ) VALUES (?, 'Alice Smith', 'active', 'granted',
                          '2026-10-01T00:00:00Z', '2026-10-01T00:00:00Z');
                """,
                (pid,),
            )
    finally:
        db.close()

    # Re-open database from disk and verify persistence
    db2 = DatabaseManager(db_file)
    try:
        assert db2.get_schema_version() == CURRENT_SCHEMA_VERSION
        with db2.readonly_cursor() as cur:
            cur.execute("SELECT display_name FROM persons WHERE person_id = ?;", (pid,))
            row = cur.fetchone()
            assert row is not None
            assert row["display_name"] == "Alice Smith"
    finally:
        db2.close()


def test_foreign_keys_pragmas_enabled() -> None:
    """Test that foreign key constraints and WAL/pragmas are strictly enforced."""
    db = DatabaseManager(":memory:")
    try:
        with db.readonly_cursor() as cur:
            cur.execute("PRAGMA foreign_keys;")
            fk_row = cur.fetchone()
            assert fk_row is not None
            assert fk_row[0] == 1

        # Attempting to insert an embedding with a non-existent person_id must fail
        with pytest.raises(sqlite3.IntegrityError), db.transaction() as cur:
            cur.execute(
                """
                INSERT INTO face_embeddings (
                    embedding_id, person_id, embedding, dimension,
                    model_name, model_version, created_at
                ) VALUES (
                    '22222222-2222-2222-2222-222222222222',
                    '99999999-9999-9999-9999-999999999999',
                    X'00000000', 128, 'SFace', '2021dec', '2026-10-01T00:00:00Z'
                );
                """
            )
    finally:
        db.close()


def test_transaction_commit_and_rollback() -> None:
    """Test that successful transactions commit and exceptions trigger rollback."""
    db = DatabaseManager(":memory:")
    bob_id = "33333333-3333-3333-3333-333333333333"
    charlie_id = "44444444-4444-4444-4444-444444444444"
    try:
        # Successful transaction
        with db.transaction() as cur:
            cur.execute(
                """
                INSERT INTO persons (
                    person_id, display_name, status, consent_status,
                    created_at, updated_at
                ) VALUES (?, 'Bob Jones', 'active', 'granted',
                          '2026-10-01T00:00:00Z', '2026-10-01T00:00:00Z');
                """,
                (bob_id,),
            )

        with db.readonly_cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM persons WHERE person_id = ?;", (bob_id,))
            assert cur.fetchone()[0] == 1

        # Failing transaction with rollback
        with (
            pytest.raises(ValueError, match="Simulated application error"),
            db.transaction() as cur,
        ):
            cur.execute(
                """
                INSERT INTO persons (
                    person_id, display_name, status, consent_status,
                    created_at, updated_at
                ) VALUES (?, 'Charlie Brown', 'active', 'granted',
                          '2026-10-01T00:00:00Z', '2026-10-01T00:00:00Z');
                """,
                (charlie_id,),
            )
            raise ValueError("Simulated application error")

        # Charlie Brown must NOT be in the database
        with db.readonly_cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM persons WHERE person_id = ?;", (charlie_id,))
            assert cur.fetchone()[0] == 0
    finally:
        db.close()


def test_single_writer_thread_locking() -> None:
    """Test that concurrent writer threads are safely synchronized without database locks."""
    db = DatabaseManager(":memory:")
    errors: list[Exception] = []

    def writer_task(thread_id: int) -> None:
        try:
            for i in range(10):
                pid = f"00000000-0000-0000-{thread_id:04d}-{i:012d}"
                with db.transaction() as cur:
                    cur.execute(
                        """
                        INSERT INTO persons (
                            person_id, display_name, status, consent_status,
                            created_at, updated_at
                        ) VALUES (?, ?, 'active', 'granted',
                                  '2026-10-01T00:00:00Z', '2026-10-01T00:00:00Z');
                        """,
                        (pid, f"Person {thread_id}-{i}"),
                    )
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=writer_task, args=(t,)) for t in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    try:
        assert len(errors) == 0
        with db.readonly_cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM persons;")
            assert cur.fetchone()[0] == 40
    finally:
        db.close()


def test_higher_schema_version_raises_error() -> None:
    """Test that encountering a higher schema version than supported raises SchemaVersionError."""
    db = DatabaseManager(":memory:")
    try:
        with db.transaction() as cur:
            sql = "INSERT INTO schema_version (version, applied_at, description) VALUES (?, ?, ?);"
            cur.execute(sql, (999, "now", "Future version"))
        with pytest.raises(SchemaVersionError, match="higher than supported version"):
            db.initialize()
    finally:
        db.close()

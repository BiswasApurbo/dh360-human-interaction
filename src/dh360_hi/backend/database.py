"""SQLite database manager for connections, locking, foreign keys, and migrations."""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from dh360_hi.backend.exceptions import SchemaVersionError
from dh360_hi.backend.schema import CURRENT_SCHEMA_VERSION, SCHEMA_MIGRATIONS


class DatabaseManager:
    """Manages SQLite connections, transactions, schema versioning, and single-writer locking."""

    def __init__(self, db_path: str | Path = ":memory:", auto_init: bool = True) -> None:
        self._db_path = str(db_path)
        self._is_memory = self._db_path == ":memory:" or ":memory:" in self._db_path
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None

        if not self._is_memory:
            db_file = Path(self._db_path).resolve()
            db_file.parent.mkdir(parents=True, exist_ok=True)
            self._db_path = str(db_file)

        if auto_init:
            self.initialize()

    @property
    def db_path(self) -> str:
        """Configured database path or :memory: URI."""
        return self._db_path

    @property
    def is_memory(self) -> bool:
        """Check if database is ephemeral in-memory."""
        return self._is_memory

    def get_connection(self) -> sqlite3.Connection:
        """Get or create the thread-safe connection with PRAGMAs configured."""
        with self._lock:
            if self._conn is None:
                self._conn = sqlite3.connect(
                    self._db_path,
                    check_same_thread=False,
                    timeout=5.0,
                )
                self._conn.row_factory = sqlite3.Row
                # Configure critical PRAGMAs
                self._conn.execute("PRAGMA foreign_keys = ON;")
                self._conn.execute("PRAGMA busy_timeout = 5000;")
                if not self._is_memory:
                    self._conn.execute("PRAGMA journal_mode = WAL;")
            return self._conn

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Cursor, None, None]:
        """Context manager for atomic write transactions with single-writer lock."""
        with self._lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            try:
                cursor.execute("BEGIN IMMEDIATE")
                yield cursor
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                cursor.close()

    @contextmanager
    def readonly_cursor(self) -> Generator[sqlite3.Cursor, None, None]:
        """Context manager for read-only database queries."""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
        finally:
            cursor.close()

    def get_schema_version(self) -> int:
        """Query currently applied schema version from database."""
        with self.readonly_cursor() as cur:
            cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version';"
            )
            if not cur.fetchone():
                return 0
            cur.execute("SELECT MAX(version) FROM schema_version;")
            row = cur.fetchone()
            return int(row[0]) if row and row[0] is not None else 0

    def initialize(self) -> int:
        """Initialize database tables and run pending schema migrations."""
        with self._lock:
            current_ver = self.get_schema_version()
            if current_ver > CURRENT_SCHEMA_VERSION:
                msg = (
                    f"Database schema version {current_ver} is higher than "
                    f"supported version {CURRENT_SCHEMA_VERSION}"
                )
                raise SchemaVersionError(msg)

            if current_ver < CURRENT_SCHEMA_VERSION:
                with self.transaction() as cur:
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS schema_version (
                            version INTEGER PRIMARY KEY,
                            applied_at TEXT NOT NULL,
                            description TEXT NOT NULL
                        );
                        """
                    )
                    for ver in range(current_ver + 1, CURRENT_SCHEMA_VERSION + 1):
                        if ver not in SCHEMA_MIGRATIONS:
                            msg = f"Missing migration script for schema version {ver}"
                            raise SchemaVersionError(msg)

                        sql_script, description = SCHEMA_MIGRATIONS[ver]
                        cur.executescript(sql_script)
                        applied_at = datetime.now(UTC).isoformat()
                        insert_sql = """
                        INSERT INTO schema_version (version, applied_at, description)
                        VALUES (?, ?, ?);
                        """
                        cur.execute(insert_sql, (ver, applied_at, description))

            return self.get_schema_version()

    def close(self) -> None:
        """Close underlying database connection."""
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
                finally:
                    self._conn = None

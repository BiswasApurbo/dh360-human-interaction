"""SQLite repository and bounded asynchronous persistence worker for session records."""

from __future__ import annotations

import contextlib
import logging
import queue
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from dh360_hi.backend.database import DatabaseManager
from dh360_hi.backend.exceptions import (
    DatabaseError,
    ForeignKeyViolationError,
    ValidationError,
)
from dh360_hi.backend.models import (
    PersonEngagementStats,
    SessionAggregateAnalytics,
    SessionQueryFilter,
    SessionRecord,
    validate_uuid,
)

if TYPE_CHECKING:
    from dh360_hi.contracts.interaction import InteractionSummary

logger = logging.getLogger(__name__)


class BaseSessionRepository(ABC):
    """Abstract interface for Interaction Session persistence operations."""

    @abstractmethod
    def save_session(self, session: SessionRecord) -> SessionRecord:
        """Persist or update an interaction session record.

        Args:
            session: SessionRecord to persist.

        Returns:
            The persisted SessionRecord.
        """

    @abstractmethod
    def get_session(self, session_id: str) -> SessionRecord | None:
        """Retrieve an interaction session record by UUID.

        Args:
            session_id: Unique UUID of the session.

        Returns:
            SessionRecord if found, otherwise None.
        """

    @abstractmethod
    def list_sessions(
        self,
        person_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[SessionRecord]:
        """List interaction sessions, optionally filtered by person ID, with pagination.

        Args:
            person_id: Optional UUID filter for identified person.
            limit: Maximum number of records to return (must be > 0).
            offset: Number of records to skip (must be >= 0).

        Returns:
            List of matching SessionRecord instances ordered chronologically descending.
        """

    @abstractmethod
    def count_sessions(self, person_id: str | None = None) -> int:
        """Count total interaction session records, optionally filtered by person.

        Args:
            person_id: Optional UUID filter for identified person.

        Returns:
            Total count of matching session records.
        """

    @abstractmethod
    def query_sessions(self, filter: SessionQueryFilter) -> list[SessionRecord]:
        """Query interaction sessions matching filter criteria with pagination.

        Args:
            filter: SessionQueryFilter containing criteria and pagination parameters.

        Returns:
            List of matching SessionRecord instances ordered chronologically descending.
        """

    @abstractmethod
    def count_query_sessions(self, filter: SessionQueryFilter) -> int:
        """Count total interaction sessions matching filter criteria.

        Args:
            filter: SessionQueryFilter containing criteria parameters.

        Returns:
            Total count of matching session records.
        """

    @abstractmethod
    def get_aggregate_analytics(
        self,
        filter: SessionQueryFilter | None = None,
    ) -> SessionAggregateAnalytics:
        """Compute aggregate statistics directly via SQL over filtered sessions.

        Args:
            filter: Optional SessionQueryFilter criteria.

        Returns:
            SessionAggregateAnalytics with computed metrics.
        """

    @abstractmethod
    def get_person_engagement_stats(
        self,
        person_id: str,
    ) -> PersonEngagementStats | None:
        """Compute historical engagement metrics for a specific enrolled person.

        Args:
            person_id: UUID of the target person.

        Returns:
            PersonEngagementStats if sessions exist for this person, otherwise None.
        """

    @abstractmethod
    def delete_session(self, session_id: str) -> bool:
        """Delete an interaction session record by UUID.

        Args:
            session_id: Unique UUID of the session to delete.

        Returns:
            True if a record was deleted, False otherwise.
        """


class SQLiteSessionRepository(BaseSessionRepository):
    """SQLite-backed implementation of BaseSessionRepository."""

    def __init__(self, db_manager: DatabaseManager) -> None:
        self._db = db_manager

    def save_session(self, session: SessionRecord) -> SessionRecord:
        """Persist or idempotently update an interaction session record in SQLite."""
        validate_uuid(session.session_id, "session_id")
        if session.person_id is not None:
            validate_uuid(session.person_id, "person_id")

        with self._db.transaction() as cur:
            try:
                sql = """
                INSERT INTO interaction_sessions (
                    session_id,
                    person_id,
                    display_name,
                    started_at,
                    ended_at,
                    duration_seconds,
                    final_state,
                    identity_was_known,
                    identity_transitions_count,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    person_id = excluded.person_id,
                    display_name = excluded.display_name,
                    started_at = excluded.started_at,
                    ended_at = excluded.ended_at,
                    duration_seconds = excluded.duration_seconds,
                    final_state = excluded.final_state,
                    identity_was_known = excluded.identity_was_known,
                    identity_transitions_count = excluded.identity_transitions_count,
                    created_at = excluded.created_at;
                """
                cur.execute(
                    sql,
                    (
                        session.session_id,
                        session.person_id,
                        session.display_name,
                        session.started_at,
                        session.ended_at,
                        session.duration_seconds,
                        session.final_state,
                        1 if session.identity_was_known else 0,
                        session.identity_transitions_count,
                        session.created_at,
                    ),
                )
                return session
            except sqlite3.IntegrityError as exc:
                err_msg = str(exc).lower()
                if "foreign key" in err_msg:
                    msg = f"Foreign key violation: person_id '{session.person_id}' does not exist"
                    raise ForeignKeyViolationError(msg) from exc
                msg = f"Failed to save session '{session.session_id}': {exc}"
                raise DatabaseError(msg) from exc

    def get_session(self, session_id: str) -> SessionRecord | None:
        """Retrieve an interaction session record by UUID."""
        validate_uuid(session_id, "session_id")

        with self._db.readonly_cursor() as cur:
            cur.execute(
                """
                SELECT
                    session_id,
                    person_id,
                    display_name,
                    started_at,
                    ended_at,
                    duration_seconds,
                    final_state,
                    identity_was_known,
                    identity_transitions_count,
                    created_at
                FROM interaction_sessions
                WHERE session_id = ?;
                """,
                (session_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return self._row_to_record(row)

    def list_sessions(
        self,
        person_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[SessionRecord]:
        """List interaction sessions, optionally filtered by person ID, with pagination."""
        if limit <= 0:
            msg = f"limit must be greater than 0, got {limit}"
            raise ValidationError(msg)
        if offset < 0:
            msg = f"offset cannot be negative, got {offset}"
            raise ValidationError(msg)
        if person_id is not None:
            validate_uuid(person_id, "person_id")

        with self._db.readonly_cursor() as cur:
            if person_id is not None:
                cur.execute(
                    """
                    SELECT
                        session_id,
                        person_id,
                        display_name,
                        started_at,
                        ended_at,
                        duration_seconds,
                        final_state,
                        identity_was_known,
                        identity_transitions_count,
                        created_at
                    FROM interaction_sessions
                    WHERE person_id = ?
                    ORDER BY ended_at DESC, created_at DESC, session_id ASC
                    LIMIT ? OFFSET ?;
                    """,
                    (person_id, limit, offset),
                )
            else:
                cur.execute(
                    """
                    SELECT
                        session_id,
                        person_id,
                        display_name,
                        started_at,
                        ended_at,
                        duration_seconds,
                        final_state,
                        identity_was_known,
                        identity_transitions_count,
                        created_at
                    FROM interaction_sessions
                    ORDER BY ended_at DESC, created_at DESC, session_id ASC
                    LIMIT ? OFFSET ?;
                    """,
                    (limit, offset),
                )
            rows = cur.fetchall()
            return [self._row_to_record(r) for r in rows]

    def count_sessions(self, person_id: str | None = None) -> int:
        """Count total interaction session records, optionally filtered by person ID."""
        if person_id is not None:
            validate_uuid(person_id, "person_id")

        with self._db.readonly_cursor() as cur:
            if person_id is not None:
                cur.execute(
                    "SELECT COUNT(*) FROM interaction_sessions WHERE person_id = ?;",
                    (person_id,),
                )
            else:
                cur.execute("SELECT COUNT(*) FROM interaction_sessions;")
            row = cur.fetchone()
            return int(row[0]) if row and row[0] is not None else 0

    def delete_session(self, session_id: str) -> bool:
        """Delete an interaction session record by UUID."""
        validate_uuid(session_id, "session_id")

        with self._db.transaction() as cur:
            cur.execute("DELETE FROM interaction_sessions WHERE session_id = ?;", (session_id,))
            return cur.rowcount > 0

    @staticmethod
    def _build_filter_clause(
        filter_criteria: SessionQueryFilter | None,
    ) -> tuple[str, list[Any]]:
        """Build dynamic parameterized WHERE clause and parameter list from SessionQueryFilter."""
        if filter_criteria is None:
            return "", []

        conditions: list[str] = []
        params: list[Any] = []

        if filter_criteria.person_id is not None:
            validate_uuid(filter_criteria.person_id, "person_id")
            conditions.append("person_id = ?")
            params.append(filter_criteria.person_id)

        if filter_criteria.is_identified is not None:
            conditions.append("identity_was_known = ?")
            params.append(1 if filter_criteria.is_identified else 0)

        if filter_criteria.final_state is not None:
            state_str = (
                filter_criteria.final_state.value
                if hasattr(filter_criteria.final_state, "value")
                else str(filter_criteria.final_state)
            )
            conditions.append("final_state = ?")
            params.append(state_str)

        if filter_criteria.started_after is not None:
            conditions.append("started_at >= ?")
            params.append(filter_criteria.started_after)

        if filter_criteria.started_before is not None:
            conditions.append("started_at <= ?")
            params.append(filter_criteria.started_before)

        if filter_criteria.ended_after is not None:
            conditions.append("ended_at >= ?")
            params.append(filter_criteria.ended_after)

        if filter_criteria.ended_before is not None:
            conditions.append("ended_at <= ?")
            params.append(filter_criteria.ended_before)

        if filter_criteria.min_duration_seconds is not None:
            conditions.append("duration_seconds >= ?")
            params.append(filter_criteria.min_duration_seconds)

        if filter_criteria.max_duration_seconds is not None:
            conditions.append("duration_seconds <= ?")
            params.append(filter_criteria.max_duration_seconds)

        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
        return where_clause, params

    def query_sessions(self, filter: SessionQueryFilter) -> list[SessionRecord]:
        """Query interaction sessions matching filter criteria with pagination."""
        where_clause, params = self._build_filter_clause(filter)
        sql = f"""
        SELECT
            session_id,
            person_id,
            display_name,
            started_at,
            ended_at,
            duration_seconds,
            final_state,
            identity_was_known,
            identity_transitions_count,
            created_at
        FROM interaction_sessions
        {where_clause}
        ORDER BY ended_at DESC, created_at DESC, session_id ASC
        LIMIT ? OFFSET ?;
        """
        exec_params = [*params, filter.limit, filter.offset]
        with self._db.readonly_cursor() as cur:
            cur.execute(sql, tuple(exec_params))
            rows = cur.fetchall()
            return [self._row_to_record(r) for r in rows]

    def count_query_sessions(self, filter: SessionQueryFilter) -> int:
        """Count total interaction sessions matching filter criteria."""
        where_clause, params = self._build_filter_clause(filter)
        sql = f"SELECT COUNT(*) FROM interaction_sessions{where_clause};"
        with self._db.readonly_cursor() as cur:
            cur.execute(sql, tuple(params))
            row = cur.fetchone()
            return int(row[0]) if row and row[0] is not None else 0

    def get_aggregate_analytics(
        self,
        filter: SessionQueryFilter | None = None,
    ) -> SessionAggregateAnalytics:
        """Compute aggregate statistics directly via SQL over filtered sessions."""
        where_clause, params = self._build_filter_clause(filter)
        sql = f"""
        SELECT
            COUNT(*) AS total_sessions,
            COALESCE(SUM(duration_seconds), 0.0) AS total_duration,
            COALESCE(AVG(duration_seconds), 0.0) AS avg_duration,
            COALESCE(MIN(duration_seconds), 0.0) AS min_duration,
            COALESCE(MAX(duration_seconds), 0.0) AS max_duration,
            COALESCE(
                SUM(CASE WHEN identity_was_known = 1 THEN 1 ELSE 0 END), 0
            ) AS identified_count,
            COALESCE(
                SUM(CASE WHEN identity_was_known = 0 THEN 1 ELSE 0 END), 0
            ) AS anonymous_count,
            MIN(started_at) AS earliest_started_at,
            MAX(ended_at) AS latest_ended_at
        FROM interaction_sessions
        {where_clause};
        """
        with self._db.readonly_cursor() as cur:
            cur.execute(sql, tuple(params))
            row = cur.fetchone()
            if row is None or row["total_sessions"] == 0:
                return SessionAggregateAnalytics(
                    total_sessions=0,
                    total_duration_seconds=0.0,
                    average_duration_seconds=0.0,
                    min_duration_seconds=0.0,
                    max_duration_seconds=0.0,
                    identified_sessions_count=0,
                    anonymous_sessions_count=0,
                    earliest_session_started_at=None,
                    latest_session_ended_at=None,
                )
            return SessionAggregateAnalytics(
                total_sessions=int(row["total_sessions"]),
                total_duration_seconds=float(row["total_duration"]),
                average_duration_seconds=float(row["avg_duration"]),
                min_duration_seconds=float(row["min_duration"]),
                max_duration_seconds=float(row["max_duration"]),
                identified_sessions_count=int(row["identified_count"]),
                anonymous_sessions_count=int(row["anonymous_count"]),
                earliest_session_started_at=float(row["earliest_started_at"])
                if row["earliest_started_at"] is not None
                else None,
                latest_session_ended_at=float(row["latest_ended_at"])
                if row["latest_ended_at"] is not None
                else None,
            )

    def get_person_engagement_stats(
        self,
        person_id: str,
    ) -> PersonEngagementStats | None:
        """Compute historical engagement metrics for a specific enrolled person."""
        validate_uuid(person_id, "person_id")

        with self._db.readonly_cursor() as cur:
            # A. Aggregate statistics across all sessions for this person
            cur.execute(
                """
                SELECT
                    COUNT(*) AS total_sessions,
                    COALESCE(SUM(duration_seconds), 0.0) AS total_duration,
                    COALESCE(AVG(duration_seconds), 0.0) AS avg_duration,
                    MIN(started_at) AS first_interaction_at,
                    MAX(ended_at) AS last_interaction_at
                FROM interaction_sessions
                WHERE person_id = ?;
                """,
                (person_id,),
            )
            stats_row = cur.fetchone()
            if stats_row is None or stats_row["total_sessions"] == 0:
                return None

            # B. Deterministic query for latest display_name from most recent session
            cur.execute(
                """
                SELECT display_name
                FROM interaction_sessions
                WHERE person_id = ?
                ORDER BY ended_at DESC, created_at DESC, session_id ASC
                LIMIT 1;
                """,
                (person_id,),
            )
            name_row = cur.fetchone()
            latest_display_name = (
                str(name_row["display_name"])
                if (name_row is not None and name_row["display_name"] is not None)
                else None
            )

            return PersonEngagementStats(
                person_id=person_id,
                display_name=latest_display_name,
                total_sessions=int(stats_row["total_sessions"]),
                total_duration_seconds=float(stats_row["total_duration"]),
                average_duration_seconds=float(stats_row["avg_duration"]),
                first_interaction_at=float(stats_row["first_interaction_at"])
                if stats_row["first_interaction_at"] is not None
                else None,
                last_interaction_at=float(stats_row["last_interaction_at"])
                if stats_row["last_interaction_at"] is not None
                else None,
            )

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> SessionRecord:
        """Map a SQLite row to an immutable SessionRecord."""
        return SessionRecord(
            session_id=str(row["session_id"]),
            person_id=str(row["person_id"]) if row["person_id"] is not None else None,
            display_name=str(row["display_name"]) if row["display_name"] is not None else None,
            started_at=float(row["started_at"]),
            ended_at=float(row["ended_at"]),
            duration_seconds=float(row["duration_seconds"]),
            final_state=str(row["final_state"]),
            identity_was_known=bool(row["identity_was_known"]),
            identity_transitions_count=int(row["identity_transitions_count"]),
            created_at=str(row["created_at"]),
        )


class SessionPersistenceWorker:
    """Dedicated background worker for bounded asynchronous session persistence.

    Offloads SQLite disk write I/O from the real-time frame processing and EventBus thread.

    Guarantees:
        - Non-blocking submission from frame/coordinator callbacks.
        - Bounded queue preventing unbounded memory accumulation.
        - Synchronous fallback if queue is saturated to guarantee zero dropped audit records.
        - Graceful flush and draining on shutdown before database connections close.
        - Captured, observable persistence errors and metrics.
    """

    def __init__(
        self,
        repository: BaseSessionRepository,
        max_queue_size: int = 128,
        flush_timeout: float = 5.0,
    ) -> None:
        self._repository = repository
        self._max_queue_size = max(1, max_queue_size)
        self._flush_timeout = flush_timeout
        self._queue: queue.Queue[SessionRecord | None] = queue.Queue(maxsize=self._max_queue_size)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.RLock()
        self._is_running = False
        self._error_count = 0
        self._last_error: str | None = None
        self._saved_count = 0

    @property
    def is_running(self) -> bool:
        """Whether the worker background thread is active."""
        with self._lock:
            return self._is_running and self._thread is not None and self._thread.is_alive()

    @property
    def queue_size(self) -> int:
        """Number of pending session records in queue."""
        return self._queue.qsize()

    @property
    def saved_count(self) -> int:
        """Total number of successfully persisted session records."""
        with self._lock:
            return self._saved_count

    @property
    def error_count(self) -> int:
        """Total number of persistence errors encountered."""
        with self._lock:
            return self._error_count

    @property
    def last_error(self) -> str | None:
        """Last captured persistence error message, if any."""
        with self._lock:
            return self._last_error

    @property
    def max_queue_size(self) -> int:
        """Configured maximum queue capacity."""
        return self._max_queue_size

    def start(self) -> None:
        """Start the background persistence worker thread."""
        with self._lock:
            if self._is_running and self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._is_running = True
            self._thread = threading.Thread(
                target=self._run,
                name="SessionPersistenceWorkerThread",
                daemon=True,
            )
            self._thread.start()

    def submit(self, session: SessionRecord | InteractionSummary) -> bool:
        """Submit a completed session record or summary for persistence.

        Non-blocking under normal operation. If queue is full, executes a bounded wait
        followed by a synchronous fallback to prevent silent loss of audit records.

        Args:
            session: SessionRecord or InteractionSummary to persist.

        Returns:
            True if enqueued or persisted successfully, False on persistence failure.
        """
        if isinstance(session, SessionRecord):
            record = session
        else:
            record = SessionRecord.from_summary(session)

        # If worker is not started or stopped, write directly
        with self._lock:
            if not self._is_running or self._stop_event.is_set():
                return self._write_record_direct(record)

        try:
            self._queue.put_nowait(record)
            return True
        except queue.Full:
            logger.warning(
                "SessionPersistenceWorker queue full (%d items); attempting bounded put",
                self._max_queue_size,
            )
            try:
                self._queue.put(record, timeout=0.1)
                return True
            except queue.Full:
                logger.error(
                    "SessionPersistenceWorker queue full after timeout; saving synchronously "
                    "for session %s to prevent record loss",
                    record.session_id,
                )
                return self._write_record_direct(record)

    def flush(self, timeout: float | None = None) -> bool:
        """Wait until all currently queued session records are written to storage.

        Args:
            timeout: Maximum seconds to wait. Defaults to configured flush_timeout.

        Returns:
            True if all records were flushed, False if timeout expired.
        """
        t_limit = timeout if timeout is not None else self._flush_timeout
        deadline = time.monotonic() + t_limit

        while self._queue.unfinished_tasks > 0:
            if time.monotonic() > deadline:
                return False
            time.sleep(0.005)
        return True

    def stop(self, timeout: float = 3.0) -> None:
        """Gracefully drain pending queue items, stop the worker thread, and release resources.

        Args:
            timeout: Maximum seconds to wait for worker thread to terminate.
        """
        with self._lock:
            if not self._is_running:
                return
            self._stop_event.set()

            # Signal worker thread to terminate
            with contextlib.suppress(queue.Full):
                self._queue.put_nowait(None)

            if self._thread is not None and self._thread.is_alive():
                self._thread.join(timeout=timeout)

            # Drain any remaining unwritten items to guarantee zero lost records
            while not self._queue.empty():
                try:
                    item = self._queue.get_nowait()
                    if item is not None:
                        self._write_record_direct(item)
                    self._queue.task_done()
                except queue.Empty:
                    break

            self._is_running = False

    def _write_record_direct(self, record: SessionRecord) -> bool:
        """Directly write record to repository and track metrics."""
        try:
            self._repository.save_session(record)
            with self._lock:
                self._saved_count += 1
            return True
        except Exception as exc:
            logger.exception(
                "Failed to persist session record %s directly: %s",
                record.session_id,
                exc,
            )
            with self._lock:
                self._error_count += 1
                self._last_error = str(exc)
            return False

    def _run(self) -> None:
        """Main worker loop continuously processing queued session records."""
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if item is None:
                self._queue.task_done()
                break

            try:
                self._repository.save_session(item)
                with self._lock:
                    self._saved_count += 1
            except Exception as exc:
                logger.exception("Failed to persist session %s: %s", item.session_id, exc)
                with self._lock:
                    self._error_count += 1
                    self._last_error = str(exc)
            finally:
                self._queue.task_done()

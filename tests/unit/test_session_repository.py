"""Unit tests for SQLiteSessionRepository, SessionRecord model, and SessionPersistenceWorker."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from dh360_hi.backend.database import DatabaseManager
from dh360_hi.backend.exceptions import (
    ForeignKeyViolationError,
    ValidationError,
)
from dh360_hi.backend.models import (
    ConsentStatus,
    PersonRecord,
    PersonStatus,
    SessionRecord,
)
from dh360_hi.backend.repository import SQLitePersonRepository
from dh360_hi.backend.session_repository import (
    BaseSessionRepository,
    SessionPersistenceWorker,
    SQLiteSessionRepository,
)
from dh360_hi.contracts.interaction import InteractionState, InteractionSummary

if TYPE_CHECKING:
    pass


def _mk_person(repo: SQLitePersonRepository, pid: str, name: str = "Alice Smith") -> PersonRecord:
    """Helper to insert a test person into the database."""
    return repo.create_person(
        PersonRecord(
            person_id=pid,
            display_name=name,
            status=PersonStatus.ACTIVE,
            consent_status=ConsentStatus.GRANTED,
        )
    )


class TestSessionRecordModel:
    def test_valid_session_record_instantiation(self) -> None:
        """Verify valid SessionRecord creation with proper defaults."""
        sid = "11111111-1111-1111-1111-111111111111"
        rec = SessionRecord(
            session_id=sid,
            person_id=None,
            display_name=None,
            started_at=100.0,
            ended_at=110.0,
            duration_seconds=10.0,
            final_state="engaged",
            identity_was_known=False,
            identity_transitions_count=0,
        )
        assert rec.session_id == sid
        assert rec.person_id is None
        assert rec.duration_seconds == 10.0
        assert rec.created_at is not None

    def test_from_summary_and_to_summary_roundtrip(self) -> None:
        """Verify bidirectional conversion between InteractionSummary and SessionRecord."""
        sid = "22222222-2222-2222-2222-222222222222"
        pid = "33333333-3333-3333-3333-333333333333"
        summary = InteractionSummary(
            session_id=sid,
            person_id=pid,
            display_name="Dr. Jane",
            person_profile=None,
            started_at=100.0,
            ended_at=105.5,
            duration_seconds=5.5,
            final_state=InteractionState.FAREWELL,
            identity_was_known=True,
            identity_transitions_count=2,
        )

        record = SessionRecord.from_summary(summary)
        assert record.session_id == sid
        assert record.person_id == pid
        assert record.display_name == "Dr. Jane"
        assert record.duration_seconds == 5.5
        assert record.final_state == "farewell"
        assert record.identity_was_known is True
        assert record.identity_transitions_count == 2

        roundtrip = record.to_summary()
        assert roundtrip.session_id == summary.session_id
        assert roundtrip.person_id == summary.person_id
        assert roundtrip.display_name == summary.display_name
        assert roundtrip.duration_seconds == summary.duration_seconds
        assert roundtrip.final_state == summary.final_state
        assert roundtrip.identity_was_known == summary.identity_was_known

    def test_validation_errors_on_invalid_data(self) -> None:
        """Verify validation errors for bad UUIDs, negative duration, or empty state."""
        sid = "44444444-4444-4444-4444-444444444444"

        # Invalid session UUID
        with pytest.raises(ValidationError, match="Invalid UUID format for session_id"):
            SessionRecord(
                session_id="not-a-uuid",
                person_id=None,
                display_name=None,
                started_at=0.0,
                ended_at=1.0,
                duration_seconds=1.0,
                final_state="engaged",
                identity_was_known=False,
                identity_transitions_count=0,
            )

        # Invalid person UUID
        with pytest.raises(ValidationError, match="Invalid UUID format for person_id"):
            SessionRecord(
                session_id=sid,
                person_id="bad-uuid",
                display_name=None,
                started_at=0.0,
                ended_at=1.0,
                duration_seconds=1.0,
                final_state="engaged",
                identity_was_known=False,
                identity_transitions_count=0,
            )

        # Negative duration
        with pytest.raises(ValidationError, match="duration_seconds cannot be negative"):
            SessionRecord(
                session_id=sid,
                person_id=None,
                display_name=None,
                started_at=0.0,
                ended_at=1.0,
                duration_seconds=-5.0,
                final_state="engaged",
                identity_was_known=False,
                identity_transitions_count=0,
            )

        # Negative transitions count
        with pytest.raises(ValidationError, match="identity_transitions_count cannot be negative"):
            SessionRecord(
                session_id=sid,
                person_id=None,
                display_name=None,
                started_at=0.0,
                ended_at=1.0,
                duration_seconds=1.0,
                final_state="engaged",
                identity_was_known=False,
                identity_transitions_count=-1,
            )

        # Empty state
        with pytest.raises(ValidationError, match="final_state cannot be empty"):
            SessionRecord(
                session_id=sid,
                person_id=None,
                display_name=None,
                started_at=0.0,
                ended_at=1.0,
                duration_seconds=1.0,
                final_state="   ",
                identity_was_known=False,
                identity_transitions_count=0,
            )


class TestSQLiteSessionRepositoryCRUD:
    def test_save_and_get_anonymous_session(self) -> None:
        """Verify saving and retrieving an anonymous interaction session."""
        db = DatabaseManager(":memory:")
        try:
            repo = SQLiteSessionRepository(db)
            sid = "aaaaaaaa-1111-1111-1111-111111111111"
            rec = SessionRecord(
                session_id=sid,
                person_id=None,
                display_name=None,
                started_at=1000.0,
                ended_at=1015.0,
                duration_seconds=15.0,
                final_state="idle",
                identity_was_known=False,
                identity_transitions_count=0,
            )
            saved = repo.save_session(rec)
            assert saved.session_id == sid

            queried = repo.get_session(sid)
            assert queried is not None
            assert queried.session_id == sid
            assert queried.person_id is None
            assert queried.display_name is None
            assert queried.started_at == 1000.0
            assert queried.ended_at == 1015.0
            assert queried.duration_seconds == 15.0
            assert queried.final_state == "idle"
            assert queried.identity_was_known is False
            assert queried.identity_transitions_count == 0
        finally:
            db.close()

    def test_save_and_get_identified_session(self) -> None:
        """Verify saving and retrieving an identified interaction session referencing a person."""
        db = DatabaseManager(":memory:")
        try:
            person_repo = SQLitePersonRepository(db)
            session_repo = SQLiteSessionRepository(db)

            pid = "bbbbbbbb-2222-2222-2222-222222222222"
            _mk_person(person_repo, pid, "Bob Jones")

            sid = "cccccccc-3333-3333-3333-333333333333"
            rec = SessionRecord(
                session_id=sid,
                person_id=pid,
                display_name="Bob Jones",
                started_at=2000.0,
                ended_at=2045.0,
                duration_seconds=45.0,
                final_state="farewell",
                identity_was_known=True,
                identity_transitions_count=1,
            )
            session_repo.save_session(rec)

            queried = session_repo.get_session(sid)
            assert queried is not None
            assert queried.session_id == sid
            assert queried.person_id == pid
            assert queried.display_name == "Bob Jones"
            assert queried.identity_was_known is True
        finally:
            db.close()

    def test_save_duplicate_session_id_is_idempotent(self) -> None:
        """Verify that saving an existing session ID updates the record cleanly."""
        db = DatabaseManager(":memory:")
        try:
            repo = SQLiteSessionRepository(db)
            sid = "dddddddd-4444-4444-4444-444444444444"

            rec1 = SessionRecord(
                session_id=sid,
                person_id=None,
                display_name=None,
                started_at=100.0,
                ended_at=110.0,
                duration_seconds=10.0,
                final_state="engaged",
                identity_was_known=False,
                identity_transitions_count=0,
            )
            repo.save_session(rec1)

            # Update duration and state
            rec2 = SessionRecord(
                session_id=sid,
                person_id=None,
                display_name=None,
                started_at=100.0,
                ended_at=125.0,
                duration_seconds=25.0,
                final_state="farewell",
                identity_was_known=False,
                identity_transitions_count=0,
            )
            repo.save_session(rec2)

            queried = repo.get_session(sid)
            assert queried is not None
            assert queried.duration_seconds == 25.0
            assert queried.final_state == "farewell"
            assert repo.count_sessions() == 1
        finally:
            db.close()

    def test_foreign_key_violation_for_unknown_person_id(self) -> None:
        """Verify that referencing an un-enrolled person_id raises ForeignKeyViolationError."""
        db = DatabaseManager(":memory:")
        try:
            repo = SQLiteSessionRepository(db)
            sid = "eeeeeeee-5555-5555-5555-555555555555"
            non_existent_pid = "ffffffff-9999-9999-9999-999999999999"

            rec = SessionRecord(
                session_id=sid,
                person_id=non_existent_pid,
                display_name="Unknown",
                started_at=10.0,
                ended_at=20.0,
                duration_seconds=10.0,
                final_state="idle",
                identity_was_known=True,
                identity_transitions_count=1,
            )
            with pytest.raises(ForeignKeyViolationError, match="Foreign key violation"):
                repo.save_session(rec)
        finally:
            db.close()

    def test_person_deletion_sets_session_person_id_to_null(self) -> None:
        """Verify that deleting a person cascades ON DELETE SET NULL to interaction_sessions."""
        db = DatabaseManager(":memory:")
        try:
            person_repo = SQLitePersonRepository(db)
            session_repo = SQLiteSessionRepository(db)

            pid = "12345678-1234-1234-1234-123456789abc"
            _mk_person(person_repo, pid, "Dr. Delete Me")

            sid = "87654321-4321-4321-4321-cba987654321"
            session_repo.save_session(
                SessionRecord(
                    session_id=sid,
                    person_id=pid,
                    display_name="Dr. Delete Me",
                    started_at=100.0,
                    ended_at=120.0,
                    duration_seconds=20.0,
                    final_state="farewell",
                    identity_was_known=True,
                    identity_transitions_count=1,
                )
            )

            # Delete the person record
            assert person_repo.delete_person(pid) is True

            # Session still exists, but person_id is now NULL (retains display_name audit)
            queried = session_repo.get_session(sid)
            assert queried is not None
            assert queried.session_id == sid
            assert queried.person_id is None
            assert queried.display_name == "Dr. Delete Me"
        finally:
            db.close()

    def test_delete_session(self) -> None:
        """Verify deleting a session record by ID."""
        db = DatabaseManager(":memory:")
        try:
            repo = SQLiteSessionRepository(db)
            sid = "11223344-5566-7788-9900-aabbccddeeff"
            repo.save_session(
                SessionRecord(
                    session_id=sid,
                    person_id=None,
                    display_name=None,
                    started_at=1.0,
                    ended_at=2.0,
                    duration_seconds=1.0,
                    final_state="idle",
                    identity_was_known=False,
                    identity_transitions_count=0,
                )
            )
            assert repo.delete_session(sid) is True
            assert repo.get_session(sid) is None
            assert repo.delete_session(sid) is False
        finally:
            db.close()


class TestSQLiteSessionRepositoryQueries:
    def test_list_and_count_sessions_with_pagination_and_ordering(self) -> None:
        """Verify chronological sorting and limit/offset pagination."""
        db = DatabaseManager(":memory:")
        try:
            person_repo = SQLitePersonRepository(db)
            session_repo = SQLiteSessionRepository(db)

            pid1 = "11111111-0000-0000-0000-000000000001"
            pid2 = "22222222-0000-0000-0000-000000000002"
            _mk_person(person_repo, pid1, "Person One")
            _mk_person(person_repo, pid2, "Person Two")

            # Create 5 sessions with distinct timestamps
            for i in range(5):
                sid = f"00000000-0000-0000-0000-{i:012d}"
                pid = pid1 if i % 2 == 0 else pid2
                session_repo.save_session(
                    SessionRecord(
                        session_id=sid,
                        person_id=pid,
                        display_name=f"Person {i}",
                        started_at=float(i * 100),
                        ended_at=float(i * 100 + 50),
                        duration_seconds=50.0,
                        final_state="engaged",
                        identity_was_known=True,
                        identity_transitions_count=1,
                    )
                )

            assert session_repo.count_sessions() == 5
            assert session_repo.count_sessions(person_id=pid1) == 3
            assert session_repo.count_sessions(person_id=pid2) == 2

            # List all: newest ended_at first (session 4, 3, 2, 1, 0)
            all_sessions = session_repo.list_sessions(limit=10, offset=0)
            assert len(all_sessions) == 5
            assert [s.session_id for s in all_sessions] == [
                "00000000-0000-0000-0000-000000000004",
                "00000000-0000-0000-0000-000000000003",
                "00000000-0000-0000-0000-000000000002",
                "00000000-0000-0000-0000-000000000001",
                "00000000-0000-0000-0000-000000000000",
            ]

            # Pagination: limit 2, offset 2 -> session 2, 1
            page2 = session_repo.list_sessions(limit=2, offset=2)
            assert len(page2) == 2
            assert page2[0].session_id == "00000000-0000-0000-0000-000000000002"
            assert page2[1].session_id == "00000000-0000-0000-0000-000000000001"

            # Filter by person_id=pid2 (sessions 3 and 1)
            p2_sessions = session_repo.list_sessions(person_id=pid2, limit=10, offset=0)
            assert len(p2_sessions) == 2
            assert p2_sessions[0].session_id == "00000000-0000-0000-0000-000000000003"
            assert p2_sessions[1].session_id == "00000000-0000-0000-0000-000000000001"
        finally:
            db.close()

    def test_invalid_pagination_parameters_raise_validation_error(self) -> None:
        """Verify limit <= 0 or offset < 0 raises ValidationError."""
        db = DatabaseManager(":memory:")
        try:
            repo = SQLiteSessionRepository(db)
            with pytest.raises(ValidationError, match="limit must be greater than 0"):
                repo.list_sessions(limit=0)

            with pytest.raises(ValidationError, match="offset cannot be negative"):
                repo.list_sessions(offset=-1)
        finally:
            db.close()


class TestSessionPersistenceWorker:
    def test_worker_startup_submit_flush_and_stop(self) -> None:
        """Verify normal asynchronous session queueing, background write, flush, and stop."""
        repo = MagicMock(spec=BaseSessionRepository)
        worker = SessionPersistenceWorker(repository=repo, max_queue_size=16)

        assert worker.is_running is False
        worker.start()
        assert worker.is_running is True

        sid = "99999999-9999-9999-9999-999999999999"
        rec = SessionRecord(
            session_id=sid,
            person_id=None,
            display_name=None,
            started_at=10.0,
            ended_at=15.0,
            duration_seconds=5.0,
            final_state="engaged",
            identity_was_known=False,
            identity_transitions_count=0,
        )

        assert worker.submit(rec) is True
        assert worker.flush(timeout=2.0) is True

        repo.save_session.assert_called_once_with(rec)
        assert worker.saved_count == 1
        assert worker.error_count == 0

        worker.stop(timeout=1.0)
        assert worker.is_running is False

    def test_worker_gracefully_drains_remaining_queue_on_stop(self) -> None:
        """Verify that calling stop drains all pending queue items even if un-flushed."""
        db = DatabaseManager(":memory:")
        try:
            repo = SQLiteSessionRepository(db)
            worker = SessionPersistenceWorker(repository=repo, max_queue_size=64)
            worker.start()

            for i in range(10):
                sid = f"aaaaaaaa-0000-0000-0000-{i:012d}"
                worker.submit(
                    SessionRecord(
                        session_id=sid,
                        person_id=None,
                        display_name=None,
                        started_at=float(i),
                        ended_at=float(i + 1),
                        duration_seconds=1.0,
                        final_state="idle",
                        identity_was_known=False,
                        identity_transitions_count=0,
                    )
                )

            # Immediately stop without manual flush
            worker.stop(timeout=2.0)
            assert worker.is_running is False
            assert repo.count_sessions() == 10
        finally:
            db.close()

    def test_worker_captures_and_observes_persistence_error(self) -> None:
        """Verify worker records errors without raising uncaught exceptions in background thread."""
        repo = MagicMock(spec=BaseSessionRepository)
        repo.save_session.side_effect = RuntimeError("Disk full simulation")

        worker = SessionPersistenceWorker(repository=repo, max_queue_size=8)
        worker.start()

        sid = "88888888-8888-8888-8888-888888888888"
        rec = SessionRecord(
            session_id=sid,
            person_id=None,
            display_name=None,
            started_at=0.0,
            ended_at=1.0,
            duration_seconds=1.0,
            final_state="idle",
            identity_was_known=False,
            identity_transitions_count=0,
        )

        worker.submit(rec)
        worker.flush(timeout=1.0)

        assert worker.error_count == 1
        assert "Disk full simulation" in str(worker.last_error)

        worker.stop(timeout=1.0)

"""Unit tests for session query filtering, aggregate analytics, and person engagement statistics."""

from __future__ import annotations

import pytest

from dh360_hi.backend.database import DatabaseManager
from dh360_hi.backend.exceptions import ValidationError
from dh360_hi.backend.models import (
    ConsentStatus,
    PersonEngagementStats,
    PersonRecord,
    PersonStatus,
    SessionAggregateAnalytics,
    SessionQueryFilter,
    SessionRecord,
)
from dh360_hi.backend.repository import SQLitePersonRepository
from dh360_hi.backend.session_repository import SQLiteSessionRepository
from dh360_hi.contracts.interaction import InteractionState


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


def _seed_sessions(
    session_repo: SQLiteSessionRepository,
    person_repo: SQLitePersonRepository | None = None,
) -> dict[str, str]:
    """Helper to populate a standard set of interaction sessions for testing queries."""
    pid1 = "11111111-1111-1111-1111-111111111111"
    pid2 = "22222222-2222-2222-2222-222222222222"

    if person_repo is not None:
        _mk_person(person_repo, pid1, "Dr. Alice")
        _mk_person(person_repo, pid2, "Bob Jones")

    # Session 1: Alice, 10s, engaged, identified, started=100.0, ended=110.0
    session_repo.save_session(
        SessionRecord(
            session_id="00000000-0000-0000-0000-000000000001",
            person_id=pid1 if person_repo else None,
            display_name="Dr. Alice" if person_repo else None,
            started_at=100.0,
            ended_at=110.0,
            duration_seconds=10.0,
            final_state="engaged",
            identity_was_known=True,
            identity_transitions_count=1,
            created_at="2026-10-01T10:00:00Z",
        )
    )

    # Session 2: Alice, 20s, farewell, identified, started=200.0, ended=220.0
    session_repo.save_session(
        SessionRecord(
            session_id="00000000-0000-0000-0000-000000000002",
            person_id=pid1 if person_repo else None,
            display_name="Dr. Alice" if person_repo else None,
            started_at=200.0,
            ended_at=220.0,
            duration_seconds=20.0,
            final_state="farewell",
            identity_was_known=True,
            identity_transitions_count=2,
            created_at="2026-10-01T11:00:00Z",
        )
    )

    # Session 3: Bob, 30s, farewell, identified, started=300.0, ended=330.0
    session_repo.save_session(
        SessionRecord(
            session_id="00000000-0000-0000-0000-000000000003",
            person_id=pid2 if person_repo else None,
            display_name="Bob Jones" if person_repo else None,
            started_at=300.0,
            ended_at=330.0,
            duration_seconds=30.0,
            final_state="farewell",
            identity_was_known=True,
            identity_transitions_count=1,
            created_at="2026-10-01T12:00:00Z",
        )
    )

    # Session 4: Anonymous, 5s, idle, unknown, started=400.0, ended=405.0
    session_repo.save_session(
        SessionRecord(
            session_id="00000000-0000-0000-0000-000000000004",
            person_id=None,
            display_name=None,
            started_at=400.0,
            ended_at=405.0,
            duration_seconds=5.0,
            final_state="idle",
            identity_was_known=False,
            identity_transitions_count=0,
            created_at="2026-10-01T13:00:00Z",
        )
    )

    # Session 5: Anonymous, 15s, engaged, unknown, started=500.0, ended=515.0
    session_repo.save_session(
        SessionRecord(
            session_id="00000000-0000-0000-0000-000000000005",
            person_id=None,
            display_name=None,
            started_at=500.0,
            ended_at=515.0,
            duration_seconds=15.0,
            final_state="engaged",
            identity_was_known=False,
            identity_transitions_count=0,
            created_at="2026-10-01T14:00:00Z",
        )
    )

    return {"alice": pid1, "bob": pid2}


class TestSessionQueryFilterValidation:
    def test_valid_filter_defaults(self) -> None:
        """Verify default filter initialization."""
        flt = SessionQueryFilter()
        assert flt.person_id is None
        assert flt.is_identified is None
        assert flt.final_state is None
        assert flt.started_after is None
        assert flt.ended_before is None
        assert flt.min_duration_seconds is None
        assert flt.max_duration_seconds is None
        assert flt.limit == 50
        assert flt.offset == 0

    def test_invalid_limit(self) -> None:
        """Verify limit <= 0 raises ValidationError."""
        with pytest.raises(ValidationError, match="limit must be greater than 0"):
            SessionQueryFilter(limit=0)
        with pytest.raises(ValidationError, match="limit must be greater than 0"):
            SessionQueryFilter(limit=-10)

    def test_invalid_offset(self) -> None:
        """Verify offset < 0 raises ValidationError."""
        with pytest.raises(ValidationError, match="offset cannot be negative"):
            SessionQueryFilter(offset=-1)

    def test_invalid_person_id(self) -> None:
        """Verify malformed person_id UUID raises ValidationError."""
        with pytest.raises(ValidationError, match="Invalid UUID format for person_id"):
            SessionQueryFilter(person_id="not-a-valid-uuid")

    def test_reversed_time_range(self) -> None:
        """Verify started_after > ended_before raises ValidationError."""
        with pytest.raises(
            ValidationError, match="started_after .* cannot be greater than ended_before"
        ):
            SessionQueryFilter(started_after=500.0, ended_before=400.0)

    def test_negative_minimum_duration(self) -> None:
        """Verify min_duration_seconds < 0 raises ValidationError."""
        with pytest.raises(ValidationError, match="min_duration_seconds cannot be negative"):
            SessionQueryFilter(min_duration_seconds=-1.0)

    def test_negative_maximum_duration(self) -> None:
        """Verify max_duration_seconds < 0 raises ValidationError."""
        with pytest.raises(ValidationError, match="max_duration_seconds cannot be negative"):
            SessionQueryFilter(max_duration_seconds=-5.0)

    def test_reversed_duration_range(self) -> None:
        """Verify min_duration_seconds > max_duration_seconds raises ValidationError."""
        with pytest.raises(
            ValidationError,
            match="min_duration_seconds .* cannot be greater than max_duration_seconds",
        ):
            SessionQueryFilter(min_duration_seconds=30.0, max_duration_seconds=10.0)

    def test_invalid_final_state(self) -> None:
        """Verify invalid final_state string or type raises ValidationError."""
        with pytest.raises(ValidationError, match="Invalid final_state"):
            SessionQueryFilter(final_state="unknown_state")  # type: ignore[arg-type]
        with pytest.raises(ValidationError, match="Invalid final_state"):
            SessionQueryFilter(final_state=123)  # type: ignore[arg-type]

    def test_valid_final_state_enum_and_string(self) -> None:
        """Verify valid final_state enum and string values initialize correctly."""
        flt_enum = SessionQueryFilter(final_state=InteractionState.ENGAGED)
        assert flt_enum.final_state == InteractionState.ENGAGED

        flt_str = SessionQueryFilter(final_state="farewell")  # type: ignore[arg-type]
        assert flt_str.final_state == InteractionState.FAREWELL


class TestSessionQueryFiltering:
    def test_query_all_with_default_filter(self) -> None:
        """Verify querying with default filter returns all records ordered descending."""
        db = DatabaseManager(":memory:")
        try:
            p_repo = SQLitePersonRepository(db)
            s_repo = SQLiteSessionRepository(db)
            _seed_sessions(s_repo, p_repo)

            flt = SessionQueryFilter()
            sessions = s_repo.query_sessions(flt)
            assert len(sessions) == 5
            # Expected order by ended_at DESC: session 5 (515), 4 (405), 3 (330), 2 (220), 1 (110)
            assert [s.session_id[-1] for s in sessions] == ["5", "4", "3", "2", "1"]
            assert s_repo.count_query_sessions(flt) == 5
        finally:
            db.close()

    def test_filter_by_person_id(self) -> None:
        """Verify querying by specific person_id."""
        db = DatabaseManager(":memory:")
        try:
            p_repo = SQLitePersonRepository(db)
            s_repo = SQLiteSessionRepository(db)
            ids = _seed_sessions(s_repo, p_repo)

            flt = SessionQueryFilter(person_id=ids["alice"])
            sessions = s_repo.query_sessions(flt)
            assert len(sessions) == 2
            assert all(s.person_id == ids["alice"] for s in sessions)
            assert s_repo.count_query_sessions(flt) == 2
        finally:
            db.close()

    def test_filter_by_identity_status(self) -> None:
        """Verify filtering for identified vs anonymous sessions."""
        db = DatabaseManager(":memory:")
        try:
            p_repo = SQLitePersonRepository(db)
            s_repo = SQLiteSessionRepository(db)
            _seed_sessions(s_repo, p_repo)

            # Identified only
            flt_identified = SessionQueryFilter(is_identified=True)
            identified = s_repo.query_sessions(flt_identified)
            assert len(identified) == 3
            assert all(s.identity_was_known is True for s in identified)
            assert s_repo.count_query_sessions(flt_identified) == 3

            # Anonymous only
            flt_anon = SessionQueryFilter(is_identified=False)
            anon = s_repo.query_sessions(flt_anon)
            assert len(anon) == 2
            assert all(s.identity_was_known is False for s in anon)
            assert s_repo.count_query_sessions(flt_anon) == 2
        finally:
            db.close()

    def test_filter_by_final_state(self) -> None:
        """Verify filtering by final_state (enum or string)."""
        db = DatabaseManager(":memory:")
        try:
            p_repo = SQLitePersonRepository(db)
            s_repo = SQLiteSessionRepository(db)
            _seed_sessions(s_repo, p_repo)

            # Enum filter
            flt_farewell = SessionQueryFilter(final_state=InteractionState.FAREWELL)
            farewell_sessions = s_repo.query_sessions(flt_farewell)
            assert len(farewell_sessions) == 2
            assert all(s.final_state == "farewell" for s in farewell_sessions)

            # String filter
            flt_engaged = SessionQueryFilter(final_state="engaged")
            engaged_sessions = s_repo.query_sessions(flt_engaged)
            assert len(engaged_sessions) == 2
            assert all(s.final_state == "engaged" for s in engaged_sessions)

            # None filter (preserves all states)
            flt_none = SessionQueryFilter(final_state=None)
            all_sessions = s_repo.query_sessions(flt_none)
            assert len(all_sessions) == 5
        finally:
            db.close()

    def test_combined_filter_with_final_state(self) -> None:
        """Verify combining final_state with identity and duration filters."""
        db = DatabaseManager(":memory:")
        try:
            p_repo = SQLitePersonRepository(db)
            s_repo = SQLiteSessionRepository(db)
            _seed_sessions(s_repo, p_repo)

            # Farewell + Identified + duration >= 25.0 -> Matches Bob (session 3: 30s, farewell, id)
            flt = SessionQueryFilter(
                final_state=InteractionState.FAREWELL,
                is_identified=True,
                min_duration_seconds=25.0,
            )
            sessions = s_repo.query_sessions(flt)
            assert len(sessions) == 1
            assert sessions[0].session_id[-1] == "3"
            assert s_repo.count_query_sessions(flt) == 1
        finally:
            db.close()

    def test_query_sessions_deterministic_ordering(self) -> None:
        """Verify query_sessions strictly obeys ended_at DESC, created_at DESC, session_id ASC."""
        db = DatabaseManager(":memory:")
        try:
            s_repo = SQLiteSessionRepository(db)

            # Insert 3 records with the SAME ended_at (100.0) but different created_at / session_id
            s1 = SessionRecord(
                session_id="00000000-0000-0000-0000-00000000000b",
                person_id=None,
                display_name=None,
                started_at=90.0,
                ended_at=100.0,
                duration_seconds=10.0,
                final_state="idle",
                identity_was_known=False,
                identity_transitions_count=0,
                created_at="2026-10-01T10:00:00Z",
            )
            s2 = SessionRecord(
                session_id="00000000-0000-0000-0000-00000000000a",
                person_id=None,
                display_name=None,
                started_at=90.0,
                ended_at=100.0,
                duration_seconds=10.0,
                final_state="idle",
                identity_was_known=False,
                identity_transitions_count=0,
                created_at="2026-10-01T12:00:00Z",  # newer created_at -> should come first
            )
            s3 = SessionRecord(
                session_id="00000000-0000-0000-0000-00000000000c",
                person_id=None,
                display_name=None,
                started_at=90.0,
                ended_at=100.0,
                duration_seconds=10.0,
                final_state="idle",
                identity_was_known=False,
                identity_transitions_count=0,
                # same created_at as s1 -> tiebreak by session_id ASC
                # ('...000b' vs '...000c')
                created_at="2026-10-01T10:00:00Z",
            )
            s_repo.save_session(s1)
            s_repo.save_session(s2)
            s_repo.save_session(s3)

            results = s_repo.query_sessions(SessionQueryFilter())
            assert [r.session_id[-1] for r in results] == ["a", "b", "c"]
        finally:
            db.close()

    def test_filter_by_time_window(self) -> None:
        """Verify filtering by started_after and ended_before timestamps."""
        db = DatabaseManager(":memory:")
        try:
            p_repo = SQLitePersonRepository(db)
            s_repo = SQLiteSessionRepository(db)
            _seed_sessions(s_repo, p_repo)

            # Range covering sessions 2 and 3 (ended 220.0 and 330.0)
            flt = SessionQueryFilter(started_after=200.0, ended_before=350.0)
            sessions = s_repo.query_sessions(flt)
            assert len(sessions) == 2
            assert [s.session_id[-1] for s in sessions] == ["3", "2"]
            assert s_repo.count_query_sessions(flt) == 2
        finally:
            db.close()

    def test_filter_by_duration_range(self) -> None:
        """Verify filtering by min and max duration seconds."""
        db = DatabaseManager(":memory:")
        try:
            p_repo = SQLitePersonRepository(db)
            s_repo = SQLiteSessionRepository(db)
            _seed_sessions(s_repo, p_repo)

            # Durations are 5, 10, 15, 20, 30. Filter for >= 15 and <= 25 (sessions 5 and 2)
            flt = SessionQueryFilter(min_duration_seconds=15.0, max_duration_seconds=25.0)
            sessions = s_repo.query_sessions(flt)
            assert len(sessions) == 2
            assert [s.session_id[-1] for s in sessions] == ["5", "2"]
            assert s_repo.count_query_sessions(flt) == 2
        finally:
            db.close()

    def test_combined_filters_and_pagination(self) -> None:
        """Verify combining multiple filters together with limit and offset."""
        db = DatabaseManager(":memory:")
        try:
            p_repo = SQLitePersonRepository(db)
            s_repo = SQLiteSessionRepository(db)
            _seed_sessions(s_repo, p_repo)

            # Combined: is_identified=True, min_duration=10.0, limit=1, offset=1
            # Matches sessions 3, 2, 1. Page with limit 1, offset 1 is session 2.
            flt = SessionQueryFilter(
                is_identified=True,
                min_duration_seconds=10.0,
                limit=1,
                offset=1,
            )
            sessions = s_repo.query_sessions(flt)
            assert len(sessions) == 1
            assert sessions[0].session_id[-1] == "2"
            assert s_repo.count_query_sessions(flt) == 3
        finally:
            db.close()

    def test_query_empty_result_set(self) -> None:
        """Verify query returning 0 matches returns empty list without error."""
        db = DatabaseManager(":memory:")
        try:
            s_repo = SQLiteSessionRepository(db)
            flt = SessionQueryFilter(started_after=999999.0)
            assert s_repo.query_sessions(flt) == []
            assert s_repo.count_query_sessions(flt) == 0
        finally:
            db.close()


class TestAggregateAnalytics:
    def test_aggregate_analytics_empty_database(self) -> None:
        """Verify aggregate analytics on an empty database returns zeroed model safely."""
        db = DatabaseManager(":memory:")
        try:
            s_repo = SQLiteSessionRepository(db)
            stats = s_repo.get_aggregate_analytics()
            assert isinstance(stats, SessionAggregateAnalytics)
            assert stats.total_sessions == 0
            assert stats.total_duration_seconds == 0.0
            assert stats.average_duration_seconds == 0.0
            assert stats.min_duration_seconds == 0.0
            assert stats.max_duration_seconds == 0.0
            assert stats.identified_sessions_count == 0
            assert stats.anonymous_sessions_count == 0
            assert stats.earliest_session_started_at is None
            assert stats.latest_session_ended_at is None
        finally:
            db.close()

    def test_aggregate_analytics_multiple_sessions(self) -> None:
        """Verify aggregate calculations across full seed dataset."""
        db = DatabaseManager(":memory:")
        try:
            p_repo = SQLitePersonRepository(db)
            s_repo = SQLiteSessionRepository(db)
            _seed_sessions(s_repo, p_repo)

            # Durations: 10, 20, 30, 5, 15 -> total = 80.0, avg = 16.0, min = 5.0, max = 30.0
            # Identified = 3, Anonymous = 2; Earliest start = 100.0, Latest end = 515.0
            stats = s_repo.get_aggregate_analytics()
            assert stats.total_sessions == 5
            assert stats.total_duration_seconds == 80.0
            assert stats.average_duration_seconds == 16.0
            assert stats.min_duration_seconds == 5.0
            assert stats.max_duration_seconds == 30.0
            assert stats.identified_sessions_count == 3
            assert stats.anonymous_sessions_count == 2
            assert stats.earliest_session_started_at == 100.0
            assert stats.latest_session_ended_at == 515.0
        finally:
            db.close()

    def test_aggregate_analytics_filtered_subset(self) -> None:
        """Verify aggregate analytics scoped to a specific filter criteria."""
        db = DatabaseManager(":memory:")
        try:
            p_repo = SQLitePersonRepository(db)
            s_repo = SQLiteSessionRepository(db)
            _seed_sessions(s_repo, p_repo)

            # Filter for anonymous only (sessions 4 and 5: 5s and 15s)
            flt = SessionQueryFilter(is_identified=False)
            stats = s_repo.get_aggregate_analytics(filter=flt)
            assert stats.total_sessions == 2
            assert stats.total_duration_seconds == 20.0
            assert stats.average_duration_seconds == 10.0
            assert stats.min_duration_seconds == 5.0
            assert stats.max_duration_seconds == 15.0
            assert stats.identified_sessions_count == 0
            assert stats.anonymous_sessions_count == 2
            assert stats.earliest_session_started_at == 400.0
            assert stats.latest_session_ended_at == 515.0
        finally:
            db.close()


class TestPersonEngagementStatistics:
    def test_person_engagement_stats_known_person(self) -> None:
        """Verify person engagement stats for an enrolled person with multiple sessions."""
        db = DatabaseManager(":memory:")
        try:
            p_repo = SQLitePersonRepository(db)
            s_repo = SQLiteSessionRepository(db)
            ids = _seed_sessions(s_repo, p_repo)

            # Alice has session 1 (10s, [100.0-110.0]) and session 2 (20s, [200.0-220.0])
            stats = s_repo.get_person_engagement_stats(ids["alice"])
            assert stats is not None
            assert isinstance(stats, PersonEngagementStats)
            assert stats.person_id == ids["alice"]
            assert stats.display_name == "Dr. Alice"
            assert stats.total_sessions == 2
            assert stats.total_duration_seconds == 30.0
            assert stats.average_duration_seconds == 15.0
            assert stats.first_interaction_at == 100.0
            assert stats.last_interaction_at == 220.0
        finally:
            db.close()

    def test_person_engagement_stats_multiple_sessions_different_display_names(
        self,
    ) -> None:
        """Verify engagement stats calculate across ALL sessions and pick latest display_name."""
        db = DatabaseManager(":memory:")
        try:
            p_repo = SQLitePersonRepository(db)
            s_repo = SQLiteSessionRepository(db)
            pid = "33333333-3333-3333-3333-333333333333"
            _mk_person(p_repo, pid, "Alice Initial")

            # Session 1: display_name = "Alice 1", 10s, ended=100.0, created="2026-10-01T08:00:00Z"
            s_repo.save_session(
                SessionRecord(
                    session_id="00000000-0000-0000-0000-000000000011",
                    person_id=pid,
                    display_name="Alice 1",
                    started_at=90.0,
                    ended_at=100.0,
                    duration_seconds=10.0,
                    final_state="engaged",
                    identity_was_known=True,
                    identity_transitions_count=1,
                    created_at="2026-10-01T08:00:00Z",
                )
            )

            # Session 2: display_name = "Alice 2", 20s, ended=200.0, created="2026-10-01T09:00:00Z"
            s_repo.save_session(
                SessionRecord(
                    session_id="00000000-0000-0000-0000-000000000012",
                    person_id=pid,
                    display_name="Alice 2",
                    started_at=180.0,
                    ended_at=200.0,
                    duration_seconds=20.0,
                    final_state="farewell",
                    identity_was_known=True,
                    identity_transitions_count=1,
                    created_at="2026-10-01T09:00:00Z",
                )
            )

            # Session 3: display_name = "Dr. Alice Latest", 30s, ended=300.0
            # created="2026-10-01T10:00:00Z"
            s_repo.save_session(
                SessionRecord(
                    session_id="00000000-0000-0000-0000-000000000013",
                    person_id=pid,
                    display_name="Dr. Alice Latest",
                    started_at=270.0,
                    ended_at=300.0,
                    duration_seconds=30.0,
                    final_state="farewell",
                    identity_was_known=True,
                    identity_transitions_count=2,
                    created_at="2026-10-01T10:00:00Z",
                )
            )

            stats = s_repo.get_person_engagement_stats(pid)
            assert stats is not None
            assert stats.person_id == pid
            # From session 3 (latest ended_at=300.0)
            assert stats.display_name == "Dr. Alice Latest"
            assert stats.total_sessions == 3  # All 3 sessions included
            assert stats.total_duration_seconds == 60.0  # 10 + 20 + 30
            assert stats.average_duration_seconds == 20.0  # 60 / 3
            assert stats.first_interaction_at == 90.0  # Earliest started_at
            assert stats.last_interaction_at == 300.0  # Latest ended_at
        finally:
            db.close()

    def test_person_engagement_stats_deterministic_tiebreaking(self) -> None:
        """Verify latest display_name tiebreaking via created_at DESC and session_id ASC."""
        db = DatabaseManager(":memory:")
        try:
            p_repo = SQLitePersonRepository(db)
            s_repo = SQLiteSessionRepository(db)
            pid = "44444444-4444-4444-4444-444444444444"
            _mk_person(p_repo, pid, "User")

            # Two sessions with exact same ended_at (200.0)
            s_repo.save_session(
                SessionRecord(
                    session_id="00000000-0000-0000-0000-000000000021",
                    person_id=pid,
                    display_name="Older Creation",
                    started_at=190.0,
                    ended_at=200.0,
                    duration_seconds=10.0,
                    final_state="engaged",
                    identity_was_known=True,
                    identity_transitions_count=1,
                    created_at="2026-10-01T08:00:00Z",
                )
            )
            s_repo.save_session(
                SessionRecord(
                    session_id="00000000-0000-0000-0000-000000000022",
                    person_id=pid,
                    display_name="Newer Creation",
                    started_at=190.0,
                    ended_at=200.0,
                    duration_seconds=10.0,
                    final_state="engaged",
                    identity_was_known=True,
                    identity_transitions_count=1,
                    created_at="2026-10-01T12:00:00Z",
                )
            )

            stats = s_repo.get_person_engagement_stats(pid)
            assert stats is not None
            assert stats.display_name == "Newer Creation"
            assert stats.total_sessions == 2
            assert stats.total_duration_seconds == 20.0
        finally:
            db.close()

    def test_person_engagement_stats_nonexistent_person(self) -> None:
        """Verify person engagement stats for person with 0 sessions returns None."""
        db = DatabaseManager(":memory:")
        try:
            s_repo = SQLiteSessionRepository(db)
            non_existent_pid = "99999999-9999-9999-9999-999999999999"
            assert s_repo.get_person_engagement_stats(non_existent_pid) is None
        finally:
            db.close()

    def test_deleted_person_historical_behavior(self) -> None:
        """Verify sessions survive person deletion with NULL person_id and display_name."""
        db = DatabaseManager(":memory:")
        try:
            p_repo = SQLitePersonRepository(db)
            s_repo = SQLiteSessionRepository(db)
            ids = _seed_sessions(s_repo, p_repo)
            alice_pid = ids["alice"]

            # Delete Alice from person repository
            assert p_repo.delete_person(alice_pid) is True

            # 1. Person-specific query by the old person_id now returns None
            assert s_repo.get_person_engagement_stats(alice_pid) is None
            assert s_repo.count_query_sessions(SessionQueryFilter(person_id=alice_pid)) == 0

            # 2. Global query shows all 5 sessions remain intact
            all_sessions = s_repo.query_sessions(SessionQueryFilter())
            assert len(all_sessions) == 5

            # 3. Alice's historical sessions have person_id=None but display_name preserved
            alice_historical = [s for s in all_sessions if s.display_name == "Dr. Alice"]
            assert len(alice_historical) == 2
            assert all(s.person_id is None for s in alice_historical)
            assert all(s.identity_was_known is True for s in alice_historical)

            # 4. Total aggregate stats still include Alice's duration (total 80.0s)
            global_analytics = s_repo.get_aggregate_analytics()
            assert global_analytics.total_sessions == 5
            assert global_analytics.total_duration_seconds == 80.0
        finally:
            db.close()


class TestPrivacyAndContractIntegrity:
    def test_no_biometric_or_vision_data_in_analytics_models(self) -> None:
        """Verify models contain zero image arrays, embeddings, landmarks, or bounding boxes."""
        analytics = SessionAggregateAnalytics(
            total_sessions=1,
            total_duration_seconds=10.0,
            average_duration_seconds=10.0,
            min_duration_seconds=10.0,
            max_duration_seconds=10.0,
            identified_sessions_count=1,
            anonymous_sessions_count=0,
            earliest_session_started_at=100.0,
            latest_session_ended_at=110.0,
        )
        stats = PersonEngagementStats(
            person_id="11111111-1111-1111-1111-111111111111",
            display_name="User",
            total_sessions=1,
            total_duration_seconds=10.0,
            average_duration_seconds=10.0,
            first_interaction_at=100.0,
            last_interaction_at=110.0,
        )

        for model in (analytics, stats):
            field_names = [f.lower() for f in model.__dataclass_fields__]
            assert "embedding" not in field_names
            assert "vector" not in field_names
            assert "image" not in field_names
            assert "frame" not in field_names
            assert "bbox" not in field_names
            assert "landmark" not in field_names
            assert "track_id" not in field_names

"""Integration tests for historical session query filtering, SQL aggregation, and analytics."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

from dh360_hi.application.config import HumanInteractionConfig
from dh360_hi.application.factory import create_interaction_service
from dh360_hi.backend.database import DatabaseManager
from dh360_hi.backend.models import (
    ConsentStatus,
    PersonStatus,
    SessionQueryFilter,
    SessionRecord,
)
from dh360_hi.backend.session_repository import SQLiteSessionRepository
from dh360_hi.contracts.interaction import InteractionState
from dh360_hi.edge.runtime import EdgeRuntimeConfig
from dh360_hi.edge.session_manager import SessionConfig
from dh360_hi.vision.identity_state import IdentityState, TrackIdentityState
from dh360_hi.vision.presence_debouncer import PresenceEvent, PresenceTransition
from dh360_hi.vision.target_selector import TargetCandidate, TargetSelectionResult
from dh360_hi.vision.two_rate_pipeline import TwoRateFrameResult, TwoRatePerceptionPipeline

if TYPE_CHECKING:
    from pathlib import Path


def _mk_target_result(
    primary_track_id: int | None = 1,
    person_id: str | None = None,
    display_name: str | None = None,
    frame_id: int = 1,
) -> TargetSelectionResult:
    cand = None
    if primary_track_id is not None:
        cand = TargetCandidate(
            track_id=primary_track_id,
            score=1.0,
            centrality_score=1.0,
            size_score=1.0,
            identity_state_score=1.0,
            dwell_score=1.0,
            is_identified=bool(person_id),
            identity_state=IdentityState.IDENTIFIED if person_id else IdentityState.COLLECTING,
            person_id=person_id,
            display_name=display_name,
            bbox=(100.0, 100.0, 200.0, 200.0),
            center=(150.0, 150.0),
        )
    return TargetSelectionResult(
        primary_target=cand,
        candidates=[cand] if cand else [],
        target_changed=False,
        previous_target_track_id=None,
        frame_id=frame_id,
    )


def _mk_scripted_frame_result(
    frame_id: int,
    person_id: str | None = None,
    display_name: str | None = None,
    track_id: int = 1,
    total_frames: int = 5,
    timestamp_offset: float = 0.0,
) -> TwoRateFrameResult:
    transitions: list[PresenceTransition] = []
    t_now = timestamp_offset + float(frame_id) * 0.033
    if frame_id == 1:
        transitions.append(
            PresenceTransition(
                event=PresenceEvent.PERSON_ENTERED,
                track_id=track_id,
                person_id=person_id,
                display_name=display_name,
                frame_id=frame_id,
                timestamp=t_now,
            )
        )
    elif frame_id == total_frames:
        transitions.append(
            PresenceTransition(
                event=PresenceEvent.PERSON_LEFT,
                track_id=track_id,
                person_id=person_id,
                display_name=display_name,
                frame_id=frame_id,
                timestamp=t_now,
            )
        )

    id_states: dict[int, TrackIdentityState] = {}
    if frame_id < total_frames:
        id_states[track_id] = TrackIdentityState(
            track_id=track_id,
            state=IdentityState.IDENTIFIED if person_id else IdentityState.COLLECTING,
            current_person_id=person_id,
            current_display_name=display_name,
        )

    primary_track = track_id if frame_id < total_frames else None
    return TwoRateFrameResult(
        frame_id=frame_id,
        timestamp=t_now,
        detections=[],
        tracks=[],
        quality_results={},
        identity_states=id_states,
        target_result=_mk_target_result(primary_track, person_id, display_name, frame_id),
        presence_transitions=transitions,
        completed_recognitions=[],
        ambiguous_crossing_track_ids=set(),
        pending_worker_tasks=0,
        frame_latency_ms=1.0,
        total_pipeline_latency_ms=1.0,
    )


class TestSessionAnalyticsIntegration:
    def test_multi_session_query_and_sql_aggregation(self) -> None:
        """Verify running consecutive interaction sessions and calculating aggregates."""
        db = DatabaseManager(":memory:")
        cfg = HumanInteractionConfig(
            runtime=EdgeRuntimeConfig(target_fps=1000.0),
            session=SessionConfig(require_confirmed_presence=True),
        )

        pipeline = MagicMock(spec=TwoRatePerceptionPipeline)
        pipeline.worker.is_running = True

        service = create_interaction_service(
            config=cfg,
            db_manager=db,
            pipeline=pipeline,
        )

        # 1. Enroll Person A and Person B
        profile_a = service.person_service.create_person(
            display_name="Dr. Alice",
            status=PersonStatus.ACTIVE,
            consent_status=ConsentStatus.GRANTED,
        )
        profile_b = service.person_service.create_person(
            display_name="Bob Jones",
            status=PersonStatus.ACTIVE,
            consent_status=ConsentStatus.GRANTED,
        )

        # Run Session 1: Alice (frames 1 to 5, t=100.0)
        for f_id in range(1, 6):
            pipeline.process_frame.return_value = _mk_scripted_frame_result(
                frame_id=f_id,
                person_id=profile_a.person_id,
                display_name=profile_a.display_name,
                total_frames=5,
                timestamp_offset=100.0,
            )
            service.step_once()

        # Run Session 2: Bob (frames 1 to 5, t=200.0)
        for f_id in range(1, 6):
            pipeline.process_frame.return_value = _mk_scripted_frame_result(
                frame_id=f_id,
                person_id=profile_b.person_id,
                display_name=profile_b.display_name,
                total_frames=5,
                timestamp_offset=200.0,
            )
            service.step_once()

        # Run Session 3: Anonymous (frames 1 to 5, t=300.0)
        for f_id in range(1, 6):
            pipeline.process_frame.return_value = _mk_scripted_frame_result(
                frame_id=f_id,
                person_id=None,
                display_name=None,
                total_frames=5,
                timestamp_offset=300.0,
            )
            service.step_once()

        # Flush worker
        assert service.flush_session_persistence(timeout=2.0) is True

        # 2. Query with SessionQueryFilter
        # Filter for identified only -> 2 sessions (Alice, Bob)
        flt_identified = SessionQueryFilter(is_identified=True)
        identified_sessions = service.query_sessions(flt_identified)
        assert len(identified_sessions) == 2
        assert service.count_query_sessions(flt_identified) == 2

        # Filter for Alice specifically
        flt_alice = SessionQueryFilter(person_id=profile_a.person_id)
        alice_sessions = service.query_sessions(flt_alice)
        assert len(alice_sessions) == 1
        assert alice_sessions[0].display_name == "Dr. Alice"

        # 3. Person Engagement Stats
        alice_stats = service.get_person_engagement_stats(profile_a.person_id)
        assert alice_stats is not None
        assert alice_stats.person_id == profile_a.person_id
        assert alice_stats.display_name == "Dr. Alice"
        assert alice_stats.total_sessions == 1
        assert alice_stats.total_duration_seconds > 0.0

        bob_stats = service.get_person_engagement_stats(profile_b.person_id)
        assert bob_stats is not None
        assert bob_stats.total_sessions == 1

        # 4. Global SQL Aggregate Analytics
        global_analytics = service.get_aggregate_analytics()
        assert global_analytics.total_sessions == 3
        assert global_analytics.identified_sessions_count == 2
        assert global_analytics.anonymous_sessions_count == 1
        assert global_analytics.total_duration_seconds > 0.0
        assert global_analytics.earliest_session_started_at is not None
        assert global_analytics.latest_session_ended_at is not None

        service.stop()

    def test_person_deletion_historical_retention_integration(self, tmp_path: Path) -> None:
        """Verify on-disk database retains session history when person is deleted, severing ID."""
        db_file = tmp_path / "deletion_history.db"
        cfg = HumanInteractionConfig(
            db_path=str(db_file),
            runtime=EdgeRuntimeConfig(target_fps=1000.0),
            session=SessionConfig(require_confirmed_presence=True),
        )

        pipeline = MagicMock(spec=TwoRatePerceptionPipeline)
        pipeline.worker.is_running = True

        service = create_interaction_service(config=cfg, pipeline=pipeline)

        # 1. Enroll user
        profile = service.person_service.create_person(
            display_name="Charlie Chaplin",
            status=PersonStatus.ACTIVE,
            consent_status=ConsentStatus.GRANTED,
        )
        pid = profile.person_id

        # 2. Run session
        for f_id in range(1, 6):
            pipeline.process_frame.return_value = _mk_scripted_frame_result(
                frame_id=f_id,
                person_id=pid,
                display_name="Charlie Chaplin",
                total_frames=5,
                timestamp_offset=100.0,
            )
            service.step_once()

        assert service.flush_session_persistence(timeout=2.0) is True

        # 3. Delete person
        assert service.person_service.delete_person(pid) is True

        # 4. Person engagement lookup by old ID returns None
        assert service.get_person_engagement_stats(pid) is None

        # 5. Global query and analytics still include the historical session record
        all_sessions = service.query_sessions()
        assert len(all_sessions) == 1
        record = all_sessions[0]
        assert record.person_id is None
        assert record.display_name == "Charlie Chaplin"
        assert record.identity_was_known is True

        analytics = service.get_aggregate_analytics()
        assert analytics.total_sessions == 1
        assert analytics.identified_sessions_count == 1

        service.stop()

    def test_large_history_sql_aggregate_efficiency(self) -> None:
        """Verify that aggregate queries over a 50-session history execute via SQL directly."""
        db = DatabaseManager(":memory:")
        repo = SQLiteSessionRepository(db)

        # Seed 50 sessions in SQLite directly
        for i in range(50):
            sid = f"aaaaaaaa-0000-0000-0000-{i:012d}"
            repo.save_session(
                SessionRecord(
                    session_id=sid,
                    person_id=None,
                    display_name=None,
                    started_at=float(i * 10),
                    ended_at=float(i * 10 + 5),
                    duration_seconds=5.0,
                    final_state=InteractionState.ENGAGED.value,
                    identity_was_known=i % 2 == 0,
                    identity_transitions_count=0,
                )
            )

        # Query total analytics via SQL
        analytics = repo.get_aggregate_analytics()
        assert analytics.total_sessions == 50
        assert analytics.total_duration_seconds == 250.0
        assert analytics.average_duration_seconds == 5.0
        assert analytics.min_duration_seconds == 5.0
        assert analytics.max_duration_seconds == 5.0
        assert analytics.identified_sessions_count == 25
        assert analytics.anonymous_sessions_count == 25
        assert analytics.earliest_session_started_at == 0.0
        assert analytics.latest_session_ended_at == 495.0

        db.close()

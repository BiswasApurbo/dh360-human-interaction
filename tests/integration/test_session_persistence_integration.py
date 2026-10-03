"""Integration tests for session persistence, multi-session survival, and service restarts."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

from dh360_hi.application.config import HumanInteractionConfig
from dh360_hi.application.factory import create_interaction_service
from dh360_hi.backend.database import DatabaseManager
from dh360_hi.backend.models import ConsentStatus, PersonStatus
from dh360_hi.backend.session_repository import (
    BaseSessionRepository,
    SessionPersistenceWorker,
    SQLiteSessionRepository,
)
from dh360_hi.contracts.dto import CreatePersonCommand
from dh360_hi.contracts.interaction import InteractionState, InteractionSummary
from dh360_hi.edge.runtime import EdgeRuntimeConfig
from dh360_hi.edge.session_manager import SessionConfig
from dh360_hi.vision.frame import SyntheticFrameSource
from dh360_hi.vision.identity_state import IdentityState, IdentityStateManager, TrackIdentityState
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
) -> TwoRateFrameResult:
    transitions: list[PresenceTransition] = []
    if frame_id == 1:
        transitions.append(
            PresenceTransition(
                event=PresenceEvent.PERSON_ENTERED,
                track_id=track_id,
                person_id=person_id,
                display_name=display_name,
                frame_id=frame_id,
                timestamp=float(frame_id) * 0.033,
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
                timestamp=float(frame_id) * 0.033,
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
        timestamp=float(frame_id) * 0.033,
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


class TestSessionPersistenceIntegration:
    def test_deleting_identified_person_detaches_active_session_before_persistence(self) -> None:
        """A session ending after person deletion persists without a dangling person FK."""
        pipeline = MagicMock(spec=TwoRatePerceptionPipeline)
        pipeline.worker.is_running = True
        pipeline.worker.quiesce.return_value = True
        identity_manager = IdentityStateManager()
        identity_manager._tracks[1] = TrackIdentityState(
            track_id=1,
            state=IdentityState.IDENTIFIED,
            current_person_id="pending-person-id",
            current_display_name="Jordan",
        )
        pipeline.identity_manager = identity_manager

        service = create_interaction_service(
            config=HumanInteractionConfig(),
            pipeline=pipeline,
            enable_perception=True,
        )
        person = service.create_person(
            CreatePersonCommand(display_name="Jordan", consent_status=ConsentStatus.GRANTED)
        )
        identity_manager._tracks[1].current_person_id = person.person_id

        try:
            runtime = service.runtime
            assert runtime is not None
            runtime.session_manager.process_frame_result(
                _mk_scripted_frame_result(
                    frame_id=1,
                    person_id=person.person_id,
                    display_name=person.display_name,
                    total_frames=10,
                )
            )

            assert service.delete_person(person.person_id) is True
            active_session = runtime.session_manager.active_session
            assert active_session is not None
            assert active_session.person_id is None
            assert active_session.display_name == "Jordan"

            ended_events = runtime.session_manager.process_frame_result(
                _mk_scripted_frame_result(
                    frame_id=10,
                    person_id=person.person_id,
                    display_name="Jordan",
                    total_frames=10,
                )
            )
            session_ended = next(
                event for event in ended_events if event.__class__.__name__ == "SessionEndedEvent"
            )
            assert session_ended.final_person_id is None
            assert session_ended.final_display_name == "Jordan"
            assert service.flush_session_persistence(timeout=2.0) is True

            summary = service.get_last_summary()
            assert summary is not None
            persisted = service.get_session(summary.session_id)
            assert persisted is not None
            assert persisted.person_id is None
            assert persisted.display_name == "Jordan"
            assert persisted.identity_was_known is True
        finally:
            service.stop()

    def test_single_anonymous_session_persisted_to_sqlite(self) -> None:
        """Verify single anonymous session runs through pipeline, finishes, and saves to SQLite."""
        source = SyntheticFrameSource(num_frames=5, fps=100.0)

        pipeline = MagicMock(spec=TwoRatePerceptionPipeline)
        pipeline.worker.is_running = True
        pipeline.process_frame.side_effect = lambda f: _mk_scripted_frame_result(
            frame_id=f.frame_id + 1,
            person_id=None,
            display_name=None,
            total_frames=5,
        )

        cfg = HumanInteractionConfig(
            runtime=EdgeRuntimeConfig(target_fps=1000.0),
            session=SessionConfig(require_confirmed_presence=True),
        )

        service = create_interaction_service(
            config=cfg,
            frame_source=source,
            pipeline=pipeline,
        )

        # Run stream to completion synchronously
        service.start(async_thread=False)

        # Flush background persistence worker
        assert service.flush_session_persistence(timeout=2.0) is True

        summary = service.get_last_summary()
        assert summary is not None
        session_id = summary.session_id

        # Query session via service query API
        persisted = service.get_session(session_id)
        assert persisted is not None
        assert persisted.session_id == session_id
        assert persisted.person_id is None
        assert persisted.display_name is None
        assert persisted.duration_seconds > 0.0
        assert persisted.identity_was_known is False

        assert service.count_sessions() == 1
        all_sessions = service.list_sessions()
        assert len(all_sessions) == 1
        assert all_sessions[0].session_id == session_id

        service.stop()

    def test_multiple_consecutive_sessions_persisted(self) -> None:
        """Verify multiple consecutive interaction sessions all persist and are queryable."""
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

        # Run Session 1 (frames 1 to 5)
        for f_id in range(1, 6):
            pipeline.process_frame.return_value = _mk_scripted_frame_result(
                frame_id=f_id,
                person_id=None,
                display_name=None,
                total_frames=5,
            )
            service.step_once()

        s1_summary = service.get_last_summary()
        assert s1_summary is not None
        s1_id = s1_summary.session_id

        # Run Session 2 (frames 1 to 5)
        for f_id in range(1, 6):
            pipeline.process_frame.return_value = _mk_scripted_frame_result(
                frame_id=f_id,
                person_id=None,
                display_name=None,
                total_frames=5,
            )
            service.step_once()

        s2_summary = service.get_last_summary()
        assert s2_summary is not None
        s2_id = s2_summary.session_id
        assert s1_id != s2_id

        assert service.flush_session_persistence(timeout=2.0) is True

        assert service.count_sessions() == 2
        sessions = service.list_sessions()
        assert len(sessions) == 2
        session_ids = [s.session_id for s in sessions]
        assert s1_id in session_ids
        assert s2_id in session_ids

        service.stop()

    def test_restart_persistence_across_service_instances(self, tmp_path: Path) -> None:
        """Verify that completed session records survive stopping and re-opening on disk DB."""
        db_file = tmp_path / "persistent_history.db"
        cfg = HumanInteractionConfig(
            db_path=str(db_file),
            runtime=EdgeRuntimeConfig(target_fps=1000.0),
            session=SessionConfig(require_confirmed_presence=True),
        )

        pipeline = MagicMock(spec=TwoRatePerceptionPipeline)
        pipeline.worker.is_running = True

        # Instance 1: create person and run session
        service1 = create_interaction_service(config=cfg, pipeline=pipeline)
        profile = service1.person_service.create_person(
            display_name="Dr. Robert Paulson",
            status=PersonStatus.ACTIVE,
            consent_status=ConsentStatus.GRANTED,
        )
        pid = profile.person_id

        for f_id in range(1, 6):
            pipeline.process_frame.return_value = _mk_scripted_frame_result(
                frame_id=f_id,
                person_id=pid,
                display_name="Dr. Robert Paulson",
                total_frames=5,
            )
            service1.step_once()

        summary1 = service1.get_last_summary()
        assert summary1 is not None
        s1_id = summary1.session_id

        # Stop service 1 (drains persistence worker and closes DB)
        service1.stop()

        # Instance 2: re-open on same database file
        service2 = create_interaction_service(config=cfg, pipeline=pipeline)

        # In-memory last summary of fresh service is None
        assert service2.get_last_summary() is None

        # But historical session is persisted and queryable from SQLite
        persisted_s1 = service2.get_session(s1_id)
        assert persisted_s1 is not None
        assert persisted_s1.session_id == s1_id
        assert persisted_s1.person_id == pid
        assert persisted_s1.display_name == "Dr. Robert Paulson"
        assert persisted_s1.identity_was_known is True

        # Filter by person_id
        person_sessions = service2.list_sessions(person_id=pid)
        assert len(person_sessions) == 1
        assert person_sessions[0].session_id == s1_id

        service2.stop()

    def test_shutdown_under_load_drains_without_worker_leak(self) -> None:
        """Verify that stop() gracefully persists pending sessions under active load."""
        db = DatabaseManager(":memory:")
        repo = SQLiteSessionRepository(db)
        worker = SessionPersistenceWorker(repository=repo, max_queue_size=128)
        worker.start()

        # Submit 20 sessions rapidly
        for i in range(20):
            sid = f"eeeeeeee-0000-0000-0000-{i:012d}"
            worker.submit(
                InteractionSummary(
                    session_id=sid,
                    person_id=None,
                    display_name=None,
                    person_profile=None,
                    started_at=float(i),
                    ended_at=float(i + 1),
                    duration_seconds=1.0,
                    final_state=InteractionState.FAREWELL,
                    identity_was_known=False,
                    identity_transitions_count=0,
                )
            )

        # Stop immediately without explicit flush
        worker.stop(timeout=3.0)

        assert worker.is_running is False
        assert repo.count_sessions() == 20
        db.close()

    def test_persistence_failure_observability(self) -> None:
        """Verify that repository write failure is recorded in worker error metrics."""
        mock_repo = MagicMock(spec=BaseSessionRepository)
        mock_repo.save_session.side_effect = RuntimeError("I/O error during save")

        worker = SessionPersistenceWorker(repository=mock_repo, max_queue_size=16)
        worker.start()

        sid = "ffffffff-1111-1111-1111-111111111111"
        worker.submit(
            InteractionSummary(
                session_id=sid,
                person_id=None,
                display_name=None,
                person_profile=None,
                started_at=1.0,
                ended_at=2.0,
                duration_seconds=1.0,
                final_state=InteractionState.IDLE,
                identity_was_known=False,
                identity_transitions_count=0,
            )
        )

        assert worker.flush(timeout=1.0) is True
        assert worker.error_count == 1
        assert "I/O error during save" in str(worker.last_error)
        assert worker.saved_count == 0

        worker.stop(timeout=1.0)

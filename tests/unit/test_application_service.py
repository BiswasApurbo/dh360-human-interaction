"""Unit tests for HumanInteractionService facade and transport-neutral boundary."""

from __future__ import annotations

import threading
import uuid
from unittest.mock import MagicMock

import numpy as np
import pytest

from dh360_hi.application.config import HumanInteractionConfig
from dh360_hi.application.coordinator import InteractionCoordinator
from dh360_hi.application.health import SubsystemHealthStatus
from dh360_hi.application.service import HumanInteractionService
from dh360_hi.backend.database import DatabaseManager
from dh360_hi.backend.models import (
    ConsentStatus,
    EmbeddingRecord,
    PersonEngagementStats,
    PersonProfile,
    PersonRecord,
    PersonStatus,
    SessionAggregateAnalytics,
    SessionQueryFilter,
    SessionRecord,
    vector_to_blob,
)
from dh360_hi.backend.person_service import BasePersonService
from dh360_hi.backend.repository import BasePersonRepository
from dh360_hi.contracts.dto import (
    CreatePersonCommand,
    EnrollmentProgressDTO,
    EnrollmentResultDTO,
    InteractionContextDTO,
    InteractionSessionDTO,
    InteractionSummaryDTO,
    PersonEngagementStatsDTO,
    PersonProfileDTO,
    RuntimeMetricsDTO,
    SessionAggregateAnalyticsDTO,
    StepResultDTO,
    UpdatePersonCommand,
)
from dh360_hi.contracts.errors import ConflictAppError, ServiceUnavailableError
from dh360_hi.contracts.events import (
    PersonEnteredEvent,
    SessionEndedEvent,
    SessionStartedEvent,
)
from dh360_hi.contracts.integration import (
    BaseIntegrationEvent,
    SessionEndedIntegrationEvent,
    SessionStartedIntegrationEvent,
)
from dh360_hi.contracts.interaction import (
    InteractionContext,
    InteractionState,
    InteractionSummary,
)
from dh360_hi.edge.event_bus import EventBus
from dh360_hi.edge.runtime import EdgeInteractionRuntime, RuntimeMetrics, RuntimeState
from dh360_hi.edge.session_manager import SessionManager
from dh360_hi.vision.enrollment import (
    EnrollmentMode,
    EnrollmentProgress,
    EnrollmentResult,
    EnrollmentState,
    FaceEnrollmentController,
)
from dh360_hi.vision.frame import SyntheticFrameSource
from dh360_hi.vision.recognition import BaseFaceRecognizer
from dh360_hi.vision.target_selector import TargetSelectionResult
from dh360_hi.vision.two_rate_pipeline import TwoRateFrameResult, TwoRatePerceptionPipeline


def _mk_service_harness(
    runtime_state: RuntimeState = RuntimeState.STOPPED,
    with_enrollment: bool = False,
) -> tuple[HumanInteractionService, dict[str, MagicMock]]:
    runtime = MagicMock(spec=EdgeInteractionRuntime)
    runtime.state = runtime_state
    runtime.metrics = RuntimeMetrics(frames_processed=42, current_fps=30.0)
    runtime.begin_mutation_barrier.return_value = runtime_state
    runtime.pipeline.worker.quiesce.return_value = True

    coordinator = MagicMock(spec=InteractionCoordinator)
    coordinator.get_context.return_value = InteractionContext(
        session_id="sess-1",
        state=InteractionState.ENGAGED,
    )
    coordinator.get_last_summary.return_value = InteractionSummary(
        session_id="sess-0",
        person_id="p-1",
        display_name="Alice",
        person_profile=None,
        started_at=100.0,
        ended_at=105.0,
        duration_seconds=5.0,
        final_state=InteractionState.ENGAGED,
        identity_was_known=True,
        identity_transitions_count=1,
    )

    person_service = MagicMock(spec=BasePersonService)
    repository = MagicMock(spec=BasePersonRepository)
    recognizer = MagicMock(spec=BaseFaceRecognizer)
    db_manager = MagicMock(spec=DatabaseManager)
    event_bus = MagicMock(spec=EventBus)
    session_repository = MagicMock()
    persistence_worker = MagicMock()
    persistence_worker.error_count = 0
    persistence_worker.flush.return_value = True
    enrollment_controller = MagicMock(spec=FaceEnrollmentController) if with_enrollment else None

    config = HumanInteractionConfig()

    svc = HumanInteractionService(
        runtime=runtime,
        coordinator=coordinator,
        person_service=person_service,
        repository=repository,
        recognizer=recognizer,
        db_manager=db_manager,
        event_bus=event_bus,
        session_repository=session_repository,
        persistence_worker=persistence_worker,
        enrollment_controller=enrollment_controller,
        config=config,
    )

    mocks = {
        "runtime": runtime,
        "coordinator": coordinator,
        "person_service": person_service,
        "repository": repository,
        "recognizer": recognizer,
        "db_manager": db_manager,
        "event_bus": event_bus,
        "session_repository": session_repository,
        "persistence_worker": persistence_worker,
        "enrollment_controller": enrollment_controller,
    }
    return svc, mocks


class TestServicePropertiesAndQueries:
    def test_property_delegations(self) -> None:
        """Verify properties reflect underlying component states."""
        svc, mocks = _mk_service_harness(runtime_state=RuntimeState.RUNNING)

        assert svc.is_running is True
        assert svc.state == RuntimeState.RUNNING
        assert isinstance(svc.context, InteractionContextDTO)
        assert svc.context.session_id == "sess-1"
        assert svc.metrics.frames_processed == 42
        assert svc.runtime is mocks["runtime"]
        assert svc.coordinator is mocks["coordinator"]
        assert svc.person_service is mocks["person_service"]
        assert svc.config.model_name == "SFace"

    def test_context_queries(self) -> None:
        """Verify get_interaction_context and get_last_summary return DTOs."""
        svc, mocks = _mk_service_harness()

        ctx = svc.get_interaction_context()
        assert isinstance(ctx, InteractionContextDTO)
        assert ctx.session_id == "sess-1"
        assert ctx.state == "engaged"
        mocks["coordinator"].get_context.assert_called_once()

        summary = svc.get_last_summary()
        assert summary is not None
        assert isinstance(summary, InteractionSummaryDTO)
        assert summary.session_id == "sess-0"
        mocks["coordinator"].get_last_summary.assert_called_once()

        metrics = svc.get_runtime_metrics()
        assert isinstance(metrics, RuntimeMetricsDTO)
        assert metrics.frames_processed == 42

        state = svc.get_runtime_state()
        assert state == RuntimeState.STOPPED

    def test_person_queries(self) -> None:
        """Verify get_person and list_persons return PersonProfileDTO."""
        svc, mocks = _mk_service_harness()
        valid_pid = str(uuid.uuid4())
        mock_profile = PersonProfile(
            person_id=valid_pid,
            display_name="Bob Jones",
            preferred_name="Bob",
            status=PersonStatus.ACTIVE,
            consent_status=ConsentStatus.GRANTED,
            consent_timestamp="2026-10-01T12:00:00Z",
            notes=None,
            created_at="2026-10-01T12:00:00Z",
            updated_at="2026-10-01T12:00:00Z",
            embedding_count=1,
        )
        mocks["person_service"].get_person.return_value = mock_profile
        mocks["person_service"].list_people.return_value = [mock_profile]

        profile = svc.get_person(valid_pid)
        assert profile is not None
        assert isinstance(profile, PersonProfileDTO)
        assert profile.display_name == "Bob Jones"
        mocks["person_service"].get_person.assert_called_once_with(valid_pid)

        persons = svc.list_persons(status=PersonStatus.ACTIVE, consent_status=ConsentStatus.GRANTED)
        assert len(persons) == 1
        assert isinstance(persons[0], PersonProfileDTO)
        assert persons[0].person_id == valid_pid
        mocks["person_service"].list_people.assert_called_once_with(status=PersonStatus.ACTIVE)

    def test_person_commands(self) -> None:
        """Verify create_person, update_person, delete_person, and reset_biometrics."""
        svc, mocks = _mk_service_harness()
        pid = str(uuid.uuid4())
        mock_profile = PersonProfile(
            person_id=pid,
            display_name="Charlie",
            preferred_name=None,
            status=PersonStatus.ACTIVE,
            consent_status=ConsentStatus.GRANTED,
            consent_timestamp="2026-10-01T12:00:00Z",
            notes=None,
            created_at="2026-10-01T12:00:00Z",
            updated_at="2026-10-01T12:00:00Z",
        )
        mocks["person_service"].create_person.return_value = mock_profile
        mocks["person_service"].update_person.return_value = mock_profile
        mocks["person_service"].delete_person.return_value = True
        mocks["person_service"].delete_person_embeddings.return_value = 3

        # Create
        create_cmd = CreatePersonCommand(display_name="Charlie")
        res_create = svc.create_person(create_cmd)
        assert isinstance(res_create, PersonProfileDTO)
        assert res_create.display_name == "Charlie"
        mocks["person_service"].create_person.assert_called_once()

        # Update
        upd_cmd = UpdatePersonCommand(preferred_name="Chuck")
        res_upd = svc.update_person(pid, upd_cmd)
        assert isinstance(res_upd, PersonProfileDTO)
        mocks["person_service"].update_person.assert_called_once()

        # Delete
        del_res = svc.delete_person(pid)
        assert del_res is True
        mocks["person_service"].delete_person.assert_called_once_with(person_id=pid)
        mocks["runtime"].session_manager.invalidate_person_identity.assert_any_call(
            pid,
            deleted=True,
        )
        mocks["coordinator"].forget_deleted_person.assert_called_once_with(pid)

        # Reset biometrics
        reset_res = svc.reset_person_biometrics(pid)
        assert reset_res == 3
        mocks["person_service"].delete_person_embeddings.assert_called_once_with(person_id=pid)
        mocks["runtime"].session_manager.invalidate_person_identity.assert_any_call(
            pid,
            deleted=False,
        )
        mocks["recognizer"].replace_gallery.assert_called()
        mocks["runtime"].end_mutation_barrier.assert_called_with(
            RuntimeState.STOPPED,
            resume=True,
        )

    def test_mutation_aborts_before_database_write_when_worker_cannot_quiesce(self) -> None:
        """A failed worker barrier leaves the profile and runtime state untouched."""
        svc, mocks = _mk_service_harness(runtime_state=RuntimeState.RUNNING)
        mocks["runtime"].pipeline.worker.quiesce.return_value = False

        with pytest.raises(ServiceUnavailableError):
            svc.delete_person(str(uuid.uuid4()))

        mocks["person_service"].delete_person.assert_not_called()
        mocks["runtime"].end_mutation_barrier.assert_called_once_with(
            RuntimeState.RUNNING,
            resume=True,
        )

    def test_mutation_aborts_with_503_when_worker_quiescence_raises(self) -> None:
        """Unexpected worker-barrier errors are normalized and prevent database mutation."""
        svc, mocks = _mk_service_harness(runtime_state=RuntimeState.RUNNING)
        mocks["runtime"].pipeline.worker.quiesce.side_effect = RuntimeError("worker failed")

        with pytest.raises(ServiceUnavailableError):
            svc.delete_person(str(uuid.uuid4()))

        mocks["person_service"].delete_person.assert_not_called()
        mocks["runtime"].end_mutation_barrier.assert_called_once_with(
            RuntimeState.RUNNING,
            resume=True,
        )

    def test_delete_aborts_when_persistence_flush_reports_write_error(self) -> None:
        """A queued session write failure prevents deletion before the DB mutation."""
        svc, mocks = _mk_service_harness(runtime_state=RuntimeState.RUNNING)

        def fail_flush(*, timeout: float) -> bool:
            del timeout
            mocks["persistence_worker"].error_count = 1
            return True

        mocks["persistence_worker"].flush.side_effect = fail_flush
        with pytest.raises(ServiceUnavailableError):
            svc.delete_person(str(uuid.uuid4()))

        mocks["person_service"].delete_person.assert_not_called()
        mocks["runtime"].end_mutation_barrier.assert_called_once_with(
            RuntimeState.RUNNING,
            resume=True,
        )

    def test_delete_aborts_with_503_when_persistence_flush_raises(self) -> None:
        """Unexpected persistence barrier errors prevent person deletion."""
        svc, mocks = _mk_service_harness(runtime_state=RuntimeState.RUNNING)
        mocks["persistence_worker"].flush.side_effect = RuntimeError("flush failed")

        with pytest.raises(ServiceUnavailableError):
            svc.delete_person(str(uuid.uuid4()))

        mocks["person_service"].delete_person.assert_not_called()
        mocks["runtime"].end_mutation_barrier.assert_called_once_with(
            RuntimeState.RUNNING,
            resume=True,
        )

    def test_postcommit_gallery_failure_leaves_runtime_paused(self) -> None:
        """A committed update is not reported successful when gallery reconciliation fails."""
        svc, mocks = _mk_service_harness(runtime_state=RuntimeState.RUNNING)
        person_id = str(uuid.uuid4())
        profile = PersonProfile(
            person_id=person_id,
            display_name="Before",
            preferred_name=None,
            status=PersonStatus.ACTIVE,
            consent_status=ConsentStatus.GRANTED,
            consent_timestamp="2026-10-01T12:00:00Z",
            notes=None,
            created_at="2026-10-01T12:00:00Z",
            updated_at="2026-10-01T12:00:00Z",
        )
        mocks["person_service"].get_person.return_value = profile
        mocks["person_service"].update_person.return_value = profile
        svc.reload_gallery = MagicMock(side_effect=RuntimeError("gallery unavailable"))

        with pytest.raises(ServiceUnavailableError, match="runtime remains paused"):
            svc.update_person(person_id, UpdatePersonCommand(display_name="After"))

        mocks["person_service"].update_person.assert_called_once()
        mocks["runtime"].end_mutation_barrier.assert_called_once_with(
            RuntimeState.RUNNING,
            resume=False,
        )

    @pytest.mark.parametrize(
        ("config", "command"),
        [
            (
                HumanInteractionConfig(require_active_for_gallery=False),
                UpdatePersonCommand(status=PersonStatus.INACTIVE),
            ),
            (
                HumanInteractionConfig(require_consent_for_gallery=False),
                UpdatePersonCommand(consent_status=ConsentStatus.WITHDRAWN),
            ),
        ],
    )
    def test_eligibility_update_rejected_when_gallery_policy_cannot_enforce_it(
        self,
        config: HumanInteractionConfig,
        command: UpdatePersonCommand,
    ) -> None:
        """Status/consent removal cannot succeed under a gallery policy that ignores it."""
        svc, mocks = _mk_service_harness()
        svc._config = config
        person_id = str(uuid.uuid4())
        mocks["person_service"].get_person.return_value = PersonProfile(
            person_id=person_id,
            display_name="Current",
            preferred_name=None,
            status=PersonStatus.ACTIVE,
            consent_status=ConsentStatus.GRANTED,
            consent_timestamp="2026-10-01T12:00:00Z",
            notes=None,
            created_at="2026-10-01T12:00:00Z",
            updated_at="2026-10-01T12:00:00Z",
        )

        with pytest.raises(ConflictAppError):
            svc.update_person(person_id, command)

        mocks["person_service"].update_person.assert_not_called()

    def test_historical_session_queries(self) -> None:
        """Verify get_session, list_sessions, and count_sessions return DTOs."""
        svc, mocks = _mk_service_harness()
        sid = str(uuid.uuid4())
        pid = str(uuid.uuid4())
        mock_record = SessionRecord(
            session_id=sid,
            person_id=pid,
            display_name="Alice",
            started_at=100.0,
            ended_at=110.0,
            duration_seconds=10.0,
            final_state="engaged",
            identity_was_known=True,
            identity_transitions_count=1,
            created_at="2026-10-01T12:00:00Z",
        )
        mocks["session_repository"].get_session.return_value = mock_record
        mocks["session_repository"].list_sessions.return_value = [mock_record]
        mocks["session_repository"].count_sessions.return_value = 1
        mocks["persistence_worker"].flush.return_value = True

        rec = svc.get_session(sid)
        assert rec is not None
        assert isinstance(rec, InteractionSessionDTO)
        assert rec.session_id == sid
        mocks["session_repository"].get_session.assert_called_once_with(sid)

        sessions = svc.list_sessions(person_id=pid, limit=10, offset=5)
        assert len(sessions) == 1
        assert isinstance(sessions[0], InteractionSessionDTO)
        mocks["session_repository"].list_sessions.assert_called_once_with(
            person_id=pid, limit=10, offset=5
        )

        count = svc.count_sessions(person_id=pid)
        assert count == 1
        mocks["session_repository"].count_sessions.assert_called_once_with(person_id=pid)

        flushed = svc.flush_session_persistence(timeout=1.0)
        assert flushed is True
        mocks["persistence_worker"].flush.assert_called_once_with(timeout=1.0)

    def test_query_sessions_and_analytics_delegation(self) -> None:
        """Verify query_sessions, analytics, and engagement return DTOs."""
        svc, mocks = _mk_service_harness()
        flt = SessionQueryFilter(is_identified=True)
        sid = str(uuid.uuid4())
        pid = "11111111-1111-1111-1111-111111111111"
        mock_rec = SessionRecord(
            session_id=sid,
            person_id=pid,
            display_name="User",
            started_at=100.0,
            ended_at=110.0,
            duration_seconds=10.0,
            final_state="engaged",
            identity_was_known=True,
            identity_transitions_count=1,
            created_at="2026-10-01T12:00:00Z",
        )
        mock_analytics = SessionAggregateAnalytics(
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
        mock_engagement = PersonEngagementStats(
            person_id="11111111-1111-1111-1111-111111111111",
            display_name="User",
            total_sessions=1,
            total_duration_seconds=10.0,
            average_duration_seconds=10.0,
            first_interaction_at=100.0,
            last_interaction_at=110.0,
        )

        mocks["session_repository"].query_sessions.return_value = [mock_rec]
        mocks["session_repository"].count_query_sessions.return_value = 1
        mocks["session_repository"].get_aggregate_analytics.return_value = mock_analytics
        mocks["session_repository"].get_person_engagement_stats.return_value = mock_engagement

        res_sessions = svc.query_sessions(flt)
        assert len(res_sessions) == 1
        assert isinstance(res_sessions[0], InteractionSessionDTO)
        assert res_sessions[0].session_id == sid
        mocks["session_repository"].query_sessions.assert_called_once_with(flt)

        res_count = svc.count_query_sessions(flt)
        assert res_count == 1
        mocks["session_repository"].count_query_sessions.assert_called_once_with(flt)

        res_analytics = svc.get_aggregate_analytics(flt)
        assert isinstance(res_analytics, SessionAggregateAnalyticsDTO)
        assert res_analytics.total_sessions == 1
        mocks["session_repository"].get_aggregate_analytics.assert_called_once_with(filter=flt)

        res_engagement = svc.get_person_engagement_stats("11111111-1111-1111-1111-111111111111")
        assert res_engagement is not None
        assert isinstance(res_engagement, PersonEngagementStatsDTO)
        assert res_engagement.person_id == "11111111-1111-1111-1111-111111111111"
        mocks["session_repository"].get_person_engagement_stats.assert_called_once_with(
            "11111111-1111-1111-1111-111111111111"
        )

    def test_service_queries_with_none_repository(self) -> None:
        """Verify service query APIs handle None repository safely without crashing."""
        runtime = MagicMock(spec=EdgeInteractionRuntime)
        coordinator = MagicMock(spec=InteractionCoordinator)
        person_service = MagicMock(spec=BasePersonService)
        repository = MagicMock(spec=BasePersonRepository)
        recognizer = MagicMock(spec=BaseFaceRecognizer)
        db_manager = MagicMock(spec=DatabaseManager)
        event_bus = MagicMock(spec=EventBus)

        svc = HumanInteractionService(
            runtime=runtime,
            coordinator=coordinator,
            person_service=person_service,
            repository=repository,
            recognizer=recognizer,
            db_manager=db_manager,
            event_bus=event_bus,
            session_repository=None,
            persistence_worker=None,
        )

        assert svc.get_session("some-id") is None
        assert svc.list_sessions() == []
        assert svc.count_sessions() == 0
        assert svc.query_sessions() == []
        assert svc.count_query_sessions() == 0
        assert svc.get_person_engagement_stats("11111111-1111-1111-1111-111111111111") is None
        analytics = svc.get_aggregate_analytics()
        assert isinstance(analytics, SessionAggregateAnalyticsDTO)
        assert analytics.total_sessions == 0
        assert analytics.total_duration_seconds == 0.0
        assert svc.flush_session_persistence() is True


class TestServiceLifecycle:
    def test_start_delegation(self) -> None:
        """Verify start command starts persistence worker and delegates to runtime."""
        svc, mocks = _mk_service_harness()

        svc.start(async_thread=True)
        mocks["persistence_worker"].start.assert_called_once()
        mocks["runtime"].start.assert_called_once_with(async_thread=True)

    def test_pause_and_resume(self) -> None:
        """Verify pause and resume commands delegate to runtime."""
        svc, mocks = _mk_service_harness(runtime_state=RuntimeState.RUNNING)

        svc.pause()
        mocks["runtime"].pause.assert_called_once()

        svc.resume()
        mocks["runtime"].resume.assert_called_once()

    def test_stop_graceful_shutdown(self) -> None:
        """Verify stop coordinates shutdown across runtime, worker, coordinator, and db."""
        svc, mocks = _mk_service_harness(runtime_state=RuntimeState.RUNNING)

        svc.stop(timeout=5.0)

        mocks["runtime"].stop.assert_called_once_with(timeout=5.0)
        mocks["persistence_worker"].stop.assert_called_once_with(timeout=5.0)
        mocks["coordinator"].close.assert_called_once()
        mocks["db_manager"].close.assert_called_once()

    def test_stop_handles_exceptions_gracefully(self) -> None:
        """Verify stop does not raise if a subsystem raises during cleanup."""
        svc, mocks = _mk_service_harness()
        mocks["runtime"].stop.side_effect = RuntimeError("Runtime stop failure")
        mocks["coordinator"].close.side_effect = RuntimeError("Coordinator close failure")
        mocks["db_manager"].close.side_effect = RuntimeError("DB close failure")

        # Must not raise
        svc.stop()

    def test_step_once_returns_step_result_dto(self) -> None:
        """Verify step_once executes frame step and returns sanitized StepResultDTO."""
        svc, mocks = _mk_service_harness()
        mock_frame_res = MagicMock()
        mock_events = [
            PersonEnteredEvent(timestamp=1.0, frame_id=1, track_id=1),
            SessionStartedEvent(
                event_id="ev-10",
                timestamp=1.0,
                frame_id=1,
                track_id=1,
                session_id="sess-1",
            ),
        ]
        mocks["runtime"].step_once.return_value = (mock_frame_res, mock_events)

        step_res = svc.step_once()

        mocks["runtime"].step_once.assert_called_once()
        assert isinstance(step_res, StepResultDTO)
        assert step_res.context.session_id == "sess-1"
        # PersonEnteredEvent is filtered out; SessionStartedEvent is mapped
        assert len(step_res.events) == 1
        assert isinstance(step_res.events[0], SessionStartedIntegrationEvent)
        assert step_res.events[0].session_id == "sess-1"
        assert not hasattr(step_res.events[0], "track_id")


class TestServiceEnrollmentCommands:
    def test_enrollment_commands_when_controller_configured(self) -> None:
        """Verify enrollment methods delegate to FaceEnrollmentController."""
        svc, mocks = _mk_service_harness(with_enrollment=True)
        ctrl = mocks["enrollment_controller"]
        prog = EnrollmentProgress(
            session_id="esess-1",
            enrollment_batch_id="batch-1",
            person_id="p-1",
            display_name="Alice",
            mode=EnrollmentMode.ADD_SAMPLES,
            state=EnrollmentState.COLLECTING,
            samples_collected=1,
            target_samples=5,
            minimum_samples=3,
            total_attempts=1,
            last_rejection_reason=None,
            last_rejection_details=None,
            rejection_counts={},
            is_viable=False,
            is_complete=False,
        )
        res = EnrollmentResult(
            session_id="esess-1",
            enrollment_batch_id="batch-1",
            person_id="p-1",
            display_name="Alice",
            mode=EnrollmentMode.ADD_SAMPLES,
            is_success=True,
            state=EnrollmentState.COMPLETED,
            samples_enrolled=5,
            persisted_embedding_ids=["e1", "e2"],
            total_attempts=5,
            error_message=None,
            rejection_counts={},
        )
        ctrl.start_session.return_value = prog
        ctrl.process_frame.return_value = prog
        ctrl.commit_session.return_value = res
        ctrl.current_progress = prog

        # Start
        start_prog = svc.start_enrollment("p-1")
        assert isinstance(start_prog, EnrollmentProgressDTO)
        assert start_prog.session_id == "esess-1"
        ctrl.start_session.assert_called_once_with(person_id="p-1", mode="add_samples")

        # Process frame
        frame_prog = svc.process_enrollment_frame(np.zeros((100, 100, 3), dtype=np.uint8))
        assert isinstance(frame_prog, EnrollmentProgressDTO)
        ctrl.process_frame.assert_called_once()

        # Get current progress
        curr_prog = svc.get_enrollment_progress()
        assert curr_prog is not None
        assert curr_prog.session_id == "esess-1"

        # Commit
        commit_res = svc.commit_enrollment()
        assert isinstance(commit_res, EnrollmentResultDTO)
        assert commit_res.is_success is True

        # Cancel
        assert svc.cancel_enrollment() is True
        ctrl.cancel_session.assert_called_once()

    def test_enrollment_commands_raise_when_unconfigured(self) -> None:
        """Verify enrollment methods raise ServiceUnavailableError when controller is None."""
        svc, _ = _mk_service_harness(with_enrollment=False)

        with pytest.raises(ServiceUnavailableError):
            svc.start_enrollment("p-1")

        with pytest.raises(ServiceUnavailableError):
            svc.process_enrollment_frame(np.zeros((10, 10, 3), dtype=np.uint8))

        with pytest.raises(ServiceUnavailableError):
            svc.cancel_enrollment()

        with pytest.raises(ServiceUnavailableError):
            svc.commit_enrollment()

        assert svc.get_enrollment_progress() is None


class TestServiceGalleryAndListeners:
    def test_mutation_lock_does_not_serialize_ordinary_perception(self) -> None:
        """The API-only mutation lock is not acquired by the normal frame loop."""
        service, _ = _mk_service_harness()
        frame_processed = threading.Event()
        pipeline = MagicMock(spec=TwoRatePerceptionPipeline)

        def process_frame(_frame: object) -> TwoRateFrameResult:
            frame_processed.set()
            return TwoRateFrameResult(
                frame_id=1,
                timestamp=1.0,
                detections=[],
                tracks=[],
                quality_results={},
                identity_states={},
                target_result=TargetSelectionResult(
                    primary_target=None,
                    candidates=[],
                    target_changed=False,
                    previous_target_track_id=None,
                    frame_id=1,
                ),
                presence_transitions=[],
                completed_recognitions=[],
                ambiguous_crossing_track_ids=set(),
                pending_worker_tasks=0,
                frame_latency_ms=0.1,
                total_pipeline_latency_ms=0.1,
            )

        pipeline.process_frame.side_effect = process_frame
        runtime = EdgeInteractionRuntime(
            SyntheticFrameSource(num_frames=100, fps=1000.0),
            pipeline,
            SessionManager(),
            EventBus(),
        )
        service._runtime = runtime
        assert service._person_mutation_lock.acquire(timeout=0.1)
        try:
            runtime.start(async_thread=True)
            assert frame_processed.wait(timeout=1.0)
            assert runtime.state == RuntimeState.RUNNING
        finally:
            service._person_mutation_lock.release()
            runtime.stop(timeout=1.0)

    def test_reload_gallery_syncs_with_repository(self) -> None:
        """Verify reload_gallery extracts records from repository and replaces gallery."""
        svc, mocks = _mk_service_harness()

        valid_pid = str(uuid.uuid4())
        valid_eid = str(uuid.uuid4())
        p_rec = PersonRecord(
            person_id=valid_pid,
            display_name="Alice",
            preferred_name=None,
            status=PersonStatus.ACTIVE,
            consent_status=ConsentStatus.GRANTED,
            consent_timestamp="2026-10-01T12:00:00Z",
            notes=None,
            created_at="2026-10-01T12:00:00Z",
            updated_at="2026-10-01T12:00:00Z",
        )
        fake_vec = np.zeros(128, dtype=np.float32)
        emb_rec = EmbeddingRecord(
            embedding_id=valid_eid,
            person_id=valid_pid,
            embedding=vector_to_blob(fake_vec),
            dimension=128,
            model_name="SFace",
            model_version="2021dec",
            quality_score=0.9,
            created_at="2026-10-01T12:00:00Z",
        )
        mocks["repository"].list_persons.return_value = [p_rec]
        mocks["repository"].get_all_embeddings.return_value = [emb_rec]

        count = svc.reload_gallery()

        assert count == 1
        mocks["recognizer"].replace_gallery.assert_called_once()

    def test_context_listener_delegation(self) -> None:
        """Verify context listener registration and unregistration with DTO adapter."""
        svc, mocks = _mk_service_harness()
        cb = MagicMock()
        mocks["coordinator"].add_context_listener.return_value = 101
        mocks["coordinator"].remove_context_listener.return_value = True

        lid = svc.add_context_listener(cb)
        assert lid == 101
        mocks["coordinator"].add_context_listener.assert_called_once()

        res = svc.remove_context_listener(lid)
        assert res is True
        mocks["coordinator"].remove_context_listener.assert_called_once_with(101)

    def test_integration_event_listener_dispatch(self) -> None:
        """Verify integration event listener registration and event dispatching."""
        svc, _ = _mk_service_harness()
        received_events: list[BaseIntegrationEvent] = []

        def _listener(event: BaseIntegrationEvent) -> None:
            received_events.append(event)

        lid = svc.add_integration_event_listener(_listener)

        # Trigger internal domain event bridge
        domain_ev = SessionEndedEvent(
            event_id="ev-99",
            timestamp=105.0,
            session_id="sess-0",
            final_person_id="p-1",
        )
        svc._handle_domain_event_for_integration(domain_ev)

        assert len(received_events) == 1
        assert isinstance(received_events[0], SessionEndedIntegrationEvent)
        assert received_events[0].session_id == "sess-0"

        # Remove listener
        assert svc.remove_integration_event_listener(lid) is True
        assert svc.remove_integration_event_listener(lid) is False

    def test_get_health_snapshot(self) -> None:
        """Verify get_health_snapshot returns non-throwing snapshot."""
        svc, _ = _mk_service_harness()
        snap = svc.get_health_snapshot()
        assert snap.status in (SubsystemHealthStatus.HEALTHY, SubsystemHealthStatus.DEGRADED)


def test_step_enrollment_drives_server_capture_source() -> None:
    """Verify start/step/cancel drive a server-side FrameSource through the facade."""
    svc, mocks = _mk_service_harness(with_enrollment=True)
    ctrl = mocks["enrollment_controller"]
    progress = EnrollmentProgress(
        session_id="esess-capture",
        enrollment_batch_id="batch-capture",
        person_id="p-1",
        display_name="Capture",
        mode=EnrollmentMode.ADD_SAMPLES,
        state=EnrollmentState.COLLECTING,
        samples_collected=0,
        target_samples=5,
        minimum_samples=3,
        total_attempts=0,
        last_rejection_reason=None,
        last_rejection_details=None,
        rejection_counts={},
        is_viable=False,
        is_complete=False,
    )
    ctrl.start_session.return_value = progress
    ctrl.process_frame.return_value = progress

    source = SyntheticFrameSource(fps=30.0, num_frames=10)
    started = svc.start_enrollment("p-1", mode="add_samples", capture_source=source)
    assert started.session_id == "esess-capture"
    assert source.is_opened is True
    ctrl.start_session.assert_called_once_with(person_id="p-1", mode="add_samples")

    stepped = svc.step_enrollment()
    assert stepped.session_id == "esess-capture"
    ctrl.process_frame.assert_called_once()

    assert svc.cancel_enrollment() is True
    assert source.is_opened is False


def test_commit_enrollment_releases_capture_and_uses_barrier() -> None:
    """Verify a successful commit traverses the mutation barrier and releases capture."""
    svc, mocks = _mk_service_harness(runtime_state=RuntimeState.RUNNING, with_enrollment=True)
    ctrl = mocks["enrollment_controller"]
    ctrl.commit_session.return_value = EnrollmentResult(
        session_id="esess-capture",
        enrollment_batch_id="batch-capture",
        person_id="p-1",
        display_name="Capture",
        mode=EnrollmentMode.ADD_SAMPLES,
        is_success=True,
        state=EnrollmentState.COMPLETED,
        samples_enrolled=5,
        persisted_embedding_ids=["e1", "e2"],
        total_attempts=5,
        error_message=None,
        rejection_counts={},
    )

    source = SyntheticFrameSource(fps=30.0, num_frames=10)
    svc.start_enrollment("p-1", capture_source=source)
    assert source.is_opened is True

    result = svc.commit_enrollment()
    assert result.is_success is True
    assert source.is_opened is False
    mocks["runtime"].begin_mutation_barrier.assert_called_once()
    mocks["runtime"].end_mutation_barrier.assert_called_once()


def test_step_enrollment_uses_runtime_latest_frame_when_running() -> None:
    """Verify step_enrollment consumes runtime.latest_frame when runtime is actively running."""
    from dh360_hi.vision.frame import Frame

    svc, mocks = _mk_service_harness(runtime_state=RuntimeState.RUNNING, with_enrollment=True)
    mocks["runtime"].is_running = True
    test_frame = Frame(
        image=np.zeros((100, 100, 3), dtype=np.uint8),
        frame_id=7,
        timestamp=1.23,
    )
    mocks["runtime"].latest_frame = test_frame

    ctrl = mocks["enrollment_controller"]
    progress = EnrollmentProgress(
        session_id="esess-live",
        enrollment_batch_id="batch-live",
        person_id="p-1",
        display_name="Live Person",
        mode=EnrollmentMode.ADD_SAMPLES,
        state=EnrollmentState.COLLECTING,
        samples_collected=1,
        target_samples=30,
        minimum_samples=5,
        total_attempts=1,
        last_rejection_reason=None,
        last_rejection_details=None,
        rejection_counts={},
        is_viable=False,
        is_complete=False,
    )
    ctrl.start_session.return_value = progress
    ctrl.process_frame.return_value = progress

    started = svc.start_enrollment("p-1", capture_source=None)
    assert started.session_id == "esess-live"

    stepped = svc.step_enrollment()
    assert stepped.session_id == "esess-live"
    ctrl.process_frame.assert_called_once_with(test_frame)

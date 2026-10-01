"""Unit tests for HumanInteractionService facade."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import numpy as np

from dh360_hi.application.config import HumanInteractionConfig
from dh360_hi.application.coordinator import InteractionCoordinator
from dh360_hi.application.service import HumanInteractionService
from dh360_hi.backend.database import DatabaseManager
from dh360_hi.backend.models import (
    ConsentStatus,
    EmbeddingRecord,
    PersonProfile,
    PersonRecord,
    PersonStatus,
    vector_to_blob,
)
from dh360_hi.backend.person_service import BasePersonService
from dh360_hi.backend.repository import BasePersonRepository
from dh360_hi.contracts.events import PersonEnteredEvent
from dh360_hi.contracts.interaction import (
    InteractionContext,
    InteractionState,
    InteractionSummary,
)
from dh360_hi.edge.event_bus import EventBus
from dh360_hi.edge.runtime import EdgeInteractionRuntime, RuntimeMetrics, RuntimeState
from dh360_hi.vision.recognition import BaseFaceRecognizer


def _mk_service_harness(
    runtime_state: RuntimeState = RuntimeState.STOPPED,
) -> tuple[HumanInteractionService, dict[str, MagicMock]]:
    runtime = MagicMock(spec=EdgeInteractionRuntime)
    runtime.state = runtime_state
    runtime.metrics = RuntimeMetrics(frames_processed=42, current_fps=30.0)

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
    }
    return svc, mocks


class TestServicePropertiesAndQueries:
    def test_property_delegations(self) -> None:
        """Verify properties reflect underlying component states."""
        svc, mocks = _mk_service_harness(runtime_state=RuntimeState.RUNNING)

        assert svc.is_running is True
        assert svc.state == RuntimeState.RUNNING
        assert svc.context.session_id == "sess-1"
        assert svc.metrics.frames_processed == 42
        assert svc.runtime is mocks["runtime"]
        assert svc.coordinator is mocks["coordinator"]
        assert svc.person_service is mocks["person_service"]
        assert svc.config.model_name == "SFace"

    def test_context_queries(self) -> None:
        """Verify get_interaction_context and get_last_summary delegation."""
        svc, mocks = _mk_service_harness()

        ctx = svc.get_interaction_context()
        assert ctx.session_id == "sess-1"
        assert ctx.state == InteractionState.ENGAGED
        mocks["coordinator"].get_context.assert_called_once()

        summary = svc.get_last_summary()
        assert summary is not None
        assert summary.session_id == "sess-0"
        mocks["coordinator"].get_last_summary.assert_called_once()

        metrics = svc.get_runtime_metrics()
        assert metrics.frames_processed == 42

        state = svc.get_runtime_state()
        assert state == RuntimeState.STOPPED

    def test_person_queries(self) -> None:
        """Verify get_person and list_persons delegate to person_service."""
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
        assert profile.display_name == "Bob Jones"
        mocks["person_service"].get_person.assert_called_once_with(valid_pid)

        persons = svc.list_persons(status=PersonStatus.ACTIVE, consent_status=ConsentStatus.GRANTED)
        assert len(persons) == 1
        assert persons[0].person_id == valid_pid
        mocks["person_service"].list_people.assert_called_once_with(status=PersonStatus.ACTIVE)

    def test_historical_session_queries(self) -> None:
        """Verify get_session, list_sessions, and count_sessions delegate to session_repository."""
        svc, mocks = _mk_service_harness()
        sid = str(uuid.uuid4())
        mock_record = MagicMock()
        mocks["session_repository"].get_session.return_value = mock_record
        mocks["session_repository"].list_sessions.return_value = [mock_record]
        mocks["session_repository"].count_sessions.return_value = 1
        mocks["persistence_worker"].flush.return_value = True

        rec = svc.get_session(sid)
        assert rec is mock_record
        mocks["session_repository"].get_session.assert_called_once_with(sid)

        sessions = svc.list_sessions(person_id="pid-1", limit=10, offset=5)
        assert len(sessions) == 1
        mocks["session_repository"].list_sessions.assert_called_once_with(
            person_id="pid-1", limit=10, offset=5
        )

        count = svc.count_sessions(person_id="pid-1")
        assert count == 1
        mocks["session_repository"].count_sessions.assert_called_once_with(person_id="pid-1")

        flushed = svc.flush_session_persistence(timeout=1.0)
        assert flushed is True
        mocks["persistence_worker"].flush.assert_called_once_with(timeout=1.0)

    def test_query_sessions_and_analytics_delegation(self) -> None:
        """Verify query_sessions, count_query_sessions, analytics, and engagement delegation."""
        svc, mocks = _mk_service_harness()
        from dh360_hi.backend.models import (
            PersonEngagementStats,
            SessionAggregateAnalytics,
            SessionQueryFilter,
        )

        flt = SessionQueryFilter(is_identified=True)
        mock_rec = MagicMock()
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
        assert res_sessions == [mock_rec]
        mocks["session_repository"].query_sessions.assert_called_once_with(flt)

        res_count = svc.count_query_sessions(flt)
        assert res_count == 1
        mocks["session_repository"].count_query_sessions.assert_called_once_with(flt)

        res_analytics = svc.get_aggregate_analytics(flt)
        assert res_analytics == mock_analytics
        mocks["session_repository"].get_aggregate_analytics.assert_called_once_with(filter=flt)

        res_engagement = svc.get_person_engagement_stats("11111111-1111-1111-1111-111111111111")
        assert res_engagement == mock_engagement
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

    def test_step_once_returns_application_contracts_only(self) -> None:
        """Verify step_once executes frame step and returns application contracts."""
        svc, mocks = _mk_service_harness()
        mock_frame_res = MagicMock()
        mock_events = [PersonEnteredEvent(timestamp=1.0, frame_id=1, track_id=1)]
        mocks["runtime"].step_once.return_value = (mock_frame_res, mock_events)

        ctx, events = svc.step_once()

        mocks["runtime"].step_once.assert_called_once()
        assert ctx.session_id == "sess-1"
        assert len(events) == 1
        assert isinstance(events[0], PersonEnteredEvent)


class TestServiceGalleryAndListeners:
    def test_reload_gallery_syncs_with_repository(self) -> None:
        """Verify reload_gallery extracts records from repository and replaces gallery."""
        svc, mocks = _mk_service_harness()

        valid_pid = str(uuid.uuid4())
        valid_eid = str(uuid.uuid4())
        # Seed repository mock with 1 active person and 1 embedding
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
        """Verify context listener registration and unregistration."""
        svc, mocks = _mk_service_harness()
        cb = MagicMock()
        mocks["coordinator"].add_context_listener.return_value = 101
        mocks["coordinator"].remove_context_listener.return_value = True

        lid = svc.add_context_listener(cb)
        assert lid == 101
        mocks["coordinator"].add_context_listener.assert_called_once_with(cb)

        res = svc.remove_context_listener(lid)
        assert res is True
        mocks["coordinator"].remove_context_listener.assert_called_once_with(101)

    def test_event_listener_delegation(self) -> None:
        """Verify event listener registration and unregistration on EventBus."""
        svc, mocks = _mk_service_harness()
        cb = MagicMock()
        mocks["event_bus"].subscribe.return_value = 202
        mocks["event_bus"].unsubscribe.return_value = True

        sub_id = svc.add_event_listener(PersonEnteredEvent, cb)
        assert sub_id == 202
        mocks["event_bus"].subscribe.assert_called_once_with(PersonEnteredEvent, cb)

        res = svc.remove_event_listener(sub_id)
        assert res is True
        mocks["event_bus"].unsubscribe.assert_called_once_with(202)

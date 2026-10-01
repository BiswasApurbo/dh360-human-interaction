"""Integration tests for EdgeInteractionRuntime and InteractionCoordinator end-to-end pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock

from dh360_hi.application.coordinator import InteractionCoordinator
from dh360_hi.backend.database import DatabaseManager
from dh360_hi.backend.models import ConsentStatus, PersonStatus
from dh360_hi.backend.person_service import PersonService
from dh360_hi.backend.repository import SQLitePersonRepository
from dh360_hi.contracts.interaction import InteractionState
from dh360_hi.edge.event_bus import EventBus
from dh360_hi.edge.runtime import EdgeInteractionRuntime, EdgeRuntimeConfig
from dh360_hi.edge.session_manager import SessionConfig, SessionManager
from dh360_hi.vision.frame import SyntheticFrameSource
from dh360_hi.vision.identity_state import IdentityState, TrackIdentityState
from dh360_hi.vision.presence_debouncer import PresenceEvent, PresenceTransition
from dh360_hi.vision.target_selector import TargetCandidate, TargetSelectionResult
from dh360_hi.vision.two_rate_pipeline import TwoRateFrameResult, TwoRatePerceptionPipeline

# ---------------------------------------------------------------------------
# Scripted Perception Generators
# ---------------------------------------------------------------------------


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
    total_frames: int = 5,
) -> TwoRateFrameResult:
    transitions: list[PresenceTransition] = []
    if frame_id == 1:
        transitions.append(
            PresenceTransition(
                event=PresenceEvent.PERSON_ENTERED,
                track_id=1,
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
                track_id=1,
                person_id=person_id,
                display_name=display_name,
                frame_id=frame_id,
                timestamp=float(frame_id) * 0.033,
            )
        )

    id_states: dict[int, TrackIdentityState] = {}
    if frame_id < total_frames:
        id_states[1] = TrackIdentityState(
            track_id=1,
            state=IdentityState.IDENTIFIED if person_id else IdentityState.COLLECTING,
            current_person_id=person_id,
            current_display_name=display_name,
        )

    primary_track = 1 if frame_id < total_frames else None
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


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


class TestInteractionPipelineIntegration:
    def test_anonymous_interaction_lifecycle(self) -> None:
        """Test anonymous person interaction from entry to departure."""
        source = SyntheticFrameSource(num_frames=5, fps=100.0)

        pipeline = MagicMock(spec=TwoRatePerceptionPipeline)
        pipeline.worker.is_running = True
        pipeline.process_frame.side_effect = lambda f: _mk_scripted_frame_result(
            frame_id=f.frame_id + 1,
            person_id=None,
            display_name=None,
            total_frames=5,
        )

        bus = EventBus()
        session_cfg = SessionConfig(require_confirmed_presence=True)
        manager = SessionManager(event_bus=bus, config=session_cfg)
        coordinator = InteractionCoordinator(bus)

        runtime = EdgeInteractionRuntime(
            frame_source=source,
            pipeline=pipeline,
            session_manager=manager,
            event_bus=bus,
            config=EdgeRuntimeConfig(target_fps=1000.0),
        )

        # Execute stream to completion
        runtime.start(async_thread=False)

        # Context returned to IDLE
        ctx = coordinator.get_context()
        assert ctx.state == InteractionState.IDLE
        assert ctx.session_id is None

        # Summary verified
        summary = coordinator.get_last_summary()
        assert summary is not None
        assert summary.person_id is None
        assert not summary.identity_was_known
        assert summary.duration_seconds > 0.0

        coordinator.close()

    def test_registered_person_interaction_with_profile_enrichment(self) -> None:
        """Test registered user session with PersonService profile enrichment."""
        db = DatabaseManager(":memory:")
        repo = SQLitePersonRepository(db)
        service = PersonService(repo)

        profile = service.create_person(
            display_name="Dr. Jane Doe",
            preferred_name="Jane",
            status=PersonStatus.ACTIVE,
            consent_status=ConsentStatus.GRANTED,
        )
        person_id = profile.person_id

        source = SyntheticFrameSource(num_frames=5, fps=100.0)

        pipeline = MagicMock(spec=TwoRatePerceptionPipeline)
        pipeline.worker.is_running = True
        pipeline.process_frame.side_effect = lambda f: _mk_scripted_frame_result(
            frame_id=f.frame_id + 1,
            person_id=person_id,
            display_name="Dr. Jane Doe",
            total_frames=5,
        )

        bus = EventBus()
        session_cfg = SessionConfig(require_confirmed_presence=True)
        manager = SessionManager(event_bus=bus, config=session_cfg)
        coordinator = InteractionCoordinator(bus, person_service=service)

        runtime = EdgeInteractionRuntime(
            frame_source=source,
            pipeline=pipeline,
            session_manager=manager,
            event_bus=bus,
            config=EdgeRuntimeConfig(target_fps=1000.0),
        )

        # Step first 2 frames and check active context
        runtime.step_once()
        runtime.step_once()

        ctx = coordinator.get_context()
        assert ctx.session_id is not None
        assert ctx.person_id == person_id
        assert ctx.display_name == "Jane"  # Preferred name resolved from PersonProfile!
        assert ctx.person_profile is not None
        assert ctx.person_profile.display_name == "Dr. Jane Doe"
        assert ctx.identity_known

        # Complete stream
        runtime.start(async_thread=False)

        summary = coordinator.get_last_summary()
        assert summary is not None
        assert summary.person_id == person_id
        assert summary.display_name == "Jane"
        assert summary.identity_was_known
        assert summary.person_profile is not None
        assert summary.person_profile.person_id == person_id

        coordinator.close()

    def test_mid_session_recognition_integration(self) -> None:
        """Test interaction starting anonymously and resolving identity mid-stream."""
        db = DatabaseManager(":memory:")
        repo = SQLitePersonRepository(db)
        service = PersonService(repo)

        profile = service.create_person(
            display_name="Alex Rivera",
            preferred_name="Alex",
        )
        person_id = profile.person_id

        # Custom scripted pipeline: frame 1=anonymous, frame 2=identified, frame 4=departure
        def scripted_pipeline(frame) -> TwoRateFrameResult:
            f_id = frame.frame_id + 1
            if f_id == 1:
                return _mk_scripted_frame_result(f_id, person_id=None, total_frames=4)
            elif f_id in (2, 3):
                res = _mk_scripted_frame_result(f_id, person_id=person_id, total_frames=4)
                # simulate async recognition completion on frame 2
                if f_id == 2:
                    res.completed_recognitions.append(
                        MagicMock(track_id=1, person_id=person_id, display_name="Alex Rivera")
                    )
                return res
            else:
                return _mk_scripted_frame_result(f_id, person_id=person_id, total_frames=4)

        source = SyntheticFrameSource(num_frames=4, fps=100.0)
        pipeline = MagicMock(spec=TwoRatePerceptionPipeline)
        pipeline.worker.is_running = True
        pipeline.process_frame.side_effect = scripted_pipeline

        bus = EventBus()
        session_cfg = SessionConfig(require_confirmed_presence=True)
        manager = SessionManager(event_bus=bus, config=session_cfg)
        coordinator = InteractionCoordinator(bus, person_service=service)

        runtime = EdgeInteractionRuntime(
            frame_source=source,
            pipeline=pipeline,
            session_manager=manager,
            event_bus=bus,
            config=EdgeRuntimeConfig(target_fps=1000.0),
        )

        # Frame 1: Anonymous start
        runtime.step_once()
        assert coordinator.get_context().session_id is not None
        assert not coordinator.get_context().identity_known

        # Frame 2: Identified
        runtime.step_once()
        assert coordinator.get_context().identity_known
        assert coordinator.get_context().person_id == person_id
        assert coordinator.get_context().display_name == "Alex"

        # Complete
        runtime.step_once()
        runtime.step_once()

        summary = coordinator.get_last_summary()
        assert summary is not None
        assert summary.identity_was_known
        assert summary.person_id == person_id

        coordinator.close()

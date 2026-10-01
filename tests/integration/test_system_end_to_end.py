"""End-to-end integration tests for the composed Human Interaction subsystem."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from dh360_hi.application.config import HumanInteractionConfig
from dh360_hi.application.factory import create_interaction_service
from dh360_hi.backend.models import ConsentStatus, PersonStatus
from dh360_hi.contracts.interaction import InteractionState
from dh360_hi.edge.runtime import EdgeRuntimeConfig, RuntimeState
from dh360_hi.edge.session_manager import SessionConfig
from dh360_hi.vision.frame import SyntheticFrameSource
from dh360_hi.vision.identity_state import IdentityState, TrackIdentityState
from dh360_hi.vision.presence_debouncer import PresenceEvent, PresenceTransition
from dh360_hi.vision.target_selector import TargetCandidate, TargetSelectionResult
from dh360_hi.vision.two_rate_pipeline import TwoRateFrameResult, TwoRatePerceptionPipeline


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


class TestSystemEndToEnd:
    def test_anonymous_interaction_full_lifecycle(self) -> None:
        """Verify anonymous user session from stream start to summary finalization."""
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

        # Context returns to IDLE on completion
        ctx = service.get_interaction_context()
        assert ctx.state == InteractionState.IDLE
        assert ctx.session_id is None

        # Summary finalized
        summary = service.get_last_summary()
        assert summary is not None
        assert summary.person_id is None
        assert not summary.identity_was_known
        assert summary.duration_seconds > 0.0

        # Metrics updated
        metrics = service.get_runtime_metrics()
        assert metrics.frames_processed == 5

        service.stop()

    def test_registered_person_profile_enrichment(self) -> None:
        """Verify registered person creation, gallery sync, and context profile enrichment."""
        cfg = HumanInteractionConfig(
            runtime=EdgeRuntimeConfig(target_fps=1000.0),
            session=SessionConfig(require_confirmed_presence=True),
        )

        # Create service with in-memory DB
        pipeline = MagicMock(spec=TwoRatePerceptionPipeline)
        pipeline.worker.is_running = True

        service = create_interaction_service(
            config=cfg,
            pipeline=pipeline,
        )

        # Register person in DB
        profile = service.person_service.create_person(
            display_name="Dr. Jane Smith",
            preferred_name="Jane",
            status=PersonStatus.ACTIVE,
            consent_status=ConsentStatus.GRANTED,
        )
        person_id = profile.person_id

        # Query person via service API
        queried = service.get_person(person_id)
        assert queried is not None
        assert queried.preferred_name == "Jane"

        # List persons via service API
        all_persons = service.list_persons()
        assert len(all_persons) == 1

        # Configure scripted pipeline returning Jane
        pipeline.process_frame.side_effect = lambda f: _mk_scripted_frame_result(
            frame_id=f.frame_id + 1,
            person_id=person_id,
            display_name="Dr. Jane Smith",
            total_frames=5,
        )

        # Step 2 frames
        service.step_once()
        service.step_once()

        ctx = service.get_interaction_context()
        assert ctx.session_id is not None
        assert ctx.person_id == person_id
        assert ctx.display_name == "Jane"  # Preferred name resolved
        assert ctx.person_profile is not None
        assert ctx.person_profile.display_name == "Dr. Jane Smith"
        assert ctx.identity_known is True

        service.stop()

    def test_active_stream_shutdown_no_deadlock(self) -> None:
        """Verify clean background execution and prompt shutdown without thread hangs."""
        source = SyntheticFrameSource(num_frames=1000, fps=100.0)

        pipeline = MagicMock(spec=TwoRatePerceptionPipeline)
        pipeline.worker.is_running = True
        pipeline.process_frame.side_effect = lambda f: _mk_scripted_frame_result(
            frame_id=f.frame_id + 1,
            person_id=None,
            display_name=None,
            total_frames=1000,
        )

        cfg = HumanInteractionConfig(
            runtime=EdgeRuntimeConfig(target_fps=200.0),
        )

        service = create_interaction_service(
            config=cfg,
            frame_source=source,
            pipeline=pipeline,
        )

        # Start async thread
        service.start(async_thread=True)
        assert service.is_running is True
        assert service.state == RuntimeState.RUNNING

        # Let it process several frames
        time.sleep(0.05)

        # Stop while actively running
        t0 = time.monotonic()
        service.stop(timeout=2.0)
        stop_duration = time.monotonic() - t0

        assert stop_duration < 1.0  # Quick shutdown
        assert service.state == RuntimeState.STOPPED
        assert service.is_running is False
        assert service.metrics.frames_processed > 0

    def test_factory_requires_detector_or_model_path(self) -> None:
        """Verify create_interaction_service raises ValueError if detector is omitted."""
        cfg = HumanInteractionConfig(
            detector_model_path=None,
        )

        with pytest.raises(ValueError, match="either a 'detector' instance or"):
            create_interaction_service(config=cfg, detector=None)

"""Integration tests for EdgeInteractionRuntime end-to-end pipeline execution."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

from dh360_hi.contracts.events import (
    BaseEvent,
    PersonEnteredEvent,
    PersonIdentifiedEvent,
    PersonLeftEvent,
    SessionEndedEvent,
    SessionStartedEvent,
)
from dh360_hi.edge.event_bus import EventBus
from dh360_hi.edge.runtime import (
    EdgeInteractionRuntime,
    EdgeRuntimeConfig,
    RuntimeState,
)
from dh360_hi.edge.session_manager import SessionConfig, SessionManager
from dh360_hi.vision.frame import SyntheticFrameSource
from dh360_hi.vision.identity_state import IdentityState, TrackIdentityState
from dh360_hi.vision.presence_debouncer import PresenceEvent, PresenceTransition
from dh360_hi.vision.target_selector import TargetCandidate, TargetSelectionResult
from dh360_hi.vision.two_rate_pipeline import TwoRateFrameResult, TwoRatePerceptionPipeline

# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------


def _mk_target_result(primary_track_id: int | None = 1, frame_id: int = 1) -> TargetSelectionResult:
    cand = None
    if primary_track_id is not None:
        cand = TargetCandidate(
            track_id=primary_track_id,
            score=1.0,
            centrality_score=1.0,
            size_score=1.0,
            identity_state_score=1.0,
            dwell_score=1.0,
            is_identified=True,
            identity_state=IdentityState.IDENTIFIED,
            person_id="user_123",
            display_name="Alice",
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


def _mk_scripted_frame_result(frame_id: int) -> TwoRateFrameResult:
    # Frame 1: PERSON_ENTERED
    # Frame 2-4: ACTIVE
    # Frame 5: PERSON_LEFT
    transitions: list[PresenceTransition] = []
    if frame_id == 1:
        transitions.append(
            PresenceTransition(
                event=PresenceEvent.PERSON_ENTERED,
                track_id=1,
                person_id="user_123",
                display_name="Alice",
                frame_id=frame_id,
                timestamp=float(frame_id) * 0.033,
            )
        )
    elif frame_id == 5:
        transitions.append(
            PresenceTransition(
                event=PresenceEvent.PERSON_LEFT,
                track_id=1,
                person_id="user_123",
                display_name="Alice",
                frame_id=frame_id,
                timestamp=float(frame_id) * 0.033,
            )
        )

    id_states: dict[int, TrackIdentityState] = {}
    if frame_id < 5:
        id_states[1] = TrackIdentityState(
            track_id=1,
            state=IdentityState.IDENTIFIED,
            current_person_id="user_123",
            current_display_name="Alice",
        )

    primary_track = 1 if frame_id < 5 else None
    return TwoRateFrameResult(
        frame_id=frame_id,
        timestamp=float(frame_id) * 0.033,
        detections=[],
        tracks=[],
        quality_results={},
        identity_states=id_states,
        target_result=_mk_target_result(primary_track, frame_id),
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


class TestEdgeRuntimeEndToEndIntegration:
    def test_synchronous_stream_processes_all_frames_and_emits_events(self) -> None:
        source = SyntheticFrameSource(num_frames=5, fps=100.0)

        pipeline = MagicMock(spec=TwoRatePerceptionPipeline)
        pipeline.worker.is_running = True
        pipeline.process_frame.side_effect = lambda f: _mk_scripted_frame_result(f.frame_id + 1)

        bus = EventBus()
        events_received: list[BaseEvent] = []
        for evt_cls in [
            PersonEnteredEvent,
            SessionStartedEvent,
            PersonIdentifiedEvent,
            PersonLeftEvent,
            SessionEndedEvent,
        ]:
            bus.subscribe(evt_cls, events_received.append)

        cfg = SessionConfig(require_confirmed_presence=True)
        manager = SessionManager(event_bus=bus, config=cfg)
        runtime = EdgeInteractionRuntime(
            frame_source=source,
            pipeline=pipeline,
            session_manager=manager,
            event_bus=bus,
            config=EdgeRuntimeConfig(target_fps=1000.0),
        )

        # Run synchronously to completion
        runtime.start(async_thread=False)

        assert runtime.state == RuntimeState.STOPPED
        assert runtime.metrics.frames_processed == 5
        assert len(events_received) >= 4

        # Verify event sequence
        types = [type(e) for e in events_received]
        assert PersonEnteredEvent in types
        assert SessionStartedEvent in types
        assert PersonIdentifiedEvent in types
        assert PersonLeftEvent in types
        assert SessionEndedEvent in types

    def test_asynchronous_runtime_starts_and_stops_cleanly(self) -> None:
        source = SyntheticFrameSource(num_frames=20, fps=100.0)

        pipeline = MagicMock(spec=TwoRatePerceptionPipeline)
        pipeline.worker.is_running = True
        pipeline.process_frame.side_effect = lambda f: _mk_scripted_frame_result(f.frame_id + 1)

        bus = EventBus()
        manager = SessionManager(event_bus=bus)
        runtime = EdgeInteractionRuntime(
            frame_source=source,
            pipeline=pipeline,
            session_manager=manager,
            event_bus=bus,
            config=EdgeRuntimeConfig(target_fps=100.0),
        )

        runtime.start(async_thread=True)
        assert runtime.state == RuntimeState.RUNNING

        # Let it run for a short duration
        time.sleep(0.05)

        runtime.stop(timeout=1.0)
        assert runtime.state == RuntimeState.STOPPED
        assert runtime.metrics.frames_processed > 0

    def test_runtime_step_once_matches_direct_session_manager(self) -> None:
        source = SyntheticFrameSource(num_frames=2)
        source.open()
        frame1 = source.read()
        assert frame1 is not None

        pipeline = MagicMock(spec=TwoRatePerceptionPipeline)
        pipeline.worker.is_running = True
        frame_res = _mk_scripted_frame_result(1)
        pipeline.process_frame.return_value = frame_res

        bus1 = EventBus()
        mgr1 = SessionManager(event_bus=bus1)
        expected_events = mgr1.process_frame_result(frame_res)

        bus2 = EventBus()
        mgr2 = SessionManager(event_bus=bus2)
        source2 = SyntheticFrameSource(num_frames=2)
        runtime = EdgeInteractionRuntime(source2, pipeline, mgr2, bus2)

        res, actual_events = runtime.step_once()
        assert res == frame_res
        assert len(actual_events) == len(expected_events)
        for exp, act in zip(expected_events, actual_events, strict=True):
            assert type(exp) is type(act)
            assert getattr(exp, "track_id", None) == getattr(act, "track_id", None)

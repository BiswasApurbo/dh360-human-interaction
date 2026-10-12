"""Unit tests for EdgeInteractionRuntime lifecycle, step_once, error handling, and metrics."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from dh360_hi.contracts.events import (
    BaseEvent,
    SessionStartedEvent,
)
from dh360_hi.edge.event_bus import EventBus
from dh360_hi.edge.runtime import (
    EdgeInteractionRuntime,
    EdgeRuntimeConfig,
    RuntimeMetrics,
    RuntimeState,
)
from dh360_hi.edge.session_manager import SessionConfig, SessionManager
from dh360_hi.vision.frame import Frame, FrameSource
from dh360_hi.vision.target_selector import TargetSelectionResult
from dh360_hi.vision.two_rate_pipeline import TwoRateFrameResult, TwoRatePerceptionPipeline

# ---------------------------------------------------------------------------
# Test Stubs / Fixtures
# ---------------------------------------------------------------------------


class StubFrameSource(FrameSource):
    """Controllable frame source stub for unit tests."""

    def __init__(
        self,
        frames: list[Frame | None] | None = None,
        infinite: bool = False,
        delay_s: float = 0.0,
        is_live: bool = False,
    ) -> None:
        self._frames = list(frames or [])
        self._infinite = infinite
        self._delay_s = delay_s
        self._is_live = is_live
        self._index = 0
        self._is_opened = False
        self.open_called = 0
        self.release_called = 0

    @property
    def is_live(self) -> bool:
        return self._is_live

    def open(self) -> None:
        self._is_opened = True
        self.open_called += 1

    def read(self) -> Frame | None:
        if not self._is_opened:
            return None
        if self._delay_s > 0:
            time.sleep(self._delay_s)
        if self._infinite:
            return _mk_test_frame(self._index + 1)
        if self._index >= len(self._frames):
            return None
        frame = self._frames[self._index]
        self._index += 1
        return frame

    def release(self) -> None:
        self._is_opened = False
        self.release_called += 1

    @property
    def is_opened(self) -> bool:
        return self._is_opened

    @property
    def resolution(self) -> tuple[int, int]:
        return (640, 480)

    @property
    def fps(self) -> float:
        return 30.0


def _mk_test_frame(frame_id: int = 1) -> Frame:
    return Frame(
        image=np.zeros((100, 100, 3), dtype=np.uint8),
        frame_id=frame_id,
        timestamp=float(frame_id) * 0.033,
        source_info="test",
    )


def _mk_frame_result(frame_id: int = 1) -> TwoRateFrameResult:
    return TwoRateFrameResult(
        frame_id=frame_id,
        timestamp=float(frame_id) * 0.033,
        detections=[],
        tracks=[],
        quality_results={},
        identity_states={},
        target_result=TargetSelectionResult(
            primary_target=None,
            candidates=[],
            target_changed=False,
            previous_target_track_id=None,
            frame_id=frame_id,
        ),
        presence_transitions=[],
        completed_recognitions=[],
        ambiguous_crossing_track_ids=set(),
        pending_worker_tasks=0,
        frame_latency_ms=1.0,
        total_pipeline_latency_ms=1.0,
    )


# ---------------------------------------------------------------------------
# Configuration Tests
# ---------------------------------------------------------------------------


class TestEdgeRuntimeConfig:
    def test_default_config_valid(self) -> None:
        cfg = EdgeRuntimeConfig()
        assert cfg.target_fps == 30.0
        assert cfg.max_consecutive_frame_errors == 10
        assert cfg.auto_start_worker is True

    def test_invalid_target_fps_raises_error(self) -> None:
        with pytest.raises(ValueError, match="target_fps must be positive"):
            EdgeRuntimeConfig(target_fps=0.0)

        with pytest.raises(ValueError, match="target_fps must be positive"):
            EdgeRuntimeConfig(target_fps=-10.0)

    def test_invalid_max_errors_raises_error(self) -> None:
        with pytest.raises(ValueError, match="max_consecutive_frame_errors must be >= 1"):
            EdgeRuntimeConfig(max_consecutive_frame_errors=0)


# ---------------------------------------------------------------------------
# Lifecycle State Transition Tests
# ---------------------------------------------------------------------------


class TestRuntimeLifecycleTransitions:
    def test_initial_state_is_stopped(self) -> None:
        source = StubFrameSource()
        pipeline = MagicMock(spec=TwoRatePerceptionPipeline)
        pipeline.worker.is_running = False
        manager = SessionManager()
        bus = EventBus()

        runtime = EdgeInteractionRuntime(source, pipeline, manager, bus)
        assert runtime.state == RuntimeState.STOPPED
        assert runtime.metrics.frames_processed == 0

    def test_start_transitions_to_running(self) -> None:
        source = StubFrameSource(infinite=True, delay_s=0.01)
        pipeline = MagicMock(spec=TwoRatePerceptionPipeline)
        pipeline.process_frame.return_value = _mk_frame_result(1)
        manager = SessionManager()
        bus = EventBus()

        runtime = EdgeInteractionRuntime(source, pipeline, manager, bus)
        runtime.start(async_thread=True)
        assert runtime.state == RuntimeState.RUNNING
        assert source.open_called == 1
        assert pipeline.start.called
        runtime.stop()

    def test_pause_and_resume_transitions(self) -> None:
        source = StubFrameSource(infinite=True, delay_s=0.01)
        pipeline = MagicMock(spec=TwoRatePerceptionPipeline)
        manager = SessionManager()
        bus = EventBus()

        runtime = EdgeInteractionRuntime(source, pipeline, manager, bus)
        runtime.start(async_thread=True)

        runtime.pause()
        assert runtime.state == RuntimeState.PAUSED

        runtime.resume()
        assert runtime.state == RuntimeState.RUNNING

        runtime.stop()
        assert runtime.state == RuntimeState.STOPPED

    def test_mutation_barrier_waits_for_inflight_frame_and_restores_state(self) -> None:
        """A mutation barrier does not return until frame, session, and event work finish."""
        source = StubFrameSource(infinite=True)
        pipeline = MagicMock(spec=TwoRatePerceptionPipeline)
        frame_started = threading.Event()
        release_frame = threading.Event()

        def block_frame(_frame: Frame) -> TwoRateFrameResult:
            frame_started.set()
            if not release_frame.wait(timeout=2.0):
                raise TimeoutError("test frame release timed out")
            return _mk_frame_result(1)

        pipeline.process_frame.side_effect = block_frame
        manager = SessionManager()
        runtime = EdgeInteractionRuntime(
            source,
            pipeline,
            manager,
            EventBus(),
            EdgeRuntimeConfig(target_fps=1000.0),
        )
        runtime.start(async_thread=True)
        assert frame_started.wait(timeout=1.0)

        def release_after_wait() -> None:
            deadline = time.monotonic() + 1.0
            while runtime.state != RuntimeState.PAUSED and time.monotonic() < deadline:
                time.sleep(0.001)
            assert runtime.state == RuntimeState.PAUSED
            time.sleep(0.05)
            release_frame.set()

        release_thread = threading.Thread(target=release_after_wait)
        release_thread.start()
        prior_state = runtime.begin_mutation_barrier(timeout=1.0)

        try:
            assert prior_state == RuntimeState.RUNNING
            assert runtime.state == RuntimeState.PAUSED
            runtime.end_mutation_barrier(prior_state)
            assert runtime.state == RuntimeState.RUNNING
        finally:
            release_frame.set()
            release_thread.join(timeout=1.0)
            runtime.stop(timeout=1.0)

    def test_mutation_barrier_timeout_restores_running_state(self) -> None:
        """A frame-boundary timeout releases the barrier without mutating runtime state."""
        source = StubFrameSource(infinite=True)
        pipeline = MagicMock(spec=TwoRatePerceptionPipeline)
        frame_started = threading.Event()
        release_frame = threading.Event()

        def block_frame(_frame: Frame) -> TwoRateFrameResult:
            frame_started.set()
            release_frame.wait(timeout=2.0)
            return _mk_frame_result(1)

        pipeline.process_frame.side_effect = block_frame
        runtime = EdgeInteractionRuntime(
            source,
            pipeline,
            SessionManager(),
            EventBus(),
            EdgeRuntimeConfig(target_fps=1000.0),
        )
        runtime.start(async_thread=True)
        try:
            assert frame_started.wait(timeout=1.0)
            assert runtime.begin_mutation_barrier(timeout=0.01) is None
            assert runtime.state == RuntimeState.RUNNING
        finally:
            release_frame.set()
            runtime.stop(timeout=1.0)

    def test_mutation_barrier_preserves_preexisting_paused_state(self) -> None:
        """A successful mutation barrier does not resume a runtime paused by its operator."""
        runtime = EdgeInteractionRuntime(
            StubFrameSource(infinite=True),
            MagicMock(spec=TwoRatePerceptionPipeline),
            SessionManager(),
            EventBus(),
        )
        runtime.start(async_thread=True)
        runtime.pause()

        prior_state = runtime.begin_mutation_barrier(timeout=1.0)
        assert prior_state == RuntimeState.PAUSED
        runtime.end_mutation_barrier(prior_state)

        assert runtime.state == RuntimeState.PAUSED
        runtime.resume()
        runtime.stop(timeout=1.0)

    def test_stop_is_idempotent(self) -> None:
        source = StubFrameSource(infinite=True, delay_s=0.01)
        pipeline = MagicMock(spec=TwoRatePerceptionPipeline)
        manager = SessionManager()
        bus = EventBus()

        runtime = EdgeInteractionRuntime(source, pipeline, manager, bus)
        # Calling stop before start
        runtime.stop()
        assert runtime.state == RuntimeState.STOPPED

        # Starting and double-stopping
        runtime.start(async_thread=True)
        runtime.stop()
        assert runtime.state == RuntimeState.STOPPED
        runtime.stop()
        assert runtime.state == RuntimeState.STOPPED

    def test_invalid_lifecycle_transitions_raise_error(self) -> None:
        source = StubFrameSource(infinite=True, delay_s=0.01)
        pipeline = MagicMock(spec=TwoRatePerceptionPipeline)
        manager = SessionManager()
        bus = EventBus()

        runtime = EdgeInteractionRuntime(source, pipeline, manager, bus)

        # Pause while stopped
        with pytest.raises(RuntimeError, match="Cannot pause runtime from state stopped"):
            runtime.pause()

        # Resume while stopped
        with pytest.raises(RuntimeError, match="Cannot resume runtime from state stopped"):
            runtime.resume()

        # Start twice
        runtime.start(async_thread=True)
        with pytest.raises(RuntimeError, match="Cannot start runtime: already in state running"):
            runtime.start(async_thread=True)

        runtime.stop()


# ---------------------------------------------------------------------------
# step_once() Deterministic Behavior Tests
# ---------------------------------------------------------------------------


class TestRuntimeStepOnce:
    def test_step_once_processes_single_frame(self) -> None:
        frame = _mk_test_frame(1)
        source = StubFrameSource([frame])
        pipeline = MagicMock(spec=TwoRatePerceptionPipeline)
        pipeline.worker.is_running = True
        frame_res = _mk_frame_result(1)
        pipeline.process_frame.return_value = frame_res

        bus = EventBus()
        cfg = SessionConfig(require_confirmed_presence=False)
        manager = SessionManager(event_bus=bus, config=cfg)

        runtime = EdgeInteractionRuntime(source, pipeline, manager, bus)
        res, events = runtime.step_once()

        assert res == frame_res
        assert pipeline.process_frame.called
        assert runtime.metrics.frames_processed == 1
        assert runtime.metrics.average_latency_ms >= 0.0

    def test_step_once_publishes_events_to_bus(self) -> None:
        frame = _mk_test_frame(1)
        source = StubFrameSource([frame])
        pipeline = MagicMock(spec=TwoRatePerceptionPipeline)
        pipeline.worker.is_running = True
        pipeline.process_frame.return_value = _mk_frame_result(1)

        bus = EventBus()
        received: list[BaseEvent] = []
        bus.subscribe(SessionStartedEvent, received.append)

        # Manager without direct bus wiring -> runtime must publish
        cfg = SessionConfig(require_confirmed_presence=False)
        manager = SessionManager(event_bus=None, config=cfg)
        # Ensure target is primary so session starts
        frame_res = _mk_frame_result(1)
        pipeline.process_frame.return_value = frame_res

        runtime = EdgeInteractionRuntime(source, pipeline, manager, bus)
        _, events = runtime.step_once()

        # Events returned must match events delivered to bus
        assert len(events) == len(received)

    def test_step_once_when_paused_raises_error(self) -> None:
        source = StubFrameSource()
        pipeline = MagicMock(spec=TwoRatePerceptionPipeline)
        manager = SessionManager()
        bus = EventBus()

        runtime = EdgeInteractionRuntime(source, pipeline, manager, bus)
        runtime.start(async_thread=True)
        runtime.pause()

        pattern = "Cannot execute step_once\\(\\) while runtime is PAUSED"
        with pytest.raises(RuntimeError, match=pattern):
            runtime.step_once()

        runtime.stop()

    def test_step_once_at_end_of_stream_raises_error(self) -> None:
        source = StubFrameSource([])  # No frames
        pipeline = MagicMock(spec=TwoRatePerceptionPipeline)
        pipeline.worker.is_running = True
        manager = SessionManager()
        bus = EventBus()

        runtime = EdgeInteractionRuntime(source, pipeline, manager, bus)
        with pytest.raises(RuntimeError, match="FrameSource returned None"):
            runtime.step_once()


# ---------------------------------------------------------------------------
# Metrics Telemetry Tests
# ---------------------------------------------------------------------------


class TestRuntimeMetrics:
    def test_metrics_initial_state(self) -> None:
        metrics = RuntimeMetrics()
        assert metrics.frames_processed == 0
        assert metrics.current_fps == 0.0
        assert metrics.average_latency_ms == 0.0
        assert metrics.last_error is None
        assert metrics.uptime_seconds == 0.0

    def test_metrics_updated_after_step(self) -> None:
        source = StubFrameSource([_mk_test_frame(1), _mk_test_frame(2)])
        pipeline = MagicMock(spec=TwoRatePerceptionPipeline)
        pipeline.worker.is_running = True
        pipeline.process_frame.return_value = _mk_frame_result(1)
        manager = SessionManager()
        bus = EventBus()

        runtime = EdgeInteractionRuntime(source, pipeline, manager, bus)
        runtime.step_once()
        runtime.step_once()

        m = runtime.metrics
        assert m.frames_processed == 2
        assert m.average_latency_ms >= 0.0
        assert m.current_fps >= 0.0


# ---------------------------------------------------------------------------
# Error Handling & Consecutive Error Tolerance Tests
# ---------------------------------------------------------------------------


class TestRuntimeErrorHandling:
    def test_transient_error_isolated_in_loop(self) -> None:
        # Source returns 1 bad frame then stops
        source = MagicMock(spec=FrameSource)
        source.is_opened = True
        source.read.side_effect = [RuntimeError("transient sensor glitch"), None]

        pipeline = MagicMock(spec=TwoRatePerceptionPipeline)
        manager = SessionManager()
        bus = EventBus()

        cfg = EdgeRuntimeConfig(max_consecutive_frame_errors=5, target_fps=100.0)
        runtime = EdgeInteractionRuntime(source, pipeline, manager, bus, config=cfg)

        # Run synchronously
        runtime.start(async_thread=False)

        assert runtime.metrics.last_error is not None
        assert "RuntimeError: transient sensor glitch" in runtime.metrics.last_error
        # Because error count (1) <= max (5), it did not enter ERROR state, stopped on EOF
        assert runtime.state == RuntimeState.STOPPED

    def test_consecutive_errors_exceeding_threshold_enters_error_state(self) -> None:
        source = MagicMock(spec=FrameSource)
        source.is_opened = True
        source.read.side_effect = RuntimeError("persistent hardware failure")

        pipeline = MagicMock(spec=TwoRatePerceptionPipeline)
        manager = SessionManager()
        bus = EventBus()

        cfg = EdgeRuntimeConfig(max_consecutive_frame_errors=3, target_fps=100.0)
        runtime = EdgeInteractionRuntime(source, pipeline, manager, bus, config=cfg)

        # Run synchronously
        runtime.start(async_thread=False)

        assert runtime.state == RuntimeState.ERROR
        assert runtime.metrics.last_error is not None

        # Clean stop from ERROR state
        runtime.stop()
        assert runtime.state == RuntimeState.STOPPED


# ---------------------------------------------------------------------------
# Dependency Ownership & Shutdown Behavior Tests
# ---------------------------------------------------------------------------


class TestRuntimeDependencyOwnership:
    def test_stop_shuts_down_pipeline_and_releases_source(self) -> None:
        source = StubFrameSource([_mk_test_frame(1)])
        pipeline = MagicMock(spec=TwoRatePerceptionPipeline)
        manager = SessionManager()
        bus = EventBus()

        runtime = EdgeInteractionRuntime(source, pipeline, manager, bus)
        runtime.start(async_thread=True)
        runtime.stop(timeout=1.0)

        assert pipeline.stop.called
        assert source.release_called == 1
        assert runtime.state == RuntimeState.STOPPED

    def test_auto_start_worker_disabled(self) -> None:
        source = StubFrameSource()
        pipeline = MagicMock(spec=TwoRatePerceptionPipeline)
        manager = SessionManager()
        bus = EventBus()

        cfg = EdgeRuntimeConfig(auto_start_worker=False)
        runtime = EdgeInteractionRuntime(source, pipeline, manager, bus, config=cfg)
        runtime.start(async_thread=True)

        assert not pipeline.start.called
        runtime.stop()


# ---------------------------------------------------------------------------
# is_running Property Tests
# ---------------------------------------------------------------------------


class TestRuntimeIsRunningProperty:
    def test_is_running_reflects_lifecycle(self) -> None:
        source = StubFrameSource(infinite=True, delay_s=0.01)
        pipeline = MagicMock(spec=TwoRatePerceptionPipeline)
        manager = SessionManager()
        bus = EventBus()

        runtime = EdgeInteractionRuntime(source, pipeline, manager, bus)
        assert runtime.is_running is False

        runtime.start(async_thread=True)
        assert runtime.is_running is True

        runtime.pause()
        assert runtime.is_running is False

        runtime.resume()
        assert runtime.is_running is True

        runtime.stop()
        assert runtime.is_running is False


# ---------------------------------------------------------------------------
# Transient Camera Error Recovery Tests
# ---------------------------------------------------------------------------


class TestRuntimeTransientCameraRecovery:
    def test_transient_camera_read_failure_recovers_when_live(self) -> None:
        # Live source returns None (transient glitch), then a valid frame, then another frame
        f1 = _mk_test_frame(1)
        f2 = _mk_test_frame(2)
        source = StubFrameSource(frames=[None, f1, f2], is_live=True)
        pipeline = MagicMock(spec=TwoRatePerceptionPipeline)
        pipeline.process_frame.return_value = _mk_frame_result(1)
        manager = SessionManager()
        bus = EventBus()

        cfg = EdgeRuntimeConfig(max_consecutive_frame_errors=5, target_fps=100.0)
        runtime = EdgeInteractionRuntime(source, pipeline, manager, bus, config=cfg)
        runtime.start(async_thread=True)

        # Wait until both valid frames are processed
        start = time.monotonic()
        while runtime.metrics.frames_processed < 2 and time.monotonic() - start < 1.0:
            time.sleep(0.01)

        assert runtime.metrics.frames_processed >= 2
        assert runtime.state == RuntimeState.RUNNING
        runtime.stop()
        assert runtime.state == RuntimeState.STOPPED

    def test_repeated_transient_camera_failures_enter_error_state(self) -> None:
        # Live source repeatedly returns None exceeding max_consecutive_frame_errors
        source = StubFrameSource(frames=[None, None, None, None, None, None], is_live=True)
        pipeline = MagicMock(spec=TwoRatePerceptionPipeline)
        manager = SessionManager()
        bus = EventBus()

        cfg = EdgeRuntimeConfig(max_consecutive_frame_errors=3, target_fps=100.0)
        runtime = EdgeInteractionRuntime(source, pipeline, manager, bus, config=cfg)
        runtime.start(async_thread=True)

        start = time.monotonic()
        while runtime.state != RuntimeState.ERROR and time.monotonic() - start < 1.0:
            time.sleep(0.01)

        assert runtime.state == RuntimeState.ERROR
        assert runtime.is_running is False
        assert "Camera capture source returned None" in (runtime.metrics.last_error or "")
        runtime.stop()
        assert runtime.state == RuntimeState.STOPPED

    def test_finite_source_stops_at_eof(self) -> None:
        # Non-live source returns frames then EOF -> transitions to STOPPED cleanly
        f1 = _mk_test_frame(1)
        source = StubFrameSource(frames=[f1, None], is_live=False)
        pipeline = MagicMock(spec=TwoRatePerceptionPipeline)
        pipeline.process_frame.return_value = _mk_frame_result(1)
        manager = SessionManager()
        bus = EventBus()

        cfg = EdgeRuntimeConfig(target_fps=100.0)
        runtime = EdgeInteractionRuntime(source, pipeline, manager, bus, config=cfg)
        runtime.start(async_thread=True)

        start = time.monotonic()
        while runtime.state == RuntimeState.RUNNING and time.monotonic() - start < 1.0:
            time.sleep(0.01)

        assert runtime.state == RuntimeState.STOPPED
        assert runtime.is_running is False
        assert runtime.metrics.frames_processed == 1

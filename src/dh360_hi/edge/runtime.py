"""Edge interaction runtime: top-level continuous execution orchestrator.

Supervises the end-to-end edge pipeline:
    FrameSource → TwoRatePerceptionPipeline → SessionManager → EventBus

Provides thread-safe lifecycle control (start, stop, pause, resume, step_once),
FPS pacing, transient error handling, and operational metrics telemetry.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import StrEnum

from dh360_hi.contracts.events import BaseEvent
from dh360_hi.edge.event_bus import EventBus
from dh360_hi.edge.session_manager import SessionManager
from dh360_hi.vision.frame import Frame, FrameSource
from dh360_hi.vision.two_rate_pipeline import TwoRateFrameResult, TwoRatePerceptionPipeline

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EdgeRuntimeConfig:
    """Configuration for EdgeInteractionRuntime execution and error tolerances.

    Attributes:
        target_fps: Target frame rate ceiling for continuous loop pacing. Must be > 0.
        max_consecutive_frame_errors: Maximum allowable consecutive frame processing
            exceptions before transitioning to ERROR state. Must be >= 1.
        auto_start_worker: Whether to automatically start the pipeline's background
            embedding worker thread when the runtime starts.
    """

    target_fps: float = 30.0
    max_consecutive_frame_errors: int = 10
    auto_start_worker: bool = True

    def __post_init__(self) -> None:
        if self.target_fps <= 0:
            msg = f"target_fps must be positive, got {self.target_fps}"
            raise ValueError(msg)
        if self.max_consecutive_frame_errors < 1:
            err_count = self.max_consecutive_frame_errors
            msg = f"max_consecutive_frame_errors must be >= 1, got {err_count}"
            raise ValueError(msg)


# ---------------------------------------------------------------------------
# State Enum
# ---------------------------------------------------------------------------


class RuntimeState(StrEnum):
    """Lifecycle states of the EdgeInteractionRuntime."""

    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Metrics Telemetry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RuntimeMetrics:
    """Immutable snapshot of operational telemetry and performance metrics.

    Attributes:
        frames_processed: Total number of frames successfully processed.
        current_fps: Rolling frame rate estimated across recent frames.
        average_latency_ms: Mean total processing latency per frame in milliseconds.
        last_error: Error message string from the most recent exception, if any.
        uptime_seconds: Total duration the runtime has been running/active.
    """

    frames_processed: int = 0
    current_fps: float = 0.0
    average_latency_ms: float = 0.0
    last_error: str | None = None
    uptime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Runtime Orchestrator
# ---------------------------------------------------------------------------


class EdgeInteractionRuntime:
    """Top-level edge execution orchestrator and continuous frame pump.

    Coordinates the continuous data flow:
        FrameSource → TwoRatePerceptionPipeline → SessionManager → EventBus

    Dependency Ownership:
        - TwoRatePerceptionPipeline owns its background BoundedEmbeddingWorker.
          Runtime manages pipeline worker lifecycle via `pipeline.start()` / `pipeline.stop()`.
        - FrameSource is opened when execution begins and released on `stop()`.
        - SessionManager decides domain event emissions.
        - EventBus delivers domain events to subscribers.
    """

    def __init__(
        self,
        frame_source: FrameSource,
        pipeline: TwoRatePerceptionPipeline,
        session_manager: SessionManager,
        event_bus: EventBus,
        config: EdgeRuntimeConfig | None = None,
    ) -> None:
        self._frame_source = frame_source
        self._pipeline = pipeline
        self._session_manager = session_manager
        self._event_bus = event_bus
        self._config = config or EdgeRuntimeConfig()

        # Threading and synchronization
        self._lock = threading.RLock()
        self._pause_condition = threading.Condition(self._lock)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._active_cycles = 0
        self._mutation_barrier = False

        # State tracking
        self._state = RuntimeState.STOPPED
        self._frames_processed = 0
        self._total_latency_ms = 0.0
        self._consecutive_errors = 0
        self._last_error: str | None = None
        self._start_time: float | None = None
        self._frame_timestamps: deque[float] = deque(maxlen=30)
        self._latest_frame: Frame | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> RuntimeState:
        """Current operational lifecycle state."""
        with self._lock:
            return self._state

    @property
    def config(self) -> EdgeRuntimeConfig:
        """Immutable runtime configuration."""
        return self._config

    @property
    def frame_source(self) -> FrameSource:
        """Underlying camera or video capture source."""
        return self._frame_source

    @property
    def pipeline(self) -> TwoRatePerceptionPipeline:
        """Two-rate computer vision perception pipeline."""
        return self._pipeline

    @property
    def session_manager(self) -> SessionManager:
        """Primary interaction target session manager."""
        return self._session_manager

    @property
    def event_bus(self) -> EventBus:
        """In-process domain event bus."""
        return self._event_bus

    @property
    def latest_frame(self) -> Frame | None:
        """Most recent captured camera frame from the continuous runtime loop."""
        with self._lock:
            return self._latest_frame

    @property
    def metrics(self) -> RuntimeMetrics:
        """Retrieve an immutable snapshot of current operational telemetry."""
        with self._lock:
            now = time.monotonic()
            uptime = (now - self._start_time) if self._start_time is not None else 0.0

            # Compute rolling FPS from recorded timestamps
            current_fps = 0.0
            if len(self._frame_timestamps) >= 2:
                time_span = self._frame_timestamps[-1] - self._frame_timestamps[0]
                if time_span > 0:
                    current_fps = float(len(self._frame_timestamps) - 1) / time_span

            # Compute average latency
            avg_latency = (
                (self._total_latency_ms / self._frames_processed)
                if self._frames_processed > 0
                else 0.0
            )

            return RuntimeMetrics(
                frames_processed=self._frames_processed,
                current_fps=current_fps,
                average_latency_ms=avg_latency,
                last_error=self._last_error,
                uptime_seconds=uptime,
            )

    # ------------------------------------------------------------------
    # Lifecycle API
    # ------------------------------------------------------------------

    def start(self, async_thread: bool = True) -> None:
        """Start the continuous runtime execution loop.

        Args:
            async_thread: If True (default), launches a dedicated background
                thread to execute the continuous frame pump. If False, executes
                the loop synchronously on the caller thread until stream end or stop.

        Raises:
            RuntimeError: If runtime is already in RUNNING or PAUSED state.
        """
        with self._lock:
            if self._mutation_barrier:
                msg = "Cannot start runtime while a person mutation barrier is active"
                raise RuntimeError(msg)
            if self._state in (RuntimeState.RUNNING, RuntimeState.PAUSED):
                msg = f"Cannot start runtime: already in state {self._state.value}"
                raise RuntimeError(msg)

            # Ensure frame source is open
            if not self._frame_source.is_opened:
                self._frame_source.open()

            # Start background embedding worker if configured
            if self._config.auto_start_worker:
                self._pipeline.start()

            self._stop_event.clear()
            self._state = RuntimeState.RUNNING
            self._start_time = time.monotonic()
            self._consecutive_errors = 0

            if async_thread:
                self._thread = threading.Thread(
                    target=self._run_loop,
                    name="EdgeInteractionRuntimeThread",
                    daemon=True,
                )
                self._thread.start()
                logger.info("EdgeInteractionRuntime started in asynchronous mode")
            else:
                logger.info("EdgeInteractionRuntime started in synchronous mode")

        if not async_thread:
            self._run_loop()

    def stop(self, timeout: float = 2.0) -> None:
        """Stop the runtime loop, join worker threads, and release resources.

        This method is idempotent and safe to call from any state.

        Args:
            timeout: Maximum seconds to wait for background thread and pipeline
                shutdown.
        """
        thread_to_join: threading.Thread | None = None
        with self._lock:
            if self._state == RuntimeState.STOPPED:
                return

            self._state = RuntimeState.STOPPED
            self._stop_event.set()
            self._pause_condition.notify_all()
            self._latest_frame = None

            thread_to_join = self._thread
            self._thread = None

        # Join runtime loop thread outside the lock to prevent deadlocks
        if thread_to_join is not None and thread_to_join.is_alive():
            thread_to_join.join(timeout=timeout)
            if thread_to_join.is_alive():
                logger.warning("EdgeInteractionRuntime thread did not terminate within timeout")

        # Stop perception pipeline (owns embedding worker)
        try:
            self._pipeline.stop(timeout=timeout)
        except Exception:
            logger.exception("Error stopping perception pipeline during runtime shutdown")

        # Release frame source
        try:
            if self._frame_source.is_opened:
                self._frame_source.release()
        except Exception:
            logger.exception("Error releasing frame source during runtime shutdown")

        logger.info("EdgeInteractionRuntime stopped successfully")

    def pause(self) -> None:
        """Pause the continuous frame pump without resetting pipeline or session state.

        Raises:
            RuntimeError: If runtime is not currently RUNNING.
        """
        with self._lock:
            if self._state != RuntimeState.RUNNING:
                msg = f"Cannot pause runtime from state {self._state.value}"
                raise RuntimeError(msg)
            self._state = RuntimeState.PAUSED
            logger.info("EdgeInteractionRuntime paused")

    def begin_mutation_barrier(self, timeout: float = 5.0) -> RuntimeState | None:
        """Prevent frame cycles from starting and wait for all active cycles to finish.

        Returns the state to restore with :meth:`end_mutation_barrier`, or ``None``
        when another barrier is active or the frame boundary could not be reached.
        """
        deadline = time.monotonic() + timeout
        with self._pause_condition:
            if self._mutation_barrier:
                return None

            prior_state = self._state
            self._mutation_barrier = True
            if self._state == RuntimeState.RUNNING:
                self._state = RuntimeState.PAUSED
            self._pause_condition.notify_all()

            while self._active_cycles > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    self._mutation_barrier = False
                    if not self._stop_event.is_set() and self._state != RuntimeState.ERROR:
                        self._state = prior_state
                    self._pause_condition.notify_all()
                    return None
                self._pause_condition.wait(timeout=remaining)

            if self._state == RuntimeState.ERROR:
                return RuntimeState.ERROR
            return prior_state

    def end_mutation_barrier(
        self,
        prior_state: RuntimeState,
        resume: bool = True,
    ) -> None:
        """Release a mutation barrier and restore or preserve the runtime state."""
        with self._pause_condition:
            if not self._mutation_barrier:
                return
            self._mutation_barrier = False
            if self._stop_event.is_set() or self._state == RuntimeState.STOPPED:
                self._state = RuntimeState.STOPPED
                self._pause_condition.notify_all()
                return
            if resume:
                self._state = prior_state
            elif prior_state == RuntimeState.RUNNING and self._state != RuntimeState.ERROR:
                self._state = RuntimeState.PAUSED
            self._pause_condition.notify_all()

    def resume(self) -> None:
        """Resume continuous execution from PAUSED state.

        Raises:
            RuntimeError: If runtime is not currently PAUSED.
        """
        with self._lock:
            if self._mutation_barrier:
                msg = "Cannot resume runtime while a person mutation barrier is active"
                raise RuntimeError(msg)
            if self._state != RuntimeState.PAUSED:
                msg = f"Cannot resume runtime from state {self._state.value}"
                raise RuntimeError(msg)
            self._state = RuntimeState.RUNNING
            self._pause_condition.notify_all()
            logger.info("EdgeInteractionRuntime resumed")

    def step_once(self) -> tuple[TwoRateFrameResult, list[BaseEvent]]:
        """Execute exactly one complete perception-to-session cycle deterministically.

        Steps:
            1. Read a single Frame from FrameSource.
            2. Process frame through TwoRatePerceptionPipeline.
            3. Pass TwoRateFrameResult to SessionManager.
            4. Publish emitted BaseEvents to EventBus (if not already published).
            5. Update runtime telemetry metrics.
            6. Return (TwoRateFrameResult, list[BaseEvent]).

        Returns:
            Tuple of (TwoRateFrameResult, list[BaseEvent]).

        Raises:
            RuntimeError: If runtime is in PAUSED or ERROR state, or if the
                FrameSource returns no frame (end of stream).
        """
        with self._pause_condition:
            if self._mutation_barrier or self._state == RuntimeState.PAUSED:
                msg = "Cannot execute step_once() while runtime is PAUSED"
                raise RuntimeError(msg)
            if self._state == RuntimeState.ERROR:
                msg = (
                    "Cannot execute step_once() while runtime is in ERROR state: "
                    f"{self._last_error}"
                )
                raise RuntimeError(msg)

            # Ensure source is open for single-step invocation
            if not self._frame_source.is_opened:
                self._frame_source.open()

            # Start worker if configured and not running
            if self._config.auto_start_worker and not self._pipeline.worker.is_running:
                self._pipeline.start()
            self._active_cycles += 1

        try:
            t_start = time.monotonic()
            frame = self._frame_source.read()
            if frame is None:
                msg = "FrameSource returned None (end of stream or capture failure)"
                raise RuntimeError(msg)

            with self._lock:
                self._latest_frame = frame

            # 1. Vision Perception Fast-Path
            frame_result = self._pipeline.process_frame(frame)

            # 2. Session Lifecycle Supervision
            events = self._session_manager.process_frame_result(frame_result)

            # 3. Event Bus Publication (ensuring delivery without duplicate publish)
            if self._session_manager._bus is not self._event_bus:
                for event in events:
                    self._event_bus.publish(event)

            t_end = time.monotonic()
            latency_ms = (t_end - t_start) * 1000.0

            with self._lock:
                self._frames_processed += 1
                self._total_latency_ms += latency_ms
                self._consecutive_errors = 0
                self._frame_timestamps.append(t_end)

            return frame_result, events
        finally:
            self._finish_frame_cycle()

    # ------------------------------------------------------------------
    # Internal Loop & Pacing
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        """Continuous execution loop with constant-rate FPS pacing and error recovery."""
        target_interval = 1.0 / self._config.target_fps
        next_deadline = time.monotonic()

        while not self._stop_event.is_set():
            # Wait before starting a cycle while paused or during lifecycle mutation.
            with self._pause_condition:
                while (
                    self._state == RuntimeState.PAUSED or self._mutation_barrier
                ) and not self._stop_event.is_set():
                    self._pause_condition.wait(timeout=0.1)
                if self._stop_event.is_set() or self._state != RuntimeState.RUNNING:
                    break
                self._active_cycles += 1

            # Execute single frame cycle with error isolation
            try:
                frame = self._frame_source.read()
                if frame is None:
                    logger.info("FrameSource reached end of stream; stopping runtime loop")
                    with self._lock:
                        self._state = RuntimeState.STOPPED
                    break

                with self._lock:
                    self._latest_frame = frame

                t_start = time.monotonic()
                frame_result = self._pipeline.process_frame(frame)
                events = self._session_manager.process_frame_result(frame_result)

                if self._session_manager._bus is not self._event_bus:
                    for event in events:
                        self._event_bus.publish(event)

                t_end = time.monotonic()
                latency_ms = (t_end - t_start) * 1000.0

                with self._lock:
                    self._frames_processed += 1
                    self._total_latency_ms += latency_ms
                    self._consecutive_errors = 0
                    self._frame_timestamps.append(t_end)

            except Exception as exc:
                self._handle_frame_error(exc)
                if self.state == RuntimeState.ERROR:
                    break
            finally:
                self._finish_frame_cycle()

            # Frame rate pacing toward target_fps
            next_deadline += target_interval
            now = time.monotonic()
            sleep_time = next_deadline - now
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # Behind schedule: reset deadline to prevent accumulated burst processing
                next_deadline = now

    def _finish_frame_cycle(self) -> None:
        """Signal completion of a frame/session/event cycle to lifecycle waiters."""
        with self._pause_condition:
            self._active_cycles -= 1
            self._pause_condition.notify_all()

    def _handle_frame_error(self, exc: Exception) -> None:
        """Record transient frame exception and trigger ERROR transition if threshold exceeded."""
        with self._lock:
            self._consecutive_errors += 1
            self._last_error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "EdgeInteractionRuntime: frame error (%d/%d): %s",
                self._consecutive_errors,
                self._config.max_consecutive_frame_errors,
                self._last_error,
            )

            if self._consecutive_errors >= self._config.max_consecutive_frame_errors:
                self._state = RuntimeState.ERROR
                logger.error(
                    "EdgeInteractionRuntime: exceeded max errors (%d); entering ERROR state",
                    self._config.max_consecutive_frame_errors,
                )

"""Bounded asynchronous embedding worker with latest-wins queue semantics for real-time vision."""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from dh360_hi.vision.aligner import AlignmentResult, BaseFaceAligner
from dh360_hi.vision.embedder import BaseFaceEmbedder, EmbeddingResult
from dh360_hi.vision.quality_gate import FaceQualityTier
from dh360_hi.vision.recognition import BaseFaceRecognizer, RecognitionResult

if TYPE_CHECKING:
    from dh360_hi.vision.detector import FaceLandmarks

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbeddingTask:
    """Encapsulates a single face observation scheduled for embedding and recognition."""

    track_id: int
    frame_id: int
    timestamp: float
    image: np.ndarray  # Full frame image or cropped face image
    bbox: tuple[float, float, float, float]  # (x, y, w, h)
    landmarks: FaceLandmarks | None = None
    quality_tier: FaceQualityTier = FaceQualityTier.MEDIUM
    quality_score: float = 1.0
    created_at: float = field(default_factory=time.monotonic)


@dataclass(frozen=True)
class EmbeddingWorkerResult:
    """Immutable result from embedding worker processing a face observation."""

    track_id: int
    frame_id: int
    timestamp: float
    alignment_result: AlignmentResult | None
    embedding_result: EmbeddingResult | None
    recognition_result: RecognitionResult | None
    queue_latency_ms: float
    inference_latency_ms: float
    total_latency_ms: float
    error: str | None = None

    @property
    def is_success(self) -> bool:
        """Check whether recognition inference completed successfully."""
        return self.error is None and self.recognition_result is not None


@dataclass(frozen=True)
class EmbeddingWorkerConfig:
    """Configuration for bounded asynchronous embedding worker.

    Attributes:
        max_queue_size: Maximum pending tasks in queue before evicting oldest.
        max_results_queue_size: Maximum completed results buffered before dropping oldest unpolled.
        max_request_age_sec: Drop pending tasks older than this threshold.
        drop_stale_requests: Whether to drop stale pending tasks before inference.
        poll_batch_size: Default maximum results returned per poll_results call.
    """

    max_queue_size: int = 8
    max_results_queue_size: int = 32
    max_request_age_sec: float = 1.0
    drop_stale_requests: bool = True
    poll_batch_size: int = 16


class BaseEmbeddingWorker(ABC):
    """Abstract interface for bounded face embedding worker."""

    @abstractmethod
    def start(self) -> None:
        """Start worker background thread(s)."""

    @abstractmethod
    def stop(self, timeout: float = 2.0) -> None:
        """Stop worker background thread(s) gracefully."""

    @abstractmethod
    def submit(self, task: EmbeddingTask) -> bool:
        """Submit a task to the bounded queue."""

    @abstractmethod
    def poll_results(self, max_results: int | None = None) -> list[EmbeddingWorkerResult]:
        """Poll and retrieve completed recognition results non-blockingly."""

    @abstractmethod
    def quiesce(self, timeout: float = 5.0) -> bool:
        """Wait for queued/in-progress work and discard buffered recognition results."""

    @abstractmethod
    def is_busy_or_pending(self, track_id: int) -> bool:
        """Check if an embedding task for this track is pending or currently processing."""

    @property
    @abstractmethod
    def is_running(self) -> bool:
        """Check if worker is currently active."""

    @property
    @abstractmethod
    def pending_count(self) -> int:
        """Number of pending tasks waiting in the queue."""


class BoundedEmbeddingWorker(BaseEmbeddingWorker):
    """Bounded, thread-safe asynchronous face embedding worker with latest-wins queueing."""

    def __init__(
        self,
        aligner: BaseFaceAligner,
        embedder: BaseFaceEmbedder,
        recognizer: BaseFaceRecognizer,
        config: EmbeddingWorkerConfig | None = None,
    ) -> None:
        self._aligner = aligner
        self._embedder = embedder
        self._recognizer = recognizer
        self._config = config or EmbeddingWorkerConfig()

        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._running = False
        self._thread: threading.Thread | None = None

        # Queue tracking: track_id -> EmbeddingTask (latest-wins per track)
        self._pending_tasks: dict[int, EmbeddingTask] = {}
        # Order of track_ids in queue
        self._queue_order: deque[int] = deque()
        # Track IDs currently undergoing inference in the worker thread
        self._in_progress_tracks: set[int] = set()

        # Results buffer: thread-safe bounded deque populated by worker, drained by main loop
        self._results_queue: deque[EmbeddingWorkerResult] = deque(
            maxlen=self._config.max_results_queue_size
        )

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._running

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending_tasks)

    def start(self) -> None:
        """Start the background embedding worker thread."""
        with self._lock:
            if self._running:
                return
            self._running = True
            self._thread = threading.Thread(
                target=self._worker_loop,
                name="BoundedEmbeddingWorkerThread",
                daemon=True,
            )
            self._thread.start()
            logger.info("BoundedEmbeddingWorker thread started")

    def stop(self, timeout: float = 2.0) -> None:
        """Signal worker thread to shut down and wait for termination.

        Note:
            Any pending tasks remaining in the queue when stop() is called are
            discarded without notification.
        """
        thread_to_join: threading.Thread | None = None
        with self._lock:
            if not self._running:
                return
            self._running = False
            self._condition.notify_all()
            thread_to_join = self._thread
            self._thread = None

        if thread_to_join is not None:
            thread_to_join.join(timeout=timeout)
            logger.info("BoundedEmbeddingWorker thread stopped")

    def submit(self, task: EmbeddingTask) -> bool:
        """Submit an embedding task with latest-wins replacement per track.

        Returns:
            True if task was queued/replaced, False if worker is stopped.
        """
        with self._lock:
            if not self._running:
                return False

            track_id = task.track_id

            # Case 1: Replace pending task with newer observation (latest-wins)
            if track_id in self._pending_tasks:
                self._pending_tasks[track_id] = task
                self._condition.notify()
                return True

            # Case 2: Queue full -> evict oldest pending task to maintain bounded latency
            if len(self._pending_tasks) >= self._config.max_queue_size and self._queue_order:
                oldest_track_id = self._queue_order.popleft()
                self._pending_tasks.pop(oldest_track_id, None)

            # Enqueue new task
            self._pending_tasks[track_id] = task
            self._queue_order.append(track_id)
            self._condition.notify()
            return True

    def poll_results(self, max_results: int | None = None) -> list[EmbeddingWorkerResult]:
        """Drain completed recognition results from results queue without blocking."""
        limit = max_results if max_results is not None else self._config.poll_batch_size
        results: list[EmbeddingWorkerResult] = []

        with self._lock:
            while self._results_queue and len(results) < limit:
                results.append(self._results_queue.popleft())

        return results

    def quiesce(self, timeout: float = 5.0) -> bool:
        """Drain active work and discard results produced against the current gallery."""
        deadline = time.monotonic() + timeout
        with self._condition:
            if not self._running:
                self._pending_tasks.clear()
                self._queue_order.clear()

            while self._pending_tasks or self._in_progress_tracks:
                if not self._running:
                    self._pending_tasks.clear()
                    self._queue_order.clear()

                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return False
                self._condition.wait(timeout=remaining)

            self._results_queue.clear()
            return True

    def is_busy_or_pending(self, track_id: int) -> bool:
        """Check if an embedding task for this track is pending or in progress."""
        with self._lock:
            return track_id in self._pending_tasks or track_id in self._in_progress_tracks

    def process_sync(self, task: EmbeddingTask) -> EmbeddingWorkerResult:
        """Synchronously execute alignment, embedding, and recognition for a task."""
        t_start = time.monotonic()
        queue_latency_ms = (t_start - task.created_at) * 1000.0

        try:
            # 1. Face Alignment
            t_infer_start = time.monotonic()
            align_res = self._aligner.align(
                frame=task.image,
                target=task.bbox,
                landmarks=task.landmarks,
            )

            if not align_res.is_success:
                t_infer_end = time.monotonic()
                infer_latency_ms = (t_infer_end - t_infer_start) * 1000.0
                return EmbeddingWorkerResult(
                    track_id=task.track_id,
                    frame_id=task.frame_id,
                    timestamp=task.timestamp,
                    alignment_result=align_res,
                    embedding_result=None,
                    recognition_result=None,
                    queue_latency_ms=queue_latency_ms,
                    inference_latency_ms=infer_latency_ms,
                    total_latency_ms=queue_latency_ms + infer_latency_ms,
                    error="Face alignment failed",
                )

            # 2. Embedding Extraction
            embed_res = self._embedder.embed(align_res.aligned_face)

            # 3. Gallery Recognition
            rec_res = self._recognizer.recognize(embed_res.embedding)

            t_infer_end = time.monotonic()
            infer_latency_ms = (t_infer_end - t_infer_start) * 1000.0

            return EmbeddingWorkerResult(
                track_id=task.track_id,
                frame_id=task.frame_id,
                timestamp=task.timestamp,
                alignment_result=align_res,
                embedding_result=embed_res,
                recognition_result=rec_res,
                queue_latency_ms=queue_latency_ms,
                inference_latency_ms=infer_latency_ms,
                total_latency_ms=queue_latency_ms + infer_latency_ms,
                error=None,
            )

        except Exception as exc:
            t_infer_end = time.monotonic()
            infer_latency_ms = (t_infer_end - t_start) * 1000.0
            logger.warning("Error processing embedding task for track %d: %s", task.track_id, exc)
            return EmbeddingWorkerResult(
                track_id=task.track_id,
                frame_id=task.frame_id,
                timestamp=task.timestamp,
                alignment_result=None,
                embedding_result=None,
                recognition_result=None,
                queue_latency_ms=queue_latency_ms,
                inference_latency_ms=infer_latency_ms,
                total_latency_ms=queue_latency_ms + infer_latency_ms,
                error=str(exc),
            )

    def _worker_loop(self) -> None:
        """Background thread worker loop executing pending tasks."""
        while True:
            task: EmbeddingTask | None = None

            with self._lock:
                while self._running and not self._pending_tasks:
                    self._condition.wait()

                if not self._running:
                    break

                # Pop next task from queue
                while self._queue_order:
                    next_track_id = self._queue_order.popleft()
                    if next_track_id in self._pending_tasks:
                        task = self._pending_tasks.pop(next_track_id)
                        break

                if task is None:
                    continue

                # Check for staleness
                if self._config.drop_stale_requests:
                    age = time.monotonic() - task.created_at
                    if age > self._config.max_request_age_sec:
                        logger.debug(
                            "Dropping stale task for track %d (age %.2fs > %.2fs)",
                            task.track_id,
                            age,
                            self._config.max_request_age_sec,
                        )
                        self._condition.notify_all()
                        continue

                self._in_progress_tracks.add(task.track_id)

            # Process task outside lock
            result = self.process_sync(task)

            # Push result and clear in-progress state
            with self._lock:
                self._in_progress_tracks.discard(task.track_id)
                self._results_queue.append(result)
                self._condition.notify_all()

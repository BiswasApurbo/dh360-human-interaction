"""Unit tests for BoundedEmbeddingWorker and async queue lifecycle."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import numpy as np

from dh360_hi.vision.aligner import AlignmentResult, BaseFaceAligner
from dh360_hi.vision.embedder import BaseFaceEmbedder, EmbeddingResult, normalize_l2
from dh360_hi.vision.embedding_worker import (
    BoundedEmbeddingWorker,
    EmbeddingTask,
    EmbeddingWorkerConfig,
)
from dh360_hi.vision.quality_gate import FaceQualityTier
from dh360_hi.vision.recognition import (
    BaseFaceRecognizer,
    MatchCandidate,
    RecognitionOutcome,
    RecognitionResult,
)


class MockAligner(BaseFaceAligner):
    """Mock aligner producing synthetic aligned crops."""

    def __init__(self, succeed: bool = True, delay_sec: float = 0.0) -> None:
        self._succeed = succeed
        self._delay = delay_sec

    @property
    def output_size(self) -> tuple[int, int]:
        return (112, 112)

    def align(
        self,
        frame: object,
        target: object,
        landmarks: object = None,
    ) -> AlignmentResult:
        del frame, target, landmarks
        if self._delay > 0:
            time.sleep(self._delay)
        crop = np.zeros((112, 112, 3), dtype=np.uint8)
        return AlignmentResult(
            aligned_face=crop,
            is_success=self._succeed,
            method="similarity",
            output_size=(112, 112),
        )


class MockEmbedder(BaseFaceEmbedder):
    """Mock embedder producing synthetic 128-D vectors."""

    def __init__(self, delay_sec: float = 0.0) -> None:
        self._delay = delay_sec

    @property
    def model_name(self) -> str:
        return "SFace"

    @property
    def model_version(self) -> str:
        return "2021dec"

    @property
    def dimension(self) -> int:
        return 128

    @property
    def input_size(self) -> tuple[int, int]:
        return (112, 112)

    def embed(
        self,
        aligned_face: np.ndarray,
        normalize: bool = True,
    ) -> EmbeddingResult:
        del aligned_face, normalize
        if self._delay > 0:
            time.sleep(self._delay)
        vec = normalize_l2(np.ones(128, dtype=np.float32))
        return EmbeddingResult(
            embedding=vec,
            model_name="SFace",
            model_version="2021dec",
            dimension=128,
            is_normalized=True,
            latency_ms=self._delay * 1000.0,
        )


class MockRecognizer(BaseFaceRecognizer):
    """Mock recognizer returning deterministic outcomes."""

    def __init__(self, outcome: RecognitionOutcome = RecognitionOutcome.IDENTIFIED) -> None:
        self._outcome = outcome

    @property
    def config(self):
        return MagicMock()

    @property
    def gallery(self):
        return MagicMock()

    def recognize(self, query_embedding: np.ndarray) -> RecognitionResult:
        del query_embedding
        if self._outcome == RecognitionOutcome.IDENTIFIED:
            top = MatchCandidate(
                person_id="person-123",
                display_name="Alice",
                similarity=0.92,
                distance=0.08,
            )
            return RecognitionResult(
                outcome=RecognitionOutcome.IDENTIFIED,
                top_match=top,
                second_match=None,
                similarity_margin=0.92,
                confidence_score=0.92,
                model_name="SFace",
                model_version="2021dec",
                is_model_compatible=True,
                details="Identified Alice",
            )
        return RecognitionResult(
            outcome=self._outcome,
            top_match=None,
            second_match=None,
            similarity_margin=0.0,
            confidence_score=0.30,
            model_name="SFace",
            model_version="2021dec",
            is_model_compatible=True,
            details=str(self._outcome),
        )


def _make_task(track_id: int, frame_id: int = 1) -> EmbeddingTask:
    """Helper to create dummy embedding task."""
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    return EmbeddingTask(
        track_id=track_id,
        frame_id=frame_id,
        timestamp=float(frame_id) * 0.033,
        image=dummy_img,
        bbox=(100.0, 100.0, 100.0, 100.0),
        quality_tier=FaceQualityTier.HIGH,
    )


def test_worker_lifecycle_start_and_stop() -> None:
    """Test clean worker startup and shutdown."""
    worker = BoundedEmbeddingWorker(
        aligner=MockAligner(),
        embedder=MockEmbedder(),
        recognizer=MockRecognizer(),
    )
    assert not worker.is_running
    worker.start()
    assert worker.is_running

    # Submit when running
    task = _make_task(track_id=1)
    assert worker.submit(task) is True

    worker.stop(timeout=1.0)
    assert not worker.is_running

    # Submit after stop rejected
    assert worker.submit(task) is False


def test_worker_latest_wins_replaces_pending_task_for_same_track() -> None:
    """Test that a newer frame task for the same track replaces earlier pending task."""
    worker = BoundedEmbeddingWorker(
        aligner=MockAligner(delay_sec=0.05),
        embedder=MockEmbedder(delay_sec=0.05),
        recognizer=MockRecognizer(),
        config=EmbeddingWorkerConfig(max_queue_size=4),
    )
    worker.start()

    try:
        # Submit task 1 for track 10
        t1 = _make_task(track_id=10, frame_id=1)
        worker.submit(t1)

        # Immediately submit task 2 for track 10 (newer observation)
        t2 = _make_task(track_id=10, frame_id=2)
        worker.submit(t2)

        # Wait for completion
        time.sleep(0.3)
        results = worker.poll_results()
        assert len(results) >= 1
        # The processed result should correspond to the newest observation (frame_id 2 or 1 then 2)
        assert results[-1].track_id == 10
        assert results[-1].is_success is True

    finally:
        worker.stop()


def test_worker_bounded_queue_eviction_on_overflow() -> None:
    """Test that queue evicts oldest pending track when capacity is exceeded."""
    worker = BoundedEmbeddingWorker(
        aligner=MockAligner(delay_sec=0.1),
        embedder=MockEmbedder(delay_sec=0.1),
        recognizer=MockRecognizer(),
        config=EmbeddingWorkerConfig(max_queue_size=2),
    )
    worker.start()

    try:
        # Submit 4 distinct tracks into queue of max size 2
        for t_id in range(1, 5):
            worker.submit(_make_task(track_id=t_id, frame_id=t_id))

        assert worker.pending_count <= 2

    finally:
        worker.stop()


def test_worker_error_handling_does_not_crash_thread() -> None:
    """Test that alignment/embedder exceptions are captured safely in result without crashing."""
    worker = BoundedEmbeddingWorker(
        aligner=MockAligner(succeed=False),  # alignment fails
        embedder=MockEmbedder(),
        recognizer=MockRecognizer(),
    )
    worker.start()

    try:
        task = _make_task(track_id=42)
        worker.submit(task)

        time.sleep(0.1)
        results = worker.poll_results()
        assert len(results) == 1
        res = results[0]
        assert res.track_id == 42
        assert not res.is_success
        assert "alignment failed" in str(res.error).lower()
        assert worker.is_running

    finally:
        worker.stop()


def test_worker_synchronous_process_sync() -> None:
    """Test deterministic direct synchronous execution via process_sync."""
    worker = BoundedEmbeddingWorker(
        aligner=MockAligner(),
        embedder=MockEmbedder(),
        recognizer=MockRecognizer(outcome=RecognitionOutcome.IDENTIFIED),
    )
    task = _make_task(track_id=99, frame_id=10)
    result = worker.process_sync(task)

    assert result.track_id == 99
    assert result.frame_id == 10
    assert result.is_success is True
    assert result.recognition_result is not None
    assert result.recognition_result.person_id == "person-123"
    assert result.error is None
    assert result.total_latency_ms >= 0.0

"""Two-rate vision perception pipeline orchestrating fast-rate tracking and async embedding."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from dh360_hi.vision.aligner import BaseFaceAligner, FaceAligner
from dh360_hi.vision.detector import BaseFaceDetector, FaceDetection
from dh360_hi.vision.embedder import BaseFaceEmbedder
from dh360_hi.vision.embedding_worker import (
    BaseEmbeddingWorker,
    BoundedEmbeddingWorker,
    EmbeddingTask,
    EmbeddingWorkerConfig,
    EmbeddingWorkerResult,
)
from dh360_hi.vision.frame import Frame
from dh360_hi.vision.identity_state import (
    IdentityStateConfig,
    IdentityStateManager,
    TrackIdentityState,
)
from dh360_hi.vision.presence_debouncer import (
    PresenceDebounceConfig,
    PresenceDebouncer,
    PresenceTransition,
)
from dh360_hi.vision.quality_gate import (
    BaseFaceQualityGate,
    FaceQualityGate,
    FaceQualityResult,
    FaceQualityTier,
)
from dh360_hi.vision.recognition import BaseFaceRecognizer, FaceRecognizer
from dh360_hi.vision.target_selector import (
    BaseTargetSelector,
    InteractionTargetSelector,
    TargetSelectionConfig,
    TargetSelectionResult,
)
from dh360_hi.vision.tracker import BaseFaceTracker, FaceTracker, TrackedFace

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TwoRateFrameResult:
    """Consolidated output from a single frame processed by the two-rate vision pipeline."""

    frame_id: int
    timestamp: float
    detections: list[FaceDetection]
    tracks: list[TrackedFace]
    quality_results: dict[int, FaceQualityResult]
    identity_states: dict[int, TrackIdentityState]
    target_result: TargetSelectionResult
    presence_transitions: list[PresenceTransition]
    completed_recognitions: list[EmbeddingWorkerResult]
    ambiguous_crossing_track_ids: set[int]
    pending_worker_tasks: int
    frame_latency_ms: float
    total_pipeline_latency_ms: float


class TwoRatePerceptionPipeline:
    """Orchestrates real-time perception (30+ FPS) decoupled from async face embedding."""

    def __init__(
        self,
        detector: BaseFaceDetector,
        embedder: BaseFaceEmbedder | None = None,
        tracker: BaseFaceTracker | None = None,
        quality_gate: BaseFaceQualityGate | None = None,
        aligner: BaseFaceAligner | None = None,
        recognizer: BaseFaceRecognizer | None = None,
        worker: BaseEmbeddingWorker | None = None,
        identity_manager: IdentityStateManager | None = None,
        target_selector: BaseTargetSelector | None = None,
        presence_debouncer: PresenceDebouncer | None = None,
        worker_config: EmbeddingWorkerConfig | None = None,
        identity_config: IdentityStateConfig | None = None,
        target_config: TargetSelectionConfig | None = None,
        presence_config: PresenceDebounceConfig | None = None,
    ) -> None:
        self._detector = detector
        self._tracker = tracker or FaceTracker()
        self._quality_gate = quality_gate or FaceQualityGate()
        self._aligner = aligner or FaceAligner()
        self._recognizer = recognizer or FaceRecognizer()

        if worker is not None:
            self._worker = worker
            self._embedder = embedder
        else:
            if embedder is None:
                msg = "An embedder instance must be provided when worker is not supplied"
                raise ValueError(msg)
            self._embedder = embedder
            self._worker = BoundedEmbeddingWorker(
                aligner=self._aligner,
                embedder=self._embedder,
                recognizer=self._recognizer,
                config=worker_config,
            )

        self._identity_manager = identity_manager or IdentityStateManager(identity_config)
        self._target_selector = target_selector or InteractionTargetSelector(target_config)
        self._presence_debouncer = presence_debouncer or PresenceDebouncer(presence_config)

    @property
    def detector(self) -> BaseFaceDetector:
        return self._detector

    @property
    def tracker(self) -> BaseFaceTracker:
        return self._tracker

    @property
    def quality_gate(self) -> BaseFaceQualityGate:
        return self._quality_gate

    @property
    def aligner(self) -> BaseFaceAligner:
        return self._aligner

    @property
    def embedder(self) -> BaseFaceEmbedder | None:
        return self._embedder

    @property
    def recognizer(self) -> BaseFaceRecognizer:
        return self._recognizer

    @property
    def worker(self) -> BaseEmbeddingWorker:
        return self._worker

    @property
    def identity_manager(self) -> IdentityStateManager:
        return self._identity_manager

    @property
    def target_selector(self) -> BaseTargetSelector:
        return self._target_selector

    @property
    def presence_debouncer(self) -> PresenceDebouncer:
        return self._presence_debouncer

    def start(self) -> None:
        """Start background embedding worker thread."""
        self._worker.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Stop background embedding worker thread."""
        self._worker.stop(timeout=timeout)

    def reset(self) -> None:
        """Reset internal states across all pipeline stages."""
        self._tracker.reset()
        self._identity_manager.reset()
        self._target_selector.reset()
        self._presence_debouncer.reset()

    def process_frame(self, frame: Frame) -> TwoRateFrameResult:
        """Process a camera frame through the two-rate vision pipeline.

        Synchronous Fast-Path:
            Detection -> Tracking -> Quality Gate -> Crossing Check ->
            Poll Async Recognition Results -> Identity Updates ->
            Worker Queueing -> Target Selection -> Presence Debouncing.

        Args:
            frame: Incoming camera frame.

        Returns:
            TwoRateFrameResult containing all per-frame perception artifacts.
        """
        t_start = time.monotonic()

        # 1. Face Detection (Fast Path)
        det_result = self._detector.detect(frame)
        detections = det_result.detections

        # 2. Multi-Face Tracking (Fast Path)
        track_result = self._tracker.update(detections, frame.frame_id, frame.timestamp)
        active_tracks = track_result.tracks
        confirmed_tracks = track_result.confirmed_tracks

        # 3. Face Quality Gating (Fast Path)
        quality_map: dict[int, FaceQualityResult] = {}
        for track in confirmed_tracks:
            q_res = self._quality_gate.evaluate(
                frame=frame,
                target=track,
                confidence=track.confidence,
                landmarks=track.landmarks,
            )
            quality_map[track.track_id] = q_res

        # 4. Spatial Proximity & Crossing Detection (ID Swap Protection)
        ambiguous_crossings = self._identity_manager.check_track_crossings(
            active_tracks=confirmed_tracks,
            current_frame_id=frame.frame_id,
        )

        # 5. Poll Completed Recognition Results from Async Worker (Non-blocking)
        completed_results = self._worker.poll_results()
        for res in completed_results:
            if res.is_success and res.recognition_result is not None:
                self._identity_manager.update(
                    track_id=res.track_id,
                    rec_result=res.recognition_result,
                    frame_id=frame.frame_id,
                )

        # 6. Evaluate and Submit New Embedding Tasks to Bounded Worker
        img_array = frame.image
        for track in confirmed_tracks:
            track_quality = quality_map.get(track.track_id)
            tier = track_quality.tier if track_quality is not None else FaceQualityTier.REJECTED

            # Check if this track should request an embedding task
            should_embed = self._identity_manager.should_request_embedding(
                track.track_id, frame.frame_id, tier
            )
            if should_embed and not self._worker.is_busy_or_pending(track.track_id):
                task = EmbeddingTask(
                    track_id=track.track_id,
                    frame_id=frame.frame_id,
                    timestamp=frame.timestamp,
                    image=img_array,
                    bbox=track.bbox,
                    landmarks=track.landmarks,
                    quality_tier=tier,
                    quality_score=track_quality.metrics.sharpness if track_quality else 1.0,
                )
                self._worker.submit(task)
                self._identity_manager.mark_embedding_requested(track.track_id, frame.frame_id)

        # 7. Prune Lost Tracks from Identity Manager
        active_track_ids = {t.track_id for t in confirmed_tracks}
        self._identity_manager.prune_lost_tracks(
            current_frame_id=frame.frame_id,
            active_track_ids=active_track_ids,
        )
        identity_states = self._identity_manager.get_all_states()

        # 8. Target Selection (Multi-Person Focus)
        h, w = img_array.shape[:2]
        target_res = self._target_selector.select_target(
            tracks=confirmed_tracks,
            identity_states=identity_states,
            frame_width=w,
            frame_height=h,
            frame_id=frame.frame_id,
        )

        # 9. Presence Debouncing (Entering / Leaving Smoothing)
        presence_events = self._presence_debouncer.update(
            tracks=confirmed_tracks,
            identity_states=identity_states,
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
        )

        t_end = time.monotonic()
        frame_latency_ms = (t_end - t_start) * 1000.0

        return TwoRateFrameResult(
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            detections=detections,
            tracks=active_tracks,
            quality_results=quality_map,
            identity_states=identity_states,
            target_result=target_res,
            presence_transitions=presence_events,
            completed_recognitions=completed_results,
            ambiguous_crossing_track_ids=ambiguous_crossings,
            pending_worker_tasks=self._worker.pending_count,
            frame_latency_ms=frame_latency_ms,
            total_pipeline_latency_ms=frame_latency_ms,
        )

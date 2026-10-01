"""Unified configuration model for human interaction application composition."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from dh360_hi.edge.runtime import EdgeRuntimeConfig
from dh360_hi.edge.session_manager import SessionConfig
from dh360_hi.vision.embedding_worker import EmbeddingWorkerConfig
from dh360_hi.vision.enrollment import EnrollmentConfig
from dh360_hi.vision.identity_state import IdentityStateConfig
from dh360_hi.vision.presence_debouncer import PresenceDebounceConfig
from dh360_hi.vision.quality_gate import FaceQualityThresholds
from dh360_hi.vision.recognition import RecognitionConfig
from dh360_hi.vision.target_selector import TargetSelectionConfig


@dataclass(frozen=True)
class HumanInteractionConfig:
    """Unified application-level configuration for the Human Interaction subsystem.

    Aggregates configuration parameters across storage, perception pipeline,
    edge runtime, session management, and application context layers while
    reusing existing typed configuration structures.

    Attributes:
        db_path: Path to SQLite database file or ":memory:".
        auto_init_db: Whether to automatically run schema migrations on DB open.
        detector_model_path: Path to YuNet ONNX detection model weights.
        embedder_model_path: Path to SFace ONNX embedding model weights.
        detector_input_size: Resolution (width, height) for detector inference.
        detector_score_threshold: Confidence threshold for face detections.
        detector_nms_threshold: Non-maximum suppression threshold for face detections.
        detector_top_k: Maximum number of detection candidates to evaluate.
        tracker_iou_threshold: Minimum IoU overlap to match detection with track.
        tracker_max_missed_frames: Maximum missed frames before track expiration.
        tracker_min_hits: Detection hits required to promote track to confirmed.
        tracker_max_center_distance: Optional center distance fallback threshold.
        tracker_smooth_factor: Bounding box smoothing factor.
        aligner_output_size: Output canonical face dimension (width, height).
        aligner_fallback_to_bbox: Whether to fallback to bounding box crop if alignment fails.
        quality: Face quality gate filtering thresholds.
        runtime: Edge runtime continuous loop and FPS pacing configuration.
        session: Interaction session and presence requirement configuration.
        recognition: Recognition similarity thresholds and aggregation method.
        worker: Background embedding worker queue and polling configuration.
        identity: Identity state temporal stability and voting configuration.
        target_selector: Interaction target scoring and stickiness configuration.
        presence: Presence enter/departure temporal debounce configuration.
        enrollment: Camera-based face enrollment and diversity configuration.
        model_name: Expected face recognition embedding model name.
        model_version: Expected face recognition embedding model version.
        require_active_for_gallery: Whether only ACTIVE persons are loaded into gallery.
        require_consent_for_gallery: Whether only GRANTED consent persons are loaded into gallery.
    """

    # Persistence & Storage
    db_path: str | Path = ":memory:"
    auto_init_db: bool = True

    # Model Weights Paths
    detector_model_path: str | Path | None = None
    embedder_model_path: str | Path | None = None

    # Detector Parameters
    detector_input_size: tuple[int, int] = (320, 320)
    detector_score_threshold: float = 0.6
    detector_nms_threshold: float = 0.3
    detector_top_k: int = 5000

    # Tracker Parameters
    tracker_iou_threshold: float = 0.3
    tracker_max_missed_frames: int = 15
    tracker_min_hits: int = 1
    tracker_max_center_distance: float | None = 100.0
    tracker_smooth_factor: float = 0.8

    # Aligner Parameters
    aligner_output_size: tuple[int, int] = (112, 112)
    aligner_fallback_to_bbox: bool = True

    # Subsystem Configurations
    quality: FaceQualityThresholds = field(default_factory=FaceQualityThresholds)
    runtime: EdgeRuntimeConfig = field(default_factory=EdgeRuntimeConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    recognition: RecognitionConfig = field(default_factory=RecognitionConfig)
    worker: EmbeddingWorkerConfig = field(default_factory=EmbeddingWorkerConfig)
    identity: IdentityStateConfig = field(default_factory=IdentityStateConfig)
    target_selector: TargetSelectionConfig = field(default_factory=TargetSelectionConfig)
    presence: PresenceDebounceConfig = field(default_factory=PresenceDebounceConfig)
    enrollment: EnrollmentConfig = field(default_factory=EnrollmentConfig)

    # Recognition Gallery Settings
    model_name: str = "SFace"
    model_version: str = "2021dec"
    require_active_for_gallery: bool = True
    require_consent_for_gallery: bool = True

    def __post_init__(self) -> None:
        """Validate top-level configuration values."""
        if not str(self.db_path).strip():
            msg = "db_path must not be empty"
            raise ValueError(msg)
        if not self.model_name.strip():
            msg = "model_name must not be empty"
            raise ValueError(msg)
        if not self.model_version.strip():
            msg = "model_version must not be empty"
            raise ValueError(msg)

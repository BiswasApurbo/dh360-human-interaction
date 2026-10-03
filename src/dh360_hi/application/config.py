"""Unified configuration model for human interaction application composition."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from dh360_hi.edge.runtime import EdgeRuntimeConfig
from dh360_hi.edge.session_manager import SessionConfig
from dh360_hi.vision.embedding_worker import EmbeddingWorkerConfig
from dh360_hi.vision.enrollment import EnrollmentConfig
from dh360_hi.vision.identity_state import IdentityStateConfig
from dh360_hi.vision.presence_debouncer import PresenceDebounceConfig
from dh360_hi.vision.quality_gate import FaceQualityThresholds
from dh360_hi.vision.recognition import RecognitionConfig
from dh360_hi.vision.target_selector import TargetSelectionConfig


class ConfigurationPreset(StrEnum):
    """Supported standard configuration presets."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


@dataclass(frozen=True)
class ApiConfig:
    """API operator authentication settings.

    The operator token is intentionally excluded from repr and serialization.
    """

    operator_token: str | None = field(default=None, repr=False, compare=False)
    operator_scopes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return non-secret API settings for configuration serialization."""
        return {"operator_scopes": list(self.operator_scopes)}


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
    api: ApiConfig = field(default_factory=ApiConfig)

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

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to a nested JSON/serialization-safe dictionary."""
        return {
            "db_path": str(self.db_path),
            "auto_init_db": self.auto_init_db,
            "detector_model_path": str(self.detector_model_path)
            if self.detector_model_path is not None
            else None,
            "embedder_model_path": str(self.embedder_model_path)
            if self.embedder_model_path is not None
            else None,
            "detector_input_size": list(self.detector_input_size),
            "detector_score_threshold": self.detector_score_threshold,
            "detector_nms_threshold": self.detector_nms_threshold,
            "detector_top_k": self.detector_top_k,
            "tracker_iou_threshold": self.tracker_iou_threshold,
            "tracker_max_missed_frames": self.tracker_max_missed_frames,
            "tracker_min_hits": self.tracker_min_hits,
            "tracker_max_center_distance": self.tracker_max_center_distance,
            "tracker_smooth_factor": self.tracker_smooth_factor,
            "aligner_output_size": list(self.aligner_output_size),
            "aligner_fallback_to_bbox": self.aligner_fallback_to_bbox,
            "quality": {
                "min_confidence": self.quality.min_confidence,
                "min_face_width": self.quality.min_face_width,
                "min_face_height": self.quality.min_face_height,
                "min_sharpness": self.quality.min_sharpness,
                "max_roll_angle_deg": self.quality.max_roll_angle_deg,
                "max_yaw_asymmetry": self.quality.max_yaw_asymmetry,
                "require_landmarks": self.quality.require_landmarks,
            },
            "runtime": {
                "target_fps": self.runtime.target_fps,
                "max_consecutive_frame_errors": self.runtime.max_consecutive_frame_errors,
                "auto_start_worker": self.runtime.auto_start_worker,
            },
            "session": {
                "require_confirmed_presence": self.session.require_confirmed_presence,
            },
            "recognition": {
                "similarity_threshold": self.recognition.similarity_threshold,
                "min_similarity_margin": self.recognition.min_similarity_margin,
                "aggregation_method": self.recognition.aggregation_method,
                "require_model_match": self.recognition.require_model_match,
            },
            "worker": {
                "max_queue_size": self.worker.max_queue_size,
                "max_results_queue_size": self.worker.max_results_queue_size,
                "max_request_age_sec": self.worker.max_request_age_sec,
                "drop_stale_requests": self.worker.drop_stale_requests,
                "poll_batch_size": self.worker.poll_batch_size,
            },
            "identity": {
                "min_observations_to_decide": self.identity.min_observations_to_decide,
                "evidence_window_size": self.identity.evidence_window_size,
                "identification_consistency_ratio": self.identity.identification_consistency_ratio,
                "hysteresis_patience_observations": self.identity.hysteresis_patience_observations,
                "max_missed_frames_to_lost": self.identity.max_missed_frames_to_lost,
                "reverification_interval_frames": self.identity.reverification_interval_frames,
                "spatial_proximity_distance_px": self.identity.spatial_proximity_distance_px,
                "spatial_proximity_min_iou": self.identity.spatial_proximity_min_iou,
            },
            "target_selector": {
                "weight_centrality": self.target_selector.weight_centrality,
                "weight_size": self.target_selector.weight_size,
                "weight_identity": self.target_selector.weight_identity,
                "weight_dwell": self.target_selector.weight_dwell,
                "hysteresis_switch_margin": self.target_selector.hysteresis_switch_margin,
                "max_lost_frames_to_release": self.target_selector.max_lost_frames_to_release,
                "max_dwell_frames_normalization": (
                    self.target_selector.max_dwell_frames_normalization
                ),
            },
            "presence": {
                "enter_patience_frames": self.presence.enter_patience_frames,
                "leave_patience_frames": self.presence.leave_patience_frames,
            },
            "enrollment": {
                "target_samples": self.enrollment.target_samples,
                "minimum_samples": self.enrollment.minimum_samples,
                "max_attempts": self.enrollment.max_attempts,
                "duplicate_similarity_threshold": self.enrollment.duplicate_similarity_threshold,
                "min_sample_interval_seconds": self.enrollment.min_sample_interval_seconds,
                "timeout_seconds": self.enrollment.timeout_seconds,
                "model_name": self.enrollment.model_name,
                "model_version": self.enrollment.model_version,
            },
            "model_name": self.model_name,
            "model_version": self.model_version,
            "require_active_for_gallery": self.require_active_for_gallery,
            "require_consent_for_gallery": self.require_consent_for_gallery,
            "api": self.api.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HumanInteractionConfig:
        """Construct HumanInteractionConfig from a dictionary with nested sub-configs."""
        d = dict(data)

        # Convert nested sub-configs if dicts
        quality_data = d.get("quality")
        if isinstance(quality_data, dict):
            quality = FaceQualityThresholds(
                min_confidence=float(quality_data.get("min_confidence", 0.80)),
                min_face_width=float(quality_data.get("min_face_width", 40.0)),
                min_face_height=float(quality_data.get("min_face_height", 40.0)),
                min_sharpness=float(quality_data.get("min_sharpness", 50.0)),
                max_roll_angle_deg=float(quality_data.get("max_roll_angle_deg", 25.0)),
                max_yaw_asymmetry=float(quality_data.get("max_yaw_asymmetry", 2.5)),
                require_landmarks=bool(quality_data.get("require_landmarks", False)),
            )
            d["quality"] = quality

        runtime_data = d.get("runtime")
        if isinstance(runtime_data, dict):
            runtime = EdgeRuntimeConfig(
                target_fps=float(runtime_data.get("target_fps", 30.0)),
                max_consecutive_frame_errors=int(
                    runtime_data.get("max_consecutive_frame_errors", 10)
                ),
                auto_start_worker=bool(runtime_data.get("auto_start_worker", True)),
            )
            d["runtime"] = runtime

        session_data = d.get("session")
        if isinstance(session_data, dict):
            session = SessionConfig(
                require_confirmed_presence=bool(
                    session_data.get("require_confirmed_presence", True)
                ),
            )
            d["session"] = session

        recognition_data = d.get("recognition")
        if isinstance(recognition_data, dict):
            recognition = RecognitionConfig(
                similarity_threshold=float(recognition_data.get("similarity_threshold", 0.65)),
                min_similarity_margin=float(recognition_data.get("min_similarity_margin", 0.08)),
                aggregation_method=str(recognition_data.get("aggregation_method", "max")),
                require_model_match=bool(recognition_data.get("require_model_match", True)),
            )
            d["recognition"] = recognition

        worker_data = d.get("worker")
        if isinstance(worker_data, dict):
            worker = EmbeddingWorkerConfig(
                max_queue_size=int(worker_data.get("max_queue_size", 8)),
                max_results_queue_size=int(worker_data.get("max_results_queue_size", 32)),
                max_request_age_sec=float(worker_data.get("max_request_age_sec", 1.0)),
                drop_stale_requests=bool(worker_data.get("drop_stale_requests", True)),
                poll_batch_size=int(worker_data.get("poll_batch_size", 16)),
            )
            d["worker"] = worker

        identity_data = d.get("identity")
        if isinstance(identity_data, dict):
            identity = IdentityStateConfig(
                min_observations_to_decide=int(identity_data.get("min_observations_to_decide", 3)),
                evidence_window_size=int(identity_data.get("evidence_window_size", 7)),
                identification_consistency_ratio=float(
                    identity_data.get("identification_consistency_ratio", 0.70)
                ),
                hysteresis_patience_observations=int(
                    identity_data.get("hysteresis_patience_observations", 4)
                ),
                max_missed_frames_to_lost=int(identity_data.get("max_missed_frames_to_lost", 30)),
                reverification_interval_frames=int(
                    identity_data.get("reverification_interval_frames", 90)
                ),
                spatial_proximity_distance_px=float(
                    identity_data.get("spatial_proximity_distance_px", 80.0)
                ),
                spatial_proximity_min_iou=float(
                    identity_data.get("spatial_proximity_min_iou", 0.08)
                ),
            )
            d["identity"] = identity

        target_data = d.get("target_selector")
        if isinstance(target_data, dict):
            target_selector = TargetSelectionConfig(
                weight_centrality=float(target_data.get("weight_centrality", 0.35)),
                weight_size=float(target_data.get("weight_size", 0.35)),
                weight_identity=float(target_data.get("weight_identity", 0.20)),
                weight_dwell=float(target_data.get("weight_dwell", 0.10)),
                hysteresis_switch_margin=float(target_data.get("hysteresis_switch_margin", 0.15)),
                max_lost_frames_to_release=int(target_data.get("max_lost_frames_to_release", 15)),
                max_dwell_frames_normalization=float(
                    target_data.get("max_dwell_frames_normalization", 150.0)
                ),
            )
            d["target_selector"] = target_selector

        presence_data = d.get("presence")
        if isinstance(presence_data, dict):
            presence = PresenceDebounceConfig(
                enter_patience_frames=int(presence_data.get("enter_patience_frames", 3)),
                leave_patience_frames=int(presence_data.get("leave_patience_frames", 10)),
            )
            d["presence"] = presence

        enrollment_data = d.get("enrollment")
        if isinstance(enrollment_data, dict):
            enrollment = EnrollmentConfig(
                target_samples=int(enrollment_data.get("target_samples", 7)),
                minimum_samples=int(enrollment_data.get("minimum_samples", 5)),
                max_attempts=int(enrollment_data.get("max_attempts", 60)),
                duplicate_similarity_threshold=float(
                    enrollment_data.get("duplicate_similarity_threshold", 0.95)
                ),
                min_sample_interval_seconds=float(
                    enrollment_data.get("min_sample_interval_seconds", 0.2)
                ),
                timeout_seconds=float(enrollment_data.get("timeout_seconds", 30.0)),
                model_name=str(enrollment_data.get("model_name", "SFace")),
                model_version=str(enrollment_data.get("model_version", "2021dec")),
            )
            d["enrollment"] = enrollment

        api_data = d.get("api")
        if isinstance(api_data, dict):
            scopes_data = api_data.get("operator_scopes", ())
            if isinstance(scopes_data, str):
                scopes = tuple(scope.strip() for scope in scopes_data.split(",") if scope.strip())
            elif isinstance(scopes_data, (list, tuple)):
                scopes = tuple(str(scope) for scope in scopes_data)
            else:
                msg = "api.operator_scopes must be a list or comma-separated string"
                raise ValueError(msg)
            token = api_data.get("operator_token")
            if token is not None and not isinstance(token, str):
                msg = "api.operator_token must be a string"
                raise ValueError(msg)
            d["api"] = ApiConfig(operator_token=token, operator_scopes=scopes)

        # Convert tuples
        if "detector_input_size" in d and isinstance(d["detector_input_size"], list):
            d["detector_input_size"] = tuple(d["detector_input_size"])
        if "aligner_output_size" in d and isinstance(d["aligner_output_size"], list):
            d["aligner_output_size"] = tuple(d["aligner_output_size"])

        return cls(**d)


def get_preset(preset: ConfigurationPreset | str) -> HumanInteractionConfig:
    """Retrieve pre-tuned configuration by preset name.

    Args:
        preset: ConfigurationPreset enum or string ('development', 'testing', 'production').

    Returns:
        HumanInteractionConfig tailored to the requested environment.

    Raises:
        ValueError: If preset name is unrecognized.
    """
    preset_name = str(preset).lower()
    if preset_name in (ConfigurationPreset.DEVELOPMENT, "dev"):
        return HumanInteractionConfig(
            db_path="dh360_dev.db",
            auto_init_db=True,
            runtime=EdgeRuntimeConfig(target_fps=30.0),
        )
    if preset_name in (ConfigurationPreset.TESTING, "test"):
        return HumanInteractionConfig(
            db_path=":memory:",
            auto_init_db=True,
            runtime=EdgeRuntimeConfig(target_fps=60.0),
            presence=PresenceDebounceConfig(enter_patience_frames=1, leave_patience_frames=2),
        )
    if preset_name in (ConfigurationPreset.PRODUCTION, "prod"):
        return HumanInteractionConfig(
            db_path="dh360_hi.db",
            auto_init_db=True,
            runtime=EdgeRuntimeConfig(target_fps=30.0, max_consecutive_frame_errors=5),
            require_active_for_gallery=True,
            require_consent_for_gallery=True,
        )
    msg = (
        f"Unknown configuration preset: '{preset}'. "
        f"Available presets: {[p.value for p in ConfigurationPreset]}"
    )
    raise ValueError(msg)

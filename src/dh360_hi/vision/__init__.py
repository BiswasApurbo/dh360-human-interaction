"""Vision pipeline core modules: detection, tracking, quality gate, and embeddings."""

from dh360_hi.vision.benchmark import (
    ConfidenceStats,
    LatencyStats,
    PipelineBenchmark,
    StageTimer,
)
from dh360_hi.vision.detector import (
    BaseFaceDetector,
    DetectionResult,
    FaceDetection,
    FaceLandmarks,
    YuNetFaceDetector,
)
from dh360_hi.vision.frame import (
    Frame,
    FrameSource,
    SyntheticFrameSource,
    VideoFileFrameSource,
    WebcamFrameSource,
)
from dh360_hi.vision.quality_gate import (
    BaseFaceQualityGate,
    FaceQualityGate,
    FaceQualityMetrics,
    FaceQualityResult,
    FaceQualityThresholds,
    FaceRejectionReason,
    compute_pose_metrics,
    compute_sharpness,
)
from dh360_hi.vision.tracker import (
    BaseFaceTracker,
    FaceTracker,
    TrackedFace,
    TrackingResult,
    TrackState,
    compute_center_distance,
    compute_iou,
)

__all__ = [
    "BaseFaceDetector",
    "BaseFaceQualityGate",
    "BaseFaceTracker",
    "ConfidenceStats",
    "DetectionResult",
    "FaceDetection",
    "FaceLandmarks",
    "FaceQualityGate",
    "FaceQualityMetrics",
    "FaceQualityResult",
    "FaceQualityThresholds",
    "FaceRejectionReason",
    "FaceTracker",
    "Frame",
    "FrameSource",
    "LatencyStats",
    "PipelineBenchmark",
    "StageTimer",
    "SyntheticFrameSource",
    "TrackState",
    "TrackedFace",
    "TrackingResult",
    "VideoFileFrameSource",
    "WebcamFrameSource",
    "YuNetFaceDetector",
    "compute_center_distance",
    "compute_iou",
    "compute_pose_metrics",
    "compute_sharpness",
]

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

__all__ = [
    "BaseFaceDetector",
    "ConfidenceStats",
    "DetectionResult",
    "FaceDetection",
    "FaceLandmarks",
    "Frame",
    "FrameSource",
    "LatencyStats",
    "PipelineBenchmark",
    "StageTimer",
    "SyntheticFrameSource",
    "VideoFileFrameSource",
    "WebcamFrameSource",
    "YuNetFaceDetector",
]

"""Unit tests for FaceDetector data structures and interface."""

from pathlib import Path

import pytest

from dh360_hi.vision.detector import (
    DetectionResult,
    FaceDetection,
    FaceLandmarks,
    YuNetFaceDetector,
)


def test_face_detection_and_landmarks_data() -> None:
    """Verify FaceDetection and FaceLandmarks property accessors."""
    landmarks = FaceLandmarks(
        right_eye=(10.0, 15.0),
        left_eye=(30.0, 15.0),
        nose_tip=(20.0, 25.0),
        right_mouth_corner=(12.0, 35.0),
        left_mouth_corner=(28.0, 35.0),
    )
    assert len(landmarks.to_list()) == 5
    assert landmarks.right_eye == (10.0, 15.0)

    detection = FaceDetection(
        bbox=(100.0, 150.0, 50.0, 60.0),
        confidence=0.92,
        landmarks=landmarks,
    )

    assert detection.x == 100.0
    assert detection.y == 150.0
    assert detection.width == 50.0
    assert detection.height == 60.0
    assert detection.area == 3000.0
    assert detection.confidence == 0.92
    assert detection.landmarks is not None


def test_detection_result_properties() -> None:
    """Verify DetectionResult container properties."""
    res = DetectionResult(
        detections=[],
        frame_id=1,
        latency_ms=8.5,
        input_size=(320, 320),
    )
    assert res.frame_id == 1
    assert res.latency_ms == 8.5
    assert res.input_size == (320, 320)
    assert len(res.detections) == 0


def test_yunet_missing_model_raises_filenotfound() -> None:
    """Verify YuNetFaceDetector raises FileNotFoundError when model path does not exist."""
    with pytest.raises(FileNotFoundError, match="YuNet ONNX model file not found"):
        YuNetFaceDetector(model_path=Path("non_existent_path.onnx"))

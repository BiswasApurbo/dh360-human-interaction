"""Integration tests for YuNet face detector execution."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from dh360_hi.vision.detector import YuNetFaceDetector
from dh360_hi.vision.frame import Frame

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = ROOT_DIR / "models" / "weights" / "face_detection_yunet_2023mar.onnx"
FIXTURE_PATH = ROOT_DIR / "tests" / "fixtures" / "sample_face.jpg"


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="YuNet model weights not downloaded")
def test_yunet_detector_execution_on_synthetic_frame() -> None:
    """Verify YuNet detector executes cleanly on synthetic frame."""
    detector = YuNetFaceDetector(
        model_path=MODEL_PATH,
        input_size=(320, 320),
        score_threshold=0.6,
    )

    # Create synthetic test frame
    image = np.zeros((320, 320, 3), dtype=np.uint8)
    frame = Frame(image=image, frame_id=1, timestamp=0.0, source_info="test")

    result = detector.detect(frame)

    assert result.frame_id == 1
    assert result.latency_ms >= 0.0
    assert result.input_size == (320, 320)
    assert isinstance(result.detections, list)
    assert len(result.detections) == 0


@pytest.mark.skipif(
    not (MODEL_PATH.exists() and FIXTURE_PATH.exists()),
    reason="YuNet model weights or sample fixture not found",
)
def test_yunet_detector_detects_face_on_fixture() -> None:
    """Verify YuNet detector successfully detects face and landmarks on standard fixture."""
    img = cv2.imread(str(FIXTURE_PATH))
    assert img is not None

    detector = YuNetFaceDetector(
        model_path=MODEL_PATH,
        score_threshold=0.6,
    )

    frame = Frame(image=img, frame_id=10, timestamp=1.0, source_info="fixture")
    result = detector.detect(frame)

    assert len(result.detections) == 1
    detection = result.detections[0]

    # Verify detection accuracy and confidence
    assert detection.confidence >= 0.85
    assert detection.width > 50 and detection.height > 50
    assert detection.landmarks is not None
    landmarks_list = detection.landmarks.to_list()
    assert len(landmarks_list) == 5

    # High threshold filter verification
    detector.confidence_threshold = 0.99
    high_thresh_res = detector.detect(frame)
    assert len(high_thresh_res.detections) == 0


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="YuNet model weights not downloaded")
def test_yunet_detector_dynamic_input_size() -> None:
    """Verify detector dynamically adapts to changing frame resolutions."""
    detector = YuNetFaceDetector(
        model_path=MODEL_PATH,
        input_size=(320, 320),
    )

    # First run on 320x320
    img1 = np.zeros((320, 320, 3), dtype=np.uint8)
    res1 = detector.detect(img1, frame_id=1)
    assert res1.input_size == (320, 320)

    # Second run on 640x480
    img2 = np.zeros((480, 640, 3), dtype=np.uint8)
    res2 = detector.detect(img2, frame_id=2)
    assert res2.input_size == (640, 480)
    assert detector.input_size == (640, 480)


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="YuNet model weights not downloaded")
def test_yunet_confidence_threshold_setter() -> None:
    """Verify setting confidence threshold updates detector property."""
    detector = YuNetFaceDetector(
        model_path=MODEL_PATH,
        score_threshold=0.5,
    )
    assert detector.confidence_threshold == 0.5
    detector.confidence_threshold = 0.8
    assert detector.confidence_threshold == 0.8

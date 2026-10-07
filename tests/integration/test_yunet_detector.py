"""Integration tests for YuNet face detector execution."""

import threading
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


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="YuNet model weights not downloaded")
def test_yunet_detector_thread_safety() -> None:
    """Verify concurrent detection calls from multiple threads run safely
    without graph buffer corruption.
    """
    detector = YuNetFaceDetector(
        model_path=MODEL_PATH,
        input_size=(320, 320),
    )

    errors: list[Exception] = []

    def worker(img: np.ndarray, frame_id_base: int) -> None:
        for idx in range(20):
            try:
                res = detector.detect(img, frame_id=frame_id_base + idx)
                assert res is not None
            except Exception as exc:
                errors.append(exc)
                break

    img_480p = np.zeros((480, 640, 3), dtype=np.uint8)
    img_720p = np.zeros((720, 1280, 3), dtype=np.uint8)
    img_320p = np.zeros((320, 320, 3), dtype=np.uint8)

    threads = [
        threading.Thread(target=worker, args=(img_480p, 100)),
        threading.Thread(target=worker, args=(img_720p, 200)),
        threading.Thread(target=worker, args=(img_320p, 300)),
        threading.Thread(target=worker, args=(img_480p, 400)),
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0, f"Concurrent detection raised errors: {errors}"


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="YuNet model weights not downloaded")
def test_yunet_detector_frame_normalization() -> None:
    """Verify detector normalizes non-contiguous, BGRA, and grayscale frames."""
    detector = YuNetFaceDetector(
        model_path=MODEL_PATH,
        input_size=(320, 320),
    )

    # 1. Non-contiguous sliced frame
    full_img = np.zeros((480, 640, 3), dtype=np.uint8)
    sliced_img = full_img[::2, ::2]  # non-contiguous view
    assert not sliced_img.flags.c_contiguous
    res_sliced = detector.detect(sliced_img)
    assert res_sliced.input_size == (320, 240)

    # 2. 4-channel BGRA frame
    bgra_img = np.zeros((240, 320, 4), dtype=np.uint8)
    res_bgra = detector.detect(bgra_img)
    assert res_bgra.input_size == (320, 240)

    # 3. 1-channel grayscale frame
    gray_img = np.zeros((240, 320), dtype=np.uint8)
    res_gray = detector.detect(gray_img)
    assert res_gray.input_size == (320, 240)

    # 4. Empty/zero dimension frame
    empty_img = np.zeros((0, 0, 3), dtype=np.uint8)
    res_empty = detector.detect(empty_img)
    assert len(res_empty.detections) == 0

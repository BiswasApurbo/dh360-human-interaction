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
    """Verify detector preserves configured ceiling, scales down 640x480 to 320x240,
    and updates when input_size is explicitly configured.
    """
    detector = YuNetFaceDetector(
        model_path=MODEL_PATH,
        input_size=(320, 320),
    )

    # First run on 320x320
    img1 = np.zeros((320, 320, 3), dtype=np.uint8)
    res1 = detector.detect(img1, frame_id=1)
    assert res1.input_size == (320, 320)
    assert detector.input_size == (320, 320)

    # Second run on 640x480: downscales to (320, 240) while keeping configured input_size ceiling
    img2 = np.zeros((480, 640, 3), dtype=np.uint8)
    res2 = detector.detect(img2, frame_id=2)
    assert res2.input_size == (320, 240)
    assert detector.input_size == (320, 320)

    # Explicitly updating input_size updates the ceiling
    detector.input_size = (640, 480)
    assert detector.input_size == (640, 480)
    res3 = detector.detect(img2, frame_id=3)
    assert res3.input_size == (640, 480)


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="YuNet model weights not downloaded")
def test_yunet_detector_resolution_scaling_and_aspect_preservation() -> None:
    """Verify detector downscaling across 640x480, 1280x720, 320x320,
    small, and non-square inputs.
    """
    detector = YuNetFaceDetector(
        model_path=MODEL_PATH,
        input_size=(320, 320),
    )

    # 1. 640x480 (4:3) -> (320, 240)
    frame_vga = np.zeros((480, 640, 3), dtype=np.uint8)
    res_vga = detector.detect(frame_vga)
    assert res_vga.input_size == (320, 240)

    # 2. 1280x720 (16:9) -> (320, 180)
    frame_hd = np.zeros((720, 1280, 3), dtype=np.uint8)
    res_hd = detector.detect(frame_hd)
    assert res_hd.input_size == (320, 180)

    # 3. 320x320 (1:1) -> (320, 320) exactly
    frame_sq = np.zeros((320, 320, 3), dtype=np.uint8)
    res_sq = detector.detect(frame_sq)
    assert res_sq.input_size == (320, 320)

    # 4. Smaller input (200x150) must NOT be upscaled -> (200, 150)
    frame_small = np.zeros((150, 200, 3), dtype=np.uint8)
    res_small = detector.detect(frame_small)
    assert res_small.input_size == (200, 150)

    # 5. Non-square input (500x300) -> scale = min(320/500, 320/300) = 0.64 -> (320, 192)
    frame_custom = np.zeros((300, 500, 3), dtype=np.uint8)
    res_custom = detector.detect(frame_custom)
    assert res_custom.input_size == (320, 192)


@pytest.mark.skipif(
    not (MODEL_PATH.exists() and FIXTURE_PATH.exists()),
    reason="YuNet model weights or sample fixture not found",
)
def test_yunet_detector_coordinate_mapping_and_downstream_alignment() -> None:
    """Verify detected bounding boxes and landmarks map back to original-frame coordinates
    sensibly and integrate cleanly with FaceAligner and SFace embeddings.
    """
    from dh360_hi.vision.aligner import FaceAligner
    from dh360_hi.vision.embedder import SFaceEmbedder

    sface_model = ROOT_DIR / "models" / "weights" / "face_recognition_sface_2021dec.onnx"
    if not sface_model.exists():
        pytest.skip("SFace model weights not found")

    fixture = cv2.imread(str(FIXTURE_PATH))
    assert fixture is not None
    face_crop = cv2.resize(fixture, (200, 200))

    # Embed face into a 640x480 canvas at known position (x=220..420, y=140..340)
    canvas_640 = np.full((480, 640, 3), 120, dtype=np.uint8)
    canvas_640[140:340, 220:420] = face_crop

    detector = YuNetFaceDetector(model_path=MODEL_PATH, input_size=(320, 320), score_threshold=0.6)
    res = detector.detect(canvas_640)

    # Verified inference ran at (320, 240)
    assert res.input_size == (320, 240)
    assert len(res.detections) == 1
    det = res.detections[0]

    # Verify bounding box mapped back into sensible 640x480 canvas bounds
    bx, by, bw, bh = det.bbox
    assert 200.0 <= bx <= 350.0
    assert 120.0 <= by <= 260.0
    assert 40.0 <= bw <= 120.0
    assert 60.0 <= bh <= 150.0
    assert bx + bw <= 640.0
    assert by + bh <= 480.0

    # Verify all 5 landmarks mapped into canvas coordinate bounds
    assert det.landmarks is not None
    for lx, ly in det.landmarks.to_list():
        assert 0.0 <= lx <= 640.0
        assert 0.0 <= ly <= 480.0

    # Downstream FaceAligner test: verify alignment produces valid 112x112 crop
    aligner = FaceAligner()
    align_res = aligner.align(canvas_640, det)
    assert align_res.is_success is True
    assert align_res.aligned_face.shape == (112, 112, 3)

    # Downstream SFace test: verify embedding is valid unit vector
    embedder = SFaceEmbedder(model_path=sface_model)
    emb_res = embedder.embed(align_res.aligned_face)
    assert emb_res.is_normalized is True
    assert len(emb_res.embedding) == 128
    assert np.isclose(np.linalg.norm(emb_res.embedding), 1.0, atol=1e-3)

    # Test 1280x720 frame: fixture resized to 1280x720
    frame_hd = cv2.resize(fixture, (1280, 720))
    res_hd = detector.detect(frame_hd)
    assert res_hd.input_size == (320, 180)
    assert len(res_hd.detections) == 1
    hdx, hdy, hdw, hdh = res_hd.detections[0].bbox
    assert 450.0 <= hdx <= 650.0
    assert 180.0 <= hdy <= 320.0
    assert hdx + hdw <= 1280.0
    assert hdy + hdh <= 720.0


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

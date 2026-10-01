"""Unit tests for FaceAligner and 2D similarity transform alignment."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from dh360_hi.vision.aligner import (
    CANONICAL_5_PTS_112,
    AlignmentResult,
    FaceAligner,
    get_canonical_landmarks,
    landmarks_to_array,
)
from dh360_hi.vision.detector import FaceDetection, FaceLandmarks
from dh360_hi.vision.frame import Frame
from dh360_hi.vision.tracker import TrackedFace

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "sample_face.jpg"


def _create_sample_image() -> np.ndarray:
    """Create a 300x300 synthetic test pattern."""
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    img[50:250, 50:250] = (120, 150, 200)
    cv2.circle(img, (100, 120), 10, (255, 0, 0), -1)  # right eye
    cv2.circle(img, (200, 120), 10, (0, 255, 0), -1)  # left eye
    cv2.circle(img, (150, 170), 8, (0, 0, 255), -1)  # nose
    cv2.circle(img, (110, 220), 6, (255, 255, 0), -1)  # right mouth
    cv2.circle(img, (190, 220), 6, (0, 255, 255), -1)  # left mouth
    return img


def _create_sample_landmarks() -> FaceLandmarks:
    return FaceLandmarks(
        right_eye=(100.0, 120.0),
        left_eye=(200.0, 120.0),
        nose_tip=(150.0, 170.0),
        right_mouth_corner=(110.0, 220.0),
        left_mouth_corner=(190.0, 220.0),
    )


def test_get_canonical_landmarks_scaling() -> None:
    """Verify canonical landmarks scale proportionally with output size."""
    pts_112 = get_canonical_landmarks((112, 112))
    assert pts_112.shape == (5, 2)
    assert pts_112[0, 0] == pytest.approx(CANONICAL_5_PTS_112[0][0])
    assert pts_112[0, 1] == pytest.approx(CANONICAL_5_PTS_112[0][1])

    # Scaled 2x
    pts_224 = get_canonical_landmarks((224, 224))
    assert pts_224.shape == (5, 2)
    assert pts_224[0, 0] == pytest.approx(pts_112[0, 0] * 2.0)
    assert pts_224[0, 1] == pytest.approx(pts_112[0, 1] * 2.0)


def test_landmarks_to_array_ordering() -> None:
    """Verify landmarks_to_array packages 5 landmarks in correct order."""
    lm = _create_sample_landmarks()
    arr = landmarks_to_array(lm)
    assert arr.shape == (5, 2)
    assert np.allclose(arr[0], [100.0, 120.0])  # right eye
    assert np.allclose(arr[1], [200.0, 120.0])  # left eye
    assert np.allclose(arr[2], [150.0, 170.0])  # nose
    assert np.allclose(arr[3], [110.0, 220.0])  # right mouth
    assert np.allclose(arr[4], [190.0, 220.0])  # left mouth


def test_aligner_initialization_validation() -> None:
    """Verify constructor rejects non-positive output sizes."""
    with pytest.raises(ValueError, match="Output size must be positive"):
        FaceAligner(output_size=(0, 112))

    with pytest.raises(ValueError, match="Output size must be positive"):
        FaceAligner(output_size=(112, -10))


def test_landmark_based_similarity_alignment() -> None:
    """Verify successful 5-point similarity transformation alignment."""
    img = _create_sample_image()
    lm = _create_sample_landmarks()
    det = FaceDetection(bbox=(50.0, 50.0, 200.0, 200.0), confidence=0.95, landmarks=lm)

    aligner = FaceAligner(output_size=(112, 112))
    res = aligner.align(img, det)

    assert isinstance(res, AlignmentResult)
    assert res.is_success is True
    assert res.method == "similarity_5pt"
    assert res.output_size == (112, 112)
    assert res.aligned_face.shape == (112, 112, 3)
    assert res.aligned_face.dtype == np.uint8
    assert res.transform_matrix is not None
    assert res.transform_matrix.shape == (2, 3)


def test_deterministic_output() -> None:
    """Verify alignment is strictly deterministic across repeated invocations."""
    img = _create_sample_image()
    lm = _create_sample_landmarks()
    det = FaceDetection(bbox=(50.0, 50.0, 200.0, 200.0), confidence=0.95, landmarks=lm)

    aligner = FaceAligner(output_size=(112, 112))
    res1 = aligner.align(img, det)
    res2 = aligner.align(img, det)

    assert np.array_equal(res1.aligned_face, res2.aligned_face)
    assert np.allclose(res1.transform_matrix, res2.transform_matrix)


def test_missing_landmarks_bbox_fallback() -> None:
    """Verify fallback to bounding-box crop when landmarks are absent."""
    img = _create_sample_image()
    det_no_lm = FaceDetection(bbox=(50.0, 50.0, 200.0, 200.0), confidence=0.95, landmarks=None)

    # Fallback enabled (default)
    aligner = FaceAligner(output_size=(112, 112), fallback_to_bbox=True)
    res = aligner.align(img, det_no_lm)

    assert res.is_success is True
    assert res.method == "bbox_crop"
    assert res.aligned_face.shape == (112, 112, 3)
    assert res.transform_matrix is None

    # Fallback disabled
    aligner_no_fallback = FaceAligner(output_size=(112, 112), fallback_to_bbox=False)
    res_no_fallback = aligner_no_fallback.align(img, det_no_lm)

    assert res_no_fallback.is_success is False
    assert res_no_fallback.method == "failed"


def test_boundary_conditions_and_partial_crops() -> None:
    """Verify alignment handles face bounding boxes near or crossing image boundaries."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[0:50, 0:50] = 200

    # Face near bottom-right edge crossing image boundary
    det_edge = FaceDetection(bbox=(80.0, 80.0, 50.0, 50.0), confidence=0.9, landmarks=None)

    aligner = FaceAligner(output_size=(112, 112))
    res = aligner.align(img, det_edge)

    assert res.is_success is True
    assert res.aligned_face.shape == (112, 112, 3)


def test_invalid_image_and_degenerate_target() -> None:
    """Verify invalid inputs return failed AlignmentResult without crashing."""
    aligner = FaceAligner(output_size=(112, 112))

    # None image
    res_none = aligner.align(None, (10.0, 10.0, 50.0, 50.0))  # type: ignore[arg-type]
    assert res_none.is_success is False
    assert res_none.method == "failed"

    # Empty image
    empty_img = np.zeros((0, 0, 3), dtype=np.uint8)
    res_empty = aligner.align(empty_img, (10.0, 10.0, 50.0, 50.0))
    assert res_empty.is_success is False

    # Degenerate zero-size bbox
    valid_img = _create_sample_image()
    res_deg = aligner.align(valid_img, (10.0, 10.0, 0.0, 0.0))
    assert res_deg.is_success is False


def test_tracked_face_and_frame_inputs() -> None:
    """Verify aligner works seamlessly with Frame and TrackedFace objects."""
    img = _create_sample_image()
    frame = Frame(image=img, frame_id=1, timestamp=0.0, source_info="test")
    track = TrackedFace(
        track_id=42,
        bbox=(50.0, 50.0, 200.0, 200.0),
        confidence=0.93,
        landmarks=_create_sample_landmarks(),
    )

    aligner = FaceAligner(output_size=(112, 112))
    res = aligner.align(frame, track)

    assert res.is_success is True
    assert res.method == "similarity_5pt"
    assert res.aligned_face.shape == (112, 112, 3)


@pytest.mark.skipif(not FIXTURE_PATH.exists(), reason="Reference fixture not found")
def test_reference_fixture_alignment() -> None:
    """Verify alignment on sample_face.jpg fixture."""
    img = cv2.imread(str(FIXTURE_PATH))
    assert img is not None

    landmarks = FaceLandmarks(
        right_eye=(252.0, 259.0),
        left_eye=(323.0, 267.0),
        nose_tip=(284.0, 313.0),
        right_mouth_corner=(254.0, 351.0),
        left_mouth_corner=(309.0, 357.0),
    )
    det = FaceDetection(bbox=(207.8, 182.5, 145.9, 206.9), confidence=0.91, landmarks=landmarks)

    aligner = FaceAligner(output_size=(112, 112))
    res = aligner.align(img, det)

    assert res.is_success is True
    assert res.method == "similarity_5pt"
    assert res.aligned_face.shape == (112, 112, 3)
    assert res.aligned_face.mean() > 50.0

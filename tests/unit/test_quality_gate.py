"""Unit tests for FaceQualityGate and quality evaluation criteria."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from dh360_hi.vision.detector import FaceDetection, FaceLandmarks
from dh360_hi.vision.frame import Frame
from dh360_hi.vision.quality_gate import (
    FaceQualityGate,
    FaceQualityThresholds,
    FaceRejectionReason,
    compute_pose_metrics,
    compute_sharpness,
)
from dh360_hi.vision.tracker import TrackedFace

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "sample_face.jpg"


def _create_synthetic_sharp_face() -> np.ndarray:
    """Create a synthetic high-contrast, high-sharpness pattern."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    for i in range(0, 200, 10):
        cv2.line(img, (i, 0), (i, 200), (255, 255, 255), 2)
        cv2.line(img, (0, i), (200, i), (255, 255, 255), 2)
    return img


def _create_synthetic_landmarks(
    re: tuple[float, float] = (70.0, 70.0),
    le: tuple[float, float] = (130.0, 70.0),
    nt: tuple[float, float] = (100.0, 100.0),
    rm: tuple[float, float] = (75.0, 140.0),
    lm: tuple[float, float] = (125.0, 140.0),
) -> FaceLandmarks:
    """Helper to generate customizable 5-point face landmarks."""
    return FaceLandmarks(
        right_eye=re,
        left_eye=le,
        nose_tip=nt,
        right_mouth_corner=rm,
        left_mouth_corner=lm,
    )


def test_compute_sharpness_clean_vs_blurred() -> None:
    """Verify Laplacian variance correctly distinguishes sharp from blurry images."""
    sharp_img = _create_synthetic_sharp_face()
    blurred_img = cv2.GaussianBlur(sharp_img, (21, 21), 7.0)

    sharpness_high = compute_sharpness(sharp_img)
    sharpness_low = compute_sharpness(blurred_img)

    assert sharpness_high > 500.0
    assert sharpness_low < 50.0
    assert sharpness_high > sharpness_low * 10.0


def test_compute_pose_metrics_roll_and_yaw() -> None:
    """Verify roll angle and yaw asymmetry calculations."""
    # 1. Perfectly centered frontal face
    lm_frontal = _create_synthetic_landmarks(
        re=(70.0, 70.0),
        le=(130.0, 70.0),
        nt=(100.0, 100.0),  # d_r = 30, d_l = 30 -> yaw_ratio = 1.0
    )
    roll, yaw = compute_pose_metrics(lm_frontal)
    assert roll == pytest.approx(0.0, abs=1e-3)
    assert yaw == pytest.approx(1.0, abs=1e-3)

    # 2. Tilted face (45 degrees roll)
    # dx = 60, dy = 60 -> arctan2(60, 60) = 45 deg
    lm_tilted = _create_synthetic_landmarks(
        re=(70.0, 70.0),
        le=(130.0, 130.0),
        nt=(100.0, 100.0),
    )
    roll_tilted, _ = compute_pose_metrics(lm_tilted)
    assert roll_tilted == pytest.approx(45.0, abs=0.1)

    # 3. Turned face (yaw asymmetry: nose close to right eye)
    # d_r = 10, d_l = 50 -> yaw_ratio = 5.0
    lm_turned = _create_synthetic_landmarks(
        re=(70.0, 70.0),
        le=(130.0, 70.0),
        nt=(80.0, 100.0),
    )
    _, yaw_turned = compute_pose_metrics(lm_turned)
    assert yaw_turned == pytest.approx(50.0 / 10.0, abs=0.1)


def test_high_quality_face_accepted() -> None:
    """Verify a high-resolution, sharp, frontal face is accepted."""
    image = _create_synthetic_sharp_face()
    landmarks = _create_synthetic_landmarks()
    det = FaceDetection(
        bbox=(20.0, 20.0, 160.0, 160.0),
        confidence=0.95,
        landmarks=landmarks,
    )

    gate = FaceQualityGate()
    result = gate.evaluate(image, det)

    assert result.is_acceptable is True
    assert len(result.rejection_reasons) == 0
    assert result.metrics.confidence == 0.95
    assert result.metrics.face_width == 160.0
    assert result.metrics.face_height == 160.0
    assert result.metrics.sharpness > 50.0
    assert result.metrics.crop_valid is True


def test_low_detector_confidence_rejected() -> None:
    """Verify face with confidence below threshold is rejected with LOW_CONFIDENCE."""
    image = _create_synthetic_sharp_face()
    det = FaceDetection(
        bbox=(20.0, 20.0, 160.0, 160.0),
        confidence=0.65,  # Default threshold is 0.80
        landmarks=_create_synthetic_landmarks(),
    )

    gate = FaceQualityGate()
    result = gate.evaluate(image, det)

    assert result.is_acceptable is False
    assert FaceRejectionReason.LOW_CONFIDENCE in result.rejection_reasons


def test_face_below_minimum_size_rejected() -> None:
    """Verify face with dimensions below min_face_width/height is rejected with TOO_SMALL."""
    image = _create_synthetic_sharp_face()
    det = FaceDetection(
        bbox=(20.0, 20.0, 30.0, 30.0),  # Default threshold is 40x40
        confidence=0.95,
        landmarks=_create_synthetic_landmarks(re=(25.0, 25.0), le=(35.0, 25.0), nt=(30.0, 30.0)),
    )

    gate = FaceQualityGate()
    result = gate.evaluate(image, det)

    assert result.is_acceptable is False
    assert FaceRejectionReason.TOO_SMALL in result.rejection_reasons


def test_blurry_face_rejected() -> None:
    """Verify out-of-focus or motion-blurred face crop is rejected with BLURRY."""
    sharp_image = _create_synthetic_sharp_face()
    blurry_image = cv2.GaussianBlur(sharp_image, (25, 25), 9.0)

    det = FaceDetection(
        bbox=(20.0, 20.0, 160.0, 160.0),
        confidence=0.95,
        landmarks=_create_synthetic_landmarks(),
    )

    gate = FaceQualityGate(thresholds=FaceQualityThresholds(min_sharpness=100.0))
    result = gate.evaluate(blurry_image, det)

    assert result.is_acceptable is False
    assert FaceRejectionReason.BLURRY in result.rejection_reasons
    assert result.metrics.sharpness < 100.0


def test_excessive_roll_and_yaw_rejected() -> None:
    """Verify non-frontal faces (tilted or turned away) are rejected."""
    image = _create_synthetic_sharp_face()

    # Excessive roll tilt (45 deg > 25 deg default threshold)
    tilted_lm = _create_synthetic_landmarks(re=(70.0, 50.0), le=(130.0, 110.0))
    det_tilted = FaceDetection(
        bbox=(20.0, 20.0, 160.0, 160.0), confidence=0.95, landmarks=tilted_lm
    )

    gate = FaceQualityGate()
    res_tilted = gate.evaluate(image, det_tilted)
    assert res_tilted.is_acceptable is False
    assert FaceRejectionReason.EXCESSIVE_ROLL in res_tilted.rejection_reasons

    # Excessive yaw turn (ratio 5.0 > 2.5 default threshold)
    turned_lm = _create_synthetic_landmarks(re=(70.0, 70.0), le=(130.0, 70.0), nt=(80.0, 100.0))
    det_turned = FaceDetection(
        bbox=(20.0, 20.0, 160.0, 160.0), confidence=0.95, landmarks=turned_lm
    )
    res_turned = gate.evaluate(image, det_turned)
    assert res_turned.is_acceptable is False
    assert FaceRejectionReason.EXCESSIVE_YAW in res_turned.rejection_reasons


def test_invalid_and_out_of_bounds_bounding_box_handled_safely() -> None:
    """Verify invalid, negative, or degenerate bounding boxes are handled safely."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    gate = FaceQualityGate()

    # Negative width/height
    res_neg = gate.evaluate(image, (-10.0, -10.0, -50.0, -50.0), confidence=0.9)
    assert res_neg.is_acceptable is False
    assert FaceRejectionReason.INVALID_BBOX in res_neg.rejection_reasons

    # Box completely outside image bounds
    res_out = gate.evaluate(image, (500.0, 500.0, 100.0, 100.0), confidence=0.9)
    assert res_out.is_acceptable is False
    assert FaceRejectionReason.INVALID_BBOX in res_out.rejection_reasons


def test_missing_landmarks_handled_safely() -> None:
    """Verify missing landmarks skips pose checks by default or rejects if required."""
    image = _create_synthetic_sharp_face()
    det_no_lm = FaceDetection(bbox=(20.0, 20.0, 160.0, 160.0), confidence=0.95, landmarks=None)

    # Default: require_landmarks = False (pose metrics None, no missing_landmarks error)
    gate_default = FaceQualityGate()
    res_default = gate_default.evaluate(image, det_no_lm)
    assert res_default.is_acceptable is True
    assert res_default.metrics.roll_angle_deg is None
    assert res_default.metrics.yaw_asymmetry is None

    # Configured: require_landmarks = True
    gate_strict = FaceQualityGate(thresholds=FaceQualityThresholds(require_landmarks=True))
    res_strict = gate_strict.evaluate(image, det_no_lm)
    assert res_strict.is_acceptable is False
    assert FaceRejectionReason.MISSING_LANDMARKS in res_strict.rejection_reasons


def test_multiple_rejection_reasons_accumulated() -> None:
    """Verify multiple failing criteria are all reported in rejection_reasons."""
    # Image that is blurry, too small, and low confidence
    small_blurry_img = cv2.GaussianBlur(np.ones((20, 20, 3), dtype=np.uint8) * 100, (5, 5), 2.0)
    det = FaceDetection(
        bbox=(2.0, 2.0, 15.0, 15.0),  # too small
        confidence=0.40,  # low confidence
        landmarks=None,
    )

    gate = FaceQualityGate()
    result = gate.evaluate(small_blurry_img, det)

    assert result.is_acceptable is False
    assert FaceRejectionReason.LOW_CONFIDENCE in result.rejection_reasons
    assert FaceRejectionReason.TOO_SMALL in result.rejection_reasons
    assert FaceRejectionReason.BLURRY in result.rejection_reasons
    assert len(result.rejection_reasons) >= 3


def test_configurable_thresholds_modify_outcome() -> None:
    """Verify modifying thresholds dynamically alters evaluation results."""
    image = _create_synthetic_sharp_face()
    det = FaceDetection(
        bbox=(20.0, 20.0, 50.0, 50.0),
        confidence=0.75,
        landmarks=_create_synthetic_landmarks(),
    )

    # With default min_confidence=0.80 -> rejected
    gate = FaceQualityGate()
    assert gate.evaluate(image, det).is_acceptable is False

    # Lower threshold to 0.70 -> accepted
    gate.thresholds = FaceQualityThresholds(min_confidence=0.70)
    assert gate.evaluate(image, det).is_acceptable is True


def test_tracked_face_and_frame_container_inputs() -> None:
    """Verify FaceQualityGate evaluates TrackedFace and Frame objects directly."""
    image = _create_synthetic_sharp_face()
    frame = Frame(image=image, frame_id=1, timestamp=0.0, source_info="test")
    track = TrackedFace(
        track_id=1,
        bbox=(20.0, 20.0, 160.0, 160.0),
        confidence=0.92,
        landmarks=_create_synthetic_landmarks(),
    )

    gate = FaceQualityGate()
    res = gate.evaluate(frame, track)
    assert res.is_acceptable is True
    assert res.metrics.confidence == 0.92


@pytest.mark.skipif(not FIXTURE_PATH.exists(), reason="Reference fixture not found")
def test_evaluation_on_reference_fixture() -> None:
    """Verify quality gate on real reference fixture sample_face.jpg."""
    img = cv2.imread(str(FIXTURE_PATH))
    assert img is not None

    # Detected face coordinates on sample_face.jpg
    bbox = (207.8, 182.5, 145.9, 206.9)
    landmarks = FaceLandmarks(
        right_eye=(252.0, 259.0),
        left_eye=(323.0, 267.0),
        nose_tip=(284.0, 313.0),
        right_mouth_corner=(254.0, 351.0),
        left_mouth_corner=(309.0, 357.0),
    )

    gate = FaceQualityGate()
    res = gate.evaluate(img, bbox, confidence=0.91, landmarks=landmarks)
    assert res.is_acceptable is True
    assert res.metrics.sharpness > 200.0
    assert res.metrics.face_width > 100.0


def test_classify_head_direction_all_five_directions() -> None:
    """Verify classify_head_direction for FRONT, RIGHT, LEFT, UP, and DOWN."""
    from dh360_hi.vision.quality_gate import HeadDirection, classify_head_direction

    # 1. Frontal face: nose centered horizontally and vertically between eyes and mouth
    lm_front = _create_synthetic_landmarks(
        re=(70.0, 70.0),
        le=(130.0, 70.0),
        nt=(100.0, 104.3),  # (104.3 - 70) / 70 ~= 0.49
        rm=(75.0, 140.0),
        lm=(125.0, 140.0),
    )
    assert classify_head_direction(lm_front) == HeadDirection.FRONT

    # 2. Right turned face: nose shifted towards right eye (image left, x=88)
    # eyes_mid = 100, (100 - 88) / 60 = +0.20 > 0.12
    lm_right = _create_synthetic_landmarks(
        re=(70.0, 70.0),
        le=(130.0, 70.0),
        nt=(88.0, 104.3),
        rm=(75.0, 140.0),
        lm=(125.0, 140.0),
    )
    assert classify_head_direction(lm_right) == HeadDirection.RIGHT

    # 3. Left turned face: nose shifted towards left eye (image right, x=112)
    # eyes_mid = 100, (100 - 112) / 60 = -0.20 < -0.12
    lm_left = _create_synthetic_landmarks(
        re=(70.0, 70.0),
        le=(130.0, 70.0),
        nt=(112.0, 104.3),
        rm=(75.0, 140.0),
        lm=(125.0, 140.0),
    )
    assert classify_head_direction(lm_left) == HeadDirection.LEFT

    # 4. Looking UP: nose shifted higher towards eyes (y=90 instead of 104.3)
    # (90 - 70) / 70 = 0.286 -> offset = 0.286 - 0.49 = -0.204 <= -0.08
    lm_up = _create_synthetic_landmarks(
        re=(70.0, 70.0),
        le=(130.0, 70.0),
        nt=(100.0, 90.0),
        rm=(75.0, 140.0),
        lm=(125.0, 140.0),
    )
    assert classify_head_direction(lm_up) == HeadDirection.UP

    # 5. Looking DOWN: nose shifted lower towards mouth (y=118 instead of 104.3)
    # (118 - 70) / 70 = 0.686 -> offset = 0.686 - 0.50 = +0.186 >= +0.08
    lm_down = _create_synthetic_landmarks(
        re=(70.0, 70.0),
        le=(130.0, 70.0),
        nt=(100.0, 118.0),
        rm=(75.0, 140.0),
        lm=(125.0, 140.0),
    )
    assert classify_head_direction(lm_down) == HeadDirection.DOWN


def test_realistic_webcam_poses_and_opposite_rejections() -> None:
    """Verify calibrated realistic webcam poses (RIGHT, LEFT, UP, DOWN, FRONT)

    and ensure opposite directions are rejected and FRONT does not count for others.
    """
    from dh360_hi.vision.quality_gate import HeadDirection, classify_head_direction

    # Realistic moderate RIGHT pose (nose offset +0.10: nt_x=94, eyes_mid=100, iod=60)
    lm_right = _create_synthetic_landmarks(
        re=(70.0, 70.0),
        le=(130.0, 70.0),
        nt=(94.0, 105.0),
        rm=(75.0, 140.0),
        lm=(125.0, 140.0),
    )
    # Realistic moderate LEFT pose (nose offset -0.10: nt_x=106, eyes_mid=100, iod=60)
    lm_left = _create_synthetic_landmarks(
        re=(70.0, 70.0),
        le=(130.0, 70.0),
        nt=(106.0, 105.0),
        rm=(75.0, 140.0),
        lm=(125.0, 140.0),
    )
    # Realistic UP pose (nose higher, y=92, span=70 -> ratio=0.314)
    lm_up = _create_synthetic_landmarks(
        re=(70.0, 70.0),
        le=(130.0, 70.0),
        nt=(100.0, 92.0),
        rm=(75.0, 140.0),
        lm=(125.0, 140.0),
    )
    # Realistic DOWN pose (nose lower, y=116, span=70 -> ratio=0.657)
    lm_down = _create_synthetic_landmarks(
        re=(70.0, 70.0),
        le=(130.0, 70.0),
        nt=(100.0, 116.0),
        rm=(75.0, 140.0),
        lm=(125.0, 140.0),
    )
    # Clean FRONT pose
    lm_front = _create_synthetic_landmarks(
        re=(70.0, 70.0),
        le=(130.0, 70.0),
        nt=(100.0, 105.0),
        rm=(75.0, 140.0),
        lm=(125.0, 140.0),
    )

    # 1. Correct classifications
    assert classify_head_direction(lm_right) == HeadDirection.RIGHT
    assert classify_head_direction(lm_left) == HeadDirection.LEFT
    assert classify_head_direction(lm_up) == HeadDirection.UP
    assert classify_head_direction(lm_down) == HeadDirection.DOWN
    assert classify_head_direction(lm_front) == HeadDirection.FRONT

    # 2. Opposite direction rejection
    assert classify_head_direction(lm_right) != HeadDirection.LEFT
    assert classify_head_direction(lm_left) != HeadDirection.RIGHT
    assert classify_head_direction(lm_up) != HeadDirection.DOWN
    assert classify_head_direction(lm_down) != HeadDirection.UP

    # 3. FRONT does not count as any directional pose
    assert classify_head_direction(lm_front) != HeadDirection.RIGHT
    assert classify_head_direction(lm_front) != HeadDirection.LEFT
    assert classify_head_direction(lm_front) != HeadDirection.UP
    assert classify_head_direction(lm_front) != HeadDirection.DOWN

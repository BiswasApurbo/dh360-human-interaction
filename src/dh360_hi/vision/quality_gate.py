"""Face quality gating module for filtering low-quality or non-frontal faces before recognition."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum

import cv2
import numpy as np

from dh360_hi.vision.detector import FaceDetection, FaceLandmarks
from dh360_hi.vision.frame import Frame
from dh360_hi.vision.tracker import TrackedFace


class FaceRejectionReason(StrEnum):
    """Enumeration of quality criteria failure reasons."""

    INVALID_IMAGE = "invalid_image"
    INVALID_BBOX = "invalid_bbox"
    LOW_CONFIDENCE = "low_confidence"
    TOO_SMALL = "too_small"
    BLURRY = "blurry"
    EXCESSIVE_ROLL = "excessive_roll"
    EXCESSIVE_YAW = "excessive_yaw"
    MISSING_LANDMARKS = "missing_landmarks"


class FaceQualityTier(StrEnum):
    """Quality tiers for categorizing face suitability for downstream embedding."""

    REJECTED = "rejected"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class FaceQualityThresholds:
    """Configurable thresholds for face quality evaluation.

    Note:
        These values represent engineering baseline defaults and are subject
        to empirical tuning for specific deployment lighting, cameras, and environments.
    """

    min_confidence: float = 0.80
    min_face_width: float = 40.0
    min_face_height: float = 40.0
    min_sharpness: float = 50.0  # Laplacian variance threshold
    max_roll_angle_deg: float = 25.0  # In-plane head tilt
    max_yaw_asymmetry: float = 2.5  # Ratio between left/right eye-to-nose horizontal distances
    require_landmarks: bool = False  # If True, missing landmarks causes rejection


@dataclass(frozen=True)
class FaceQualityMetrics:
    """Numerical quality measurements extracted from a face crop and its landmarks."""

    confidence: float
    face_width: float
    face_height: float
    sharpness: float  # Variance of Laplacian on grayscale face crop
    roll_angle_deg: float | None = None
    yaw_asymmetry: float | None = None
    crop_valid: bool = True


@dataclass(frozen=True)
class FaceQualityResult:
    """Evaluation result for a single face against quality gate criteria."""

    is_acceptable: bool
    metrics: FaceQualityMetrics
    rejection_reasons: list[FaceRejectionReason]
    thresholds: FaceQualityThresholds
    tier: FaceQualityTier = FaceQualityTier.MEDIUM

    @property
    def has_rejection_reason(self) -> bool:
        """Check if any rejection reasons were recorded."""
        return len(self.rejection_reasons) > 0


def compute_sharpness(crop: np.ndarray) -> float:
    """Compute sharpness metric using the variance of the Laplacian operator on grayscale."""
    if crop.size == 0:
        return 0.0

    if len(crop.shape) == 3 and crop.shape[2] == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    elif len(crop.shape) == 2:
        gray = crop
    else:
        return 0.0

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


def compute_pose_metrics(
    landmarks: FaceLandmarks,
) -> tuple[float, float]:
    """Compute roll angle (degrees) and yaw asymmetry ratio from 5-point landmarks.

    Returns:
        tuple of (roll_angle_deg, yaw_asymmetry_ratio)
    """
    re_x, re_y = landmarks.right_eye
    le_x, le_y = landmarks.left_eye
    nt_x, _ = landmarks.nose_tip

    # Roll angle: angle of inter-ocular axis relative to horizontal
    dx = le_x - re_x
    dy = le_y - re_y
    roll_angle_deg = float(abs(np.degrees(np.arctan2(dy, dx))))

    # Yaw asymmetry ratio: comparison of horizontal distance from nose to each eye
    d_right = abs(nt_x - re_x)
    d_left = abs(le_x - nt_x)
    min_dist = max(min(d_right, d_left), 1e-4)
    max_dist = max(d_right, d_left)
    yaw_asymmetry = float(max_dist / min_dist)

    return roll_angle_deg, yaw_asymmetry


class HeadDirection(StrEnum):
    """Head gaze/pose orientation relative to camera for guided biometric enrollment."""

    FRONT = "front"
    RIGHT = "right"
    LEFT = "left"
    UP = "up"
    DOWN = "down"


def compute_face_pitch_ratio(landmarks: FaceLandmarks) -> float:
    """Compute vertical position ratio of the nose tip between eyes and mouth.

    Returns:
        float: (nose_tip_y - eyes_mid_y) / vert_span
    """
    eyes_mid_y = (landmarks.right_eye[1] + landmarks.left_eye[1]) / 2.0
    mouth_mid_y = (landmarks.right_mouth_corner[1] + landmarks.left_mouth_corner[1]) / 2.0
    vert_span = max(abs(mouth_mid_y - eyes_mid_y), 1e-4)
    return float((landmarks.nose_tip[1] - eyes_mid_y) / vert_span)


def compute_yaw_offset(
    landmarks: FaceLandmarks,
    *,
    mirrored: bool = False,
) -> float:
    """Compute signed roll-corrected horizontal yaw offset of the nose tip.

    Projects the displacement from the eye midpoint to the nose tip onto the
    unit inter-ocular axis (from ``right_eye`` to ``left_eye``), normalized by
    the inter-ocular distance (IOD). Projecting onto the inter-ocular vector
    makes horizontal yaw estimation invariant to in-plane head roll/tilt.

    Sign convention (unmirrored camera frame, ``mirrored=False``):
    - ``right_eye`` is on image-left (smaller x) and ``left_eye`` is on image-right.
    - Positive ``yaw_offset`` (> 0): nose shifts toward ``right_eye`` (subject turning
      their head to their own physical RIGHT).
    - Negative ``yaw_offset`` (< 0): nose shifts toward ``left_eye`` (subject turning
      their head to their own physical LEFT).

    When ``mirrored=True`` (horizontally flipped camera stream), the sign is inverted
    so positive still represents the subject's physical RIGHT.
    """
    re_x, re_y = landmarks.right_eye
    le_x, le_y = landmarks.left_eye
    nt_x, nt_y = landmarks.nose_tip

    eye_dx = le_x - re_x
    eye_dy = le_y - re_y
    iod_sq = float(eye_dx * eye_dx + eye_dy * eye_dy)
    if iod_sq < 1e-8:
        return 0.0

    eyes_mid_x = (re_x + le_x) / 2.0
    eyes_mid_y = (re_y + le_y) / 2.0

    # Project (eyes_mid - nose_tip) onto (left_eye - right_eye) normalized by iod^2.
    yaw_offset = float(((eyes_mid_x - nt_x) * eye_dx + (eyes_mid_y - nt_y) * eye_dy) / iod_sq)
    if mirrored:
        yaw_offset = -yaw_offset
    return yaw_offset


def classify_head_direction(
    landmarks: FaceLandmarks,
    yaw_threshold: float = 0.08,
    pitch_up_threshold: float = 0.06,
    pitch_down_threshold: float = 0.06,
    baseline_pitch_ratio: float = 0.50,
    *,
    mirrored: bool = False,
    include_vertical: bool = True,
) -> HeadDirection:
    """Classify 5-point facial landmarks into a discrete head orientation direction.

    Geometry:
    - Eye midpoint and inter-ocular vector define scale and roll-corrected horizontal axis.
    - Yaw is computed via :func:`compute_yaw_offset` by projecting the nose-to-eye-midpoint
      vector onto the inter-ocular axis, normalized by inter-ocular distance. Positive
      deviation indicates the subject turning their head RIGHT; negative indicates LEFT.
    - When ``include_vertical=True`` (default), pitch is also evaluated from the vertical
      position of the nose tip relative to the eye-to-mouth span against ``baseline_pitch_ratio``.
      When ``include_vertical=False`` (3-stage horizontal guided enrollment: FRONT/RIGHT/LEFT),
      classification is restricted to horizontal pose directions (FRONT, RIGHT, LEFT) so
      natural vertical facial proportions or slight pitch shifts cannot misclassify horizontal
      stages as UP/DOWN.
    """
    yaw_offset = compute_yaw_offset(landmarks, mirrored=mirrored)
    is_yaw = abs(yaw_offset) >= yaw_threshold

    if not include_vertical:
        if is_yaw:
            return HeadDirection.RIGHT if yaw_offset > 0 else HeadDirection.LEFT
        return HeadDirection.FRONT

    # Pitch offset: vertical position of nose within eye-to-mouth span
    current_pitch_ratio = compute_face_pitch_ratio(landmarks)
    pitch_offset = current_pitch_ratio - baseline_pitch_ratio

    # Determine dominant deviation
    is_pitch_up = pitch_offset <= -pitch_up_threshold
    is_pitch_down = pitch_offset >= pitch_down_threshold

    if not is_yaw and not is_pitch_up and not is_pitch_down:
        return HeadDirection.FRONT

    yaw_mag = abs(yaw_offset)
    pitch_mag = abs(pitch_offset)

    # When both horizontal turn and vertical tilt reach threshold,
    # the larger deviation determines the orientation.
    if is_yaw and (is_pitch_up or is_pitch_down):
        if pitch_mag >= yaw_mag:
            return HeadDirection.UP if is_pitch_up else HeadDirection.DOWN
        return HeadDirection.RIGHT if yaw_offset > 0 else HeadDirection.LEFT

    if is_yaw:
        return HeadDirection.RIGHT if yaw_offset > 0 else HeadDirection.LEFT

    if is_pitch_up:
        return HeadDirection.UP
    if is_pitch_down:
        return HeadDirection.DOWN

    return HeadDirection.FRONT


class BaseFaceQualityGate(ABC):
    """Abstract interface for face quality evaluation."""

    @abstractmethod
    def evaluate(
        self,
        frame: Frame | np.ndarray,
        target: FaceDetection | TrackedFace | tuple[float, float, float, float],
        confidence: float | None = None,
        landmarks: FaceLandmarks | None = None,
    ) -> FaceQualityResult:
        """Evaluate a detected or tracked face for recognition suitability."""

    @property
    @abstractmethod
    def thresholds(self) -> FaceQualityThresholds:
        """Current quality thresholds."""

    @thresholds.setter
    @abstractmethod
    def thresholds(self, thresholds: FaceQualityThresholds) -> None:
        """Update quality thresholds."""


class FaceQualityGate(BaseFaceQualityGate):
    """Standard face quality gate based on size, confidence, Laplacian sharpness, and pose."""

    def __init__(self, thresholds: FaceQualityThresholds | None = None) -> None:
        self._thresholds = thresholds or FaceQualityThresholds()

    @property
    def thresholds(self) -> FaceQualityThresholds:
        return self._thresholds

    @thresholds.setter
    def thresholds(self, thresholds: FaceQualityThresholds) -> None:
        self._thresholds = thresholds

    def evaluate(
        self,
        frame: Frame | np.ndarray,
        target: FaceDetection | TrackedFace | tuple[float, float, float, float],
        confidence: float | None = None,
        landmarks: FaceLandmarks | None = None,
    ) -> FaceQualityResult:
        """Evaluate face quality against configured thresholds."""
        # 1. Extract image array
        image = frame.image if isinstance(frame, Frame) else frame

        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return FaceQualityResult(
                is_acceptable=False,
                metrics=FaceQualityMetrics(
                    confidence=0.0,
                    face_width=0.0,
                    face_height=0.0,
                    sharpness=0.0,
                    crop_valid=False,
                ),
                rejection_reasons=[FaceRejectionReason.INVALID_IMAGE],
                thresholds=self._thresholds,
            )

        # 2. Extract bounding box, confidence, and landmarks from target
        if isinstance(target, (FaceDetection, TrackedFace)):
            bbox = target.bbox
            target_conf = target.confidence if confidence is None else confidence
            target_lm = target.landmarks if landmarks is None else landmarks
        else:
            bbox = target
            target_conf = 1.0 if confidence is None else confidence
            target_lm = landmarks

        x, y, w, h = bbox
        img_h, img_w = image.shape[:2]

        rejection_reasons: list[FaceRejectionReason] = []

        # 3. Check Confidence
        if target_conf < self._thresholds.min_confidence:
            rejection_reasons.append(FaceRejectionReason.LOW_CONFIDENCE)

        # 4. Check Dimensions & Bounds
        if w <= 0 or h <= 0 or not np.isfinite(x) or not np.isfinite(y):
            return FaceQualityResult(
                is_acceptable=False,
                metrics=FaceQualityMetrics(
                    confidence=target_conf,
                    face_width=max(0.0, w),
                    face_height=max(0.0, h),
                    sharpness=0.0,
                    crop_valid=False,
                ),
                rejection_reasons=[FaceRejectionReason.INVALID_BBOX],
                thresholds=self._thresholds,
            )

        if w < self._thresholds.min_face_width or h < self._thresholds.min_face_height:
            rejection_reasons.append(FaceRejectionReason.TOO_SMALL)

        # 5. Extract and Crop Face
        x1 = max(0, int(np.floor(x)))
        y1 = max(0, int(np.floor(y)))
        x2 = min(img_w, int(np.ceil(x + w)))
        y2 = min(img_h, int(np.ceil(y + h)))

        crop = image[y1:y2, x1:x2]
        if crop.size == 0 or (x2 - x1) <= 0 or (y2 - y1) <= 0:
            rejection_reasons.append(FaceRejectionReason.INVALID_BBOX)
            sharpness = 0.0
            crop_valid = False
        else:
            sharpness = compute_sharpness(crop)
            crop_valid = True
            if sharpness < self._thresholds.min_sharpness:
                rejection_reasons.append(FaceRejectionReason.BLURRY)

        # 6. Pose & Frontality Evaluation
        roll_angle_deg: float | None = None
        yaw_asymmetry: float | None = None

        if target_lm is not None:
            roll_angle_deg, yaw_asymmetry = compute_pose_metrics(target_lm)

            if roll_angle_deg > self._thresholds.max_roll_angle_deg:
                rejection_reasons.append(FaceRejectionReason.EXCESSIVE_ROLL)

            if yaw_asymmetry > self._thresholds.max_yaw_asymmetry:
                rejection_reasons.append(FaceRejectionReason.EXCESSIVE_YAW)
        elif self._thresholds.require_landmarks:
            rejection_reasons.append(FaceRejectionReason.MISSING_LANDMARKS)

        metrics = FaceQualityMetrics(
            confidence=target_conf,
            face_width=float(w),
            face_height=float(h),
            sharpness=sharpness,
            roll_angle_deg=roll_angle_deg,
            yaw_asymmetry=yaw_asymmetry,
            crop_valid=crop_valid,
        )

        is_acceptable = len(rejection_reasons) == 0
        tier = compute_quality_tier(is_acceptable, metrics, self._thresholds)

        return FaceQualityResult(
            is_acceptable=is_acceptable,
            metrics=metrics,
            rejection_reasons=rejection_reasons,
            thresholds=self._thresholds,
            tier=tier,
        )


def compute_quality_tier(
    is_acceptable: bool,
    metrics: FaceQualityMetrics,
    thresholds: FaceQualityThresholds,
) -> FaceQualityTier:
    """Classify face quality metrics into discrete operational tiers."""
    if not is_acceptable or not metrics.crop_valid:
        return FaceQualityTier.REJECTED

    # High tier: superior confidence, sharpness, resolution, and near-perfect frontality
    is_high_conf = metrics.confidence >= max(0.90, thresholds.min_confidence + 0.05)
    is_high_sharpness = metrics.sharpness >= max(80.0, thresholds.min_sharpness * 1.5)
    is_high_size = metrics.face_width >= max(
        60.0, thresholds.min_face_width * 1.5
    ) and metrics.face_height >= max(60.0, thresholds.min_face_height * 1.5)
    is_high_pose = (metrics.roll_angle_deg is None or metrics.roll_angle_deg <= 12.0) and (
        metrics.yaw_asymmetry is None or metrics.yaw_asymmetry <= 1.6
    )

    if is_high_conf and is_high_sharpness and is_high_size and is_high_pose:
        return FaceQualityTier.HIGH

    # Medium tier: satisfies standard quality thresholds
    is_med_conf = metrics.confidence >= thresholds.min_confidence
    is_med_sharpness = metrics.sharpness >= thresholds.min_sharpness
    is_med_size = (
        metrics.face_width >= thresholds.min_face_width
        and metrics.face_height >= thresholds.min_face_height
    )
    is_med_pose = (
        metrics.roll_angle_deg is None or metrics.roll_angle_deg <= thresholds.max_roll_angle_deg
    ) and (metrics.yaw_asymmetry is None or metrics.yaw_asymmetry <= thresholds.max_yaw_asymmetry)

    if is_med_conf and is_med_sharpness and is_med_size and is_med_pose:
        return FaceQualityTier.MEDIUM

    return FaceQualityTier.LOW

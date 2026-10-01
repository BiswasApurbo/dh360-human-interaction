"""Face alignment module for normalizing face orientation, scale, and position."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import cv2
import numpy as np

from dh360_hi.vision.detector import FaceDetection, FaceLandmarks
from dh360_hi.vision.frame import Frame
from dh360_hi.vision.tracker import TrackedFace

# Standard ArcFace/InsightFace/SFace canonical 5-point landmarks on 112x112 canvas
# Coordinates in (x, y) pixels relative to (112, 112)
CANONICAL_5_PTS_112: tuple[tuple[float, float], ...] = (
    (38.2946, 51.6963),  # right eye (subject right / image left)
    (73.5318, 51.5014),  # left eye (subject left / image right)
    (56.0252, 71.7366),  # nose tip
    (41.5493, 92.3655),  # right mouth corner
    (70.7299, 92.2041),  # left mouth corner
)


@dataclass(frozen=True)
class AlignmentResult:
    """Result of face alignment transformation."""

    aligned_face: np.ndarray
    is_success: bool
    method: str
    output_size: tuple[int, int]
    transform_matrix: np.ndarray | None = None


def get_canonical_landmarks(output_size: tuple[int, int] = (112, 112)) -> np.ndarray:
    """Scale standard 5-point canonical landmarks to the target output size (width, height)."""
    target_w, target_h = output_size
    scale_x = target_w / 112.0
    scale_y = target_h / 112.0

    scaled = []
    for x, y in CANONICAL_5_PTS_112:
        scaled.append([x * scale_x, y * scale_y])

    return np.array(scaled, dtype=np.float32)


def landmarks_to_array(landmarks: FaceLandmarks) -> np.ndarray:
    """Convert a FaceLandmarks instance to a (5, 2) float32 array in canonical order."""
    return np.array(
        [
            landmarks.right_eye,
            landmarks.left_eye,
            landmarks.nose_tip,
            landmarks.right_mouth_corner,
            landmarks.left_mouth_corner,
        ],
        dtype=np.float32,
    )


class BaseFaceAligner(ABC):
    """Abstract interface for face alignment."""

    @abstractmethod
    def align(
        self,
        frame: Frame | np.ndarray,
        target: FaceDetection | TrackedFace | tuple[float, float, float, float],
        landmarks: FaceLandmarks | None = None,
    ) -> AlignmentResult:
        """Align a detected or tracked face to canonical geometry."""

    @property
    @abstractmethod
    def output_size(self) -> tuple[int, int]:
        """Target canonical face dimensions (width, height)."""


class FaceAligner(BaseFaceAligner):
    """Deterministic 2D similarity transform face aligner using 5 facial landmarks."""

    def __init__(
        self,
        output_size: tuple[int, int] = (112, 112),
        fallback_to_bbox: bool = True,
    ) -> None:
        if output_size[0] <= 0 or output_size[1] <= 0:
            msg = f"Output size must be positive, got {output_size}"
            raise ValueError(msg)

        self._output_size = output_size
        self._fallback_to_bbox = fallback_to_bbox
        self._canonical_pts = get_canonical_landmarks(output_size)

    @property
    def output_size(self) -> tuple[int, int]:
        return self._output_size

    def align(
        self,
        frame: Frame | np.ndarray,
        target: FaceDetection | TrackedFace | tuple[float, float, float, float],
        landmarks: FaceLandmarks | None = None,
    ) -> AlignmentResult:
        """Align face using 2D similarity transform or fallback bounding box crop."""
        image = frame.image if isinstance(frame, Frame) else frame

        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return AlignmentResult(
                aligned_face=np.zeros(
                    (self._output_size[1], self._output_size[0], 3), dtype=np.uint8
                ),
                is_success=False,
                method="failed",
                output_size=self._output_size,
                transform_matrix=None,
            )

        # Extract bbox and landmarks from target
        if isinstance(target, (FaceDetection, TrackedFace)):
            bbox = target.bbox
            target_lm = target.landmarks if landmarks is None else landmarks
        else:
            bbox = target
            target_lm = landmarks

        # 1. Attempt Landmark-based Similarity Alignment
        if target_lm is not None:
            src_pts = landmarks_to_array(target_lm)
            if np.all(np.isfinite(src_pts)):
                matrix, _ = cv2.estimateAffinePartial2D(
                    src_pts, self._canonical_pts, method=cv2.LMEDS
                )
                if matrix is not None and np.all(np.isfinite(matrix)):
                    aligned = cv2.warpAffine(
                        image,
                        matrix,
                        self._output_size,
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(0, 0, 0),
                    )
                    return AlignmentResult(
                        aligned_face=aligned,
                        is_success=True,
                        method="similarity_5pt",
                        output_size=self._output_size,
                        transform_matrix=matrix,
                    )

        # 2. Bounding-Box Fallback Alignment (if enabled)
        if self._fallback_to_bbox:
            x, y, w, h = bbox
            img_h, img_w = image.shape[:2]

            if w > 0 and h > 0 and np.isfinite(x) and np.isfinite(y):
                x1 = max(0, int(np.floor(x)))
                y1 = max(0, int(np.floor(y)))
                x2 = min(img_w, int(np.ceil(x + w)))
                y2 = min(img_h, int(np.ceil(y + h)))

                crop = image[y1:y2, x1:x2]
                if crop.size > 0 and crop.shape[0] > 0 and crop.shape[1] > 0:
                    aligned = cv2.resize(
                        crop,
                        self._output_size,
                        interpolation=cv2.INTER_LINEAR,
                    )
                    return AlignmentResult(
                        aligned_face=aligned,
                        is_success=True,
                        method="bbox_crop",
                        output_size=self._output_size,
                        transform_matrix=None,
                    )

        # 3. Alignment Failure
        return AlignmentResult(
            aligned_face=np.zeros((self._output_size[1], self._output_size[0], 3), dtype=np.uint8),
            is_success=False,
            method="failed",
            output_size=self._output_size,
            transform_matrix=None,
        )

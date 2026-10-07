"""Face detector interface and YuNet ONNX implementation."""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from dh360_hi.vision.frame import Frame


@dataclass(frozen=True)
class FaceLandmarks:
    """Five-point 2D facial landmark coordinates (x, y)."""

    right_eye: tuple[float, float]
    left_eye: tuple[float, float]
    nose_tip: tuple[float, float]
    right_mouth_corner: tuple[float, float]
    left_mouth_corner: tuple[float, float]

    def to_list(self) -> list[tuple[float, float]]:
        """Return landmarks as a list of (x, y) coordinate tuples."""
        return [
            self.right_eye,
            self.left_eye,
            self.nose_tip,
            self.right_mouth_corner,
            self.left_mouth_corner,
        ]


@dataclass(frozen=True)
class FaceDetection:
    """Represents a single detected face bounding box, score, and landmarks."""

    bbox: tuple[float, float, float, float]  # (x, y, width, height) in pixels
    confidence: float
    landmarks: FaceLandmarks | None = None

    @property
    def x(self) -> float:
        return self.bbox[0]

    @property
    def y(self) -> float:
        return self.bbox[1]

    @property
    def width(self) -> float:
        return self.bbox[2]

    @property
    def height(self) -> float:
        return self.bbox[3]

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass(frozen=True)
class DetectionResult:
    """Output of a detector execution on a single frame."""

    detections: list[FaceDetection]
    frame_id: int
    latency_ms: float
    input_size: tuple[int, int]  # (width, height) used for detector inference


class BaseFaceDetector(ABC):
    """Abstract interface for face detection models."""

    @abstractmethod
    def detect(self, frame: Frame | np.ndarray, frame_id: int = 0) -> DetectionResult:
        """Execute face detection on the input frame.

        Args:
            frame: A Frame instance or raw BGR NumPy image array.
            frame_id: Sequential frame index (used when frame is a raw np.ndarray).

        Returns:
            DetectionResult containing list of FaceDetection objects and metrics.
        """

    @property
    @abstractmethod
    def input_size(self) -> tuple[int, int]:
        """Current detector input size (width, height)."""

    @input_size.setter
    @abstractmethod
    def input_size(self, size: tuple[int, int]) -> None:
        """Update detector input size."""

    @property
    @abstractmethod
    def confidence_threshold(self) -> float:
        """Minimum confidence threshold for keeping detections."""

    @confidence_threshold.setter
    @abstractmethod
    def confidence_threshold(self, threshold: float) -> None:
        """Update confidence threshold."""


class YuNetFaceDetector(BaseFaceDetector):
    """YuNet Face Detector backed by OpenCV FaceDetectorYN and ONNX weights."""

    def __init__(
        self,
        model_path: str | Path,
        input_size: tuple[int, int] = (320, 320),
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
        backend_id: int = cv2.dnn.DNN_BACKEND_OPENCV,
        target_id: int = cv2.dnn.DNN_TARGET_CPU,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"YuNet ONNX model file not found at: {self.model_path}\n"
                f"Please run 'python scripts/download_models.py' to download model weights."
            )

        self._lock = threading.RLock()
        self._input_size = input_size
        self._score_threshold = float(score_threshold)
        self._nms_threshold = float(nms_threshold)
        self._top_k = int(top_k)
        self._backend_id = backend_id
        self._target_id = target_id

        self._detector: cv2.FaceDetectorYN = cv2.FaceDetectorYN.create(
            model=str(self.model_path),
            config="",
            input_size=self._input_size,
            score_threshold=self._score_threshold,
            nms_threshold=self._nms_threshold,
            top_k=self._top_k,
            backend_id=self._backend_id,
            target_id=self._target_id,
        )

    def detect(self, frame: Frame | np.ndarray, frame_id: int = 0) -> DetectionResult:
        """Run YuNet inference on a single frame."""
        if isinstance(frame, Frame):
            image = frame.image
            current_frame_id = frame.frame_id
        else:
            image = frame
            current_frame_id = frame_id

        if image is None or image.size == 0 or image.ndim < 2:
            return DetectionResult(
                detections=[],
                frame_id=current_frame_id,
                latency_ms=0.0,
                input_size=self._input_size,
            )

        # Normalize channel layout: YuNet expects 3-channel BGR
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        elif image.ndim == 3 and image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Ensure contiguous uint8 array
        if not image.flags.c_contiguous:
            image = np.ascontiguousarray(image)
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        height, width = image.shape[:2]
        frame_size = (width, height)

        with self._lock:
            # Update input size on detector if frame dimensions changed
            if self._input_size != frame_size:
                self.input_size = frame_size

            start_time = time.perf_counter()
            _, raw_faces = self._detector.detect(image)
            latency_ms = (time.perf_counter() - start_time) * 1000.0

            detections: list[FaceDetection] = []
            if raw_faces is not None and len(raw_faces) > 0:
                for face_row in raw_faces:
                    # Raw row format: [x, y, w, h, re_x, re_y, le_x, le_y, nt_x, nt_y, ...]
                    bbox = (
                        float(face_row[0]),
                        float(face_row[1]),
                        float(face_row[2]),
                        float(face_row[3]),
                    )
                    landmarks = FaceLandmarks(
                        right_eye=(float(face_row[4]), float(face_row[5])),
                        left_eye=(float(face_row[6]), float(face_row[7])),
                        nose_tip=(float(face_row[8]), float(face_row[9])),
                        right_mouth_corner=(float(face_row[10]), float(face_row[11])),
                        left_mouth_corner=(float(face_row[12]), float(face_row[13])),
                    )
                    score = float(face_row[14])
                    detections.append(
                        FaceDetection(
                            bbox=bbox,
                            confidence=score,
                            landmarks=landmarks,
                        )
                    )

            return DetectionResult(
                detections=detections,
                frame_id=current_frame_id,
                latency_ms=latency_ms,
                input_size=self._input_size,
            )

    @property
    def input_size(self) -> tuple[int, int]:
        with self._lock:
            return self._input_size

    @input_size.setter
    def input_size(self, size: tuple[int, int]) -> None:
        with self._lock:
            self._input_size = size
            self._detector.setInputSize(self._input_size)

    @property
    def confidence_threshold(self) -> float:
        with self._lock:
            return self._score_threshold

    @confidence_threshold.setter
    def confidence_threshold(self, threshold: float) -> None:
        with self._lock:
            self._score_threshold = float(threshold)
            self._detector.setScoreThreshold(self._score_threshold)

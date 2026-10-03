"""Scripted perception components for deterministic scenario replay."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import numpy as np

from dh360_hi.testing.contracts import (
    ScenarioFrame,
)
from dh360_hi.vision.aligner import AlignmentResult, BaseFaceAligner, FaceAligner
from dh360_hi.vision.detector import (
    BaseFaceDetector,
    DetectionResult,
    FaceDetection,
    FaceLandmarks,
)
from dh360_hi.vision.embedder import BaseFaceEmbedder, EmbeddingResult, normalize_l2
from dh360_hi.vision.embedding_worker import (
    BaseEmbeddingWorker,
    EmbeddingTask,
    EmbeddingWorkerResult,
)
from dh360_hi.vision.frame import Frame, FrameSource
from dh360_hi.vision.recognition import BaseFaceRecognizer

if TYPE_CHECKING:
    pass


def make_deterministic_embedding(index: int, dimension: int = 128) -> np.ndarray:
    """Generate a 128-D unit vector with a 1.0 at the specified index."""
    vec = np.zeros(dimension, dtype=np.float32)
    vec[index % dimension] = 1.0
    return normalize_l2(vec)


class ScriptedFaceDetector(BaseFaceDetector):
    """Face detector that emits predetermined detections mapped to frame sequence IDs."""

    def __init__(
        self,
        frames: list[ScenarioFrame] | None = None,
        input_size: tuple[int, int] = (320, 320),
        confidence_threshold: float = 0.6,
    ) -> None:
        self._input_size = input_size
        self._confidence_threshold = confidence_threshold
        self._detections_by_frame: dict[int, list[FaceDetection]] = {}
        self._embedding_labels_by_frame: dict[int, list[str | None]] = {}

        if frames:
            for sf in frames:
                self.load_scenario_frame(sf)

    def load_scenario_frame(self, frame: ScenarioFrame) -> None:
        """Register detections for a specific scenario frame."""
        det_list: list[FaceDetection] = []
        labels: list[str | None] = []
        for d in frame.detections:
            lm = None
            if d.landmarks is not None:
                lm = FaceLandmarks(
                    right_eye=d.landmarks.right_eye,
                    left_eye=d.landmarks.left_eye,
                    nose_tip=d.landmarks.nose_tip,
                    right_mouth_corner=d.landmarks.right_mouth_corner,
                    left_mouth_corner=d.landmarks.left_mouth_corner,
                )
            det = FaceDetection(
                bbox=d.bbox,
                confidence=d.confidence,
                landmarks=lm,
            )
            det_list.append(det)
            labels.append(d.embedding_label)

        self._detections_by_frame[frame.frame_id] = det_list
        self._embedding_labels_by_frame[frame.frame_id] = labels

    def set_frame_detections(
        self,
        frame_id: int,
        detections: list[FaceDetection],
        labels: list[str | None] | None = None,
    ) -> None:
        """Set detections dynamically for a given frame ID."""
        self._detections_by_frame[frame_id] = list(detections)
        self._embedding_labels_by_frame[frame_id] = (
            list(labels) if labels is not None else [None] * len(detections)
        )

    def get_labels_for_frame(self, frame_id: int) -> list[str | None]:
        """Retrieve embedding labels for a given frame ID."""
        return self._embedding_labels_by_frame.get(frame_id, [])

    def get_label_for_bbox(
        self,
        frame_id: int,
        bbox: tuple[float, float, float, float],
    ) -> str | None:
        """Retrieve embedding label for the detection in the frame best matching bbox."""
        dets = self._detections_by_frame.get(frame_id, [])
        labels = self._embedding_labels_by_frame.get(frame_id, [])
        if not dets or not labels:
            return None

        best_iou = -1.0
        best_label: str | None = None
        for det, lbl in zip(dets, labels, strict=False):
            x1 = max(det.bbox[0], bbox[0])
            y1 = max(det.bbox[1], bbox[1])
            x2 = min(det.bbox[0] + det.bbox[2], bbox[0] + bbox[2])
            y2 = min(det.bbox[1] + det.bbox[3], bbox[1] + bbox[3])
            inter_w = max(0.0, x2 - x1)
            inter_h = max(0.0, y2 - y1)
            inter_area = inter_w * inter_h
            area1 = det.bbox[2] * det.bbox[3]
            area2 = bbox[2] * bbox[3]
            union_area = area1 + area2 - inter_area
            iou = inter_area / union_area if union_area > 0 else 0.0

            if iou > best_iou:
                best_iou = iou
                best_label = lbl

        return best_label if best_iou > 0.05 else (labels[0] if labels else None)

    @property
    def input_size(self) -> tuple[int, int]:
        return self._input_size

    @input_size.setter
    def input_size(self, size: tuple[int, int]) -> None:
        self._input_size = size

    @property
    def confidence_threshold(self) -> float:
        return self._confidence_threshold

    @confidence_threshold.setter
    def confidence_threshold(self, threshold: float) -> None:
        self._confidence_threshold = threshold

    def detect(self, frame: Frame | np.ndarray, frame_id: int = 0) -> DetectionResult:
        """Return scripted detections matching the frame index."""
        f_id = frame.frame_id if isinstance(frame, Frame) else frame_id
        detections = self._detections_by_frame.get(f_id, [])
        return DetectionResult(
            detections=list(detections),
            frame_id=f_id,
            latency_ms=0.1,
            input_size=self._input_size,
        )


class ScriptedFaceEmbedder(BaseFaceEmbedder):
    """Deterministic face feature embedder returning configured or synthetic embeddings."""

    def __init__(
        self,
        label_to_index: dict[str, int] | None = None,
        dimension: int = 128,
        model_name: str = "SFace",
        model_version: str = "2021dec",
    ) -> None:
        self._label_to_index = dict(label_to_index or {})
        self._dimension = dimension
        self._model_name = model_name
        self._model_version = model_version
        self._default_vector = make_deterministic_embedding(0, dimension)
        self._active_label: str | None = None
        self._track_labels: dict[int, str] = {}

    def register_label(self, label: str, index: int) -> None:
        """Associate a person ID / label with a specific unit vector index."""
        self._label_to_index[label] = index

    def set_active_label(self, label: str | None) -> None:
        """Set the active label for subsequent embed() calls."""
        self._active_label = label

    def set_track_label(self, track_id: int, label: str) -> None:
        """Associate a track ID with a specific label."""
        self._track_labels[track_id] = label

    def get_embedding_for_label(self, label: str | None) -> np.ndarray:
        """Return normalized 128-D vector for a label."""
        if label is not None and label in self._label_to_index:
            idx = self._label_to_index[label]
            return make_deterministic_embedding(idx, self._dimension)
        return self._default_vector

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def input_size(self) -> tuple[int, int]:
        return (112, 112)

    def embed(self, aligned_face: np.ndarray | AlignmentResult) -> EmbeddingResult:
        """Extract a deterministic normalized embedding."""
        _ = aligned_face
        emb = self.get_embedding_for_label(self._active_label)
        return EmbeddingResult(
            embedding=emb,
            model_name=self._model_name,
            model_version=self._model_version,
            dimension=self._dimension,
            is_normalized=True,
            latency_ms=0.1,
        )


class DirectEmbeddingWorker(BaseEmbeddingWorker):
    """Synchronous, deterministic embedding worker that processes tasks inline."""

    def __init__(
        self,
        embedder: ScriptedFaceEmbedder,
        recognizer: BaseFaceRecognizer,
        aligner: BaseFaceAligner | None = None,
        detector: ScriptedFaceDetector | None = None,
    ) -> None:
        self._embedder = embedder
        self._recognizer = recognizer
        self._aligner = aligner or FaceAligner()
        self._detector = detector
        self._completed_results: deque[EmbeddingWorkerResult] = deque()
        self._is_running = True

    def start(self) -> None:
        self._is_running = True

    def stop(self, timeout: float = 2.0) -> None:
        _ = timeout
        self._is_running = False

    def submit(self, task: EmbeddingTask) -> bool:
        """Execute embedding and recognition synchronously and buffer result."""
        # Check if detector has a label for this frame matching bbox
        label: str | None = None
        if self._detector is not None:
            label = self._detector.get_label_for_bbox(task.frame_id, task.bbox)

        # Check embedder track labels if no frame label
        if label is None:
            label = self._embedder._track_labels.get(task.track_id)

        self._embedder.set_active_label(label)
        emb_res = self._embedder.embed(task.image)
        rec_res = self._recognizer.recognize(emb_res)

        res = EmbeddingWorkerResult(
            track_id=task.track_id,
            frame_id=task.frame_id,
            timestamp=task.timestamp,
            alignment_result=None,
            embedding_result=emb_res,
            recognition_result=rec_res,
            queue_latency_ms=0.0,
            inference_latency_ms=0.1,
            total_latency_ms=0.1,
            error=None,
        )
        self._completed_results.append(res)
        return True

    def poll_results(self, max_results: int | None = None) -> list[EmbeddingWorkerResult]:
        """Retrieve completed recognition results."""
        results: list[EmbeddingWorkerResult] = []
        limit = max_results if max_results is not None else len(self._completed_results)
        while self._completed_results and len(results) < limit:
            results.append(self._completed_results.popleft())
        return results

    def quiesce(self, timeout: float = 5.0) -> bool:
        """Discard inline results at a deterministic lifecycle boundary."""
        del timeout
        self._completed_results.clear()
        return True

    def is_busy_or_pending(self, track_id: int) -> bool:
        """In synchronous execution, tasks never block."""
        _ = track_id
        return False

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def pending_count(self) -> int:
        return len(self._completed_results)


class ScriptedFrameSource(FrameSource):
    """Deterministic frame source yielding frames specified by scenario sequence."""

    def __init__(
        self,
        frames: list[ScenarioFrame],
        width: int = 320,
        height: int = 320,
    ) -> None:
        self._frames = list(frames)
        self._width = width
        self._height = height
        self._index = 0
        self._is_opened = False

    def open(self) -> None:
        self._is_opened = True
        self._index = 0

    def read(self) -> Frame | None:
        if not self._is_opened or self._index >= len(self._frames):
            return None

        sf = self._frames[self._index]
        self._index += 1

        # Generate textured synthetic frame image so face quality gate sharpness check passes
        image = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        image[:, :] = (35, 35, 35)

        for det in sf.detections:
            x, y, w, h = (int(det.bbox[0]), int(det.bbox[1]), int(det.bbox[2]), int(det.bbox[3]))
            x1 = max(0, min(self._width, x))
            y1 = max(0, min(self._height, y))
            x2 = max(0, min(self._width, x + w))
            y2 = max(0, min(self._height, y + h))
            if x2 > x1 and y2 > y1:
                # High-contrast checkerboard inside bbox for Laplacian sharpness > 100.0
                for r in range(y1, y2, 4):
                    for c in range(x1, x2, 4):
                        if ((r // 4) + (c // 4)) % 2 == 0:
                            image[r : min(r + 4, y2), c : min(c + 4, x2)] = (220, 220, 220)
                        else:
                            image[r : min(r + 4, y2), c : min(c + 4, x2)] = (30, 30, 30)

        return Frame(
            image=image,
            frame_id=sf.frame_id,
            timestamp=sf.timestamp,
            source_info="scripted_frame_source",
        )

    def release(self) -> None:
        self._is_opened = False

    @property
    def is_opened(self) -> bool:
        return self._is_opened

    @property
    def resolution(self) -> tuple[int, int]:
        return self._width, self._height

    @property
    def fps(self) -> float:
        return 30.0

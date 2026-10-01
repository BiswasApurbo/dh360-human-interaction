"""Lightweight, modular face tracking on top of M1 FaceDetection results."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from dh360_hi.vision.detector import DetectionResult, FaceDetection, FaceLandmarks


class TrackState(StrEnum):
    """Lifecycle state of a tracked face."""

    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"
    DELETED = "deleted"


@dataclass
class TrackedFace:
    """Represents a continuous track of a detected face across frames."""

    track_id: int
    bbox: tuple[float, float, float, float]  # (x, y, width, height) in pixels
    confidence: float
    landmarks: FaceLandmarks | None = None
    state: TrackState = TrackState.TENTATIVE
    start_frame_id: int = 0
    last_seen_frame_id: int = 0
    start_timestamp: float = 0.0
    last_seen_timestamp: float = 0.0
    hits: int = 1
    misses: int = 0
    age: int = 1

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
    def center(self) -> tuple[float, float]:
        """Center coordinate (cx, cy) of bounding box."""
        return self.x + self.width / 2.0, self.y + self.height / 2.0

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass(frozen=True)
class TrackingResult:
    """Output of a tracker update step on a single frame."""

    tracks: list[TrackedFace]
    frame_id: int
    latency_ms: float
    new_track_ids: list[int] = field(default_factory=list)
    lost_track_ids: list[int] = field(default_factory=list)
    removed_track_ids: list[int] = field(default_factory=list)

    @property
    def confirmed_tracks(self) -> list[TrackedFace]:
        """Filter tracks to only confirmed active faces."""
        return [t for t in self.tracks if t.state == TrackState.CONFIRMED]


def compute_iou(
    bbox1: tuple[float, float, float, float],
    bbox2: tuple[float, float, float, float],
) -> float:
    """Compute Intersection-over-Union (IoU) between two bounding boxes (x, y, w, h)."""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_w = max(0.0, xi2 - xi1)
    inter_h = max(0.0, yi2 - yi1)
    inter_area = inter_w * inter_h

    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area

    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def compute_center_distance(
    bbox1: tuple[float, float, float, float],
    bbox2: tuple[float, float, float, float],
) -> float:
    """Compute Euclidean distance between centers of two bounding boxes."""
    c1_x = bbox1[0] + bbox1[2] / 2.0
    c1_y = bbox1[1] + bbox1[3] / 2.0
    c2_x = bbox2[0] + bbox2[2] / 2.0
    c2_y = bbox2[1] + bbox2[3] / 2.0
    return float(np.hypot(c1_x - c2_x, c1_y - c2_y))


class BaseFaceTracker(ABC):
    """Abstract interface for face tracking algorithms."""

    @abstractmethod
    def update(
        self,
        detections: Sequence[FaceDetection] | DetectionResult,
        frame_id: int = 0,
        timestamp: float = 0.0,
    ) -> TrackingResult:
        """Update tracker with new detections for a frame."""

    @abstractmethod
    def reset(self) -> None:
        """Reset internal tracker state and IDs."""

    @property
    @abstractmethod
    def active_tracks(self) -> list[TrackedFace]:
        """Return currently active tracked faces."""


class FaceTracker(BaseFaceTracker):
    """Modular, deterministic face tracker based on IoU and spatial centroid matching."""

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_missed_frames: int = 15,
        min_hits: int = 1,
        max_center_distance: float | None = 100.0,
        smooth_factor: float = 0.8,
    ) -> None:
        """Initialize face tracker.

        Args:
            iou_threshold: Minimum IoU overlap required to match a detection with a track.
            max_missed_frames: Maximum consecutive missed frames before track expiration.
            min_hits: Number of detection hits required to promote track to CONFIRMED state.
            max_center_distance: Optional maximum center pixel distance fallback when IoU is zero.
            smooth_factor: Exponential smoothing factor for bounding boxes (1.0 = no smoothing).
        """
        self.iou_threshold = float(iou_threshold)
        self.max_missed_frames = int(max_missed_frames)
        self.min_hits = int(min_hits)
        self.max_center_distance = (
            float(max_center_distance) if max_center_distance is not None else None
        )
        self.smooth_factor = float(smooth_factor)

        self._next_id: int = 1
        self._tracks: dict[int, TrackedFace] = {}

    def reset(self) -> None:
        """Clear all active tracks and reset ID counter."""
        self._next_id = 1
        self._tracks.clear()

    @property
    def active_tracks(self) -> list[TrackedFace]:
        """Return list of active (non-deleted) tracks."""
        return [t for t in self._tracks.values() if t.state != TrackState.DELETED]

    def update(
        self,
        detections: Sequence[FaceDetection] | DetectionResult,
        frame_id: int = 0,
        timestamp: float = 0.0,
    ) -> TrackingResult:
        """Process incoming detections and update track states."""
        start_time = time.perf_counter()

        if isinstance(detections, DetectionResult):
            det_list = detections.detections
            current_frame_id = detections.frame_id
        else:
            det_list = list(detections)
            current_frame_id = frame_id

        current_timestamp = timestamp or time.monotonic()

        # Update track ages
        for track in self._tracks.values():
            track.age += 1

        active_track_ids = [tid for tid, t in self._tracks.items() if t.state != TrackState.DELETED]
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()

        new_track_ids: list[int] = []
        lost_track_ids: list[int] = []
        removed_track_ids: list[int] = []

        # 1. Primary Matching via IoU
        if active_track_ids and det_list:
            iou_matrix = np.zeros((len(active_track_ids), len(det_list)), dtype=np.float32)
            for i, tid in enumerate(active_track_ids):
                for j, det in enumerate(det_list):
                    iou_matrix[i, j] = compute_iou(self._tracks[tid].bbox, det.bbox)

            # Greedy bipartite matching on highest IoU
            while True:
                max_val = float(np.max(iou_matrix)) if iou_matrix.size > 0 else 0.0
                if max_val < self.iou_threshold:
                    break
                max_pos = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                i, j = int(max_pos[0]), int(max_pos[1])
                tid = active_track_ids[i]

                self._update_track_detection(
                    self._tracks[tid],
                    det_list[j],
                    current_frame_id,
                    current_timestamp,
                )
                matched_tracks.add(tid)
                matched_detections.add(j)

                # Invalidate matched row and column
                iou_matrix[i, :] = -1.0
                iou_matrix[:, j] = -1.0

        # 2. Secondary Fallback Matching via Centroid Distance
        if self.max_center_distance is not None:
            unmatched_tids = [tid for tid in active_track_ids if tid not in matched_tracks]
            unmatched_det_indices = [j for j in range(len(det_list)) if j not in matched_detections]

            if unmatched_tids and unmatched_det_indices:
                dist_matrix = np.full(
                    (len(unmatched_tids), len(unmatched_det_indices)),
                    fill_value=1e6,
                    dtype=np.float32,
                )
                for i, tid in enumerate(unmatched_tids):
                    for j_idx, j in enumerate(unmatched_det_indices):
                        dist = compute_center_distance(self._tracks[tid].bbox, det_list[j].bbox)
                        if dist <= self.max_center_distance:
                            dist_matrix[i, j_idx] = dist

                while True:
                    min_val = float(np.min(dist_matrix)) if dist_matrix.size > 0 else 1e6
                    if min_val > self.max_center_distance:
                        break
                    min_pos = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
                    i, j_idx = int(min_pos[0]), int(min_pos[1])
                    tid = unmatched_tids[i]
                    j = unmatched_det_indices[j_idx]

                    self._update_track_detection(
                        self._tracks[tid],
                        det_list[j],
                        current_frame_id,
                        current_timestamp,
                    )
                    matched_tracks.add(tid)
                    matched_detections.add(j)

                    dist_matrix[i, :] = 1e6
                    dist_matrix[:, j_idx] = 1e6

        # 3. Handle Unmatched Active Tracks (Misses & Expirations)
        for tid in active_track_ids:
            if tid not in matched_tracks:
                track = self._tracks[tid]
                track.misses += 1
                if track.state == TrackState.CONFIRMED:
                    track.state = TrackState.LOST
                    lost_track_ids.append(tid)

                if track.misses > self.max_missed_frames:
                    track.state = TrackState.DELETED
                    removed_track_ids.append(tid)

        # 4. Handle Unmatched Detections (New Tracks)
        for j, det in enumerate(det_list):
            if j not in matched_detections:
                new_track = TrackedFace(
                    track_id=self._next_id,
                    bbox=det.bbox,
                    confidence=det.confidence,
                    landmarks=det.landmarks,
                    state=TrackState.CONFIRMED if self.min_hits <= 1 else TrackState.TENTATIVE,
                    start_frame_id=current_frame_id,
                    last_seen_frame_id=current_frame_id,
                    start_timestamp=current_timestamp,
                    last_seen_timestamp=current_timestamp,
                    hits=1,
                    misses=0,
                    age=1,
                )
                self._tracks[self._next_id] = new_track
                new_track_ids.append(self._next_id)
                self._next_id += 1

        # Clean up deleted tracks from internal storage
        for tid in removed_track_ids:
            self._tracks.pop(tid, None)

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return TrackingResult(
            tracks=list(self._tracks.values()),
            frame_id=current_frame_id,
            latency_ms=latency_ms,
            new_track_ids=new_track_ids,
            lost_track_ids=lost_track_ids,
            removed_track_ids=removed_track_ids,
        )

    def _update_track_detection(
        self,
        track: TrackedFace,
        detection: FaceDetection,
        frame_id: int,
        timestamp: float,
    ) -> None:
        """Update existing track attributes with newly matched detection."""
        track.hits += 1
        track.misses = 0
        track.last_seen_frame_id = frame_id
        track.last_seen_timestamp = timestamp
        track.confidence = detection.confidence
        track.landmarks = detection.landmarks

        if track.hits >= self.min_hits and track.state != TrackState.DELETED:
            track.state = TrackState.CONFIRMED

        # Apply exponential smoothing to bounding box
        if self.smooth_factor >= 1.0:
            track.bbox = detection.bbox
        else:
            alpha = self.smooth_factor
            old_x, old_y, old_w, old_h = track.bbox
            new_x, new_y, new_w, new_h = detection.bbox
            smoothed_bbox = (
                alpha * new_x + (1.0 - alpha) * old_x,
                alpha * new_y + (1.0 - alpha) * old_y,
                alpha * new_w + (1.0 - alpha) * old_w,
                alpha * new_h + (1.0 - alpha) * old_h,
            )
            track.bbox = smoothed_bbox

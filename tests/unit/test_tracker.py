"""Unit tests for BaseFaceTracker and FaceTracker implementations."""

import pytest

from dh360_hi.vision.detector import DetectionResult, FaceDetection, FaceLandmarks
from dh360_hi.vision.tracker import (
    FaceTracker,
    TrackState,
    compute_center_distance,
    compute_iou,
)


def _make_det(
    x: float,
    y: float,
    w: float = 50.0,
    h: float = 60.0,
    conf: float = 0.9,
) -> FaceDetection:
    """Helper creating a test FaceDetection."""
    return FaceDetection(
        bbox=(x, y, w, h),
        confidence=conf,
        landmarks=FaceLandmarks(
            right_eye=(x + 15, y + 20),
            left_eye=(x + 35, y + 20),
            nose_tip=(x + 25, y + 35),
            right_mouth_corner=(x + 18, y + 48),
            left_mouth_corner=(x + 32, y + 48),
        ),
    )


def test_compute_iou_geometry() -> None:
    """Test IoU overlap calculations across standard geometric relationships."""
    # Exact overlap
    b1 = (100.0, 100.0, 50.0, 50.0)
    assert compute_iou(b1, b1) == pytest.approx(1.0)

    # Disjoint boxes (no overlap)
    b2 = (200.0, 200.0, 50.0, 50.0)
    assert compute_iou(b1, b2) == 0.0

    # 50% horizontal overlap: width 50, overlapping width 25, height 50
    # intersection = 25*50 = 1250, union = 2500 + 2500 - 1250 = 3750, IoU = 1/3
    b3 = (125.0, 100.0, 50.0, 50.0)
    assert compute_iou(b1, b3) == pytest.approx(1250.0 / 3750.0)


def test_compute_center_distance() -> None:
    """Test center Euclidean distance calculation."""
    b1 = (0.0, 0.0, 10.0, 10.0)  # center = (5, 5)
    b2 = (10.0, 0.0, 10.0, 10.0)  # center = (15, 5)
    assert compute_center_distance(b1, b2) == pytest.approx(10.0)


def test_single_face_maintains_same_track_id() -> None:
    """Verify a face moving across consecutive frames maintains the same track_id."""
    tracker = FaceTracker(iou_threshold=0.3)

    # Frame 1
    d1 = _make_det(100.0, 100.0)
    res1 = tracker.update([d1], frame_id=1)
    assert len(res1.tracks) == 1
    track_id = res1.tracks[0].track_id
    assert track_id == 1
    assert res1.new_track_ids == [1]
    assert res1.tracks[0].hits == 1

    # Frame 2: slight translation (high IoU)
    d2 = _make_det(104.0, 102.0)
    res2 = tracker.update([d2], frame_id=2)
    assert len(res2.tracks) == 1
    assert res2.tracks[0].track_id == track_id
    assert res2.tracks[0].hits == 2
    assert res2.tracks[0].misses == 0
    assert len(res2.new_track_ids) == 0

    # Frame 3: further slight translation
    d3 = _make_det(108.0, 105.0)
    res3 = tracker.update([d3], frame_id=3)
    assert len(res3.tracks) == 1
    assert res3.tracks[0].track_id == track_id
    assert res3.tracks[0].hits == 3


def test_two_faces_maintain_separate_track_ids() -> None:
    """Verify two distinct simultaneous faces are assigned and maintain separate track IDs."""
    tracker = FaceTracker(iou_threshold=0.3)

    # Frame 1: Face A on left (x=100), Face B on right (x=400)
    face_a1 = _make_det(100.0, 100.0)
    face_b1 = _make_det(400.0, 100.0)
    res1 = tracker.update([face_a1, face_b1], frame_id=1)

    assert len(res1.tracks) == 2
    track_ids = {t.track_id for t in res1.tracks}
    assert len(track_ids) == 2

    track_a_id = next(t.track_id for t in res1.tracks if t.x < 200)
    track_b_id = next(t.track_id for t in res1.tracks if t.x > 300)
    assert track_a_id != track_b_id

    # Frame 2: both move slightly
    face_a2 = _make_det(103.0, 102.0)
    face_b2 = _make_det(402.0, 98.0)
    res2 = tracker.update([face_a2, face_b2], frame_id=2)

    assert len(res2.tracks) == 2
    res2_a = next(t for t in res2.tracks if t.track_id == track_a_id)
    res2_b = next(t for t in res2.tracks if t.track_id == track_b_id)

    assert res2_a.x == pytest.approx(103.0, abs=1.0)
    assert res2_b.x == pytest.approx(402.0, abs=1.0)


def test_temporarily_missing_face_retains_track() -> None:
    """Verify a face missing for a few frames retains its track ID when it reappears."""
    tracker = FaceTracker(max_missed_frames=5)

    # Frame 1: Face appears
    d1 = _make_det(100.0, 100.0)
    res1 = tracker.update([d1], frame_id=1)
    tid = res1.tracks[0].track_id

    # Frames 2-4: Face missed (e.g. occlusion or false negative)
    for f in range(2, 5):
        res = tracker.update([], frame_id=f)
        assert len(res.tracks) == 1
        assert res.tracks[0].track_id == tid
        assert res.tracks[0].state == TrackState.LOST
        assert res.tracks[0].misses == f - 1

    # Frame 5: Face reappears at nearby position
    d_reappear = _make_det(105.0, 102.0)
    res5 = tracker.update([d_reappear], frame_id=5)

    assert len(res5.tracks) == 1
    assert res5.tracks[0].track_id == tid
    assert res5.tracks[0].state == TrackState.CONFIRMED
    assert res5.tracks[0].misses == 0
    assert res5.tracks[0].hits == 2


def test_expired_track_is_removed() -> None:
    """Verify tracks exceeding max_missed_frames are deleted."""
    tracker = FaceTracker(max_missed_frames=3)

    # Frame 1: Face appears
    tracker.update([_make_det(100.0, 100.0)], frame_id=1)

    # Miss for 3 frames (within tolerance)
    tracker.update([], frame_id=2)
    tracker.update([], frame_id=3)
    res4 = tracker.update([], frame_id=4)
    assert len(res4.tracks) == 1
    assert res4.tracks[0].misses == 3

    # Frame 5: 4th consecutive miss exceeds max_missed_frames (3)
    res5 = tracker.update([], frame_id=5)
    assert len(res5.tracks) == 0
    assert res5.removed_track_ids == [1]


def test_new_face_receives_new_track_id() -> None:
    """Verify new detections in different locations receive newly incremented track IDs."""
    tracker = FaceTracker()

    # First face
    res1 = tracker.update([_make_det(50.0, 50.0)], frame_id=1)
    assert res1.tracks[0].track_id == 1

    # Second face appearing later in completely different region
    res2 = tracker.update([_make_det(50.0, 50.0), _make_det(350.0, 350.0)], frame_id=2)
    assert len(res2.tracks) == 2
    track_ids = [t.track_id for t in res2.tracks]
    assert 1 in track_ids
    assert 2 in track_ids
    assert res2.new_track_ids == [2]


def test_no_detections_on_empty_tracker() -> None:
    """Verify tracker handles empty frame on cold start cleanly."""
    tracker = FaceTracker()
    res = tracker.update([], frame_id=1)

    assert len(res.tracks) == 0
    assert len(res.new_track_ids) == 0
    assert len(res.lost_track_ids) == 0
    assert len(res.removed_track_ids) == 0
    assert res.latency_ms >= 0.0


def test_detection_result_object_input() -> None:
    """Verify tracker accepts DetectionResult container object directly."""
    tracker = FaceTracker()
    det_res = DetectionResult(
        detections=[_make_det(120.0, 140.0)],
        frame_id=10,
        latency_ms=4.2,
        input_size=(640, 480),
    )

    res = tracker.update(det_res)
    assert res.frame_id == 10
    assert len(res.tracks) == 1
    assert res.tracks[0].track_id == 1


def test_tracker_reset() -> None:
    """Verify reset() clears internal state and resets next ID."""
    tracker = FaceTracker()
    tracker.update([_make_det(100.0, 100.0)], frame_id=1)
    assert len(tracker.active_tracks) == 1

    tracker.reset()
    assert len(tracker.active_tracks) == 0

    # Next track should restart at ID 1
    res = tracker.update([_make_det(200.0, 200.0)], frame_id=1)
    assert res.tracks[0].track_id == 1

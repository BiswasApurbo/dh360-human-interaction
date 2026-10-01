"""Unit tests for InteractionTargetSelector multi-person target selection and stickiness."""

from __future__ import annotations

from dh360_hi.vision.identity_state import IdentityState, TrackIdentityState
from dh360_hi.vision.target_selector import (
    InteractionTargetSelector,
    TargetSelectionConfig,
)
from dh360_hi.vision.tracker import TrackedFace, TrackState


def _create_track(
    track_id: int,
    bbox: tuple[float, float, float, float],
    state: TrackState = TrackState.CONFIRMED,
    age: int = 10,
) -> TrackedFace:
    """Helper to create a TrackedFace instance."""
    return TrackedFace(
        track_id=track_id,
        bbox=bbox,
        confidence=0.95,
        state=state,
        age=age,
    )


def test_empty_tracks_target_selection() -> None:
    """Test target selection when no active tracks are present."""
    selector = InteractionTargetSelector()
    res = selector.select_target(
        tracks=[],
        identity_states={},
        frame_width=640,
        frame_height=480,
        frame_id=1,
    )
    assert res.primary_target is None
    assert len(res.candidates) == 0
    assert not res.has_target


def test_single_track_target_selection() -> None:
    """Test target selection with a single centered face."""
    selector = InteractionTargetSelector()
    track = _create_track(track_id=1, bbox=(270.0, 190.0, 100.0, 100.0))  # Centered in 640x480
    id_state = TrackIdentityState(
        track_id=1,
        state=IdentityState.IDENTIFIED,
        current_person_id="p1",
        current_display_name="Alice",
    )

    res = selector.select_target(
        tracks=[track],
        identity_states={1: id_state},
        frame_width=640,
        frame_height=480,
        frame_id=1,
    )

    assert res.has_target
    assert res.primary_track_id == 1
    assert res.primary_target is not None
    assert res.primary_target.person_id == "p1"
    assert res.primary_target.display_name == "Alice"
    assert res.primary_target.is_identified is True
    assert res.target_changed is True


def test_centrality_preference() -> None:
    """Test that a centered face is preferred over an off-center face of equal size and state."""
    selector = InteractionTargetSelector()
    # Centered face (track 1) vs Corner face (track 2)
    t1 = _create_track(track_id=1, bbox=(270.0, 190.0, 100.0, 100.0))  # center: (320, 240)
    t2 = _create_track(track_id=2, bbox=(10.0, 10.0, 100.0, 100.0))  # center: (60, 60)

    res = selector.select_target(
        tracks=[t1, t2],
        identity_states={},
        frame_width=640,
        frame_height=480,
        frame_id=1,
    )

    assert res.primary_track_id == 1
    assert res.candidates[0].track_id == 1
    assert res.candidates[0].centrality_score > res.candidates[1].centrality_score


def test_size_and_proximity_preference() -> None:
    """Test that a larger face is preferred over a tiny face near center."""
    selector = InteractionTargetSelector()
    # Large face (track 1) vs Tiny face (track 2)
    t1 = _create_track(track_id=1, bbox=(220.0, 140.0, 200.0, 200.0))  # Large
    t2 = _create_track(track_id=2, bbox=(310.0, 230.0, 20.0, 20.0))  # Tiny

    res = selector.select_target(
        tracks=[t1, t2],
        identity_states={},
        frame_width=640,
        frame_height=480,
        frame_id=1,
    )

    assert res.primary_track_id == 1
    assert res.candidates[0].size_score > res.candidates[1].size_score


def test_identified_bonus_preference() -> None:
    """Test that an IDENTIFIED user receives a state score bonus over UNREGISTERED/NEW."""
    selector = InteractionTargetSelector()
    t1 = _create_track(track_id=1, bbox=(150.0, 190.0, 100.0, 100.0))
    t2 = _create_track(track_id=2, bbox=(390.0, 190.0, 100.0, 100.0))

    id1 = TrackIdentityState(track_id=1, state=IdentityState.IDENTIFIED, current_person_id="p1")
    id2 = TrackIdentityState(track_id=2, state=IdentityState.UNREGISTERED)

    res = selector.select_target(
        tracks=[t1, t2],
        identity_states={1: id1, 2: id2},
        frame_width=640,
        frame_height=480,
        frame_id=1,
    )

    assert res.primary_track_id == 1
    assert res.candidates[0].identity_state_score > res.candidates[1].identity_state_score


def test_stickiness_hysteresis_prevents_target_flickering() -> None:
    """Test that current target is retained when a rival candidate's score

    is only slightly higher.
    """
    config = TargetSelectionConfig(hysteresis_switch_margin=0.15)
    selector = InteractionTargetSelector(config)

    # Frame 1: Track 1 is primary target
    t1 = _create_track(track_id=1, bbox=(270.0, 190.0, 100.0, 100.0))
    t2 = _create_track(track_id=2, bbox=(100.0, 100.0, 80.0, 80.0))
    res1 = selector.select_target([t1, t2], {}, 640, 480, frame_id=1)
    assert res1.primary_track_id == 1

    # Frame 2: Track 2 moves slightly closer to center, scoring barely higher
    # than Track 1 (< 0.15 margin)
    t1_f2 = _create_track(track_id=1, bbox=(250.0, 190.0, 95.0, 95.0))
    t2_f2 = _create_track(track_id=2, bbox=(260.0, 190.0, 98.0, 98.0))
    res2 = selector.select_target([t1_f2, t2_f2], {}, 640, 480, frame_id=2)

    # Due to stickiness hysteresis, target 1 is retained
    assert res2.primary_track_id == 1
    assert res2.target_changed is False

    # Frame 3: Track 2 grows much larger (exceeds switch margin)
    t2_f3 = _create_track(track_id=2, bbox=(200.0, 100.0, 240.0, 240.0))
    res3 = selector.select_target([t1_f2, t2_f3], {}, 640, 480, frame_id=3)

    # Now target switches to Track 2
    assert res3.primary_track_id == 2
    assert res3.target_changed is True


def test_target_release_after_lost_frames() -> None:
    """Test that target focus is released when the target track is lost for > max_lost_frames."""
    config = TargetSelectionConfig(max_lost_frames_to_release=5)
    selector = InteractionTargetSelector(config)

    t1 = _create_track(track_id=1, bbox=(270.0, 190.0, 100.0, 100.0))
    selector.select_target([t1], {}, 640, 480, frame_id=1)
    assert selector.current_target_track_id == 1

    # Empty frame at frame_id 10 (> 5 frames lost)
    res = selector.select_target([], {}, 640, 480, frame_id=10)
    assert res.primary_target is None
    assert selector.current_target_track_id is None

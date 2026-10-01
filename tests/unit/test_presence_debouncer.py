"""Unit tests for PresenceDebouncer temporal entry and departure smoothing."""

from __future__ import annotations

from dh360_hi.vision.identity_state import IdentityState, TrackIdentityState
from dh360_hi.vision.presence_debouncer import (
    PresenceDebounceConfig,
    PresenceDebouncer,
    PresenceEvent,
)
from dh360_hi.vision.tracker import TrackedFace, TrackState


def _make_track(track_id: int) -> TrackedFace:
    return TrackedFace(
        track_id=track_id,
        bbox=(100.0, 100.0, 100.0, 100.0),
        confidence=0.95,
        state=TrackState.CONFIRMED,
    )


def test_enter_patience_debouncing() -> None:
    """Test that PERSON_ENTERED is only emitted after enter_patience_frames

    consecutive sightings.
    """
    config = PresenceDebounceConfig(enter_patience_frames=3, leave_patience_frames=5)
    debouncer = PresenceDebouncer(config)
    track = _make_track(track_id=1)
    id_state = TrackIdentityState(
        track_id=1,
        state=IdentityState.IDENTIFIED,
        current_person_id="p1",
        current_display_name="Alice",
    )
    id_map = {1: id_state}

    # Frame 1: 1st sighting -> No transition yet (entering state)
    t1 = debouncer.update([track], id_map, frame_id=1, timestamp=0.1)
    assert len(t1) == 0
    assert 1 not in debouncer.get_confirmed_present_tracks()

    # Frame 2: 2nd sighting -> Still debouncing
    t2 = debouncer.update([track], id_map, frame_id=2, timestamp=0.2)
    assert len(t2) == 0
    assert 1 not in debouncer.get_confirmed_present_tracks()

    # Frame 3: 3rd sighting -> Reached enter patience -> Emits PERSON_ENTERED
    t3 = debouncer.update([track], id_map, frame_id=3, timestamp=0.3)
    assert len(t3) == 1
    assert t3[0].event == PresenceEvent.PERSON_ENTERED
    assert t3[0].track_id == 1
    assert t3[0].person_id == "p1"
    assert t3[0].display_name == "Alice"
    assert 1 in debouncer.get_confirmed_present_tracks()

    # Frame 4: 4th sighting -> Already present, no new transition
    t4 = debouncer.update([track], id_map, frame_id=4, timestamp=0.4)
    assert len(t4) == 0


def test_leave_patience_and_temporary_glitch_recovery() -> None:
    """Test departure debouncing and recovery from temporary detection loss."""
    config = PresenceDebounceConfig(enter_patience_frames=2, leave_patience_frames=4)
    debouncer = PresenceDebouncer(config)
    track = _make_track(track_id=1)

    # Establish presence across frames 1 and 2
    debouncer.update([track], {}, frame_id=1, timestamp=0.1)
    events_enter = debouncer.update([track], {}, frame_id=2, timestamp=0.2)
    assert len(events_enter) == 1
    assert events_enter[0].event == PresenceEvent.PERSON_ENTERED
    assert 1 in debouncer.get_confirmed_present_tracks()

    # Frame 3: 1 frame detection miss (e.g. occlusion/glitch) -> Should NOT emit PERSON_LEFT
    e3 = debouncer.update([], {}, frame_id=3, timestamp=0.3)
    assert len(e3) == 0

    # Frame 4: 2nd missed frame -> Still in leaving debounce window
    e4 = debouncer.update([], {}, frame_id=4, timestamp=0.4)
    assert len(e4) == 0

    # Frame 5: Track returns! -> Recovers to PRESENT without having left
    e5 = debouncer.update([track], {}, frame_id=5, timestamp=0.5)
    assert len(e5) == 0
    assert 1 in debouncer.get_confirmed_present_tracks()

    # Now simulate persistent departure: frames 6, 7, 8, 9 (4 consecutive missed frames)
    debouncer.update([], {}, frame_id=6, timestamp=0.6)
    debouncer.update([], {}, frame_id=7, timestamp=0.7)
    debouncer.update([], {}, frame_id=8, timestamp=0.8)
    e9 = debouncer.update([], {}, frame_id=9, timestamp=0.9)

    # At 4th missed frame (frame 9), emits PERSON_LEFT
    assert len(e9) == 1
    assert e9[0].event == PresenceEvent.PERSON_LEFT
    assert e9[0].track_id == 1
    assert 1 not in debouncer.get_confirmed_present_tracks()

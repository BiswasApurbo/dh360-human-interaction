"""Temporal presence debounce for smoothing entry and departure transitions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from dh360_hi.vision.identity_state import TrackIdentityState
from dh360_hi.vision.tracker import TrackedFace, TrackState


class PresenceEvent(StrEnum):
    """Presence transition event emitted by debouncer."""

    PERSON_ENTERED = "person_entered"
    PERSON_LEFT = "person_left"
    PERSON_PRESENT = "person_present"


class PresenceState(StrEnum):
    """Internal state of presence tracking for a track."""

    ABSENT = "absent"
    ENTERING = "entering"  # Undergoing entry debounce
    PRESENT = "present"  # Confirmed present
    LEAVING = "leaving"  # Undergoing departure debounce


@dataclass(frozen=True)
class PresenceTransition:
    """Represents a presence event emitted on a state change."""

    event: PresenceEvent
    track_id: int
    person_id: str | None
    display_name: str | None
    frame_id: int
    timestamp: float


@dataclass(frozen=True)
class PresenceDebounceConfig:
    """Parameters governing temporal entry and departure debouncing."""

    enter_patience_frames: int = 3  # Must be visible consecutively for N frames before ENTERED
    leave_patience_frames: int = 10  # Must be missing consecutively for N frames before LEFT


@dataclass
class _TrackPresenceState:
    """Mutable internal state for a single track's presence lifecycle."""

    track_id: int
    presence_state: PresenceState = PresenceState.ABSENT
    consecutive_present_frames: int = 0
    consecutive_missed_frames: int = 0
    first_seen_frame_id: int = 0
    last_seen_frame_id: int = 0
    person_id: str | None = None
    display_name: str | None = None


class PresenceDebouncer:
    """Debounces presence state transitions across active tracks to eliminate detection flicker."""

    def __init__(self, config: PresenceDebounceConfig | None = None) -> None:
        self._config = config or PresenceDebounceConfig()
        self._states: dict[int, _TrackPresenceState] = {}

    @property
    def config(self) -> PresenceDebounceConfig:
        return self._config

    @config.setter
    def config(self, config: PresenceDebounceConfig) -> None:
        self._config = config

    def reset(self) -> None:
        """Reset all presence debouncer history."""
        self._states.clear()

    def update(
        self,
        tracks: Sequence[TrackedFace],
        identity_states: dict[int, TrackIdentityState],
        frame_id: int,
        timestamp: float,
    ) -> list[PresenceTransition]:
        """Update presence states from current frame tracks and return newly triggered transitions.

        Args:
            tracks: Currently active tracks.
            identity_states: Current per-track identity states.
            frame_id: Current frame index.
            timestamp: Frame acquisition timestamp.

        Returns:
            List of emitted PresenceTransition events (ENTERED / LEFT).
        """
        transitions: list[PresenceTransition] = []
        active_confirmed_tracks = {t.track_id: t for t in tracks if t.state == TrackState.CONFIRMED}

        # 1. Update existing and new active tracks
        for track_id in active_confirmed_tracks:
            id_state = identity_states.get(track_id)
            person_id = id_state.current_person_id if id_state else None
            display_name = id_state.current_display_name if id_state else None

            if track_id not in self._states:
                # Initialize new presence tracker
                p_state = _TrackPresenceState(
                    track_id=track_id,
                    presence_state=PresenceState.ENTERING,
                    consecutive_present_frames=1,
                    consecutive_missed_frames=0,
                    first_seen_frame_id=frame_id,
                    last_seen_frame_id=frame_id,
                    person_id=person_id,
                    display_name=display_name,
                )
                self._states[track_id] = p_state
            else:
                p_state = self._states[track_id]
                p_state.consecutive_present_frames += 1
                p_state.consecutive_missed_frames = 0
                p_state.last_seen_frame_id = frame_id
                if person_id is not None:
                    p_state.person_id = person_id
                if display_name is not None:
                    p_state.display_name = display_name

            # Check for ENTERED transition
            if p_state.presence_state in (PresenceState.ABSENT, PresenceState.ENTERING):
                if p_state.consecutive_present_frames >= self._config.enter_patience_frames:
                    p_state.presence_state = PresenceState.PRESENT
                    transitions.append(
                        PresenceTransition(
                            event=PresenceEvent.PERSON_ENTERED,
                            track_id=track_id,
                            person_id=p_state.person_id,
                            display_name=p_state.display_name,
                            frame_id=frame_id,
                            timestamp=timestamp,
                        )
                    )
            elif p_state.presence_state == PresenceState.LEAVING:
                # Recovered before full leave patience elapsed
                p_state.presence_state = PresenceState.PRESENT

        # 2. Check for missing tracks (potential departures)
        tracks_to_remove: list[int] = []
        for track_id, p_state in self._states.items():
            if track_id not in active_confirmed_tracks:
                p_state.consecutive_missed_frames += 1
                p_state.consecutive_present_frames = 0

                if p_state.presence_state == PresenceState.PRESENT:
                    p_state.presence_state = PresenceState.LEAVING

                if p_state.consecutive_missed_frames >= self._config.leave_patience_frames:
                    if p_state.presence_state == PresenceState.LEAVING:
                        transitions.append(
                            PresenceTransition(
                                event=PresenceEvent.PERSON_LEFT,
                                track_id=track_id,
                                person_id=p_state.person_id,
                                display_name=p_state.display_name,
                                frame_id=frame_id,
                                timestamp=timestamp,
                            )
                        )
                    tracks_to_remove.append(track_id)

        for track_id in tracks_to_remove:
            del self._states[track_id]

        return transitions

    def get_confirmed_present_tracks(self) -> set[int]:
        """Return track IDs currently confirmed in PRESENT state."""
        return {
            t_id for t_id, s in self._states.items() if s.presence_state == PresenceState.PRESENT
        }

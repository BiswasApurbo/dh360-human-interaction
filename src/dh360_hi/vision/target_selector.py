"""Multi-person interaction target selection with deterministic scoring and hysteresis."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from dh360_hi.vision.identity_state import IdentityState, TrackIdentityState
from dh360_hi.vision.tracker import TrackedFace, TrackState


@dataclass(frozen=True)
class TargetCandidate:
    """Evaluated interaction candidate with composite and breakdown scores."""

    track_id: int
    score: float
    centrality_score: float
    size_score: float
    identity_state_score: float
    dwell_score: float
    is_identified: bool
    identity_state: IdentityState
    person_id: str | None
    display_name: str | None
    bbox: tuple[float, float, float, float]
    center: tuple[float, float]


@dataclass(frozen=True)
class TargetSelectionResult:
    """Immutable result of interaction target selection for a single frame."""

    primary_target: TargetCandidate | None
    candidates: list[TargetCandidate]
    target_changed: bool
    previous_target_track_id: int | None
    frame_id: int

    @property
    def has_target(self) -> bool:
        """Check if an active target was selected."""
        return self.primary_target is not None

    @property
    def primary_track_id(self) -> int | None:
        """Track ID of primary target if any."""
        return self.primary_target.track_id if self.primary_target else None


@dataclass(frozen=True)
class TargetSelectionConfig:
    """Configuration weights and hysteresis parameters for target selection."""

    weight_centrality: float = 0.35
    weight_size: float = 0.35
    weight_identity: float = 0.20
    weight_dwell: float = 0.10
    # Margin above current target score required for a rival candidate to seize focus
    hysteresis_switch_margin: float = 0.15
    max_lost_frames_to_release: int = 15
    max_dwell_frames_normalization: float = 150.0  # ~5 seconds at 30 fps for full dwell weight


class BaseTargetSelector(ABC):
    """Abstract interface for interaction target selection."""

    @abstractmethod
    def select_target(
        self,
        tracks: Sequence[TrackedFace],
        identity_states: dict[int, TrackIdentityState],
        frame_width: int,
        frame_height: int,
        frame_id: int,
    ) -> TargetSelectionResult:
        """Select primary interaction target from visible tracks and identity states."""

    @abstractmethod
    def reset(self) -> None:
        """Reset internal target selection history and hysteresis state."""


class InteractionTargetSelector(BaseTargetSelector):
    """Deterministic, score-based interaction target selector with stickiness hysteresis."""

    def __init__(self, config: TargetSelectionConfig | None = None) -> None:
        self._config = config or TargetSelectionConfig()
        self._current_target_track_id: int | None = None
        self._last_target_seen_frame_id: int = 0

    @property
    def current_target_track_id(self) -> int | None:
        """Currently engaged primary target track ID."""
        return self._current_target_track_id

    def reset(self) -> None:
        """Reset active target selection."""
        self._current_target_track_id = None
        self._last_target_seen_frame_id = 0

    def select_target(
        self,
        tracks: Sequence[TrackedFace],
        identity_states: dict[int, TrackIdentityState],
        frame_width: int,
        frame_height: int,
        frame_id: int,
    ) -> TargetSelectionResult:
        """Select primary interaction target with deterministic tie-breaking and hysteresis."""
        # 1. Filter to confirmed active tracks
        active_tracks = [t for t in tracks if t.state == TrackState.CONFIRMED]

        if not active_tracks or frame_width <= 0 or frame_height <= 0:
            previous_target = self._current_target_track_id
            if (
                self._current_target_track_id is not None
                and (frame_id - self._last_target_seen_frame_id)
                > self._config.max_lost_frames_to_release
            ):
                self._current_target_track_id = None

            target_changed = self._current_target_track_id != previous_target
            return TargetSelectionResult(
                primary_target=None,
                candidates=[],
                target_changed=target_changed,
                previous_target_track_id=previous_target,
                frame_id=frame_id,
            )

        center_x = frame_width / 2.0
        center_y = frame_height / 2.0
        max_dist = math.hypot(center_x, center_y)
        frame_area = float(frame_width * frame_height)

        # 2. Score each candidate
        candidate_list: list[TargetCandidate] = []
        for track in active_tracks:
            cx, cy = track.center

            # Centrality score: 1.0 at image center, decaying towards edges
            dist_to_center = math.hypot(cx - center_x, cy - center_y)
            centrality = max(0.0, 1.0 - (dist_to_center / max_dist))

            # Size score: square root of bounding box area ratio relative to frame
            area_ratio = min(1.0, track.area / (frame_area * 0.25))  # 25% frame is max score
            size_score = math.sqrt(area_ratio)

            # Identity State score: bonus for confirmed registered users
            identity_state = identity_states.get(track.track_id)
            if identity_state is not None:
                state_enum = identity_state.state
                person_id = identity_state.current_person_id
                display_name = identity_state.current_display_name
                is_identified = identity_state.is_confirmed

                if state_enum == IdentityState.IDENTIFIED:
                    state_score = 1.0
                elif state_enum in (IdentityState.COLLECTING, IdentityState.UNREGISTERED):
                    state_score = 0.5
                else:  # UNCERTAIN, NEW, etc.
                    state_score = 0.2
            else:
                state_enum = IdentityState.NEW
                person_id = None
                display_name = None
                is_identified = False
                state_score = 0.2

            # Dwell score: track longevity
            dwell_frames = float(track.age)
            dwell_score = min(1.0, dwell_frames / self._config.max_dwell_frames_normalization)

            # Composite weighted score
            composite_score = (
                self._config.weight_centrality * centrality
                + self._config.weight_size * size_score
                + self._config.weight_identity * state_score
                + self._config.weight_dwell * dwell_score
            )

            candidate_list.append(
                TargetCandidate(
                    track_id=track.track_id,
                    score=composite_score,
                    centrality_score=centrality,
                    size_score=size_score,
                    identity_state_score=state_score,
                    dwell_score=dwell_score,
                    is_identified=is_identified,
                    identity_state=state_enum,
                    person_id=person_id,
                    display_name=display_name,
                    bbox=track.bbox,
                    center=(cx, cy),
                )
            )

        # 3. Sort candidates deterministically by score descending, with track_id as tie-breaker
        candidate_list.sort(key=lambda c: (-c.score, c.track_id))

        top_candidate = candidate_list[0]
        previous_target = self._current_target_track_id

        # 4. Apply Target Stickiness Hysteresis
        chosen_target: TargetCandidate = top_candidate
        if self._current_target_track_id is not None:
            # Find candidate corresponding to currently engaged target
            current_target_cand = next(
                (c for c in candidate_list if c.track_id == self._current_target_track_id),
                None,
            )

            # Retain current target unless rival exceeds its score + hysteresis margin
            if (
                current_target_cand is not None
                and top_candidate.track_id != self._current_target_track_id
                and (top_candidate.score - current_target_cand.score)
                < self._config.hysteresis_switch_margin
            ):
                chosen_target = current_target_cand

        # 5. Update state
        self._current_target_track_id = chosen_target.track_id
        self._last_target_seen_frame_id = frame_id
        target_changed = chosen_target.track_id != previous_target

        return TargetSelectionResult(
            primary_target=chosen_target,
            candidates=candidate_list,
            target_changed=target_changed,
            previous_target_track_id=previous_target,
            frame_id=frame_id,
        )

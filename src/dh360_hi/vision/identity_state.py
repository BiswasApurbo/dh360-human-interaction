"""Per-track identity state machine with temporal evidence voting and hysteresis."""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from dh360_hi.vision.quality_gate import FaceQualityTier
from dh360_hi.vision.recognition import RecognitionOutcome, RecognitionResult
from dh360_hi.vision.tracker import TrackedFace, compute_iou


class IdentityState(StrEnum):
    """Lifecycle states of a tracked person's identity."""

    NEW = "new"  # Newly initialized track, no recognition evidence yet
    COLLECTING = "collecting"  # Accumulating multi-frame recognition observations
    IDENTIFIED = "identified"  # Stable, verified match against registered gallery identity
    UNREGISTERED = "unregistered"  # Consistently observed but matches no gallery entry
    UNCERTAIN = "uncertain"  # Ambiguous scores, conflicting evidence, or borderlines
    LOST = "lost"  # Track expired or terminated


@dataclass(frozen=True)
class IdentityStateConfig:
    """Parameters and thresholds governing the identity state machine."""

    min_observations_to_decide: int = 3
    evidence_window_size: int = 7
    identification_consistency_ratio: float = 0.70
    hysteresis_patience_observations: int = 4
    max_missed_frames_to_lost: int = 30
    reverification_interval_frames: int = 90
    spatial_proximity_distance_px: float = 80.0
    spatial_proximity_min_iou: float = 0.08


@dataclass
class TrackIdentityState:
    """Internal mutable and observable state for a single tracked face's identity."""

    track_id: int
    state: IdentityState = IdentityState.NEW
    current_person_id: str | None = None
    current_display_name: str | None = None
    confidence_score: float = 0.0
    observation_count: int = 0
    consecutive_conflicts: int = 0
    last_seen_frame_id: int = 0
    state_change_frame_id: int = 0
    last_verification_frame_id: int = 0
    last_embedding_request_frame_id: int = 0
    is_spatial_ambiguous: bool = False
    last_spatial_ambiguity_frame_id: int = 0
    recent_observations: deque[RecognitionResult] = field(default_factory=lambda: deque(maxlen=7))

    @property
    def is_confirmed(self) -> bool:
        """Check if identity is currently in IDENTIFIED state."""
        return self.state == IdentityState.IDENTIFIED


class IdentityStateManager:
    """Manages identity state machines and temporal evidence smoothing across all active tracks.

    Threading Contract:
        IdentityStateManager is designed for single-threaded execution on the primary
        vision perception loop thread. External operations (such as PersonService
        profile deletion or biometric reset) must either be invoked when the perception
        loop is paused or between frame processing steps.
    """

    def __init__(self, config: IdentityStateConfig | None = None) -> None:
        self._config = config or IdentityStateConfig()
        self._tracks: dict[int, TrackIdentityState] = {}

    @property
    def config(self) -> IdentityStateConfig:
        return self._config

    @config.setter
    def config(self, config: IdentityStateConfig) -> None:
        self._config = config

    def get_track_state(self, track_id: int) -> TrackIdentityState | None:
        """Retrieve current identity state for a specific track ID."""
        return self._tracks.get(track_id)

    def get_all_states(self) -> dict[int, TrackIdentityState]:
        """Retrieve a copy of all active track identity states."""
        return dict(self._tracks)

    def ensure_track(self, track_id: int, frame_id: int) -> TrackIdentityState:
        """Ensure a track exists in the identity manager and update its last seen frame."""
        if track_id not in self._tracks:
            track_state = TrackIdentityState(
                track_id=track_id,
                state=IdentityState.NEW,
                last_seen_frame_id=frame_id,
                state_change_frame_id=frame_id,
                last_verification_frame_id=frame_id,
                recent_observations=deque(maxlen=self._config.evidence_window_size),
            )
            self._tracks[track_id] = track_state
        else:
            track_state = self._tracks[track_id]
            track_state.last_seen_frame_id = frame_id
        return track_state

    def update(
        self,
        track_id: int,
        rec_result: RecognitionResult | None,
        frame_id: int,
        create_if_missing: bool = True,
    ) -> TrackIdentityState | None:
        """Update a track's identity state with an optional recognition observation.

        Args:
            track_id: Target track ID.
            rec_result: Recognition result observation (or None).
            frame_id: Current frame index.
            create_if_missing: If True, initialize a new TrackIdentityState if not present.
                If False (e.g. for asynchronous worker results), late results for pruned
                or nonexistent tracks are discarded and return None.

        Returns:
            Updated TrackIdentityState, or None if track does not exist and
            create_if_missing is False.
        """
        if track_id not in self._tracks:
            if not create_if_missing:
                return None
            # Initialize new track state
            track_state = TrackIdentityState(
                track_id=track_id,
                state=IdentityState.NEW,
                last_seen_frame_id=frame_id,
                state_change_frame_id=frame_id,
                last_verification_frame_id=frame_id,
                recent_observations=deque(maxlen=self._config.evidence_window_size),
            )
            self._tracks[track_id] = track_state
        else:
            track_state = self._tracks[track_id]
            track_state.last_seen_frame_id = frame_id

        # If no recognition observation (e.g. frame skipped or quality failed), return current state
        if rec_result is None or rec_result.outcome == RecognitionOutcome.NO_MATCH:
            return track_state

        # Record observation
        track_state.recent_observations.append(rec_result)
        track_state.observation_count += 1

        self._evaluate_state_transitions(track_state, rec_result, frame_id)
        return track_state

    def _evaluate_state_transitions(
        self,
        track_state: TrackIdentityState,
        latest_rec: RecognitionResult,
        frame_id: int,
    ) -> None:
        """Evaluate evidence window and state transitions."""
        obs_window = list(track_state.recent_observations)
        n_obs = len(obs_window)

        # 1. NEW -> COLLECTING transition
        if track_state.state == IdentityState.NEW:
            track_state.state = IdentityState.COLLECTING
            track_state.state_change_frame_id = frame_id

        # If not enough observations yet, remain in COLLECTING
        if n_obs < self._config.min_observations_to_decide:
            return

        # 2. Analyze Observation Window
        identified_obs = [
            o for o in obs_window if o.outcome == RecognitionOutcome.IDENTIFIED and o.person_id
        ]
        unregistered_obs = [o for o in obs_window if o.outcome == RecognitionOutcome.UNREGISTERED]
        uncertain_obs = [o for o in obs_window if o.outcome == RecognitionOutcome.UNCERTAIN]

        person_counter = Counter([o.person_id for o in identified_obs if o.person_id is not None])

        top_person_id, top_count = person_counter.most_common(1)[0] if person_counter else (None, 0)
        consistency_ratio = float(top_count / n_obs) if n_obs > 0 else 0.0

        # 3. State Machine Logic with Hysteresis
        if track_state.state == IdentityState.IDENTIFIED:
            # Hysteresis: check if current identity is contested
            if latest_rec.is_identified and latest_rec.person_id == track_state.current_person_id:
                # Reinforce existing identity
                track_state.consecutive_conflicts = 0
                track_state.confidence_score = latest_rec.confidence_score
                track_state.last_verification_frame_id = frame_id
            else:
                # Contradictory observation
                track_state.consecutive_conflicts += 1
                if (
                    track_state.consecutive_conflicts
                    >= self._config.hysteresis_patience_observations
                ):
                    # Patience exhausted -> demote to COLLECTING/UNCERTAIN
                    self._transition_to(track_state, IdentityState.UNCERTAIN, frame_id)
                    track_state.current_person_id = None
                    track_state.current_display_name = None
                    track_state.consecutive_conflicts = 0

        elif track_state.state in (
            IdentityState.COLLECTING,
            IdentityState.UNCERTAIN,
            IdentityState.UNREGISTERED,
        ):
            # Evaluate for IDENTIFIED
            if (
                top_person_id is not None
                and consistency_ratio >= self._config.identification_consistency_ratio
            ):
                top_match_sample = next(o for o in identified_obs if o.person_id == top_person_id)
                self._transition_to(track_state, IdentityState.IDENTIFIED, frame_id)
                track_state.current_person_id = top_person_id
                track_state.current_display_name = top_match_sample.display_name
                track_state.confidence_score = top_match_sample.confidence_score
                track_state.consecutive_conflicts = 0
                track_state.last_verification_frame_id = frame_id

            # Evaluate for UNREGISTERED
            elif len(unregistered_obs) / n_obs >= self._config.identification_consistency_ratio:
                self._transition_to(track_state, IdentityState.UNREGISTERED, frame_id)
                track_state.current_person_id = None
                track_state.current_display_name = None
                track_state.confidence_score = float(
                    np.mean([o.confidence_score for o in unregistered_obs])
                )
                track_state.consecutive_conflicts = 0

            # Evaluate for UNCERTAIN (conflicting identities or low consistency)
            elif (
                len(person_counter) > 1
                or len(uncertain_obs) / n_obs > 0.30
                or (
                    len(identified_obs) > 0
                    and consistency_ratio < self._config.identification_consistency_ratio
                )
            ):
                self._transition_to(track_state, IdentityState.UNCERTAIN, frame_id)
                track_state.current_person_id = None
                track_state.current_display_name = None
                track_state.confidence_score = latest_rec.confidence_score

    def _transition_to(
        self, track_state: TrackIdentityState, new_state: IdentityState, frame_id: int
    ) -> None:
        """Execute a state transition and update frame metadata."""
        if track_state.state != new_state:
            track_state.state = new_state
            track_state.state_change_frame_id = frame_id

    def prune_lost_tracks(
        self, current_frame_id: int, active_track_ids: set[int] | None = None
    ) -> list[int]:
        """Prune tracks that have disappeared beyond the maximum allowed frames."""
        pruned = []
        for track_id, track_state in list(self._tracks.items()):
            is_active = active_track_ids is not None and track_id in active_track_ids
            missed_frames = current_frame_id - track_state.last_seen_frame_id

            if not is_active and missed_frames > self._config.max_missed_frames_to_lost:
                track_state.state = IdentityState.LOST
                track_state.state_change_frame_id = current_frame_id
                pruned.append(track_id)

        for track_id in pruned:
            del self._tracks[track_id]

        return pruned

    def check_track_crossings(
        self,
        active_tracks: Sequence[TrackedFace],
        current_frame_id: int,
    ) -> set[int]:
        """Detect spatial crossings/proximity among active tracks to protect against ID swaps.

        When two tracks come into close proximity (low centroid distance or bounding box overlap),
        both tracks are flagged with spatial ambiguity so identity stability requires fresh
        independent recognition evidence rather than momentum.

        Args:
            active_tracks: Currently visible active tracks.
            current_frame_id: Current frame index.

        Returns:
            Set of track IDs that are in spatial crossing/proximity.
        """
        # Reset spatial ambiguity flag for all active tracks for current frame evaluation
        for t in active_tracks:
            state = self._tracks.get(t.track_id)
            if state is not None:
                state.is_spatial_ambiguous = False

        ambiguous_tracks: set[int] = set()
        n_tracks = len(active_tracks)
        if n_tracks < 2:
            return ambiguous_tracks

        for i in range(n_tracks):
            t1 = active_tracks[i]
            c1_x, c1_y = t1.center
            for j in range(i + 1, n_tracks):
                t2 = active_tracks[j]
                c2_x, c2_y = t2.center

                # Calculate spatial centroid distance
                dist = float(np.hypot(c1_x - c2_x, c1_y - c2_y))
                iou = compute_iou(t1.bbox, t2.bbox)

                if (
                    dist <= self._config.spatial_proximity_distance_px
                    or iou >= self._config.spatial_proximity_min_iou
                ):
                    ambiguous_tracks.add(t1.track_id)
                    ambiguous_tracks.add(t2.track_id)

        # Mark ambiguity on track identity states
        for track_id in ambiguous_tracks:
            state = self._tracks.get(track_id)
            if state is not None:
                state.is_spatial_ambiguous = True
                state.last_spatial_ambiguity_frame_id = current_frame_id

        return ambiguous_tracks

    def should_request_embedding(
        self,
        track_id: int,
        frame_id: int,
        quality_tier: FaceQualityTier,
    ) -> bool:
        """Evaluate whether a face crop from a track should be submitted for embedding/recognition.

        Decision rules:
        - Rejected / Low quality: do not submit (saves worker/CPU budget).
        - NEW / COLLECTING / UNCERTAIN: submit on MEDIUM or HIGH to gather evidence.
        - UNREGISTERED: submit periodically (every ~30 frames) on MEDIUM or HIGH.
        - IDENTIFIED: submit on HIGH or MEDIUM if spatial ambiguity is active or occurred recently,
          or if reverification interval has elapsed.

        Args:
            track_id: Target track ID.
            frame_id: Current frame index.
            quality_tier: Assessed quality tier from FaceQualityGate.

        Returns:
            True if an embedding task should be queued, False otherwise.
        """
        if quality_tier in (FaceQualityTier.REJECTED, FaceQualityTier.LOW):
            return False

        track_state = self._tracks.get(track_id)
        if track_state is None:
            return True  # New track, needs embedding

        if track_state.state in (
            IdentityState.NEW,
            IdentityState.COLLECTING,
            IdentityState.UNCERTAIN,
        ):
            return True

        if track_state.state == IdentityState.UNREGISTERED:
            # Periodic check for newly enrolled persons
            return frame_id - track_state.last_embedding_request_frame_id >= (
                self._config.reverification_interval_frames // 3
            )

        if track_state.state == IdentityState.IDENTIFIED:
            # Re-verify if spatial crossing/proximity is active or occurred within last 30 frames
            if track_state.is_spatial_ambiguous or (
                track_state.last_spatial_ambiguity_frame_id > 0
                and (frame_id - track_state.last_spatial_ambiguity_frame_id) <= 30
            ):
                return True

            # Periodic re-verification
            return (
                frame_id - track_state.last_verification_frame_id
                >= self._config.reverification_interval_frames
            )

        return False

    def mark_embedding_requested(self, track_id: int, frame_id: int) -> None:
        """Record timestamp/frame_id of an embedding submission."""
        track_state = self._tracks.get(track_id)
        if track_state is not None:
            track_state.last_embedding_request_frame_id = frame_id

    def invalidate_person(self, person_id: str, frame_id: int = 0) -> list[int]:
        """Invalidate identity state for any active tracks currently associated with person_id.

        Args:
            person_id: The person ID being deleted or whose biometrics were removed.
            frame_id: Optional current frame index.

        Returns:
            List of track IDs whose identity state was invalidated.
        """
        invalidated_tracks: list[int] = []
        for track_id, track_state in self._tracks.items():
            has_person_id = track_state.current_person_id == person_id
            has_in_history = any(
                obs.person_id == person_id for obs in track_state.recent_observations
            )
            if has_person_id or has_in_history:
                track_state.current_person_id = None
                track_state.current_display_name = None
                track_state.confidence_score = 0.0
                track_state.consecutive_conflicts = 0
                track_state.recent_observations.clear()
                self._transition_to(track_state, IdentityState.UNREGISTERED, frame_id)
                invalidated_tracks.append(track_id)

        return invalidated_tracks

    def reset(self) -> None:
        """Reset all track identity states."""
        self._tracks.clear()

"""Unit tests for the per-track identity state machine and temporal smoothing."""

from dh360_hi.vision.identity_state import (
    IdentityState,
    IdentityStateConfig,
    IdentityStateManager,
)
from dh360_hi.vision.recognition import (
    MatchCandidate,
    RecognitionOutcome,
    RecognitionResult,
)


def _create_mock_recognition_result(
    outcome: RecognitionOutcome,
    person_id: str | None = None,
    display_name: str | None = None,
    similarity: float = 0.85,
) -> RecognitionResult:
    """Helper to create a deterministic recognition result."""
    top_match = (
        MatchCandidate(
            person_id=person_id,
            display_name=display_name or person_id or "Unknown",
            similarity=similarity,
            distance=1.0 - similarity,
        )
        if person_id
        else None
    )
    return RecognitionResult(
        outcome=outcome,
        top_match=top_match,
        second_match=None,
        similarity_margin=0.20 if top_match else 0.0,
        confidence_score=similarity,
        model_name="SFace",
        model_version="2021dec",
        is_model_compatible=True,
        details="Mock test recognition",
    )


def test_new_track_initialization_and_collection() -> None:
    """Verify newly tracked face initializes in NEW state and enters COLLECTING."""
    manager = IdentityStateManager(config=IdentityStateConfig(min_observations_to_decide=3))

    # Frame 1: update with no recognition yet
    s1 = manager.update(track_id=1, rec_result=None, frame_id=1)
    assert s1.state == IdentityState.NEW
    assert s1.observation_count == 0

    # Frame 2: first observation enters COLLECTING
    rec_obs = _create_mock_recognition_result(RecognitionOutcome.IDENTIFIED, "p001", "Alice")
    s2 = manager.update(track_id=1, rec_result=rec_obs, frame_id=2)
    assert s2.state == IdentityState.COLLECTING
    assert s2.observation_count == 1
    assert s2.current_person_id is None


def test_transition_to_identified_after_consistent_evidence() -> None:
    """Verify track transitions to IDENTIFIED after consistent multi-frame matches."""
    manager = IdentityStateManager(
        config=IdentityStateConfig(
            min_observations_to_decide=3, identification_consistency_ratio=0.70
        )
    )

    rec_alice = _create_mock_recognition_result(
        RecognitionOutcome.IDENTIFIED, "p001", "Alice Smith", 0.88
    )

    manager.update(track_id=1, rec_result=rec_alice, frame_id=1)
    manager.update(track_id=1, rec_result=rec_alice, frame_id=2)
    s3 = manager.update(track_id=1, rec_result=rec_alice, frame_id=3)

    assert s3.state == IdentityState.IDENTIFIED
    assert s3.is_confirmed is True
    assert s3.current_person_id == "p001"
    assert s3.current_display_name == "Alice Smith"
    assert s3.confidence_score == 0.88


def test_transition_to_unregistered_when_consistently_unknown() -> None:
    """Verify track transitions to UNREGISTERED when observations consistently fail match."""
    manager = IdentityStateManager(
        config=IdentityStateConfig(
            min_observations_to_decide=3, identification_consistency_ratio=0.70
        )
    )

    rec_unknown = _create_mock_recognition_result(RecognitionOutcome.UNREGISTERED, similarity=0.30)

    manager.update(track_id=2, rec_result=rec_unknown, frame_id=1)
    manager.update(track_id=2, rec_result=rec_unknown, frame_id=2)
    s3 = manager.update(track_id=2, rec_result=rec_unknown, frame_id=3)

    assert s3.state == IdentityState.UNREGISTERED
    assert s3.current_person_id is None


def test_transition_to_uncertain_on_conflicting_evidence() -> None:
    """Verify track enters UNCERTAIN state when evidence is split between different identities."""
    manager = IdentityStateManager(
        config=IdentityStateConfig(
            min_observations_to_decide=3,
            evidence_window_size=4,
            identification_consistency_ratio=0.75,
        )
    )

    rec_alice = _create_mock_recognition_result(RecognitionOutcome.IDENTIFIED, "p001", "Alice")
    rec_bob = _create_mock_recognition_result(RecognitionOutcome.IDENTIFIED, "p002", "Bob")

    manager.update(track_id=3, rec_result=rec_alice, frame_id=1)
    manager.update(track_id=3, rec_result=rec_bob, frame_id=2)
    manager.update(track_id=3, rec_result=rec_alice, frame_id=3)
    s4 = manager.update(track_id=3, rec_result=rec_bob, frame_id=4)

    assert s4.state == IdentityState.UNCERTAIN
    assert s4.current_person_id is None


def test_hysteresis_prevents_flipping_on_isolated_conflict() -> None:
    """Verify single contradictory observation does not revoke confirmed IDENTIFIED state."""
    manager = IdentityStateManager(
        config=IdentityStateConfig(min_observations_to_decide=3, hysteresis_patience_observations=3)
    )

    rec_alice = _create_mock_recognition_result(RecognitionOutcome.IDENTIFIED, "p001", "Alice")
    rec_intruder = _create_mock_recognition_result(
        RecognitionOutcome.IDENTIFIED, "p999", "Intruder"
    )

    # Establish Alice
    for fid in range(1, 4):
        manager.update(track_id=1, rec_result=rec_alice, frame_id=fid)

    s_confirmed = manager.get_track_state(1)
    assert s_confirmed is not None
    assert s_confirmed.state == IdentityState.IDENTIFIED

    # 1 conflicting observation -> Hysteresis absorbs it
    s_contested_1 = manager.update(track_id=1, rec_result=rec_intruder, frame_id=4)
    assert s_contested_1.state == IdentityState.IDENTIFIED
    assert s_contested_1.current_person_id == "p001"
    assert s_contested_1.consecutive_conflicts == 1

    # Return to Alice observation -> Clears conflict counter
    s_restored = manager.update(track_id=1, rec_result=rec_alice, frame_id=5)
    assert s_restored.state == IdentityState.IDENTIFIED
    assert s_restored.consecutive_conflicts == 0


def test_hysteresis_demotes_after_sustained_conflicts() -> None:
    """Verify sustained conflicting observations eventually demote confirmed identity."""
    manager = IdentityStateManager(
        config=IdentityStateConfig(min_observations_to_decide=3, hysteresis_patience_observations=3)
    )

    rec_alice = _create_mock_recognition_result(RecognitionOutcome.IDENTIFIED, "p001", "Alice")
    rec_unknown = _create_mock_recognition_result(RecognitionOutcome.UNREGISTERED, similarity=0.2)

    for fid in range(1, 4):
        manager.update(track_id=1, rec_result=rec_alice, frame_id=fid)

    # 3 consecutive unknown observations -> exceeds patience of 3
    manager.update(track_id=1, rec_result=rec_unknown, frame_id=4)
    manager.update(track_id=1, rec_result=rec_unknown, frame_id=5)
    s_demoted = manager.update(track_id=1, rec_result=rec_unknown, frame_id=6)

    assert s_demoted.state == IdentityState.UNCERTAIN
    assert s_demoted.current_person_id is None


def test_prune_lost_tracks() -> None:
    """Verify inactive tracks exceeding max_missed_frames are marked LOST and removed."""
    manager = IdentityStateManager(config=IdentityStateConfig(max_missed_frames_to_lost=10))

    manager.update(track_id=1, rec_result=None, frame_id=5)
    manager.update(track_id=2, rec_result=None, frame_id=18)

    # At frame 20: track 1 missed 15 frames (> 10), track 2 missed 2 frames
    pruned = manager.prune_lost_tracks(current_frame_id=20, active_track_ids={2})

    assert pruned == [1]
    assert manager.get_track_state(1) is None
    assert manager.get_track_state(2) is not None


def test_multiple_simultaneous_tracks_independent_states() -> None:
    """Verify multiple simultaneous tracks maintain separate identity states."""
    manager = IdentityStateManager(config=IdentityStateConfig(min_observations_to_decide=2))

    rec_alice = _create_mock_recognition_result(RecognitionOutcome.IDENTIFIED, "p001", "Alice")
    rec_bob = _create_mock_recognition_result(RecognitionOutcome.IDENTIFIED, "p002", "Bob")

    # Track 1 is Alice, Track 2 is Bob
    manager.update(track_id=1, rec_result=rec_alice, frame_id=1)
    manager.update(track_id=2, rec_result=rec_bob, frame_id=1)
    s1 = manager.update(track_id=1, rec_result=rec_alice, frame_id=2)
    s2 = manager.update(track_id=2, rec_result=rec_bob, frame_id=2)

    assert s1.state == IdentityState.IDENTIFIED
    assert s1.current_person_id == "p001"

    assert s2.state == IdentityState.IDENTIFIED
    assert s2.current_person_id == "p002"

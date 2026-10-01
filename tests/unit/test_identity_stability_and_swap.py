"""Unit tests for identity stability, hysteresis, ID swap protection, and request policy."""

from __future__ import annotations

from dh360_hi.vision.identity_state import (
    IdentityState,
    IdentityStateConfig,
    IdentityStateManager,
)
from dh360_hi.vision.quality_gate import FaceQualityTier
from dh360_hi.vision.recognition import (
    MatchCandidate,
    RecognitionOutcome,
    RecognitionResult,
)
from dh360_hi.vision.tracker import TrackedFace, TrackState


def _make_rec(
    person_id: str | None,
    name: str | None,
    outcome: RecognitionOutcome,
    score: float = 0.90,
) -> RecognitionResult:
    """Helper creating a recognition result."""
    top = (
        MatchCandidate(person_id=person_id, display_name=name, similarity=score, distance=1 - score)
        if person_id and name
        else None
    )
    return RecognitionResult(
        outcome=outcome,
        top_match=top,
        second_match=None,
        similarity_margin=score,
        confidence_score=score,
        model_name="SFace",
        model_version="2021dec",
        is_model_compatible=True,
        details="test",
    )


def _make_track(track_id: int, bbox: tuple[float, float, float, float]) -> TrackedFace:
    return TrackedFace(
        track_id=track_id,
        bbox=bbox,
        confidence=0.95,
        state=TrackState.CONFIRMED,
    )


def test_stable_identity_across_temporary_recognition_failures() -> None:
    """Test that temporary noisy/failed recognition observations do not erase stable identity."""
    config = IdentityStateConfig(
        min_observations_to_decide=3,
        hysteresis_patience_observations=4,
    )
    mgr = IdentityStateManager(config)

    rec_alice = _make_rec("p1", "Alice", RecognitionOutcome.IDENTIFIED, 0.92)

    # 1. Establish Alice as IDENTIFIED across 3 frames
    mgr.update(track_id=1, rec_result=rec_alice, frame_id=1)
    mgr.update(track_id=1, rec_result=rec_alice, frame_id=2)
    s = mgr.update(track_id=1, rec_result=rec_alice, frame_id=3)
    assert s.state == IdentityState.IDENTIFIED
    assert s.current_person_id == "p1"

    # 2. Temporary recognition failure / uncertain frame
    rec_uncertain = _make_rec(None, None, RecognitionOutcome.UNCERTAIN, 0.40)
    s4 = mgr.update(track_id=1, rec_result=rec_uncertain, frame_id=4)
    # Hysteresis retains Alice
    assert s4.state == IdentityState.IDENTIFIED
    assert s4.current_person_id == "p1"
    assert s4.consecutive_conflicts == 1

    # 3. Another frame missing recognition (None)
    s5 = mgr.update(track_id=1, rec_result=None, frame_id=5)
    assert s5.state == IdentityState.IDENTIFIED
    assert s5.current_person_id == "p1"

    # 4. Next frame Alice recognized again -> conflicts reset
    s6 = mgr.update(track_id=1, rec_result=rec_alice, frame_id=6)
    assert s6.state == IdentityState.IDENTIFIED
    assert s6.current_person_id == "p1"
    assert s6.consecutive_conflicts == 0


def test_conflicting_recognition_requires_sustained_evidence_to_demote() -> None:
    """Test that a rival identity (Bob) cannot immediately overwrite Alice

    without sustained evidence.
    """
    config = IdentityStateConfig(
        min_observations_to_decide=3,
        hysteresis_patience_observations=3,
        identification_consistency_ratio=0.70,
    )
    mgr = IdentityStateManager(config)

    rec_alice = _make_rec("p1", "Alice", RecognitionOutcome.IDENTIFIED, 0.90)
    rec_bob = _make_rec("p2", "Bob", RecognitionOutcome.IDENTIFIED, 0.90)

    # Establish Alice
    for f in range(1, 4):
        mgr.update(track_id=1, rec_result=rec_alice, frame_id=f)
    assert mgr.get_track_state(1).state == IdentityState.IDENTIFIED
    assert mgr.get_track_state(1).current_person_id == "p1"

    # Frame 4: 1st Bob sighting -> Alice retained
    s4 = mgr.update(track_id=1, rec_result=rec_bob, frame_id=4)
    assert s4.state == IdentityState.IDENTIFIED
    assert s4.current_person_id == "p1"
    assert s4.consecutive_conflicts == 1

    # Frame 5: 2nd Bob sighting -> Alice still retained
    s5 = mgr.update(track_id=1, rec_result=rec_bob, frame_id=5)
    assert s5.state == IdentityState.IDENTIFIED
    assert s5.consecutive_conflicts == 2

    # Frame 6: 3rd Bob sighting -> Patience (3) exhausted -> Demotes to UNCERTAIN
    s6 = mgr.update(track_id=1, rec_result=rec_bob, frame_id=6)
    assert s6.state == IdentityState.UNCERTAIN
    assert s6.current_person_id is None

    # Frame 7 & 8: Bob continues consistently -> Transitions to IDENTIFIED as Bob
    mgr.update(track_id=1, rec_result=rec_bob, frame_id=7)
    s8 = mgr.update(track_id=1, rec_result=rec_bob, frame_id=8)
    assert s8.state == IdentityState.IDENTIFIED
    assert s8.current_person_id == "p2"
    assert s8.current_display_name == "Bob"


def test_track_reappearance_requires_fresh_evidence_for_new_track_id() -> None:
    """Test that when a track expires and a new track ID appears nearby, it starts in NEW state."""
    config = IdentityStateConfig(min_observations_to_decide=3, max_missed_frames_to_lost=5)
    mgr = IdentityStateManager(config)

    rec_alice = _make_rec("p1", "Alice", RecognitionOutcome.IDENTIFIED, 0.90)

    # Track 1 established as Alice
    for f in range(1, 4):
        mgr.update(track_id=1, rec_result=rec_alice, frame_id=f)
    assert mgr.get_track_state(1).state == IdentityState.IDENTIFIED

    # Track 1 disappears and is pruned
    pruned = mgr.prune_lost_tracks(current_frame_id=10, active_track_ids=set())
    assert 1 in pruned
    assert mgr.get_track_state(1) is None

    # New Track 2 appears in scene -> Starts as NEW, not Alice
    s2 = mgr.update(track_id=2, rec_result=None, frame_id=11)
    assert s2.state == IdentityState.NEW
    assert s2.current_person_id is None


def test_spatial_crossing_detection_flags_ambiguity() -> None:
    """Test that check_track_crossings detects close bounding boxes and flags both tracks."""
    mgr = IdentityStateManager()

    # Track 1 at (100, 100, 80, 80) and Track 2 far away at (400, 100, 80, 80)
    t1 = _make_track(1, (100.0, 100.0, 80.0, 80.0))
    t2 = _make_track(2, (400.0, 100.0, 80.0, 80.0))

    mgr.update(1, None, frame_id=1)
    mgr.update(2, None, frame_id=1)

    # No crossing when far apart
    crossings = mgr.check_track_crossings([t1, t2], current_frame_id=1)
    assert len(crossings) == 0
    assert not mgr.get_track_state(1).is_spatial_ambiguous

    # Frame 2: Tracks move close / overlap (crossing)
    t1_close = _make_track(1, (200.0, 100.0, 80.0, 80.0))
    t2_close = _make_track(2, (220.0, 100.0, 80.0, 80.0))  # Centroid distance = 20px < 80px

    crossings_f2 = mgr.check_track_crossings([t1_close, t2_close], current_frame_id=2)
    assert crossings_f2 == {1, 2}
    assert mgr.get_track_state(1).is_spatial_ambiguous is True
    assert mgr.get_track_state(2).is_spatial_ambiguous is True
    assert mgr.get_track_state(1).last_spatial_ambiguity_frame_id == 2


def test_should_request_embedding_policy() -> None:
    """Test embedding task submission policy for different states and quality tiers."""
    mgr = IdentityStateManager()

    # 1. Rejected and Low quality tiers should never request embeddings
    assert not mgr.should_request_embedding(1, 1, FaceQualityTier.REJECTED)
    assert not mgr.should_request_embedding(1, 1, FaceQualityTier.LOW)

    # 2. New track with MEDIUM/HIGH quality should request embedding
    assert mgr.should_request_embedding(1, 1, FaceQualityTier.HIGH) is True

    # 3. Established IDENTIFIED track should NOT request embedding immediately on next frame
    rec_alice = _make_rec("p1", "Alice", RecognitionOutcome.IDENTIFIED, 0.90)
    for f in range(1, 4):
        mgr.update(track_id=1, rec_result=rec_alice, frame_id=f)

    assert mgr.get_track_state(1).state == IdentityState.IDENTIFIED
    # Frame 4: not reverification time yet -> False
    assert mgr.should_request_embedding(1, 4, FaceQualityTier.HIGH) is False

    # Frame 100: reverification interval (90) elapsed -> True
    assert mgr.should_request_embedding(1, 100, FaceQualityTier.HIGH) is True

    # Spatial ambiguity triggers immediate re-verification
    mgr.get_track_state(1).is_spatial_ambiguous = True
    mgr.get_track_state(1).last_spatial_ambiguity_frame_id = 10
    assert mgr.should_request_embedding(1, 15, FaceQualityTier.HIGH) is True

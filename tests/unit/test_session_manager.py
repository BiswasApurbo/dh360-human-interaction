"""Unit tests for SessionManager.

All tests use lightweight stubs — no camera, no model, no disk IO.
``TwoRateFrameResult`` objects are built from scratch with minimal fields.
"""

from __future__ import annotations

import pytest

from dh360_hi.contracts.events import (
    BaseEvent,
    PersonEnteredEvent,
    PersonIdentifiedEvent,
    PersonLeftEvent,
    PersonLostIdentityEvent,
    SessionEndedEvent,
    SessionStartedEvent,
)
from dh360_hi.edge.event_bus import EventBus
from dh360_hi.edge.session_manager import SessionConfig, SessionManager
from dh360_hi.vision.identity_state import IdentityState, TrackIdentityState
from dh360_hi.vision.presence_debouncer import PresenceEvent, PresenceTransition
from dh360_hi.vision.target_selector import TargetCandidate, TargetSelectionResult
from dh360_hi.vision.two_rate_pipeline import TwoRateFrameResult

# ---------------------------------------------------------------------------
# Test helpers / builders
# ---------------------------------------------------------------------------

_FRAME_ID = 0


def _next_frame() -> int:
    global _FRAME_ID
    _FRAME_ID += 1
    return _FRAME_ID


def _mk_id_state(
    track_id: int,
    state: IdentityState = IdentityState.NEW,
    person_id: str | None = None,
    display_name: str | None = None,
) -> TrackIdentityState:
    return TrackIdentityState(
        track_id=track_id,
        state=state,
        current_person_id=person_id,
        current_display_name=display_name,
    )


def _mk_candidate(
    track_id: int,
    person_id: str | None = None,
    state: IdentityState = IdentityState.NEW,
) -> TargetCandidate:
    return TargetCandidate(
        track_id=track_id,
        score=1.0,
        centrality_score=1.0,
        size_score=1.0,
        identity_state_score=1.0,
        dwell_score=1.0,
        is_identified=state == IdentityState.IDENTIFIED,
        identity_state=state,
        person_id=person_id,
        display_name=None,
        bbox=(0.0, 0.0, 100.0, 100.0),
        center=(50.0, 50.0),
    )


def _mk_target_result(
    primary_track_id: int | None = None,
    candidates: list[TargetCandidate] | None = None,
    frame_id: int = 1,
) -> TargetSelectionResult:
    primary = None
    if primary_track_id is not None:
        cands = candidates or [_mk_candidate(primary_track_id)]
        primary = next((c for c in cands if c.track_id == primary_track_id), None)
        if primary is None:
            primary = _mk_candidate(primary_track_id)
    return TargetSelectionResult(
        primary_target=primary,
        candidates=candidates or ([primary] if primary else []),
        target_changed=False,
        previous_target_track_id=None,
        frame_id=frame_id,
    )


def _mk_transition(
    event: PresenceEvent,
    track_id: int,
    person_id: str | None = None,
    frame_id: int = 1,
    timestamp: float = 1.0,
) -> PresenceTransition:
    return PresenceTransition(
        event=event,
        track_id=track_id,
        person_id=person_id,
        display_name=None,
        frame_id=frame_id,
        timestamp=timestamp,
    )


def _mk_frame(
    frame_id: int = 1,
    timestamp: float = 1.0,
    primary_track_id: int | None = None,
    identity_states: dict[int, TrackIdentityState] | None = None,
    presence_transitions: list[PresenceTransition] | None = None,
) -> TwoRateFrameResult:
    target_result = _mk_target_result(
        primary_track_id=primary_track_id,
        frame_id=frame_id,
    )
    return TwoRateFrameResult(
        frame_id=frame_id,
        timestamp=timestamp,
        detections=[],
        tracks=[],
        quality_results={},
        identity_states=identity_states or {},
        target_result=target_result,
        presence_transitions=presence_transitions or [],
        completed_recognitions=[],
        ambiguous_crossing_track_ids=set(),
        pending_worker_tasks=0,
        frame_latency_ms=0.0,
        total_pipeline_latency_ms=0.0,
    )


def _manager(require_confirmed_presence: bool = True) -> SessionManager:
    cfg = SessionConfig(require_confirmed_presence=require_confirmed_presence)
    return SessionManager(config=cfg)


# Reset global frame counter before each test
@pytest.fixture(autouse=True)
def reset_frame_counter() -> None:
    global _FRAME_ID
    _FRAME_ID = 0


# ---------------------------------------------------------------------------
# PersonEnteredEvent
# ---------------------------------------------------------------------------


class TestPersonEnteredEvent:
    def test_entered_emitted_for_primary_target_on_presence_transition(self) -> None:
        mgr = _manager()
        frame = _mk_frame(
            frame_id=1,
            primary_track_id=5,
            presence_transitions=[_mk_transition(PresenceEvent.PERSON_ENTERED, track_id=5)],
        )
        events = mgr.process_frame_result(frame)
        entered = [e for e in events if isinstance(e, PersonEnteredEvent)]
        assert len(entered) == 1
        assert entered[0].track_id == 5

    def test_entered_not_emitted_twice_for_same_session(self) -> None:
        mgr = _manager()
        t = _mk_transition(PresenceEvent.PERSON_ENTERED, track_id=5)
        frame1 = _mk_frame(frame_id=1, primary_track_id=5, presence_transitions=[t])
        frame2 = _mk_frame(frame_id=2, primary_track_id=5)
        events1 = mgr.process_frame_result(frame1)
        events2 = mgr.process_frame_result(frame2)
        assert any(isinstance(e, PersonEnteredEvent) for e in events1)
        assert not any(isinstance(e, PersonEnteredEvent) for e in events2)

    def test_entered_carries_person_id_from_identity_state(self) -> None:
        mgr = _manager()
        id_state = _mk_id_state(7, IdentityState.IDENTIFIED, person_id="pid-abc")
        frame = _mk_frame(
            frame_id=1,
            primary_track_id=7,
            identity_states={7: id_state},
            presence_transitions=[_mk_transition(PresenceEvent.PERSON_ENTERED, track_id=7)],
        )
        events = mgr.process_frame_result(frame)
        entered = next(e for e in events if isinstance(e, PersonEnteredEvent))
        assert entered.person_id == "pid-abc"

    def test_entered_person_id_none_for_unregistered(self) -> None:
        mgr = _manager()
        id_state = _mk_id_state(3, IdentityState.UNREGISTERED, person_id=None)
        frame = _mk_frame(
            frame_id=1,
            primary_track_id=3,
            identity_states={3: id_state},
            presence_transitions=[_mk_transition(PresenceEvent.PERSON_ENTERED, track_id=3)],
        )
        events = mgr.process_frame_result(frame)
        entered = next(e for e in events if isinstance(e, PersonEnteredEvent))
        assert entered.person_id is None

    def test_no_entered_event_for_background_person(self) -> None:
        mgr = _manager(require_confirmed_presence=True)
        frame = _mk_frame(
            frame_id=1,
            primary_track_id=1,
            presence_transitions=[
                _mk_transition(PresenceEvent.PERSON_ENTERED, track_id=1),
                _mk_transition(PresenceEvent.PERSON_ENTERED, track_id=2),
            ],
        )
        events = mgr.process_frame_result(frame)
        entered = [e for e in events if isinstance(e, PersonEnteredEvent)]
        assert len(entered) == 1
        assert entered[0].track_id == 1


# ---------------------------------------------------------------------------
# Session start
# ---------------------------------------------------------------------------


class TestSessionStart:
    def test_session_started_when_primary_target_and_confirmed_presence(self) -> None:
        mgr = _manager(require_confirmed_presence=True)
        frame = _mk_frame(
            frame_id=1,
            primary_track_id=1,
            presence_transitions=[_mk_transition(PresenceEvent.PERSON_ENTERED, track_id=1)],
        )
        events = mgr.process_frame_result(frame)
        assert any(isinstance(e, SessionStartedEvent) for e in events)

    def test_session_not_started_without_presence_confirmation(self) -> None:
        """With require_confirmed_presence=True, no session without PERSON_ENTERED."""
        mgr = _manager(require_confirmed_presence=True)
        frame = _mk_frame(frame_id=1, primary_track_id=1)  # no presence transition
        events = mgr.process_frame_result(frame)
        assert not any(isinstance(e, SessionStartedEvent) for e in events)

    def test_session_started_without_presence_requirement(self) -> None:
        """With require_confirmed_presence=False, session starts on primary track alone."""
        mgr = _manager(require_confirmed_presence=False)
        frame = _mk_frame(frame_id=1, primary_track_id=1)
        events = mgr.process_frame_result(frame)
        assert any(isinstance(e, SessionStartedEvent) for e in events)

    def test_session_started_event_fields(self) -> None:
        mgr = _manager(require_confirmed_presence=False)
        id_state = _mk_id_state(1, IdentityState.NEW)
        frame = _mk_frame(
            frame_id=5, timestamp=5.0, primary_track_id=1, identity_states={1: id_state}
        )
        events = mgr.process_frame_result(frame)
        started = next(e for e in events if isinstance(e, SessionStartedEvent))
        assert started.track_id == 1
        assert started.frame_id == 5
        assert started.timestamp == 5.0
        assert started.session_id != ""

    def test_session_started_exactly_once(self) -> None:
        mgr = _manager(require_confirmed_presence=False)
        f1 = _mk_frame(frame_id=1, primary_track_id=1)
        f2 = _mk_frame(frame_id=2, primary_track_id=1)
        events1 = mgr.process_frame_result(f1)
        events2 = mgr.process_frame_result(f2)
        started_count = sum(1 for e in events1 + events2 if isinstance(e, SessionStartedEvent))
        assert started_count == 1

    def test_session_with_unregistered_person(self) -> None:
        mgr = _manager(require_confirmed_presence=False)
        id_state = _mk_id_state(1, IdentityState.UNREGISTERED, person_id=None)
        frame = _mk_frame(frame_id=1, primary_track_id=1, identity_states={1: id_state})
        events = mgr.process_frame_result(frame)
        started = next(e for e in events if isinstance(e, SessionStartedEvent))
        assert started.person_id is None  # Must support unregistered


# ---------------------------------------------------------------------------
# Event ordering guarantees
# ---------------------------------------------------------------------------


class TestEventOrdering:
    def test_entered_before_session_started(self) -> None:
        mgr = _manager(require_confirmed_presence=True)
        frame = _mk_frame(
            frame_id=1,
            primary_track_id=1,
            presence_transitions=[_mk_transition(PresenceEvent.PERSON_ENTERED, track_id=1)],
        )
        events = mgr.process_frame_result(frame)
        types = [type(e) for e in events]
        assert types.index(PersonEnteredEvent) < types.index(SessionStartedEvent)

    def test_entered_before_session_before_identified(self) -> None:
        mgr = _manager(require_confirmed_presence=True)
        id_state = _mk_id_state(1, IdentityState.IDENTIFIED, person_id="p1")
        frame = _mk_frame(
            frame_id=1,
            primary_track_id=1,
            identity_states={1: id_state},
            presence_transitions=[_mk_transition(PresenceEvent.PERSON_ENTERED, track_id=1)],
        )
        events = mgr.process_frame_result(frame)
        types = [type(e) for e in events]
        assert types.index(PersonEnteredEvent) < types.index(SessionStartedEvent)
        assert types.index(SessionStartedEvent) < types.index(PersonIdentifiedEvent)

    def test_left_before_session_ended(self) -> None:
        mgr = _manager(require_confirmed_presence=False)
        f1 = _mk_frame(frame_id=1, primary_track_id=1)
        mgr.process_frame_result(f1)

        f2 = _mk_frame(
            frame_id=2,
            primary_track_id=None,
            presence_transitions=[_mk_transition(PresenceEvent.PERSON_LEFT, track_id=1)],
        )
        events = mgr.process_frame_result(f2)
        types = [type(e) for e in events]
        assert types.index(PersonLeftEvent) < types.index(SessionEndedEvent)

    def test_lost_identity_before_re_identified(self) -> None:
        """PersonLostIdentityEvent must precede the new PersonIdentifiedEvent."""
        mgr = _manager(require_confirmed_presence=False)

        # Start session with person A identified
        id_state_a = _mk_id_state(1, IdentityState.IDENTIFIED, person_id="alice")
        f1 = _mk_frame(frame_id=1, primary_track_id=1, identity_states={1: id_state_a})
        mgr.process_frame_result(f1)

        # Identity becomes uncertain
        id_state_u = _mk_id_state(1, IdentityState.UNCERTAIN)
        f2 = _mk_frame(frame_id=2, primary_track_id=1, identity_states={1: id_state_u})
        mgr.process_frame_result(f2)

        # Identity resolves to person B
        id_state_b = _mk_id_state(1, IdentityState.IDENTIFIED, person_id="bob")
        f3 = _mk_frame(frame_id=3, primary_track_id=1, identity_states={1: id_state_b})
        events = mgr.process_frame_result(f3)
        types = [type(e) for e in events]
        assert PersonLostIdentityEvent not in types  # Was already emitted in f2
        assert PersonIdentifiedEvent in types


# ---------------------------------------------------------------------------
# PersonIdentifiedEvent deduplication
# ---------------------------------------------------------------------------


class TestPersonIdentifiedDedup:
    def test_identified_emitted_once_per_stable_identity(self) -> None:
        mgr = _manager(require_confirmed_presence=False)
        id_state = _mk_id_state(1, IdentityState.IDENTIFIED, person_id="alice")

        all_events: list[BaseEvent] = []
        for frame_id in range(1, 6):
            f = _mk_frame(
                frame_id=frame_id,
                primary_track_id=1,
                identity_states={1: id_state},
            )
            all_events.extend(mgr.process_frame_result(f))

        identified = [e for e in all_events if isinstance(e, PersonIdentifiedEvent)]
        assert len(identified) == 1

    def test_identified_on_session_start_if_already_identified(self) -> None:
        mgr = _manager(require_confirmed_presence=False)
        id_state = _mk_id_state(1, IdentityState.IDENTIFIED, person_id="alice")
        frame = _mk_frame(frame_id=1, primary_track_id=1, identity_states={1: id_state})
        events = mgr.process_frame_result(frame)
        identified = [e for e in events if isinstance(e, PersonIdentifiedEvent)]
        assert len(identified) == 1
        assert identified[0].person_id == "alice"

    def test_identified_emitted_after_collecting_phase(self) -> None:
        mgr = _manager(require_confirmed_presence=False)

        # Start session in COLLECTING state
        f1 = _mk_frame(
            frame_id=1,
            primary_track_id=1,
            identity_states={1: _mk_id_state(1, IdentityState.COLLECTING)},
        )
        e1 = mgr.process_frame_result(f1)
        assert not any(isinstance(e, PersonIdentifiedEvent) for e in e1)

        # Now identity resolves
        f2 = _mk_frame(
            frame_id=2,
            primary_track_id=1,
            identity_states={1: _mk_id_state(1, IdentityState.IDENTIFIED, person_id="carol")},
        )
        e2 = mgr.process_frame_result(f2)
        identified = [e for e in e2 if isinstance(e, PersonIdentifiedEvent)]
        assert len(identified) == 1
        assert identified[0].person_id == "carol"


# ---------------------------------------------------------------------------
# PersonLostIdentityEvent
# ---------------------------------------------------------------------------


class TestPersonLostIdentityEvent:
    def test_lost_emitted_when_identity_degrades(self) -> None:
        mgr = _manager(require_confirmed_presence=False)

        f1 = _mk_frame(
            frame_id=1,
            primary_track_id=1,
            identity_states={1: _mk_id_state(1, IdentityState.IDENTIFIED, person_id="alice")},
        )
        mgr.process_frame_result(f1)

        f2 = _mk_frame(
            frame_id=2,
            primary_track_id=1,
            identity_states={1: _mk_id_state(1, IdentityState.UNCERTAIN)},
        )
        events = mgr.process_frame_result(f2)
        lost = [e for e in events if isinstance(e, PersonLostIdentityEvent)]
        assert len(lost) == 1
        assert lost[0].previous_person_id == "alice"

    def test_lost_not_emitted_without_prior_identification(self) -> None:
        """No LostIdentity if identity was never previously stable."""
        mgr = _manager(require_confirmed_presence=False)

        f1 = _mk_frame(
            frame_id=1,
            primary_track_id=1,
            identity_states={1: _mk_id_state(1, IdentityState.COLLECTING)},
        )
        mgr.process_frame_result(f1)

        f2 = _mk_frame(
            frame_id=2,
            primary_track_id=1,
            identity_states={1: _mk_id_state(1, IdentityState.UNCERTAIN)},
        )
        events = mgr.process_frame_result(f2)
        assert not any(isinstance(e, PersonLostIdentityEvent) for e in events)

    def test_lost_resets_dedup_guard(self) -> None:
        """After a LostIdentity event, re-identification must emit a new IdentifiedEvent."""
        mgr = _manager(require_confirmed_presence=False)

        f1 = _mk_frame(
            frame_id=1,
            primary_track_id=1,
            identity_states={1: _mk_id_state(1, IdentityState.IDENTIFIED, person_id="alice")},
        )
        mgr.process_frame_result(f1)

        # Loses identity
        f2 = _mk_frame(
            frame_id=2,
            primary_track_id=1,
            identity_states={1: _mk_id_state(1, IdentityState.UNCERTAIN)},
        )
        mgr.process_frame_result(f2)

        # Re-identified (same person)
        f3 = _mk_frame(
            frame_id=3,
            primary_track_id=1,
            identity_states={1: _mk_id_state(1, IdentityState.IDENTIFIED, person_id="alice")},
        )
        events = mgr.process_frame_result(f3)
        identified = [e for e in events if isinstance(e, PersonIdentifiedEvent)]
        assert len(identified) == 1


# ---------------------------------------------------------------------------
# Session end
# ---------------------------------------------------------------------------


class TestSessionEnd:
    def test_session_ended_on_person_left(self) -> None:
        mgr = _manager(require_confirmed_presence=False)
        f1 = _mk_frame(frame_id=1, primary_track_id=1)
        mgr.process_frame_result(f1)

        f2 = _mk_frame(
            frame_id=2,
            primary_track_id=None,
            presence_transitions=[_mk_transition(PresenceEvent.PERSON_LEFT, track_id=1)],
        )
        events = mgr.process_frame_result(f2)
        assert any(isinstance(e, SessionEndedEvent) for e in events)

    def test_session_ended_exactly_once(self) -> None:
        mgr = _manager(require_confirmed_presence=False)
        f1 = _mk_frame(frame_id=1, primary_track_id=1)
        mgr.process_frame_result(f1)

        f2 = _mk_frame(
            frame_id=2,
            presence_transitions=[_mk_transition(PresenceEvent.PERSON_LEFT, track_id=1)],
        )
        f3 = _mk_frame(frame_id=3)
        e2 = mgr.process_frame_result(f2)
        e3 = mgr.process_frame_result(f3)
        ended_count = sum(1 for e in e2 + e3 if isinstance(e, SessionEndedEvent))
        assert ended_count == 1

    def test_session_ended_carries_session_id(self) -> None:
        mgr = _manager(require_confirmed_presence=False)
        f1 = _mk_frame(frame_id=1, primary_track_id=1)
        e1 = mgr.process_frame_result(f1)
        started = next(e for e in e1 if isinstance(e, SessionStartedEvent))

        f2 = _mk_frame(
            frame_id=2,
            presence_transitions=[_mk_transition(PresenceEvent.PERSON_LEFT, track_id=1)],
        )
        e2 = mgr.process_frame_result(f2)
        ended = next(e for e in e2 if isinstance(e, SessionEndedEvent))
        assert ended.session_id == started.session_id

    def test_no_session_ended_without_active_session(self) -> None:
        mgr = _manager(require_confirmed_presence=False)
        f = _mk_frame(
            frame_id=1,
            presence_transitions=[_mk_transition(PresenceEvent.PERSON_LEFT, track_id=99)],
        )
        events = mgr.process_frame_result(f)
        assert not any(isinstance(e, SessionEndedEvent) for e in events)

    def test_new_session_starts_after_previous_ended(self) -> None:
        mgr = _manager(require_confirmed_presence=False)

        # First session
        f1 = _mk_frame(frame_id=1, primary_track_id=1)
        mgr.process_frame_result(f1)
        f2 = _mk_frame(
            frame_id=2,
            presence_transitions=[_mk_transition(PresenceEvent.PERSON_LEFT, track_id=1)],
        )
        mgr.process_frame_result(f2)

        # Second person arrives
        f3 = _mk_frame(
            frame_id=3,
            primary_track_id=2,
            presence_transitions=[_mk_transition(PresenceEvent.PERSON_ENTERED, track_id=2)],
        )
        e3 = mgr.process_frame_result(f3)
        assert any(isinstance(e, SessionStartedEvent) for e in e3)


# ---------------------------------------------------------------------------
# Target switching — no fake PERSON_LEFT
# ---------------------------------------------------------------------------


class TestTargetSwitching:
    def test_ranking_change_does_not_end_session(self) -> None:
        """If old primary track is still present, ranking change must not end the session."""
        mgr = _manager(require_confirmed_presence=False)

        # Start session on track 1
        f1 = _mk_frame(frame_id=1, primary_track_id=1)
        e1 = mgr.process_frame_result(f1)
        assert any(isinstance(e, SessionStartedEvent) for e in e1)

        # Target selector now ranks track 2 higher — but track 1 is NOT in left_transitions
        f2 = _mk_frame(frame_id=2, primary_track_id=2)
        e2 = mgr.process_frame_result(f2)
        # Session must NOT end
        assert not any(isinstance(e, SessionEndedEvent) for e in e2)
        # Session must NOT start a new one
        assert not any(isinstance(e, SessionStartedEvent) for e in e2)

    def test_no_fake_left_event_from_ranking_change(self) -> None:
        mgr = _manager(require_confirmed_presence=False)

        f1 = _mk_frame(frame_id=1, primary_track_id=1)
        mgr.process_frame_result(f1)

        f2 = _mk_frame(frame_id=2, primary_track_id=2)
        events = mgr.process_frame_result(f2)
        assert not any(isinstance(e, PersonLeftEvent) for e in events)

    def test_session_ends_only_on_real_left_transition(self) -> None:
        mgr = _manager(require_confirmed_presence=False)

        f1 = _mk_frame(frame_id=1, primary_track_id=1)
        mgr.process_frame_result(f1)

        # Primary changes but no PERSON_LEFT for track 1 → session retained
        f2 = _mk_frame(frame_id=2, primary_track_id=2)
        e2 = mgr.process_frame_result(f2)
        assert not any(isinstance(e, SessionEndedEvent) for e in e2)

        # Now track 1 actually leaves
        f3 = _mk_frame(
            frame_id=3,
            primary_track_id=2,
            presence_transitions=[_mk_transition(PresenceEvent.PERSON_LEFT, track_id=1)],
        )
        e3 = mgr.process_frame_result(f3)
        assert any(isinstance(e, SessionEndedEvent) for e in e3)


# ---------------------------------------------------------------------------
# Multiple people — single primary session
# ---------------------------------------------------------------------------


class TestMultiplePeople:
    def test_only_primary_gets_session(self) -> None:
        mgr = _manager(require_confirmed_presence=False)

        id_states = {
            1: _mk_id_state(1, IdentityState.NEW),
            2: _mk_id_state(2, IdentityState.NEW),
        }
        frame = _mk_frame(frame_id=1, primary_track_id=1, identity_states=id_states)
        events = mgr.process_frame_result(frame)
        sessions_started = [e for e in events if isinstance(e, SessionStartedEvent)]
        assert len(sessions_started) == 1
        assert sessions_started[0].track_id == 1

    def test_background_person_does_not_emit_events(self) -> None:
        """Background face entering and leaving produces no domain events."""
        mgr = _manager(require_confirmed_presence=True)

        # Primary track 1 and background track 2 enter
        f1 = _mk_frame(
            frame_id=1,
            primary_track_id=1,
            presence_transitions=[
                _mk_transition(PresenceEvent.PERSON_ENTERED, track_id=1),
                _mk_transition(PresenceEvent.PERSON_ENTERED, track_id=2),
            ],
        )
        e1 = mgr.process_frame_result(f1)
        # Only track 1 gets events
        assert all(getattr(e, "track_id", None) == 1 for e in e1)

        # Background track 2 leaves
        f2 = _mk_frame(
            frame_id=2,
            primary_track_id=1,
            presence_transitions=[_mk_transition(PresenceEvent.PERSON_LEFT, track_id=2)],
        )
        e2 = mgr.process_frame_result(f2)
        # No events emitted for background departure
        assert len(e2) == 0

    def test_promoted_background_person_starts_session(self) -> None:
        """When old primary leaves and background person becomes primary, a new session starts."""
        mgr = _manager(require_confirmed_presence=True)

        # Track 1 (primary) and Track 2 (background) enter
        f1 = _mk_frame(
            frame_id=1,
            primary_track_id=1,
            presence_transitions=[
                _mk_transition(PresenceEvent.PERSON_ENTERED, track_id=1),
                _mk_transition(PresenceEvent.PERSON_ENTERED, track_id=2),
            ],
        )
        mgr.process_frame_result(f1)

        # Track 1 leaves, Track 2 becomes primary
        f2 = _mk_frame(
            frame_id=2,
            primary_track_id=2,
            presence_transitions=[_mk_transition(PresenceEvent.PERSON_LEFT, track_id=1)],
        )
        events = mgr.process_frame_result(f2)
        # Old session ends (track 1)
        left = next(e for e in events if isinstance(e, PersonLeftEvent))
        assert left.track_id == 1
        ended = next(e for e in events if isinstance(e, SessionEndedEvent))
        assert ended.track_id == 1

        # New session starts (track 2)
        entered = next(e for e in events if isinstance(e, PersonEnteredEvent))
        assert entered.track_id == 2
        started = next(e for e in events if isinstance(e, SessionStartedEvent))
        assert started.track_id == 2


# ---------------------------------------------------------------------------
# EventBus integration
# ---------------------------------------------------------------------------


class TestSessionManagerWithEventBus:
    def test_events_published_to_bus(self) -> None:
        bus = EventBus()
        received: list[BaseEvent] = []
        bus.subscribe(SessionStartedEvent, received.append)

        mgr = SessionManager(event_bus=bus, config=SessionConfig(require_confirmed_presence=False))
        frame = _mk_frame(frame_id=1, primary_track_id=1)
        mgr.process_frame_result(frame)

        assert len(received) == 1
        assert isinstance(received[0], SessionStartedEvent)

    def test_all_events_published_and_returned(self) -> None:
        bus = EventBus()
        published: list[BaseEvent] = []
        for event_type in [
            PersonEnteredEvent,
            SessionStartedEvent,
            PersonIdentifiedEvent,
        ]:
            bus.subscribe(event_type, published.append)

        mgr = SessionManager(event_bus=bus, config=SessionConfig(require_confirmed_presence=True))
        id_state = _mk_id_state(1, IdentityState.IDENTIFIED, person_id="alice")
        frame = _mk_frame(
            frame_id=1,
            primary_track_id=1,
            identity_states={1: id_state},
            presence_transitions=[_mk_transition(PresenceEvent.PERSON_ENTERED, track_id=1)],
        )
        returned = mgr.process_frame_result(frame)
        assert len(returned) == len(published)

    def test_no_bus_no_publish_error(self) -> None:
        mgr = SessionManager(event_bus=None, config=SessionConfig(require_confirmed_presence=False))
        frame = _mk_frame(frame_id=1, primary_track_id=1)
        events = mgr.process_frame_result(frame)
        assert any(isinstance(e, SessionStartedEvent) for e in events)

    def test_person_invalidation_emits_one_loss_and_detaches_deleted_id(self) -> None:
        person_id = "person-1"
        bus = EventBus()
        lost_events: list[PersonLostIdentityEvent] = []
        bus.subscribe(PersonLostIdentityEvent, lost_events.append)
        mgr = SessionManager(event_bus=bus, config=SessionConfig(require_confirmed_presence=False))
        mgr.process_frame_result(
            _mk_frame(
                frame_id=1,
                primary_track_id=7,
                identity_states={
                    7: _mk_id_state(
                        7,
                        IdentityState.IDENTIFIED,
                        person_id=person_id,
                        display_name="Riley",
                    )
                },
            )
        )
        session = mgr.active_session
        assert session is not None
        assert session.last_emitted_person_id == person_id

        lost = mgr.invalidate_person_identity(person_id, deleted=True, timestamp=2.0)
        assert lost is not None
        assert len(lost_events) == 1
        assert lost_events[0].previous_person_id == person_id
        assert session.person_id is None
        assert session.display_name == "Riley"
        assert session.last_emitted_person_id is None

        next_events = mgr.process_frame_result(
            _mk_frame(
                frame_id=2,
                primary_track_id=7,
                identity_states={7: _mk_id_state(7, IdentityState.UNREGISTERED)},
            )
        )
        assert not any(isinstance(event, PersonLostIdentityEvent) for event in next_events)

        ended = mgr.process_frame_result(
            _mk_frame(
                frame_id=3,
                presence_transitions=[
                    _mk_transition(PresenceEvent.PERSON_LEFT, track_id=7, timestamp=3.0)
                ],
            )
        )
        end_event = next(event for event in ended if isinstance(event, SessionEndedEvent))
        assert end_event.final_person_id is None
        assert end_event.final_display_name == "Riley"

    def test_biometric_reset_preserves_session_history_link(self) -> None:
        person_id = "person-2"
        mgr = SessionManager(config=SessionConfig(require_confirmed_presence=False))
        mgr.process_frame_result(
            _mk_frame(
                frame_id=1,
                primary_track_id=8,
                identity_states={
                    8: _mk_id_state(
                        8,
                        IdentityState.IDENTIFIED,
                        person_id=person_id,
                        display_name="Morgan",
                    )
                },
            )
        )

        mgr.invalidate_person_identity(person_id, deleted=False, timestamp=2.0)
        assert mgr.active_session is not None
        assert mgr.active_session.person_id == person_id
        assert mgr.active_session.display_name == "Morgan"


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


class TestSessionManagerReset:
    def test_reset_clears_active_session(self) -> None:
        mgr = _manager(require_confirmed_presence=False)
        f1 = _mk_frame(frame_id=1, primary_track_id=1)
        mgr.process_frame_result(f1)
        assert mgr.active_session is not None

        mgr.reset()
        assert mgr.active_session is None

    def test_reset_allows_fresh_session_start(self) -> None:
        mgr = _manager(require_confirmed_presence=False)
        f1 = _mk_frame(frame_id=1, primary_track_id=1)
        mgr.process_frame_result(f1)
        mgr.reset()

        f2 = _mk_frame(frame_id=2, primary_track_id=1)
        events = mgr.process_frame_result(f2)
        assert any(isinstance(e, SessionStartedEvent) for e in events)

    def test_reset_clears_present_tracks(self) -> None:
        mgr = _manager(require_confirmed_presence=True)
        frame = _mk_frame(
            frame_id=1,
            primary_track_id=5,
            presence_transitions=[_mk_transition(PresenceEvent.PERSON_ENTERED, track_id=5)],
        )
        mgr.process_frame_result(frame)
        mgr.reset()

        # Same track enters again — should be treated as new
        frame2 = _mk_frame(
            frame_id=2,
            primary_track_id=5,
            presence_transitions=[_mk_transition(PresenceEvent.PERSON_ENTERED, track_id=5)],
        )
        events = mgr.process_frame_result(frame2)
        assert any(isinstance(e, PersonEnteredEvent) for e in events)


# ---------------------------------------------------------------------------
# Active session property
# ---------------------------------------------------------------------------


class TestActiveSessionProperty:
    def test_active_session_none_initially(self) -> None:
        mgr = _manager()
        assert mgr.active_session is None

    def test_active_session_set_after_start(self) -> None:
        mgr = _manager(require_confirmed_presence=False)
        frame = _mk_frame(frame_id=1, primary_track_id=42)
        mgr.process_frame_result(frame)
        assert mgr.active_session is not None
        assert mgr.active_session.track_id == 42

    def test_active_session_cleared_after_end(self) -> None:
        mgr = _manager(require_confirmed_presence=False)
        f1 = _mk_frame(frame_id=1, primary_track_id=1)
        mgr.process_frame_result(f1)
        f2 = _mk_frame(
            frame_id=2,
            presence_transitions=[_mk_transition(PresenceEvent.PERSON_LEFT, track_id=1)],
        )
        mgr.process_frame_result(f2)
        assert mgr.active_session is None

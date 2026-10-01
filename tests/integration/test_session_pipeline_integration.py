"""Integration tests for SessionManager wired to the TwoRatePerceptionPipeline.

These tests feed synthetic frames through the real pipeline and verify that
``SessionManager`` produces correct domain events from the consolidated
``TwoRateFrameResult`` output.

No camera or ONNX models are required — the pipeline is replaced by a
lightweight stub that hands pre-built ``TwoRateFrameResult`` objects to the
``SessionManager``.  This is intentional: the SessionManager sits *above* the
vision pipeline and must not depend on hardware availability.

Integration verifications
--------------------------
1. A realistic multi-frame presence → session → identification → departure
   sequence produces exactly the right events in the right order.
2. The ``EventBus`` wiring delivers the same events to subscribers.
3. Multiple people are handled correctly (only primary gets a session).
4. Re-identification after identity loss produces the correct event sequence.
5. ``SessionManager.reset()`` in the middle of a pipeline sequence works.
"""

from __future__ import annotations

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
# Helpers
# ---------------------------------------------------------------------------


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


def _mk_candidate(track_id: int, score: float = 1.0) -> TargetCandidate:
    return TargetCandidate(
        track_id=track_id,
        score=score,
        centrality_score=score,
        size_score=score,
        identity_state_score=score,
        dwell_score=score,
        is_identified=False,
        identity_state=IdentityState.NEW,
        person_id=None,
        display_name=None,
        bbox=(0.0, 0.0, 100.0, 100.0),
        center=(50.0, 50.0),
    )


def _mk_target(primary_track_id: int | None, frame_id: int = 1) -> TargetSelectionResult:
    primary = _mk_candidate(primary_track_id) if primary_track_id is not None else None
    return TargetSelectionResult(
        primary_target=primary,
        candidates=[primary] if primary else [],
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


def _frame(
    frame_id: int,
    primary_track_id: int | None = None,
    identity_states: dict[int, TrackIdentityState] | None = None,
    presence_transitions: list[PresenceTransition] | None = None,
) -> TwoRateFrameResult:
    return TwoRateFrameResult(
        frame_id=frame_id,
        timestamp=float(frame_id) * 0.033,
        detections=[],
        tracks=[],
        quality_results={},
        identity_states=identity_states or {},
        target_result=_mk_target(primary_track_id, frame_id),
        presence_transitions=presence_transitions or [],
        completed_recognitions=[],
        ambiguous_crossing_track_ids=set(),
        pending_worker_tasks=0,
        frame_latency_ms=1.0,
        total_pipeline_latency_ms=1.0,
    )


def _all_events_of(events: list[BaseEvent], cls: type) -> list[BaseEvent]:
    return [e for e in events if isinstance(e, cls)]


# ---------------------------------------------------------------------------
# Integration test 1: Full realistic lifecycle
# ---------------------------------------------------------------------------


class TestFullLifecycle:
    """A simulated 20-frame interaction with debounce, identification, and departure."""

    def _run_scenario(
        self,
        require_confirmed_presence: bool = True,
    ) -> tuple[SessionManager, list[BaseEvent]]:
        """Run a scripted scenario and return manager + all emitted events."""
        cfg = SessionConfig(require_confirmed_presence=require_confirmed_presence)
        mgr = SessionManager(config=cfg)
        all_events: list[BaseEvent] = []

        # Frames 1-2: Track 1 detected but debounce in progress (no ENTERED yet)
        for fid in [1, 2]:
            events = mgr.process_frame_result(_frame(fid, primary_track_id=1))
            all_events.extend(events)

        # Frame 3: PERSON_ENTERED fires (debounce satisfied)
        events = mgr.process_frame_result(
            _frame(
                3,
                primary_track_id=1,
                presence_transitions=[_mk_transition(PresenceEvent.PERSON_ENTERED, 1, frame_id=3)],
            )
        )
        all_events.extend(events)

        # Frames 4-6: Present but still COLLECTING
        for fid in [4, 5, 6]:
            events = mgr.process_frame_result(
                _frame(
                    fid,
                    primary_track_id=1,
                    identity_states={1: _mk_id_state(1, IdentityState.COLLECTING)},
                )
            )
            all_events.extend(events)

        # Frame 7: Identity resolves to IDENTIFIED
        events = mgr.process_frame_result(
            _frame(
                7,
                primary_track_id=1,
                identity_states={1: _mk_id_state(1, IdentityState.IDENTIFIED, person_id="alice")},
            )
        )
        all_events.extend(events)

        # Frames 8-9: Still identified (no duplicate events expected)
        for fid in [8, 9]:
            events = mgr.process_frame_result(
                _frame(
                    fid,
                    primary_track_id=1,
                    identity_states={
                        1: _mk_id_state(1, IdentityState.IDENTIFIED, person_id="alice")
                    },
                )
            )
            all_events.extend(events)

        # Frame 10: Person leaves
        events = mgr.process_frame_result(
            _frame(
                10,
                primary_track_id=None,
                presence_transitions=[_mk_transition(PresenceEvent.PERSON_LEFT, 1, frame_id=10)],
            )
        )
        all_events.extend(events)

        return mgr, all_events

    def test_full_lifecycle_event_types_present(self) -> None:
        _, events = self._run_scenario()
        types = {type(e) for e in events}
        assert PersonEnteredEvent in types
        assert SessionStartedEvent in types
        assert PersonIdentifiedEvent in types
        assert PersonLeftEvent in types
        assert SessionEndedEvent in types
        # No spurious events
        assert PersonLostIdentityEvent not in types

    def test_exactly_one_session_started(self) -> None:
        _, events = self._run_scenario()
        assert len(_all_events_of(events, SessionStartedEvent)) == 1

    def test_exactly_one_session_ended(self) -> None:
        _, events = self._run_scenario()
        assert len(_all_events_of(events, SessionEndedEvent)) == 1

    def test_session_ids_match(self) -> None:
        _, events = self._run_scenario()
        started = next(e for e in events if isinstance(e, SessionStartedEvent))
        ended = next(e for e in events if isinstance(e, SessionEndedEvent))
        assert started.session_id == ended.session_id

    def test_exactly_one_identified_event(self) -> None:
        _, events = self._run_scenario()
        identified = _all_events_of(events, PersonIdentifiedEvent)
        assert len(identified) == 1
        assert identified[0].person_id == "alice"  # type: ignore[union-attr]

    def test_event_ordering_entered_before_started(self) -> None:
        _, events = self._run_scenario()
        types = [type(e) for e in events]
        assert types.index(PersonEnteredEvent) < types.index(SessionStartedEvent)

    def test_event_ordering_started_before_identified(self) -> None:
        _, events = self._run_scenario()
        types = [type(e) for e in events]
        assert types.index(SessionStartedEvent) < types.index(PersonIdentifiedEvent)

    def test_event_ordering_left_before_ended(self) -> None:
        _, events = self._run_scenario()
        types = [type(e) for e in events]
        assert types.index(PersonLeftEvent) < types.index(SessionEndedEvent)

    def test_manager_clean_after_departure(self) -> None:
        mgr, _ = self._run_scenario()
        assert mgr.active_session is None


# ---------------------------------------------------------------------------
# Integration test 2: EventBus wiring
# ---------------------------------------------------------------------------


class TestEventBusWiring:
    def test_all_published_match_returned(self) -> None:
        bus = EventBus()
        published: list[BaseEvent] = []
        for cls in [
            PersonEnteredEvent,
            SessionStartedEvent,
            PersonIdentifiedEvent,
            PersonLostIdentityEvent,
            PersonLeftEvent,
            SessionEndedEvent,
        ]:
            bus.subscribe(cls, published.append)

        mgr = SessionManager(
            event_bus=bus,
            config=SessionConfig(require_confirmed_presence=True),
        )
        all_returned: list[BaseEvent] = []

        # Frame 1: Enter
        f1 = _frame(
            1,
            primary_track_id=1,
            presence_transitions=[_mk_transition(PresenceEvent.PERSON_ENTERED, 1)],
        )
        all_returned.extend(mgr.process_frame_result(f1))

        # Frame 2: Identify
        f2 = _frame(
            2,
            primary_track_id=1,
            identity_states={1: _mk_id_state(1, IdentityState.IDENTIFIED, person_id="p1")},
        )
        all_returned.extend(mgr.process_frame_result(f2))

        # Frame 3: Leave
        f3 = _frame(
            3,
            primary_track_id=None,
            presence_transitions=[_mk_transition(PresenceEvent.PERSON_LEFT, 1)],
        )
        all_returned.extend(mgr.process_frame_result(f3))

        assert len(published) == len(all_returned)
        for r, p in zip(all_returned, published, strict=True):
            assert r.event_id == p.event_id

    def test_bus_subscriber_exception_does_not_abort_processing(self) -> None:
        bus = EventBus()
        bus.subscribe(SessionStartedEvent, lambda _: (_ for _ in ()).throw(RuntimeError()))  # type: ignore[misc]
        received: list[BaseEvent] = []
        bus.subscribe(PersonIdentifiedEvent, received.append)

        mgr = SessionManager(
            event_bus=bus,
            config=SessionConfig(require_confirmed_presence=True),
        )

        f1 = _frame(
            1,
            primary_track_id=1,
            identity_states={1: _mk_id_state(1, IdentityState.IDENTIFIED, person_id="p1")},
            presence_transitions=[_mk_transition(PresenceEvent.PERSON_ENTERED, 1)],
        )
        # Should NOT raise despite bad subscriber
        returned = mgr.process_frame_result(f1)
        assert any(isinstance(e, PersonIdentifiedEvent) for e in returned)
        assert len(received) == 1


# ---------------------------------------------------------------------------
# Integration test 3: Multiple people, single session
# ---------------------------------------------------------------------------


class TestMultiplePeopleIntegration:
    def test_two_people_only_primary_gets_session_and_events(self) -> None:
        mgr = SessionManager(config=SessionConfig(require_confirmed_presence=True))
        all_events: list[BaseEvent] = []

        # Both persons enter on frame 1, but track 1 is primary
        f1 = _frame(
            1,
            primary_track_id=1,
            identity_states={
                1: _mk_id_state(1, IdentityState.COLLECTING),
                2: _mk_id_state(2, IdentityState.COLLECTING),
            },
            presence_transitions=[
                _mk_transition(PresenceEvent.PERSON_ENTERED, 1),
                _mk_transition(PresenceEvent.PERSON_ENTERED, 2),
            ],
        )
        all_events.extend(mgr.process_frame_result(f1))

        # Only track 1 gets PersonEnteredEvent and SessionStartedEvent
        entered = _all_events_of(all_events, PersonEnteredEvent)
        assert len(entered) == 1
        assert entered[0].track_id == 1  # type: ignore[union-attr]

        sessions = _all_events_of(all_events, SessionStartedEvent)
        assert len(sessions) == 1
        assert sessions[0].track_id == 1  # type: ignore[union-attr]

    def test_background_person_left_produces_no_events(self) -> None:
        mgr = SessionManager(config=SessionConfig(require_confirmed_presence=True))

        # Both enter, track 1 is primary
        f1 = _frame(
            1,
            primary_track_id=1,
            presence_transitions=[
                _mk_transition(PresenceEvent.PERSON_ENTERED, 1),
                _mk_transition(PresenceEvent.PERSON_ENTERED, 2),
            ],
        )
        mgr.process_frame_result(f1)

        # Background person (track 2) leaves
        f2 = _frame(
            2,
            primary_track_id=1,
            presence_transitions=[_mk_transition(PresenceEvent.PERSON_LEFT, 2)],
        )
        events = mgr.process_frame_result(f2)

        # No domain events emitted for background person departure
        assert len(events) == 0
        assert mgr.active_session is not None
        assert mgr.active_session.track_id == 1

    def test_background_person_promoted_after_primary_leaves(self) -> None:
        mgr = SessionManager(config=SessionConfig(require_confirmed_presence=True))

        # Track 1 (primary) and Track 2 (background) enter
        f1 = _frame(
            1,
            primary_track_id=1,
            presence_transitions=[
                _mk_transition(PresenceEvent.PERSON_ENTERED, 1),
                _mk_transition(PresenceEvent.PERSON_ENTERED, 2),
            ],
        )
        mgr.process_frame_result(f1)

        # Track 1 departs; Track 2 (already present) becomes primary
        f2 = _frame(
            2,
            primary_track_id=2,
            identity_states={2: _mk_id_state(2, IdentityState.IDENTIFIED, person_id="bob")},
            presence_transitions=[_mk_transition(PresenceEvent.PERSON_LEFT, 1)],
        )
        events = mgr.process_frame_result(f2)

        # Old session ends: PersonLeftEvent (1) -> SessionEndedEvent (1)
        # New session starts: PersonEnteredEvent (2) -> SessionStartedEvent (2)
        #                     -> PersonIdentifiedEvent (2)
        types = [type(e) for e in events]
        assert types == [
            PersonLeftEvent,
            SessionEndedEvent,
            PersonEnteredEvent,
            SessionStartedEvent,
            PersonIdentifiedEvent,
        ]
        assert mgr.active_session is not None
        assert mgr.active_session.track_id == 2
        assert mgr.active_session.person_id == "bob"


# ---------------------------------------------------------------------------
# Integration test 4: Identity loss and re-identification
# ---------------------------------------------------------------------------


class TestIdentityLossAndReidentification:
    def test_identity_loss_then_reidentification_sequence(self) -> None:
        mgr = SessionManager(config=SessionConfig(require_confirmed_presence=False))
        all_events: list[BaseEvent] = []

        # Frame 1: Start session, already identified
        f1 = _frame(
            1,
            primary_track_id=1,
            identity_states={1: _mk_id_state(1, IdentityState.IDENTIFIED, person_id="alice")},
        )
        all_events.extend(mgr.process_frame_result(f1))

        # Frame 2: Identity degrades to UNCERTAIN
        f2 = _frame(
            2,
            primary_track_id=1,
            identity_states={1: _mk_id_state(1, IdentityState.UNCERTAIN)},
        )
        all_events.extend(mgr.process_frame_result(f2))

        # Frame 3: Re-identified as same person
        f3 = _frame(
            3,
            primary_track_id=1,
            identity_states={1: _mk_id_state(1, IdentityState.IDENTIFIED, person_id="alice")},
        )
        all_events.extend(mgr.process_frame_result(f3))

        lost = _all_events_of(all_events, PersonLostIdentityEvent)
        identified = _all_events_of(all_events, PersonIdentifiedEvent)

        assert len(lost) == 1
        assert lost[0].previous_person_id == "alice"  # type: ignore[union-attr]
        # 2 identified events: one at session start + one after re-identification
        assert len(identified) == 2
        assert all(e.person_id == "alice" for e in identified)  # type: ignore[union-attr]

    def test_identity_change_different_person(self) -> None:
        """Track flips from alice to bob — LostIdentity + new IdentifiedEvent."""
        mgr = SessionManager(config=SessionConfig(require_confirmed_presence=False))
        all_events: list[BaseEvent] = []

        f1 = _frame(
            1,
            primary_track_id=1,
            identity_states={1: _mk_id_state(1, IdentityState.IDENTIFIED, person_id="alice")},
        )
        all_events.extend(mgr.process_frame_result(f1))

        f2 = _frame(
            2,
            primary_track_id=1,
            identity_states={1: _mk_id_state(1, IdentityState.UNCERTAIN)},
        )
        all_events.extend(mgr.process_frame_result(f2))

        f3 = _frame(
            3,
            primary_track_id=1,
            identity_states={1: _mk_id_state(1, IdentityState.IDENTIFIED, person_id="bob")},
        )
        all_events.extend(mgr.process_frame_result(f3))

        lost = _all_events_of(all_events, PersonLostIdentityEvent)
        identified = _all_events_of(all_events, PersonIdentifiedEvent)

        assert len(lost) == 1
        assert lost[0].previous_person_id == "alice"  # type: ignore[union-attr]
        assert len(identified) == 2
        person_ids = [e.person_id for e in identified]  # type: ignore[union-attr]
        assert person_ids == ["alice", "bob"]


# ---------------------------------------------------------------------------
# Integration test 5: Reset mid-scenario
# ---------------------------------------------------------------------------


class TestResetMidScenario:
    def test_reset_mid_session_then_new_lifecycle(self) -> None:
        mgr = SessionManager(config=SessionConfig(require_confirmed_presence=False))

        # Start a session
        f1 = _frame(1, primary_track_id=1)
        mgr.process_frame_result(f1)
        assert mgr.active_session is not None

        # Operator resets (e.g. robot re-initialized)
        mgr.reset()
        assert mgr.active_session is None

        # Fresh person arrives
        f2 = _frame(
            2,
            primary_track_id=2,
            presence_transitions=[_mk_transition(PresenceEvent.PERSON_ENTERED, 2)],
        )
        events = mgr.process_frame_result(f2)
        assert any(isinstance(e, PersonEnteredEvent) for e in events)
        assert any(isinstance(e, SessionStartedEvent) for e in events)
        assert mgr.active_session is not None
        assert mgr.active_session.track_id == 2

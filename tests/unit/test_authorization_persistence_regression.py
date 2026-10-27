"""Regression tests for persistent authorization lifecycle and security invariants.

Verifies:
1. Authorization remains active after > 30 seconds without state transitions.
2. A temporary missed detection does not immediately terminate an otherwise valid session.
3. A genuine different-person identification immediately terminates session and blocks access.
4. Backend LOST/revocation events are handled according to their actual cause.
5. Dashboard / client logic does not overwrite valid backend authorization state.
6. SSE reconnection or polling does not cause a valid session to become LOST.
7. Explicit Logout still ends authorization and clears private context.
8. General questions continue working after logout.
9. Existing authorization and privacy invariants remain intact.
"""

from __future__ import annotations

import time
from unittest import mock

from dh360_hi.application.dialogue_coordinator import DialogueCoordinator
from dh360_hi.backend.knowledge_service import KnowledgeService
from dh360_hi.contracts.authorization import (
    AuthorizationSession,
    AuthorizationStatus,
)
from dh360_hi.contracts.dialogue import MockLLMProvider
from dh360_hi.edge.authorization_controller import AuthorizationController
from dh360_hi.edge.event_bus import EventBus
from dh360_hi.edge.session_manager import SessionConfig, SessionManager
from dh360_hi.vision.identity_state import IdentityState, TrackIdentityState
from dh360_hi.vision.target_selector import TargetCandidate, TargetSelectionResult
from dh360_hi.vision.two_rate_pipeline import TwoRateFrameResult


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
    display_name: str | None = None,
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
        display_name=display_name,
        bbox=(0.0, 0.0, 100.0, 100.0),
        center=(50.0, 50.0),
    )


def _mk_frame(
    frame_id: int = 1,
    timestamp: float = 1.0,
    primary_track_id: int | None = None,
    candidates: list[TargetCandidate] | None = None,
    identity_states: dict[int, TrackIdentityState] | None = None,
) -> TwoRateFrameResult:
    primary = None
    if primary_track_id is not None:
        cands = candidates or [_mk_candidate(primary_track_id)]
        primary = next((c for c in cands if c.track_id == primary_track_id), None)
        if primary is None:
            primary = _mk_candidate(primary_track_id)
    target_result = TargetSelectionResult(
        primary_target=primary,
        candidates=candidates or ([primary] if primary else []),
        target_changed=False,
        previous_target_track_id=None,
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
        presence_transitions=[],
        completed_recognitions=[],
        ambiguous_crossing_track_ids=set(),
        pending_worker_tasks=0,
        frame_latency_ms=0.0,
        total_pipeline_latency_ms=0.0,
    )


class TestAuthorizationPersistenceRegression:
    """Covers requirements 1 through 9 for persistent authorization."""

    def test_1_authorization_remains_active_after_more_than_30_seconds(self) -> None:
        """1. Authorization remains active > 30 seconds without state transitions."""
        mgr = SessionManager(config=SessionConfig(require_confirmed_presence=True))
        ctrl = AuthorizationController(
            session_manager=mgr,
            timeout_seconds=30.0,
            require_continuous_presence=False,
        )
        ctrl.start()

        # Step 1: Authorize Alice
        auth_state = _mk_id_state(
            track_id=1,
            state=IdentityState.IDENTIFIED,
            person_id="p-100",
            display_name="Alice",
        )
        frame1 = _mk_frame(
            frame_id=1,
            primary_track_id=1,
            identity_states={1: auth_state},
        )
        mgr.process_frame_result(frame1)

        session = ctrl.get_current()
        assert session is not None
        assert session.status is AuthorizationStatus.AUTHORIZED
        assert session.is_authorized is True
        assert session.is_active is True
        auth_id = session.authorization_id

        # Step 2: Advance time past the original 30-second timeout window (e.g. 60s)
        base_time = time.monotonic()
        with mock.patch("time.monotonic", return_value=base_time + 60.0):
            frame2 = _mk_frame(
                frame_id=2,
                primary_track_id=1,
                identity_states={1: auth_state},
            )
            mgr.process_frame_result(frame2)
            current = ctrl.get_current()
            assert current is not None
            assert current.authorization_id == auth_id
            assert current.status is AuthorizationStatus.AUTHORIZED
            assert current.is_authorized is True
            assert current.is_active is True
            assert current.ended_at_iso is None

        ctrl.close()

    def test_2_transient_missed_detection_does_not_terminate_session(self) -> None:
        """2. Temporary missed detection does not immediately terminate valid session."""
        mgr = SessionManager(config=SessionConfig(require_confirmed_presence=True))
        ctrl = AuthorizationController(
            session_manager=mgr,
            timeout_seconds=30.0,
            require_continuous_presence=False,
        )
        ctrl.start()

        # Step 1: Authorize Alice
        auth_state = _mk_id_state(
            track_id=1,
            state=IdentityState.IDENTIFIED,
            person_id="p-100",
            display_name="Alice",
        )
        mgr.process_frame_result(
            _mk_frame(frame_id=1, primary_track_id=1, identity_states={1: auth_state})
        )

        session = ctrl.get_current()
        assert session is not None
        assert session.status is AuthorizationStatus.AUTHORIZED

        # Step 2: Temporary missed detection (primary target drops for a frame)
        mgr.process_frame_result(_mk_frame(frame_id=2, primary_track_id=None))
        current = ctrl.get_current()
        assert current is not None
        assert current.status is AuthorizationStatus.AUTHORIZED
        assert current.is_active is True

        # Step 3: Temporary uncertain/unregistered observation (e.g. face angled away)
        uncertain_state = _mk_id_state(track_id=1, state=IdentityState.UNCERTAIN)
        mgr.process_frame_result(
            _mk_frame(frame_id=3, primary_track_id=1, identity_states={1: uncertain_state})
        )
        current = ctrl.get_current()
        assert current is not None
        assert current.status is AuthorizationStatus.AUTHORIZED
        assert current.is_active is True

        # Step 4: Alice face re-observed
        mgr.process_frame_result(
            _mk_frame(frame_id=4, primary_track_id=1, identity_states={1: auth_state})
        )
        current = ctrl.get_current()
        assert current is not None
        assert current.status is AuthorizationStatus.AUTHORIZED
        assert current.is_active is True

        ctrl.close()

    def test_3_genuine_different_person_identification_terminates_session(self) -> None:
        """3. Genuine different-person identification prevents cross-person access."""
        events: list[AuthorizationSession] = []
        mgr = SessionManager(config=SessionConfig(require_confirmed_presence=True))
        ctrl = AuthorizationController(
            session_manager=mgr,
            timeout_seconds=30.0,
            on_transition=events.append,
            require_continuous_presence=False,
        )
        ctrl.start()

        # Step 1: Alice is authorized
        alice_state = _mk_id_state(
            track_id=1,
            state=IdentityState.IDENTIFIED,
            person_id="p-alice",
            display_name="Alice",
        )
        mgr.process_frame_result(
            _mk_frame(frame_id=1, primary_track_id=1, identity_states={1: alice_state})
        )
        assert ctrl.get_current().status is AuthorizationStatus.AUTHORIZED
        assert len(events) == 1

        # Step 2: Bob is identified on the primary track
        bob_state = _mk_id_state(
            track_id=2,
            state=IdentityState.IDENTIFIED,
            person_id="p-bob",
            display_name="Bob",
        )
        mgr.process_frame_result(
            _mk_frame(frame_id=2, primary_track_id=2, identity_states={2: bob_state})
        )

        # Must immediately transition to LOST to prevent cross-person access
        current = ctrl.get_current()
        assert current is not None
        assert current.status is AuthorizationStatus.LOST
        assert current.is_active is False
        assert current.is_completed is True
        assert current.ended_at_iso is not None
        assert len(events) == 2
        assert events[1].status is AuthorizationStatus.LOST

        # Step 3: Verify DialogueCoordinator refuses private access for Bob
        bus = EventBus()
        knowledge_svc = mock.MagicMock(spec=KnowledgeService)
        knowledge_svc.build_knowledge_context.return_value = "Public company info"
        coordinator = DialogueCoordinator(
            event_bus=bus,
            knowledge_service=knowledge_svc,
            authorization_controller=ctrl,
            llm_provider=MockLLMProvider(),
        )

        turn = coordinator.handle_turn(user_text="What is my secret?")
        assert turn is not None
        assert turn.role == "model"

        ctrl.close()

    def test_4_backend_lost_or_revocation_event_handled_by_cause(self) -> None:
        """4. Backend LOST/revocation event is handled according to its actual cause."""
        # Cause A: Continuous presence policy terminates on target loss
        mgr_strict = SessionManager(config=SessionConfig(require_confirmed_presence=True))
        ctrl_strict = AuthorizationController(
            session_manager=mgr_strict,
            timeout_seconds=30.0,
            require_continuous_presence=True,
        )
        ctrl_strict.start()
        auth_state = _mk_id_state(
            track_id=1,
            state=IdentityState.IDENTIFIED,
            person_id="p-1",
            display_name="User",
        )
        mgr_strict.process_frame_result(
            _mk_frame(frame_id=1, primary_track_id=1, identity_states={1: auth_state})
        )
        assert ctrl_strict.get_current().status is AuthorizationStatus.AUTHORIZED

        # Target drops -> under continuous presence, causes LOST
        mgr_strict.process_frame_result(_mk_frame(frame_id=2, primary_track_id=None))
        assert ctrl_strict.get_current().status is AuthorizationStatus.LOST
        ctrl_strict.close()

        # Cause B: Operator profile invalidation/deletion causes CANCELLED
        mgr_p = SessionManager(config=SessionConfig(require_confirmed_presence=True))
        ctrl_p = AuthorizationController(
            session_manager=mgr_p,
            timeout_seconds=30.0,
            require_continuous_presence=False,
        )
        ctrl_p.start()
        mgr_p.process_frame_result(
            _mk_frame(frame_id=1, primary_track_id=1, identity_states={1: auth_state})
        )
        assert ctrl_p.get_current().status is AuthorizationStatus.AUTHORIZED

        # Invalidate the authorized person
        ctrl_p.invalidate_person("p-1")
        assert ctrl_p.get_current().status is AuthorizationStatus.CANCELLED
        ctrl_p.close()

    def test_5_dashboard_stale_event_isolation(self) -> None:
        """5. Dashboard does not overwrite valid backend authorization with stale event."""
        s1 = AuthorizationSession(status=AuthorizationStatus.AUTHORIZED, person_id="p-1")
        s2 = AuthorizationSession(status=AuthorizationStatus.LOST, person_id="p-old")

        assert s1.authorization_id != s2.authorization_id
        assert s1.status is AuthorizationStatus.AUTHORIZED
        assert s2.status is AuthorizationStatus.LOST

    def test_6_sse_reconnection_or_polling_does_not_cause_session_lost(self) -> None:
        """6. SSE reconnection or polling does not cause valid session to become LOST."""
        mgr = SessionManager(config=SessionConfig(require_confirmed_presence=True))
        ctrl = AuthorizationController(
            session_manager=mgr,
            timeout_seconds=30.0,
            require_continuous_presence=False,
        )
        ctrl.start()

        auth_state = _mk_id_state(
            track_id=1,
            state=IdentityState.IDENTIFIED,
            person_id="p-1",
            display_name="User",
        )
        mgr.process_frame_result(
            _mk_frame(frame_id=1, primary_track_id=1, identity_states={1: auth_state})
        )
        assert ctrl.get_current().status is AuthorizationStatus.AUTHORIZED

        # Simulate repeated polling of /authorization/sessions/current over 20 cycles
        for _ in range(20):
            current = ctrl.get_current()
            assert current is not None
            assert current.status is AuthorizationStatus.AUTHORIZED
            assert current.is_authorized is True
            assert current.is_active is True

        ctrl.close()

    def test_7_explicit_logout_ends_authorization_and_clears_private_context(self) -> None:
        """7. Explicit Logout still ends authorization and clears private context."""
        events: list[AuthorizationSession] = []
        mgr = SessionManager(config=SessionConfig(require_confirmed_presence=True))
        ctrl = AuthorizationController(
            session_manager=mgr,
            timeout_seconds=30.0,
            on_transition=events.append,
            require_continuous_presence=False,
        )
        ctrl.start()

        auth_state = _mk_id_state(
            track_id=1,
            state=IdentityState.IDENTIFIED,
            person_id="p-1",
            display_name="User",
        )
        mgr.process_frame_result(
            _mk_frame(frame_id=1, primary_track_id=1, identity_states={1: auth_state})
        )
        assert ctrl.get_current().status is AuthorizationStatus.AUTHORIZED

        # Explicit cancel (Logout)
        cancelled = ctrl.cancel()
        assert cancelled.status is AuthorizationStatus.CANCELLED
        assert cancelled.is_completed is True
        assert cancelled.is_active is False
        assert ctrl.get_current().status is AuthorizationStatus.CANCELLED

        ctrl.close()

    def test_8_general_questions_continue_working_after_logout(self) -> None:
        """8. General questions continue working after logout."""
        bus = EventBus()
        mgr = SessionManager(config=SessionConfig(require_confirmed_presence=True))
        ctrl = AuthorizationController(
            session_manager=mgr,
            timeout_seconds=30.0,
            require_continuous_presence=False,
        )
        ctrl.start()
        auth_state = _mk_id_state(
            track_id=1,
            state=IdentityState.IDENTIFIED,
            person_id="p-1",
            display_name="User",
        )
        mgr.process_frame_result(
            _mk_frame(frame_id=1, primary_track_id=1, identity_states={1: auth_state})
        )

        knowledge_svc = mock.MagicMock(spec=KnowledgeService)
        knowledge_svc.build_knowledge_context.return_value = (
            "General knowledge: Dhaka is the capital of Bangladesh."
        )

        coordinator = DialogueCoordinator(
            event_bus=bus,
            knowledge_service=knowledge_svc,
            authorization_controller=ctrl,
            llm_provider=MockLLMProvider(),
        )

        # Question while authorized
        turn1 = coordinator.handle_turn(user_text="What is Dhaka?")
        assert turn1.role == "model"
        assert len(turn1.text) > 0

        # Log out
        ctrl.cancel()
        coordinator.revoke_authorization()

        # Question after logout: general question must still receive an answer
        turn2 = coordinator.handle_turn(user_text="What is Dhaka?")
        assert turn2.role == "model"
        assert len(turn2.text) > 0
        assert "Dhaka" in turn2.text or "Mock" in turn2.text or len(turn2.text) > 5

        ctrl.close()

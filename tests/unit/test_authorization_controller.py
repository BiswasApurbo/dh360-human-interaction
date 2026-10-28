"""Unit tests for AuthorizationController.

Tests cover:
- construction and observer registration
- single active session enforcement
- close/unregister/idempotence
- AUTHORIZED status derivation (primary target identified with person_id)
- UNKNOWN status derivation
- UNCERTAIN status derivation
- NO_FACE status derivation (no primary target selected)
- AUTHORIZED -> LOST transition (terminal, immutable, preserves person_id/display_name)
- TIMEOUT behavior (deadline expiration, terminal)
- terminal immutability (frames ignored once session is completed/terminal)
- primary target only considered (non-primary tracks ignored)
- no prohibited runtime/DB/network calls
- regression tests for the frozen AuthorizationSession contract
"""

from __future__ import annotations

from unittest import mock

import pytest

from dh360_hi.contracts.authorization import (
    AuthorizationSession,
    AuthorizationStatus,
)
from dh360_hi.edge.authorization_controller import AuthorizationController
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


def _mk_frame(
    frame_id: int = 1,
    timestamp: float = 1.0,
    primary_track_id: int | None = None,
    candidates: list[TargetCandidate] | None = None,
    identity_states: dict[int, TrackIdentityState] | None = None,
) -> TwoRateFrameResult:
    target_result = _mk_target_result(
        primary_track_id=primary_track_id,
        candidates=candidates,
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


def _manager() -> SessionManager:
    cfg = SessionConfig(require_confirmed_presence=True)
    return SessionManager(config=cfg)


@pytest.fixture
def controller() -> AuthorizationController:
    mgr = _manager()
    ctrl = AuthorizationController(session_manager=mgr, timeout_seconds=5.0)
    yield ctrl
    ctrl.close()


class TestAuthorizationControllerLifecycle:
    def test_construction_registers_observer(self, controller: AuthorizationController) -> None:
        mgr = controller._session_manager
        assert controller._on_frame in mgr._observers

    def test_close_unregisters_observer_and_is_idempotent(
        self, controller: AuthorizationController
    ) -> None:
        mgr = controller._session_manager
        assert controller._on_frame in mgr._observers

        controller.close()
        assert controller._on_frame not in mgr._observers

        # Calling close() again must be safe and idempotent
        controller.close()
        assert controller._on_frame not in mgr._observers

    def test_start_creates_active_session(self, controller: AuthorizationController) -> None:
        assert controller.get_current() is None
        sess = controller.start()
        assert isinstance(sess, AuthorizationSession)
        assert sess.status is AuthorizationStatus.ACTIVE
        assert sess.is_active is True
        assert sess.is_completed is False
        assert controller.get_current() == sess

    def test_second_start_while_active_raises_value_error(
        self, controller: AuthorizationController
    ) -> None:
        controller.start()
        with pytest.raises(ValueError, match="already exists"):
            controller.start()

    def test_start_allowed_after_previous_session_completed(
        self, controller: AuthorizationController
    ) -> None:
        sess1 = controller.start()
        # Complete sess1 via timeout
        with mock.patch("time.monotonic", return_value=controller._deadline + 1):  # type: ignore[operator]
            timed_out = controller.get_current()
            assert timed_out is not None
            assert timed_out.status is AuthorizationStatus.TIMEOUT
            assert timed_out.is_completed is True

        # Now start should succeed and create a fresh session
        sess2 = controller.start()
        assert sess2.authorization_id != sess1.authorization_id
        assert sess2.status is AuthorizationStatus.ACTIVE

    def test_start_allowed_when_previous_session_expired_without_prior_get_current(
        self, controller: AuthorizationController
    ) -> None:
        sess1 = controller.start()
        # Advance time past deadline without calling get_current() or pumping frames
        with mock.patch("time.monotonic", return_value=controller._deadline + 1):  # type: ignore[operator]
            sess2 = controller.start()
            assert sess2.authorization_id != sess1.authorization_id
            assert sess2.status is AuthorizationStatus.ACTIVE
            assert sess2.is_active is True

    def test_cancel_active_session(self, controller: AuthorizationController) -> None:
        """Verify an active session can be explicitly cancelled."""
        sess1 = controller.start()
        assert sess1.is_active

        sess2 = controller.cancel()
        assert not sess2.is_active
        assert sess2.status == AuthorizationStatus.CANCELLED

        # Verify a new session can be started after cancellation
        sess3 = controller.start()
        assert sess3.is_active
        assert sess3.authorization_id != sess1.authorization_id

    def test_cancel_raises_when_no_active_session(
        self, controller: AuthorizationController
    ) -> None:
        """Verify cancel raises ValueError if no active session."""
        import pytest

        with pytest.raises(ValueError, match="No active authorization session to cancel"):
            controller.cancel()

    def test_start_raises_after_controller_closed(
        self, controller: AuthorizationController
    ) -> None:
        controller.close()
        with pytest.raises(RuntimeError, match="closed"):
            controller.start()


class TestAuthorizationDerivations:
    def test_authorized_when_primary_identified(self, controller: AuthorizationController) -> None:
        controller.start()
        mgr = controller._session_manager

        id_state = _mk_id_state(
            track_id=1,
            state=IdentityState.IDENTIFIED,
            person_id="p-100",
            display_name="Alice",
        )
        frame = _mk_frame(primary_track_id=1, identity_states={1: id_state})
        mgr.process_frame_result(frame)

        current = controller.get_current()
        assert current is not None
        assert current.status is AuthorizationStatus.AUTHORIZED
        assert current.is_authorized is True
        assert current.person_id == "p-100"
        assert current.display_name == "Alice"

    def test_unknown_when_primary_unregistered(self, controller: AuthorizationController) -> None:
        controller.start()
        mgr = controller._session_manager

        id_state = _mk_id_state(track_id=1, state=IdentityState.UNREGISTERED)
        frame = _mk_frame(primary_track_id=1, identity_states={1: id_state})
        mgr.process_frame_result(frame)

        current = controller.get_current()
        assert current is not None
        assert current.status is AuthorizationStatus.UNKNOWN
        assert current.person_id is None

    def test_uncertain_when_primary_uncertain(self, controller: AuthorizationController) -> None:
        controller.start()
        mgr = controller._session_manager

        id_state = _mk_id_state(track_id=1, state=IdentityState.UNCERTAIN)
        frame = _mk_frame(primary_track_id=1, identity_states={1: id_state})
        mgr.process_frame_result(frame)

        current = controller.get_current()
        assert current is not None
        assert current.status is AuthorizationStatus.UNCERTAIN
        assert current.person_id is None

    def test_no_face_when_no_primary_target(self, controller: AuthorizationController) -> None:
        controller.start()
        mgr = controller._session_manager

        frame = _mk_frame(primary_track_id=None)
        mgr.process_frame_result(frame)

        current = controller.get_current()
        assert current is not None
        assert current.status is AuthorizationStatus.NO_FACE
        assert current.person_id is None

    def test_primary_target_only_evaluated(self, controller: AuthorizationController) -> None:
        """Secondary candidates do not trigger authorization."""
        controller.start()
        mgr = controller._session_manager

        primary_id = _mk_id_state(track_id=1, state=IdentityState.NEW)
        secondary_id = _mk_id_state(
            track_id=2,
            state=IdentityState.IDENTIFIED,
            person_id="p-200",
            display_name="Bob",
        )
        cand1 = _mk_candidate(1, state=IdentityState.NEW)
        cand2 = _mk_candidate(2, person_id="p-200", state=IdentityState.IDENTIFIED)

        frame = _mk_frame(
            primary_track_id=1,
            candidates=[cand1, cand2],
            identity_states={1: primary_id, 2: secondary_id},
        )
        mgr.process_frame_result(frame)

        current = controller.get_current()
        assert current is not None
        assert current.status is AuthorizationStatus.ACTIVE
        assert current.person_id is None


class TestAuthorizedToLostSemantics:
    def test_authorized_to_lost_transition_lifecycle(
        self, controller: AuthorizationController
    ) -> None:
        """AUTHORIZED -> LOST transition completes the session while preserving identity."""
        controller.start()
        mgr = controller._session_manager

        # Frame 1: Authorize
        auth_state = _mk_id_state(
            track_id=1,
            state=IdentityState.IDENTIFIED,
            person_id="p-100",
            display_name="Alice",
        )
        mgr.process_frame_result(_mk_frame(primary_track_id=1, identity_states={1: auth_state}))

        auth_session = controller.get_current()
        assert auth_session is not None
        assert auth_session.status is AuthorizationStatus.AUTHORIZED
        assert auth_session.person_id == "p-100"
        assert auth_session.display_name == "Alice"
        assert auth_session.is_active is True

        # Frame 2: Target lost (no primary target)
        mgr.process_frame_result(_mk_frame(primary_track_id=None))

        lost_session = controller.get_current()
        assert lost_session is not None
        assert lost_session.status is AuthorizationStatus.LOST
        # Preserves authorization_id and authorized person identity
        assert lost_session.authorization_id == auth_session.authorization_id
        assert lost_session.person_id == "p-100"
        assert lost_session.display_name == "Alice"
        # LOST is terminal: marked completed with ended_at_iso
        assert lost_session.is_completed is True
        assert lost_session.is_active is False
        assert lost_session.ended_at_iso is not None

        # Respects frozen AuthorizationSession contract (new value object returned)
        assert isinstance(lost_session, AuthorizationSession)
        assert lost_session is not auth_session

    def test_terminal_immutability_after_lost(self, controller: AuthorizationController) -> None:
        """Once LOST, subsequent frames cannot alter the completed session."""
        controller.start()
        mgr = controller._session_manager

        auth_state = _mk_id_state(
            track_id=1,
            state=IdentityState.IDENTIFIED,
            person_id="p-100",
            display_name="Alice",
        )
        mgr.process_frame_result(_mk_frame(primary_track_id=1, identity_states={1: auth_state}))
        mgr.process_frame_result(_mk_frame(primary_track_id=None))

        lost_session = controller.get_current()
        assert lost_session is not None
        assert lost_session.status is AuthorizationStatus.LOST

        # Later frame arrives with another identified person
        other_state = _mk_id_state(
            track_id=2,
            state=IdentityState.IDENTIFIED,
            person_id="p-200",
            display_name="Bob",
        )
        mgr.process_frame_result(_mk_frame(primary_track_id=2, identity_states={2: other_state}))

        current = controller.get_current()
        assert current == lost_session
        assert current.status is AuthorizationStatus.LOST
        assert current.person_id == "p-100"


class TestTimeoutAndTerminality:
    def test_timeout_expiration_completes_session(
        self, controller: AuthorizationController
    ) -> None:
        controller.start()
        assert controller.get_current() is not None
        assert controller.get_current().status is AuthorizationStatus.ACTIVE  # type: ignore[union-attr]

        with mock.patch("time.monotonic", return_value=controller._deadline + 1):  # type: ignore[operator]
            timed_out = controller.get_current()
            assert timed_out is not None
            assert timed_out.status is AuthorizationStatus.TIMEOUT
            assert timed_out.is_completed is True
            assert timed_out.ended_at_iso is not None

    def test_frame_after_timeout_does_not_modify_session(
        self, controller: AuthorizationController
    ) -> None:
        controller.start()
        mgr = controller._session_manager

        with mock.patch("time.monotonic", return_value=controller._deadline + 1):  # type: ignore[operator]
            # Frame observation checks timeout and completes session
            mgr.process_frame_result(_mk_frame(primary_track_id=1))

        current = controller.get_current()
        assert current is not None
        assert current.status is AuthorizationStatus.TIMEOUT
        assert current.is_completed is True


class TestProhibitedOperationsAndPrivacy:
    def test_privacy_invariants_in_snapshot(self, controller: AuthorizationController) -> None:
        controller.start()
        mgr = controller._session_manager

        auth_state = _mk_id_state(
            track_id=1,
            state=IdentityState.IDENTIFIED,
            person_id="p-100",
            display_name="Alice",
        )
        mgr.process_frame_result(_mk_frame(primary_track_id=1, identity_states={1: auth_state}))

        current = controller.get_current()
        assert current is not None
        payload = current.to_dict()

        # Only transport-safe fields
        assert set(payload.keys()) == {
            "authorization_id",
            "status",
            "person_id",
            "display_name",
            "created_at_iso",
            "updated_at_iso",
            "ended_at_iso",
        }
        forbidden = [
            "track_id",
            "frame_id",
            "bbox",
            "embedding",
            "embeddings",
            "recognition",
            "RecognitionResult",
        ]
        for field in forbidden:
            assert not hasattr(current, field)
            assert field not in payload

    def test_controller_never_calls_process_frame_result_or_runtime(
        self, controller: AuthorizationController
    ) -> None:
        """Controller does not invoke pipeline step or frame processing."""
        mgr = controller._session_manager
        with mock.patch.object(mgr, "process_frame_result") as mock_process:
            controller.start()
            controller.get_current()
            controller.close()
            assert mock_process.call_count == 0


class TestAuthorizationControllerTransitionNotification:
    """Tests covering on_transition callback invocation and exception safety."""

    def test_callback_fires_on_active_to_authorized(self) -> None:
        events: list[AuthorizationSession] = []
        mgr = SessionManager(config=SessionConfig(require_confirmed_presence=True))
        ctrl = AuthorizationController(
            session_manager=mgr,
            timeout_seconds=5.0,
            on_transition=events.append,
        )
        ctrl.start()
        assert len(events) == 0

        # Frame 1: Transitions to AUTHORIZED
        auth_state = _mk_id_state(
            track_id=1,
            state=IdentityState.IDENTIFIED,
            person_id="p-100",
            display_name="Alice",
        )
        mgr.process_frame_result(_mk_frame(primary_track_id=1, identity_states={1: auth_state}))

        assert len(events) == 1
        assert events[0].status is AuthorizationStatus.AUTHORIZED
        assert events[0].person_id == "p-100"
        assert events[0].display_name == "Alice"
        assert events[0].is_authorized is True

    def test_callback_fires_on_authorized_to_lost(self) -> None:
        events: list[AuthorizationSession] = []
        mgr = SessionManager(config=SessionConfig(require_confirmed_presence=True))
        ctrl = AuthorizationController(
            session_manager=mgr,
            timeout_seconds=5.0,
            on_transition=events.append,
        )
        ctrl.start()

        auth_state = _mk_id_state(
            track_id=1,
            state=IdentityState.IDENTIFIED,
            person_id="p-100",
            display_name="Alice",
        )
        mgr.process_frame_result(_mk_frame(primary_track_id=1, identity_states={1: auth_state}))
        assert len(events) == 1

        # Frame 2: Primary target drops -> LOST
        mgr.process_frame_result(_mk_frame(primary_track_id=None))

        assert len(events) == 2
        assert events[1].status is AuthorizationStatus.LOST
        assert events[1].person_id == "p-100"
        assert events[1].is_completed is True
        assert events[1].ended_at_iso is not None

    def test_callback_fires_on_unknown_uncertain_no_face(self) -> None:
        events: list[AuthorizationSession] = []
        mgr = SessionManager(config=SessionConfig(require_confirmed_presence=True))
        ctrl = AuthorizationController(
            session_manager=mgr,
            timeout_seconds=5.0,
            on_transition=events.append,
        )
        ctrl.start()

        # Frame 1: UNKNOWN
        unreg_state = _mk_id_state(track_id=1, state=IdentityState.UNREGISTERED)
        mgr.process_frame_result(_mk_frame(primary_track_id=1, identity_states={1: unreg_state}))
        assert len(events) == 1
        assert events[0].status is AuthorizationStatus.UNKNOWN

        # Frame 2: UNCERTAIN
        unc_state = _mk_id_state(track_id=1, state=IdentityState.UNCERTAIN)
        mgr.process_frame_result(_mk_frame(primary_track_id=1, identity_states={1: unc_state}))
        assert len(events) == 2
        assert events[1].status is AuthorizationStatus.UNCERTAIN

        # Frame 3: NO_FACE
        mgr.process_frame_result(_mk_frame(primary_track_id=None))
        assert len(events) == 3
        assert events[2].status is AuthorizationStatus.NO_FACE

    def test_callback_does_not_fire_repeatedly_when_status_unchanged(self) -> None:
        events: list[AuthorizationSession] = []
        mgr = SessionManager(config=SessionConfig(require_confirmed_presence=True))
        ctrl = AuthorizationController(
            session_manager=mgr,
            timeout_seconds=5.0,
            on_transition=events.append,
        )
        ctrl.start()

        # Frame 1: UNKNOWN
        unreg_state = _mk_id_state(track_id=1, state=IdentityState.UNREGISTERED)
        mgr.process_frame_result(_mk_frame(primary_track_id=1, identity_states={1: unreg_state}))
        assert len(events) == 1

        # Frame 2: Still UNKNOWN
        mgr.process_frame_result(_mk_frame(primary_track_id=1, identity_states={1: unreg_state}))
        assert len(events) == 1  # Did NOT fire again

    def test_callback_fires_on_timeout_via_frame_or_get_current(self) -> None:
        events: list[AuthorizationSession] = []
        mgr = SessionManager(config=SessionConfig(require_confirmed_presence=True))
        ctrl = AuthorizationController(
            session_manager=mgr,
            timeout_seconds=0.01,
            on_transition=events.append,
        )
        ctrl.start()

        # Manually force deadline expiration
        ctrl._deadline = 0.0

        # Frame triggers timeout
        mgr.process_frame_result(_mk_frame(primary_track_id=None))
        assert len(events) == 1
        assert events[0].status is AuthorizationStatus.TIMEOUT
        assert events[0].is_completed is True

        # Second test: get_current timeout
        events2: list[AuthorizationSession] = []
        ctrl2 = AuthorizationController(
            session_manager=mgr,
            timeout_seconds=0.01,
            on_transition=events2.append,
        )
        ctrl2.start()
        ctrl2._deadline = 0.0
        curr = ctrl2.get_current()
        assert curr is not None
        assert curr.status is AuthorizationStatus.TIMEOUT
        assert len(events2) == 1
        assert events2[0].status is AuthorizationStatus.TIMEOUT

    def test_callback_exceptions_are_isolated(self) -> None:
        def failing_callback(_session: AuthorizationSession) -> None:
            msg = "listener failed unexpectedly"
            raise RuntimeError(msg)

        mgr = SessionManager(config=SessionConfig(require_confirmed_presence=True))
        ctrl = AuthorizationController(
            session_manager=mgr,
            timeout_seconds=5.0,
            on_transition=failing_callback,
        )
        ctrl.start()

        # Processing frame should not throw even though callback raises
        auth_state = _mk_id_state(
            track_id=1,
            state=IdentityState.IDENTIFIED,
            person_id="p-100",
            display_name="Alice",
        )
        mgr.process_frame_result(_mk_frame(primary_track_id=1, identity_states={1: auth_state}))
        current = ctrl.get_current()
        assert current is not None
        assert current.status is AuthorizationStatus.AUTHORIZED

    def test_invalidate_person_cancels_only_matching_person(self) -> None:
        events: list[AuthorizationSession] = []
        mgr = SessionManager(config=SessionConfig(require_confirmed_presence=True))
        ctrl = AuthorizationController(
            session_manager=mgr,
            timeout_seconds=30.0,
            on_transition=events.append,
        )
        ctrl.start()

        auth_state = _mk_id_state(
            track_id=1,
            state=IdentityState.IDENTIFIED,
            person_id="p-target",
            display_name="Target Person",
        )
        mgr.process_frame_result(_mk_frame(primary_track_id=1, identity_states={1: auth_state}))

        curr = ctrl.get_current()
        assert curr is not None
        assert curr.status is AuthorizationStatus.AUTHORIZED
        assert curr.person_id == "p-target"

        # Invalidate a different person: should have no effect
        res_other = ctrl.invalidate_person("p-other")
        assert res_other is None
        curr_after = ctrl.get_current()
        assert curr_after is not None
        assert curr_after.status is AuthorizationStatus.AUTHORIZED
        assert curr_after.person_id == "p-target"

        # Invalidate matching person: should cancel session and notify observer
        res_target = ctrl.invalidate_person("p-target")
        assert res_target is not None
        assert res_target.status is AuthorizationStatus.CANCELLED
        assert res_target.person_id is None
        assert res_target.is_completed is True
        assert ctrl.get_current() is not None
        assert ctrl.get_current().status is AuthorizationStatus.CANCELLED
        assert events[-1].status is AuthorizationStatus.CANCELLED

        # Invalidate again when already cancelled: should return None
        res_again = ctrl.invalidate_person("p-target")
        assert res_again is None

    def test_invalidate_person_when_no_active_session(self) -> None:
        mgr = SessionManager(config=SessionConfig(require_confirmed_presence=True))
        ctrl = AuthorizationController(session_manager=mgr)
        assert ctrl.invalidate_person("p-any") is None
        ctrl.close()
        assert ctrl.invalidate_person("p-any") is None

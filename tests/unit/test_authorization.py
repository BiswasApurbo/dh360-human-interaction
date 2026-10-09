"""Unit tests for the M3.3 authorization domain model (status vocabulary + session snapshot)."""

from __future__ import annotations

import dataclasses
import json
import uuid

import pytest

# Preload the contracts package before any ``dh360_hi.backend.*`` import. The
# repository has a pre-existing import-order sensitivity: ``backend.models``
# imports ``contracts.interaction`` at runtime (models.py:14), so a backend-first
# entry reaches ``contracts/__init__`` while ``backend.models`` is only partially
# initialized and fails. Initializing contracts first (as the full test suite and
# application entry points do) avoids that cycle. Straight imports are placed
# before from-imports by ruff/isort, so this stays in canonical order.
import dh360_hi.contracts  # noqa: F401
from dh360_hi.backend.exceptions import ValidationError
from dh360_hi.contracts import (
    AuthorizationSession,
    AuthorizationStatus,
    derive_authorization_status,
)
from dh360_hi.vision.identity_state import IdentityState, IdentityStateManager
from dh360_hi.vision.recognition import RecognitionOutcome, RecognitionResult
from dh360_hi.vision.target_selector import TargetCandidate, TargetSelectionResult

FORBIDDEN_INTERNAL_FIELDS = (
    "track_id",
    "primary_track_id",
    "frame_id",
    "image",
    "images",
    "frame",
    "embedding",
    "embeddings",
    "recognition_result",
    "camera",
    "bbox",
    "landmarks",
    "target_result",
    "identity_state",
    "gallery",
    "profile",
)


def _make_target_result(has_primary: bool, *, frame_id: int = 1) -> TargetSelectionResult:
    """Build a real TargetSelectionResult with or without a primary target."""
    primary: TargetCandidate | None = None
    if has_primary:
        primary = TargetCandidate(
            track_id=7,
            score=0.9,
            centrality_score=0.8,
            size_score=0.7,
            identity_state_score=0.6,
            dwell_score=0.5,
            is_identified=False,
            identity_state=IdentityState.NEW,
            person_id=None,
            display_name=None,
            bbox=(10.0, 10.0, 110.0, 110.0),
            center=(60.0, 60.0),
        )
    return TargetSelectionResult(
        primary_target=primary,
        candidates=[primary] if primary is not None else [],
        target_changed=False,
        previous_target_track_id=None,
        frame_id=frame_id,
    )


def _make_rec(outcome: RecognitionOutcome) -> RecognitionResult:
    """Build a RecognitionResult (mirrors test_identity_stability_and_swap helper)."""
    return RecognitionResult(
        outcome=outcome,
        top_match=None,
        second_match=None,
        similarity_margin=0.0,
        confidence_score=0.0,
        model_name="SFace",
        model_version="2021dec",
        is_model_compatible=True,
        details="test",
    )


def _authorized_session() -> AuthorizationSession:
    """A running session transitioned to AUTHORIZED for person 'p-100' (Ada)."""
    return AuthorizationSession().transition(
        AuthorizationStatus.AUTHORIZED,
        person_id="p-100",
        display_name="Ada",
    )


class TestAuthorizationStatus:
    def test_status_values_match_domain_vocabulary(self) -> None:
        """All seven externally meaningful statuses exist with stable wire values."""
        assert {status.value for status in AuthorizationStatus} == {
            "active",
            "authorized",
            "unknown",
            "uncertain",
            "no_face",
            "lost",
            "timeout",
            "cancelled",
        }

    @pytest.mark.parametrize(
        ("status", "kwargs"),
        [
            (AuthorizationStatus.ACTIVE, {}),
            (AuthorizationStatus.AUTHORIZED, {"person_id": "p-100"}),
            (AuthorizationStatus.UNKNOWN, {}),
            (AuthorizationStatus.UNCERTAIN, {}),
            (AuthorizationStatus.NO_FACE, {}),
            (AuthorizationStatus.LOST, {"person_id": "p-100"}),
            (AuthorizationStatus.TIMEOUT, {}),
        ],
    )
    def test_every_status_is_constructible(
        self, status: AuthorizationStatus, kwargs: dict[str, str]
    ) -> None:
        """Every status value builds a valid snapshot with the correct identity shape."""
        session = AuthorizationSession(status=status, **kwargs)
        assert session.status is status
        assert session.to_dict()["status"] == status.value

    def test_status_accepts_plain_wire_strings(self) -> None:
        """Transport payloads carrying plain strings deserialize into the enum."""
        session = AuthorizationSession(status="authorized", person_id="p-100")
        assert session.status is AuthorizationStatus.AUTHORIZED

    def test_unknown_status_value_rejected(self) -> None:
        """Garbage status strings raise ValidationError, not raw ValueError."""
        with pytest.raises(ValidationError):
            AuthorizationSession(status="bogus")


class TestSessionLifecycleState:
    def test_valid_initial_session_state(self) -> None:
        """A fresh session is ACTIVE, running, and carries no identity or decision."""
        session = AuthorizationSession()
        assert session.status is AuthorizationStatus.ACTIVE
        assert session.is_active is True
        assert session.is_completed is False
        assert session.is_authorized is False
        assert session.person_id is None
        assert session.display_name is None
        assert session.ended_at_iso is None
        uuid.UUID(session.authorization_id)  # unique UUID string by default
        assert session.created_at_iso and session.updated_at_iso

    def test_session_is_frozen(self) -> None:
        """Snapshots are immutable, per repository DTO conventions."""
        session = AuthorizationSession()
        with pytest.raises(dataclasses.FrozenInstanceError):
            session.status = AuthorizationStatus.TIMEOUT  # type: ignore[misc]


class TestAuthorizedResultIdentity:
    def test_authorized_result_carries_person_identity(self) -> None:
        """An AUTHORIZED result carries the person identity and display name."""
        session = _authorized_session()
        assert session.status is AuthorizationStatus.AUTHORIZED
        assert session.is_authorized is True
        assert session.person_id == "p-100"
        assert session.display_name == "Ada"
        payload = session.to_dict()
        assert payload["person_id"] == "p-100"
        assert payload["display_name"] == "Ada"

    def test_authorized_requires_person_id(self) -> None:
        """AUTHORIZED without an identity is rejected (state-combination guard)."""
        with pytest.raises(ValidationError):
            AuthorizationSession(status=AuthorizationStatus.AUTHORIZED)

    def test_authorized_rejects_blank_person_id(self) -> None:
        """Blank/whitespace person ids are rejected like other contract payloads."""
        with pytest.raises(ValidationError):
            AuthorizationSession(status=AuthorizationStatus.AUTHORIZED, person_id="   ")

    def test_authorized_display_name_is_optional(self) -> None:
        """The preferred-name policy may yield no displayable name."""
        session = AuthorizationSession(status=AuthorizationStatus.AUTHORIZED, person_id="p-1")
        assert session.display_name is None

    def test_authorized_rejects_blank_display_name(self) -> None:
        """Blank display names are rejected per CreatePersonCommand conventions."""
        with pytest.raises(ValidationError):
            AuthorizationSession(
                status=AuthorizationStatus.AUTHORIZED,
                person_id="p-1",
                display_name="  ",
            )

    def test_lost_requires_previously_authorized_person(self) -> None:
        """LOST snapshots structurally require an authorized person identity."""
        with pytest.raises(ValidationError):
            AuthorizationSession(status=AuthorizationStatus.LOST)

    def test_lost_carries_previous_person_identity(self) -> None:
        """A directly built LOST snapshot keeps the previous authorized identity."""
        session = AuthorizationSession(
            status=AuthorizationStatus.LOST,
            person_id="p-100",
            display_name="Ada",
        )
        assert session.person_id == "p-100"
        assert session.display_name == "Ada"


class TestTransportNeutralResult:
    def test_to_dict_exposes_only_transport_safe_fields(self) -> None:
        """The result payload is exactly the documented transport shape."""
        payload = _authorized_session().to_dict()
        assert set(payload) == {
            "authorization_id",
            "status",
            "person_id",
            "display_name",
            "created_at_iso",
            "updated_at_iso",
            "ended_at_iso",
        }
        json.dumps(payload)  # fully JSON-serializable, no custom objects

    def test_no_internal_vision_or_database_fields_exposed(self) -> None:
        """No images, embeddings, frame/track ids, RecognitionResult, or DB objects."""
        session = _authorized_session()
        payload = session.to_dict()
        for name in FORBIDDEN_INTERNAL_FIELDS:
            assert not hasattr(session, name)
            assert name not in payload


class TestInvalidStateCombinations:
    @pytest.mark.parametrize(
        "status",
        [
            AuthorizationStatus.ACTIVE,
            AuthorizationStatus.UNKNOWN,
            AuthorizationStatus.UNCERTAIN,
            AuthorizationStatus.NO_FACE,
            AuthorizationStatus.TIMEOUT,
        ],
    )
    def test_identity_less_statuses_reject_person_id(self, status: AuthorizationStatus) -> None:
        """Only AUTHORIZED/LOST may claim an identity."""
        with pytest.raises(ValidationError):
            AuthorizationSession(status=status, person_id="p-100")

    @pytest.mark.parametrize(
        "status",
        [
            AuthorizationStatus.ACTIVE,
            AuthorizationStatus.UNKNOWN,
            AuthorizationStatus.UNCERTAIN,
            AuthorizationStatus.NO_FACE,
            AuthorizationStatus.TIMEOUT,
        ],
    )
    def test_identity_less_statuses_reject_display_name(self, status: AuthorizationStatus) -> None:
        """Names surface only alongside an authorized identity (policy invariants)."""
        with pytest.raises(ValidationError):
            AuthorizationSession(status=status, display_name="Ada")

    def test_active_session_cannot_have_end_timestamp(self) -> None:
        """A running ACTIVE snapshot cannot claim completion."""
        with pytest.raises(ValidationError):
            AuthorizationSession(
                status=AuthorizationStatus.ACTIVE,
                ended_at_iso="2026-01-01T00:00:00+00:00",
            )

    def test_blank_end_timestamp_rejected(self) -> None:
        """Completion timestamps are validated when present (None means running)."""
        with pytest.raises(ValidationError):
            AuthorizationSession(status=AuthorizationStatus.UNKNOWN, ended_at_iso="  ")

    def test_blank_authorization_id_rejected(self) -> None:
        """The session id is a required, non-empty opaque string."""
        with pytest.raises(ValidationError):
            AuthorizationSession(authorization_id=" ")


class TestTransitionAndCompletion:
    def test_transition_keeps_session_running(self) -> None:
        """Transitions produce a new running snapshot with preserved identity fields."""
        session = AuthorizationSession()
        observed = session.transition(
            AuthorizationStatus.UNKNOWN, at_iso="2026-01-01T00:00:01+00:00"
        )
        assert observed.status is AuthorizationStatus.UNKNOWN
        assert observed.is_active is True
        assert observed.is_completed is False
        assert observed.updated_at_iso == "2026-01-01T00:00:01+00:00"
        assert observed.authorization_id == session.authorization_id
        assert observed.created_at_iso == session.created_at_iso

    def test_transition_to_authorized_requires_person_id(self) -> None:
        """Reaching AUTHORIZED must supply the verified person identity."""
        with pytest.raises(ValidationError):
            AuthorizationSession().transition(AuthorizationStatus.AUTHORIZED)

    @pytest.mark.parametrize(
        "status",
        [
            AuthorizationStatus.ACTIVE,
            AuthorizationStatus.UNKNOWN,
            AuthorizationStatus.NO_FACE,
        ],
    )
    def test_lost_rejected_without_prior_authorized_status(
        self, status: AuthorizationStatus
    ) -> None:
        """LOST is unreachable from non-AUTHORIZED statuses, even with a person."""
        session = AuthorizationSession(status=status)
        with pytest.raises(ValidationError):
            session.transition(AuthorizationStatus.LOST, person_id="p-100")

    def test_lost_inherits_identity_from_authorized_session(self) -> None:
        """Identity loss keeps the previously authorized person identity."""
        lost = _authorized_session().transition(AuthorizationStatus.LOST)
        assert lost.status is AuthorizationStatus.LOST
        assert lost.person_id == "p-100"
        assert lost.display_name == "Ada"
        assert lost.is_active is True  # running until completed
        assert lost.is_completed is False

    @pytest.mark.parametrize(
        "status",
        [
            AuthorizationStatus.ACTIVE,
            AuthorizationStatus.UNKNOWN,
            AuthorizationStatus.UNCERTAIN,
            AuthorizationStatus.NO_FACE,
        ],
    )
    def test_authorized_session_cannot_regress_to_observation_statuses(
        self, status: AuthorizationStatus
    ) -> None:
        """An AUTHORIZED session only evolves to LOST/TIMEOUT or completion."""
        with pytest.raises(ValidationError):
            _authorized_session().transition(status)

    def test_authorized_session_can_time_out_and_clears_identity(self) -> None:
        """TIMEOUT is a controller outcome that drops the identity claim."""
        timed_out = _authorized_session().transition(AuthorizationStatus.TIMEOUT)
        assert timed_out.status is AuthorizationStatus.TIMEOUT
        assert timed_out.person_id is None
        assert timed_out.display_name is None

    def test_completed_session_rejects_further_transitions(self) -> None:
        """Completion is terminal for the session snapshot."""
        completed = _authorized_session().complete()
        assert completed.is_completed is True
        assert completed.status is AuthorizationStatus.AUTHORIZED
        assert completed.person_id == "p-100"
        assert completed.ended_at_iso is not None
        with pytest.raises(ValidationError):
            completed.transition(AuthorizationStatus.TIMEOUT)
        with pytest.raises(ValidationError):
            completed.complete(AuthorizationStatus.LOST)

    def test_completion_requires_final_status(self) -> None:
        """ACTIVE is a running state and can never be a completion decision."""
        session = AuthorizationSession()
        with pytest.raises(ValidationError):
            session.complete()
        with pytest.raises(ValidationError):
            session.complete(AuthorizationStatus.ACTIVE)

    def test_completion_records_end_timestamp(self) -> None:
        """Completing stamps the end time and marks the session completed."""
        done = AuthorizationSession().complete(
            AuthorizationStatus.UNKNOWN, at_iso="2026-01-01T00:00:05+00:00"
        )
        assert done.status is AuthorizationStatus.UNKNOWN
        assert done.ended_at_iso == "2026-01-01T00:00:05+00:00"
        assert done.updated_at_iso == "2026-01-01T00:00:05+00:00"
        assert done.is_completed is True
        assert done.is_active is False

    def test_completion_of_authorized_session_retains_identity(self) -> None:
        """Default completion keeps the authorized identity intact."""
        done = _authorized_session().complete(at_iso="2026-01-01T00:00:09+00:00")
        assert done.status is AuthorizationStatus.AUTHORIZED
        assert done.person_id == "p-100"
        assert done.display_name == "Ada"
        assert done.ended_at_iso == "2026-01-01T00:00:09+00:00"


class TestDeriveAuthorizationStatus:
    def test_no_face_comes_from_target_semantics(self) -> None:
        """NO_FACE comes only from a missing primary target, never identity state."""
        no_target = _make_target_result(False)
        assert (
            derive_authorization_status(None, target_result=no_target, previously_authorized=False)
            is AuthorizationStatus.NO_FACE
        )
        assert (
            derive_authorization_status(
                IdentityState.UNREGISTERED,
                target_result=no_target,
                previously_authorized=False,
            )
            is AuthorizationStatus.NO_FACE
        )

    def test_target_without_identity_record_is_active(self) -> None:
        """A target with no identity record yet is still gathering evidence."""
        status = derive_authorization_status(
            None,
            target_result=_make_target_result(True),
            previously_authorized=False,
        )
        assert status is AuthorizationStatus.ACTIVE

    @pytest.mark.parametrize(
        ("identity_state", "expected"),
        [
            (IdentityState.NEW, AuthorizationStatus.ACTIVE),
            (IdentityState.COLLECTING, AuthorizationStatus.ACTIVE),
            (IdentityState.IDENTIFIED, AuthorizationStatus.AUTHORIZED),
            (IdentityState.UNREGISTERED, AuthorizationStatus.UNKNOWN),
            (IdentityState.UNCERTAIN, AuthorizationStatus.UNCERTAIN),
            # Track expiry BEFORE any authorization is not identity loss:
            (IdentityState.LOST, AuthorizationStatus.ACTIVE),
        ],
    )
    def test_pre_authorization_state_mapping(
        self, identity_state: IdentityState, expected: AuthorizationStatus
    ) -> None:
        """Documented IdentityState -> AuthorizationStatus mapping (pre-authorization)."""
        status = derive_authorization_status(
            identity_state,
            target_result=_make_target_result(True),
            previously_authorized=False,
        )
        assert status is expected

    def test_perception_never_derives_lost_or_timeout_before_authorization(self) -> None:
        """Guards: TIMEOUT is controller-only; LOST requires a prior AUTHORIZED."""
        for identity_state in (None, *IdentityState):
            for has_target in (True, False):
                status = derive_authorization_status(
                    identity_state,
                    target_result=_make_target_result(has_target),
                    previously_authorized=False,
                )
                assert status not in (AuthorizationStatus.LOST, AuthorizationStatus.TIMEOUT)

    @pytest.mark.parametrize(
        ("identity_state", "has_target"),
        [
            (IdentityState.IDENTIFIED, True),
            (IdentityState.IDENTIFIED, False),
            (IdentityState.UNREGISTERED, True),
            (IdentityState.UNCERTAIN, True),
            (IdentityState.NEW, True),
            (IdentityState.LOST, True),
            (None, True),
            (None, False),
        ],
    )
    def test_post_authorization_frames_only_yield_authorized_or_lost(
        self, identity_state: IdentityState | None, has_target: bool
    ) -> None:
        """After authorization, only AUTHORIZED (verified) or LOST (loss) occur."""
        status = derive_authorization_status(
            identity_state,
            target_result=_make_target_result(has_target),
            previously_authorized=True,
        )
        if identity_state is IdentityState.IDENTIFIED and has_target:
            assert status is AuthorizationStatus.AUTHORIZED
        else:
            assert status is AuthorizationStatus.LOST

    def test_no_match_observation_never_creates_unknown_decision(self) -> None:
        """Documented actual behavior of RecognitionOutcome.NO_MATCH observations.

        IdentityStateManager.update() early-returns for NO_MATCH (empty gallery /
        no candidates), so NO_MATCH never feeds UNKNOWN; the track keeps its
        previous IdentityState (NEW here) and authorization stays ACTIVE.
        """
        manager = IdentityStateManager()
        state = manager.update(
            track_id=1,
            rec_result=_make_rec(RecognitionOutcome.NO_MATCH),
            frame_id=1,
        )
        assert state is not None
        assert state.state is IdentityState.NEW
        status = derive_authorization_status(
            state.state,
            target_result=_make_target_result(True),
            previously_authorized=False,
        )
        assert status is AuthorizationStatus.ACTIVE

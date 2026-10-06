"""Authorization domain model: session snapshot, status vocabulary, and perception mapping.

M3.3 Step 2 defines only the transport-neutral authorization domain layer consumed
by the (later) authorization controller. It follows the repository contract
conventions: frozen dataclasses, ``__post_init__`` validation raising
``backend.exceptions.ValidationError``, ISO-8601 UTC timestamps, opaque string ids,
and ``to_dict()`` payloads with zero internal vision/database structures.

Status mapping actually supported by the inspected perception types
-------------------------------------------------------------------
``derive_authorization_status()`` maps one frame's perception to a status using:

* ``dh360_hi.vision.target_selector.TargetSelectionResult`` -- ``primary_target`` /
  ``primary_track_id`` (the ``has_target`` property). ``NO_FACE`` is produced ONLY
  from this target/presence signal (no primary target selected). It is never derived
  from ``RecognitionOutcome``, which has no ``NO_FACE`` value.
* ``dh360_hi.vision.identity_state.IdentityState`` (state of the primary track):

  - ``NEW`` / ``COLLECTING`` -> ``ACTIVE``      no decision evidence yet
  - ``IDENTIFIED``           -> ``AUTHORIZED``  stable gallery match; the person
    identity is supplied by ``TrackIdentityState.current_person_id`` /
    ``current_display_name`` when the controller builds the session.
  - ``UNREGISTERED``         -> ``UNKNOWN``     recognized but matches no gallery
    entry (``RecognitionOutcome.UNREGISTERED`` below-threshold semantics, as
    documented on the ``IdentityState.UNREGISTERED`` enum member).
  - ``UNCERTAIN``            -> ``UNCERTAIN``   ambiguous/conflicting evidence.
  - ``LOST``                 -> ``ACTIVE``      the track expired BEFORE any
    authorization decision; ``LOST`` status is reserved for identity loss after a
    prior ``AUTHORIZED`` decision (presence/target semantics report ``NO_FACE``
    once the target is released).

* ``RecognitionOutcome.NO_MATCH`` (empty gallery / no valid candidates) does NOT
  feed the state machine: ``IdentityStateManager.update()`` early-returns for it
  (``rec_result.outcome == RecognitionOutcome.NO_MATCH`` in
  ``src/dh360_hi/vision/identity_state.py``), so it never directly produces
  ``UNKNOWN``; the track keeps its previous ``IdentityState``.

* ``previously_authorized=True`` (the session reached ``AUTHORIZED``): the session
  stays ``AUTHORIZED`` only while a primary target AND ``IdentityState.IDENTIFIED``
  both hold; every other frame (no target, state left ``IDENTIFIED``, missing
  identity record, track expired) is identity loss -> ``LOST``. This mirrors
  ``PersonLostIdentityEvent`` (a previously identified person whose state
  transitions away from ``IDENTIFIED``) and structurally guarantees ``LOST`` is
  only reachable after a prior ``AUTHORIZED`` decision.

``TIMEOUT`` is NEVER derived from perception: it is a controller/session-lifecycle
state produced only through ``AuthorizationSession.transition()`` / ``complete()``.

Privacy invariants (transport-neutral result): the snapshot exposes only
``authorization_id``, ``status``, ``person_id``, ``display_name``, and ISO
timestamps -- no images, embeddings, frame/track IDs, ``RecognitionResult``,
camera references, or database objects. ``person_id``/``display_name`` appear only
on ``AUTHORIZED``/``LOST``; ``display_name`` carries the gallery-resolved
preferred-name-first value (``backend/gallery_loader.py``: ``preferred_name or
display_name``) exactly as surfaced via ``TrackIdentityState.current_display_name``.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from dh360_hi.backend.exceptions import ValidationError
from dh360_hi.backend.models import utc_now_iso
from dh360_hi.vision.identity_state import IdentityState

if TYPE_CHECKING:
    from dh360_hi.vision.target_selector import TargetSelectionResult


class AuthorizationStatus(StrEnum):
    """Externally meaningful authorization statuses for one session."""

    ACTIVE = "active"  # Running; no decision yet (or pre-decision observations)
    AUTHORIZED = "authorized"  # Identity verified against a registered gallery person
    UNKNOWN = "unknown"  # Established as not registered / no gallery match
    UNCERTAIN = "uncertain"  # Ambiguous or conflicting identity evidence
    NO_FACE = "no_face"  # No primary target selected (target/presence semantics)
    LOST = "lost"  # Identity lost AFTER a prior AUTHORIZED decision
    TIMEOUT = "timeout"  # Controller lifecycle outcome; never perception-derived


def _coerce_status(status: AuthorizationStatus | str) -> AuthorizationStatus:
    """Convert a wire-level string into AuthorizationStatus, or raise ValidationError."""
    if isinstance(status, AuthorizationStatus):
        return status
    try:
        return AuthorizationStatus(status)
    except (ValueError, TypeError) as exc:
        msg = f"unknown authorization status: {status!r}"
        raise ValidationError(msg) from exc


@dataclass(frozen=True)
class AuthorizationSession:
    """Transport-neutral snapshot of one authorization session (state + result).

    State combinations enforced by ``__post_init__``:

    * ``AUTHORIZED`` and ``LOST`` require a non-empty ``person_id`` (an authorized
      identity must exist -- this structurally encodes that ``LOST`` represents
      identity loss after a prior ``AUTHORIZED`` decision); every other status must
      carry no identity claim at all.
    * ``display_name`` is optional on ``AUTHORIZED``/``LOST``; blank values are
      rejected (``ValidationError``), matching ``CreatePersonCommand`` conventions.
    * ``ACTIVE`` never has ``ended_at_iso``; a non-None ``ended_at_iso`` marks a
      completed session (``is_completed``) while ``is_active`` means "still running".

    ``transition()`` is the sanctioned path for status evolution and additionally
    enforces that ``LOST`` is only entered from ``AUTHORIZED`` and that an
    ``AUTHORIZED`` session may only remain ``AUTHORIZED``, move to ``LOST``/
    ``TIMEOUT``, or be completed. Direct construction validates only the snapshot
    it can see (repository convention for frozen contract dataclasses).
    """

    authorization_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: AuthorizationStatus = AuthorizationStatus.ACTIVE
    person_id: str | None = None
    display_name: str | None = None
    created_at_iso: str = field(default_factory=utc_now_iso)
    updated_at_iso: str = field(default_factory=utc_now_iso)
    ended_at_iso: str | None = None

    def __post_init__(self) -> None:
        # Accept plain wire strings (transport payloads) by coercing into the enum.
        coerced = _coerce_status(self.status)
        if coerced is not self.status:
            object.__setattr__(self, "status", coerced)

        if not self.authorization_id or not self.authorization_id.strip():
            msg = "authorization_id cannot be empty or whitespace"
            raise ValidationError(msg)

        if self.ended_at_iso is not None and not self.ended_at_iso.strip():
            msg = "ended_at_iso cannot be blank; use None while the session is running"
            raise ValidationError(msg)
        if self.status is AuthorizationStatus.ACTIVE and self.ended_at_iso is not None:
            msg = "ACTIVE authorization session cannot carry ended_at_iso"
            raise ValidationError(msg)

        if self.status in (AuthorizationStatus.AUTHORIZED, AuthorizationStatus.LOST):
            if not self.person_id or not self.person_id.strip():
                msg = f"{self.status.value} authorization session requires a non-empty person_id"
                raise ValidationError(msg)
            if self.display_name is not None and not self.display_name.strip():
                msg = "display_name cannot be empty or whitespace"
                raise ValidationError(msg)
        else:
            # ACTIVE / UNKNOWN / UNCERTAIN / NO_FACE / TIMEOUT claim no identity.
            if self.person_id is not None:
                msg = f"{self.status.value} authorization session cannot carry person_id"
                raise ValidationError(msg)
            if self.display_name is not None:
                msg = f"{self.status.value} authorization session cannot carry display_name"
                raise ValidationError(msg)

    @property
    def is_active(self) -> bool:
        """True while the session is still running (no completion timestamp yet)."""
        return self.ended_at_iso is None

    @property
    def is_completed(self) -> bool:
        """True once the session has been completed with a final decision."""
        return self.ended_at_iso is not None

    @property
    def is_authorized(self) -> bool:
        """True when the current status grants authorization."""
        return self.status is AuthorizationStatus.AUTHORIZED

    def transition(
        self,
        status: AuthorizationStatus | str,
        *,
        person_id: str | None = None,
        display_name: str | None = None,
        at_iso: str | None = None,
    ) -> AuthorizationSession:
        """Return an updated running snapshot; rejects invalid state combinations.

        Rules (beyond constructor validation):

        * Completed sessions cannot transition.
        * ``LOST`` may only be entered while the current status is ``AUTHORIZED``
          (identity loss after a prior authorization decision); the authorized
          ``person_id``/``display_name`` are inherited unless explicitly replaced.
        * An ``AUTHORIZED`` session may only remain ``AUTHORIZED``, move to
          ``LOST``/``TIMEOUT`` (observation statuses are pre-authorization only);
          ``TIMEOUT`` drops the identity claim (controller lifecycle outcome).
        """
        if self.is_completed:
            msg = "completed authorization sessions cannot transition"
            raise ValidationError(msg)

        target_status = _coerce_status(status)

        if (
            target_status is AuthorizationStatus.LOST
            and self.status is not AuthorizationStatus.AUTHORIZED
        ):
            msg = "LOST requires current status AUTHORIZED (identity loss after authorization)"
            raise ValidationError(msg)

        if self.status is AuthorizationStatus.AUTHORIZED and target_status not in (
            AuthorizationStatus.AUTHORIZED,
            AuthorizationStatus.LOST,
            AuthorizationStatus.TIMEOUT,
        ):
            msg = "AUTHORIZED session may only stay AUTHORIZED, move to LOST/TIMEOUT, or complete"
            raise ValidationError(msg)

        if target_status in (AuthorizationStatus.AUTHORIZED, AuthorizationStatus.LOST):
            # Inherit the authorized identity when the caller does not replace it;
            # a replacement person_id must bring its own display_name (or none).
            if person_id is None and display_name is None:
                person_id = self.person_id
                display_name = self.display_name
            elif person_id is None:
                person_id = self.person_id

        return AuthorizationSession(
            authorization_id=self.authorization_id,
            status=target_status,
            person_id=person_id,
            display_name=display_name,
            created_at_iso=self.created_at_iso,
            updated_at_iso=at_iso or utc_now_iso(),
            ended_at_iso=None,
        )

    def complete(
        self,
        status: AuthorizationStatus | str | None = None,
        *,
        person_id: str | None = None,
        display_name: str | None = None,
        at_iso: str | None = None,
    ) -> AuthorizationSession:
        """Return the completed snapshot with a final (non-ACTIVE) decision status.

        Defaults to completing with the current status (e.g. a running
        ``AUTHORIZED`` session is completed as ``AUTHORIZED``). ``ACTIVE`` is a
        running state and can never be a completion status.
        """
        timestamp = at_iso or utc_now_iso()
        target_status = self.status if status is None else _coerce_status(status)
        if target_status is AuthorizationStatus.ACTIVE:
            msg = "session cannot be completed with ACTIVE; a final decision is required"
            raise ValidationError(msg)
        settled = self.transition(
            target_status,
            person_id=person_id,
            display_name=display_name,
            at_iso=timestamp,
        )
        return replace(settled, ended_at_iso=timestamp)

    def to_dict(self) -> dict[str, Any]:
        """Convert the snapshot to a JSON-serializable dictionary (transport shape)."""
        return {
            "authorization_id": self.authorization_id,
            "status": self.status.value,
            "person_id": self.person_id,
            "display_name": self.display_name,
            "created_at_iso": self.created_at_iso,
            "updated_at_iso": self.updated_at_iso,
            "ended_at_iso": self.ended_at_iso,
        }


def derive_authorization_status(
    identity_state: IdentityState | None,
    *,
    target_result: TargetSelectionResult,
    previously_authorized: bool,
) -> AuthorizationStatus:
    """Map one frame's perception to an authorization status.

    Pure, transport-neutral mapping documented against the actual types (see the
    module docstring for the full table and citations):

    1. ``previously_authorized`` (the session reached ``AUTHORIZED``): ``AUTHORIZED``
       only while ``target_result.has_target`` holds AND the primary track's
       ``IdentityState`` is ``IDENTIFIED``; every other frame is ``LOST`` (identity
       loss after prior authorization -- mirrors ``PersonLostIdentityEvent``).
       Pass ``previously_authorized=session.status is AuthorizationStatus.AUTHORIZED``:
       an ``AUTHORIZED`` session can only remain ``AUTHORIZED``, move to
       ``LOST``/``TIMEOUT``, or be completed, so the running status encodes the
       authorization history.
    2. Otherwise, a missing primary target (``target_result.has_target`` is False /
       ``primary_track_id`` is None) yields ``NO_FACE`` -- target/presence semantics
       only, never ``RecognitionOutcome`` (which has no such value).
    3. No identity record for the primary track yet -> ``ACTIVE``.
    4. Pre-authorization ``IdentityState`` mapping: ``IDENTIFIED`` -> ``AUTHORIZED``,
       ``UNREGISTERED`` -> ``UNKNOWN``, ``UNCERTAIN`` -> ``UNCERTAIN``,
       ``NEW``/``COLLECTING``/``LOST`` -> ``ACTIVE`` (no decision yet; track expiry
       before any authorization is not identity loss -- presence semantics will
       report ``NO_FACE`` once the target is released).

    ``TIMEOUT`` is never produced here; it is a controller lifecycle state.
    """
    if previously_authorized:
        if target_result.has_target and identity_state is IdentityState.IDENTIFIED:
            return AuthorizationStatus.AUTHORIZED
        return AuthorizationStatus.LOST

    if not target_result.has_target:
        return AuthorizationStatus.NO_FACE

    if identity_state is None:
        return AuthorizationStatus.ACTIVE

    if identity_state is IdentityState.IDENTIFIED:
        return AuthorizationStatus.AUTHORIZED
    if identity_state is IdentityState.UNREGISTERED:
        return AuthorizationStatus.UNKNOWN
    if identity_state is IdentityState.UNCERTAIN:
        return AuthorizationStatus.UNCERTAIN
    # IdentityState.NEW / COLLECTING: no decision evidence yet.
    # IdentityState.LOST without prior authorization: the track expired before any
    # authorization decision, so the LOST status (post-authorization identity loss)
    # does not apply.
    return AuthorizationStatus.ACTIVE

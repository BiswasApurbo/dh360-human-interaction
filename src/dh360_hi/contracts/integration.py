"""Transport-neutral integration events for external application subscribers.

Integration events cross the boundary between the Human Interaction subsystem and
external orchestrators / consumers (REST API, WebSockets, dialogue managers, UI).

Privacy & Decoupling Invariants:
1. Internal vision tracker IDs (`track_id`) and frame sequence numbers (`frame_id`)
   are strictly STRIPPED from all integration events.
2. Raw biometric data (embeddings, bounding boxes, landmarks, crop arrays) is never exposed.
3. Timestamps are converted from float epoch seconds to standard ISO 8601 UTC strings.
4. All integration events are frozen dataclasses supporting `.to_dict()`.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from dh360_hi.backend.models import utc_now_iso
from dh360_hi.contracts.dto import (
    InteractionContextDTO,
    InteractionSummaryDTO,
    epoch_to_iso,
)
from dh360_hi.contracts.events import (
    BaseEvent,
    PersonIdentifiedEvent,
    PersonLostIdentityEvent,
    SessionEndedEvent,
    SessionStartedEvent,
)


@dataclass(frozen=True)
class BaseIntegrationEvent:
    """Base class for all transport-neutral integration events."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = "base"
    timestamp_iso: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        """Convert integration event to a JSON-serializable dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class SessionStartedIntegrationEvent(BaseIntegrationEvent):
    """Emitted externally when an interaction session begins."""

    event_type: str = "session_started"
    session_id: str = ""
    person_id: str | None = None
    display_name: str | None = None


@dataclass(frozen=True)
class PersonIdentifiedIntegrationEvent(BaseIntegrationEvent):
    """Emitted externally when a subject is positively identified."""

    event_type: str = "person_identified"
    session_id: str = ""
    person_id: str = ""
    display_name: str | None = None


@dataclass(frozen=True)
class PersonLostIdentityIntegrationEvent(BaseIntegrationEvent):
    """Emitted externally when a previously identified subject loses identity."""

    event_type: str = "person_lost_identity"
    session_id: str = ""
    previous_person_id: str | None = None


@dataclass(frozen=True)
class SessionEndedIntegrationEvent(BaseIntegrationEvent):
    """Emitted externally when an interaction session terminates."""

    event_type: str = "session_ended"
    session_id: str = ""
    final_person_id: str | None = None
    final_display_name: str | None = None
    summary: InteractionSummaryDTO | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert integration event to a dictionary with nested summary DTO."""
        data = asdict(self)
        if self.summary is not None:
            data["summary"] = self.summary.to_dict()
        return data


@dataclass(frozen=True)
class InteractionContextUpdatedIntegrationEvent(BaseIntegrationEvent):
    """Emitted externally whenever the active interaction context state changes."""

    event_type: str = "context_updated"
    context: InteractionContextDTO = field(default_factory=InteractionContextDTO)

    def to_dict(self) -> dict[str, Any]:
        """Convert integration event to a dictionary with nested context DTO."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "timestamp_iso": self.timestamp_iso,
            "context": self.context.to_dict(),
        }


@dataclass(frozen=True)
class AuthorizationUpdatedIntegrationEvent(BaseIntegrationEvent):
    """Emitted externally when an authorization session state transition occurs."""

    event_type: str = "authorization_updated"
    authorization_id: str = ""
    status: str = "active"
    person_id: str | None = None
    display_name: str | None = None
    started_at: str | None = None
    ended_at: str | None = None
    updated_at: str | None = None
    is_active: bool = True
    is_completed: bool = False
    is_authorized: bool = False

    @classmethod
    def from_session(cls, session: Any) -> AuthorizationUpdatedIntegrationEvent:
        """Create an event from an AuthorizationSession instance."""
        status_str = (
            session.status.value if hasattr(session.status, "value") else str(session.status)
        )
        return cls(
            authorization_id=session.authorization_id,
            status=status_str,
            person_id=session.person_id,
            display_name=session.display_name,
            started_at=getattr(session, "created_at_iso", None),
            ended_at=getattr(session, "ended_at_iso", None),
            updated_at=getattr(session, "updated_at_iso", None),
            is_active=session.is_active,
            is_completed=session.is_completed,
            is_authorized=session.is_authorized,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert integration event to a JSON-serializable dictionary."""
        return asdict(self)


def map_domain_event_to_integration(
    event: BaseEvent,
    summary: InteractionSummaryDTO | None = None,
) -> BaseIntegrationEvent | None:
    """Transform an internal domain event into a sanitized external integration event.

    Returns `None` for domain events that are purely internal to the perception
    pipeline (e.g. `PersonEnteredEvent`, `PersonLeftEvent`).
    """
    ts_iso = epoch_to_iso(event.timestamp) or utc_now_iso()

    if isinstance(event, SessionStartedEvent):
        return SessionStartedIntegrationEvent(
            event_id=event.event_id,
            timestamp_iso=ts_iso,
            session_id=event.session_id,
            person_id=event.person_id,
            display_name=event.display_name,
        )
    if isinstance(event, PersonIdentifiedEvent):
        return PersonIdentifiedIntegrationEvent(
            event_id=event.event_id,
            timestamp_iso=ts_iso,
            session_id=event.session_id,
            person_id=event.person_id,
            display_name=event.display_name,
        )
    if isinstance(event, PersonLostIdentityEvent):
        return PersonLostIdentityIntegrationEvent(
            event_id=event.event_id,
            timestamp_iso=ts_iso,
            session_id=event.session_id,
            previous_person_id=event.previous_person_id,
        )
    if isinstance(event, SessionEndedEvent):
        return SessionEndedIntegrationEvent(
            event_id=event.event_id,
            timestamp_iso=ts_iso,
            session_id=event.session_id,
            final_person_id=event.final_person_id,
            final_display_name=event.final_display_name,
            summary=summary,
        )

    return None

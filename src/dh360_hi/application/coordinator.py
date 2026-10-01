"""Application-level interaction context coordinator and event bridge.

Translates low-level edge session and identity lifecycle events into a stable,
queryable application interaction context, enriches identified individuals with
PersonProfile metadata via PersonService, and generates InteractionSummary records.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from dh360_hi.contracts.events import (
    PersonEnteredEvent,
    PersonIdentifiedEvent,
    PersonLeftEvent,
    PersonLostIdentityEvent,
    SessionEndedEvent,
    SessionStartedEvent,
)
from dh360_hi.contracts.interaction import (
    InteractionContext,
    InteractionState,
    InteractionSummary,
)
from dh360_hi.edge.event_bus import EventBus

if TYPE_CHECKING:
    from dh360_hi.backend.models import PersonProfile
    from dh360_hi.backend.person_service import BasePersonService
    from dh360_hi.backend.session_repository import (
        BaseSessionRepository,
        SessionPersistenceWorker,
    )

logger = logging.getLogger(__name__)


class InteractionCoordinator:
    """Application-level interaction coordinator and context aggregator.

    Supervises application interaction context without reproducing perception,
    target selection, or session lifecycle state machine logic.

    Responsibilities:
        - Subscribes to EventBus for session and identity domain events.
        - Enriches identified individuals with PersonProfile metadata via PersonService.
        - Maintains thread-safe, immutable InteractionContext snapshots.
        - Generates finalized InteractionSummary records on session termination.
        - Persists completed interaction sessions to repository storage.
        - Provides clean listener callbacks for downstream application adapters.
    """

    def __init__(
        self,
        event_bus: EventBus,
        person_service: BasePersonService | None = None,
        session_repository: BaseSessionRepository | None = None,
        persistence_worker: SessionPersistenceWorker | None = None,
    ) -> None:
        self._event_bus = event_bus
        self._person_service = person_service
        self._session_repository = session_repository
        self._persistence_worker = persistence_worker
        self._lock = threading.RLock()

        # Active interaction state
        self._context = InteractionContext(last_updated_at=time.time())
        self._last_summary: InteractionSummary | None = None
        self._identity_was_known = False

        # Listener callbacks
        self._listeners: dict[int, Callable[[InteractionContext], None]] = {}
        self._next_listener_id = 1

        # Subscription IDs on EventBus
        self._subscription_ids: list[int] = []
        self._subscribe_events()

    # ------------------------------------------------------------------
    # Properties & Query API
    # ------------------------------------------------------------------

    @property
    def session_repository(self) -> BaseSessionRepository | None:
        """Configured session repository, if any."""
        return self._session_repository

    @property
    def persistence_worker(self) -> SessionPersistenceWorker | None:
        """Configured session persistence worker, if any."""
        return self._persistence_worker

    @property
    def context(self) -> InteractionContext:
        """Current thread-safe immutable interaction context snapshot."""
        with self._lock:
            return self._context

    @property
    def last_summary(self) -> InteractionSummary | None:
        """Most recently completed interaction session summary, if any."""
        with self._lock:
            return self._last_summary

    def get_context(self) -> InteractionContext:
        """Retrieve current immutable interaction context snapshot."""
        return self.context

    def get_last_summary(self) -> InteractionSummary | None:
        """Retrieve most recently completed interaction session summary."""
        return self.last_summary

    # ------------------------------------------------------------------
    # Listener Registration
    # ------------------------------------------------------------------

    def add_context_listener(self, callback: Callable[[InteractionContext], None]) -> int:
        """Register a callback for context state transition notifications.

        Args:
            callback: Function invoked with the new immutable InteractionContext.

        Returns:
            Unique listener ID for subsequent unregistration.
        """
        with self._lock:
            lid = self._next_listener_id
            self._next_listener_id += 1
            self._listeners[lid] = callback
            return lid

    def remove_context_listener(self, listener_id: int) -> bool:
        """Unregister a previously registered context listener.

        Args:
            listener_id: Listener ID returned by add_context_listener.

        Returns:
            True if removed, False if listener_id was not found.
        """
        with self._lock:
            return self._listeners.pop(listener_id, None) is not None

    # ------------------------------------------------------------------
    # Lifecycle Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Unsubscribe from EventBus and release listeners. Idempotent."""
        with self._lock:
            for sub_id in self._subscription_ids:
                try:
                    self._event_bus.unsubscribe(sub_id)
                except Exception:
                    logger.exception("Error unsubscribing handler %d from EventBus", sub_id)
            self._subscription_ids.clear()
            self._listeners.clear()

    # ------------------------------------------------------------------
    # Event Subscriptions & Handlers
    # ------------------------------------------------------------------

    def _subscribe_events(self) -> None:
        """Subscribe to relevant domain events on the EventBus."""
        self._subscription_ids.append(
            self._event_bus.subscribe(PersonEnteredEvent, self._on_person_entered)
        )
        self._subscription_ids.append(
            self._event_bus.subscribe(SessionStartedEvent, self._on_session_started)
        )
        self._subscription_ids.append(
            self._event_bus.subscribe(PersonIdentifiedEvent, self._on_person_identified)
        )
        self._subscription_ids.append(
            self._event_bus.subscribe(PersonLostIdentityEvent, self._on_person_lost_identity)
        )
        self._subscription_ids.append(
            self._event_bus.subscribe(PersonLeftEvent, self._on_person_left)
        )
        self._subscription_ids.append(
            self._event_bus.subscribe(SessionEndedEvent, self._on_session_ended)
        )

    def _resolve_profile(self, person_id: str | None) -> PersonProfile | None:
        """Safely lookup public PersonProfile without leaking DB exceptions."""
        if person_id is None or self._person_service is None:
            return None
        try:
            return self._person_service.get_person(person_id)
        except Exception:
            logger.exception("Failed to resolve PersonProfile for person_id: %s", person_id)
            return None

    def _on_person_entered(self, event: PersonEnteredEvent) -> None:
        """Handle presence entrance notification."""
        with self._lock:
            if self._context.state == InteractionState.IDLE:
                self._context = InteractionContext(
                    session_id=None,
                    person_id=event.person_id,
                    display_name=event.display_name,
                    person_profile=None,
                    state=InteractionState.IDLE,
                    started_at=None,
                    last_updated_at=event.timestamp,
                    identity_known=bool(event.person_id),
                    last_known_person_id=event.person_id,
                    identity_transitions_count=0,
                )

    def _on_session_started(self, event: SessionStartedEvent) -> None:
        """Handle authoritative session start event."""
        profile = self._resolve_profile(event.person_id)
        display_name = event.display_name
        if profile is not None:
            display_name = profile.preferred_name or profile.display_name

        identity_known = bool(event.person_id)
        with self._lock:
            self._identity_was_known = identity_known
            self._context = InteractionContext(
                session_id=event.session_id,
                person_id=event.person_id,
                display_name=display_name,
                person_profile=profile,
                state=InteractionState.GREETING,
                started_at=event.timestamp,
                last_updated_at=event.timestamp,
                identity_known=identity_known,
                last_known_person_id=event.person_id,
                identity_transitions_count=1 if identity_known else 0,
            )
            snapshot = self._context

        self._notify_listeners(snapshot)

    def _on_person_identified(self, event: PersonIdentifiedEvent) -> None:
        """Handle identity recognition or confirmation during active session."""
        profile = self._resolve_profile(event.person_id)
        display_name = event.display_name
        if profile is not None:
            display_name = profile.preferred_name or profile.display_name

        with self._lock:
            # Only apply if active session exists
            if self._context.session_id is None:
                return

            self._identity_was_known = True
            new_state = (
                InteractionState.ENGAGED
                if self._context.state in (InteractionState.GREETING, InteractionState.ENGAGED)
                else self._context.state
            )
            self._context = InteractionContext(
                session_id=self._context.session_id,
                person_id=event.person_id,
                display_name=display_name,
                person_profile=profile or self._context.person_profile,
                state=new_state,
                started_at=self._context.started_at,
                last_updated_at=event.timestamp,
                identity_known=True,
                last_known_person_id=event.person_id,
                identity_transitions_count=self._context.identity_transitions_count + 1,
            )
            snapshot = self._context

        self._notify_listeners(snapshot)

    def _on_person_lost_identity(self, event: PersonLostIdentityEvent) -> None:
        """Handle identity degradation while preserving active interaction session."""
        with self._lock:
            if self._context.session_id is None:
                return

            last_known = (
                event.previous_person_id
                or self._context.person_id
                or self._context.last_known_person_id
            )
            self._context = InteractionContext(
                session_id=self._context.session_id,
                person_id=None,
                display_name=None,
                person_profile=None,
                state=self._context.state,
                started_at=self._context.started_at,
                last_updated_at=event.timestamp,
                identity_known=False,
                last_known_person_id=last_known,
                identity_transitions_count=self._context.identity_transitions_count + 1,
            )
            snapshot = self._context

        self._notify_listeners(snapshot)

    def _on_person_left(self, event: PersonLeftEvent) -> None:
        """Handle physical departure of active subject."""
        with self._lock:
            if self._context.session_id is not None:
                self._context = InteractionContext(
                    session_id=self._context.session_id,
                    person_id=self._context.person_id,
                    display_name=self._context.display_name,
                    person_profile=self._context.person_profile,
                    state=InteractionState.FAREWELL,
                    started_at=self._context.started_at,
                    last_updated_at=event.timestamp,
                    identity_known=self._context.identity_known,
                    last_known_person_id=self._context.last_known_person_id,
                    identity_transitions_count=self._context.identity_transitions_count,
                )
                snapshot = self._context
            else:
                snapshot = None

        if snapshot is not None:
            self._notify_listeners(snapshot)

    def _on_session_ended(self, event: SessionEndedEvent) -> None:
        """Handle authoritative session termination and generate summary."""
        with self._lock:
            started_at = (
                self._context.started_at
                if self._context.started_at is not None
                else event.timestamp
            )
            ended_at = event.timestamp
            duration = max(0.0, ended_at - started_at)

            final_person_id = self._context.person_id or event.final_person_id
            final_display_name = self._context.display_name or event.final_display_name
            final_profile = self._context.person_profile

            summary = InteractionSummary(
                session_id=event.session_id,
                person_id=final_person_id,
                display_name=final_display_name,
                person_profile=final_profile,
                started_at=started_at,
                ended_at=ended_at,
                duration_seconds=duration,
                final_state=self._context.state,
                identity_was_known=self._identity_was_known or bool(final_person_id),
                identity_transitions_count=self._context.identity_transitions_count,
            )
            self._last_summary = summary
            self._identity_was_known = False

            # Reset context back to IDLE
            self._context = InteractionContext(
                session_id=None,
                person_id=None,
                display_name=None,
                person_profile=None,
                state=InteractionState.IDLE,
                started_at=None,
                last_updated_at=event.timestamp,
                identity_known=False,
                last_known_person_id=None,
                identity_transitions_count=0,
            )
            snapshot = self._context

        # Asynchronously offload session persistence or fallback to direct repository
        if self._persistence_worker is not None:
            self._persistence_worker.submit(summary)
        elif self._session_repository is not None:
            try:
                from dh360_hi.backend.models import SessionRecord

                record = SessionRecord.from_summary(summary)
                self._session_repository.save_session(record)
            except Exception:
                logger.exception("Failed to persist session %s directly", summary.session_id)

        self._notify_listeners(snapshot)

    def _notify_listeners(self, snapshot: InteractionContext) -> None:
        """Deliver context updates to registered listeners with exception isolation."""
        with self._lock:
            callbacks = list(self._listeners.values())

        for cb in callbacks:
            try:
                cb(snapshot)
            except Exception:
                logger.exception("Error executing InteractionContext listener callback")

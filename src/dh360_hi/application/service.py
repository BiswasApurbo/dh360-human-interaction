"""Unified human interaction application service facade.

Provides a thin, typed, in-process application facade coordinating edge perception
runtime, identity recognition, session management, and context aggregation without
duplicating underlying domain logic.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

from dh360_hi.application.config import HumanInteractionConfig
from dh360_hi.application.coordinator import InteractionCoordinator
from dh360_hi.backend.gallery_loader import load_gallery_from_repository
from dh360_hi.contracts.events import BaseEvent
from dh360_hi.contracts.interaction import InteractionContext, InteractionSummary
from dh360_hi.edge.event_bus import EventBus
from dh360_hi.edge.runtime import EdgeInteractionRuntime, RuntimeMetrics, RuntimeState

if TYPE_CHECKING:
    from dh360_hi.backend.database import DatabaseManager
    from dh360_hi.backend.models import (
        ConsentStatus,
        PersonEngagementStats,
        PersonProfile,
        PersonStatus,
        SessionAggregateAnalytics,
        SessionQueryFilter,
        SessionRecord,
    )
    from dh360_hi.backend.person_service import BasePersonService
    from dh360_hi.backend.repository import BasePersonRepository
    from dh360_hi.backend.session_repository import (
        BaseSessionRepository,
        SessionPersistenceWorker,
    )
    from dh360_hi.vision.enrollment import FaceEnrollmentController
    from dh360_hi.vision.recognition import BaseFaceRecognizer

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseEvent)


class HumanInteractionService:
    """Unified application service facade for the Human Interaction subsystem.

    Coordinates lifecycle management, context queries, person profiles, historical
    session persistence, and recognition gallery synchronization across subsystem components.

    Invariants:
        - Contains zero raw computer vision, tensor manipulation, or SQL execution.
        - Exposes strictly application-safe contracts without leaking internal biometric
          embeddings or image arrays.
        - Delegates all domain responsibilities to specialized underlying services.
    """

    def __init__(
        self,
        runtime: EdgeInteractionRuntime,
        coordinator: InteractionCoordinator,
        person_service: BasePersonService,
        repository: BasePersonRepository,
        recognizer: BaseFaceRecognizer,
        db_manager: DatabaseManager,
        event_bus: EventBus,
        session_repository: BaseSessionRepository | None = None,
        persistence_worker: SessionPersistenceWorker | None = None,
        enrollment_controller: FaceEnrollmentController | None = None,
        config: HumanInteractionConfig | None = None,
    ) -> None:
        self._runtime = runtime
        self._coordinator = coordinator
        self._person_service = person_service
        self._repository = repository
        self._recognizer = recognizer
        self._db_manager = db_manager
        self._event_bus = event_bus
        self._session_repository = session_repository
        self._persistence_worker = persistence_worker
        self._enrollment_controller = enrollment_controller
        self._config = config or HumanInteractionConfig()
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> HumanInteractionConfig:
        """Configured application system settings."""
        return self._config

    @property
    def is_running(self) -> bool:
        """Whether the edge interaction runtime is actively running."""
        return self._runtime.state == RuntimeState.RUNNING

    @property
    def state(self) -> RuntimeState:
        """Current operational lifecycle state of the edge runtime."""
        return self._runtime.state

    @property
    def context(self) -> InteractionContext:
        """Current immutable interaction context snapshot."""
        return self._coordinator.get_context()

    @property
    def metrics(self) -> RuntimeMetrics:
        """Current operational telemetry metrics."""
        return self._runtime.metrics

    @property
    def runtime(self) -> EdgeInteractionRuntime:
        """Composed edge interaction runtime instance."""
        return self._runtime

    @property
    def coordinator(self) -> InteractionCoordinator:
        """Composed interaction context coordinator instance."""
        return self._coordinator

    @property
    def person_service(self) -> BasePersonService:
        """Composed person service instance."""
        return self._person_service

    @property
    def session_repository(self) -> BaseSessionRepository | None:
        """Composed session repository instance, if configured."""
        return self._session_repository

    @property
    def persistence_worker(self) -> SessionPersistenceWorker | None:
        """Composed session persistence worker instance, if configured."""
        return self._persistence_worker

    # ------------------------------------------------------------------
    # Interaction Context Queries
    # ------------------------------------------------------------------

    def get_interaction_context(self) -> InteractionContext:
        """Retrieve the current thread-safe immutable interaction context snapshot."""
        return self._coordinator.get_context()

    def get_last_summary(self) -> InteractionSummary | None:
        """Retrieve the summary record of the most recently completed session, if any."""
        return self._coordinator.get_last_summary()

    def get_runtime_metrics(self) -> RuntimeMetrics:
        """Retrieve current operational telemetry metrics."""
        return self._runtime.metrics

    def get_runtime_state(self) -> RuntimeState:
        """Retrieve current runtime execution state."""
        return self._runtime.state

    # ------------------------------------------------------------------
    # Person & Profile Queries
    # ------------------------------------------------------------------

    def get_person(self, person_id: str) -> PersonProfile | None:
        """Retrieve a public person profile by UUID identifier.

        Args:
            person_id: UUID of the target person.

        Returns:
            PersonProfile if found and registered, otherwise None.
        """
        return self._person_service.get_person(person_id)

    def list_persons(
        self,
        status: PersonStatus | None = None,
        consent_status: ConsentStatus | None = None,
    ) -> list[PersonProfile]:
        """List registered person profiles with optional filtering.

        Args:
            status: Optional PersonStatus filter (e.g. ACTIVE).
            consent_status: Optional ConsentStatus filter (e.g. GRANTED).

        Returns:
            List of matching PersonProfile records.
        """
        people = self._person_service.list_people(status=status)
        if consent_status is not None:
            people = [p for p in people if p.consent_status == consent_status]
        return people

    # ------------------------------------------------------------------
    # Historical Session Queries & Persistence
    # ------------------------------------------------------------------

    def get_session(self, session_id: str) -> SessionRecord | None:
        """Retrieve an interaction session record by UUID identifier.

        Args:
            session_id: UUID of the target session.

        Returns:
            SessionRecord if found in persistence, otherwise None.
        """
        if self._session_repository is None:
            return None
        return self._session_repository.get_session(session_id)

    def list_sessions(
        self,
        person_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[SessionRecord]:
        """List historical interaction sessions with optional person filtering and pagination.

        Args:
            person_id: Optional UUID filter for identified subject.
            limit: Maximum number of records to return (must be > 0).
            offset: Number of records to skip for pagination (must be >= 0).

        Returns:
            List of matching SessionRecord entries ordered chronologically descending.
        """
        if self._session_repository is None:
            return []
        return self._session_repository.list_sessions(
            person_id=person_id,
            limit=limit,
            offset=offset,
        )

    def count_sessions(self, person_id: str | None = None) -> int:
        """Count total historical interaction sessions, optionally filtered by person.

        Args:
            person_id: Optional UUID filter for identified subject.

        Returns:
            Total number of matching session records.
        """
        if self._session_repository is None:
            return 0
        return self._session_repository.count_sessions(person_id=person_id)

    def query_sessions(
        self,
        filter: SessionQueryFilter | None = None,
    ) -> list[SessionRecord]:
        """Query historical interaction sessions with rich filtering and pagination.

        Args:
            filter: Filter criteria model (defaults to default SessionQueryFilter()).

        Returns:
            List of matching SessionRecord entries ordered chronologically descending.
        """
        if self._session_repository is None:
            return []
        from dh360_hi.backend.models import SessionQueryFilter

        criteria = filter if filter is not None else SessionQueryFilter()
        return self._session_repository.query_sessions(criteria)

    def count_query_sessions(
        self,
        filter: SessionQueryFilter | None = None,
    ) -> int:
        """Count historical interaction sessions matching structured criteria.

        Args:
            filter: Filter criteria model (defaults to default SessionQueryFilter()).

        Returns:
            Total count of matching session records.
        """
        if self._session_repository is None:
            return 0
        from dh360_hi.backend.models import SessionQueryFilter

        criteria = filter if filter is not None else SessionQueryFilter()
        return self._session_repository.count_query_sessions(criteria)

    def get_aggregate_analytics(
        self,
        filter: SessionQueryFilter | None = None,
    ) -> SessionAggregateAnalytics:
        """Compute aggregate statistics directly in SQLite across filtered sessions.

        Args:
            filter: Optional filter criteria to restrict the analytical scope.

        Returns:
            SessionAggregateAnalytics model with aggregate metrics.
        """
        if self._session_repository is None:
            from dh360_hi.backend.models import SessionAggregateAnalytics

            return SessionAggregateAnalytics(
                total_sessions=0,
                total_duration_seconds=0.0,
                average_duration_seconds=0.0,
                min_duration_seconds=0.0,
                max_duration_seconds=0.0,
                identified_sessions_count=0,
                anonymous_sessions_count=0,
                earliest_session_started_at=None,
                latest_session_ended_at=None,
            )
        return self._session_repository.get_aggregate_analytics(filter=filter)

    def get_person_engagement_stats(
        self,
        person_id: str,
    ) -> PersonEngagementStats | None:
        """Retrieve historical interaction metrics for a specific person.

        Args:
            person_id: UUID of the target person.

        Returns:
            PersonEngagementStats if sessions exist for this person, otherwise None.
        """
        if self._session_repository is None:
            return None
        return self._session_repository.get_person_engagement_stats(person_id)

    def flush_session_persistence(self, timeout: float = 2.0) -> bool:
        """Flush pending session persistence writes to storage.

        Args:
            timeout: Maximum seconds to wait.

        Returns:
            True if all pending sessions were successfully written before timeout.
        """
        if self._persistence_worker is not None:
            return self._persistence_worker.flush(timeout=timeout)
        return True

    # ------------------------------------------------------------------
    # Lifecycle Commands
    # ------------------------------------------------------------------

    def start(self, async_thread: bool = True) -> None:
        """Start the continuous edge interaction runtime frame pump and persistence worker.

        Args:
            async_thread: If True (default), runs continuous processing in a background
                thread. If False, runs synchronously on the caller thread.
        """
        with self._lock:
            if self._persistence_worker is not None:
                self._persistence_worker.start()
            self._runtime.start(async_thread=async_thread)

    def stop(self, timeout: float = 3.0) -> None:
        """Gracefully stop the runtime, terminate background workers, and release resources.

        Enforces deterministic shutdown ordering:
            1. Stop runtime frame pump and embedding worker threads.
            2. Drain and stop background session persistence worker.
            3. Unsubscribe InteractionCoordinator from EventBus.
            4. Close DatabaseManager SQLite connection.

        Args:
            timeout: Maximum seconds to wait for background thread termination.
        """
        with self._lock:
            try:
                self._runtime.stop(timeout=timeout)
            except Exception:
                logger.exception("Error stopping EdgeInteractionRuntime")

            if self._persistence_worker is not None:
                try:
                    self._persistence_worker.stop(timeout=timeout)
                except Exception:
                    logger.exception("Error stopping SessionPersistenceWorker")

            try:
                self._coordinator.close()
            except Exception:
                logger.exception("Error closing InteractionCoordinator")

            try:
                self._db_manager.close()
            except Exception:
                logger.exception("Error closing DatabaseManager")

    def pause(self) -> None:
        """Pause continuous frame pumping without resetting perception or session state."""
        with self._lock:
            self._runtime.pause()

    def resume(self) -> None:
        """Resume continuous frame pumping from PAUSED state."""
        with self._lock:
            self._runtime.resume()

    def step_once(self) -> tuple[InteractionContext, list[BaseEvent]]:
        """Execute a single deterministic frame-to-interaction step.

        Runs one frame through perception, session management, and event dispatch,
        and returns the resulting application context snapshot and domain events.

        Returns:
            Tuple of (InteractionContext snapshot, list of emitted BaseEvents).
        """
        with self._lock:
            _frame_result, events = self._runtime.step_once()
            ctx = self._coordinator.get_context()
            return ctx, events

    # ------------------------------------------------------------------
    # Gallery & Recognition Management
    # ------------------------------------------------------------------

    def reload_gallery(self) -> int:
        """Synchronize in-memory face recognition gallery with persisted embeddings.

        Extracts active, consented embeddings matching the configured model from
        the repository and atomically hot-swaps the recognition gallery.

        Returns:
            Number of enrolled person identities loaded into the active gallery.
        """
        with self._lock:
            gallery = load_gallery_from_repository(
                repository=self._repository,
                model_name=self._config.model_name,
                model_version=self._config.model_version,
                require_active=self._config.require_active_for_gallery,
                require_consent=self._config.require_consent_for_gallery,
            )
            self._recognizer.replace_gallery(gallery)
            return len(gallery.list_entries())

    # ------------------------------------------------------------------
    # Listener Registration Delegation
    # ------------------------------------------------------------------

    def add_context_listener(self, callback: Callable[[InteractionContext], None]) -> int:
        """Register a callback for application interaction context transitions.

        Args:
            callback: Function invoked synchronously on context updates.

        Returns:
            Unique listener ID for subsequent unregistration.
        """
        return self._coordinator.add_context_listener(callback)

    def remove_context_listener(self, listener_id: int) -> bool:
        """Unregister a previously registered context listener.

        Args:
            listener_id: Listener ID returned by add_context_listener.

        Returns:
            True if removed, False if listener was not found.
        """
        return self._coordinator.remove_context_listener(listener_id)

    def add_event_listener(
        self,
        event_type: type[T],
        callback: Callable[[T], None],
    ) -> int:
        """Subscribe directly to structured domain events on the in-process EventBus.

        Args:
            event_type: Concrete subclass of BaseEvent to subscribe to.
            callback: Function invoked synchronously when matching event is published.

        Returns:
            Unique subscription ID for subsequent unregistration.
        """
        return self._event_bus.subscribe(event_type, callback)

    def remove_event_listener(self, subscription_id: int) -> bool:
        """Unsubscribe a previously registered event listener from EventBus.

        Args:
            subscription_id: Subscription ID returned by add_event_listener.

        Returns:
            True if unsubscribed, False if subscription was not found.
        """
        return self._event_bus.unsubscribe(subscription_id)

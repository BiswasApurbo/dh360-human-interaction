"""Unified human interaction application service facade.

Provides a thin, typed, in-process application facade coordinating edge perception
runtime, identity recognition, session management, face enrollment, and context aggregation
without duplicating underlying domain logic.

Invariants:
    - Transport-neutral boundary returning frozen, serialization-safe DTOs.
    - Zero raw computer vision, tensor manipulation, or SQL execution directly in facade.
    - Exposes strictly application-safe contracts without leaking internal biometric
      embeddings, tracker IDs, frame IDs, or raw camera image arrays.
    - Delegates all domain responsibilities to specialized underlying services.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

import numpy as np

from dh360_hi.application.config import HumanInteractionConfig
from dh360_hi.application.coordinator import InteractionCoordinator
from dh360_hi.application.health import ServiceHealthSnapshot, evaluate_service_health
from dh360_hi.backend.exceptions import (
    PersonNotFoundError,
)
from dh360_hi.backend.exceptions import (
    ValidationError as BackendValidationError,
)
from dh360_hi.backend.gallery_loader import load_gallery_from_repository
from dh360_hi.contracts.dto import (
    CreatePersonCommand,
    EnrollmentProgressDTO,
    EnrollmentResultDTO,
    InteractionContextDTO,
    InteractionSessionDTO,
    InteractionSummaryDTO,
    PersonEngagementStatsDTO,
    PersonProfileDTO,
    RuntimeMetricsDTO,
    SessionAggregateAnalyticsDTO,
    StepResultDTO,
    UpdatePersonCommand,
)
from dh360_hi.contracts.errors import ConflictAppError, ServiceUnavailableError
from dh360_hi.contracts.events import (
    BaseEvent,
    PersonIdentifiedEvent,
    PersonLostIdentityEvent,
    SessionEndedEvent,
    SessionStartedEvent,
)
from dh360_hi.contracts.integration import (
    BaseIntegrationEvent,
    map_domain_event_to_integration,
)
from dh360_hi.edge.event_bus import EventBus
from dh360_hi.edge.runtime import EdgeInteractionRuntime, RuntimeMetrics, RuntimeState
from dh360_hi.vision.frame import Frame, FrameSource

if TYPE_CHECKING:
    from dh360_hi.backend.database import DatabaseManager
    from dh360_hi.backend.models import (
        ConsentStatus,
        PersonStatus,
        SessionQueryFilter,
    )
    from dh360_hi.backend.person_service import BasePersonService
    from dh360_hi.backend.repository import BasePersonRepository
    from dh360_hi.backend.session_repository import (
        BaseSessionRepository,
        SessionPersistenceWorker,
    )
    from dh360_hi.contracts.interaction import InteractionContext
    from dh360_hi.vision.enrollment import EnrollmentMode, FaceEnrollmentController
    from dh360_hi.vision.recognition import BaseFaceRecognizer

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseEvent)


class HumanInteractionService:
    """Unified application service facade for the Human Interaction subsystem.

    Coordinates lifecycle management, context queries, person profiles, historical
    session persistence, face enrollment workflows, and recognition gallery synchronization.
    """

    def __init__(
        self,
        coordinator: InteractionCoordinator,
        person_service: BasePersonService,
        repository: BasePersonRepository,
        recognizer: BaseFaceRecognizer,
        db_manager: DatabaseManager,
        event_bus: EventBus,
        runtime: EdgeInteractionRuntime | None = None,
        session_repository: BaseSessionRepository | None = None,
        persistence_worker: SessionPersistenceWorker | None = None,
        enrollment_controller: FaceEnrollmentController | None = None,
        config: HumanInteractionConfig | None = None,
    ) -> None:
        self._coordinator = coordinator
        self._person_service = person_service
        self._repository = repository
        self._recognizer = recognizer
        self._db_manager = db_manager
        self._event_bus = event_bus
        self._runtime = runtime
        self._session_repository = session_repository
        self._persistence_worker = persistence_worker
        self._enrollment_controller = enrollment_controller
        self._enrollment_capture: FrameSource | None = None
        self._config = config or HumanInteractionConfig()
        self._lock = threading.RLock()
        self._person_mutation_lock = threading.Lock()

        # Integration event subscribers
        self._integration_listeners: dict[int, Callable[[BaseIntegrationEvent], None]] = {}
        self._next_integration_listener_id = 1
        self._context_listener_adapters: dict[int, Callable[[InteractionContext], None]] = {}

        # Subscribe to internal domain events to broadcast sanitized integration events
        self._domain_sub_ids: list[int] = [
            self._event_bus.subscribe(
                ev_cls,
                self._handle_domain_event_for_integration,
            )
            for ev_cls in (
                SessionStartedEvent,
                PersonIdentifiedEvent,
                PersonLostIdentityEvent,
                SessionEndedEvent,
            )
        ]

    def _handle_domain_event_for_integration(self, event: BaseEvent) -> None:
        """Bridge internal domain events to external integration event subscribers."""
        summary_dto: InteractionSummaryDTO | None = None
        if isinstance(event, SessionEndedEvent):
            raw_summary = self._coordinator.get_last_summary()
            if raw_summary is not None:
                summary_dto = InteractionSummaryDTO.from_summary(raw_summary)

        integ_event = map_domain_event_to_integration(event, summary=summary_dto)
        if integ_event is not None:
            with self._lock:
                listeners = list(self._integration_listeners.values())
            for listener in listeners:
                try:
                    listener(integ_event)
                except Exception:
                    logger.exception("Error dispatching event %s", integ_event.event_type)

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
        return self._runtime.state == RuntimeState.RUNNING if self._runtime is not None else False

    @property
    def state(self) -> RuntimeState:
        """Current operational lifecycle state of the edge runtime."""
        return self._runtime.state if self._runtime is not None else RuntimeState.STOPPED

    @property
    def context(self) -> InteractionContextDTO:
        """Current immutable interaction context snapshot DTO."""
        return InteractionContextDTO.from_context(self._coordinator.get_context())

    @property
    def metrics(self) -> RuntimeMetrics:
        """Current operational telemetry metrics."""
        if self._runtime is None:
            msg = "Perception runtime is not enabled"
            raise ServiceUnavailableError("runtime", msg)
        return self._runtime.metrics

    @property
    def runtime(self) -> EdgeInteractionRuntime | None:
        """Composed edge interaction runtime instance, if configured."""
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

    @property
    def enrollment_controller(self) -> FaceEnrollmentController | None:
        """Composed face enrollment controller instance, if configured."""
        return self._enrollment_controller

    # ------------------------------------------------------------------
    # Interaction Context Queries (DTO Boundary)
    # ------------------------------------------------------------------

    def get_interaction_context(self) -> InteractionContextDTO:
        """Retrieve the current thread-safe immutable interaction context snapshot."""
        return InteractionContextDTO.from_context(self._coordinator.get_context())

    def get_last_summary(self) -> InteractionSummaryDTO | None:
        """Retrieve the summary record of the most recently completed session, if any."""
        summary = self._coordinator.get_last_summary()
        return InteractionSummaryDTO.from_summary(summary) if summary is not None else None

    def get_runtime_metrics(self) -> RuntimeMetricsDTO:
        """Retrieve current operational telemetry metrics snapshot DTO."""
        if self._runtime is None:
            msg = "Perception runtime is not enabled"
            raise ServiceUnavailableError("runtime", msg)
        return RuntimeMetricsDTO.from_metrics(self._runtime.metrics)

    def get_runtime_state(self) -> RuntimeState:
        """Retrieve current runtime execution state."""
        return self._runtime.state if self._runtime is not None else RuntimeState.STOPPED

    # ------------------------------------------------------------------
    # Person & Profile Queries and Commands (DTO Boundary)
    # ------------------------------------------------------------------

    def get_person(self, person_id: str) -> PersonProfileDTO | None:
        """Retrieve a sanitized public person profile by UUID identifier.

        Args:
            person_id: UUID of the target person.

        Returns:
            PersonProfileDTO if found and registered, otherwise None.
        """
        profile = self._person_service.get_person(person_id)
        return PersonProfileDTO.from_profile(profile) if profile is not None else None

    def list_persons(
        self,
        status: PersonStatus | str | None = None,
        consent_status: ConsentStatus | str | None = None,
    ) -> list[PersonProfileDTO]:
        """List registered person profiles with optional filtering.

        Args:
            status: Optional PersonStatus filter (e.g. ACTIVE).
            consent_status: Optional ConsentStatus filter (e.g. GRANTED).

        Returns:
            List of matching PersonProfileDTO records.
        """
        people = self._person_service.list_people(status=status)
        if consent_status is not None:
            c_val = (
                consent_status.value if hasattr(consent_status, "value") else str(consent_status)
            )
            people = [
                p
                for p in people
                if (
                    p.consent_status.value
                    if hasattr(p.consent_status, "value")
                    else str(p.consent_status)
                )
                == c_val
            ]
        return [PersonProfileDTO.from_profile(p) for p in people]

    def create_person(self, command: CreatePersonCommand) -> PersonProfileDTO:
        """Create and enroll a new person profile.

        Args:
            command: Validated CreatePersonCommand payload.

        Returns:
            Created PersonProfileDTO.
        """
        profile = self._person_service.create_person(
            display_name=command.display_name,
            preferred_name=command.preferred_name,
            status=command.status,
            consent_status=command.consent_status,
            notes=command.notes,
            person_id=command.person_id,
        )
        return PersonProfileDTO.from_profile(profile)

    def update_person(self, person_id: str, command: UpdatePersonCommand) -> PersonProfileDTO:
        """Update an existing person profile.

        Args:
            person_id: UUID of the target person.
            command: Validated UpdatePersonCommand payload.

        Returns:
            Updated PersonProfileDTO.
        """
        prior_state = self._begin_person_mutation()
        resume_runtime = True
        mutation_started = False
        try:
            with self._lock:
                existing = self._person_service.get_person(person_id)
                if existing is None:
                    raise PersonNotFoundError(f"Person '{person_id}' not found")

                self._validate_gallery_eligibility_update(command)
                mutation_started = True
                try:
                    profile = self._person_service.update_person(
                        person_id=person_id,
                        display_name=command.display_name,
                        preferred_name=command.preferred_name,
                        status=command.status,
                        consent_status=command.consent_status,
                        notes=command.notes,
                    )
                except (BackendValidationError, PersonNotFoundError):
                    raise
                except Exception as exc:
                    resume_runtime = False
                    raise ServiceUnavailableError(
                        "person_mutation",
                        "Profile update outcome is uncertain; runtime remains paused",
                    ) from exc

                self._synchronize_person_mutation(person_id, deleted=False)
                return PersonProfileDTO.from_profile(profile)
        except Exception as exc:
            if mutation_started and not isinstance(
                exc,
                (BackendValidationError, PersonNotFoundError, ServiceUnavailableError),
            ):
                resume_runtime = False
                raise ServiceUnavailableError(
                    "person_mutation",
                    "Profile was updated but runtime synchronization failed; "
                    "runtime remains paused",
                ) from exc
            if isinstance(exc, ServiceUnavailableError):
                resume_runtime = False
            raise
        finally:
            self._finish_person_mutation(prior_state, resume=resume_runtime)

    def delete_person(self, person_id: str) -> bool:
        """Delete a person profile, stored embeddings, and synchronize runtime identity state.

        Args:
            person_id: UUID of the target person.

        Returns:
            True if person was deleted, False if not found.
        """
        prior_state = self._begin_person_mutation(flush_session_records=True)
        resume_runtime = True
        mutation_started = False
        try:
            with self._lock:
                try:
                    deleted = self._person_service.delete_person(person_id=person_id)
                except (BackendValidationError, PersonNotFoundError):
                    raise
                except Exception as exc:
                    resume_runtime = False
                    raise ServiceUnavailableError(
                        "person_mutation",
                        "Person deletion outcome is uncertain; runtime remains paused",
                    ) from exc

                if not deleted:
                    return False
                mutation_started = True
                self._synchronize_person_mutation(person_id, deleted=True)
                return True
        except Exception as exc:
            if mutation_started and not isinstance(exc, ServiceUnavailableError):
                resume_runtime = False
                raise ServiceUnavailableError(
                    "person_mutation",
                    "Person deletion committed but runtime synchronization failed; "
                    "runtime remains paused",
                ) from exc
            if isinstance(exc, ServiceUnavailableError):
                resume_runtime = False
            raise
        finally:
            self._finish_person_mutation(prior_state, resume=resume_runtime)

    def reset_person_biometrics(self, person_id: str) -> int:
        """Delete all biometric embeddings for a person and update runtime recognizer gallery.

        Args:
            person_id: UUID of the target person.

        Returns:
            Count of deleted biometric embedding records.
        """
        prior_state = self._begin_person_mutation()
        resume_runtime = True
        mutation_started = False
        try:
            with self._lock:
                try:
                    deleted_count = self._person_service.delete_person_embeddings(
                        person_id=person_id
                    )
                except (BackendValidationError, PersonNotFoundError):
                    raise
                except Exception as exc:
                    resume_runtime = False
                    raise ServiceUnavailableError(
                        "person_mutation",
                        "Biometric reset outcome is uncertain; runtime remains paused",
                    ) from exc

                mutation_started = True
                self._synchronize_person_mutation(person_id, deleted=False)
                return deleted_count
        except Exception as exc:
            if mutation_started and not isinstance(exc, ServiceUnavailableError):
                resume_runtime = False
                raise ServiceUnavailableError(
                    "person_mutation",
                    "Biometric reset committed but runtime synchronization failed; "
                    "runtime remains paused",
                ) from exc
            if isinstance(exc, ServiceUnavailableError):
                resume_runtime = False
            raise
        finally:
            self._finish_person_mutation(prior_state, resume=resume_runtime)

    def _begin_person_mutation(self, flush_session_records: bool = False) -> RuntimeState | None:
        """Serialize operator writes, quiesce vision, and optionally flush session history."""
        if not self._person_mutation_lock.acquire(timeout=5.0):
            raise ConflictAppError("Another person mutation is already in progress")

        prior_state: RuntimeState | None = None
        barrier_started = False
        try:
            if self._runtime is not None:
                try:
                    prior_state = self._runtime.begin_mutation_barrier(timeout=5.0)
                except Exception as exc:
                    raise ServiceUnavailableError(
                        "runtime", "Could not establish a safe frame boundary"
                    ) from exc
                if prior_state is None:
                    raise ServiceUnavailableError(
                        "runtime", "Could not reach a frame boundary before timeout"
                    )
                barrier_started = True
                try:
                    worker_quiesced = self._runtime.pipeline.worker.quiesce(timeout=5.0)
                except Exception as exc:
                    raise ServiceUnavailableError(
                        "recognition_worker", "Could not quiesce recognition work"
                    ) from exc
                if not worker_quiesced:
                    raise ServiceUnavailableError(
                        "recognition_worker", "Could not quiesce recognition work before timeout"
                    )

            if flush_session_records and self._persistence_worker is not None:
                try:
                    errors_before_flush = self._persistence_worker.error_count
                    flushed = self.flush_session_persistence(timeout=5.0)
                    errors_after_flush = self._persistence_worker.error_count
                except Exception as exc:
                    raise ServiceUnavailableError(
                        "session_persistence", "Could not verify pending session writes"
                    ) from exc
                if not flushed or errors_after_flush != errors_before_flush:
                    raise ServiceUnavailableError(
                        "session_persistence", "Could not safely flush pending session records"
                    )
            return prior_state
        except Exception:
            try:
                if barrier_started and self._runtime is not None and prior_state is not None:
                    self._runtime.end_mutation_barrier(prior_state, resume=True)
            finally:
                self._person_mutation_lock.release()
            raise

    def _finish_person_mutation(
        self,
        prior_state: RuntimeState | None,
        resume: bool,
    ) -> None:
        """Release the mutation gate and restore or preserve the prior runtime state."""
        try:
            if self._runtime is not None and prior_state is not None:
                try:
                    self._runtime.end_mutation_barrier(prior_state, resume=resume)
                except Exception as exc:
                    raise ServiceUnavailableError(
                        "runtime", "Could not restore the runtime mutation barrier state"
                    ) from exc
        finally:
            self._person_mutation_lock.release()

    def _validate_gallery_eligibility_update(self, command: UpdatePersonCommand) -> None:
        """Reject consent/status changes that configured gallery policy cannot enforce."""
        if command.status is not None:
            status_value = (
                command.status.value if hasattr(command.status, "value") else str(command.status)
            )
            if status_value != "active" and not self._config.require_active_for_gallery:
                raise ConflictAppError(
                    "Cannot deactivate a person while inactive profiles are allowed "
                    "in the recognition gallery"
                )
        if command.consent_status is not None:
            consent_value = (
                command.consent_status.value
                if hasattr(command.consent_status, "value")
                else str(command.consent_status)
            )
            if consent_value != "granted" and not self._config.require_consent_for_gallery:
                raise ConflictAppError(
                    "Cannot withdraw consent while consent is not required for the "
                    "recognition gallery"
                )

    def _synchronize_person_mutation(self, person_id: str, deleted: bool) -> None:
        """Reload gallery and reconcile live identity/session state at a safe point."""
        self.reload_gallery()

        identity_manager = (
            self._runtime.pipeline.identity_manager if self._runtime is not None else None
        )
        if identity_manager is not None:
            identity_manager.invalidate_person(person_id)

        if self._runtime is not None:
            self._runtime.session_manager.invalidate_person_identity(
                person_id,
                deleted=deleted,
            )

        if deleted:
            self._coordinator.forget_deleted_person(person_id)

    # ------------------------------------------------------------------
    # Face Enrollment Workflow Commands (DTO Boundary)
    # ------------------------------------------------------------------

    def start_enrollment(
        self,
        person_id: str,
        mode: EnrollmentMode | str = "add_samples",
        capture_source: FrameSource | None = None,
    ) -> EnrollmentProgressDTO:
        """Start a guided face enrollment session for an enrolled, consented person.

        Args:
            person_id: Target person UUID.
            mode: Enrollment mode ('add_samples' or 'replace').
            capture_source: Optional server-side FrameSource to open and drive via
                step_enrollment(). When omitted, the caller feeds frames directly through
                process_enrollment_frame() (existing CLI behavior).

        Returns:
            Initial EnrollmentProgressDTO.

        Raises:
            ServiceUnavailableError: If the enrollment controller or capture source is
                unavailable.
        """
        if self._enrollment_controller is None:
            msg = "FaceEnrollmentController is not configured"
            raise ServiceUnavailableError("enrollment", msg)

        with self._lock:
            if capture_source is not None:
                self._release_enrollment_capture()
                try:
                    capture_source.open()
                except Exception as exc:
                    raise ServiceUnavailableError(
                        "enrollment_capture", f"Failed to open capture source: {exc}"
                    ) from exc
                self._enrollment_capture = capture_source

            try:
                progress = self._enrollment_controller.start_session(person_id=person_id, mode=mode)
            except Exception:
                self._release_enrollment_capture()
                raise
            return EnrollmentProgressDTO.from_progress(progress)

    def step_enrollment(self) -> EnrollmentProgressDTO:
        """Pull one frame from the active server-side capture source and advance the session.

        Returns:
            Updated EnrollmentProgressDTO after processing exactly one captured frame.

        Raises:
            ServiceUnavailableError: If the enrollment controller, capture source, or
                frame capture is unavailable.
        """
        if self._enrollment_controller is None:
            msg = "FaceEnrollmentController is not configured"
            raise ServiceUnavailableError("enrollment", msg)

        with self._lock:
            if self._enrollment_capture is None:
                msg = "No active server-side capture source for this enrollment session"
                raise ServiceUnavailableError("enrollment_capture", msg)
            try:
                frame = self._enrollment_capture.read()
                if frame is None:
                    msg = "Camera capture failed or reached end of stream"
                    raise ServiceUnavailableError("enrollment_capture", msg)
                progress = self._enrollment_controller.process_frame(frame)
                return EnrollmentProgressDTO.from_progress(progress)
            except Exception:
                # Ensure capture source is released on any error (including timeout)
                self._release_enrollment_capture()
                raise

    def process_enrollment_frame(
        self,
        frame: Frame | np.ndarray,
    ) -> EnrollmentProgressDTO:
        """Process a camera frame against the active face enrollment session.

        Args:
            frame: Camera Frame or raw BGR numpy array.

        Returns:
            Updated EnrollmentProgressDTO.

        Raises:
            ServiceUnavailableError: If enrollment controller is not configured.
        """
        if self._enrollment_controller is None:
            msg = "FaceEnrollmentController is not configured"
            raise ServiceUnavailableError("enrollment", msg)
        if isinstance(frame, np.ndarray):
            import time

            frame_obj = Frame(image=frame, frame_id=0, timestamp=time.time())
        else:
            frame_obj = frame
        progress = self._enrollment_controller.process_frame(frame_obj)
        return EnrollmentProgressDTO.from_progress(progress)

    def cancel_enrollment(self, reason: str = "User cancelled") -> bool:
        """Cancel an in-progress face enrollment session without persisting samples.

        Args:
            reason: Optional cancellation reason.

        Returns:
            True if cancelled.

        Raises:
            ServiceUnavailableError: If enrollment controller is not configured.
        """
        if self._enrollment_controller is None:
            msg = "FaceEnrollmentController is not configured"
            raise ServiceUnavailableError("enrollment", msg)
        with self._lock:
            _ = reason
            self._enrollment_controller.cancel_session()
            self._release_enrollment_capture()
            return True

    def commit_enrollment(self) -> EnrollmentResultDTO:
        """Persist collected enrollment samples within a safe person-mutation barrier.

        Reuses the M3.2.2 mutation barrier: live perception is paused (when enabled), the
        recognition embedding worker is quiesced, the samples are persisted through the
        person service, and the recognition gallery plus live identity/session state are
        reconciled at a stable frame boundary before the runtime is restored.

        Returns:
            EnrollmentResultDTO outcome.

        Raises:
            ServiceUnavailableError: If the enrollment controller is not configured or the
                barrier could not be established / synchronized.
        """
        if self._enrollment_controller is None:
            msg = "FaceEnrollmentController is not configured"
            raise ServiceUnavailableError("enrollment", msg)

        prior_state = self._begin_person_mutation()
        resume_runtime = True
        mutation_started = False
        try:
            with self._lock:
                try:
                    res = self._enrollment_controller.commit_session()
                except Exception as exc:
                    if isinstance(exc, (ServiceUnavailableError, ConflictAppError)):
                        raise
                    resume_runtime = False
                    raise ServiceUnavailableError(
                        "person_mutation",
                        "Enrollment commit outcome is uncertain; runtime remains paused",
                    ) from exc

                mutation_started = True
                if res.is_success:
                    self._synchronize_person_mutation(res.person_id, deleted=False)
                self._release_enrollment_capture()
                return EnrollmentResultDTO.from_result(res)
        except Exception as exc:
            if mutation_started and not isinstance(exc, ServiceUnavailableError):
                resume_runtime = False
                raise ServiceUnavailableError(
                    "person_mutation",
                    "Enrollment committed but runtime synchronization failed; "
                    "runtime remains paused",
                ) from exc
            if isinstance(exc, ServiceUnavailableError):
                resume_runtime = False
            raise
        finally:
            self._release_enrollment_capture()
            self._finish_person_mutation(prior_state, resume=resume_runtime)

    def _release_enrollment_capture(self) -> None:
        """Release and clear any active server-side enrollment capture source."""
        capture = self._enrollment_capture
        self._enrollment_capture = None
        if capture is not None:
            try:
                capture.release()
            except Exception:
                logger.exception("Error releasing enrollment capture source")

    def get_enrollment_progress(self) -> EnrollmentProgressDTO | None:
        """Retrieve current progress of the active enrollment session, if any."""
        if self._enrollment_controller is None:
            return None
        p = self._enrollment_controller.current_progress
        return EnrollmentProgressDTO.from_progress(p) if p is not None else None

    # ------------------------------------------------------------------
    # Historical Session Queries & Analytics (DTO Boundary)
    # ------------------------------------------------------------------

    def get_session(self, session_id: str) -> InteractionSessionDTO | None:
        """Retrieve an interaction session record DTO by UUID identifier."""
        if self._session_repository is None:
            return None
        rec = self._session_repository.get_session(session_id)
        return InteractionSessionDTO.from_record(rec) if rec is not None else None

    def list_sessions(
        self,
        person_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[InteractionSessionDTO]:
        """List historical interaction session DTOs with pagination."""
        if self._session_repository is None:
            return []
        recs = self._session_repository.list_sessions(
            person_id=person_id,
            limit=limit,
            offset=offset,
        )
        return [InteractionSessionDTO.from_record(r) for r in recs]

    def count_sessions(self, person_id: str | None = None) -> int:
        """Count total historical interaction sessions."""
        if self._session_repository is None:
            return 0
        return self._session_repository.count_sessions(person_id=person_id)

    def query_sessions(
        self,
        filter: SessionQueryFilter | None = None,
    ) -> list[InteractionSessionDTO]:
        """Query historical interaction sessions with rich filtering and pagination."""
        if self._session_repository is None:
            return []
        from dh360_hi.backend.models import SessionQueryFilter

        criteria = filter if filter is not None else SessionQueryFilter()
        recs = self._session_repository.query_sessions(criteria)
        return [InteractionSessionDTO.from_record(r) for r in recs]

    def count_query_sessions(
        self,
        filter: SessionQueryFilter | None = None,
    ) -> int:
        """Count historical interaction sessions matching structured criteria."""
        if self._session_repository is None:
            return 0
        from dh360_hi.backend.models import SessionQueryFilter

        criteria = filter if filter is not None else SessionQueryFilter()
        return self._session_repository.count_query_sessions(criteria)

    def get_aggregate_analytics(
        self,
        filter: SessionQueryFilter | None = None,
    ) -> SessionAggregateAnalyticsDTO:
        """Compute aggregate statistics directly in SQLite across filtered sessions."""
        if self._session_repository is None:
            from dh360_hi.backend.models import SessionAggregateAnalytics

            analytics = SessionAggregateAnalytics(
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
        else:
            analytics = self._session_repository.get_aggregate_analytics(filter=filter)
        return SessionAggregateAnalyticsDTO.from_analytics(analytics)

    def get_person_engagement_stats(
        self,
        person_id: str,
    ) -> PersonEngagementStatsDTO | None:
        """Retrieve historical interaction metrics DTO for a specific person."""
        if self._session_repository is None:
            return None
        stats = self._session_repository.get_person_engagement_stats(person_id)
        return PersonEngagementStatsDTO.from_stats(stats) if stats is not None else None

    def flush_session_persistence(self, timeout: float = 2.0) -> bool:
        """Flush pending session persistence writes to storage."""
        if self._persistence_worker is not None:
            return self._persistence_worker.flush(timeout=timeout)
        return True

    # ------------------------------------------------------------------
    # Health Diagnostics
    # ------------------------------------------------------------------

    def get_health_snapshot(self) -> ServiceHealthSnapshot:
        """Retrieve a comprehensive health snapshot of the service and all subsystems."""
        return evaluate_service_health(self)

    # ------------------------------------------------------------------
    # Lifecycle Commands
    # ------------------------------------------------------------------

    def start(self, async_thread: bool = True) -> None:
        """Start the continuous edge interaction runtime frame pump and persistence worker."""
        with self._person_mutation_lock, self._lock:
            if self._runtime is None:
                msg = "Cannot start service: perception runtime is not configured or enabled"
                raise ServiceUnavailableError("runtime", msg)
            if self._persistence_worker is not None:
                self._persistence_worker.start()
            self._runtime.start(async_thread=async_thread)

    def stop(self, timeout: float = 3.0) -> None:
        """Gracefully stop the runtime, terminate background workers, and release resources."""
        with self._person_mutation_lock, self._lock:
            if self._domain_sub_ids:
                for sid in self._domain_sub_ids:
                    try:
                        self._event_bus.unsubscribe(sid)
                    except Exception:
                        logger.exception("Error unsubscribing domain event bridge")
                self._domain_sub_ids.clear()

            if self._runtime is not None:
                try:
                    self._runtime.stop(timeout=timeout)
                except Exception:
                    logger.exception("Error stopping EdgeInteractionRuntime")

            if self._persistence_worker is not None:
                try:
                    self._persistence_worker.stop(timeout=timeout)
                except Exception:
                    logger.exception("Error stopping SessionPersistenceWorker")

            if self._coordinator is not None:
                try:
                    self._coordinator.close()
                except Exception:
                    logger.exception("Error closing InteractionCoordinator")

            if self._db_manager is not None:
                try:
                    self._db_manager.close()
                except Exception:
                    logger.exception("Error closing DatabaseManager")

            self._release_enrollment_capture()

    def pause(self) -> None:
        """Pause continuous frame pumping without resetting perception or session state."""
        with self._person_mutation_lock, self._lock:
            if self._runtime is None:
                msg = "Perception runtime is not enabled"
                raise ServiceUnavailableError("runtime", msg)
            self._runtime.pause()

    def resume(self) -> None:
        """Resume continuous frame pumping from PAUSED state."""
        with self._person_mutation_lock, self._lock:
            if self._runtime is None:
                msg = "Perception runtime is not enabled"
                raise ServiceUnavailableError("runtime", msg)
            self._runtime.resume()

    def step_once(self) -> StepResultDTO:
        """Execute a single deterministic frame-to-interaction step.

        Runs one frame through perception, session management, and event dispatch,
        and returns the resulting transport-neutral StepResultDTO.

        Returns:
            StepResultDTO containing InteractionContextDTO and list[BaseIntegrationEvent].
        """
        with self._lock:
            if self._runtime is None:
                msg = "Perception runtime is not enabled"
                raise ServiceUnavailableError("runtime", msg)
            _frame_result, domain_events = self._runtime.step_once()
            ctx_dto = InteractionContextDTO.from_context(self._coordinator.get_context())

            summary_dto: InteractionSummaryDTO | None = None
            raw_summary = self._coordinator.get_last_summary()
            if raw_summary is not None:
                summary_dto = InteractionSummaryDTO.from_summary(raw_summary)

            integ_events: list[BaseIntegrationEvent] = []
            for ev in domain_events:
                mapped = map_domain_event_to_integration(ev, summary=summary_dto)
                if mapped is not None:
                    integ_events.append(mapped)

            return StepResultDTO(context=ctx_dto, events=integ_events)

    # ------------------------------------------------------------------
    # Gallery & Recognition Management
    # ------------------------------------------------------------------

    def reload_gallery(self) -> int:
        """Synchronize in-memory face recognition gallery with persisted embeddings."""
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
    # Integration & Domain Event Subscriptions
    # ------------------------------------------------------------------

    def add_integration_event_listener(
        self,
        callback: Callable[[BaseIntegrationEvent], None],
    ) -> int:
        """Register a callback for sanitized transport-neutral integration events.

        Args:
            callback: Function invoked synchronously on integration events.

        Returns:
            Unique listener ID for subsequent unregistration.
        """
        with self._lock:
            lid = self._next_integration_listener_id
            self._next_integration_listener_id += 1
            self._integration_listeners[lid] = callback
            return lid

    def remove_integration_event_listener(self, listener_id: int) -> bool:
        """Unregister an integration event listener.

        Args:
            listener_id: ID returned by add_integration_event_listener.

        Returns:
            True if removed, False if not found.
        """
        with self._lock:
            return self._integration_listeners.pop(listener_id, None) is not None

    def add_context_listener(self, callback: Callable[[InteractionContextDTO], None]) -> int:
        """Register a callback for application interaction context transitions.

        Args:
            callback: Function invoked with InteractionContextDTO on context updates.

        Returns:
            Unique listener ID for subsequent unregistration.
        """

        def _adapter(ctx: InteractionContext) -> None:
            callback(InteractionContextDTO.from_context(ctx))

        with self._lock:
            lid = self._coordinator.add_context_listener(_adapter)
            self._context_listener_adapters[lid] = _adapter
            return lid

    def remove_context_listener(self, listener_id: int) -> bool:
        """Unregister a previously registered context listener.

        Args:
            listener_id: Listener ID returned by add_context_listener.

        Returns:
            True if removed, False if listener was not found.
        """
        with self._lock:
            self._context_listener_adapters.pop(listener_id, None)
            return self._coordinator.remove_context_listener(listener_id)

    def add_event_listener(
        self,
        event_type: type[T],
        callback: Callable[[T], None],
    ) -> int:
        """Subscribe directly to internal domain events on the in-process EventBus.

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

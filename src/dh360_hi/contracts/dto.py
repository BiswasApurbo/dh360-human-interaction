"""Transport-neutral Data Transfer Objects (DTOs) for application integration.

All DTOs are frozen dataclasses that contain only serialization-safe primitives,
dictionaries, or nested DTOs. They provide explicit ISO 8601 UTC timestamp formatting
and contain zero internal vision structures (no numpy arrays, OpenCV objects,
bounding boxes, landmarks, embeddings, tracker IDs, or frame IDs).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from dh360_hi.backend.exceptions import ValidationError
from dh360_hi.backend.models import (
    ConsentStatus,
    PersonStatus,
    utc_now_iso,
    validate_uuid,
)

if TYPE_CHECKING:
    from dh360_hi.backend.models import (
        PersonEngagementStats,
        PersonProfile,
        SessionAggregateAnalytics,
        SessionRecord,
    )
    from dh360_hi.contracts.integration import BaseIntegrationEvent
    from dh360_hi.contracts.interaction import InteractionContext, InteractionSummary
    from dh360_hi.edge.runtime import RuntimeMetrics
    from dh360_hi.vision.enrollment import EnrollmentProgress, EnrollmentResult


def epoch_to_iso(ts: float | None) -> str | None:
    """Convert float epoch seconds to an ISO 8601 UTC string safely."""
    if ts is None or ts <= 0.0:
        return None
    try:
        return datetime.fromtimestamp(ts, tz=UTC).isoformat()
    except (ValueError, OSError, OverflowError):
        return None


# ---------------------------------------------------------------------------
# Person & Profile DTOs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PersonProfileDTO:
    """Public profile DTO for an enrolled person.

    Privacy Invariant: Excludes internal biometric implementation details
    (raw vectors, embedding counts, model name) and free-form notes.
    """

    person_id: str
    display_name: str
    preferred_name: str | None = None
    status: str = "active"
    consent_status: str = "granted"
    consent_timestamp: str | None = None
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert DTO to a JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_profile(cls, profile: PersonProfile) -> PersonProfileDTO:
        """Construct PersonProfileDTO from an internal PersonProfile record."""
        return cls(
            person_id=profile.person_id,
            display_name=profile.display_name,
            preferred_name=profile.preferred_name,
            status=(
                profile.status.value if hasattr(profile.status, "value") else str(profile.status)
            ),
            consent_status=profile.consent_status.value
            if hasattr(profile.consent_status, "value")
            else str(profile.consent_status),
            consent_timestamp=profile.consent_timestamp,
            created_at=profile.created_at,
            updated_at=profile.updated_at,
        )


@dataclass(frozen=True)
class CreatePersonCommand:
    """Command payload to create and enroll a new person."""

    display_name: str
    preferred_name: str | None = None
    status: PersonStatus | str = PersonStatus.ACTIVE
    consent_status: ConsentStatus | str = ConsentStatus.GRANTED
    notes: str | None = None
    person_id: str | None = None

    def __post_init__(self) -> None:
        if not self.display_name or not self.display_name.strip():
            msg = "display_name cannot be empty or whitespace"
            raise ValidationError(msg)
        if self.person_id is not None:
            validate_uuid(self.person_id, "person_id")

    def to_dict(self) -> dict[str, Any]:
        """Convert command to a dictionary."""
        return {
            "display_name": self.display_name.strip(),
            "preferred_name": self.preferred_name.strip() if self.preferred_name else None,
            "status": self.status.value if hasattr(self.status, "value") else str(self.status),
            "consent_status": self.consent_status.value
            if hasattr(self.consent_status, "value")
            else str(self.consent_status),
            "notes": self.notes.strip() if self.notes else None,
            "person_id": self.person_id,
        }


@dataclass(frozen=True)
class UpdatePersonCommand:
    """Command payload to update an existing person profile."""

    display_name: str | None = None
    preferred_name: str | None = None
    status: PersonStatus | str | None = None
    consent_status: ConsentStatus | str | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        if self.display_name is not None and not self.display_name.strip():
            msg = "display_name cannot be empty or whitespace"
            raise ValidationError(msg)

    def to_dict(self) -> dict[str, Any]:
        """Convert command to a dictionary."""
        return {
            "display_name": self.display_name.strip() if self.display_name is not None else None,
            "preferred_name": self.preferred_name.strip()
            if self.preferred_name is not None
            else None,
            "status": (self.status.value if hasattr(self.status, "value") else str(self.status))
            if self.status is not None
            else None,
            "consent_status": (
                self.consent_status.value
                if hasattr(self.consent_status, "value")
                else str(self.consent_status)
            )
            if self.consent_status is not None
            else None,
            "notes": self.notes.strip() if self.notes is not None else None,
        }


# ---------------------------------------------------------------------------
# Interaction Context & Summary DTOs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InteractionContextDTO:
    """Transport-neutral snapshot of active interaction context."""

    session_id: str | None = None
    person_id: str | None = None
    display_name: str | None = None
    state: str = "idle"
    started_at_iso: str | None = None
    last_updated_at_iso: str = ""
    identity_known: bool = False
    last_known_person_id: str | None = None
    identity_transitions_count: int = 0
    person_profile: PersonProfileDTO | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert DTO to a JSON-serializable dictionary."""
        data = asdict(self)
        if self.person_profile is not None:
            data["person_profile"] = self.person_profile.to_dict()
        return data

    @classmethod
    def from_context(cls, ctx: InteractionContext) -> InteractionContextDTO:
        """Construct InteractionContextDTO from an internal InteractionContext."""
        profile_dto = (
            PersonProfileDTO.from_profile(ctx.person_profile)
            if ctx.person_profile is not None
            else None
        )
        return cls(
            session_id=ctx.session_id,
            person_id=ctx.person_id,
            display_name=ctx.display_name,
            state=ctx.state.value if hasattr(ctx.state, "value") else str(ctx.state),
            started_at_iso=epoch_to_iso(ctx.started_at),
            last_updated_at_iso=epoch_to_iso(ctx.last_updated_at) or utc_now_iso(),
            identity_known=ctx.identity_known,
            last_known_person_id=ctx.last_known_person_id,
            identity_transitions_count=ctx.identity_transitions_count,
            person_profile=profile_dto,
        )


@dataclass(frozen=True)
class InteractionSummaryDTO:
    """Transport-neutral summary record of a completed interaction session."""

    session_id: str
    person_id: str | None
    display_name: str | None
    started_at_iso: str
    ended_at_iso: str
    duration_seconds: float
    final_state: str
    identity_was_known: bool
    identity_transitions_count: int
    person_profile: PersonProfileDTO | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert DTO to a JSON-serializable dictionary."""
        data = asdict(self)
        if self.person_profile is not None:
            data["person_profile"] = self.person_profile.to_dict()
        return data

    @classmethod
    def from_summary(cls, summary: InteractionSummary) -> InteractionSummaryDTO:
        """Construct InteractionSummaryDTO from an internal InteractionSummary."""
        profile_dto = (
            PersonProfileDTO.from_profile(summary.person_profile)
            if summary.person_profile is not None
            else None
        )
        return cls(
            session_id=summary.session_id,
            person_id=summary.person_id,
            display_name=summary.display_name,
            started_at_iso=epoch_to_iso(summary.started_at) or utc_now_iso(),
            ended_at_iso=epoch_to_iso(summary.ended_at) or utc_now_iso(),
            duration_seconds=summary.duration_seconds,
            final_state=summary.final_state.value
            if hasattr(summary.final_state, "value")
            else str(summary.final_state),
            identity_was_known=summary.identity_was_known,
            identity_transitions_count=summary.identity_transitions_count,
            person_profile=profile_dto,
        )


# ---------------------------------------------------------------------------
# Historical Session & Analytics DTOs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InteractionSessionDTO:
    """Transport-neutral historical interaction session record."""

    session_id: str
    person_id: str | None
    display_name: str | None
    started_at_iso: str
    ended_at_iso: str
    duration_seconds: float
    final_state: str
    identity_was_known: bool
    identity_transitions_count: int
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        """Convert DTO to a JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_record(cls, rec: SessionRecord) -> InteractionSessionDTO:
        """Construct InteractionSessionDTO from an internal SessionRecord."""
        return cls(
            session_id=rec.session_id,
            person_id=rec.person_id,
            display_name=rec.display_name,
            started_at_iso=epoch_to_iso(rec.started_at) or rec.created_at,
            ended_at_iso=epoch_to_iso(rec.ended_at) or rec.created_at,
            duration_seconds=rec.duration_seconds,
            final_state=rec.final_state,
            identity_was_known=rec.identity_was_known,
            identity_transitions_count=rec.identity_transitions_count,
            created_at=rec.created_at,
        )


@dataclass(frozen=True)
class SessionAggregateAnalyticsDTO:
    """Transport-neutral aggregated historical session statistics."""

    total_sessions: int
    total_duration_seconds: float
    average_duration_seconds: float
    min_duration_seconds: float
    max_duration_seconds: float
    identified_sessions_count: int
    anonymous_sessions_count: int
    earliest_session_started_at_iso: str | None = None
    latest_session_ended_at_iso: str | None = None

    @property
    def earliest_session_started_at(self) -> str | None:
        """Alias for earliest_session_started_at_iso."""
        return self.earliest_session_started_at_iso

    @property
    def latest_session_ended_at(self) -> str | None:
        """Alias for latest_session_ended_at_iso."""
        return self.latest_session_ended_at_iso

    def to_dict(self) -> dict[str, Any]:
        """Convert DTO to a JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_analytics(cls, analytics: SessionAggregateAnalytics) -> SessionAggregateAnalyticsDTO:
        """Construct SessionAggregateAnalyticsDTO from internal model."""
        return cls(
            total_sessions=analytics.total_sessions,
            total_duration_seconds=analytics.total_duration_seconds,
            average_duration_seconds=analytics.average_duration_seconds,
            min_duration_seconds=analytics.min_duration_seconds,
            max_duration_seconds=analytics.max_duration_seconds,
            identified_sessions_count=analytics.identified_sessions_count,
            anonymous_sessions_count=analytics.anonymous_sessions_count,
            earliest_session_started_at_iso=epoch_to_iso(analytics.earliest_session_started_at),
            latest_session_ended_at_iso=epoch_to_iso(analytics.latest_session_ended_at),
        )


@dataclass(frozen=True)
class PersonEngagementStatsDTO:
    """Transport-neutral engagement statistics for a single enrolled person."""

    person_id: str
    display_name: str | None
    total_sessions: int
    total_duration_seconds: float
    average_duration_seconds: float
    first_interaction_at_iso: str | None = None
    last_interaction_at_iso: str | None = None

    @property
    def first_interaction_at(self) -> str | None:
        """Alias for first_interaction_at_iso."""
        return self.first_interaction_at_iso

    @property
    def last_interaction_at(self) -> str | None:
        """Alias for last_interaction_at_iso."""
        return self.last_interaction_at_iso

    def to_dict(self) -> dict[str, Any]:
        """Convert DTO to a JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_stats(cls, stats: PersonEngagementStats) -> PersonEngagementStatsDTO:
        """Construct PersonEngagementStatsDTO from internal model."""
        return cls(
            person_id=stats.person_id,
            display_name=stats.display_name,
            total_sessions=stats.total_sessions,
            total_duration_seconds=stats.total_duration_seconds,
            average_duration_seconds=stats.average_duration_seconds,
            first_interaction_at_iso=epoch_to_iso(stats.first_interaction_at),
            last_interaction_at_iso=epoch_to_iso(stats.last_interaction_at),
        )


# ---------------------------------------------------------------------------
# Enrollment Workflow DTOs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnrollmentProgressDTO:
    """Transport-neutral snapshot of face enrollment progress."""

    session_id: str
    enrollment_batch_id: str
    person_id: str
    display_name: str
    mode: str
    state: str
    samples_collected: int
    target_samples: int
    minimum_samples: int
    total_attempts: int
    is_viable: bool
    is_complete: bool
    last_rejection_reason: str | None = None
    last_rejection_details: str | None = None
    rejection_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert DTO to a JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_progress(cls, progress: EnrollmentProgress) -> EnrollmentProgressDTO:
        """Construct EnrollmentProgressDTO from internal EnrollmentProgress."""
        return cls(
            session_id=progress.session_id,
            enrollment_batch_id=progress.enrollment_batch_id,
            person_id=progress.person_id,
            display_name=progress.display_name,
            mode=progress.mode.value if hasattr(progress.mode, "value") else str(progress.mode),
            state=progress.state.value if hasattr(progress.state, "value") else str(progress.state),
            samples_collected=progress.samples_collected,
            target_samples=progress.target_samples,
            minimum_samples=progress.minimum_samples,
            total_attempts=progress.total_attempts,
            is_viable=progress.is_viable,
            is_complete=progress.is_complete,
            last_rejection_reason=(
                progress.last_rejection_reason.value
                if hasattr(progress.last_rejection_reason, "value")
                else str(progress.last_rejection_reason)
            )
            if progress.last_rejection_reason is not None
            else None,
            last_rejection_details=progress.last_rejection_details,
            rejection_counts=dict(progress.rejection_counts),
        )


@dataclass(frozen=True)
class EnrollmentResultDTO:
    """Transport-neutral outcome of an enrollment commit."""

    session_id: str
    enrollment_batch_id: str
    person_id: str
    display_name: str
    mode: str
    is_success: bool
    state: str
    samples_enrolled: int
    persisted_embedding_count: int
    total_attempts: int
    error_message: str | None = None
    rejection_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert DTO to a JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_result(cls, result: EnrollmentResult) -> EnrollmentResultDTO:
        """Construct EnrollmentResultDTO from internal EnrollmentResult."""
        return cls(
            session_id=result.session_id,
            enrollment_batch_id=result.enrollment_batch_id,
            person_id=result.person_id,
            display_name=result.display_name,
            mode=result.mode.value if hasattr(result.mode, "value") else str(result.mode),
            is_success=result.is_success,
            state=result.state.value if hasattr(result.state, "value") else str(result.state),
            samples_enrolled=result.samples_enrolled,
            persisted_embedding_count=len(result.persisted_embedding_ids),
            total_attempts=result.total_attempts,
            error_message=result.error_message,
            rejection_counts=dict(result.rejection_counts),
        )


# ---------------------------------------------------------------------------
# Operational Telemetry & Step Result DTOs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RuntimeMetricsDTO:
    """Transport-neutral operational performance telemetry snapshot."""

    frames_processed: int
    current_fps: float
    average_latency_ms: float
    uptime_seconds: float
    last_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert DTO to a JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_metrics(cls, metrics: RuntimeMetrics) -> RuntimeMetricsDTO:
        """Construct RuntimeMetricsDTO from internal RuntimeMetrics."""
        return cls(
            frames_processed=metrics.frames_processed,
            current_fps=metrics.current_fps,
            average_latency_ms=metrics.average_latency_ms,
            uptime_seconds=metrics.uptime_seconds,
            last_error=metrics.last_error,
        )


@dataclass(frozen=True)
class StepResultDTO:
    """Transport-neutral outcome of a single deterministic simulation step."""

    context: InteractionContextDTO
    events: list[BaseIntegrationEvent] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert DTO to a JSON-serializable dictionary."""
        return {
            "context": self.context.to_dict(),
            "events": [e.to_dict() for e in self.events],
        }

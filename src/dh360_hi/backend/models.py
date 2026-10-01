"""Domain models, records, enums, and serialization helpers for SQLite persistence."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np

from dh360_hi.backend.exceptions import ValidationError
from dh360_hi.contracts.interaction import InteractionState

if TYPE_CHECKING:
    from dh360_hi.contracts.interaction import InteractionSummary


class PersonStatus(StrEnum):
    """Lifecycle status of an enrolled person."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    ARCHIVED = "archived"


class ConsentStatus(StrEnum):
    """Consent status for facial recognition processing."""

    GRANTED = "granted"
    WITHDRAWN = "withdrawn"
    PENDING = "pending"
    REFUSED = "refused"


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(UTC).isoformat()


def validate_uuid(val: str, field_name: str = "id") -> str:
    """Validate that a string is a valid UUID representation."""
    if not val or not isinstance(val, str):
        msg = f"{field_name} must be a non-empty string UUID"
        raise ValidationError(msg)
    try:
        parsed = uuid.UUID(val.strip())
        return str(parsed)
    except (ValueError, AttributeError) as exc:
        msg = f"Invalid UUID format for {field_name}: '{val}'"
        raise ValidationError(msg) from exc


def vector_to_blob(vector: np.ndarray, expected_dim: int = 128) -> bytes:
    """Convert a numpy vector to float32 binary blob with dimension validation."""
    if vector is None or not isinstance(vector, np.ndarray):
        msg = "Embedding vector must be a numpy ndarray"
        raise ValidationError(msg)

    flat = vector.flatten().astype(np.float32)
    if flat.shape[0] != expected_dim:
        msg = f"Embedding dimension {flat.shape[0]} does not match expected {expected_dim}"
        raise ValidationError(msg)

    if not np.all(np.isfinite(flat)):
        msg = "Embedding vector contains NaN or infinite values"
        raise ValidationError(msg)

    return flat.tobytes()


def blob_to_vector(blob: bytes, expected_dim: int = 128) -> np.ndarray:
    """Convert a binary blob back to float32 numpy vector with validation."""
    if blob is None or not isinstance(blob, (bytes, memoryview)):
        msg = "Embedding blob must be bytes"
        raise ValidationError(msg)

    raw_bytes = bytes(blob)
    expected_bytes_len = expected_dim * 4  # float32 = 4 bytes
    if len(raw_bytes) != expected_bytes_len:
        msg = (
            f"Embedding blob byte length {len(raw_bytes)} does not match "
            f"expected {expected_bytes_len} bytes for dimension {expected_dim}"
        )
        raise ValidationError(msg)

    vec = np.frombuffer(raw_bytes, dtype=np.float32)
    if not np.all(np.isfinite(vec)):
        msg = "Deserialized embedding vector contains NaN or infinite values"
        raise ValidationError(msg)

    return vec


@dataclass(frozen=True)
class PersonRecord:
    """Low-level SQLite database record for a person entity."""

    person_id: str
    display_name: str
    preferred_name: str | None = None
    status: PersonStatus = PersonStatus.ACTIVE
    consent_status: ConsentStatus = ConsentStatus.GRANTED
    consent_timestamp: str | None = None
    notes: str | None = None
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)

    def __post_init__(self) -> None:
        validate_uuid(self.person_id, "person_id")
        if not self.display_name or not self.display_name.strip():
            msg = "display_name cannot be empty or whitespace"
            raise ValidationError(msg)


@dataclass(frozen=True)
class EmbeddingRecord:
    """Low-level SQLite database record for a face embedding vector."""

    embedding_id: str
    person_id: str
    embedding: bytes
    dimension: int = 128
    model_name: str = "SFace"
    model_version: str = "2021dec"
    quality_score: float | None = None
    enrollment_batch_id: str | None = None
    created_at: str = field(default_factory=utc_now_iso)

    def __post_init__(self) -> None:
        validate_uuid(self.embedding_id, "embedding_id")
        validate_uuid(self.person_id, "person_id")
        if self.enrollment_batch_id is not None:
            validate_uuid(self.enrollment_batch_id, "enrollment_batch_id")
        if not self.model_name or not self.model_name.strip():
            msg = "model_name cannot be empty"
            raise ValidationError(msg)
        if not self.model_version or not self.model_version.strip():
            msg = "model_version cannot be empty"
            raise ValidationError(msg)

    @property
    def vector(self) -> np.ndarray:
        """Deserialize float32 array from raw bytes."""
        return blob_to_vector(self.embedding, expected_dim=self.dimension)


@dataclass(frozen=True)
class PersonProfile:
    """Public profile view of an enrolled person (excludes raw embedding vectors)."""

    person_id: str
    display_name: str
    preferred_name: str | None
    status: PersonStatus
    consent_status: ConsentStatus
    consent_timestamp: str | None
    notes: str | None
    created_at: str
    updated_at: str
    embedding_count: int = 0


@dataclass(frozen=True)
class EmbeddingMetadata:
    """Public metadata view of a stored face embedding (excludes raw vector bytes)."""

    embedding_id: str
    person_id: str
    dimension: int
    model_name: str
    model_version: str
    quality_score: float | None
    enrollment_batch_id: str | None
    created_at: str


@dataclass(frozen=True)
class SessionRecord:
    """Low-level SQLite database record for an interaction session."""

    session_id: str
    person_id: str | None
    display_name: str | None
    started_at: float
    ended_at: float
    duration_seconds: float
    final_state: str
    identity_was_known: bool
    identity_transitions_count: int
    created_at: str = field(default_factory=utc_now_iso)

    def __post_init__(self) -> None:
        validate_uuid(self.session_id, "session_id")
        if self.person_id is not None:
            validate_uuid(self.person_id, "person_id")
        if self.duration_seconds < 0.0:
            msg = "duration_seconds cannot be negative"
            raise ValidationError(msg)
        if self.identity_transitions_count < 0:
            msg = "identity_transitions_count cannot be negative"
            raise ValidationError(msg)
        if not self.final_state or not self.final_state.strip():
            msg = "final_state cannot be empty"
            raise ValidationError(msg)

    @classmethod
    def from_summary(
        cls,
        summary: InteractionSummary,
        created_at: str | None = None,
    ) -> SessionRecord:
        """Construct a SessionRecord from an application InteractionSummary."""
        final_state_str = (
            summary.final_state.value
            if hasattr(summary.final_state, "value")
            else str(summary.final_state)
        )
        return cls(
            session_id=summary.session_id,
            person_id=summary.person_id,
            display_name=summary.display_name,
            started_at=summary.started_at,
            ended_at=summary.ended_at,
            duration_seconds=summary.duration_seconds,
            final_state=final_state_str,
            identity_was_known=summary.identity_was_known,
            identity_transitions_count=summary.identity_transitions_count,
            created_at=created_at or utc_now_iso(),
        )

    def to_summary(
        self,
        person_profile: PersonProfile | None = None,
    ) -> InteractionSummary:
        """Convert SessionRecord to application-level InteractionSummary."""
        from dh360_hi.contracts.interaction import InteractionState, InteractionSummary

        try:
            state = InteractionState(self.final_state)
        except ValueError:
            state = InteractionState.IDLE

        return InteractionSummary(
            session_id=self.session_id,
            person_id=self.person_id,
            display_name=self.display_name,
            person_profile=person_profile,
            started_at=self.started_at,
            ended_at=self.ended_at,
            duration_seconds=self.duration_seconds,
            final_state=state,
            identity_was_known=self.identity_was_known,
            identity_transitions_count=self.identity_transitions_count,
        )


@dataclass(frozen=True)
class SessionQueryFilter:
    """Structured query filter criteria for historical interaction sessions."""

    person_id: str | None = None
    is_identified: bool | None = None
    final_state: InteractionState | None = None
    started_after: float | None = None
    started_before: float | None = None
    ended_after: float | None = None
    ended_before: float | None = None
    min_duration_seconds: float | None = None
    max_duration_seconds: float | None = None
    limit: int = 50
    offset: int = 0

    def __post_init__(self) -> None:
        if self.limit <= 0:
            msg = f"limit must be greater than 0, got {self.limit}"
            raise ValidationError(msg)
        if self.offset < 0:
            msg = f"offset cannot be negative, got {self.offset}"
            raise ValidationError(msg)
        if self.person_id is not None:
            validate_uuid(self.person_id, "person_id")
        if (
            self.started_after is not None
            and self.started_before is not None
            and self.started_after > self.started_before
        ):
            msg = (
                f"started_after ({self.started_after}) cannot be greater than "
                f"started_before ({self.started_before})"
            )
            raise ValidationError(msg)
        if (
            self.ended_after is not None
            and self.ended_before is not None
            and self.ended_after > self.ended_before
        ):
            msg = (
                f"ended_after ({self.ended_after}) cannot be greater than "
                f"ended_before ({self.ended_before})"
            )
            raise ValidationError(msg)
        if (
            self.started_after is not None
            and self.ended_before is not None
            and self.started_after > self.ended_before
        ):
            msg = (
                f"started_after ({self.started_after}) cannot be greater than "
                f"ended_before ({self.ended_before})"
            )
            raise ValidationError(msg)
        if self.min_duration_seconds is not None and self.min_duration_seconds < 0.0:
            msg = f"min_duration_seconds cannot be negative, got {self.min_duration_seconds}"
            raise ValidationError(msg)
        if self.max_duration_seconds is not None and self.max_duration_seconds < 0.0:
            msg = f"max_duration_seconds cannot be negative, got {self.max_duration_seconds}"
            raise ValidationError(msg)
        if (
            self.min_duration_seconds is not None
            and self.max_duration_seconds is not None
            and self.min_duration_seconds > self.max_duration_seconds
        ):
            msg = (
                f"min_duration_seconds ({self.min_duration_seconds}) cannot be greater than "
                f"max_duration_seconds ({self.max_duration_seconds})"
            )
            raise ValidationError(msg)
        if self.final_state is not None and not isinstance(self.final_state, InteractionState):
            try:
                object.__setattr__(self, "final_state", InteractionState(self.final_state))
            except (ValueError, TypeError) as exc:
                msg = f"Invalid final_state: '{self.final_state}'"
                raise ValidationError(msg) from exc


@dataclass(frozen=True)
class SessionAggregateAnalytics:
    """Aggregate statistics computed directly via SQL over interaction sessions."""

    total_sessions: int
    total_duration_seconds: float
    average_duration_seconds: float
    min_duration_seconds: float
    max_duration_seconds: float
    identified_sessions_count: int
    anonymous_sessions_count: int
    earliest_session_started_at: float | None
    latest_session_ended_at: float | None

    def __post_init__(self) -> None:
        if self.total_sessions < 0:
            msg = "total_sessions cannot be negative"
            raise ValidationError(msg)
        if self.total_duration_seconds < 0.0:
            msg = "total_duration_seconds cannot be negative"
            raise ValidationError(msg)
        if self.average_duration_seconds < 0.0:
            msg = "average_duration_seconds cannot be negative"
            raise ValidationError(msg)
        if self.min_duration_seconds < 0.0:
            msg = "min_duration_seconds cannot be negative"
            raise ValidationError(msg)
        if self.max_duration_seconds < 0.0:
            msg = "max_duration_seconds cannot be negative"
            raise ValidationError(msg)
        if self.identified_sessions_count < 0:
            msg = "identified_sessions_count cannot be negative"
            raise ValidationError(msg)
        if self.anonymous_sessions_count < 0:
            msg = "anonymous_sessions_count cannot be negative"
            raise ValidationError(msg)


@dataclass(frozen=True)
class PersonEngagementStats:
    """Historical engagement metrics for an enrolled person entity."""

    person_id: str
    display_name: str | None
    total_sessions: int
    total_duration_seconds: float
    average_duration_seconds: float
    first_interaction_at: float | None
    last_interaction_at: float | None

    def __post_init__(self) -> None:
        validate_uuid(self.person_id, "person_id")
        if self.total_sessions < 0:
            msg = "total_sessions cannot be negative"
            raise ValidationError(msg)
        if self.total_duration_seconds < 0.0:
            msg = "total_duration_seconds cannot be negative"
            raise ValidationError(msg)
        if self.average_duration_seconds < 0.0:
            msg = "average_duration_seconds cannot be negative"
            raise ValidationError(msg)

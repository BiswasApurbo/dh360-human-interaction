"""Domain models, records, enums, and serialization helpers for SQLite persistence."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

import numpy as np

from dh360_hi.backend.exceptions import ValidationError


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

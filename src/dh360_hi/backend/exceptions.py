"""Backend domain exceptions for database, repository, validation, and person service."""

from __future__ import annotations


class BackendError(Exception):
    """Base exception for all backend errors."""


class DatabaseError(BackendError):
    """General SQLite or database operation error."""


class SchemaVersionError(DatabaseError):
    """Schema version mismatch or migration failure."""


class ValidationError(BackendError):
    """Data validation or constraint violation error."""


class PersonNotFoundError(BackendError):
    """Requested person ID was not found in the repository."""


class DuplicatePersonError(BackendError):
    """Person with the given ID already exists."""


class DuplicateEmbeddingError(BackendError):
    """Embedding with the given ID already exists."""


class ForeignKeyViolationError(DatabaseError):
    """Referential integrity constraint violated."""


class IncompatibleModelError(BackendError):
    """Embedding model name or version incompatible with target system."""

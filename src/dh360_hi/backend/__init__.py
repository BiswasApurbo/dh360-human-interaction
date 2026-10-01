"""Backend package: SQLite persistence, repository, person service, and gallery loader."""

from dh360_hi.backend.database import DatabaseManager
from dh360_hi.backend.exceptions import (
    BackendError,
    DatabaseError,
    DuplicateEmbeddingError,
    DuplicatePersonError,
    ForeignKeyViolationError,
    IncompatibleModelError,
    PersonNotFoundError,
    SchemaVersionError,
    ValidationError,
)
from dh360_hi.backend.gallery_loader import load_gallery_from_repository
from dh360_hi.backend.models import (
    ConsentStatus,
    EmbeddingMetadata,
    EmbeddingRecord,
    PersonProfile,
    PersonRecord,
    PersonStatus,
    blob_to_vector,
    validate_uuid,
    vector_to_blob,
)
from dh360_hi.backend.person_service import BasePersonService, PersonService
from dh360_hi.backend.repository import BasePersonRepository, SQLitePersonRepository
from dh360_hi.backend.schema import CURRENT_SCHEMA_VERSION

__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "BackendError",
    "BasePersonRepository",
    "BasePersonService",
    "ConsentStatus",
    "DatabaseError",
    "DatabaseManager",
    "DuplicateEmbeddingError",
    "DuplicatePersonError",
    "EmbeddingMetadata",
    "EmbeddingRecord",
    "ForeignKeyViolationError",
    "IncompatibleModelError",
    "PersonNotFoundError",
    "PersonProfile",
    "PersonRecord",
    "PersonService",
    "PersonStatus",
    "SQLitePersonRepository",
    "SchemaVersionError",
    "ValidationError",
    "blob_to_vector",
    "load_gallery_from_repository",
    "validate_uuid",
    "vector_to_blob",
]

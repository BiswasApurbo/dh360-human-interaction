"""SQLite repository implementation for Person and Face Embedding persistence."""

from __future__ import annotations

import sqlite3
from abc import ABC, abstractmethod

from dh360_hi.backend.database import DatabaseManager
from dh360_hi.backend.exceptions import (
    DuplicateEmbeddingError,
    DuplicatePersonError,
    ForeignKeyViolationError,
    PersonNotFoundError,
)
from dh360_hi.backend.models import (
    ConsentStatus,
    EmbeddingRecord,
    PersonRecord,
    PersonStatus,
    validate_uuid,
)


class BasePersonRepository(ABC):
    """Abstract interface for Person and Face Embedding persistence operations."""

    @abstractmethod
    def create_person(self, person: PersonRecord) -> PersonRecord:
        """Persist a new person record."""

    @abstractmethod
    def get_person(self, person_id: str) -> PersonRecord | None:
        """Retrieve a person record by person ID."""

    @abstractmethod
    def list_persons(self, status: PersonStatus | None = None) -> list[PersonRecord]:
        """List person records, optionally filtered by lifecycle status."""

    @abstractmethod
    def update_person(self, person: PersonRecord) -> PersonRecord:
        """Update an existing person record."""

    @abstractmethod
    def delete_person(self, person_id: str) -> bool:
        """Delete a person record and cascade delete their embeddings."""

    @abstractmethod
    def add_embedding(self, embedding: EmbeddingRecord) -> EmbeddingRecord:
        """Persist a single face embedding vector."""

    @abstractmethod
    def add_embeddings(self, embeddings: list[EmbeddingRecord]) -> list[EmbeddingRecord]:
        """Persist a batch of face embeddings in a single atomic transaction."""

    @abstractmethod
    def get_embedding(self, embedding_id: str) -> EmbeddingRecord | None:
        """Retrieve a single face embedding record by ID."""

    @abstractmethod
    def get_embeddings_for_person(self, person_id: str) -> list[EmbeddingRecord]:
        """Retrieve all embedding records associated with a specific person ID."""

    @abstractmethod
    def get_all_embeddings(
        self,
        model_name: str | None = None,
        model_version: str | None = None,
    ) -> list[EmbeddingRecord]:
        """Retrieve all stored embeddings, optionally filtered by model name/version."""

    @abstractmethod
    def delete_embedding(self, embedding_id: str) -> bool:
        """Delete a specific face embedding record by ID."""

    @abstractmethod
    def delete_embeddings_for_person(self, person_id: str) -> int:
        """Delete all face embeddings associated with a specific person ID."""

    @abstractmethod
    def replace_embeddings_for_person(
        self,
        person_id: str,
        embeddings: list[EmbeddingRecord],
    ) -> list[EmbeddingRecord]:
        """Atomically delete old embeddings for a person and persist a new batch."""

    @abstractmethod
    def count_persons(self, status: PersonStatus | None = None) -> int:
        """Count total person records in the repository."""

    @abstractmethod
    def count_embeddings(self, person_id: str | None = None) -> int:
        """Count total embedding records in the repository."""


class SQLitePersonRepository(BasePersonRepository):
    """SQLite-backed implementation of BasePersonRepository."""

    def __init__(self, db_manager: DatabaseManager) -> None:
        self._db = db_manager

    @property
    def db_manager(self) -> DatabaseManager:
        """Underlying DatabaseManager instance."""
        return self._db

    @staticmethod
    def _row_to_person(row: sqlite3.Row) -> PersonRecord:
        """Convert a database row to a PersonRecord instance."""
        return PersonRecord(
            person_id=row["person_id"],
            display_name=row["display_name"],
            preferred_name=row["preferred_name"],
            status=PersonStatus(row["status"]),
            consent_status=ConsentStatus(row["consent_status"]),
            consent_timestamp=row["consent_timestamp"],
            notes=row["notes"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _row_to_embedding(row: sqlite3.Row) -> EmbeddingRecord:
        """Convert a database row to an EmbeddingRecord instance."""
        return EmbeddingRecord(
            embedding_id=row["embedding_id"],
            person_id=row["person_id"],
            embedding=bytes(row["embedding"]),
            dimension=int(row["dimension"]),
            model_name=row["model_name"],
            model_version=row["model_version"],
            quality_score=float(row["quality_score"]) if row["quality_score"] is not None else None,
            enrollment_batch_id=row["enrollment_batch_id"],
            created_at=row["created_at"],
        )

    def create_person(self, person: PersonRecord) -> PersonRecord:
        """Persist a new person record."""
        query = """
        INSERT INTO persons (
            person_id, display_name, preferred_name, status,
            consent_status, consent_timestamp, notes, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        try:
            with self._db.transaction() as cur:
                cur.execute(
                    query,
                    (
                        person.person_id,
                        person.display_name,
                        person.preferred_name,
                        person.status.value,
                        person.consent_status.value,
                        person.consent_timestamp,
                        person.notes,
                        person.created_at,
                        person.updated_at,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            msg = f"Person with ID '{person.person_id}' already exists: {exc}"
            raise DuplicatePersonError(msg) from exc

        return person

    def get_person(self, person_id: str) -> PersonRecord | None:
        """Retrieve a person record by person ID."""
        valid_id = validate_uuid(person_id, "person_id")
        query = "SELECT * FROM persons WHERE person_id = ?;"
        with self._db.readonly_cursor() as cur:
            cur.execute(query, (valid_id,))
            row = cur.fetchone()
            return self._row_to_person(row) if row else None

    def list_persons(self, status: PersonStatus | None = None) -> list[PersonRecord]:
        """List person records, optionally filtered by lifecycle status."""
        if status is not None:
            query = "SELECT * FROM persons WHERE status = ? ORDER BY created_at ASC;"
            params: tuple[str, ...] = (status.value,)
        else:
            query = "SELECT * FROM persons ORDER BY created_at ASC;"
            params = ()

        with self._db.readonly_cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
            return [self._row_to_person(r) for r in rows]

    def update_person(self, person: PersonRecord) -> PersonRecord:
        """Update an existing person record."""
        query = """
        UPDATE persons SET
            display_name = ?,
            preferred_name = ?,
            status = ?,
            consent_status = ?,
            consent_timestamp = ?,
            notes = ?,
            updated_at = ?
        WHERE person_id = ?;
        """
        with self._db.transaction() as cur:
            cur.execute(
                query,
                (
                    person.display_name,
                    person.preferred_name,
                    person.status.value,
                    person.consent_status.value,
                    person.consent_timestamp,
                    person.notes,
                    person.updated_at,
                    person.person_id,
                ),
            )
            if cur.rowcount == 0:
                msg = f"Cannot update: person '{person.person_id}' not found"
                raise PersonNotFoundError(msg)

        return person

    def delete_person(self, person_id: str) -> bool:
        """Delete a person record and cascade delete their embeddings."""
        valid_id = validate_uuid(person_id, "person_id")
        query = "DELETE FROM persons WHERE person_id = ?;"
        with self._db.transaction() as cur:
            cur.execute(query, (valid_id,))
            return cur.rowcount > 0

    def add_embedding(self, embedding: EmbeddingRecord) -> EmbeddingRecord:
        """Persist a single face embedding vector."""
        query = """
        INSERT INTO face_embeddings (
            embedding_id, person_id, embedding, dimension,
            model_name, model_version, quality_score, enrollment_batch_id, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        try:
            with self._db.transaction() as cur:
                cur.execute(
                    query,
                    (
                        embedding.embedding_id,
                        embedding.person_id,
                        embedding.embedding,
                        embedding.dimension,
                        embedding.model_name,
                        embedding.model_version,
                        embedding.quality_score,
                        embedding.enrollment_batch_id,
                        embedding.created_at,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            err_msg = str(exc).lower()
            if "foreign key" in err_msg:
                msg = f"Foreign key violation: Person '{embedding.person_id}' does not exist"
                raise ForeignKeyViolationError(msg) from exc
            if "unique" in err_msg or "primary key" in err_msg:
                msg = f"Embedding with ID '{embedding.embedding_id}' already exists"
                raise DuplicateEmbeddingError(msg) from exc
            raise

        return embedding

    def add_embeddings(self, embeddings: list[EmbeddingRecord]) -> list[EmbeddingRecord]:
        """Persist a batch of face embeddings in a single atomic transaction."""
        if not embeddings:
            return []

        query = """
        INSERT INTO face_embeddings (
            embedding_id, person_id, embedding, dimension,
            model_name, model_version, quality_score, enrollment_batch_id, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        payload = [
            (
                e.embedding_id,
                e.person_id,
                e.embedding,
                e.dimension,
                e.model_name,
                e.model_version,
                e.quality_score,
                e.enrollment_batch_id,
                e.created_at,
            )
            for e in embeddings
        ]
        try:
            with self._db.transaction() as cur:
                cur.executemany(query, payload)
        except sqlite3.IntegrityError as exc:
            err_msg = str(exc).lower()
            if "foreign key" in err_msg:
                msg = f"Foreign key violation inserting embeddings batch: {exc}"
                raise ForeignKeyViolationError(msg) from exc
            if "unique" in err_msg or "primary key" in err_msg:
                msg = f"Duplicate embedding key violation in batch: {exc}"
                raise DuplicateEmbeddingError(msg) from exc
            raise

        return embeddings

    def get_embedding(self, embedding_id: str) -> EmbeddingRecord | None:
        """Retrieve a single face embedding record by ID."""
        valid_id = validate_uuid(embedding_id, "embedding_id")
        query = "SELECT * FROM face_embeddings WHERE embedding_id = ?;"
        with self._db.readonly_cursor() as cur:
            cur.execute(query, (valid_id,))
            row = cur.fetchone()
            return self._row_to_embedding(row) if row else None

    def get_embeddings_for_person(self, person_id: str) -> list[EmbeddingRecord]:
        """Retrieve all embedding records associated with a specific person ID."""
        valid_id = validate_uuid(person_id, "person_id")
        query = "SELECT * FROM face_embeddings WHERE person_id = ? ORDER BY created_at ASC;"
        with self._db.readonly_cursor() as cur:
            cur.execute(query, (valid_id,))
            rows = cur.fetchall()
            return [self._row_to_embedding(r) for r in rows]

    def get_all_embeddings(
        self,
        model_name: str | None = None,
        model_version: str | None = None,
    ) -> list[EmbeddingRecord]:
        """Retrieve all stored embeddings, optionally filtered by model name/version."""
        clauses = []
        params: list[str] = []
        if model_name is not None:
            clauses.append("model_name = ?")
            params.append(model_name)
        if model_version is not None:
            clauses.append("model_version = ?")
            params.append(model_version)

        where_str = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"SELECT * FROM face_embeddings {where_str} ORDER BY created_at ASC;"

        with self._db.readonly_cursor() as cur:
            cur.execute(query, tuple(params))
            rows = cur.fetchall()
            return [self._row_to_embedding(r) for r in rows]

    def delete_embedding(self, embedding_id: str) -> bool:
        """Delete a specific face embedding record by ID."""
        valid_id = validate_uuid(embedding_id, "embedding_id")
        query = "DELETE FROM face_embeddings WHERE embedding_id = ?;"
        with self._db.transaction() as cur:
            cur.execute(query, (valid_id,))
            return cur.rowcount > 0

    def delete_embeddings_for_person(self, person_id: str) -> int:
        """Delete all face embeddings associated with a specific person ID."""
        valid_id = validate_uuid(person_id, "person_id")
        query = "DELETE FROM face_embeddings WHERE person_id = ?;"
        with self._db.transaction() as cur:
            cur.execute(query, (valid_id,))
            return int(cur.rowcount)

    def replace_embeddings_for_person(
        self,
        person_id: str,
        embeddings: list[EmbeddingRecord],
    ) -> list[EmbeddingRecord]:
        """Atomically delete old embeddings for a person and persist a new batch."""
        valid_id = validate_uuid(person_id, "person_id")
        person = self.get_person(valid_id)
        if person is None:
            msg = f"Cannot replace embeddings: person '{valid_id}' not found"
            raise PersonNotFoundError(msg)

        delete_query = "DELETE FROM face_embeddings WHERE person_id = ?;"
        insert_query = """
        INSERT INTO face_embeddings (
            embedding_id, person_id, embedding, dimension,
            model_name, model_version, quality_score, enrollment_batch_id, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        payload = [
            (
                e.embedding_id,
                e.person_id,
                e.embedding,
                e.dimension,
                e.model_name,
                e.model_version,
                e.quality_score,
                e.enrollment_batch_id,
                e.created_at,
            )
            for e in embeddings
        ]
        try:
            with self._db.transaction() as cur:
                cur.execute(delete_query, (valid_id,))
                if payload:
                    cur.executemany(insert_query, payload)
        except sqlite3.IntegrityError as exc:
            err_msg = str(exc).lower()
            if "foreign key" in err_msg:
                msg = f"Foreign key violation replacing embeddings for person '{valid_id}': {exc}"
                raise ForeignKeyViolationError(msg) from exc
            if "unique" in err_msg or "primary key" in err_msg:
                msg = f"Duplicate embedding key violation in replacement batch: {exc}"
                raise DuplicateEmbeddingError(msg) from exc
            raise

        return embeddings

    def count_persons(self, status: PersonStatus | None = None) -> int:
        """Count total person records in the repository."""
        if status is not None:
            query = "SELECT COUNT(*) FROM persons WHERE status = ?;"
            params: tuple[str, ...] = (status.value,)
        else:
            query = "SELECT COUNT(*) FROM persons;"
            params = ()

        with self._db.readonly_cursor() as cur:
            cur.execute(query, params)
            row = cur.fetchone()
            return int(row[0]) if row else 0

    def count_embeddings(self, person_id: str | None = None) -> int:
        """Count total embedding records in the repository."""
        if person_id is not None:
            valid_id = validate_uuid(person_id, "person_id")
            query = "SELECT COUNT(*) FROM face_embeddings WHERE person_id = ?;"
            params: tuple[str, ...] = (valid_id,)
        else:
            query = "SELECT COUNT(*) FROM face_embeddings;"
            params = ()

        with self._db.readonly_cursor() as cur:
            cur.execute(query, params)
            row = cur.fetchone()
            return int(row[0]) if row else 0

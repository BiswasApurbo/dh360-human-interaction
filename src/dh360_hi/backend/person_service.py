"""Person Service layer managing person lifecycle semantics, validation, and profile abstraction."""

from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from dh360_hi.backend.exceptions import PersonNotFoundError, ValidationError
from dh360_hi.backend.models import (
    ConsentStatus,
    EmbeddingMetadata,
    EmbeddingRecord,
    PersonProfile,
    PersonRecord,
    PersonStatus,
    utc_now_iso,
    validate_uuid,
    vector_to_blob,
)
from dh360_hi.backend.repository import BasePersonRepository
from dh360_hi.vision.recognition import InMemoryGallery

if TYPE_CHECKING:
    from dh360_hi.vision.identity_state import IdentityStateManager
    from dh360_hi.vision.recognition import FaceRecognizer


@dataclass(frozen=True)
class QAPair:
    """Structured question-and-answer pair for privileged AI context."""

    question: str
    answer: str


@dataclass(frozen=True)
class PersonContextDTO:
    """Application-facing structured representation of person profile context."""

    person_id: str
    display_name: str
    preferred_name: str | None
    role: str | None
    department: str | None
    office_room: str | None
    working_hours: str | None
    profile_description: str | None
    qa_pairs: list[QAPair]


def _validate_context_field(value: str | None, max_len: int, field_name: str) -> str | None:
    """Validate string length for a context field."""
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    if len(cleaned) > max_len:
        msg = f"{field_name} must not exceed {max_len} characters"
        raise ValidationError(msg)
    return cleaned


def _serialize_qa_pairs(qa_pairs: list[QAPair | dict[str, str]] | None) -> str:
    """Validate and serialize Q&A pairs to JSON text."""
    if qa_pairs is None:
        return "[]"
    if not isinstance(qa_pairs, list):
        msg = "qa_pairs must be a list"
        raise ValidationError(msg)
    if len(qa_pairs) > 20:
        msg = "qa_pairs must not exceed 20 pairs"
        raise ValidationError(msg)

    serialized: list[dict[str, str]] = []
    for idx, item in enumerate(qa_pairs):
        if isinstance(item, QAPair):
            q, a = item.question, item.answer
        elif isinstance(item, dict):
            q = item.get("question")  # type: ignore[assignment]
            a = item.get("answer")  # type: ignore[assignment]
        else:
            msg = f"Q&A item at index {idx} must be a QAPair or dict with 'question' and 'answer'"
            raise ValidationError(msg)

        if not isinstance(q, str) or not q.strip():
            msg = f"Question at index {idx} must be a non-empty string"
            raise ValidationError(msg)
        if not isinstance(a, str) or not a.strip():
            msg = f"Answer at index {idx} must be a non-empty string"
            raise ValidationError(msg)

        serialized.append({"question": q.strip(), "answer": a.strip()})

    return json.dumps(serialized)


def _deserialize_qa_pairs(raw_json: str | None) -> list[QAPair]:
    """Safely deserialize stored JSON into a list of QAPair objects.

    Fails safely on malformed JSON or corrupted schemas.
    """
    if not raw_json or not isinstance(raw_json, str):
        return []
    try:
        parsed: Any = json.loads(raw_json)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return []

    if not isinstance(parsed, list):
        return []

    pairs: list[QAPair] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        q = item.get("question")
        a = item.get("answer")
        if isinstance(q, str) and isinstance(a, str) and q.strip() and a.strip():
            pairs.append(QAPair(question=q.strip(), answer=a.strip()))
    return pairs


class BasePersonService(ABC):
    """Abstract interface for Person management and profile lifecycle services."""

    @property
    @abstractmethod
    def repository(self) -> BasePersonRepository:
        """Underlying persistence repository."""

    @abstractmethod
    def create_person(
        self,
        display_name: str,
        preferred_name: str | None = None,
        status: PersonStatus | str = PersonStatus.ACTIVE,
        consent_status: ConsentStatus | str = ConsentStatus.GRANTED,
        notes: str | None = None,
        person_id: str | None = None,
        role: str | None = None,
        department: str | None = None,
        office_room: str | None = None,
        working_hours: str | None = None,
        profile_description: str | None = None,
        qa_pairs: list[QAPair | dict[str, str]] | None = None,
    ) -> PersonProfile:
        """Create and persist a new person profile."""

    @abstractmethod
    def get_person(self, person_id: str) -> PersonProfile | None:
        """Retrieve a person's public profile by ID."""

    @abstractmethod
    def get_person_record(self, person_id: str) -> PersonRecord | None:
        """Retrieve the underlying PersonRecord including raw context fields."""

    @abstractmethod
    def get_person_context(self, person_id: str) -> PersonContextDTO | None:
        """Retrieve structured person context including deserialized Q&A pairs."""

    @abstractmethod
    def list_people(self, status: PersonStatus | str | None = None) -> list[PersonProfile]:
        """List person profiles, optionally filtered by lifecycle status."""

    @abstractmethod
    def update_person(
        self,
        person_id: str,
        display_name: str | None = None,
        preferred_name: str | None = None,
        status: PersonStatus | str | None = None,
        consent_status: ConsentStatus | str | None = None,
        notes: str | None = None,
        role: str | None = None,
        department: str | None = None,
        office_room: str | None = None,
        working_hours: str | None = None,
        profile_description: str | None = None,
        qa_pairs: list[QAPair | dict[str, str]] | None = None,
    ) -> PersonProfile:
        """Update an existing person profile."""

    @abstractmethod
    def delete_person(
        self,
        person_id: str,
        recognizer: FaceRecognizer | None = None,
        identity_manager: IdentityStateManager | None = None,
    ) -> bool:
        """Delete a person and cascade delete their stored embeddings."""

    @abstractmethod
    def add_person_embedding(
        self,
        person_id: str,
        embedding: np.ndarray,
        model_name: str = "SFace",
        model_version: str = "2021dec",
        quality_score: float | None = None,
        enrollment_batch_id: str | None = None,
        embedding_id: str | None = None,
    ) -> EmbeddingMetadata:
        """Store a new face embedding vector for a person and return its metadata."""

    @abstractmethod
    def replace_person_embeddings(
        self,
        person_id: str,
        embeddings: list[np.ndarray],
        model_name: str = "SFace",
        model_version: str = "2021dec",
        quality_scores: list[float | None] | None = None,
        enrollment_batch_id: str | None = None,
        recognizer: FaceRecognizer | None = None,
    ) -> list[EmbeddingMetadata]:
        """Atomically replace all face embeddings for a person with a new set."""

    @abstractmethod
    def add_person_embeddings(
        self,
        person_id: str,
        embeddings: list[np.ndarray],
        model_name: str = "SFace",
        model_version: str = "2021dec",
        quality_scores: list[float | None] | None = None,
        enrollment_batch_id: str | None = None,
        recognizer: FaceRecognizer | None = None,
    ) -> list[EmbeddingMetadata]:
        """Atomically append a batch of face embeddings for a person in one transaction."""

    @abstractmethod
    def get_person_embedding_vectors(self, person_id: str) -> list[np.ndarray]:
        """Retrieve the raw embedding vectors for all of a person's samples."""

    @abstractmethod
    def get_person_embeddings(self, person_id: str) -> list[EmbeddingMetadata]:
        """Retrieve metadata for all embeddings associated with a person."""

    @abstractmethod
    def delete_person_embedding(self, embedding_id: str) -> bool:
        """Delete a specific embedding record by ID."""

    @abstractmethod
    def delete_person_embeddings(
        self,
        person_id: str,
        recognizer: FaceRecognizer | None = None,
        identity_manager: IdentityStateManager | None = None,
    ) -> int:
        """Delete all face embeddings associated with a person while retaining profile."""

    @abstractmethod
    def load_recognition_gallery(
        self,
        model_name: str = "SFace",
        model_version: str = "2021dec",
        require_active: bool = True,
        require_consent: bool = True,
    ) -> InMemoryGallery:
        """Load an isolated InMemoryGallery from persisted database records."""

    def get_setting(self, key: str, default: str | None = None) -> str | None:
        """Retrieve a persistent setting value by key."""
        return self.repository.get_setting(key, default)

    def set_setting(self, key: str, value: str) -> None:
        """Store or update a persistent setting value by key."""
        self.repository.set_setting(key, value)


class PersonService(BasePersonService):
    """Concrete PersonService coordinating domain validation and repository persistence."""

    def __init__(self, repository: BasePersonRepository) -> None:
        self._repo = repository

    @property
    def repository(self) -> BasePersonRepository:
        """Underlying repository instance."""
        return self._repo

    def _record_to_profile(self, rec: PersonRecord, emb_count: int | None = None) -> PersonProfile:
        """Convert a PersonRecord into a public PersonProfile with embedding count."""
        count = (
            emb_count
            if emb_count is not None
            else self._repo.count_embeddings(person_id=rec.person_id)
        )
        return PersonProfile(
            person_id=rec.person_id,
            display_name=rec.display_name,
            preferred_name=rec.preferred_name,
            status=rec.status,
            consent_status=rec.consent_status,
            consent_timestamp=rec.consent_timestamp,
            notes=rec.notes,
            created_at=rec.created_at,
            updated_at=rec.updated_at,
            embedding_count=count,
        )

    def create_person(
        self,
        display_name: str,
        preferred_name: str | None = None,
        status: PersonStatus | str = PersonStatus.ACTIVE,
        consent_status: ConsentStatus | str = ConsentStatus.GRANTED,
        notes: str | None = None,
        person_id: str | None = None,
        role: str | None = None,
        department: str | None = None,
        office_room: str | None = None,
        working_hours: str | None = None,
        profile_description: str | None = None,
        qa_pairs: list[QAPair | dict[str, str]] | None = None,
    ) -> PersonProfile:
        """Create and persist a new person profile."""
        if not display_name or not display_name.strip():
            msg = "display_name cannot be empty or whitespace"
            raise ValidationError(msg)

        pid = person_id if person_id is not None else str(uuid.uuid4())
        valid_pid = validate_uuid(pid, "person_id")

        p_status = PersonStatus(status) if isinstance(status, str) else status
        c_status = (
            ConsentStatus(consent_status) if isinstance(consent_status, str) else consent_status
        )

        valid_role = _validate_context_field(role, 200, "role")
        valid_dept = _validate_context_field(department, 200, "department")
        valid_office = _validate_context_field(office_room, 100, "office_room")
        valid_hours = _validate_context_field(working_hours, 200, "working_hours")
        valid_profile_desc = _validate_context_field(
            profile_description, 2000, "profile_description"
        )
        serialized_qa = _serialize_qa_pairs(qa_pairs)

        now_str = utc_now_iso()
        consent_ts = now_str if c_status == ConsentStatus.GRANTED else None

        record = PersonRecord(
            person_id=valid_pid,
            display_name=display_name.strip(),
            preferred_name=preferred_name.strip() if preferred_name else None,
            status=p_status,
            consent_status=c_status,
            consent_timestamp=consent_ts,
            notes=notes.strip() if notes else None,
            role=valid_role,
            department=valid_dept,
            office_room=valid_office,
            working_hours=valid_hours,
            profile_description=valid_profile_desc,
            ai_qa_pairs_json=serialized_qa,
            created_at=now_str,
            updated_at=now_str,
        )

        persisted = self._repo.create_person(record)
        return self._record_to_profile(persisted, emb_count=0)

    def get_person(self, person_id: str) -> PersonProfile | None:
        """Retrieve a person's public profile by ID."""
        rec = self._repo.get_person(person_id)
        return self._record_to_profile(rec) if rec else None

    def get_person_record(self, person_id: str) -> PersonRecord | None:
        """Retrieve the underlying PersonRecord including raw context fields."""
        return self._repo.get_person(person_id)

    def get_person_context(self, person_id: str) -> PersonContextDTO | None:
        """Retrieve structured person context including deserialized Q&A pairs."""
        rec = self._repo.get_person(person_id)
        if rec is None:
            return None
        return PersonContextDTO(
            person_id=rec.person_id,
            display_name=rec.display_name,
            preferred_name=rec.preferred_name,
            role=rec.role,
            department=rec.department,
            office_room=rec.office_room,
            working_hours=rec.working_hours,
            profile_description=rec.profile_description,
            qa_pairs=_deserialize_qa_pairs(rec.ai_qa_pairs_json),
        )

    def list_people(self, status: PersonStatus | str | None = None) -> list[PersonProfile]:
        """List person profiles, optionally filtered by lifecycle status."""
        target_status = PersonStatus(status) if isinstance(status, str) else status
        records = self._repo.list_persons(status=target_status)
        return [self._record_to_profile(r) for r in records]

    def update_person(
        self,
        person_id: str,
        display_name: str | None = None,
        preferred_name: str | None = None,
        status: PersonStatus | str | None = None,
        consent_status: ConsentStatus | str | None = None,
        notes: str | None = None,
        role: str | None = None,
        department: str | None = None,
        office_room: str | None = None,
        working_hours: str | None = None,
        profile_description: str | None = None,
        qa_pairs: list[QAPair | dict[str, str]] | None = None,
    ) -> PersonProfile:
        """Update an existing person profile."""
        existing = self._repo.get_person(person_id)
        if existing is None:
            msg = f"Person '{person_id}' not found"
            raise PersonNotFoundError(msg)

        new_display_name = (
            display_name.strip() if display_name is not None else existing.display_name
        )
        if not new_display_name:
            msg = "display_name cannot be empty or whitespace"
            raise ValidationError(msg)

        new_preferred_name = (
            preferred_name.strip() if preferred_name is not None else existing.preferred_name
        )
        new_status = (
            (PersonStatus(status) if isinstance(status, str) else status)
            if status is not None
            else existing.status
        )
        new_consent_status = (
            (ConsentStatus(consent_status) if isinstance(consent_status, str) else consent_status)
            if consent_status is not None
            else existing.consent_status
        )
        new_notes = notes.strip() if notes is not None else existing.notes

        new_role = _validate_context_field(role, 200, "role") if role is not None else existing.role
        new_dept = (
            _validate_context_field(department, 200, "department")
            if department is not None
            else existing.department
        )
        new_office = (
            _validate_context_field(office_room, 100, "office_room")
            if office_room is not None
            else existing.office_room
        )
        new_hours = (
            _validate_context_field(working_hours, 200, "working_hours")
            if working_hours is not None
            else existing.working_hours
        )
        new_profile_desc = (
            _validate_context_field(profile_description, 2000, "profile_description")
            if profile_description is not None
            else existing.profile_description
        )
        new_qa_json = (
            _serialize_qa_pairs(qa_pairs) if qa_pairs is not None else existing.ai_qa_pairs_json
        )

        now_str = utc_now_iso()
        consent_ts = existing.consent_timestamp
        if consent_status is not None:
            consent_ts = now_str if new_consent_status == ConsentStatus.GRANTED else None

        updated_rec = PersonRecord(
            person_id=existing.person_id,
            display_name=new_display_name,
            preferred_name=new_preferred_name,
            status=new_status,
            consent_status=new_consent_status,
            consent_timestamp=consent_ts,
            notes=new_notes,
            role=new_role,
            department=new_dept,
            office_room=new_office,
            working_hours=new_hours,
            profile_description=new_profile_desc,
            ai_qa_pairs_json=new_qa_json,
            created_at=existing.created_at,
            updated_at=now_str,
        )

        persisted = self._repo.update_person(updated_rec)
        return self._record_to_profile(persisted)

    def delete_person(
        self,
        person_id: str,
        recognizer: FaceRecognizer | None = None,
        identity_manager: IdentityStateManager | None = None,
    ) -> bool:
        """Delete a person and cascade delete their stored embeddings."""
        success = self._repo.delete_person(person_id)
        if success:
            if recognizer is not None:
                new_gallery = self.load_recognition_gallery()
                recognizer.replace_gallery(new_gallery)
            if identity_manager is not None and hasattr(identity_manager, "invalidate_person"):
                identity_manager.invalidate_person(person_id)
        return success

    def add_person_embedding(
        self,
        person_id: str,
        embedding: np.ndarray,
        model_name: str = "SFace",
        model_version: str = "2021dec",
        quality_score: float | None = None,
        enrollment_batch_id: str | None = None,
        embedding_id: str | None = None,
    ) -> EmbeddingMetadata:
        """Store a new face embedding vector for a person and return its metadata."""
        existing = self._repo.get_person(person_id)
        if existing is None:
            msg = f"Cannot add embedding: person '{person_id}' does not exist"
            raise PersonNotFoundError(msg)

        eid = embedding_id if embedding_id is not None else str(uuid.uuid4())
        valid_eid = validate_uuid(eid, "embedding_id")

        blob = vector_to_blob(embedding, expected_dim=128)
        now_str = utc_now_iso()

        emb_rec = EmbeddingRecord(
            embedding_id=valid_eid,
            person_id=existing.person_id,
            embedding=blob,
            dimension=128,
            model_name=model_name,
            model_version=model_version,
            quality_score=quality_score,
            enrollment_batch_id=enrollment_batch_id,
            created_at=now_str,
        )

        persisted = self._repo.add_embedding(emb_rec)
        return EmbeddingMetadata(
            embedding_id=persisted.embedding_id,
            person_id=persisted.person_id,
            dimension=persisted.dimension,
            model_name=persisted.model_name,
            model_version=persisted.model_version,
            quality_score=persisted.quality_score,
            enrollment_batch_id=persisted.enrollment_batch_id,
            created_at=persisted.created_at,
        )

    def replace_person_embeddings(
        self,
        person_id: str,
        embeddings: list[np.ndarray],
        model_name: str = "SFace",
        model_version: str = "2021dec",
        quality_scores: list[float | None] | None = None,
        enrollment_batch_id: str | None = None,
        recognizer: FaceRecognizer | None = None,
    ) -> list[EmbeddingMetadata]:
        """Atomically replace all face embeddings for a person with a new set."""
        existing = self._repo.get_person(person_id)
        if existing is None:
            msg = f"Cannot replace embeddings: person '{person_id}' not found"
            raise PersonNotFoundError(msg)

        batch_id = enrollment_batch_id or str(uuid.uuid4())
        now_str = utc_now_iso()
        records: list[EmbeddingRecord] = []
        for idx, emb in enumerate(embeddings):
            eid = str(uuid.uuid4())
            blob = vector_to_blob(emb, expected_dim=128)
            q_score = (
                quality_scores[idx]
                if quality_scores is not None and idx < len(quality_scores)
                else None
            )
            records.append(
                EmbeddingRecord(
                    embedding_id=eid,
                    person_id=existing.person_id,
                    embedding=blob,
                    dimension=128,
                    model_name=model_name,
                    model_version=model_version,
                    quality_score=q_score,
                    enrollment_batch_id=batch_id,
                    created_at=now_str,
                )
            )

        persisted = self._repo.replace_embeddings_for_person(existing.person_id, records)
        if recognizer is not None:
            new_gallery = self.load_recognition_gallery(
                model_name=model_name,
                model_version=model_version,
            )
            recognizer.replace_gallery(new_gallery)

        return [
            EmbeddingMetadata(
                embedding_id=r.embedding_id,
                person_id=r.person_id,
                dimension=r.dimension,
                model_name=r.model_name,
                model_version=r.model_version,
                quality_score=r.quality_score,
                enrollment_batch_id=r.enrollment_batch_id,
                created_at=r.created_at,
            )
            for r in persisted
        ]

    def add_person_embeddings(
        self,
        person_id: str,
        embeddings: list[np.ndarray],
        model_name: str = "SFace",
        model_version: str = "2021dec",
        quality_scores: list[float | None] | None = None,
        enrollment_batch_id: str | None = None,
        recognizer: FaceRecognizer | None = None,
    ) -> list[EmbeddingMetadata]:
        """Atomically append a batch of face embeddings for a person.

        Persists all supplied vectors in a single transaction (via the repository's
        batch writer) and, when a recognizer is supplied, reloads the recognition
        gallery exactly once so the in-memory index stays consistent with storage.
        """
        existing = self._repo.get_person(person_id)
        if existing is None:
            msg = f"Cannot add embeddings: person '{person_id}' not found"
            raise PersonNotFoundError(msg)

        batch_id = enrollment_batch_id or str(uuid.uuid4())
        now_str = utc_now_iso()
        records: list[EmbeddingRecord] = []
        for idx, emb in enumerate(embeddings):
            eid = str(uuid.uuid4())
            blob = vector_to_blob(emb, expected_dim=128)
            q_score = (
                quality_scores[idx]
                if quality_scores is not None and idx < len(quality_scores)
                else None
            )
            records.append(
                EmbeddingRecord(
                    embedding_id=eid,
                    person_id=existing.person_id,
                    embedding=blob,
                    dimension=128,
                    model_name=model_name,
                    model_version=model_version,
                    quality_score=q_score,
                    enrollment_batch_id=batch_id,
                    created_at=now_str,
                )
            )

        persisted = self._repo.add_embeddings(records)
        if recognizer is not None:
            new_gallery = self.load_recognition_gallery(
                model_name=model_name,
                model_version=model_version,
            )
            recognizer.replace_gallery(new_gallery)

        return [
            EmbeddingMetadata(
                embedding_id=r.embedding_id,
                person_id=r.person_id,
                dimension=r.dimension,
                model_name=r.model_name,
                model_version=r.model_version,
                quality_score=r.quality_score,
                enrollment_batch_id=r.enrollment_batch_id,
                created_at=r.created_at,
            )
            for r in persisted
        ]

    def get_person_embedding_vectors(self, person_id: str) -> list[np.ndarray]:
        """Retrieve the raw embedding vectors for all of a person's samples."""
        existing = self._repo.get_person(person_id)
        if existing is None:
            msg = f"Person '{person_id}' not found"
            raise PersonNotFoundError(msg)
        return [r.vector for r in self._repo.get_embeddings_for_person(person_id)]

    def get_person_embeddings(self, person_id: str) -> list[EmbeddingMetadata]:
        """Retrieve metadata for all embeddings associated with a person."""
        existing = self._repo.get_person(person_id)
        if existing is None:
            msg = f"Person '{person_id}' not found"
            raise PersonNotFoundError(msg)

        records = self._repo.get_embeddings_for_person(person_id)
        return [
            EmbeddingMetadata(
                embedding_id=r.embedding_id,
                person_id=r.person_id,
                dimension=r.dimension,
                model_name=r.model_name,
                model_version=r.model_version,
                quality_score=r.quality_score,
                enrollment_batch_id=r.enrollment_batch_id,
                created_at=r.created_at,
            )
            for r in records
        ]

    def delete_person_embedding(self, embedding_id: str) -> bool:
        """Delete a specific embedding record by ID."""
        return self._repo.delete_embedding(embedding_id)

    def delete_person_embeddings(
        self,
        person_id: str,
        recognizer: FaceRecognizer | None = None,
        identity_manager: IdentityStateManager | None = None,
    ) -> int:
        """Delete all face embeddings for a person, syncing gallery and identity state."""
        existing = self._repo.get_person(person_id)
        if existing is None:
            msg = f"Person '{person_id}' not found"
            raise PersonNotFoundError(msg)

        count = self._repo.delete_embeddings_for_person(person_id)
        if recognizer is not None:
            new_gallery = self.load_recognition_gallery()
            recognizer.replace_gallery(new_gallery)
        if identity_manager is not None and hasattr(identity_manager, "invalidate_person"):
            identity_manager.invalidate_person(person_id)
        return count

    def load_recognition_gallery(
        self,
        model_name: str = "SFace",
        model_version: str = "2021dec",
        require_active: bool = True,
        require_consent: bool = True,
    ) -> InMemoryGallery:
        """Load an isolated InMemoryGallery from persisted database records."""
        from dh360_hi.backend.gallery_loader import load_gallery_from_repository

        return load_gallery_from_repository(
            repository=self._repo,
            model_name=model_name,
            model_version=model_version,
            require_active=require_active,
            require_consent=require_consent,
        )

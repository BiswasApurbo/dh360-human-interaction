"""Unit tests for SQLitePersonRepository, referential integrity, and cascading operations."""

from __future__ import annotations

import numpy as np
import pytest

from dh360_hi.backend.database import DatabaseManager
from dh360_hi.backend.exceptions import (
    DuplicateEmbeddingError,
    DuplicatePersonError,
    ForeignKeyViolationError,
    PersonNotFoundError,
    ValidationError,
)
from dh360_hi.backend.models import (
    ConsentStatus,
    EmbeddingRecord,
    PersonRecord,
    PersonStatus,
    blob_to_vector,
    validate_uuid,
    vector_to_blob,
)
from dh360_hi.backend.repository import SQLitePersonRepository


@pytest.fixture
def repo() -> SQLitePersonRepository:
    """Provide a fresh SQLitePersonRepository with an in-memory database."""
    db = DatabaseManager(":memory:")
    return SQLitePersonRepository(db)


def test_vector_blob_serialization_and_validation() -> None:
    """Test vector float32 serialization, dimension validation, and deserialization."""
    vec = np.random.randn(128).astype(np.float32)
    blob = vector_to_blob(vec, expected_dim=128)
    assert len(blob) == 512

    recovered = blob_to_vector(blob, expected_dim=128)
    assert np.allclose(vec, recovered)

    # Wrong dimension
    with pytest.raises(ValidationError, match="Embedding dimension 64"):
        vector_to_blob(np.zeros(64, dtype=np.float32), expected_dim=128)

    # Non-finite values
    nan_vec = np.ones(128, dtype=np.float32)
    nan_vec[0] = np.nan
    with pytest.raises(ValidationError, match="NaN or infinite"):
        vector_to_blob(nan_vec, expected_dim=128)

    # Invalid blob length
    with pytest.raises(ValidationError, match="byte length 10"):
        blob_to_vector(b"0123456789", expected_dim=128)


def test_uuid_validation() -> None:
    """Test UUID format validation."""
    valid_uuid = "a1b2c3d4-e5f6-4a5b-8c9d-0e1f2a3b4c5d"
    assert validate_uuid(valid_uuid) == valid_uuid

    with pytest.raises(ValidationError, match="Invalid UUID"):
        validate_uuid("not-a-uuid")

    with pytest.raises(ValidationError, match="non-empty string"):
        validate_uuid("")


def test_person_crud_operations(repo: SQLitePersonRepository) -> None:
    """Test standard CRUD operations on person records."""
    pid = "10000000-0000-0000-0000-000000000001"
    person = PersonRecord(
        person_id=pid,
        display_name="Dr. Jane Doe",
        preferred_name="Jane",
        status=PersonStatus.ACTIVE,
        consent_status=ConsentStatus.GRANTED,
        notes="Cardiology department head",
    )

    # 1. Create
    created = repo.create_person(person)
    assert created.person_id == pid
    assert repo.count_persons() == 1

    # 2. Duplicate rejection
    with pytest.raises(DuplicatePersonError):
        repo.create_person(person)

    # 3. Read
    fetched = repo.get_person(pid)
    assert fetched is not None
    assert fetched.display_name == "Dr. Jane Doe"
    assert fetched.preferred_name == "Jane"
    assert fetched.status == PersonStatus.ACTIVE

    # 4. Update
    updated_rec = PersonRecord(
        person_id=pid,
        display_name="Dr. Jane Smith",
        preferred_name="Jane S.",
        status=PersonStatus.INACTIVE,
        consent_status=ConsentStatus.WITHDRAWN,
        notes="Transferred",
    )
    repo.update_person(updated_rec)
    refetched = repo.get_person(pid)
    assert refetched is not None
    assert refetched.display_name == "Dr. Jane Smith"
    assert refetched.status == PersonStatus.INACTIVE
    assert refetched.consent_status == ConsentStatus.WITHDRAWN

    # 5. List with filters
    active_list = repo.list_persons(status=PersonStatus.ACTIVE)
    assert len(active_list) == 0
    inactive_list = repo.list_persons(status=PersonStatus.INACTIVE)
    assert len(inactive_list) == 1

    # 6. Delete
    assert repo.delete_person(pid) is True
    assert repo.get_person(pid) is None
    assert repo.count_persons() == 0


def test_update_nonexistent_person_raises_error(repo: SQLitePersonRepository) -> None:
    """Test updating a non-existent person raises PersonNotFoundError."""
    missing = PersonRecord(
        person_id="99999999-9999-9999-9999-999999999999",
        display_name="Nobody",
    )
    with pytest.raises(PersonNotFoundError):
        repo.update_person(missing)


def test_embedding_crud_and_referential_integrity(repo: SQLitePersonRepository) -> None:
    """Test embedding storage, batch insertion, retrieval, and cascade deletion."""
    pid = "20000000-0000-0000-0000-000000000002"
    repo.create_person(
        PersonRecord(
            person_id=pid,
            display_name="Patient Two",
        )
    )

    # 1. Foreign key violation when person does not exist
    vec = np.random.randn(128).astype(np.float32)
    with pytest.raises(ForeignKeyViolationError):
        repo.add_embedding(
            EmbeddingRecord(
                embedding_id="e0000000-0000-0000-0000-000000000001",
                person_id="99999999-9999-9999-9999-999999999999",
                embedding=vector_to_blob(vec),
            )
        )

    # 2. Add single embedding
    emb1 = EmbeddingRecord(
        embedding_id="e0000000-0000-0000-0000-000000000001",
        person_id=pid,
        embedding=vector_to_blob(vec),
        dimension=128,
        model_name="SFace",
        model_version="2021dec",
        quality_score=0.92,
    )
    repo.add_embedding(emb1)
    assert repo.count_embeddings(person_id=pid) == 1
    assert repo.count_embeddings() == 1

    # 3. Duplicate embedding ID rejection
    with pytest.raises(DuplicateEmbeddingError):
        repo.add_embedding(emb1)

    # 4. Batch add embeddings
    emb2 = EmbeddingRecord(
        embedding_id="e0000000-0000-0000-0000-000000000002",
        person_id=pid,
        embedding=vector_to_blob(np.random.randn(128).astype(np.float32)),
        model_name="SFace",
        model_version="2021dec",
    )
    emb3 = EmbeddingRecord(
        embedding_id="e0000000-0000-0000-0000-000000000003",
        person_id=pid,
        embedding=vector_to_blob(np.random.randn(128).astype(np.float32)),
        model_name="MobileFaceNet",
        model_version="2024",
    )
    repo.add_embeddings([emb2, emb3])
    assert repo.count_embeddings(person_id=pid) == 3

    # 5. Model version filtering
    sface_embs = repo.get_all_embeddings(model_name="SFace", model_version="2021dec")
    assert len(sface_embs) == 2
    mfn_embs = repo.get_all_embeddings(model_name="MobileFaceNet")
    assert len(mfn_embs) == 1

    # 6. Delete single embedding
    assert repo.delete_embedding("e0000000-0000-0000-0000-000000000001") is True
    assert repo.get_embedding("e0000000-0000-0000-0000-000000000001") is None
    assert repo.count_embeddings(person_id=pid) == 2

    # 7. Foreign key CASCADE delete on person deletion
    assert repo.delete_person(pid) is True
    assert repo.get_person(pid) is None
    # All embeddings for this person MUST be automatically deleted
    assert repo.count_embeddings(person_id=pid) == 0
    assert repo.count_embeddings() == 0

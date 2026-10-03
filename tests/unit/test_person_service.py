"""Unit tests for PersonService business logic, profile privacy, validation, and gallery loading."""

from __future__ import annotations

import numpy as np
import pytest

from dh360_hi.backend.database import DatabaseManager
from dh360_hi.backend.exceptions import PersonNotFoundError, ValidationError
from dh360_hi.backend.models import ConsentStatus, PersonStatus
from dh360_hi.backend.person_service import PersonService
from dh360_hi.backend.repository import SQLitePersonRepository
from dh360_hi.vision.recognition import FaceRecognizer


@pytest.fixture
def service() -> PersonService:
    """Provide a fresh PersonService backed by an in-memory SQLite database."""
    db = DatabaseManager(":memory:")
    repo = SQLitePersonRepository(db)
    return PersonService(repo)


def test_create_person_validation_and_defaults(service: PersonService) -> None:
    """Test person creation defaults, UUID generation, and field validation."""
    # 1. Successful creation with defaults
    profile = service.create_person(display_name="Sarah Connor")
    assert profile.person_id is not None
    assert len(profile.person_id) == 36
    assert profile.display_name == "Sarah Connor"
    assert profile.status == PersonStatus.ACTIVE
    assert profile.consent_status == ConsentStatus.GRANTED
    assert profile.consent_timestamp is not None
    assert profile.embedding_count == 0

    # 2. Rejection of empty or whitespace name
    with pytest.raises(ValidationError, match="display_name cannot be empty"):
        service.create_person(display_name="   ")

    # 3. Custom person_id validation
    with pytest.raises(ValidationError, match="Invalid UUID format"):
        service.create_person(display_name="John Doe", person_id="invalid-uuid")


def test_person_lifecycle_and_updates(service: PersonService) -> None:
    """Test updating person status, consent, and profile fields."""
    profile = service.create_person(
        display_name="John Connor",
        preferred_name="Johnny",
        status=PersonStatus.ACTIVE,
        consent_status=ConsentStatus.PENDING,
    )
    assert profile.consent_timestamp is None

    # Update to GRANTED updates consent_timestamp
    updated = service.update_person(
        person_id=profile.person_id,
        display_name="John Connor Jr.",
        consent_status=ConsentStatus.GRANTED,
        notes="Consent signed on tablet",
    )
    assert updated.display_name == "John Connor Jr."
    assert updated.consent_status == ConsentStatus.GRANTED
    assert updated.consent_timestamp is not None
    assert updated.notes == "Consent signed on tablet"

    # Non-existent person update
    with pytest.raises(PersonNotFoundError):
        service.update_person(
            person_id="99999999-9999-9999-9999-999999999999",
            display_name="Ghost",
        )


def test_embedding_management_and_privacy(service: PersonService) -> None:
    """Test adding embeddings, ensuring raw vectors are NOT exposed through profile/metadata API."""
    profile = service.create_person(display_name="Ellen Ripley")

    # Add 2 synthetic embeddings
    vec1 = np.random.randn(128).astype(np.float32)
    vec2 = np.random.randn(128).astype(np.float32)

    meta1 = service.add_person_embedding(
        person_id=profile.person_id,
        embedding=vec1,
        model_name="SFace",
        model_version="2021dec",
        quality_score=0.95,
    )
    assert meta1.embedding_id is not None
    assert meta1.person_id == profile.person_id
    assert meta1.dimension == 128
    assert not hasattr(meta1, "embedding")  # Vector is not in metadata object

    meta2 = service.add_person_embedding(
        person_id=profile.person_id,
        embedding=vec2,
        model_name="SFace",
        model_version="2021dec",
    )
    assert meta2.embedding_id != meta1.embedding_id

    # Check updated embedding count in profile
    refetched = service.get_person(profile.person_id)
    assert refetched is not None
    assert refetched.embedding_count == 2

    # Get embeddings metadata list
    embs_meta = service.get_person_embeddings(profile.person_id)
    assert len(embs_meta) == 2
    for meta in embs_meta:
        assert meta.person_id == profile.person_id
        assert not hasattr(meta, "embedding")

    # Delete single embedding
    assert service.delete_person_embedding(meta1.embedding_id) is True
    assert len(service.get_person_embeddings(profile.person_id)) == 1
    assert service.get_person(profile.person_id).embedding_count == 1  # type: ignore[union-attr]


def test_load_recognition_gallery_filtering(service: PersonService) -> None:
    """Test loading recognition gallery with model version, status, and consent filters."""
    # Person 1: Active + Consented + SFace 2021dec
    p1 = service.create_person("Active Patient", consent_status=ConsentStatus.GRANTED)
    service.add_person_embedding(
        person_id=p1.person_id,
        embedding=np.random.randn(128).astype(np.float32),
        model_name="SFace",
        model_version="2021dec",
    )

    # Person 2: Active + Consented + Other Model
    p2 = service.create_person("Other Model Patient", consent_status=ConsentStatus.GRANTED)
    service.add_person_embedding(
        person_id=p2.person_id,
        embedding=np.random.randn(128).astype(np.float32),
        model_name="ArcFace",
        model_version="v1",
    )

    # Person 3: Inactive + Consented + SFace 2021dec
    p3 = service.create_person(
        "Inactive Patient",
        status=PersonStatus.INACTIVE,
        consent_status=ConsentStatus.GRANTED,
    )
    service.add_person_embedding(
        person_id=p3.person_id,
        embedding=np.random.randn(128).astype(np.float32),
        model_name="SFace",
        model_version="2021dec",
    )

    # Person 4: Active + Consent Withdrawn + SFace 2021dec
    p4 = service.create_person(
        "Withdrawn Patient",
        status=PersonStatus.ACTIVE,
        consent_status=ConsentStatus.WITHDRAWN,
    )
    service.add_person_embedding(
        person_id=p4.person_id,
        embedding=np.random.randn(128).astype(np.float32),
        model_name="SFace",
        model_version="2021dec",
    )

    # 1. Load default gallery (SFace 2021dec, active, consented)
    gallery = service.load_recognition_gallery(model_name="SFace", model_version="2021dec")
    assert len(gallery) == 1
    assert gallery.get_entry(p1.person_id) is not None
    assert gallery.get_entry(p2.person_id) is None
    assert gallery.get_entry(p3.person_id) is None
    assert gallery.get_entry(p4.person_id) is None

    # Verify vector normalization in gallery
    entry = gallery.get_entry(p1.person_id)
    assert entry is not None
    assert len(entry.embeddings) == 1
    norm = np.linalg.norm(entry.embeddings[0])
    assert np.isclose(norm, 1.0, atol=1e-5)


def test_batch_add_and_vector_retrieval(service: PersonService) -> None:
    """Test M3.2.3 batch persistence, single-batch attribution, and raw vector retrieval."""
    profile = service.create_person("Batch Patient")

    vectors = [np.random.randn(128).astype(np.float32) for _ in range(3)]
    metas = service.add_person_embeddings(
        profile.person_id,
        vectors,
        quality_scores=[0.9, 0.8, 0.7],
        enrollment_batch_id="11111111-1111-1111-1111-111111111111",
        model_name="SFace",
        model_version="2021dec",
    )
    assert len(metas) == 3
    assert {m.enrollment_batch_id for m in metas} == {"11111111-1111-1111-1111-111111111111"}
    assert all(not hasattr(m, "embedding") for m in metas)
    assert service.get_person(profile.person_id).embedding_count == 3  # type: ignore[union-attr]

    # Raw vectors are retrievable for diversity/duplicate checks and are additive.
    raw = service.get_person_embedding_vectors(profile.person_id)
    assert len(raw) == 3
    assert all(v.shape == (128,) for v in raw)

    extra = service.add_person_embeddings(
        profile.person_id, [np.random.randn(128).astype(np.float32)]
    )
    assert len(extra) == 1
    assert service.get_person(profile.person_id).embedding_count == 4  # type: ignore[union-attr]


def test_batch_add_reloads_gallery_via_recognizer(service: PersonService) -> None:
    """Test that a supplied recognizer receives a single gallery refresh after batch add."""
    profile = service.create_person("Gallery Batch")
    recognizer = FaceRecognizer(gallery=service.load_recognition_gallery())

    metas = service.add_person_embeddings(
        profile.person_id,
        [np.random.randn(128).astype(np.float32) for _ in range(2)],
        enrollment_batch_id="22222222-2222-2222-2222-222222222222",
        recognizer=recognizer,
    )
    assert len(metas) == 2
    entry = recognizer.gallery.get_entry(profile.person_id)
    assert entry is not None
    assert len(entry.embeddings) == 2


def test_vector_retrieval_missing_person_raises(service: PersonService) -> None:
    """Test that vector retrieval and batch add raise for an unknown person."""
    missing = "99999999-9999-9999-9999-999999999999"
    with pytest.raises(PersonNotFoundError):
        service.get_person_embedding_vectors(missing)
    with pytest.raises(PersonNotFoundError):
        service.add_person_embeddings(missing, [np.random.randn(128).astype(np.float32)])

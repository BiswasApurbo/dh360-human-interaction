"""Unit tests for PersonService lifecycle operations, privacy boundaries, and gallery sync."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dh360_hi.backend.database import DatabaseManager
from dh360_hi.backend.exceptions import PersonNotFoundError, ValidationError
from dh360_hi.backend.models import ConsentStatus, PersonStatus
from dh360_hi.backend.person_service import PersonService
from dh360_hi.backend.repository import SQLitePersonRepository
from dh360_hi.vision.embedder import normalize_l2
from dh360_hi.vision.identity_state import IdentityState, IdentityStateManager, TrackIdentityState
from dh360_hi.vision.recognition import FaceRecognizer, RecognitionOutcome


@pytest.fixture
def memory_service() -> tuple[PersonService, SQLitePersonRepository, DatabaseManager]:
    """Provide an in-memory PersonService instance."""
    db = DatabaseManager(":memory:")
    repo = SQLitePersonRepository(db)
    service = PersonService(repo)
    return service, repo, db


def test_view_and_list_people_with_embedding_counts(memory_service: tuple) -> None:
    """Test retrieving individual person profile and listing people with embedding counts."""
    service, _, _ = memory_service

    p1 = service.create_person("Alice Smith", preferred_name="Alice", notes="Cardiology")
    p2 = service.create_person("Bob Jones", status=PersonStatus.INACTIVE)

    # 1. View individual person
    profile1 = service.get_person(p1.person_id)
    assert profile1 is not None
    assert profile1.person_id == p1.person_id
    assert profile1.display_name == "Alice Smith"
    assert profile1.preferred_name == "Alice"
    assert profile1.notes == "Cardiology"
    assert profile1.status == PersonStatus.ACTIVE
    assert profile1.consent_status == ConsentStatus.GRANTED
    assert profile1.embedding_count == 0

    # 2. Add embeddings to Alice
    v1 = normalize_l2(np.ones(128, dtype=np.float32))
    v2 = normalize_l2(np.ones(128, dtype=np.float32) * 2.0)
    service.add_person_embedding(p1.person_id, v1)
    service.add_person_embedding(p1.person_id, v2)

    profile1_updated = service.get_person(p1.person_id)
    assert profile1_updated is not None
    assert profile1_updated.embedding_count == 2

    # 3. List people - all
    all_people = service.list_people()
    assert len(all_people) == 2
    assert {p.display_name for p in all_people} == {"Alice Smith", "Bob Jones"}

    # 4. List people - active only
    active_people = service.list_people(status=PersonStatus.ACTIVE)
    assert len(active_people) == 1
    assert active_people[0].person_id == p1.person_id

    # 5. List people - inactive only
    inactive_people = service.list_people(status=PersonStatus.INACTIVE)
    assert len(inactive_people) == 1
    assert inactive_people[0].person_id == p2.person_id


def test_edit_basic_profile_and_consent_preservation(memory_service: tuple) -> None:
    """Test editing profile metadata, ensuring consent is never silently altered."""
    service, _, _ = memory_service

    person = service.create_person("Charlie Brown", preferred_name="Chuck")
    initial_consent_ts = person.consent_timestamp
    assert initial_consent_ts is not None

    # 1. Update preferred name and notes without touching consent
    updated = service.update_person(
        person.person_id,
        preferred_name="Charlie",
        notes="Updated notes",
    )
    assert updated.display_name == "Charlie Brown"
    assert updated.preferred_name == "Charlie"
    assert updated.notes == "Updated notes"
    assert updated.consent_status == ConsentStatus.GRANTED
    assert updated.consent_timestamp == initial_consent_ts  # Preserved

    # 2. Update status to INACTIVE
    updated_status = service.update_person(person.person_id, status=PersonStatus.INACTIVE)
    assert updated_status.status == PersonStatus.INACTIVE
    assert updated_status.consent_status == ConsentStatus.GRANTED
    assert updated_status.consent_timestamp == initial_consent_ts

    # 3. Explicitly update consent to WITHDRAWN
    withdrawn = service.update_person(person.person_id, consent_status=ConsentStatus.WITHDRAWN)
    assert withdrawn.consent_status == ConsentStatus.WITHDRAWN
    assert withdrawn.consent_timestamp is None

    # 4. Explicitly re-grant consent
    regranted = service.update_person(person.person_id, consent_status=ConsentStatus.GRANTED)
    assert regranted.consent_status == ConsentStatus.GRANTED
    assert regranted.consent_timestamp is not None
    assert regranted.consent_timestamp != initial_consent_ts


def test_profile_update_validation_errors(memory_service: tuple) -> None:
    """Test validation errors when updating profile with invalid parameters."""
    service, _, _ = memory_service
    person = service.create_person("David Miller")

    # Non-existent person
    with pytest.raises(PersonNotFoundError):
        service.update_person("00000000-0000-0000-0000-000000000000", display_name="Ghost")

    # Empty display name
    with pytest.raises(ValidationError, match="display_name cannot be empty"):
        service.update_person(person.person_id, display_name="   ")


def test_person_profile_privacy_boundary(memory_service: tuple) -> None:
    """Test that PersonProfile and EmbeddingMetadata never expose raw vector arrays or blobs."""
    service, _, _ = memory_service
    person = service.create_person("Eva Green")

    vec = normalize_l2(np.random.randn(128).astype(np.float32))
    meta = service.add_person_embedding(person.person_id, vec, quality_score=0.92)

    # EmbeddingMetadata has metadata but no vector / bytes attribute
    assert hasattr(meta, "embedding_id")
    assert hasattr(meta, "dimension")
    assert hasattr(meta, "model_name")
    assert hasattr(meta, "quality_score")
    assert not hasattr(meta, "vector")
    assert not hasattr(meta, "embedding")

    # PersonProfile has count but no vector details
    profile = service.get_person(person.person_id)
    assert profile is not None
    assert profile.embedding_count == 1
    assert not hasattr(profile, "embeddings")
    assert not hasattr(profile, "vectors")


def test_replace_person_embeddings_programmatic(memory_service: tuple) -> None:
    """Test replacing face embeddings programmatically with atomic gallery synchronization."""
    service, repo, _ = memory_service
    recognizer = FaceRecognizer()

    person = service.create_person("Frank Wright")
    old_vecs = [normalize_l2(np.ones(128, dtype=np.float32) * (i + 1)) for i in range(3)]
    for v in old_vecs:
        service.add_person_embedding(person.person_id, v)

    gallery_initial = service.load_recognition_gallery()
    recognizer._gallery = gallery_initial
    assert repo.count_embeddings(person.person_id) == 3
    assert len(gallery_initial.get_entry(person.person_id).embeddings) == 3

    # Replace with 2 new vectors
    new_vecs = [normalize_l2(np.ones(128, dtype=np.float32) * (i + 10)) for i in range(2)]
    replaced_meta = service.replace_person_embeddings(
        person.person_id,
        new_vecs,
        recognizer=recognizer,
    )

    assert len(replaced_meta) == 2
    assert repo.count_embeddings(person.person_id) == 2
    assert len(recognizer._gallery.get_entry(person.person_id).embeddings) == 2


def test_delete_person_embeddings_resets_biometrics(memory_service: tuple) -> None:
    """Test deleting person embeddings while keeping the person profile intact."""
    service, repo, _ = memory_service
    recognizer = FaceRecognizer()
    identity_manager = IdentityStateManager()

    person = service.create_person("Grace Hopper")
    vec = normalize_l2(np.ones(128, dtype=np.float32))
    service.add_person_embedding(person.person_id, vec)

    recognizer._gallery = service.load_recognition_gallery()
    assert len(recognizer._gallery) == 1

    # Simulate active track in IDENTIFIED state
    track_state = TrackIdentityState(
        track_id=1,
        state=IdentityState.IDENTIFIED,
        current_person_id=person.person_id,
        current_display_name=person.display_name,
    )
    identity_manager._tracks[1] = track_state

    # Delete embeddings
    deleted_count = service.delete_person_embeddings(
        person.person_id,
        recognizer=recognizer,
        identity_manager=identity_manager,
    )
    assert deleted_count == 1
    assert repo.count_embeddings(person.person_id) == 0

    # Person still exists
    profile = service.get_person(person.person_id)
    assert profile is not None
    assert profile.embedding_count == 0

    # Recognizer gallery updated (empty of embeddings for Grace)
    assert len(recognizer._gallery) == 0

    # Track state invalidated
    assert track_state.state == IdentityState.UNREGISTERED
    assert track_state.current_person_id is None


def test_delete_person_cascade_and_identity_invalidation(memory_service: tuple) -> None:
    """Test full person deletion with cascade, gallery reload, and track invalidation."""
    service, repo, _ = memory_service
    recognizer = FaceRecognizer()
    identity_manager = IdentityStateManager()

    p1 = service.create_person("Person To Delete")
    p2 = service.create_person("Person To Keep")

    v1 = normalize_l2(np.ones(128, dtype=np.float32))
    v2 = normalize_l2(np.ones(128, dtype=np.float32) * 5.0)
    service.add_person_embedding(p1.person_id, v1)
    service.add_person_embedding(p2.person_id, v2)

    recognizer._gallery = service.load_recognition_gallery()
    assert len(recognizer._gallery) == 2

    # Active track for Person 1
    track1 = TrackIdentityState(
        track_id=10,
        state=IdentityState.IDENTIFIED,
        current_person_id=p1.person_id,
        current_display_name=p1.display_name,
    )
    # Active track for Person 2
    track2 = TrackIdentityState(
        track_id=20,
        state=IdentityState.IDENTIFIED,
        current_person_id=p2.person_id,
        current_display_name=p2.display_name,
    )
    identity_manager._tracks[10] = track1
    identity_manager._tracks[20] = track2

    # Delete Person 1
    success = service.delete_person(
        p1.person_id,
        recognizer=recognizer,
        identity_manager=identity_manager,
    )
    assert success is True
    assert service.get_person(p1.person_id) is None
    assert repo.count_embeddings(p1.person_id) == 0  # Cascade deleted

    # Person 2 intact
    assert service.get_person(p2.person_id) is not None
    assert repo.count_embeddings(p2.person_id) == 1

    # Gallery updated: only Person 2 remains
    assert len(recognizer._gallery) == 1
    assert recognizer._gallery.get_entry(p1.person_id) is None
    assert recognizer._gallery.get_entry(p2.person_id) is not None

    # Track 10 invalidated; Track 20 untouched
    assert track1.state == IdentityState.UNREGISTERED
    assert track1.current_person_id is None
    assert track2.state == IdentityState.IDENTIFIED
    assert track2.current_person_id == p2.person_id


def test_empty_gallery_recognition_after_deleting_all_people(memory_service: tuple) -> None:
    """Test that after deleting all people, gallery is empty and recognizer returns NO_MATCH."""
    service, _, _ = memory_service
    recognizer = FaceRecognizer()

    person = service.create_person("Solo Person")
    service.add_person_embedding(person.person_id, normalize_l2(np.ones(128, dtype=np.float32)))

    recognizer._gallery = service.load_recognition_gallery()
    assert len(recognizer._gallery) == 1

    service.delete_person(person.person_id, recognizer=recognizer)
    assert len(recognizer._gallery) == 0

    # Query recognizer against empty gallery
    res = recognizer.recognize(normalize_l2(np.ones(128, dtype=np.float32)))
    assert res.outcome == RecognitionOutcome.NO_MATCH
    assert res.top_match is None
    assert "empty" in res.details.lower()


def test_persistence_across_simulated_service_restart(tmp_path: Path) -> None:
    """Test that records, profiles, embeddings, and gallery reload survive app restart."""
    db_file = tmp_path / "lifecycle_persistence.db"

    # 1. First Session: create DB, person, embeddings
    db1 = DatabaseManager(db_file)
    repo1 = SQLitePersonRepository(db1)
    service1 = PersonService(repo1)

    p = service1.create_person("Dr. Gregory House", preferred_name="House", notes="Diagnostics")
    vec = normalize_l2(np.arange(128, dtype=np.float32))
    service1.add_person_embedding(p.person_id, vec, quality_score=0.98)
    db1.close()

    # 2. Second Session: Re-open same DB file
    db2 = DatabaseManager(db_file)
    repo2 = SQLitePersonRepository(db2)
    service2 = PersonService(repo2)

    loaded_person = service2.get_person(p.person_id)
    assert loaded_person is not None
    assert loaded_person.display_name == "Dr. Gregory House"
    assert loaded_person.preferred_name == "House"
    assert loaded_person.notes == "Diagnostics"
    assert loaded_person.embedding_count == 1

    gallery = service2.load_recognition_gallery()
    assert len(gallery) == 1
    entry = gallery.get_entry(p.person_id)
    assert entry is not None
    assert entry.display_name == "House"
    assert len(entry.embeddings) == 1
    assert np.allclose(entry.embeddings[0], vec, atol=1e-5)

    db2.close()

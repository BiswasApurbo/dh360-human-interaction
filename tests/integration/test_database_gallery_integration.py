"""End-to-End integration test: SQLite persistence -> Gallery Loader -> FaceRecognizer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from dh360_hi.backend.database import DatabaseManager
from dh360_hi.backend.models import ConsentStatus, PersonStatus
from dh360_hi.backend.person_service import PersonService
from dh360_hi.backend.repository import SQLitePersonRepository
from dh360_hi.vision.embedder import normalize_l2
from dh360_hi.vision.recognition import FaceRecognizer, RecognitionOutcome

if TYPE_CHECKING:
    from pathlib import Path


def test_sqlite_persistence_to_recognition_integration(tmp_path: Path) -> None:
    """Validate full flow from database writes to gallery load, matching, and restarts."""
    db_file = tmp_path / "dh360_integration.db"

    # Step 1: Initialize Database and PersonService on disk
    db_manager = DatabaseManager(db_path=db_file)
    repo = SQLitePersonRepository(db_manager)
    service = PersonService(repo)

    # Generate synthetic, distinguishable base embeddings (unit vectors)
    rng = np.random.default_rng(seed=42)
    base_emb_alice = normalize_l2(rng.standard_normal(128).astype(np.float32))
    base_emb_bob = normalize_l2(rng.standard_normal(128).astype(np.float32))

    # Step 2: Enroll Alice and Bob with multiple variations
    alice = service.create_person(
        display_name="Alice Walker",
        preferred_name="Alice",
        status=PersonStatus.ACTIVE,
        consent_status=ConsentStatus.GRANTED,
    )
    # 2 embeddings for Alice (base + slight variation)
    alice_var = normalize_l2(base_emb_alice + 0.05 * rng.standard_normal(128).astype(np.float32))
    service.add_person_embedding(alice.person_id, base_emb_alice)
    service.add_person_embedding(alice.person_id, alice_var)

    bob = service.create_person(
        display_name="Bob Martinez",
        preferred_name="Bob",
        status=PersonStatus.ACTIVE,
        consent_status=ConsentStatus.GRANTED,
    )
    service.add_person_embedding(bob.person_id, base_emb_bob)

    # Verify counts in DB
    assert service.get_person(alice.person_id).embedding_count == 2  # type: ignore[union-attr]
    assert service.get_person(bob.person_id).embedding_count == 1  # type: ignore[union-attr]

    # Step 3: Load Gallery into FaceRecognizer
    gallery = service.load_recognition_gallery()
    assert len(gallery) == 2
    recognizer = FaceRecognizer(gallery=gallery)

    # Query matching Alice
    query_alice = normalize_l2(base_emb_alice + 0.02 * rng.standard_normal(128).astype(np.float32))
    res_alice = recognizer.recognize(query_alice)
    assert res_alice.outcome == RecognitionOutcome.IDENTIFIED
    assert res_alice.person_id == alice.person_id
    assert res_alice.display_name == "Alice"
    assert res_alice.confidence_score > 0.90

    # Query matching Bob
    query_bob = normalize_l2(base_emb_bob + 0.02 * rng.standard_normal(128).astype(np.float32))
    res_bob = recognizer.recognize(query_bob)
    assert res_bob.outcome == RecognitionOutcome.IDENTIFIED
    assert res_bob.person_id == bob.person_id
    assert res_bob.display_name == "Bob"

    # Query unknown impostor
    impostor_emb = normalize_l2(rng.standard_normal(128).astype(np.float32))
    res_impostor = recognizer.recognize(impostor_emb)
    assert res_impostor.outcome == RecognitionOutcome.UNREGISTERED

    # Step 4: Deletion & Atomic Gallery Reload
    service.delete_person(alice.person_id)

    # Reload gallery into recognizer
    gallery_reloaded = service.load_recognition_gallery()
    assert len(gallery_reloaded) == 1
    recognizer = FaceRecognizer(gallery=gallery_reloaded)

    # Alice should now be UNREGISTERED
    res_alice_deleted = recognizer.recognize(query_alice)
    assert res_alice_deleted.outcome == RecognitionOutcome.UNREGISTERED

    # Close first connection
    db_manager.close()

    # Step 5: Re-open Database from Disk in a fresh instance and verify recovery
    new_db = DatabaseManager(db_path=db_file)
    new_repo = SQLitePersonRepository(new_db)
    new_service = PersonService(new_repo)

    recovered_gallery = new_service.load_recognition_gallery()
    assert len(recovered_gallery) == 1

    new_recognizer = FaceRecognizer(gallery=recovered_gallery)
    res_bob_recovered = new_recognizer.recognize(query_bob)
    assert res_bob_recovered.outcome == RecognitionOutcome.IDENTIFIED
    assert res_bob_recovered.person_id == bob.person_id

    new_db.close()

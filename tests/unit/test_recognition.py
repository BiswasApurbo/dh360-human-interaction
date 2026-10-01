"""Unit tests for in-memory face recognition, gallery matching, and similarity margins."""

import numpy as np
import pytest

from dh360_hi.vision.embedder import EmbeddingResult, normalize_l2
from dh360_hi.vision.recognition import (
    FaceRecognizer,
    GalleryEntry,
    InMemoryGallery,
    RecognitionConfig,
    RecognitionOutcome,
)


def _create_unit_vector(dim: int = 128, axis: int = 0) -> np.ndarray:
    """Create a unit basis vector with 1.0 along the specified axis."""
    vec = np.zeros(dim, dtype=np.float32)
    vec[axis] = 1.0
    return vec


def _create_synthetic_embedding(seed_val: float, dim: int = 128) -> np.ndarray:
    """Create a deterministic non-trivial unit embedding vector."""
    np.random.seed(int(abs(seed_val * 1000)))  # noqa: NPY002
    raw = np.random.randn(dim).astype(np.float32)  # noqa: NPY002
    return normalize_l2(raw)


def test_empty_gallery_returns_no_match() -> None:
    """Verify recognizer returns NO_MATCH when gallery contains no enrolled identities."""
    recognizer = FaceRecognizer(gallery=InMemoryGallery())
    query = _create_unit_vector(128, 0)
    result = recognizer.recognize(query)

    assert result.outcome == RecognitionOutcome.NO_MATCH
    assert result.top_match is None
    assert result.is_identified is False


def test_single_entry_successful_identification() -> None:
    """Verify query matching registered identity above threshold is IDENTIFIED."""
    emb_alice = _create_synthetic_embedding(1.0)
    gallery = InMemoryGallery(
        [
            GalleryEntry(
                person_id="p001",
                display_name="Alice Smith",
                embeddings=[emb_alice],
                model_name="SFace",
                model_version="2021dec",
            )
        ]
    )
    recognizer = FaceRecognizer(
        gallery=gallery, config=RecognitionConfig(similarity_threshold=0.60)
    )

    # Query with identical vector (cosine similarity = 1.0)
    res = recognizer.recognize(emb_alice)

    assert res.outcome == RecognitionOutcome.IDENTIFIED
    assert res.is_identified is True
    assert res.person_id == "p001"
    assert res.display_name == "Alice Smith"
    assert res.confidence_score == pytest.approx(1.0, abs=1e-4)


def test_unregistered_person_below_threshold() -> None:
    """Verify query with similarity below threshold returns UNREGISTERED."""
    emb_registered = _create_unit_vector(128, 0)
    emb_unknown = _create_unit_vector(128, 1)  # Orthogonal vector (similarity = 0.0)

    gallery = InMemoryGallery(
        [
            GalleryEntry(
                person_id="p001",
                display_name="Alice",
                embeddings=[emb_registered],
            )
        ]
    )
    recognizer = FaceRecognizer(
        gallery=gallery, config=RecognitionConfig(similarity_threshold=0.65)
    )

    res = recognizer.recognize(emb_unknown)

    assert res.outcome == RecognitionOutcome.UNREGISTERED
    assert res.is_identified is False
    assert res.top_match is not None
    assert res.top_match.similarity < 0.65


def test_ambiguous_top_matches_returns_uncertain() -> None:
    """Verify outcome is UNCERTAIN when top-1 and top-2 are too close (below margin)."""
    base = _create_unit_vector(128, 0)
    # Create two very similar embeddings
    emb_alice = normalize_l2(base + 0.01 * _create_unit_vector(128, 1))
    emb_bob = normalize_l2(base + 0.02 * _create_unit_vector(128, 1))

    gallery = InMemoryGallery(
        [
            GalleryEntry(person_id="p001", display_name="Alice", embeddings=[emb_alice]),
            GalleryEntry(person_id="p002", display_name="Bob", embeddings=[emb_bob]),
        ]
    )
    # Requires similarity >= 0.60 and margin >= 0.10
    recognizer = FaceRecognizer(
        gallery=gallery,
        config=RecognitionConfig(similarity_threshold=0.60, min_similarity_margin=0.10),
    )

    res = recognizer.recognize(base)

    assert res.outcome == RecognitionOutcome.UNCERTAIN
    assert res.is_identified is False
    assert res.top_match is not None
    assert res.second_match is not None
    assert res.similarity_margin < 0.10


def test_multiple_embeddings_per_person_aggregation() -> None:
    """Verify multi-embedding aggregation correctly evaluates similarity across varied poses."""
    e1 = _create_unit_vector(128, 0)
    e2 = _create_unit_vector(128, 1)

    gallery = InMemoryGallery(
        [
            GalleryEntry(
                person_id="p001",
                display_name="MultiPose User",
                embeddings=[e1, e2],
            )
        ]
    )

    # Max aggregation (default): query matching e2 will score 1.0
    rec_max = FaceRecognizer(
        gallery=gallery,
        config=RecognitionConfig(aggregation_method="max", similarity_threshold=0.60),
    )
    res_max = rec_max.recognize(e2)
    assert res_max.outcome == RecognitionOutcome.IDENTIFIED
    assert res_max.confidence_score == pytest.approx(1.0, abs=1e-4)

    # Mean aggregation: query matching e2 will score (0 + 1) / 2 = 0.50
    rec_mean = FaceRecognizer(
        gallery=gallery,
        config=RecognitionConfig(aggregation_method="mean", similarity_threshold=0.60),
    )
    res_mean = rec_mean.recognize(e2)
    assert res_mean.outcome == RecognitionOutcome.UNREGISTERED  # 0.50 < 0.60


def test_model_version_mismatch_rejected() -> None:
    """Verify recognizer rejects gallery entries when model name/version mismatches query."""
    emb = _create_unit_vector(128, 0)
    gallery = InMemoryGallery(
        [
            GalleryEntry(
                person_id="p001",
                display_name="Legacy Enrolled",
                embeddings=[emb],
                model_name="SFace",
                model_version="2020jun",  # Incompatible version
            )
        ]
    )

    recognizer = FaceRecognizer(
        gallery=gallery,
        model_name="SFace",
        model_version="2021dec",
        config=RecognitionConfig(require_model_match=True),
    )

    query_res = EmbeddingResult(
        embedding=emb,
        model_name="SFace",
        model_version="2021dec",
        dimension=128,
        is_normalized=True,
    )
    res = recognizer.recognize(query_res)

    assert res.outcome == RecognitionOutcome.NO_MATCH
    assert res.is_model_compatible is False
    assert "Model mismatch" in res.details


def test_invalid_query_embedding_handled_safely() -> None:
    """Verify empty or zero-norm embeddings return NO_MATCH safely."""
    gallery = InMemoryGallery(
        [
            GalleryEntry(
                person_id="p001",
                display_name="Alice",
                embeddings=[_create_unit_vector(128, 0)],
            )
        ]
    )
    recognizer = FaceRecognizer(gallery=gallery)

    # Zero vector
    zeros = np.zeros(128, dtype=np.float32)
    res_zero = recognizer.recognize(zeros)
    assert res_zero.outcome == RecognitionOutcome.NO_MATCH

    # Empty vector
    empty = np.array([], dtype=np.float32)
    res_empty = recognizer.recognize(empty)
    assert res_empty.outcome == RecognitionOutcome.NO_MATCH


def test_recognizer_replace_gallery_updates_recognition() -> None:
    """Verify replace_gallery replaces the active gallery and affects subsequent queries."""
    emb_alice = _create_unit_vector(128, 0)
    emb_bob = _create_unit_vector(128, 1)

    initial_gallery = InMemoryGallery(
        [GalleryEntry(person_id="p001", display_name="Alice", embeddings=[emb_alice])]
    )
    recognizer = FaceRecognizer(gallery=initial_gallery)

    # Query with Bob against Alice-only gallery -> UNREGISTERED
    res1 = recognizer.recognize(emb_bob)
    assert res1.outcome == RecognitionOutcome.UNREGISTERED

    # Atomically replace gallery with Bob
    updated_gallery = InMemoryGallery(
        [GalleryEntry(person_id="p002", display_name="Bob", embeddings=[emb_bob])]
    )
    recognizer.replace_gallery(updated_gallery)

    # Query with Bob now succeeds
    res2 = recognizer.recognize(emb_bob)
    assert res2.outcome == RecognitionOutcome.IDENTIFIED
    assert res2.person_id == "p002"
    assert res2.display_name == "Bob"

    # Query with Alice is now UNREGISTERED
    res3 = recognizer.recognize(emb_alice)
    assert res3.outcome == RecognitionOutcome.UNREGISTERED

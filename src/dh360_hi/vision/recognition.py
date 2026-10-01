"""Face recognition, in-memory gallery matching, and similarity evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from dh360_hi.vision.embedder import (
    EmbeddingResult,
    compute_cosine_similarity,
    normalize_l2,
)


class RecognitionOutcome(StrEnum):
    """Possible outcomes of a gallery face recognition query."""

    IDENTIFIED = "identified"
    UNREGISTERED = "unregistered"
    UNCERTAIN = "uncertain"
    NO_MATCH = "no_match"


@dataclass(frozen=True)
class GalleryEntry:
    """Represents a registered person identity in the in-memory recognition gallery."""

    person_id: str
    display_name: str
    embeddings: list[np.ndarray] = field(default_factory=list)
    model_name: str = "SFace"
    model_version: str = "2021dec"

    def __post_init__(self) -> None:
        if not self.person_id:
            msg = "person_id cannot be empty"
            raise ValueError(msg)
        if not self.display_name:
            msg = "display_name cannot be empty"
            raise ValueError(msg)


@dataclass(frozen=True)
class MatchCandidate:
    """Scored identity candidate from gallery matching."""

    person_id: str
    display_name: str
    similarity: float
    distance: float


@dataclass(frozen=True)
class RecognitionConfig:
    """Configuration thresholds and parameters for gallery recognition."""

    similarity_threshold: float = 0.65
    min_similarity_margin: float = 0.08
    aggregation_method: str = "max"  # "max" or "mean"
    require_model_match: bool = True


@dataclass(frozen=True)
class RecognitionResult:
    """Immutable outcome of face embedding recognition against gallery."""

    outcome: RecognitionOutcome
    top_match: MatchCandidate | None
    second_match: MatchCandidate | None
    similarity_margin: float
    confidence_score: float
    model_name: str
    model_version: str
    is_model_compatible: bool
    details: str

    @property
    def is_identified(self) -> bool:
        """Convenience property to check if person is confirmed identified."""
        return self.outcome == RecognitionOutcome.IDENTIFIED

    @property
    def person_id(self) -> str | None:
        """Matched person ID if identified, else None."""
        return self.top_match.person_id if self.top_match else None

    @property
    def display_name(self) -> str | None:
        """Matched display name if identified, else None."""
        return self.top_match.display_name if self.top_match else None


class InMemoryGallery:
    """In-memory gallery storing enrolled identities and their feature embeddings."""

    def __init__(self, entries: list[GalleryEntry] | None = None) -> None:
        self._entries: dict[str, GalleryEntry] = {}
        if entries:
            for entry in entries:
                self.add_entry(entry)

    def add_entry(self, entry: GalleryEntry) -> None:
        """Add or update an identity entry in the gallery."""
        # Ensure all embeddings in entry are L2-normalized
        normalized_embs = [normalize_l2(e) for e in entry.embeddings if e.size > 0]
        self._entries[entry.person_id] = GalleryEntry(
            person_id=entry.person_id,
            display_name=entry.display_name,
            embeddings=normalized_embs,
            model_name=entry.model_name,
            model_version=entry.model_version,
        )

    def add_embedding(self, person_id: str, embedding: np.ndarray) -> bool:
        """Add an embedding vector to an existing person entry."""
        if person_id not in self._entries:
            return False
        entry = self._entries[person_id]
        new_embs = list(entry.embeddings) + [normalize_l2(embedding)]
        self._entries[person_id] = GalleryEntry(
            person_id=entry.person_id,
            display_name=entry.display_name,
            embeddings=new_embs,
            model_name=entry.model_name,
            model_version=entry.model_version,
        )
        return True

    def get_entry(self, person_id: str) -> GalleryEntry | None:
        """Retrieve a gallery entry by person ID."""
        return self._entries.get(person_id)

    def list_entries(self) -> list[GalleryEntry]:
        """List all gallery entries."""
        return list(self._entries.values())

    def clear(self) -> None:
        """Remove all entries from the gallery."""
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)


class BaseFaceRecognizer(ABC):
    """Abstract interface for face recognition matcher."""

    @abstractmethod
    def recognize(
        self,
        query: np.ndarray | EmbeddingResult,
    ) -> RecognitionResult:
        """Match a query face embedding against the gallery."""

    @property
    @abstractmethod
    def gallery(self) -> InMemoryGallery:
        """Underlying recognition gallery."""

    @property
    @abstractmethod
    def config(self) -> RecognitionConfig:
        """Recognition configuration thresholds."""


class FaceRecognizer(BaseFaceRecognizer):
    """Face recognizer evaluating query embeddings against an in-memory gallery."""

    def __init__(
        self,
        gallery: InMemoryGallery | None = None,
        config: RecognitionConfig | None = None,
        model_name: str = "SFace",
        model_version: str = "2021dec",
    ) -> None:
        self._gallery = gallery or InMemoryGallery()
        self._config = config or RecognitionConfig()
        self._model_name = model_name
        self._model_version = model_version

    @property
    def gallery(self) -> InMemoryGallery:
        return self._gallery

    @property
    def config(self) -> RecognitionConfig:
        return self._config

    @config.setter
    def config(self, config: RecognitionConfig) -> None:
        self._config = config

    def _compute_person_similarity(self, query_norm: np.ndarray, entry: GalleryEntry) -> float:
        """Compute aggregated similarity between query embedding and person gallery embeddings."""
        if not entry.embeddings:
            return -1.0

        sims = [compute_cosine_similarity(query_norm, e) for e in entry.embeddings]
        if self._config.aggregation_method == "mean":
            return float(np.mean(sims))
        # Default: "max" similarity
        return float(np.max(sims))

    def recognize(
        self,
        query: np.ndarray | EmbeddingResult,
    ) -> RecognitionResult:
        """Recognize a query face embedding against the gallery."""
        # 1. Unpack Query
        if isinstance(query, EmbeddingResult):
            if not query.is_normalized and np.allclose(query.embedding, 0.0):
                return RecognitionResult(
                    outcome=RecognitionOutcome.NO_MATCH,
                    top_match=None,
                    second_match=None,
                    similarity_margin=0.0,
                    confidence_score=0.0,
                    model_name=query.model_name,
                    model_version=query.model_version,
                    is_model_compatible=True,
                    details="Invalid or unnormalized zero embedding input",
                )
            query_emb = query.embedding
            q_model_name = query.model_name
            q_model_version = query.model_version
        else:
            query_emb = query
            q_model_name = self._model_name
            q_model_version = self._model_version

        norm_val = (
            float(np.linalg.norm(query_emb))
            if query_emb is not None and query_emb.size > 0
            else 0.0
        )
        if (
            query_emb is None
            or query_emb.size == 0
            or norm_val == 0.0
            or not np.all(np.isfinite(query_emb))
        ):
            return RecognitionResult(
                outcome=RecognitionOutcome.NO_MATCH,
                top_match=None,
                second_match=None,
                similarity_margin=0.0,
                confidence_score=0.0,
                model_name=q_model_name,
                model_version=q_model_version,
                is_model_compatible=True,
                details="Empty, zero-norm, or non-finite query embedding",
            )

        query_norm = normalize_l2(query_emb)

        # 2. Check Empty Gallery
        if len(self._gallery) == 0:
            return RecognitionResult(
                outcome=RecognitionOutcome.NO_MATCH,
                top_match=None,
                second_match=None,
                similarity_margin=0.0,
                confidence_score=0.0,
                model_name=q_model_name,
                model_version=q_model_version,
                is_model_compatible=True,
                details="Gallery is empty",
            )

        # 3. Model Compatibility & Candidate Scoring
        candidates: list[MatchCandidate] = []

        for entry in self._gallery.list_entries():
            if self._config.require_model_match and (
                entry.model_name != q_model_name or entry.model_version != q_model_version
            ):
                continue

            sim = self._compute_person_similarity(query_norm, entry)
            if sim >= -1.0:
                candidates.append(
                    MatchCandidate(
                        person_id=entry.person_id,
                        display_name=entry.display_name,
                        similarity=sim,
                        distance=float(1.0 - sim),
                    )
                )

        if not candidates and self._config.require_model_match:
            return RecognitionResult(
                outcome=RecognitionOutcome.NO_MATCH,
                top_match=None,
                second_match=None,
                similarity_margin=0.0,
                confidence_score=0.0,
                model_name=q_model_name,
                model_version=q_model_version,
                is_model_compatible=False,
                details=(
                    f"Model mismatch: no gallery entries compatible with "
                    f"{q_model_name} v{q_model_version}"
                ),
            )

        # 4. Sort candidates by similarity descending
        candidates.sort(key=lambda c: c.similarity, reverse=True)
        top_match = candidates[0] if len(candidates) > 0 else None
        second_match = candidates[1] if len(candidates) > 1 else None

        if top_match is None:
            return RecognitionResult(
                outcome=RecognitionOutcome.NO_MATCH,
                top_match=None,
                second_match=None,
                similarity_margin=0.0,
                confidence_score=0.0,
                model_name=q_model_name,
                model_version=q_model_version,
                is_model_compatible=True,
                details="No candidates with valid embeddings in gallery",
            )

        similarity_margin = (
            float(top_match.similarity - second_match.similarity)
            if second_match
            else float(top_match.similarity)
        )

        # 5. Evaluate Decision Thresholds
        thresh = self._config.similarity_threshold
        min_margin = self._config.min_similarity_margin

        if top_match.similarity < thresh:
            # Below identification threshold -> Unregistered / Unknown
            return RecognitionResult(
                outcome=RecognitionOutcome.UNREGISTERED,
                top_match=top_match,
                second_match=second_match,
                similarity_margin=similarity_margin,
                confidence_score=top_match.similarity,
                model_name=q_model_name,
                model_version=q_model_version,
                is_model_compatible=True,
                details=f"Top similarity {top_match.similarity:.4f} below threshold {thresh:.2f}",
            )

        if second_match is not None and similarity_margin < min_margin:
            # Above threshold but ambiguous margin -> Uncertain
            return RecognitionResult(
                outcome=RecognitionOutcome.UNCERTAIN,
                top_match=top_match,
                second_match=second_match,
                similarity_margin=similarity_margin,
                confidence_score=top_match.similarity,
                model_name=q_model_name,
                model_version=q_model_version,
                is_model_compatible=True,
                details=(
                    f"Ambiguous match: margin {similarity_margin:.4f} "
                    f"between '{top_match.person_id}' and '{second_match.person_id}' "
                    f"below required {min_margin:.2f}"
                ),
            )

        details_str = (
            f"Identified '{top_match.person_id}' ({top_match.display_name}) "
            f"with similarity {top_match.similarity:.4f}"
        )
        return RecognitionResult(
            outcome=RecognitionOutcome.IDENTIFIED,
            top_match=top_match,
            second_match=second_match,
            similarity_margin=similarity_margin,
            confidence_score=top_match.similarity,
            model_name=q_model_name,
            model_version=q_model_version,
            is_model_compatible=True,
            details=details_str,
        )

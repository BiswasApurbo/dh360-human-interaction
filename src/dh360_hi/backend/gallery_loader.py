"""Gallery loader extracting persisted embeddings from SQLite into InMemoryGallery."""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np

from dh360_hi.backend.models import ConsentStatus, PersonStatus
from dh360_hi.backend.repository import BasePersonRepository
from dh360_hi.vision.embedder import normalize_l2
from dh360_hi.vision.recognition import GalleryEntry, InMemoryGallery

logger = logging.getLogger(__name__)


def load_gallery_from_repository(
    repository: BasePersonRepository,
    model_name: str = "SFace",
    model_version: str = "2021dec",
    require_active: bool = True,
    require_consent: bool = True,
) -> InMemoryGallery:
    """Extract, validate, and load registered person embeddings into an InMemoryGallery.

    Args:
        repository: The persistence repository.
        model_name: Expected embedding model name (default: "SFace").
        model_version: Expected embedding model version (default: "2021dec").
        require_active: Whether to restrict gallery to persons with status ACTIVE.
        require_consent: Whether to restrict gallery to persons with consent GRANTED.

    Returns:
        A freshly constructed, isolated InMemoryGallery instance.
    """
    # 1. Fetch eligible persons
    target_status = PersonStatus.ACTIVE if require_active else None
    persons = repository.list_persons(status=target_status)

    if require_consent:
        persons = [p for p in persons if p.consent_status == ConsentStatus.GRANTED]

    if not persons:
        return InMemoryGallery()

    person_map = {p.person_id: p for p in persons}

    # 2. Fetch all stored embeddings compatible with model name and version
    embeddings = repository.get_all_embeddings(
        model_name=model_name,
        model_version=model_version,
    )

    # 3. Group embeddings by person_id
    person_embeddings: dict[str, list[np.ndarray]] = defaultdict(list)
    for emb_rec in embeddings:
        if emb_rec.person_id not in person_map:
            continue
        try:
            vec = emb_rec.vector
            if vec.size > 0 and np.all(np.isfinite(vec)):
                norm_vec = normalize_l2(vec)
                person_embeddings[emb_rec.person_id].append(norm_vec)
        except Exception as exc:
            logger.warning(
                "Skipping invalid embedding '%s' for person '%s': %s",
                emb_rec.embedding_id,
                emb_rec.person_id,
                exc,
            )

    # 4. Build GalleryEntry list for eligible persons with at least 1 valid embedding
    entries: list[GalleryEntry] = []
    for person_id, person in person_map.items():
        embs = person_embeddings.get(person_id, [])
        if embs:
            entries.append(
                GalleryEntry(
                    person_id=person.person_id,
                    display_name=person.preferred_name or person.display_name,
                    embeddings=embs,
                    model_name=model_name,
                    model_version=model_version,
                )
            )

    logger.info(
        "Loaded %d gallery identities (%d embeddings) for %s v%s",
        len(entries),
        sum(len(e.embeddings) for e in entries),
        model_name,
        model_version,
    )

    return InMemoryGallery(entries)

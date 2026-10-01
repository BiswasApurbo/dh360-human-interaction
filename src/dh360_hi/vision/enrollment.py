"""Camera-based face enrollment controller with quality gating and diversity checks."""

from __future__ import annotations

import logging
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from dh360_hi.backend.exceptions import PersonNotFoundError, ValidationError
from dh360_hi.backend.models import (
    ConsentStatus,
    EmbeddingRecord,
    PersonStatus,
    utc_now_iso,
    vector_to_blob,
)
from dh360_hi.backend.person_service import BasePersonService
from dh360_hi.vision.aligner import BaseFaceAligner
from dh360_hi.vision.detector import BaseFaceDetector
from dh360_hi.vision.embedder import (
    BaseFaceEmbedder,
    compute_cosine_similarity,
    normalize_l2,
)
from dh360_hi.vision.frame import Frame
from dh360_hi.vision.quality_gate import BaseFaceQualityGate
from dh360_hi.vision.recognition import FaceRecognizer

logger = logging.getLogger(__name__)


class EnrollmentState(StrEnum):
    """Lifecycle states of an enrollment session."""

    IDLE = "idle"
    COLLECTING = "collecting"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SampleRejectionReason(StrEnum):
    """Reasons why a candidate camera frame or face was rejected during enrollment."""

    NO_FACE = "no_face"
    MULTIPLE_FACES = "multiple_faces"
    QUALITY_REJECTED = "quality_rejected"
    ALIGNMENT_FAILED = "alignment_failed"
    EMBEDDING_FAILED = "embedding_failed"
    DUPLICATE_SAMPLE = "duplicate_sample"
    CONSENT_MISSING = "consent_missing"
    INACTIVE_PERSON = "inactive_person"
    TIMEOUT = "timeout"
    MAX_ATTEMPTS_EXCEEDED = "max_attempts_exceeded"


@dataclass(frozen=True)
class EnrollmentSample:
    """A single accepted face embedding sample collected during an enrollment session."""

    sample_id: str
    embedding: np.ndarray
    quality_score: float
    roll_angle_deg: float | None
    yaw_asymmetry: float | None
    frame_id: int
    timestamp: float


@dataclass(frozen=True)
class EnrollmentConfig:
    """Configuration governing face enrollment sample requirements and diversity thresholds."""

    target_samples: int = 7
    minimum_samples: int = 5
    max_attempts: int = 60
    duplicate_similarity_threshold: float = 0.95
    min_sample_interval_seconds: float = 0.2
    timeout_seconds: float = 30.0
    model_name: str = "SFace"
    model_version: str = "2021dec"

    def __post_init__(self) -> None:
        if self.minimum_samples < 1:
            msg = "minimum_samples must be at least 1"
            raise ValueError(msg)
        if self.target_samples < self.minimum_samples:
            msg = "target_samples cannot be less than minimum_samples"
            raise ValueError(msg)
        if not (0.0 < self.duplicate_similarity_threshold <= 1.0):
            msg = "duplicate_similarity_threshold must be between 0.0 and 1.0"
            raise ValueError(msg)


@dataclass
class _ActiveSessionData:
    """Internal mutable tracking state for an in-progress enrollment session."""

    session_id: str
    enrollment_batch_id: str
    person_id: str
    display_name: str
    state: EnrollmentState = EnrollmentState.COLLECTING
    samples: list[EnrollmentSample] = field(default_factory=list)
    attempts: int = 0
    start_time: float = field(default_factory=time.monotonic)
    last_sample_time: float = 0.0
    last_rejection_reason: SampleRejectionReason | None = None
    last_rejection_details: str | None = None
    rejection_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))


@dataclass(frozen=True)
class EnrollmentProgress:
    """Snapshot of current progress during an enrollment session."""

    session_id: str
    enrollment_batch_id: str
    person_id: str
    display_name: str
    state: EnrollmentState
    samples_collected: int
    target_samples: int
    minimum_samples: int
    total_attempts: int
    last_rejection_reason: SampleRejectionReason | None
    last_rejection_details: str | None
    rejection_counts: dict[str, int]
    is_viable: bool
    is_complete: bool


@dataclass(frozen=True)
class EnrollmentResult:
    """Final outcome of an enrollment session commit or completion."""

    session_id: str
    enrollment_batch_id: str
    person_id: str
    display_name: str
    is_success: bool
    state: EnrollmentState
    samples_enrolled: int
    persisted_embedding_ids: list[str]
    total_attempts: int
    error_message: str | None
    rejection_counts: dict[str, int]


class BaseFaceEnrollment(ABC):
    """Abstract interface for operator-controlled face enrollment."""

    @abstractmethod
    def start_session(self, person_id: str) -> EnrollmentProgress:
        """Start a new enrollment session for a consented person."""

    @abstractmethod
    def process_frame(self, frame: Frame) -> EnrollmentProgress:
        """Process a camera frame against the active enrollment session."""

    @abstractmethod
    def cancel_session(self) -> EnrollmentProgress:
        """Cancel the currently active enrollment session without persisting."""

    @abstractmethod
    def commit_session(self) -> EnrollmentResult:
        """Persist collected embeddings transactionally and reload gallery."""

    @property
    @abstractmethod
    def is_active(self) -> bool:
        """Check if an enrollment session is currently in progress."""

    @property
    @abstractmethod
    def current_progress(self) -> EnrollmentProgress | None:
        """Retrieve current progress of the active session if any."""


class FaceEnrollmentController(BaseFaceEnrollment):
    """Controls the camera face enrollment workflow with quality gating and diversity checks."""

    def __init__(
        self,
        detector: BaseFaceDetector,
        quality_gate: BaseFaceQualityGate,
        aligner: BaseFaceAligner,
        embedder: BaseFaceEmbedder,
        person_service: BasePersonService,
        config: EnrollmentConfig | None = None,
        recognizer: FaceRecognizer | None = None,
    ) -> None:
        self._detector = detector
        self._quality_gate = quality_gate
        self._aligner = aligner
        self._embedder = embedder
        self._person_service = person_service
        self._config = config or EnrollmentConfig()
        self._recognizer = recognizer
        self._session: _ActiveSessionData | None = None

    @property
    def config(self) -> EnrollmentConfig:
        """Enrollment configuration."""
        return self._config

    @property
    def is_active(self) -> bool:
        """Check if an enrollment session is currently in progress."""
        return self._session is not None and self._session.state in (
            EnrollmentState.COLLECTING,
            EnrollmentState.COMPLETED,
        )

    @property
    def current_progress(self) -> EnrollmentProgress | None:
        """Retrieve current progress of the active session."""
        return self._build_progress(self._session) if self._session is not None else None

    def _build_progress(self, s: _ActiveSessionData) -> EnrollmentProgress:
        """Construct an immutable EnrollmentProgress from session data."""
        return EnrollmentProgress(
            session_id=s.session_id,
            enrollment_batch_id=s.enrollment_batch_id,
            person_id=s.person_id,
            display_name=s.display_name,
            state=s.state,
            samples_collected=len(s.samples),
            target_samples=self._config.target_samples,
            minimum_samples=self._config.minimum_samples,
            total_attempts=s.attempts,
            last_rejection_reason=s.last_rejection_reason,
            last_rejection_details=s.last_rejection_details,
            rejection_counts=dict(s.rejection_counts),
            is_viable=len(s.samples) >= self._config.minimum_samples,
            is_complete=s.state == EnrollmentState.COMPLETED,
        )

    def start_session(self, person_id: str) -> EnrollmentProgress:
        """Start a new enrollment session for an existing, consented person."""
        if self._session is not None and self._session.state == EnrollmentState.COLLECTING:
            msg = (
                f"Cannot start enrollment: another session ({self._session.session_id}) "
                f"is already active for person '{self._session.person_id}'"
            )
            raise ValidationError(msg)

        # Verify person exists in PersonService
        person = self._person_service.get_person(person_id)
        if person is None:
            msg = f"Person '{person_id}' does not exist in person repository"
            raise PersonNotFoundError(msg)

        if person.status != PersonStatus.ACTIVE:
            msg = f"Cannot enroll inactive person '{person_id}' (status: {person.status})"
            raise ValidationError(msg)

        if person.consent_status != ConsentStatus.GRANTED:
            msg = (
                f"Cannot enroll person '{person_id}': consent status is "
                f"'{person.consent_status}', but GRANTED is strictly required"
            )
            raise ValidationError(msg)

        session_id = str(uuid.uuid4())
        batch_id = str(uuid.uuid4())

        self._session = _ActiveSessionData(
            session_id=session_id,
            enrollment_batch_id=batch_id,
            person_id=person.person_id,
            display_name=person.preferred_name or person.display_name,
            state=EnrollmentState.COLLECTING,
        )

        logger.info(
            "Started face enrollment session %s for person '%s' (%s), batch %s",
            session_id,
            person.person_id,
            person.display_name,
            batch_id,
        )
        return self._build_progress(self._session)

    def process_frame(self, frame: Frame) -> EnrollmentProgress:
        """Process a camera frame against the active enrollment session."""
        if self._session is None:
            msg = "No active enrollment session to process frame"
            raise ValidationError(msg)

        s = self._session
        if s.state != EnrollmentState.COLLECTING:
            return self._build_progress(s)

        # 1. Check Timeout
        elapsed = time.monotonic() - s.start_time
        if elapsed > self._config.timeout_seconds:
            if len(s.samples) >= self._config.minimum_samples:
                s.state = EnrollmentState.COMPLETED
            else:
                s.state = EnrollmentState.FAILED
                s.last_rejection_reason = SampleRejectionReason.TIMEOUT
                s.last_rejection_details = (
                    f"Session timed out after {elapsed:.1f}s with only "
                    f"{len(s.samples)}/{self._config.minimum_samples} minimum samples"
                )
                s.rejection_counts[SampleRejectionReason.TIMEOUT.value] += 1
            return self._build_progress(s)

        # 2. Check Max Attempts
        if s.attempts >= self._config.max_attempts:
            if len(s.samples) >= self._config.minimum_samples:
                s.state = EnrollmentState.COMPLETED
            else:
                s.state = EnrollmentState.FAILED
                s.last_rejection_reason = SampleRejectionReason.MAX_ATTEMPTS_EXCEEDED
                s.last_rejection_details = (
                    f"Reached max attempts ({s.attempts}) with only "
                    f"{len(s.samples)}/{self._config.minimum_samples} minimum samples"
                )
                s.rejection_counts[SampleRejectionReason.MAX_ATTEMPTS_EXCEEDED.value] += 1
            return self._build_progress(s)

        s.attempts += 1

        # 3. Detection
        det_result = self._detector.detect(frame)
        if len(det_result.detections) == 0:
            s.last_rejection_reason = SampleRejectionReason.NO_FACE
            s.last_rejection_details = "No face detected in camera frame"
            s.rejection_counts[SampleRejectionReason.NO_FACE.value] += 1
            return self._build_progress(s)

        # 4. Single-Face Requirement (Dominant-Face Lock)
        if len(det_result.detections) > 1:
            s.last_rejection_reason = SampleRejectionReason.MULTIPLE_FACES
            s.last_rejection_details = (
                f"{len(det_result.detections)} faces detected in frame; "
                f"exactly one face required during enrollment"
            )
            s.rejection_counts[SampleRejectionReason.MULTIPLE_FACES.value] += 1
            return self._build_progress(s)

        target_detection = det_result.detections[0]

        # 5. Quality Gate Evaluation
        q_result = self._quality_gate.evaluate(frame, target_detection)
        if not q_result.is_acceptable:
            reasons_str = ", ".join(r.value for r in q_result.rejection_reasons)
            s.last_rejection_reason = SampleRejectionReason.QUALITY_REJECTED
            s.last_rejection_details = f"Quality gate rejected: {reasons_str}"
            s.rejection_counts[SampleRejectionReason.QUALITY_REJECTED.value] += 1
            return self._build_progress(s)

        # 6. Face Alignment
        align_result = self._aligner.align(frame, target_detection)
        if not align_result.is_success:
            s.last_rejection_reason = SampleRejectionReason.ALIGNMENT_FAILED
            s.last_rejection_details = "Similarity landmark alignment failed"
            s.rejection_counts[SampleRejectionReason.ALIGNMENT_FAILED.value] += 1
            return self._build_progress(s)

        # 7. Embedding Extraction
        emb_result = self._embedder.embed(align_result)
        if (
            emb_result is None
            or emb_result.embedding is None
            or emb_result.embedding.size == 0
            or not np.all(np.isfinite(emb_result.embedding))
        ):
            s.last_rejection_reason = SampleRejectionReason.EMBEDDING_FAILED
            s.last_rejection_details = "Embedding extraction returned empty or non-finite vector"
            s.rejection_counts[SampleRejectionReason.EMBEDDING_FAILED.value] += 1
            return self._build_progress(s)

        candidate_vector = normalize_l2(emb_result.embedding)

        # 8. Sample Diversity & Duplicate Protection
        curr_time = time.monotonic()
        if (
            s.last_sample_time > 0.0
            and (curr_time - s.last_sample_time) < self._config.min_sample_interval_seconds
        ):
            s.last_rejection_reason = SampleRejectionReason.DUPLICATE_SAMPLE
            s.last_rejection_details = "Sample captured too quickly after previous sample"
            s.rejection_counts[SampleRejectionReason.DUPLICATE_SAMPLE.value] += 1
            return self._build_progress(s)

        for existing_sample in s.samples:
            sim = compute_cosine_similarity(candidate_vector, existing_sample.embedding)
            if sim >= self._config.duplicate_similarity_threshold:
                s.last_rejection_reason = SampleRejectionReason.DUPLICATE_SAMPLE
                s.last_rejection_details = (
                    f"Sample similarity {sim:.4f} exceeds duplicate threshold "
                    f"{self._config.duplicate_similarity_threshold:.2f}"
                )
                s.rejection_counts[SampleRejectionReason.DUPLICATE_SAMPLE.value] += 1
                return self._build_progress(s)

        # 9. Accept Sample
        sample = EnrollmentSample(
            sample_id=str(uuid.uuid4()),
            embedding=candidate_vector,
            quality_score=float(q_result.metrics.confidence),
            roll_angle_deg=q_result.metrics.roll_angle_deg,
            yaw_asymmetry=q_result.metrics.yaw_asymmetry,
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
        )
        s.samples.append(sample)
        s.last_sample_time = curr_time
        s.last_rejection_reason = None
        s.last_rejection_details = None

        logger.debug(
            "Accepted enrollment sample %d/%d for person '%s' (quality: %.2f)",
            len(s.samples),
            self._config.target_samples,
            s.person_id,
            sample.quality_score,
        )

        if len(s.samples) >= self._config.target_samples:
            s.state = EnrollmentState.COMPLETED
            logger.info(
                "Enrollment session %s reached target samples (%d/%d)",
                s.session_id,
                len(s.samples),
                self._config.target_samples,
            )

        return self._build_progress(s)

    def cancel_session(self) -> EnrollmentProgress:
        """Cancel the active enrollment session without persisting any samples."""
        if self._session is None:
            msg = "No active enrollment session to cancel"
            raise ValidationError(msg)

        s = self._session
        s.state = EnrollmentState.CANCELLED
        progress = self._build_progress(s)
        self._session = None
        logger.info("Cancelled enrollment session %s", progress.session_id)
        return progress

    def commit_session(self) -> EnrollmentResult:
        """Persist collected samples transactionally to PersonService and reload gallery."""
        if self._session is None:
            msg = "No active enrollment session to commit"
            raise ValidationError(msg)

        s = self._session

        # 1. Validate Minimum Samples
        if len(s.samples) < self._config.minimum_samples:
            s.state = EnrollmentState.FAILED
            err_msg = (
                f"Cannot commit enrollment: collected {len(s.samples)} samples, "
                f"which is below required minimum of {self._config.minimum_samples}"
            )
            result = EnrollmentResult(
                session_id=s.session_id,
                enrollment_batch_id=s.enrollment_batch_id,
                person_id=s.person_id,
                display_name=s.display_name,
                is_success=False,
                state=EnrollmentState.FAILED,
                samples_enrolled=0,
                persisted_embedding_ids=[],
                total_attempts=s.attempts,
                error_message=err_msg,
                rejection_counts=dict(s.rejection_counts),
            )
            self._session = None
            return result

        # 2. Build Transactional Embedding Records
        now_str = utc_now_iso()
        records: list[EmbeddingRecord] = []
        for sample in s.samples:
            blob = vector_to_blob(sample.embedding, expected_dim=128)
            records.append(
                EmbeddingRecord(
                    embedding_id=sample.sample_id,
                    person_id=s.person_id,
                    embedding=blob,
                    dimension=128,
                    model_name=self._config.model_name,
                    model_version=self._config.model_version,
                    quality_score=sample.quality_score,
                    enrollment_batch_id=s.enrollment_batch_id,
                    created_at=now_str,
                )
            )

        # 3. Transactional Persistence via Repository
        try:
            persisted = self._person_service.repository.add_embeddings(records)
            persisted_ids = [r.embedding_id for r in persisted]
        except Exception as exc:
            s.state = EnrollmentState.FAILED
            err_msg = f"Database persistence failed during enrollment commit: {exc}"
            logger.error(err_msg)
            result = EnrollmentResult(
                session_id=s.session_id,
                enrollment_batch_id=s.enrollment_batch_id,
                person_id=s.person_id,
                display_name=s.display_name,
                is_success=False,
                state=EnrollmentState.FAILED,
                samples_enrolled=0,
                persisted_embedding_ids=[],
                total_attempts=s.attempts,
                error_message=err_msg,
                rejection_counts=dict(s.rejection_counts),
            )
            self._session = None
            return result

        # 4. Atomic Gallery Reload
        try:
            new_gallery = self._person_service.load_recognition_gallery(
                model_name=self._config.model_name,
                model_version=self._config.model_version,
            )
            if self._recognizer is not None:
                self._recognizer._gallery = new_gallery
        except Exception as exc:
            logger.warning("Gallery reload encountered non-fatal error: %s", exc)

        s.state = EnrollmentState.COMPLETED
        result = EnrollmentResult(
            session_id=s.session_id,
            enrollment_batch_id=s.enrollment_batch_id,
            person_id=s.person_id,
            display_name=s.display_name,
            is_success=True,
            state=EnrollmentState.COMPLETED,
            samples_enrolled=len(persisted_ids),
            persisted_embedding_ids=persisted_ids,
            total_attempts=s.attempts,
            error_message=None,
            rejection_counts=dict(s.rejection_counts),
        )

        logger.info(
            "Successfully enrolled person '%s' (%s) with %d samples (batch %s)",
            s.person_id,
            s.display_name,
            len(persisted_ids),
            s.enrollment_batch_id,
        )
        self._session = None
        return result

"""Unit tests for FaceEnrollmentController and transactional commits."""

from __future__ import annotations

import numpy as np
import pytest

from dh360_hi.backend.database import DatabaseManager
from dh360_hi.backend.exceptions import PersonNotFoundError, ValidationError
from dh360_hi.backend.models import ConsentStatus, PersonStatus
from dh360_hi.backend.person_service import PersonService
from dh360_hi.backend.repository import SQLitePersonRepository
from dh360_hi.vision.aligner import AlignmentResult, BaseFaceAligner
from dh360_hi.vision.detector import BaseFaceDetector, DetectionResult, FaceDetection
from dh360_hi.vision.embedder import BaseFaceEmbedder, EmbeddingResult, normalize_l2
from dh360_hi.vision.enrollment import (
    EnrollmentConfig,
    EnrollmentMode,
    EnrollmentState,
    FaceEnrollmentController,
    SampleRejectionReason,
)
from dh360_hi.vision.frame import Frame
from dh360_hi.vision.quality_gate import (
    BaseFaceQualityGate,
    FaceQualityMetrics,
    FaceQualityResult,
    FaceQualityThresholds,
    FaceRejectionReason,
)
from dh360_hi.vision.recognition import FaceRecognizer


class MockDetector(BaseFaceDetector):
    """Mock detector returning configurable detections."""

    def __init__(self, detections: list[FaceDetection] | None = None) -> None:
        self._detections = detections or []
        self._input_size = (640, 480)
        self._confidence_threshold = 0.6

    def set_detections(self, detections: list[FaceDetection]) -> None:
        self._detections = detections

    def detect(self, frame: Frame | np.ndarray, frame_id: int = 0) -> DetectionResult:
        fid = frame.frame_id if isinstance(frame, Frame) else frame_id
        return DetectionResult(
            detections=self._detections,
            frame_id=fid,
            latency_ms=1.0,
            input_size=self._input_size,
        )

    @property
    def input_size(self) -> tuple[int, int]:
        return self._input_size

    @input_size.setter
    def input_size(self, size: tuple[int, int]) -> None:
        self._input_size = size

    @property
    def confidence_threshold(self) -> float:
        return self._confidence_threshold

    @confidence_threshold.setter
    def confidence_threshold(self, threshold: float) -> None:
        self._confidence_threshold = threshold


class MockQualityGate(BaseFaceQualityGate):
    """Mock quality gate returning configurable acceptance."""

    def __init__(self, is_acceptable: bool = True) -> None:
        self._is_acceptable = is_acceptable

    def set_acceptable(self, is_acceptable: bool) -> None:
        self._is_acceptable = is_acceptable

    def evaluate(self, frame: Frame, face: FaceDetection) -> FaceQualityResult:
        _ = (frame, face)
        metrics = FaceQualityMetrics(
            confidence=0.95,
            face_width=100.0,
            face_height=100.0,
            sharpness=150.0,
            roll_angle_deg=2.0,
            yaw_asymmetry=1.0,
        )
        reasons = [] if self._is_acceptable else [FaceRejectionReason.LOW_CONFIDENCE]
        return FaceQualityResult(
            is_acceptable=self._is_acceptable,
            metrics=metrics,
            rejection_reasons=reasons,
            thresholds=FaceQualityThresholds(),
        )

    @property
    def thresholds(self) -> FaceQualityThresholds:
        return FaceQualityThresholds()


class MockAligner(BaseFaceAligner):
    """Mock aligner returning successful alignment."""

    def align(self, frame: Frame, face: FaceDetection) -> AlignmentResult:
        _ = (frame, face)
        return AlignmentResult(
            aligned_face=np.zeros((112, 112, 3), dtype=np.uint8),
            is_success=True,
            method="similarity_5pts",
            output_size=(112, 112),
        )

    @property
    def output_size(self) -> tuple[int, int]:
        return (112, 112)


class MockEmbedder(BaseFaceEmbedder):
    """Mock embedder returning controllable synthetic vectors."""

    def __init__(self, vector: np.ndarray | None = None) -> None:
        self._vector = (
            vector if vector is not None else normalize_l2(np.ones(128, dtype=np.float32))
        )

    def set_vector(self, vector: np.ndarray) -> None:
        self._vector = normalize_l2(vector)

    def embed(self, aligned: AlignmentResult | np.ndarray) -> EmbeddingResult:
        _ = aligned
        return EmbeddingResult(
            embedding=self._vector,
            dimension=128,
            model_name="SFace",
            model_version="2021dec",
            is_normalized=True,
            latency_ms=1.0,
        )

    @property
    def model_name(self) -> str:
        return "SFace"

    @property
    def model_version(self) -> str:
        return "2021dec"

    @property
    def dimension(self) -> int:
        return 128

    @property
    def input_size(self) -> tuple[int, int]:
        return (112, 112)


@pytest.fixture
def mock_pipeline() -> tuple[
    MockDetector,
    MockQualityGate,
    MockAligner,
    MockEmbedder,
    PersonService,
    FaceEnrollmentController,
]:
    """Provide a mocked enrollment pipeline."""
    db = DatabaseManager(":memory:")
    repo = SQLitePersonRepository(db)
    service = PersonService(repo)

    detector = MockDetector([FaceDetection(bbox=(10.0, 10.0, 50.0, 50.0), confidence=0.95)])
    quality_gate = MockQualityGate(is_acceptable=True)
    aligner = MockAligner()
    embedder = MockEmbedder()

    config = EnrollmentConfig(
        target_samples=5,
        minimum_samples=3,
        duplicate_similarity_threshold=0.95,
        min_sample_interval_seconds=0.0,
        timeout_seconds=10.0,
    )

    controller = FaceEnrollmentController(
        detector=detector,
        quality_gate=quality_gate,
        aligner=aligner,
        embedder=embedder,
        person_service=service,
        config=config,
    )
    return detector, quality_gate, aligner, embedder, service, controller


def test_enrollment_session_lifecycle(mock_pipeline: tuple) -> None:
    """Test starting, tracking progress, and cancelling an enrollment session."""
    _, _, _, _, service, controller = mock_pipeline
    person = service.create_person("Sarah Connor", consent_status=ConsentStatus.GRANTED)

    # 1. Start Session
    prog = controller.start_session(person.person_id)
    assert controller.is_active is True
    assert prog.state == EnrollmentState.COLLECTING
    assert prog.person_id == person.person_id
    assert prog.samples_collected == 0
    assert prog.target_samples == 5

    # 2. Cancel Session
    cancelled_prog = controller.cancel_session()
    assert controller.is_active is False
    assert cancelled_prog.state == EnrollmentState.CANCELLED
    assert controller.current_progress is None


def test_enrollment_consent_and_status_validation(mock_pipeline: tuple) -> None:
    """Test that enrollment strictly requires an active, consented person."""
    _, _, _, _, service, controller = mock_pipeline

    # Non-existent person
    with pytest.raises(PersonNotFoundError):
        controller.start_session("00000000-0000-0000-0000-000000000000")

    # Inactive person
    p_inactive = service.create_person(
        "Inactive Person",
        status=PersonStatus.INACTIVE,
        consent_status=ConsentStatus.GRANTED,
    )
    with pytest.raises(ValidationError, match="inactive person"):
        controller.start_session(p_inactive.person_id)

    # Unconsented person
    p_unconsented = service.create_person(
        "Unconsented Person",
        status=PersonStatus.ACTIVE,
        consent_status=ConsentStatus.PENDING,
    )
    with pytest.raises(ValidationError, match="consent status is 'pending'"):
        controller.start_session(p_unconsented.person_id)


def test_single_active_session_enforcement(mock_pipeline: tuple) -> None:
    """Test that only one enrollment session can be active at any given time."""
    _, _, _, _, service, controller = mock_pipeline
    p1 = service.create_person("Patient 1")
    p2 = service.create_person("Patient 2")

    controller.start_session(p1.person_id)
    assert controller.is_active is True

    # Starting another session while first is active must raise ValidationError
    with pytest.raises(ValidationError, match="another session .* is already active"):
        controller.start_session(p2.person_id)

    controller.cancel_session()


def test_no_face_and_multiple_faces_rejection(mock_pipeline: tuple) -> None:
    """Test rejection when 0 faces or multiple faces are visible in the frame."""
    detector, _, _, _, service, controller = mock_pipeline
    person = service.create_person("John Doe")
    controller.start_session(person.person_id)
    dummy_frame = Frame(image=np.zeros((480, 640, 3), dtype=np.uint8), frame_id=1, timestamp=0.0)

    # 1. 0 detections -> NO_FACE
    detector.set_detections([])
    prog1 = controller.process_frame(dummy_frame)
    assert prog1.samples_collected == 0
    assert prog1.last_rejection_reason == SampleRejectionReason.NO_FACE
    assert prog1.rejection_counts[SampleRejectionReason.NO_FACE.value] == 1

    # 2. 2 detections -> MULTIPLE_FACES
    detector.set_detections(
        [
            FaceDetection(bbox=(10.0, 10.0, 50.0, 50.0), confidence=0.9),
            FaceDetection(bbox=(100.0, 10.0, 50.0, 50.0), confidence=0.9),
        ]
    )
    prog2 = controller.process_frame(dummy_frame)
    assert prog2.samples_collected == 0
    assert prog2.last_rejection_reason == SampleRejectionReason.MULTIPLE_FACES

    controller.cancel_session()


def test_quality_gate_rejection(mock_pipeline: tuple) -> None:
    """Test that candidate faces failing the quality gate are rejected."""
    detector, quality_gate, _, _, service, controller = mock_pipeline
    person = service.create_person("John Doe")
    controller.start_session(person.person_id)
    dummy_frame = Frame(image=np.zeros((480, 640, 3), dtype=np.uint8), frame_id=1, timestamp=0.0)

    detector.set_detections([FaceDetection(bbox=(10.0, 10.0, 50.0, 50.0), confidence=0.5)])
    quality_gate.set_acceptable(False)

    prog = controller.process_frame(dummy_frame)
    assert prog.samples_collected == 0
    assert prog.last_rejection_reason == SampleRejectionReason.QUALITY_REJECTED
    assert prog.rejection_counts[SampleRejectionReason.QUALITY_REJECTED.value] == 1

    controller.cancel_session()


def test_sample_diversity_and_duplicate_protection(mock_pipeline: tuple) -> None:
    """Test that near-identical frames are rejected to enforce sample diversity."""
    detector, quality_gate, _, embedder, service, controller = mock_pipeline
    person = service.create_person("John Doe")
    controller.start_session(person.person_id)
    dummy_frame = Frame(image=np.zeros((480, 640, 3), dtype=np.uint8), frame_id=1, timestamp=0.0)

    detector.set_detections([FaceDetection(bbox=(10.0, 10.0, 50.0, 50.0), confidence=0.95)])
    quality_gate.set_acceptable(True)

    # 1. First sample accepted
    v1 = normalize_l2(np.ones(128, dtype=np.float32))
    embedder.set_vector(v1)
    prog1 = controller.process_frame(dummy_frame)
    assert prog1.samples_collected == 1

    # 2. Identical vector -> DUPLICATE_SAMPLE rejection
    prog2 = controller.process_frame(dummy_frame)
    assert prog2.samples_collected == 1
    assert prog2.last_rejection_reason == SampleRejectionReason.DUPLICATE_SAMPLE

    # 3. Sufficiently distinct vector -> Accepted
    v2 = normalize_l2(np.array([1.0] * 64 + [-1.0] * 64, dtype=np.float32))
    embedder.set_vector(v2)
    prog3 = controller.process_frame(dummy_frame)
    assert prog3.samples_collected == 2
    assert prog3.last_rejection_reason is None

    controller.cancel_session()


def test_minimum_sample_requirement_on_commit(mock_pipeline: tuple) -> None:
    """Test that committing before meeting minimum samples fails and leaves no partial records."""
    detector, quality_gate, _, embedder, service, controller = mock_pipeline
    person = service.create_person("John Doe")
    controller.start_session(person.person_id)
    dummy_frame = Frame(image=np.zeros((480, 640, 3), dtype=np.uint8), frame_id=1, timestamp=0.0)

    detector.set_detections([FaceDetection(bbox=(10.0, 10.0, 50.0, 50.0), confidence=0.95)])
    quality_gate.set_acceptable(True)

    # Collect only 1 sample (minimum is 3)
    embedder.set_vector(np.random.randn(128).astype(np.float32))
    controller.process_frame(dummy_frame)

    # Commit must fail
    res = controller.commit_session()
    assert res.is_success is False
    assert res.state == EnrollmentState.FAILED
    assert "below required minimum" in str(res.error_message)

    # Verify zero embeddings stored in DB
    assert service.repository.count_embeddings(person.person_id) == 0


def test_successful_enrollment_and_single_gallery_sync(mock_pipeline: tuple) -> None:
    """Commit persists transactionally; gallery sync happens once at the app boundary."""
    detector, quality_gate, _, embedder, service, controller = mock_pipeline
    recognizer = FaceRecognizer()

    person = service.create_person("Alice Smith")
    controller.start_session(person.person_id)
    dummy_frame = Frame(image=np.zeros((480, 640, 3), dtype=np.uint8), frame_id=1, timestamp=0.0)

    detector.set_detections([FaceDetection(bbox=(10.0, 10.0, 50.0, 50.0), confidence=0.95)])
    quality_gate.set_acceptable(True)

    # Collect 5 diverse samples
    rng = np.random.default_rng(seed=123)
    for _ in range(5):
        vec = normalize_l2(rng.standard_normal(128).astype(np.float32))
        embedder.set_vector(vec)
        controller.process_frame(dummy_frame)

    assert controller.current_progress.is_complete is True  # type: ignore[union-attr]

    # Commit session
    res = controller.commit_session()
    assert res.is_success is True
    assert res.samples_enrolled == 5
    assert len(res.persisted_embedding_ids) == 5

    # Check persistence in SQLite
    assert service.repository.count_embeddings(person.person_id) == 5
    records = service.repository.get_embeddings_for_person(person.person_id)
    # Check that all records share the same batch ID
    batch_ids = {r.enrollment_batch_id for r in records}
    assert len(batch_ids) == 1
    assert res.enrollment_batch_id in batch_ids

    # Check that in-memory recognizer gallery was NOT synchronized by the controller:
    # the ApplicationService mutation barrier owns the single post-commit reload.
    assert len(recognizer.gallery) == 0

    # The single application-boundary synchronization produces the complete entry.
    recognizer.replace_gallery(service.load_recognition_gallery())
    assert len(recognizer.gallery) == 1
    gallery_entry = recognizer.gallery.get_entry(person.person_id)
    assert gallery_entry is not None
    assert len(gallery_entry.embeddings) == 5


def test_enrollment_add_samples_mode_and_existing_duplicate_protection(
    mock_pipeline: tuple,
) -> None:
    """Test ADD_SAMPLES mode rejects samples that duplicate already-persisted embeddings."""
    detector, quality_gate, _, embedder, service, controller = mock_pipeline
    person = service.create_person("David Smith")
    dummy_frame = Frame(image=np.zeros((480, 640, 3), dtype=np.uint8), frame_id=1, timestamp=0.0)

    detector.set_detections([FaceDetection(bbox=(10.0, 10.0, 50.0, 50.0), confidence=0.95)])
    quality_gate.set_acceptable(True)

    # 1. Pre-populate 1 existing embedding in DB
    existing_vec = normalize_l2(np.ones(128, dtype=np.float32))
    service.add_person_embedding(person.person_id, existing_vec)
    assert service.repository.count_embeddings(person.person_id) == 1

    # 2. Start ADD_SAMPLES session
    controller.start_session(person.person_id, mode=EnrollmentMode.ADD_SAMPLES)

    # 3. Present frame matching existing embedding -> Rejection
    embedder.set_vector(existing_vec)
    prog = controller.process_frame(dummy_frame)
    assert prog.samples_collected == 0
    assert prog.last_rejection_reason == SampleRejectionReason.DUPLICATE_SAMPLE
    assert "existing enrolled sample" in str(prog.last_rejection_details)

    # 4. Present 3 diverse new vectors
    rng = np.random.default_rng(seed=456)
    for _ in range(3):
        vec = normalize_l2(rng.standard_normal(128).astype(np.float32))
        embedder.set_vector(vec)
        controller.process_frame(dummy_frame)

    assert controller.current_progress.samples_collected == 3  # type: ignore[union-attr]

    # 5. Commit session -> total embeddings in DB should be 1 + 3 = 4
    res = controller.commit_session()
    assert res.is_success is True
    assert res.samples_enrolled == 3
    assert service.repository.count_embeddings(person.person_id) == 4


def test_staged_replacement_success_and_deferred_gallery_sync(mock_pipeline: tuple) -> None:
    """REPLACE persists atomically; gallery sync happens once at the app boundary."""
    detector, quality_gate, _, embedder, service, controller = mock_pipeline
    recognizer = FaceRecognizer()

    person = service.create_person("Old Enrollment Person")
    dummy_frame = Frame(image=np.zeros((480, 640, 3), dtype=np.uint8), frame_id=1, timestamp=0.0)
    detector.set_detections([FaceDetection(bbox=(10.0, 10.0, 50.0, 50.0), confidence=0.95)])
    quality_gate.set_acceptable(True)

    # 1. Old enrollment (3 embeddings)
    for _ in range(3):
        service.add_person_embedding(
            person.person_id,
            normalize_l2(np.random.randn(128).astype(np.float32)),
        )
    recognizer.replace_gallery(service.load_recognition_gallery())
    assert service.repository.count_embeddings(person.person_id) == 3
    assert len(recognizer.gallery.get_entry(person.person_id).embeddings) == 3

    # 2. Start REPLACE session
    prog = controller.start_session(person.person_id, mode=EnrollmentMode.REPLACE)
    assert prog.mode == EnrollmentMode.REPLACE

    # Old embeddings remain intact while collecting
    assert service.repository.count_embeddings(person.person_id) == 3

    # 3. Collect 4 diverse new samples
    rng = np.random.default_rng(seed=789)
    new_vectors = []
    for _ in range(4):
        vec = normalize_l2(rng.standard_normal(128).astype(np.float32))
        new_vectors.append(vec)
        embedder.set_vector(vec)
        controller.process_frame(dummy_frame)

    # Old embeddings still intact in DB before commit
    assert service.repository.count_embeddings(person.person_id) == 3

    # 4. Commit replacement
    res = controller.commit_session()
    assert res.is_success is True
    assert res.samples_enrolled == 4
    assert res.mode == EnrollmentMode.REPLACE

    # Verify old embeddings deleted and only new 4 remain
    assert service.repository.count_embeddings(person.person_id) == 4
    records = service.repository.get_embeddings_for_person(person.person_id)
    assert len(records) == 4
    assert {r.enrollment_batch_id for r in records} == {res.enrollment_batch_id}

    # Verify the controller did NOT sync the gallery itself: the stale entry still
    # reflects the pre-commit state, proving the single sync lives at the app boundary.
    stale_entry = recognizer.gallery.get_entry(person.person_id)
    assert stale_entry is not None
    assert len(stale_entry.embeddings) == 3

    # The application-boundary synchronization refreshes to the committed state.
    recognizer.replace_gallery(service.load_recognition_gallery())
    gallery_entry = recognizer.gallery.get_entry(person.person_id)
    assert gallery_entry is not None
    assert len(gallery_entry.embeddings) == 4


def test_staged_replacement_failure_preserves_old_enrollment(mock_pipeline: tuple) -> None:
    """Test failed/cancelled replacement preserves old enrollment completely intact."""
    detector, quality_gate, _, embedder, service, controller = mock_pipeline
    person = service.create_person("Protected Person")
    dummy_frame = Frame(image=np.zeros((480, 640, 3), dtype=np.uint8), frame_id=1, timestamp=0.0)
    detector.set_detections([FaceDetection(bbox=(10.0, 10.0, 50.0, 50.0), confidence=0.95)])
    quality_gate.set_acceptable(True)

    # 1. Existing enrollment of 4 samples
    for _ in range(4):
        service.add_person_embedding(
            person.person_id,
            normalize_l2(np.random.randn(128).astype(np.float32)),
        )
    initial_records = service.repository.get_embeddings_for_person(person.person_id)
    initial_ids = {r.embedding_id for r in initial_records}
    assert len(initial_ids) == 4

    # 2. Start REPLACE session and collect insufficient samples (< 3 minimum)
    controller.start_session(person.person_id, mode=EnrollmentMode.REPLACE)
    embedder.set_vector(normalize_l2(np.random.randn(128).astype(np.float32)))
    controller.process_frame(dummy_frame)  # only 1 sample

    # 3. Commit fails
    res = controller.commit_session()
    assert res.is_success is False
    assert res.state == EnrollmentState.FAILED

    # 4. Confirm original 4 embeddings are completely intact
    after_records = service.repository.get_embeddings_for_person(person.person_id)
    after_ids = {r.embedding_id for r in after_records}
    assert after_ids == initial_ids
    assert len(after_records) == 4


def test_invalid_enrollment_mode_raises_validation_error(mock_pipeline: tuple) -> None:
    """Test that passing an unrecognized mode string raises ValidationError."""
    _, _, _, _, service, controller = mock_pipeline
    person = service.create_person("Test Person")

    with pytest.raises(ValidationError, match="Invalid enrollment mode"):
        controller.start_session(person.person_id, mode="unknown_mode")  # type: ignore[arg-type]


def test_controller_persists_only_through_person_service(
    mock_pipeline: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M3.2.3: the controller must reach persistence only through PersonService.

    Poisoning the public ``repository`` property proves the controller persists via the
    service's transactional public methods (get vectors / add / replace) and never reaches
    directly into the persistence layer.
    """
    detector, quality_gate, _, embedder, service, controller = mock_pipeline

    def _forbid(_self: object) -> object:
        raise AssertionError("FaceEnrollmentController accessed PersonService.repository")

    monkeypatch.setattr(PersonService, "repository", property(_forbid))

    person = service.create_person("Boundary Person")
    # Pre-seed one embedding so ADD_SAMPLES exercises the vector-retrieval path.
    service.add_person_embedding(
        person.person_id, normalize_l2(np.random.randn(128).astype(np.float32))
    )

    dummy_frame = Frame(image=np.zeros((480, 640, 3), dtype=np.uint8), frame_id=1, timestamp=0.0)
    detector.set_detections([FaceDetection(bbox=(10.0, 10.0, 50.0, 50.0), confidence=0.95)])
    quality_gate.set_acceptable(True)

    # ADD_SAMPLES: uses get_person_embedding_vectors + add_person_embeddings.
    controller.start_session(person.person_id, mode=EnrollmentMode.ADD_SAMPLES)
    rng = np.random.default_rng(seed=99)
    for _ in range(5):
        embedder.set_vector(normalize_l2(rng.standard_normal(128).astype(np.float32)))
        controller.process_frame(dummy_frame)
    add_res = controller.commit_session()
    assert add_res.is_success is True
    assert service._repo.count_embeddings(person.person_id) == 6  # 1 seeded + 5 new

    # REPLACE: uses replace_person_embeddings.
    controller.start_session(person.person_id, mode=EnrollmentMode.REPLACE)
    rng2 = np.random.default_rng(seed=100)
    for _ in range(5):
        embedder.set_vector(normalize_l2(rng2.standard_normal(128).astype(np.float32)))
        controller.process_frame(dummy_frame)
    replace_res = controller.commit_session()
    assert replace_res.is_success is True
    assert service._repo.count_embeddings(person.person_id) == 5

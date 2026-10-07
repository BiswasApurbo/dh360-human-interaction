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


def test_first_frame_resets_timeout_clock(
    mock_pipeline: tuple,
) -> None:
    """Regression: the 30s timeout must measure active collection time, not the gap
    between session creation and the operator's first step call.

    A live webcam can take 5-10s to open (cv2.VideoCapture), plus operator setup
    time. The timeout clock should reset on the first frame so that the 30s window
    gives the operator 30 seconds of *active* sample collection, not 30 seconds
    from session creation (which may include idle time while the camera warms up
    and the operator positions the subject).
    """
    import time

    detector, _, _, _, service, controller = mock_pipeline
    person = service.create_person("Timed Person")
    controller.start_session(person.person_id)

    # Simulate 15 seconds having elapsed since start_session (beyond the
    # fixture's timeout_seconds=10.0) — e.g. camera open + operator setup.
    assert controller._session is not None
    controller._session.start_time = time.monotonic() - 15.0

    frame = Frame(image=np.zeros((480, 640, 3), dtype=np.uint8), frame_id=1, timestamp=0.0)
    detector.set_detections([FaceDetection(bbox=(10.0, 10.0, 50.0, 50.0), confidence=0.95)])

    # Without the fix, the first process_frame would immediately hit the
    # timeout check (15s > 10s) and mark the session FAILED with TIMEOUT.
    # With the fix, start_time is reset on the first frame (attempts==0),
    # so the frame is processed normally.
    prog = controller.process_frame(frame)

    assert prog.state == EnrollmentState.COLLECTING, (
        f"Expected COLLECTING but got {prog.state} — "
        f"first frame should reset the timeout clock, not immediately timeout"
    )
    assert prog.total_attempts == 1
    assert (
        prog.last_rejection_reason == SampleRejectionReason.NO_FACE or prog.samples_collected == 1
    )

    controller.cancel_session()


def test_timeout_still_fires_after_active_collection(
    mock_pipeline: tuple,
) -> None:
    """The timeout still fires when >timeout_seconds elapse *after* the first step."""
    import time

    detector, _, _, _, service, controller = mock_pipeline
    person = service.create_person("Timeout Person")
    controller.start_session(person.person_id)

    frame = Frame(image=np.zeros((480, 640, 3), dtype=np.uint8), frame_id=1, timestamp=0.0)
    detector.set_detections([])  # No face → frame processed but no sample

    # First frame: resets the clock, increments attempts to 1.
    prog1 = controller.process_frame(frame)
    assert prog1.total_attempts == 1
    assert prog1.state == EnrollmentState.COLLECTING

    # Simulate 15 seconds of active stepping (beyond timeout_seconds=10.0)
    # with no samples collected.
    assert controller._session is not None
    controller._session.start_time = time.monotonic() - 15.0

    # Second frame: should hit the timeout now.
    prog2 = controller.process_frame(frame)
    assert prog2.state == EnrollmentState.FAILED
    assert prog2.last_rejection_reason == SampleRejectionReason.TIMEOUT
    assert prog2.samples_collected == 0

    controller.cancel_session()


def test_guided_enrollment_direction_counting_and_quota_enforcement(
    mock_pipeline: tuple,
) -> None:
    """Verify 5-per-direction sample counting, quota full rejection, and progress fields."""
    from dh360_hi.contracts.dto import EnrollmentProgressDTO
    from dh360_hi.vision.detector import FaceLandmarks
    from dh360_hi.vision.enrollment import EnrollmentConfig, FaceEnrollmentController
    from dh360_hi.vision.quality_gate import HeadDirection

    detector, quality_gate, aligner, embedder, service, _ = mock_pipeline
    guided_config = EnrollmentConfig(
        target_samples=25,
        minimum_samples=5,
        min_sample_interval_seconds=0.0,
    )
    controller = FaceEnrollmentController(
        detector=detector,
        quality_gate=quality_gate,
        aligner=aligner,
        embedder=embedder,
        person_service=service,
        config=guided_config,
    )
    person = service.create_person("Guided Person")
    prog = controller.start_session(person.person_id)

    # Initial progress has default targets and zero counts
    assert prog.direction_targets[HeadDirection.FRONT.value] == 5
    assert prog.direction_targets[HeadDirection.RIGHT.value] == 5
    assert prog.direction_counts[HeadDirection.FRONT.value] == 0
    assert prog.current_direction == HeadDirection.FRONT.value

    # DTO round-trip preserves direction data
    dto = EnrollmentProgressDTO.from_progress(prog)
    assert dto.current_direction == HeadDirection.FRONT.value
    assert dto.direction_targets[HeadDirection.FRONT.value] == 5
    assert dto.direction_counts[HeadDirection.FRONT.value] == 0

    frame = Frame(image=np.zeros((480, 640, 3), dtype=np.uint8), frame_id=1, timestamp=0.0)

    # Configure frontal landmarks
    lm_front = FaceLandmarks(
        right_eye=(70.0, 70.0),
        left_eye=(130.0, 70.0),
        nose_tip=(100.0, 104.3),
        right_mouth_corner=(75.0, 140.0),
        left_mouth_corner=(125.0, 140.0),
    )
    detector.set_detections(
        [FaceDetection(bbox=(20.0, 20.0, 100.0, 100.0), confidence=0.95, landmarks=lm_front)]
    )

    # Collect 5 frontal samples with unique embeddings
    for i in range(5):
        vec = np.zeros(128, dtype=np.float32)
        vec[i] = 1.0
        embedder.set_vector(vec)
        prog = controller.process_frame(frame)
        assert prog.direction_counts[HeadDirection.FRONT.value] == i + 1

    assert prog.direction_counts[HeadDirection.FRONT.value] == 5

    # 6th frontal sample must be rejected with DIRECTION_QUOTA_FULL
    vec6 = np.zeros(128, dtype=np.float32)
    vec6[5] = 1.0
    embedder.set_vector(vec6)
    prog = controller.process_frame(frame)
    assert prog.last_rejection_reason == SampleRejectionReason.DIRECTION_QUOTA_FULL
    assert prog.direction_counts[HeadDirection.FRONT.value] == 5
    assert prog.samples_collected == 5

    controller.cancel_session()


def test_guided_enrollment_completes_at_25_samples_across_all_directions(
    mock_pipeline: tuple,
) -> None:
    """Verify session completes when all 5 directions have 5 accepted samples (25 total)."""
    from dh360_hi.vision.detector import FaceLandmarks
    from dh360_hi.vision.enrollment import EnrollmentConfig, FaceEnrollmentController
    from dh360_hi.vision.quality_gate import HeadDirection

    detector, quality_gate, aligner, embedder, service, _ = mock_pipeline
    guided_config = EnrollmentConfig(
        target_samples=25,
        minimum_samples=5,
        min_sample_interval_seconds=0.0,
    )
    controller = FaceEnrollmentController(
        detector=detector,
        quality_gate=quality_gate,
        aligner=aligner,
        embedder=embedder,
        person_service=service,
        config=guided_config,
    )
    person = service.create_person("All Directions Person")
    controller.start_session(person.person_id)

    frame = Frame(image=np.zeros((480, 640, 3), dtype=np.uint8), frame_id=1, timestamp=0.0)

    directions_landmarks = {
        HeadDirection.FRONT: FaceLandmarks(
            right_eye=(70.0, 70.0),
            left_eye=(130.0, 70.0),
            nose_tip=(100.0, 104.3),
            right_mouth_corner=(75.0, 140.0),
            left_mouth_corner=(125.0, 140.0),
        ),
        HeadDirection.RIGHT: FaceLandmarks(
            right_eye=(70.0, 70.0),
            left_eye=(130.0, 70.0),
            nose_tip=(88.0, 104.3),
            right_mouth_corner=(75.0, 140.0),
            left_mouth_corner=(125.0, 140.0),
        ),
        HeadDirection.LEFT: FaceLandmarks(
            right_eye=(70.0, 70.0),
            left_eye=(130.0, 70.0),
            nose_tip=(112.0, 104.3),
            right_mouth_corner=(75.0, 140.0),
            left_mouth_corner=(125.0, 140.0),
        ),
        HeadDirection.UP: FaceLandmarks(
            right_eye=(70.0, 70.0),
            left_eye=(130.0, 70.0),
            nose_tip=(100.0, 90.0),
            right_mouth_corner=(75.0, 140.0),
            left_mouth_corner=(125.0, 140.0),
        ),
        HeadDirection.DOWN: FaceLandmarks(
            right_eye=(70.0, 70.0),
            left_eye=(130.0, 70.0),
            nose_tip=(100.0, 118.0),
            right_mouth_corner=(75.0, 140.0),
            left_mouth_corner=(125.0, 140.0),
        ),
    }

    sample_idx = 0
    prog = None
    for _direction, lm in directions_landmarks.items():
        detector.set_detections(
            [FaceDetection(bbox=(20.0, 20.0, 100.0, 100.0), confidence=0.95, landmarks=lm)]
        )
        for _ in range(5):
            vec = np.zeros(128, dtype=np.float32)
            vec[sample_idx % 128] = 1.0
            vec[(sample_idx + 1) % 128] = float(sample_idx + 1)
            embedder.set_vector(vec)
            sample_idx += 1
            prog = controller.process_frame(frame)

    assert prog is not None
    assert prog.samples_collected == 25
    assert prog.is_complete is True
    assert prog.state == EnrollmentState.COMPLETED
    for d in directions_landmarks:
        assert prog.direction_counts[d.value] == 5

    # Commit persists all 25 embeddings
    res = controller.commit_session()
    assert res.is_success is True
    assert res.samples_enrolled == 25
    stored_vectors = service.get_person_embedding_vectors(person.person_id)
    assert len(stored_vectors) == 25


def test_front_samples_alone_cannot_complete_enrollment_or_commit(
    mock_pipeline: tuple,
) -> None:
    """CRITICAL: Presenting 25 FRONT frames cannot complete enrollment or commit.

    The session must stop accepting FRONT frames at 5, reject further FRONT frames with
    DIRECTION_QUOTA_FULL, advance stage guidance to RIGHT, remain incomplete, and reject
    commit if all 5 direction targets are not satisfied.
    """
    from dh360_hi.vision.detector import FaceLandmarks
    from dh360_hi.vision.enrollment import EnrollmentConfig, FaceEnrollmentController
    from dh360_hi.vision.quality_gate import HeadDirection

    detector, quality_gate, aligner, embedder, service, _ = mock_pipeline
    guided_config = EnrollmentConfig(
        target_samples=25,
        minimum_samples=5,
        min_sample_interval_seconds=0.0,
    )
    controller = FaceEnrollmentController(
        detector=detector,
        quality_gate=quality_gate,
        aligner=aligner,
        embedder=embedder,
        person_service=service,
        config=guided_config,
    )
    person = service.create_person("Front Only Person")
    prog = controller.start_session(person.person_id)
    assert prog.current_direction == HeadDirection.FRONT.value

    frame = Frame(image=np.zeros((480, 640, 3), dtype=np.uint8), frame_id=1, timestamp=0.0)
    lm_front = FaceLandmarks(
        right_eye=(70.0, 70.0),
        left_eye=(130.0, 70.0),
        nose_tip=(100.0, 104.3),
        right_mouth_corner=(75.0, 140.0),
        left_mouth_corner=(125.0, 140.0),
    )
    detector.set_detections(
        [FaceDetection(bbox=(20.0, 20.0, 100.0, 100.0), confidence=0.95, landmarks=lm_front)]
    )

    # 1. Present 5 valid FRONT frames -> accepted
    for i in range(5):
        vec = np.zeros(128, dtype=np.float32)
        vec[i] = 1.0
        embedder.set_vector(vec)
        prog = controller.process_frame(frame)
        assert prog.direction_counts[HeadDirection.FRONT.value] == i + 1

    assert prog.direction_counts[HeadDirection.FRONT.value] == 5
    assert prog.samples_collected == 5
    # Next required stage must be RIGHT
    assert prog.current_direction == HeadDirection.RIGHT.value
    assert prog.is_complete is False

    # 2. Present 20 more FRONT frames (making 25 total attempts with FRONT)
    for i in range(5, 25):
        vec = np.zeros(128, dtype=np.float32)
        vec[i % 128] = float(i + 1)
        embedder.set_vector(vec)
        prog = controller.process_frame(frame)
        assert prog.last_rejection_reason == SampleRejectionReason.DIRECTION_QUOTA_FULL
        assert prog.samples_collected == 5
        assert prog.direction_counts[HeadDirection.FRONT.value] == 5
        assert prog.current_direction == HeadDirection.RIGHT.value
        assert prog.is_complete is False

    # Total accepted samples is still strictly 5, NOT 25
    assert prog.samples_collected == 5
    assert prog.is_complete is False
    assert prog.state == EnrollmentState.COLLECTING

    # 3. Attempting to commit must fail because 4 directions are unmet
    res = controller.commit_session()
    assert res.is_success is False
    assert res.state == EnrollmentState.FAILED
    assert res.samples_enrolled == 0
    assert len(res.persisted_embedding_ids) == 0
    assert "unmet direction targets" in str(res.error_message)

    # Database has 0 stored embeddings for this person
    stored = service.get_person_embedding_vectors(person.person_id)
    assert len(stored) == 0


def test_guided_enrollment_enforces_strict_stage_sequence(
    mock_pipeline: tuple,
) -> None:
    """Verify strict stage ordering: FRONT -> RIGHT -> LEFT -> UP -> DOWN.

    Frames presented out of sequence are rejected with WRONG_DIRECTION.
    Only frames matching the active stage direction are accepted.
    """
    from dh360_hi.vision.detector import FaceLandmarks
    from dh360_hi.vision.enrollment import EnrollmentConfig, FaceEnrollmentController
    from dh360_hi.vision.quality_gate import HeadDirection

    detector, quality_gate, aligner, embedder, service, _ = mock_pipeline
    guided_config = EnrollmentConfig(
        target_samples=25,
        minimum_samples=5,
        min_sample_interval_seconds=0.0,
    )
    controller = FaceEnrollmentController(
        detector=detector,
        quality_gate=quality_gate,
        aligner=aligner,
        embedder=embedder,
        person_service=service,
        config=guided_config,
    )
    person = service.create_person("Strict Sequence Person")
    prog = controller.start_session(person.person_id)

    frame = Frame(image=np.zeros((480, 640, 3), dtype=np.uint8), frame_id=1, timestamp=0.0)

    landmarks_by_dir = {
        HeadDirection.FRONT: FaceLandmarks(
            right_eye=(70.0, 70.0),
            left_eye=(130.0, 70.0),
            nose_tip=(100.0, 104.3),
            right_mouth_corner=(75.0, 140.0),
            left_mouth_corner=(125.0, 140.0),
        ),
        HeadDirection.RIGHT: FaceLandmarks(
            right_eye=(70.0, 70.0),
            left_eye=(130.0, 70.0),
            nose_tip=(88.0, 104.3),
            right_mouth_corner=(75.0, 140.0),
            left_mouth_corner=(125.0, 140.0),
        ),
        HeadDirection.LEFT: FaceLandmarks(
            right_eye=(70.0, 70.0),
            left_eye=(130.0, 70.0),
            nose_tip=(112.0, 104.3),
            right_mouth_corner=(75.0, 140.0),
            left_mouth_corner=(125.0, 140.0),
        ),
        HeadDirection.UP: FaceLandmarks(
            right_eye=(70.0, 70.0),
            left_eye=(130.0, 70.0),
            nose_tip=(100.0, 90.0),
            right_mouth_corner=(75.0, 140.0),
            left_mouth_corner=(125.0, 140.0),
        ),
        HeadDirection.DOWN: FaceLandmarks(
            right_eye=(70.0, 70.0),
            left_eye=(130.0, 70.0),
            nose_tip=(100.0, 118.0),
            right_mouth_corner=(75.0, 140.0),
            left_mouth_corner=(125.0, 140.0),
        ),
    }

    # Stage 1: Active stage is FRONT.
    assert prog.current_direction == HeadDirection.FRONT.value
    # Present RIGHT frame out of turn -> rejected with WRONG_DIRECTION
    detector.set_detections(
        [
            FaceDetection(
                bbox=(20.0, 20.0, 100.0, 100.0),
                confidence=0.95,
                landmarks=landmarks_by_dir[HeadDirection.RIGHT],
            )
        ]
    )
    vec = np.zeros(128, dtype=np.float32)
    vec[0] = 1.0
    embedder.set_vector(vec)
    prog = controller.process_frame(frame)
    assert prog.last_rejection_reason == SampleRejectionReason.WRONG_DIRECTION
    assert prog.samples_collected == 0
    assert prog.direction_counts[HeadDirection.RIGHT.value] == 0
    assert prog.direction_counts[HeadDirection.FRONT.value] == 0
    assert prog.current_direction == HeadDirection.FRONT.value

    # Now step through each stage in order: FRONT, RIGHT, LEFT, UP, DOWN
    sequence = [
        HeadDirection.FRONT,
        HeadDirection.RIGHT,
        HeadDirection.LEFT,
        HeadDirection.UP,
        HeadDirection.DOWN,
    ]
    sample_counter = 0
    for stage_idx, stage_dir in enumerate(sequence):
        assert prog.current_direction == stage_dir.value
        detector.set_detections(
            [
                FaceDetection(
                    bbox=(20.0, 20.0, 100.0, 100.0),
                    confidence=0.95,
                    landmarks=landmarks_by_dir[stage_dir],
                )
            ]
        )
        for count in range(1, 6):
            sample_counter += 1
            v = np.zeros(128, dtype=np.float32)
            v[sample_counter % 128] = float(sample_counter)
            embedder.set_vector(v)
            prog = controller.process_frame(frame)
            assert prog.direction_counts[stage_dir.value] == count
            assert prog.samples_collected == stage_idx * 5 + count

        if stage_idx < len(sequence) - 1:
            next_stage = sequence[stage_idx + 1]
            assert prog.current_direction == next_stage.value
            assert prog.is_complete is False

    # After DOWN 5/5: enrollment is complete
    assert prog.samples_collected == 25
    assert prog.is_complete is True
    assert prog.state == EnrollmentState.COMPLETED

    res = controller.commit_session()
    assert res.is_success is True
    assert res.samples_enrolled == 25
    assert len(service.get_person_embedding_vectors(person.person_id)) == 25


def test_guided_enrollment_commit_fails_if_any_direction_unmet(
    mock_pipeline: tuple,
) -> None:
    """Verify commit fails if only 4 out of 5 directions have reached their quota."""
    from dh360_hi.vision.detector import FaceLandmarks
    from dh360_hi.vision.enrollment import EnrollmentConfig, FaceEnrollmentController
    from dh360_hi.vision.quality_gate import HeadDirection

    detector, quality_gate, aligner, embedder, service, _ = mock_pipeline
    guided_config = EnrollmentConfig(
        target_samples=25,
        minimum_samples=5,
        min_sample_interval_seconds=0.0,
    )
    controller = FaceEnrollmentController(
        detector=detector,
        quality_gate=quality_gate,
        aligner=aligner,
        embedder=embedder,
        person_service=service,
        config=guided_config,
    )
    person = service.create_person("Four Directions Person")
    controller.start_session(person.person_id)

    frame = Frame(image=np.zeros((480, 640, 3), dtype=np.uint8), frame_id=1, timestamp=0.0)

    landmarks_by_dir = {
        HeadDirection.FRONT: FaceLandmarks(
            right_eye=(70.0, 70.0),
            left_eye=(130.0, 70.0),
            nose_tip=(100.0, 104.3),
            right_mouth_corner=(75.0, 140.0),
            left_mouth_corner=(125.0, 140.0),
        ),
        HeadDirection.RIGHT: FaceLandmarks(
            right_eye=(70.0, 70.0),
            left_eye=(130.0, 70.0),
            nose_tip=(88.0, 104.3),
            right_mouth_corner=(75.0, 140.0),
            left_mouth_corner=(125.0, 140.0),
        ),
        HeadDirection.LEFT: FaceLandmarks(
            right_eye=(70.0, 70.0),
            left_eye=(130.0, 70.0),
            nose_tip=(112.0, 104.3),
            right_mouth_corner=(75.0, 140.0),
            left_mouth_corner=(125.0, 140.0),
        ),
        HeadDirection.UP: FaceLandmarks(
            right_eye=(70.0, 70.0),
            left_eye=(130.0, 70.0),
            nose_tip=(100.0, 90.0),
            right_mouth_corner=(75.0, 140.0),
            left_mouth_corner=(125.0, 140.0),
        ),
    }

    sample_counter = 0
    for _d, lm in landmarks_by_dir.items():
        detector.set_detections(
            [FaceDetection(bbox=(20.0, 20.0, 100.0, 100.0), confidence=0.95, landmarks=lm)]
        )
        for _ in range(5):
            sample_counter += 1
            v = np.zeros(128, dtype=np.float32)
            v[sample_counter % 128] = float(sample_counter)
            embedder.set_vector(v)
            controller.process_frame(frame)

    # 20 samples collected (FRONT, RIGHT, LEFT, UP all 5, DOWN is 0)
    assert controller.current_progress.samples_collected == 20
    assert controller.current_progress.current_direction == HeadDirection.DOWN.value
    assert controller.current_progress.is_complete is False

    # Attempt commit before DOWN
    res = controller.commit_session()
    assert res.is_success is False
    assert res.state == EnrollmentState.FAILED
    assert res.samples_enrolled == 0
    assert "unmet direction targets" in str(res.error_message)
    assert len(service.get_person_embedding_vectors(person.person_id)) == 0


def test_real_turned_face_accepted_during_directional_stages(
    mock_pipeline: tuple,
) -> None:
    """Verify that a realistic turned face (with yaw asymmetry > 2.5) is accepted as RIGHT

    and not rejected by quality gate, while opposite poses and FRONT are rejected.
    """
    from dh360_hi.vision.detector import FaceLandmarks
    from dh360_hi.vision.enrollment import EnrollmentConfig, FaceEnrollmentController
    from dh360_hi.vision.quality_gate import HeadDirection

    detector, quality_gate, aligner, embedder, service, _ = mock_pipeline
    guided_config = EnrollmentConfig(
        target_samples=25,
        minimum_samples=5,
        min_sample_interval_seconds=0.0,
    )
    controller = FaceEnrollmentController(
        detector=detector,
        quality_gate=quality_gate,
        aligner=aligner,
        embedder=embedder,
        person_service=service,
        config=guided_config,
    )
    person = service.create_person("Turned Pose Person")
    prog = controller.start_session(person.person_id)

    frame = Frame(image=np.zeros((480, 640, 3), dtype=np.uint8), frame_id=1, timestamp=0.0)

    # 1. Complete Stage 1 (FRONT) with 5 samples
    lm_front = FaceLandmarks(
        right_eye=(70.0, 70.0),
        left_eye=(130.0, 70.0),
        nose_tip=(100.0, 105.0),
        right_mouth_corner=(75.0, 140.0),
        left_mouth_corner=(125.0, 140.0),
    )
    detector.set_detections(
        [FaceDetection(bbox=(20.0, 20.0, 100.0, 100.0), confidence=0.95, landmarks=lm_front)]
    )
    for i in range(5):
        vec = np.zeros(128, dtype=np.float32)
        vec[i] = 1.0
        embedder.set_vector(vec)
        prog = controller.process_frame(frame)

    assert prog.direction_counts[HeadDirection.FRONT.value] == 5
    assert prog.current_direction == HeadDirection.RIGHT.value

    # 2. In Stage 2 (RIGHT):
    # Present realistic RIGHT pose: nose shifted to x=82 (yaw asymmetry = 48/12 = 4.0 > 2.5)
    lm_right_strong = FaceLandmarks(
        right_eye=(70.0, 70.0),
        left_eye=(130.0, 70.0),
        nose_tip=(82.0, 105.0),
        right_mouth_corner=(75.0, 140.0),
        left_mouth_corner=(125.0, 140.0),
    )
    detector.set_detections(
        [FaceDetection(bbox=(20.0, 20.0, 100.0, 100.0), confidence=0.95, landmarks=lm_right_strong)]
    )
    vec = np.zeros(128, dtype=np.float32)
    vec[10] = 1.0
    embedder.set_vector(vec)
    prog = controller.process_frame(frame)

    # MUST be accepted as RIGHT, not rejected as excessive_yaw!
    assert prog.direction_counts[HeadDirection.RIGHT.value] == 1
    assert prog.samples_collected == 6
    assert prog.last_rejection_reason is None

    # Present FRONT pose while in RIGHT stage:
    detector.set_detections(
        [FaceDetection(bbox=(20.0, 20.0, 100.0, 100.0), confidence=0.95, landmarks=lm_front)]
    )
    prog = controller.process_frame(frame)
    # Must be rejected, must NOT increment RIGHT counter
    assert prog.direction_counts[HeadDirection.RIGHT.value] == 1
    assert prog.last_rejection_reason == SampleRejectionReason.DIRECTION_QUOTA_FULL

    # Present opposite (LEFT) pose while in RIGHT stage:
    lm_left_strong = FaceLandmarks(
        right_eye=(70.0, 70.0),
        left_eye=(130.0, 70.0),
        nose_tip=(118.0, 105.0),
        right_mouth_corner=(75.0, 140.0),
        left_mouth_corner=(125.0, 140.0),
    )
    detector.set_detections(
        [FaceDetection(bbox=(20.0, 20.0, 100.0, 100.0), confidence=0.95, landmarks=lm_left_strong)]
    )
    prog = controller.process_frame(frame)
    # Must be rejected, must NOT increment RIGHT counter
    assert prog.direction_counts[HeadDirection.RIGHT.value] == 1
    assert prog.last_rejection_reason == SampleRejectionReason.WRONG_DIRECTION


def test_guided_enrollment_up_and_down_lifecycle_completion(
    mock_pipeline: tuple,
) -> None:
    """Verify UP and DOWN directional stage enforcement, rejection of incorrect poses,

    advancement from UP to DOWN after 5 UP samples, and completion at 25/25 total.
    """
    from dh360_hi.vision.detector import FaceLandmarks
    from dh360_hi.vision.enrollment import (
        EnrollmentConfig,
        FaceEnrollmentController,
        SampleRejectionReason,
    )
    from dh360_hi.vision.quality_gate import HeadDirection

    detector, quality_gate, aligner, embedder, service, _ = mock_pipeline
    guided_config = EnrollmentConfig(
        target_samples=25,
        minimum_samples=5,
        min_sample_interval_seconds=0.0,
    )
    controller = FaceEnrollmentController(
        detector=detector,
        quality_gate=quality_gate,
        aligner=aligner,
        embedder=embedder,
        person_service=service,
        config=guided_config,
    )
    person = service.create_person("UpDown Lifecycle Person")
    prog = controller.start_session(person.person_id)

    frame = Frame(image=np.zeros((480, 640, 3), dtype=np.uint8), frame_id=1, timestamp=0.0)

    # Realistic landmarks for each direction
    # Resting face has slight natural asymmetry/offset (ratio = 0.52: nt_y = 106.4)
    lm_front = FaceLandmarks(
        right_eye=(70.0, 70.0),
        left_eye=(130.0, 70.0),
        nose_tip=(100.0, 106.4),
        right_mouth_corner=(75.0, 140.0),
        left_mouth_corner=(125.0, 140.0),
    )
    lm_right = FaceLandmarks(
        right_eye=(70.0, 70.0),
        left_eye=(130.0, 70.0),
        nose_tip=(93.0, 106.4),
        right_mouth_corner=(75.0, 140.0),
        left_mouth_corner=(125.0, 140.0),
    )
    lm_left = FaceLandmarks(
        right_eye=(70.0, 70.0),
        left_eye=(130.0, 70.0),
        nose_tip=(107.0, 106.4),
        right_mouth_corner=(75.0, 140.0),
        left_mouth_corner=(125.0, 140.0),
    )
    # UP pose with slight natural yaw (nt_x=96.0, nt_y=99.0 -> ratio=0.414, pitch_off=-0.106)
    lm_up = FaceLandmarks(
        right_eye=(70.0, 70.0),
        left_eye=(130.0, 70.0),
        nose_tip=(96.0, 99.0),
        right_mouth_corner=(75.0, 140.0),
        left_mouth_corner=(125.0, 140.0),
    )
    # DOWN pose with slight natural yaw (nt_x=104.0, nt_y=113.8 -> ratio=0.626, pitch_off=+0.106)
    lm_down = FaceLandmarks(
        right_eye=(70.0, 70.0),
        left_eye=(130.0, 70.0),
        nose_tip=(104.0, 113.8),
        right_mouth_corner=(75.0, 140.0),
        left_mouth_corner=(125.0, 140.0),
    )

    sample_counter = 0

    # 1. Complete Stage 1 (FRONT): 5 frames
    detector.set_detections(
        [FaceDetection(bbox=(20, 20, 100, 100), confidence=0.95, landmarks=lm_front)]
    )
    for _ in range(5):
        vec = np.zeros(128, dtype=np.float32)
        vec[sample_counter % 128] = 1.0
        vec[(sample_counter + 1) % 128] = float(sample_counter + 1)
        embedder.set_vector(vec)
        sample_counter += 1
        prog = controller.process_frame(frame)

    assert prog.direction_counts[HeadDirection.FRONT.value] == 5
    assert prog.current_direction == HeadDirection.RIGHT.value

    # 2. Complete Stage 2 (RIGHT): 5 frames
    detector.set_detections(
        [FaceDetection(bbox=(20, 20, 100, 100), confidence=0.95, landmarks=lm_right)]
    )
    for _ in range(5):
        vec = np.zeros(128, dtype=np.float32)
        vec[sample_counter % 128] = 1.0
        vec[(sample_counter + 1) % 128] = float(sample_counter + 1)
        embedder.set_vector(vec)
        sample_counter += 1
        prog = controller.process_frame(frame)

    assert prog.direction_counts[HeadDirection.RIGHT.value] == 5
    assert prog.current_direction == HeadDirection.LEFT.value

    # 3. Complete Stage 3 (LEFT): 5 frames
    detector.set_detections(
        [FaceDetection(bbox=(20, 20, 100, 100), confidence=0.95, landmarks=lm_left)]
    )
    for _ in range(5):
        vec = np.zeros(128, dtype=np.float32)
        vec[sample_counter % 128] = 1.0
        vec[(sample_counter + 1) % 128] = float(sample_counter + 1)
        embedder.set_vector(vec)
        sample_counter += 1
        prog = controller.process_frame(frame)

    assert prog.direction_counts[HeadDirection.LEFT.value] == 5
    assert prog.samples_collected == 15
    # Must now advance to UP
    assert prog.current_direction == HeadDirection.UP.value
    assert prog.direction_counts[HeadDirection.UP.value] == 0

    # 4. In Stage 4 (UP): test rejection of non-UP poses
    # Feeding FRONT pose -> DIRECTION_QUOTA_FULL (FRONT is already 5/5)
    detector.set_detections(
        [FaceDetection(bbox=(20, 20, 100, 100), confidence=0.95, landmarks=lm_front)]
    )
    prog = controller.process_frame(frame)
    assert prog.last_rejection_reason == SampleRejectionReason.DIRECTION_QUOTA_FULL
    assert prog.direction_counts[HeadDirection.UP.value] == 0
    assert prog.samples_collected == 15

    # Feeding DOWN pose -> WRONG_DIRECTION (DOWN is not the active stage)
    detector.set_detections(
        [FaceDetection(bbox=(20, 20, 100, 100), confidence=0.95, landmarks=lm_down)]
    )
    prog = controller.process_frame(frame)
    assert prog.last_rejection_reason == SampleRejectionReason.WRONG_DIRECTION
    assert prog.direction_counts[HeadDirection.UP.value] == 0
    assert prog.samples_collected == 15

    # Feed 5 valid UP frames -> accepted one-by-one
    detector.set_detections(
        [FaceDetection(bbox=(20, 20, 100, 100), confidence=0.95, landmarks=lm_up)]
    )
    for i in range(5):
        vec = np.zeros(128, dtype=np.float32)
        vec[sample_counter % 128] = 1.0
        vec[(sample_counter + 1) % 128] = float(sample_counter + 1)
        embedder.set_vector(vec)
        sample_counter += 1
        prog = controller.process_frame(frame)
        assert prog.direction_counts[HeadDirection.UP.value] == i + 1

    assert prog.direction_counts[HeadDirection.UP.value] == 5
    assert prog.samples_collected == 20
    # Must now advance to Stage 5 (DOWN)
    assert prog.current_direction == HeadDirection.DOWN.value
    assert prog.direction_counts[HeadDirection.DOWN.value] == 0

    # 5. In Stage 5 (DOWN): test rejection of non-DOWN poses
    # Feeding UP pose -> DIRECTION_QUOTA_FULL (UP is already 5/5)
    detector.set_detections(
        [FaceDetection(bbox=(20, 20, 100, 100), confidence=0.95, landmarks=lm_up)]
    )
    prog = controller.process_frame(frame)
    assert prog.last_rejection_reason == SampleRejectionReason.DIRECTION_QUOTA_FULL
    assert prog.direction_counts[HeadDirection.DOWN.value] == 0
    assert prog.samples_collected == 20

    # Feeding FRONT pose -> DIRECTION_QUOTA_FULL (FRONT is already 5/5)
    detector.set_detections(
        [FaceDetection(bbox=(20, 20, 100, 100), confidence=0.95, landmarks=lm_front)]
    )
    prog = controller.process_frame(frame)
    assert prog.last_rejection_reason == SampleRejectionReason.DIRECTION_QUOTA_FULL
    assert prog.direction_counts[HeadDirection.DOWN.value] == 0
    assert prog.samples_collected == 20

    # Feed 5 valid DOWN frames -> accepted one-by-one
    detector.set_detections(
        [FaceDetection(bbox=(20, 20, 100, 100), confidence=0.95, landmarks=lm_down)]
    )
    for i in range(5):
        vec = np.zeros(128, dtype=np.float32)
        vec[sample_counter % 128] = 1.0
        vec[(sample_counter + 1) % 128] = float(sample_counter + 1)
        embedder.set_vector(vec)
        sample_counter += 1
        prog = controller.process_frame(frame)
        assert prog.direction_counts[HeadDirection.DOWN.value] == i + 1

    # 6. Completion: All 5 directions at 5/5, total 25/25
    assert prog.samples_collected == 25
    assert prog.is_complete is True
    assert prog.state == EnrollmentState.COMPLETED
    for d in ["front", "right", "left", "up", "down"]:
        assert prog.direction_counts[d] == 5

    # Commit succeeds and stores 25 embeddings
    res = controller.commit_session()
    assert res.is_success is True
    assert res.samples_enrolled == 25
    stored_vectors = service.get_person_embedding_vectors(person.person_id)
    assert len(stored_vectors) == 25

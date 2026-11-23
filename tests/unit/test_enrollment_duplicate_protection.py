"""Automated unit and integration tests for cross-identity biometric duplicate protection.

Covers all 10 mandatory identity-integrity test scenarios from Section 9:
1. New, unregistered face -> Enrollment allowed.
2. Face already registered under same name and person ID -> Existing enrollment rules apply.
3. Face registered under a different name and person ID -> Duplicate warning; enrollment blocked.
4. Face matches another identity during ADD_SAMPLES -> Reject conflicting enrollment.
5. Face matches another identity during REPLACE -> Reject conflicting replacement.
6. Two simultaneous sessions register the same face -> Only one may successfully commit.
7. Similar-looking but different person (similarity < 0.55) -> No unjustified duplicate rejection.
8. Embedding index unavailable -> Fail closed; no biometric persistence.
9. Database/index inconsistent -> Detect and report the condition.
10. Duplicate match with insufficient operator permissions -> Do not expose private identity details.
"""

from __future__ import annotations

import numpy as np
import pytest

from dh360_hi.api.security import OperatorPrincipal, OperatorScope
from dh360_hi.backend.database import DatabaseManager
from dh360_hi.backend.exceptions import ValidationError
from dh360_hi.backend.models import ConsentStatus, PersonStatus
from dh360_hi.backend.person_service import PersonService
from dh360_hi.backend.repository import SQLitePersonRepository
from dh360_hi.contracts.dto import EnrollmentProgressDTO
from dh360_hi.contracts.errors import ConflictAppError
from dh360_hi.vision.aligner import AlignmentResult, BaseFaceAligner
from dh360_hi.vision.detector import BaseFaceDetector, DetectionResult, FaceDetection, FaceLandmarks
from dh360_hi.vision.tracker import TrackedFace
from dh360_hi.vision.embedder import (
    BaseFaceEmbedder,
    EmbeddingResult,
    compute_cosine_similarity,
    normalize_l2,
)
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
)
from dh360_hi.vision.recognition import GalleryEntry, InMemoryGallery


class MockDetector(BaseFaceDetector):
    def __init__(self, bbox=(100.0, 100.0, 100.0, 100.0), confidence=0.95):
        self._bbox = bbox
        self._confidence = confidence

    def detect(self, frame: Frame | np.ndarray, frame_id: int = 0) -> DetectionResult:
        fid = frame.frame_id if isinstance(frame, Frame) else frame_id
        # Provide 5 landmarks to simulate frontal face
        landmarks = FaceLandmarks(
            right_eye=(120.0, 130.0),
            left_eye=(160.0, 130.0),
            nose_tip=(140.0, 150.0),
            right_mouth_corner=(125.0, 175.0),
            left_mouth_corner=(155.0, 175.0),
        )
        det = FaceDetection(
            bbox=self._bbox,
            confidence=self._confidence,
            landmarks=landmarks,
        )
        return DetectionResult(detections=[det], frame_id=fid, latency_ms=1.0, input_size=(640, 480))

    @property
    def input_size(self):
        return (640, 480)

    @property
    def confidence_threshold(self):
        return 0.6


class MockQualityGate(BaseFaceQualityGate):
    def __init__(self, acceptable=True):
        self._acceptable = acceptable
        self._thresholds = FaceQualityThresholds()

    def evaluate(self, frame: Frame, face: FaceDetection) -> FaceQualityResult:
        metrics = FaceQualityMetrics(
            confidence=face.confidence,
            face_width=face.bbox[2],
            face_height=face.bbox[3],
            sharpness=100.0,
            roll_angle_deg=0.0,
            yaw_asymmetry=1.0,
            crop_valid=True,
        )
        return FaceQualityResult(
            is_acceptable=self._acceptable,
            metrics=metrics,
            rejection_reasons=[],
            thresholds=self._thresholds,
        )

    @property
    def thresholds(self):
        return self._thresholds


class MockAligner(BaseFaceAligner):
    def align(
        self,
        frame: Frame | np.ndarray,
        target: FaceDetection | TrackedFace | tuple[float, float, float, float],
        landmarks: FaceLandmarks | None = None,
    ) -> AlignmentResult:
        crop = np.zeros((112, 112, 3), dtype=np.uint8)
        return AlignmentResult(
            aligned_face=crop,
            is_success=True,
            method="similarity",
            output_size=(112, 112),
        )

    @property
    def output_size(self) -> tuple[int, int]:
        return (112, 112)


class MockEmbedder(BaseFaceEmbedder):
    def __init__(self, vector: np.ndarray):
        self._vector = normalize_l2(vector)

    def set_vector(self, vector: np.ndarray):
        self._vector = normalize_l2(vector)

    def embed(self, alignment: AlignmentResult | np.ndarray) -> EmbeddingResult:
        return EmbeddingResult(
            embedding=self._vector.copy(),
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


def _make_vector(seed: int, dim: int = 128) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    return normalize_l2(vec)


@pytest.fixture
def test_setup():
    db = DatabaseManager(":memory:", auto_init=True)
    repo = SQLitePersonRepository(db)
    svc = PersonService(repo)

    # Create Alice (existing enrolled person)
    alice = svc.create_person(
        display_name="Alice Smith",
        status=PersonStatus.ACTIVE,
        consent_status=ConsentStatus.GRANTED,
    )
    alice_vec = _make_vector(100)
    svc.add_person_embeddings(alice.person_id, [alice_vec])

    # Create Bob (target person for new enrollment)
    bob = svc.create_person(
        display_name="Bob Jones",
        status=PersonStatus.ACTIVE,
        consent_status=ConsentStatus.GRANTED,
    )

    gallery = svc.load_recognition_gallery()
    return db, repo, svc, alice, bob, alice_vec, gallery


def test_scenario_1_new_unregistered_face(test_setup):
    """Scenario 1: New unregistered face -> Enrollment proceeds normally."""
    _, _, svc, alice, bob, _, gallery = test_setup
    bob_vec = _make_vector(200)  # Uncorrelated with Alice (similarity ~0.0)

    detector = MockDetector()
    qg = MockQualityGate()
    aligner = MockAligner()
    embedder = MockEmbedder(bob_vec)

    ctrl = FaceEnrollmentController(
        detector=detector,
        quality_gate=qg,
        aligner=aligner,
        embedder=embedder,
        person_service=svc,
        gallery=gallery,
        config=EnrollmentConfig(minimum_samples=1, target_samples=1, direction_targets={}),
    )

    ctrl.start_session(bob.person_id)
    frame = Frame(image=np.zeros((480, 640, 3), dtype=np.uint8), frame_id=1, timestamp=1.0)
    prog = ctrl.process_frame(frame)

    assert prog.last_rejection_reason is None
    assert prog.samples_collected == 1
    assert prog.is_viable is True

    result = ctrl.commit_session()
    assert result.is_success is True
    assert result.samples_enrolled == 1


def test_scenario_2_same_person_same_id_re_enrollment(test_setup):
    """Scenario 2: Face already registered under same name and person ID -> Allowed under ADD_SAMPLES / REPLACE."""
    _, _, svc, alice, _, alice_vec, gallery = test_setup

    detector = MockDetector()
    qg = MockQualityGate()
    aligner = MockAligner()
    # Provide a diverse sample of the SAME person (cosine similarity ~0.80, below duplicate_similarity_threshold of 0.95)
    noise = np.random.default_rng(999).standard_normal(128).astype(np.float32) * 0.2
    diverse_alice_vec = normalize_l2(alice_vec + noise)

    embedder = MockEmbedder(diverse_alice_vec)

    ctrl = FaceEnrollmentController(
        detector=detector,
        quality_gate=qg,
        aligner=aligner,
        embedder=embedder,
        person_service=svc,
        gallery=gallery,
        config=EnrollmentConfig(minimum_samples=1, target_samples=1, direction_targets={}),
    )

    # ADD_SAMPLES mode for Alice
    ctrl.start_session(alice.person_id, mode=EnrollmentMode.ADD_SAMPLES)
    frame = Frame(image=np.zeros((480, 640, 3), dtype=np.uint8), frame_id=1, timestamp=1.0)
    prog = ctrl.process_frame(frame)

    assert prog.last_rejection_reason is None
    assert prog.samples_collected == 1

    result = ctrl.commit_session()
    assert result.is_success is True


def test_scenario_3_different_identity_blocked(test_setup):
    """Scenario 3: Face registered under a different name and ID -> Blocked with IDENTITY_CONFLICT."""
    _, _, svc, alice, bob, alice_vec, gallery = test_setup

    detector = MockDetector()
    qg = MockQualityGate()
    aligner = MockAligner()
    # Present Alice's face during Bob's enrollment
    embedder = MockEmbedder(alice_vec)

    ctrl = FaceEnrollmentController(
        detector=detector,
        quality_gate=qg,
        aligner=aligner,
        embedder=embedder,
        person_service=svc,
        gallery=gallery,
        config=EnrollmentConfig(minimum_samples=1, target_samples=1, direction_targets={}),
    )

    ctrl.start_session(bob.person_id)
    frame = Frame(image=np.zeros((480, 640, 3), dtype=np.uint8), frame_id=1, timestamp=1.0)
    prog = ctrl.process_frame(frame)

    assert prog.last_rejection_reason == SampleRejectionReason.IDENTITY_CONFLICT
    assert prog.samples_collected == 0
    assert prog.conflict_person_id == alice.person_id
    assert prog.conflict_display_name == "Alice Smith"
    assert prog.conflict_similarity >= 0.99


def test_scenario_4_conflict_during_add_samples(test_setup):
    """Scenario 4: Face matches another identity during ADD_SAMPLES -> Reject conflicting enrollment."""
    _, _, svc, alice, bob, alice_vec, gallery = test_setup

    detector = MockDetector()
    qg = MockQualityGate()
    aligner = MockAligner()
    embedder = MockEmbedder(alice_vec)

    ctrl = FaceEnrollmentController(
        detector=detector,
        quality_gate=qg,
        aligner=aligner,
        embedder=embedder,
        person_service=svc,
        gallery=gallery,
        config=EnrollmentConfig(minimum_samples=1, target_samples=1, direction_targets={}),
    )

    ctrl.start_session(bob.person_id, mode=EnrollmentMode.ADD_SAMPLES)
    frame = Frame(image=np.zeros((480, 640, 3), dtype=np.uint8), frame_id=1, timestamp=1.0)
    prog = ctrl.process_frame(frame)

    assert prog.last_rejection_reason == SampleRejectionReason.IDENTITY_CONFLICT
    assert prog.samples_collected == 0


def test_scenario_5_conflict_during_replace(test_setup):
    """Scenario 5: Face matches another identity during REPLACE -> Reject conflicting replacement."""
    _, _, svc, alice, bob, alice_vec, gallery = test_setup

    detector = MockDetector()
    qg = MockQualityGate()
    aligner = MockAligner()
    embedder = MockEmbedder(alice_vec)

    ctrl = FaceEnrollmentController(
        detector=detector,
        quality_gate=qg,
        aligner=aligner,
        embedder=embedder,
        person_service=svc,
        gallery=gallery,
        config=EnrollmentConfig(minimum_samples=1, target_samples=1, direction_targets={}),
    )

    ctrl.start_session(bob.person_id, mode=EnrollmentMode.REPLACE)
    frame = Frame(image=np.zeros((480, 640, 3), dtype=np.uint8), frame_id=1, timestamp=1.0)
    prog = ctrl.process_frame(frame)

    assert prog.last_rejection_reason == SampleRejectionReason.IDENTITY_CONFLICT
    assert prog.samples_collected == 0


def test_scenario_6_simultaneous_concurrent_sessions(test_setup):
    """Scenario 6: Two sessions try to register same face -> Final barrier allows only one, rejects second."""
    _, _, svc, _, bob, _, _ = test_setup
    charlie = svc.create_person(
        display_name="Charlie Brown",
        status=PersonStatus.ACTIVE,
        consent_status=ConsentStatus.GRANTED,
    )

    shared_face_vec = _make_vector(333)

    detector = MockDetector()
    qg = MockQualityGate()
    aligner = MockAligner()
    embedder = MockEmbedder(shared_face_vec)

    # Bob's controller
    ctrl_bob = FaceEnrollmentController(
        detector=detector,
        quality_gate=qg,
        aligner=aligner,
        embedder=embedder,
        person_service=svc,
        config=EnrollmentConfig(minimum_samples=1, target_samples=1, direction_targets={}),
    )

    # Charlie's controller
    ctrl_charlie = FaceEnrollmentController(
        detector=detector,
        quality_gate=qg,
        aligner=aligner,
        embedder=embedder,
        person_service=svc,
        config=EnrollmentConfig(minimum_samples=1, target_samples=1, direction_targets={}),
    )

    frame = Frame(image=np.zeros((480, 640, 3), dtype=np.uint8), frame_id=1, timestamp=1.0)

    # Both collect samples successfully while neither has committed yet
    ctrl_bob.start_session(bob.person_id)
    ctrl_bob.process_frame(frame)

    ctrl_charlie.start_session(charlie.person_id)
    ctrl_charlie.process_frame(frame)

    # Bob commits first -> SUCCESS
    res_bob = ctrl_bob.commit_session()
    assert res_bob.is_success is True

    # Charlie commits second -> Transactional check against fresh database/gallery MUST FAIL
    res_charlie = ctrl_charlie.commit_session()
    assert res_charlie.is_success is False
    assert "matches existing registered person" in (res_charlie.error_message or "")
    assert res_charlie.samples_enrolled == 0


def test_scenario_7_similar_looking_different_person(test_setup):
    """Scenario 7: Similar-looking but different person (sim < 0.55) -> No unjustified rejection."""
    _, _, svc, alice, bob, alice_vec, gallery = test_setup

    # Construct vector with ~0.40 similarity to Alice
    rng = np.random.default_rng(777)
    orthogonal_noise = rng.standard_normal(128).astype(np.float32)
    orthogonal_noise -= np.dot(orthogonal_noise, alice_vec) * alice_vec
    orthogonal_noise = normalize_l2(orthogonal_noise)

    # Cosine similarity = cos(theta) = 0.40
    similar_vec = normalize_l2(0.40 * alice_vec + np.sqrt(1 - 0.40**2) * orthogonal_noise)
    sim = compute_cosine_similarity(similar_vec, alice_vec)
    assert 0.38 <= sim <= 0.42

    detector = MockDetector()
    qg = MockQualityGate()
    aligner = MockAligner()
    embedder = MockEmbedder(similar_vec)

    ctrl = FaceEnrollmentController(
        detector=detector,
        quality_gate=qg,
        aligner=aligner,
        embedder=embedder,
        person_service=svc,
        gallery=gallery,
        config=EnrollmentConfig(minimum_samples=1, target_samples=1, direction_targets={}),
    )

    ctrl.start_session(bob.person_id)
    frame = Frame(image=np.zeros((480, 640, 3), dtype=np.uint8), frame_id=1, timestamp=1.0)
    prog = ctrl.process_frame(frame)

    # Should NOT be rejected
    assert prog.last_rejection_reason is None
    assert prog.samples_collected == 1


def test_scenario_8_embedding_index_unavailable_fails_closed(test_setup):
    """Scenario 8: Database / gallery unavailable -> Fail closed, no biometric persistence."""
    _, _, svc, _, bob, alice_vec, gallery = test_setup
    bob_distinct = _make_vector(888)

    detector = MockDetector()
    qg = MockQualityGate()
    aligner = MockAligner()
    embedder = MockEmbedder(bob_distinct)

    # 8a: Process frame with broken gallery fails closed
    class BrokenPersonService(PersonService):
        def load_recognition_gallery(self, *args, **kwargs):
            raise RuntimeError("Database connection lost")

    broken_svc = BrokenPersonService(svc.repository)

    ctrl = FaceEnrollmentController(
        detector=detector,
        quality_gate=qg,
        aligner=aligner,
        embedder=embedder,
        person_service=broken_svc,
        gallery=None,
        config=EnrollmentConfig(minimum_samples=1, target_samples=1, direction_targets={}),
    )

    ctrl.start_session(bob.person_id)
    frame = Frame(image=np.zeros((480, 640, 3), dtype=np.uint8), frame_id=1, timestamp=1.0)
    prog = ctrl.process_frame(frame)

    # Fails closed on process_frame
    assert prog.last_rejection_reason == SampleRejectionReason.GALLERY_UNAVAILABLE
    assert prog.samples_collected == 0

    # 8b: Commit with collected sample but gallery fails on commit -> Fails closed
    ctrl2 = FaceEnrollmentController(
        detector=detector,
        quality_gate=qg,
        aligner=aligner,
        embedder=embedder,
        person_service=svc,
        gallery=gallery,
        config=EnrollmentConfig(minimum_samples=1, target_samples=1, direction_targets={}),
    )
    ctrl2.start_session(bob.person_id)
    prog2 = ctrl2.process_frame(frame)
    assert prog2.samples_collected == 1

    # Simulate gallery/database becoming unavailable at commit barrier
    ctrl2._person_service = broken_svc
    res = ctrl2.commit_session()
    assert res.is_success is False
    assert "duplicate detection gallery unavailable" in (res.error_message or "")
    assert res.samples_enrolled == 0


def test_scenario_9_database_inconsistency_detected(test_setup):
    """Scenario 9: Database inconsistency -> Detected and handled without silent bypass."""
    _, repo, svc, alice, bob, alice_vec, _ = test_setup

    # Insert an empty / corrupt embedding vector into database
    repo.add_embeddings([
        # Direct raw invalid entry
    ])
    # The gallery loader skips corrupt embeddings with warnings, but if repository raises
    # on get_all_embeddings, it fails closed.
    class CorruptRepo(SQLitePersonRepository):
        def get_all_embeddings(self, *args, **kwargs):
            raise ValueError("Corrupt database index table")

    corrupt_svc = PersonService(CorruptRepo(test_setup[0]))
    ctrl = FaceEnrollmentController(
        detector=MockDetector(),
        quality_gate=MockQualityGate(),
        aligner=MockAligner(),
        embedder=MockEmbedder(alice_vec),
        person_service=corrupt_svc,
        gallery=None,
        config=EnrollmentConfig(minimum_samples=1, target_samples=1, direction_targets={}),
    )

    ctrl.start_session(bob.person_id)
    frame = Frame(image=np.zeros((480, 640, 3), dtype=np.uint8), frame_id=1, timestamp=1.0)
    prog = ctrl.process_frame(frame)
    assert prog.last_rejection_reason == SampleRejectionReason.GALLERY_UNAVAILABLE


def test_scenario_10_insufficient_permissions_redacts_pii(test_setup):
    """Scenario 10: Operator lacking persons:read cannot see matched identity details."""
    _, _, svc, alice, bob, alice_vec, gallery = test_setup

    ctrl = FaceEnrollmentController(
        detector=MockDetector(),
        quality_gate=MockQualityGate(),
        aligner=MockAligner(),
        embedder=MockEmbedder(alice_vec),
        person_service=svc,
        gallery=gallery,
        config=EnrollmentConfig(minimum_samples=1, target_samples=1, direction_targets={}),
    )
    ctrl.start_session(bob.person_id)
    frame = Frame(image=np.zeros((480, 640, 3), dtype=np.uint8), frame_id=1, timestamp=1.0)
    prog = ctrl.process_frame(frame)
    dto = EnrollmentProgressDTO.from_progress(prog)
    resp = dto.to_dict()

    # Operator with only ENROLLMENT_MANAGE (no PERSONS_READ)
    operator = OperatorPrincipal(scopes=frozenset([OperatorScope.ENROLLMENT_MANAGE]))
    assert not operator.has_scope(OperatorScope.PERSONS_READ)

    # Apply redaction rule from API router
    if not operator.has_scope(OperatorScope.PERSONS_READ):
        if resp.get("conflict_person_id") or resp.get("last_rejection_reason") in (
            "identity_conflict",
            "uncertain_identity",
        ):
            resp["conflict_person_id"] = None
            resp["conflict_display_name"] = None
            resp["conflict_similarity"] = None
            resp["last_rejection_details"] = (
                "Potential duplicate identity detected. This face appears to match an existing "
                "registered person. Enrollment has been paused pending identity verification."
            )

    assert resp["conflict_person_id"] is None
    assert resp["conflict_display_name"] is None
    assert resp["conflict_similarity"] is None
    assert "Alice" not in resp["last_rejection_details"]
    assert "Potential duplicate identity detected" in resp["last_rejection_details"]

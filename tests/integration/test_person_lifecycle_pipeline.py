"""Integration test covering full Person Lifecycle: Create -> Enroll -> Recognize -> Delete."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from dh360_hi.backend.database import DatabaseManager
from dh360_hi.backend.models import ConsentStatus, PersonStatus
from dh360_hi.backend.person_service import PersonService
from dh360_hi.backend.repository import SQLitePersonRepository
from dh360_hi.vision.aligner import FaceAligner
from dh360_hi.vision.detector import YuNetFaceDetector
from dh360_hi.vision.embedder import SFaceEmbedder
from dh360_hi.vision.enrollment import (
    EnrollmentConfig,
    EnrollmentMode,
    FaceEnrollmentController,
)
from dh360_hi.vision.frame import Frame
from dh360_hi.vision.identity_state import IdentityState, IdentityStateManager
from dh360_hi.vision.quality_gate import FaceQualityGate
from dh360_hi.vision.recognition import FaceRecognizer, RecognitionOutcome

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
YUNET_PATH = PROJECT_ROOT / "models" / "weights" / "face_detection_yunet_2023mar.onnx"
SFACE_PATH = PROJECT_ROOT / "models" / "weights" / "face_recognition_sface_2021dec.onnx"
FIXTURE_PATH = PROJECT_ROOT / "tests" / "fixtures" / "sample_face.jpg"


def _create_jittered_frame(
    base_image: np.ndarray,
    frame_id: int,
    angle_deg: float = 0.0,
    scale: float = 1.0,
) -> Frame:
    """Helper creating a subtly perturbed camera frame from reference image."""
    h, w = base_image.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, scale)
    transformed = cv2.warpAffine(base_image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
    return Frame(
        image=transformed,
        frame_id=frame_id,
        timestamp=float(frame_id) * 0.1,
        source_info="simulated_camera",
    )


def test_full_person_lifecycle_pipeline(tmp_path: Path) -> None:
    """Validate full lifecycle: Create -> Enroll -> Recognize -> Add -> Replace -> Delete."""
    if not YUNET_PATH.exists() or not SFACE_PATH.exists() or not FIXTURE_PATH.exists():
        pytest.skip("Pretrained model weights or test fixtures missing")

    raw_fixture = cv2.imread(str(FIXTURE_PATH))
    assert raw_fixture is not None, "Fixture image must load"

    # 1. Setup SQLite Persistence & Vision Components
    db_path = tmp_path / "lifecycle_pipeline.db"
    db_manager = DatabaseManager(db_path)
    repository = SQLitePersonRepository(db_manager)
    person_service = PersonService(repository)

    detector = YuNetFaceDetector(YUNET_PATH, input_size=(512, 512), score_threshold=0.6)
    quality_gate = FaceQualityGate()
    aligner = FaceAligner()
    embedder = SFaceEmbedder(SFACE_PATH)
    recognizer = FaceRecognizer()
    identity_manager = IdentityStateManager()

    config = EnrollmentConfig(
        target_samples=5,
        minimum_samples=3,
        duplicate_similarity_threshold=0.999,
        min_sample_interval_seconds=0.0,
        timeout_seconds=30.0,
    )
    controller = FaceEnrollmentController(
        detector=detector,
        quality_gate=quality_gate,
        aligner=aligner,
        embedder=embedder,
        person_service=person_service,
        config=config,
    )

    # 2. Create Person
    person = person_service.create_person(
        "Dr. Sarah Connor",
        preferred_name="Sarah",
        status=PersonStatus.ACTIVE,
        consent_status=ConsentStatus.GRANTED,
        notes="Chief of Surgery",
    )
    assert person.embedding_count == 0

    # 3. Initial Enrollment (5 samples)
    controller.start_session(person.person_id, mode=EnrollmentMode.ADD_SAMPLES)
    perturbations = [
        (0.0, 1.0),
        (-1.0, 0.99),
        (1.0, 1.01),
        (-2.0, 0.98),
        (2.0, 1.02),
        (3.0, 0.99),
    ]
    for idx, (angle, scale) in enumerate(perturbations):
        frame = _create_jittered_frame(raw_fixture, frame_id=idx + 1, angle_deg=angle, scale=scale)
        prog = controller.process_frame(frame)
        if prog.is_complete:
            break

    assert controller.current_progress.is_viable is True  # type: ignore[union-attr]
    commit_res = controller.commit_session()
    assert commit_res.is_success is True
    initial_sample_count = commit_res.samples_enrolled
    assert initial_sample_count >= 3
    assert person_service.repository.count_embeddings(person.person_id) == initial_sample_count
    # The application service performs the single post-commit gallery synchronization.
    recognizer.replace_gallery(person_service.load_recognition_gallery())
    assert len(recognizer.gallery) == 1

    # 4. Recognition Loop: Confirm Identification
    query_frame = _create_jittered_frame(raw_fixture, frame_id=10, angle_deg=0.0, scale=1.0)
    det = detector.detect(query_frame)
    assert len(det.detections) == 1
    aligned = aligner.align(query_frame, det.detections[0])
    emb = embedder.embed(aligned)

    for f_id in range(10, 15):
        rec = recognizer.recognize(emb)
        assert rec.outcome == RecognitionOutcome.IDENTIFIED
        assert rec.top_match is not None
        assert rec.top_match.person_id == person.person_id

        track_state = identity_manager.update(track_id=1, rec_result=rec, frame_id=f_id)

    assert track_state.state == IdentityState.IDENTIFIED
    assert track_state.current_person_id == person.person_id

    # 5. Add More Face Samples (ADD_SAMPLES mode)
    controller.start_session(person.person_id, mode=EnrollmentMode.ADD_SAMPLES)
    extra_perturbations = [
        (-2.5, 0.97),
        (2.5, 1.03),
        (-1.5, 1.005),
        (1.5, 0.995),
    ]
    for idx, (angle, scale) in enumerate(extra_perturbations):
        frame = _create_jittered_frame(raw_fixture, frame_id=idx + 20, angle_deg=angle, scale=scale)
        controller.process_frame(frame)

    add_res = controller.commit_session()
    assert add_res.is_success is True
    assert add_res.samples_enrolled >= 3
    new_total = initial_sample_count + add_res.samples_enrolled
    assert person_service.repository.count_embeddings(person.person_id) == new_total
    recognizer.replace_gallery(person_service.load_recognition_gallery())
    assert len(recognizer.gallery.get_entry(person.person_id).embeddings) == new_total

    # 6. Staged Replacement (REPLACE mode)
    # A. Simulate a failed replacement first (only 1 sample, then cancel)
    controller.start_session(person.person_id, mode=EnrollmentMode.REPLACE)
    f_fail = _create_jittered_frame(raw_fixture, frame_id=30, angle_deg=0.0, scale=1.0)
    controller.process_frame(f_fail)
    controller.cancel_session()
    # Confirm embeddings still intact in DB and recognized
    assert person_service.repository.count_embeddings(person.person_id) == new_total

    # B. Successful replacement with 4 fresh samples
    controller.start_session(person.person_id, mode=EnrollmentMode.REPLACE)
    replace_perturbations = [
        (-0.5, 1.002),
        (0.5, 0.998),
        (-1.2, 1.015),
        (1.2, 0.985),
    ]
    for idx, (angle, scale) in enumerate(replace_perturbations):
        frame = _create_jittered_frame(raw_fixture, frame_id=idx + 40, angle_deg=angle, scale=scale)
        controller.process_frame(frame)

    replace_res = controller.commit_session()
    assert replace_res.is_success is True
    assert replace_res.mode == EnrollmentMode.REPLACE
    assert replace_res.samples_enrolled >= 3
    # Verify old embeddings wiped out, exactly replace_res.samples_enrolled remain
    persisted_count = person_service.repository.count_embeddings(person.person_id)
    assert persisted_count == replace_res.samples_enrolled
    recognizer.replace_gallery(person_service.load_recognition_gallery())
    gallery_entry = recognizer.gallery.get_entry(person.person_id)
    assert gallery_entry is not None
    assert len(gallery_entry.embeddings) == replace_res.samples_enrolled

    # 7. Delete Person
    del_success = person_service.delete_person(
        person.person_id,
        recognizer=recognizer,
        identity_manager=identity_manager,
    )
    assert del_success is True
    assert person_service.get_person(person.person_id) is None
    assert person_service.repository.count_embeddings(person.person_id) == 0
    assert len(recognizer.gallery) == 0

    # Verify track 1 invalidated
    t_after = identity_manager.get_track_state(1)
    assert t_after is not None
    assert t_after.state == IdentityState.UNREGISTERED
    assert t_after.current_person_id is None

    # Subsequent recognition on same frame returns NO_MATCH
    rec_after = recognizer.recognize(emb)
    assert rec_after.outcome == RecognitionOutcome.NO_MATCH

    db_manager.close()

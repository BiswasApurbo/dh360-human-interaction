"""Integration test: FaceEnrollmentController -> Real Vision Components -> SQLite -> Recognition."""

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
from dh360_hi.vision.embedder import SFaceEmbedder, normalize_l2
from dh360_hi.vision.enrollment import (
    EnrollmentConfig,
    EnrollmentState,
    FaceEnrollmentController,
)
from dh360_hi.vision.frame import Frame
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


def test_full_camera_enrollment_to_recognition_integration(tmp_path: Path) -> None:
    """Validate end-to-end flow from camera frames to enrollment, DB commit, and recognition."""
    if not YUNET_PATH.exists() or not SFACE_PATH.exists() or not FIXTURE_PATH.exists():
        pytest.skip("Model weights or test fixtures missing")

    db_path = tmp_path / "enrollment_integration.db"
    db_manager = DatabaseManager(db_path)
    repository = SQLitePersonRepository(db_manager)
    person_service = PersonService(repository)

    # Initialize vision components
    detector = YuNetFaceDetector(model_path=YUNET_PATH, input_size=(512, 512))
    quality_gate = FaceQualityGate()
    aligner = FaceAligner(output_size=(112, 112))
    embedder = SFaceEmbedder(model_path=SFACE_PATH)
    recognizer = FaceRecognizer()

    config = EnrollmentConfig(
        target_samples=5,
        minimum_samples=3,
        duplicate_similarity_threshold=0.999,  # High threshold for slight synthetic jitters
        min_sample_interval_seconds=0.0,
        timeout_seconds=30.0,
    )

    enrollment_controller = FaceEnrollmentController(
        detector=detector,
        quality_gate=quality_gate,
        aligner=aligner,
        embedder=embedder,
        person_service=person_service,
        config=config,
        recognizer=recognizer,
    )

    # 1. Create a consented person in PersonService
    person = person_service.create_person(
        display_name="Jane Doe",
        preferred_name="Jane",
        status=PersonStatus.ACTIVE,
        consent_status=ConsentStatus.GRANTED,
    )

    # 2. Start Enrollment Session
    start_progress = enrollment_controller.start_session(person.person_id)
    assert start_progress.state == EnrollmentState.COLLECTING
    assert enrollment_controller.is_active is True

    # 3. Stream camera frames with subtle head tilt/scaling variation
    base_img = cv2.imread(str(FIXTURE_PATH))
    assert base_img is not None

    variations = [
        (0.0, 1.0),
        (3.0, 0.98),
        (-3.0, 1.02),
        (5.0, 0.99),
        (-5.0, 1.01),
        (2.0, 1.0),
    ]

    for idx, (angle, scale) in enumerate(variations):
        frame = _create_jittered_frame(base_img, frame_id=idx + 1, angle_deg=angle, scale=scale)
        prog = enrollment_controller.process_frame(frame)
        if prog.is_complete:
            break

    assert enrollment_controller.current_progress.is_viable is True  # type: ignore[union-attr]

    # 4. Commit Enrollment Session
    commit_result = enrollment_controller.commit_session()
    assert commit_result.is_success is True
    assert commit_result.state == EnrollmentState.COMPLETED
    assert commit_result.samples_enrolled >= 3
    assert len(commit_result.persisted_embedding_ids) >= 3

    # 5. Verify Database State
    stored_embeddings = repository.get_embeddings_for_person(person.person_id)
    assert len(stored_embeddings) == commit_result.samples_enrolled
    # Verify all embeddings in batch share the same batch ID
    batch_ids = {e.enrollment_batch_id for e in stored_embeddings}
    assert len(batch_ids) == 1
    assert commit_result.enrollment_batch_id in batch_ids

    # 6. Verify Recognition Using Automatically Reloaded Gallery
    assert len(recognizer.gallery) == 1
    gallery_entry = recognizer.gallery.get_entry(person.person_id)
    assert gallery_entry is not None
    assert len(gallery_entry.embeddings) == commit_result.samples_enrolled

    # Query with the original fixture image
    query_frame = Frame(image=base_img, frame_id=999, timestamp=99.9)
    det_res = detector.detect(query_frame)
    align_res = aligner.align(query_frame, det_res.detections[0])
    emb_res = embedder.embed(align_res)

    rec_res = recognizer.recognize(emb_res)
    assert rec_res.outcome == RecognitionOutcome.IDENTIFIED
    assert rec_res.person_id == person.person_id
    assert rec_res.display_name == "Jane"
    assert rec_res.confidence_score > 0.90

    # Query with impostor/random vector
    rng = np.random.default_rng(seed=999)
    impostor_vec = normalize_l2(rng.standard_normal(128).astype(np.float32))
    rec_impostor = recognizer.recognize(impostor_vec)
    assert rec_impostor.outcome == RecognitionOutcome.UNREGISTERED

    db_manager.close()

"""Integration test for TwoRatePerceptionPipeline with real ONNX models and async worker."""

from __future__ import annotations

import time
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
from dh360_hi.vision.embedding_worker import EmbeddingWorkerConfig
from dh360_hi.vision.frame import Frame
from dh360_hi.vision.identity_state import IdentityState
from dh360_hi.vision.presence_debouncer import PresenceEvent
from dh360_hi.vision.recognition import FaceRecognizer
from dh360_hi.vision.two_rate_pipeline import TwoRatePerceptionPipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
YUNET_PATH = PROJECT_ROOT / "models" / "weights" / "face_detection_yunet_2023mar.onnx"
SFACE_PATH = PROJECT_ROOT / "models" / "weights" / "face_recognition_sface_2021dec.onnx"
FIXTURE_PATH = PROJECT_ROOT / "tests" / "fixtures" / "sample_face.jpg"


def _create_frame(image: np.ndarray, frame_id: int) -> Frame:
    """Helper to wrap image into Frame."""
    return Frame(
        image=image.copy(),
        frame_id=frame_id,
        timestamp=float(frame_id) * 0.033,
        source_info="test_camera",
    )


def test_two_rate_pipeline_end_to_end_recognition_and_target_selection(tmp_path: Path) -> None:
    """Validate two-rate pipeline with real detection, tracking, async embedding, and debouncing."""
    if not YUNET_PATH.exists() or not SFACE_PATH.exists() or not FIXTURE_PATH.exists():
        pytest.skip("Pretrained model weights or test fixtures missing")

    raw_fixture = cv2.imread(str(FIXTURE_PATH))
    assert raw_fixture is not None, "Fixture image must load"

    # 1. Setup Database, Person, and Gallery
    db_path = tmp_path / "two_rate.db"
    db_mgr = DatabaseManager(db_path)
    repo = SQLitePersonRepository(db_mgr)
    service = PersonService(repo)

    person = service.create_person(
        display_name="Dr. Eleanor Vance",
        preferred_name="Eleanor",
        status=PersonStatus.ACTIVE,
        consent_status=ConsentStatus.GRANTED,
    )

    detector = YuNetFaceDetector(model_path=YUNET_PATH)
    aligner = FaceAligner()
    embedder = SFaceEmbedder(model_path=SFACE_PATH)
    recognizer = FaceRecognizer()

    # Enroll fixture face into person
    det_res = detector.detect(raw_fixture)
    assert len(det_res.detections) >= 1
    face_det = det_res.detections[0]
    align_res = aligner.align(frame=raw_fixture, target=face_det)
    assert align_res.is_success
    embed_res = embedder.embed(align_res.aligned_face)
    service.add_person_embedding(person.person_id, embed_res.embedding)

    # Sync into recognizer gallery
    recognizer._gallery = service.load_recognition_gallery()
    assert len(recognizer._gallery) == 1

    # 2. Instantiate Two-Rate Pipeline
    pipeline = TwoRatePerceptionPipeline(
        detector=detector,
        embedder=embedder,
        aligner=aligner,
        recognizer=recognizer,
        worker_config=EmbeddingWorkerConfig(max_queue_size=4),
    )

    pipeline.start()

    entered_events = []
    final_result = None

    try:
        # Feed 15 consecutive frames of the fixture face
        for f_id in range(1, 16):
            frame = _create_frame(raw_fixture, f_id)
            res = pipeline.process_frame(frame)
            final_result = res

            for event in res.presence_transitions:
                if event.event == PresenceEvent.PERSON_ENTERED:
                    entered_events.append(event)

            # Tiny sleep to allow background worker thread to process between frames
            time.sleep(0.01)

        # Allow worker to finish final items
        time.sleep(0.1)
        # One more frame to collect any late worker results
        final_result = pipeline.process_frame(_create_frame(raw_fixture, 16))

    finally:
        pipeline.stop(timeout=2.0)

    # 3. Assert pipeline outputs
    assert final_result is not None
    assert len(final_result.tracks) >= 1

    # Presence debounce should have emitted PERSON_ENTERED once for track 1
    assert len(entered_events) >= 1
    assert entered_events[0].track_id == 1

    # Target Selector should have focused on Eleanor
    assert final_result.target_result.has_target is True
    assert final_result.target_result.primary_target is not None
    assert final_result.target_result.primary_target.display_name == "Eleanor"
    assert final_result.target_result.primary_target.person_id == person.person_id

    # Identity state should be IDENTIFIED
    primary_track_id = final_result.target_result.primary_track_id
    assert primary_track_id is not None
    id_state = final_result.identity_states.get(primary_track_id)
    assert id_state is not None
    assert id_state.state == IdentityState.IDENTIFIED
    assert id_state.current_person_id == person.person_id

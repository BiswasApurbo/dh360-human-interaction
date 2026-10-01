"""Integration tests for the full end-to-end vision and recognition pipeline."""

from pathlib import Path

import cv2
import pytest

from dh360_hi.vision.aligner import FaceAligner
from dh360_hi.vision.detector import YuNetFaceDetector
from dh360_hi.vision.embedder import SFaceEmbedder
from dh360_hi.vision.frame import Frame
from dh360_hi.vision.identity_state import IdentityState, IdentityStateManager
from dh360_hi.vision.quality_gate import FaceQualityGate
from dh360_hi.vision.recognition import (
    FaceRecognizer,
    GalleryEntry,
    InMemoryGallery,
    RecognitionOutcome,
)
from dh360_hi.vision.tracker import FaceTracker

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
YUNET_PATH = ROOT_DIR / "models" / "weights" / "face_detection_yunet_2023mar.onnx"
SFACE_PATH = ROOT_DIR / "models" / "weights" / "face_recognition_sface_2021dec.onnx"
FIXTURE_PATH = ROOT_DIR / "tests" / "fixtures" / "sample_face.jpg"


@pytest.mark.skipif(
    not (YUNET_PATH.exists() and SFACE_PATH.exists() and FIXTURE_PATH.exists()),
    reason="Required models or test fixture not found",
)
def test_end_to_end_recognition_and_identity_confirmation() -> None:
    """Verify end-to-end stream: Frame -> Detection -> Tracking -> Quality -> Match -> State."""
    img = cv2.imread(str(FIXTURE_PATH))
    assert img is not None

    detector = YuNetFaceDetector(model_path=YUNET_PATH)
    tracker = FaceTracker()
    quality_gate = FaceQualityGate()
    aligner = FaceAligner(output_size=(112, 112))
    embedder = SFaceEmbedder(model_path=SFACE_PATH)

    # 1. First extract enrolled embedding from reference fixture
    frame_0 = Frame(image=img, frame_id=0, timestamp=0.0, source_info="enroll")
    det_0 = detector.detect(frame_0)
    align_0 = aligner.align(frame_0, det_0.detections[0])
    emb_0 = embedder.embed(align_0)

    # Populate gallery
    gallery = InMemoryGallery(
        [
            GalleryEntry(
                person_id="patient_101",
                display_name="Jane Doe",
                embeddings=[emb_0.embedding],
                model_name="SFace",
                model_version="2021dec",
            )
        ]
    )
    recognizer = FaceRecognizer(gallery=gallery)
    id_manager = IdentityStateManager()

    # 2. Simulate multi-frame video stream (5 frames)
    for frame_id in range(1, 6):
        frame = Frame(
            image=img, frame_id=frame_id, timestamp=frame_id * 0.033, source_info="stream"
        )

        # Detection & Tracking
        det_result = detector.detect(frame)
        track_result = tracker.update(det_result, frame_id=frame_id)

        assert len(track_result.tracks) == 1
        track = track_result.tracks[0]

        # Quality Gate
        q_result = quality_gate.evaluate(frame, track)
        assert q_result.is_acceptable is True

        # Alignment & Embedding
        align_res = aligner.align(frame, track)
        assert align_res.is_success is True

        emb_res = embedder.embed(align_res)
        assert emb_res.is_normalized is True

        # Recognition
        rec_res = recognizer.recognize(emb_res)
        assert rec_res.outcome == RecognitionOutcome.IDENTIFIED
        assert rec_res.person_id == "patient_101"

        # Identity State Machine Update
        state = id_manager.update(track_id=track.track_id, rec_result=rec_res, frame_id=frame_id)

        if frame_id < 3:
            assert state.state in (IdentityState.NEW, IdentityState.COLLECTING)
        else:
            # After 3 consistent observations, state must be IDENTIFIED
            assert state.state == IdentityState.IDENTIFIED
            assert state.current_person_id == "patient_101"
            assert state.current_display_name == "Jane Doe"

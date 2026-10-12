"""Integration tests for FaceAligner and SFaceEmbedder in end-to-end vision stream."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from dh360_hi.vision.aligner import FaceAligner
from dh360_hi.vision.detector import YuNetFaceDetector
from dh360_hi.vision.embedder import SFaceEmbedder
from dh360_hi.vision.frame import Frame
from dh360_hi.vision.quality_gate import FaceQualityGate
from dh360_hi.vision.tracker import FaceTracker

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
YUNET_PATH = ROOT_DIR / "models" / "weights" / "face_detection_yunet_2023mar.onnx"
SFACE_FP32_PATH = ROOT_DIR / "models" / "weights" / "face_recognition_sface_2021dec.onnx"
SFACE_INT8_PATH = ROOT_DIR / "models" / "weights" / "face_recognition_sface_2021dec_int8.onnx"
FIXTURE_PATH = ROOT_DIR / "tests" / "fixtures" / "sample_face.jpg"


@pytest.mark.skipif(
    not (YUNET_PATH.exists() and SFACE_FP32_PATH.exists() and FIXTURE_PATH.exists()),
    reason="Required model weights or test fixture not found",
)
def test_full_pipeline_detection_tracking_quality_alignment_embedding() -> None:
    """Verify end-to-end flow: Detect -> Track -> Quality Gate -> Align -> Embed."""
    img = cv2.imread(str(FIXTURE_PATH))
    assert img is not None

    detector = YuNetFaceDetector(model_path=YUNET_PATH, input_size=(512, 512))
    tracker = FaceTracker()
    quality_gate = FaceQualityGate()
    aligner = FaceAligner(output_size=(112, 112))
    embedder = SFaceEmbedder(model_path=SFACE_FP32_PATH)

    frame = Frame(image=img, frame_id=1, timestamp=0.0, source_info="fixture")

    # 1. Detect
    det_result = detector.detect(frame)
    assert len(det_result.detections) == 1

    # 2. Track
    track_result = tracker.update(det_result, frame_id=frame.frame_id)
    assert len(track_result.tracks) == 1
    track = track_result.tracks[0]

    # 3. Quality Gate
    q_result = quality_gate.evaluate(frame, track)
    assert q_result.is_acceptable is True

    # 4. Align
    align_result = aligner.align(frame, track)
    assert align_result.is_success is True
    assert align_result.aligned_face.shape == (112, 112, 3)

    # 5. Embed
    emb_result = embedder.embed(align_result)
    assert emb_result.is_normalized is True
    assert emb_result.dimension == 128
    assert emb_result.embedding.shape == (128,)
    assert np.linalg.norm(emb_result.embedding) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.skipif(
    not (SFACE_FP32_PATH.exists() and SFACE_INT8_PATH.exists() and FIXTURE_PATH.exists()),
    reason="SFace models or test fixture not found",
)
def test_fp32_vs_int8_embedding_consistency() -> None:
    """Verify SFace FP32 and INT8 produce highly correlated embeddings on the same face."""
    img = cv2.imread(str(FIXTURE_PATH))
    assert img is not None

    aligner = FaceAligner(output_size=(112, 112))
    embedder_fp32 = SFaceEmbedder(model_path=SFACE_FP32_PATH, model_name="SFace-FP32")
    embedder_int8 = SFaceEmbedder(model_path=SFACE_INT8_PATH, model_name="SFace-INT8")

    align_result = aligner.align(img, (207.8, 182.5, 145.9, 206.9))
    assert align_result.is_success is True

    emb_fp32 = embedder_fp32.embed(align_result)
    emb_int8 = embedder_int8.embed(align_result)

    # Cosine similarity between FP32 and INT8 embeddings of the same face
    cosine_sim = float(np.dot(emb_fp32.embedding, emb_int8.embedding))
    assert cosine_sim > 0.95  # Quantization maintains high fidelity

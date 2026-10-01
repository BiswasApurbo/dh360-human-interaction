"""Unit tests for FaceEmbedder interface and SFace embedding adapter."""

from pathlib import Path

import numpy as np
import pytest

from dh360_hi.vision.aligner import AlignmentResult
from dh360_hi.vision.embedder import (
    EmbeddingResult,
    SFaceEmbedder,
    compute_cosine_distance,
    compute_cosine_similarity,
    normalize_l2,
)

SFACE_FP32_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "models"
    / "weights"
    / "face_recognition_sface_2021dec.onnx"
)
SFACE_INT8_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "models"
    / "weights"
    / "face_recognition_sface_2021dec_int8.onnx"
)


def test_normalize_l2_unit_vector() -> None:
    """Verify L2 normalization produces vectors with unit norm (||v||_2 == 1.0)."""
    vec = np.array([3.0, 4.0, 0.0], dtype=np.float32)
    normed = normalize_l2(vec)

    assert normed.shape == (3,)
    assert np.linalg.norm(normed) == pytest.approx(1.0, abs=1e-6)
    assert np.allclose(normed, [0.6, 0.8, 0.0])


def test_normalize_l2_zero_vector() -> None:
    """Verify L2 normalization handles zero vectors gracefully without division by zero."""
    zeros = np.zeros(128, dtype=np.float32)
    normed = normalize_l2(zeros)

    assert normed.shape == (128,)
    assert np.allclose(normed, 0.0)


def test_cosine_similarity_and_distance() -> None:
    """Verify cosine similarity and distance properties for parallel and orthogonal vectors."""
    v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v3 = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    assert compute_cosine_similarity(v1, v2) == pytest.approx(1.0, abs=1e-6)
    assert compute_cosine_distance(v1, v2) == pytest.approx(0.0, abs=1e-6)

    assert compute_cosine_similarity(v1, v3) == pytest.approx(0.0, abs=1e-6)
    assert compute_cosine_distance(v1, v3) == pytest.approx(1.0, abs=1e-6)


def test_sface_missing_model_raises_filenotfound() -> None:
    """Verify SFaceEmbedder raises FileNotFoundError for non-existent model path."""
    with pytest.raises(FileNotFoundError, match="SFace model file not found"):
        SFaceEmbedder(model_path="non_existent_sface_model.onnx")


@pytest.mark.skipif(not SFACE_FP32_PATH.exists(), reason="SFace FP32 weights not downloaded")
def test_sface_fp32_embed_valid_aligned_face() -> None:
    """Verify SFace FP32 extracts 128-D normalized embedding with model metadata."""
    embedder = SFaceEmbedder(
        model_path=SFACE_FP32_PATH, model_name="SFace-FP32", model_version="2021dec"
    )

    assert embedder.model_name == "SFace-FP32"
    assert embedder.model_version == "2021dec"
    assert embedder.dimension == 128
    assert embedder.input_size == (112, 112)

    # 112x112 synthetic face crop
    dummy_face = np.full((112, 112, 3), 128, dtype=np.uint8)
    res = embedder.embed(dummy_face)

    assert isinstance(res, EmbeddingResult)
    assert res.model_name == "SFace-FP32"
    assert res.model_version == "2021dec"
    assert res.dimension == 128
    assert res.embedding.shape == (128,)
    assert res.embedding.dtype == np.float32
    assert res.is_normalized is True
    assert np.linalg.norm(res.embedding) == pytest.approx(1.0, abs=1e-5)
    assert res.latency_ms > 0.0


@pytest.mark.skipif(not SFACE_INT8_PATH.exists(), reason="SFace INT8 weights not downloaded")
def test_sface_int8_embed_valid_aligned_face() -> None:
    """Verify SFace INT8 extracts 128-D normalized embedding from an aligned face."""
    embedder = SFaceEmbedder(model_path=SFACE_INT8_PATH, model_name="SFace-INT8")

    assert embedder.model_name == "SFace-INT8"
    assert embedder.model_version == "2021dec"
    assert embedder.dimension == 128

    dummy_face = np.full((112, 112, 3), 128, dtype=np.uint8)
    res = embedder.embed(dummy_face)

    assert isinstance(res, EmbeddingResult)
    assert res.model_name == "SFace-INT8"
    assert res.dimension == 128
    assert res.embedding.shape == (128,)
    assert res.is_normalized is True
    assert np.linalg.norm(res.embedding) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.skipif(not SFACE_FP32_PATH.exists(), reason="SFace FP32 weights not downloaded")
def test_sface_handles_alignment_result_input() -> None:
    """Verify SFaceEmbedder processes AlignmentResult objects seamlessly."""
    embedder = SFaceEmbedder(model_path=SFACE_FP32_PATH)

    # Successful alignment
    valid_aligned = np.full((112, 112, 3), 100, dtype=np.uint8)
    align_success = AlignmentResult(
        aligned_face=valid_aligned,
        is_success=True,
        method="similarity_5pt",
        output_size=(112, 112),
    )
    res_success = embedder.embed(align_success)
    assert res_success.is_normalized is True
    assert np.linalg.norm(res_success.embedding) == pytest.approx(1.0, abs=1e-5)

    # Failed alignment
    align_fail = AlignmentResult(
        aligned_face=np.zeros((112, 112, 3), dtype=np.uint8),
        is_success=False,
        method="failed",
        output_size=(112, 112),
    )
    res_fail = embedder.embed(align_fail)
    assert res_fail.is_normalized is False
    assert np.allclose(res_fail.embedding, 0.0)


@pytest.mark.skipif(not SFACE_FP32_PATH.exists(), reason="SFace FP32 weights not downloaded")
def test_sface_handles_invalid_and_empty_images() -> None:
    """Verify embedder handles empty or None images gracefully."""
    embedder = SFaceEmbedder(model_path=SFACE_FP32_PATH)

    # Empty array
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    res_empty = embedder.embed(empty)
    assert res_empty.is_normalized is False
    assert np.allclose(res_empty.embedding, 0.0)

    # None input
    res_none = embedder.embed(None)  # type: ignore[arg-type]
    assert res_none.is_normalized is False

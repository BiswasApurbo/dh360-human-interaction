"""Face embedding extraction module for lightweight edge models."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from dh360_hi.vision.aligner import AlignmentResult


@dataclass(frozen=True)
class EmbeddingResult:
    """Immutable representation of a face embedding extraction result with model metadata."""

    embedding: np.ndarray  # 1D float32 normalized vector
    model_name: str
    model_version: str
    dimension: int
    is_normalized: bool
    latency_ms: float = 0.0


def normalize_l2(vector: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize a 1D or 2D feature vector such that ||vector||_2 == 1.0.

    Args:
        vector: NumPy array of shape (D,) or (N, D).
        eps: Small epsilon to prevent division by zero.

    Returns:
        L2-normalized float32 NumPy array with the same shape.
    """
    flat = vector.flatten().astype(np.float32)
    norm = float(np.linalg.norm(flat))
    if norm > eps:
        return flat / norm
    return flat


def compute_cosine_similarity(e1: np.ndarray, e2: np.ndarray) -> float:
    """Compute cosine similarity between two normalized embedding vectors.

    Assumes vectors are L2-normalized.
    """
    v1 = normalize_l2(e1)
    v2 = normalize_l2(e2)
    return float(np.dot(v1, v2))


def compute_cosine_distance(e1: np.ndarray, e2: np.ndarray) -> float:
    """Compute cosine distance (1.0 - cosine_similarity) between two embedding vectors."""
    return float(1.0 - compute_cosine_similarity(e1, e2))


class BaseFaceEmbedder(ABC):
    """Abstract interface for face feature embedding models."""

    @abstractmethod
    def embed(
        self,
        aligned_face: np.ndarray | AlignmentResult,
    ) -> EmbeddingResult:
        """Extract an L2-normalized feature vector embedding from an aligned face.

        Args:
            aligned_face: (H, W, 3) BGR image array or AlignmentResult.

        Returns:
            EmbeddingResult containing 1D float32 normalized vector and metadata.
        """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of the embedding model."""

    @property
    @abstractmethod
    def model_version(self) -> str:
        """Version identifier of the embedding model."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality of the output embedding vector."""

    @property
    @abstractmethod
    def input_size(self) -> tuple[int, int]:
        """Expected canonical input image dimensions (width, height)."""


class SFaceEmbedder(BaseFaceEmbedder):
    """SFace face feature embedder backed by OpenCV Zoo and cv2.FaceRecognizerSF."""

    def __init__(
        self,
        model_path: str | Path,
        model_name: str = "SFace",
        model_version: str = "2021dec",
        backend_id: int = cv2.dnn.DNN_BACKEND_OPENCV,
        target_id: int = cv2.dnn.DNN_TARGET_CPU,
    ) -> None:
        self._path = Path(model_path).resolve()
        if not self._path.exists():
            msg = f"SFace model file not found at {self._path}"
            raise FileNotFoundError(msg)

        self._model_name = model_name
        self._model_version = model_version
        self._input_size = (112, 112)
        self._dimension = 128

        self._recognizer = cv2.FaceRecognizerSF.create(
            str(self._path),
            "",
            backend_id=backend_id,
            target_id=target_id,
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def input_size(self) -> tuple[int, int]:
        return self._input_size

    @property
    def model_path(self) -> Path:
        return self._path

    def embed(
        self,
        aligned_face: np.ndarray | AlignmentResult,
    ) -> EmbeddingResult:
        """Extract a 128-D L2-normalized feature embedding from an aligned face."""
        if isinstance(aligned_face, AlignmentResult):
            if not aligned_face.is_success:
                return EmbeddingResult(
                    embedding=np.zeros(self._dimension, dtype=np.float32),
                    model_name=self._model_name,
                    model_version=self._model_version,
                    dimension=self._dimension,
                    is_normalized=False,
                    latency_ms=0.0,
                )
            img = aligned_face.aligned_face
        else:
            img = aligned_face

        if img is None or not isinstance(img, np.ndarray) or img.size == 0:
            return EmbeddingResult(
                embedding=np.zeros(self._dimension, dtype=np.float32),
                model_name=self._model_name,
                model_version=self._model_version,
                dimension=self._dimension,
                is_normalized=False,
                latency_ms=0.0,
            )

        # Resize if dimensions differ from canonical (112, 112)
        if (img.shape[1], img.shape[0]) != self._input_size:
            img = cv2.resize(img, self._input_size, interpolation=cv2.INTER_LINEAR)

        # Ensure 3-channel BGR format
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        t_start = time.perf_counter()
        raw_feat = self._recognizer.feature(img)
        latency_ms = (time.perf_counter() - t_start) * 1000.0

        normalized = normalize_l2(raw_feat)

        return EmbeddingResult(
            embedding=normalized,
            model_name=self._model_name,
            model_version=self._model_version,
            dimension=self._dimension,
            is_normalized=True,
            latency_ms=latency_ms,
        )

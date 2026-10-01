"""Unit tests for HumanInteractionConfig."""

from __future__ import annotations

import dataclasses

import pytest

from dh360_hi.application.config import HumanInteractionConfig
from dh360_hi.edge.runtime import EdgeRuntimeConfig
from dh360_hi.edge.session_manager import SessionConfig
from dh360_hi.vision.embedding_worker import EmbeddingWorkerConfig
from dh360_hi.vision.enrollment import EnrollmentConfig
from dh360_hi.vision.identity_state import IdentityStateConfig
from dh360_hi.vision.presence_debouncer import PresenceDebounceConfig
from dh360_hi.vision.quality_gate import FaceQualityThresholds
from dh360_hi.vision.recognition import RecognitionConfig
from dh360_hi.vision.target_selector import TargetSelectionConfig


class TestHumanInteractionConfig:
    def test_default_configuration_instantiation(self) -> None:
        """Verify all default settings and sub-configurations."""
        cfg = HumanInteractionConfig()

        assert cfg.db_path == ":memory:"
        assert cfg.auto_init_db is True
        assert cfg.detector_model_path is None
        assert cfg.embedder_model_path is None
        assert cfg.detector_input_size == (320, 320)
        assert cfg.detector_score_threshold == 0.6
        assert cfg.detector_nms_threshold == 0.3
        assert cfg.detector_top_k == 5000
        assert cfg.tracker_iou_threshold == 0.3
        assert cfg.tracker_max_missed_frames == 15
        assert cfg.tracker_min_hits == 1
        assert cfg.tracker_max_center_distance == 100.0
        assert cfg.tracker_smooth_factor == 0.8
        assert cfg.aligner_output_size == (112, 112)
        assert cfg.aligner_fallback_to_bbox is True

        assert isinstance(cfg.quality, FaceQualityThresholds)
        assert isinstance(cfg.runtime, EdgeRuntimeConfig)
        assert isinstance(cfg.session, SessionConfig)
        assert isinstance(cfg.recognition, RecognitionConfig)
        assert isinstance(cfg.worker, EmbeddingWorkerConfig)
        assert isinstance(cfg.identity, IdentityStateConfig)
        assert isinstance(cfg.target_selector, TargetSelectionConfig)
        assert isinstance(cfg.presence, PresenceDebounceConfig)
        assert isinstance(cfg.enrollment, EnrollmentConfig)

        assert cfg.model_name == "SFace"
        assert cfg.model_version == "2021dec"
        assert cfg.require_active_for_gallery is True
        assert cfg.require_consent_for_gallery is True

    def test_custom_overrides(self) -> None:
        """Verify custom settings applied to configuration."""
        runtime_cfg = EdgeRuntimeConfig(target_fps=60.0)
        session_cfg = SessionConfig(require_confirmed_presence=False)
        quality_cfg = FaceQualityThresholds(min_confidence=0.85)

        cfg = HumanInteractionConfig(
            db_path="/tmp/test.db",
            auto_init_db=False,
            detector_model_path="models/yunet.onnx",
            embedder_model_path="models/sface.onnx",
            detector_input_size=(640, 640),
            detector_score_threshold=0.8,
            tracker_iou_threshold=0.5,
            aligner_output_size=(160, 160),
            quality=quality_cfg,
            runtime=runtime_cfg,
            session=session_cfg,
            model_name="CustomModel",
            model_version="v2",
            require_active_for_gallery=False,
            require_consent_for_gallery=False,
        )

        assert cfg.db_path == "/tmp/test.db"
        assert cfg.auto_init_db is False
        assert cfg.detector_model_path == "models/yunet.onnx"
        assert cfg.embedder_model_path == "models/sface.onnx"
        assert cfg.detector_input_size == (640, 640)
        assert cfg.detector_score_threshold == 0.8
        assert cfg.tracker_iou_threshold == 0.5
        assert cfg.aligner_output_size == (160, 160)
        assert cfg.quality.min_confidence == 0.85
        assert cfg.runtime.target_fps == 60.0
        assert cfg.session.require_confirmed_presence is False
        assert cfg.model_name == "CustomModel"
        assert cfg.model_version == "v2"
        assert cfg.require_active_for_gallery is False
        assert cfg.require_consent_for_gallery is False

    def test_immutability(self) -> None:
        """Verify configuration dataclass is frozen."""
        cfg = HumanInteractionConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.db_path = "new_path.db"  # type: ignore[misc]

    def test_empty_db_path_rejected(self) -> None:
        """Verify empty database path raises ValueError."""
        with pytest.raises(ValueError, match="db_path must not be empty"):
            HumanInteractionConfig(db_path="")

    def test_empty_model_name_rejected(self) -> None:
        """Verify empty model name raises ValueError."""
        with pytest.raises(ValueError, match="model_name must not be empty"):
            HumanInteractionConfig(model_name="  ")

    def test_empty_model_version_rejected(self) -> None:
        """Verify empty model version raises ValueError."""
        with pytest.raises(ValueError, match="model_version must not be empty"):
            HumanInteractionConfig(model_version="")

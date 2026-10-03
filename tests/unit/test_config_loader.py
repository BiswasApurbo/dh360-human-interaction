"""Unit tests for configuration loader, presets, TOML parsing, and validation."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from dh360_hi.application.config import (
    ConfigurationPreset,
    HumanInteractionConfig,
    get_preset,
)
from dh360_hi.application.loader import (
    deep_merge,
    load_config,
    load_toml_file,
    parse_env_vars,
    validate_config,
)
from dh360_hi.contracts.errors import ValidationAppError
from dh360_hi.vision.identity_state import IdentityStateConfig


def test_default_config_properties() -> None:
    """Verify standard default configuration parameters."""
    cfg = HumanInteractionConfig()
    assert str(cfg.db_path) == ":memory:"
    assert cfg.auto_init_db is True
    assert cfg.runtime.target_fps == 30.0
    assert cfg.quality.min_confidence == 0.80
    assert cfg.quality.min_face_width == 40.0
    assert cfg.recognition.similarity_threshold == 0.65
    assert cfg.worker.max_queue_size == 8
    assert cfg.identity.evidence_window_size == 7
    assert cfg.presence.enter_patience_frames == 3
    assert cfg.enrollment.target_samples == 7


def test_presets_resolution() -> None:
    """Verify built-in preset configurations."""
    dev = get_preset(ConfigurationPreset.DEVELOPMENT)
    assert str(dev.db_path) == "dh360_dev.db"
    assert dev.runtime.target_fps == 30.0

    test_cfg = get_preset("testing")
    assert str(test_cfg.db_path) == ":memory:"
    assert test_cfg.runtime.target_fps == 60.0
    assert test_cfg.presence.enter_patience_frames == 1

    prod = get_preset("production")
    assert str(prod.db_path) == "dh360_hi.db"
    assert prod.runtime.max_consecutive_frame_errors == 5

    with pytest.raises(ValueError, match="Unknown configuration preset"):
        get_preset("invalid_preset_name")


def test_to_dict_and_from_dict_symmetry() -> None:
    """Verify serialization and deserialization symmetry."""
    original = HumanInteractionConfig(
        db_path="test.db",
        detector_model_path="models/det.onnx",
        detector_score_threshold=0.85,
    )
    d = original.to_dict()
    assert d["db_path"] == "test.db"
    assert d["detector_score_threshold"] == 0.85
    assert isinstance(d["quality"], dict)

    reconstructed = HumanInteractionConfig.from_dict(d)
    assert str(reconstructed.db_path) == "test.db"
    assert reconstructed.detector_score_threshold == 0.85
    assert reconstructed.quality.min_confidence == original.quality.min_confidence


def test_deep_merge() -> None:
    """Verify recursive dictionary merging."""
    base = {
        "a": 1,
        "nested": {"x": 10, "y": 20},
        "list_val": [1, 2],
    }
    override = {
        "b": 2,
        "nested": {"y": 99, "z": 100},
    }
    merged = deep_merge(base, override)
    assert merged["a"] == 1
    assert merged["b"] == 2
    assert merged["nested"]["x"] == 10
    assert merged["nested"]["y"] == 99
    assert merged["nested"]["z"] == 100
    assert merged["list_val"] == [1, 2]


def test_load_toml_file_success(tmp_path: Path) -> None:
    """Verify parsing valid TOML configuration files."""
    toml_content = """
    db_path = "custom.db"
    auto_init_db = false
    detector_score_threshold = 0.75

    [runtime]
    target_fps = 45.0
    max_consecutive_frame_errors = 3

    [quality]
    min_confidence = 0.85
    min_sharpness = 55.0
    """
    toml_file = tmp_path / "config.toml"
    toml_file.write_text(toml_content)

    loaded = load_toml_file(toml_file)
    assert loaded["db_path"] == "custom.db"
    assert loaded["auto_init_db"] is False
    assert loaded["runtime"]["target_fps"] == 45.0
    assert loaded["quality"]["min_sharpness"] == 55.0


def test_load_toml_file_not_found() -> None:
    """Verify error raised on non-existent TOML file."""
    with pytest.raises(ValidationAppError) as exc_info:
        load_toml_file("non_existent_file_12345.toml")
    assert exc_info.value.code == "VALIDATION_ERROR"
    assert "not found" in exc_info.value.message


def test_load_toml_file_invalid_syntax(tmp_path: Path) -> None:
    """Verify error raised on malformed TOML syntax."""
    bad_toml = tmp_path / "bad.toml"
    bad_toml.write_text("db_path = [unclosed_bracket")

    with pytest.raises(ValidationAppError) as exc_info:
        load_toml_file(bad_toml)
    assert exc_info.value.code == "VALIDATION_ERROR"
    assert "Failed to parse TOML" in exc_info.value.message


def test_parse_env_vars() -> None:
    """Verify environment variable extraction and type parsing."""
    mock_env = {
        "DH360_HI_DB_PATH": "/tmp/env_test.db",
        "DH360_HI_AUTO_INIT_DB": "false",
        "DH360_HI_DETECTOR_SCORE_THRESHOLD": "0.88",
        "DH360_HI_RUNTIME_TARGET_FPS": "25.5",
        "DH360_HI_QUALITY__MIN_SHARPNESS": "60.0",
        "OTHER_ENV_VAR": "ignored",
    }
    with patch.dict(os.environ, mock_env, clear=True):
        parsed = parse_env_vars(prefix="DH360_HI_")
        assert parsed["db_path"] == "/tmp/env_test.db"
        assert parsed["auto_init_db"] is False
        assert parsed["detector_score_threshold"] == 0.88
        assert parsed["runtime"]["target_fps"] == 25.5
        assert parsed["quality"]["min_sharpness"] == 60.0
        assert "other_env_var" not in parsed


def test_load_config_precedence(tmp_path: Path) -> None:
    """Verify deterministic configuration precedence:

    defaults -> preset -> file -> env -> overrides.
    """
    # 1. TOML file specifies db_path and runtime.target_fps
    toml_content = """
    db_path = "from_file.db"

    [runtime]
    target_fps = 20.0
    """
    toml_file = tmp_path / "app.toml"
    toml_file.write_text(toml_content)

    # 2. Env var overrides target_fps
    mock_env = {
        "DH360_HI_RUNTIME_TARGET_FPS": "22.0",
        "DH360_HI_QUALITY__MIN_CONFIDENCE": "0.78",
    }

    # 3. Explicit override dictionary overrides db_path
    overrides = {
        "db_path": "from_overrides.db",
    }

    with patch.dict(os.environ, mock_env, clear=True):
        config = load_config(
            preset="development",
            config_file=toml_file,
            overrides=overrides,
        )

        # Overrides won for db_path
        assert str(config.db_path) == "from_overrides.db"
        # Env won for target_fps over file/preset
        assert config.runtime.target_fps == 22.0
        # Env applied quality.min_confidence
        assert config.quality.min_confidence == 0.78
        # Preset defaults preserved where untouched
        assert config.auto_init_db is True


def test_validate_config_success() -> None:
    """Verify valid configuration passes validation without error."""
    config = HumanInteractionConfig()
    validate_config(config)  # Should not raise


def test_validate_config_failures() -> None:
    """Verify invalid configuration parameters raise ValidationAppError."""
    # Invalid threshold
    with pytest.raises(ValidationAppError) as exc_info:
        bad_cfg = HumanInteractionConfig(
            detector_score_threshold=1.5,
        )
        validate_config(bad_cfg)
    assert "detector_score_threshold must be in [0.0, 1.0]" in str(exc_info.value)

    # Invalid aligner size
    with pytest.raises(ValidationAppError) as exc_info:
        bad_cfg = HumanInteractionConfig(
            aligner_output_size=(0, 112),
        )
        validate_config(bad_cfg)
    assert "aligner_output_size" in str(exc_info.value)

    # Invalid detector top_k
    with pytest.raises(ValidationAppError) as exc_info:
        bad_cfg = HumanInteractionConfig(
            detector_top_k=0,
        )
        validate_config(bad_cfg)
    assert "detector_top_k" in str(exc_info.value)

    # Invalid identity window vs observations
    with pytest.raises(ValidationAppError) as exc_info:
        bad_cfg = HumanInteractionConfig(
            identity=IdentityStateConfig(evidence_window_size=5, min_observations_to_decide=10),
        )
        validate_config(bad_cfg)
    assert "identity.min_observations_to_decide" in str(exc_info.value)

"""Configuration loader, environment manager, and validation engine.

Provides deterministic, precedence-aware configuration composition for the
Human Interaction subsystem supporting defaults, presets, TOML files,
environment variables, and explicit overrides.
"""

from __future__ import annotations

import logging
import os
import tomllib
from pathlib import Path
from typing import Any

from dh360_hi.application.config import (
    DEFAULT_DB_PATH,
    ConfigurationPreset,
    HumanInteractionConfig,
    get_preset,
)
from dh360_hi.contracts.errors import ValidationAppError

logger = logging.getLogger(__name__)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override dictionary into base dictionary."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_toml_file(path: str | Path) -> dict[str, Any]:
    """Load and parse a TOML configuration file into a dictionary.

    Args:
        path: Path to the .toml file.

    Returns:
        Dictionary parsed from TOML.

    Raises:
        ValidationAppError: If file not found or invalid TOML syntax.
    """
    file_path = Path(path)
    if not file_path.exists():
        msg = f"Configuration file not found: {file_path}"
        raise ValidationAppError(message=msg, field="config_file", details={"path": str(file_path)})

    try:
        with open(file_path, "rb") as f:
            return tomllib.load(f)
    except tomllib.TOMLDecodeError as exc:
        msg = f"Failed to parse TOML configuration from {file_path}: {exc}"
        raise ValidationAppError(
            message=msg, field="config_file", details={"path": str(file_path), "error": str(exc)}
        ) from exc
    except Exception as exc:
        msg = f"Error reading configuration file {file_path}: {exc}"
        raise ValidationAppError(
            message=msg, field="config_file", details={"path": str(file_path), "error": str(exc)}
        ) from exc


def _parse_env_val(val: str) -> Any:
    """Helper to convert string environment variable to appropriate Python type."""
    val_stripped = val.strip()
    # Booleans
    if val_stripped.lower() in ("true", "1", "yes", "on"):
        return True
    if val_stripped.lower() in ("false", "0", "no", "off"):
        return False
    # None
    if val_stripped.lower() in ("none", "null"):
        return None
    # Integers
    try:
        return int(val_stripped)
    except ValueError:
        pass
    # Floats
    try:
        return float(val_stripped)
    except ValueError:
        pass
    return val_stripped


def parse_env_vars(prefix: str = "DH360_HI_") -> dict[str, Any]:
    """Extract and parse configuration settings from environment variables.

    Supports:
        - Flat keys: DH360_HI_DB_PATH -> {"db_path": "..."}
        - Double-underscore nesting: DH360_HI_RUNTIME__TARGET_FPS -> {"runtime": {"target_fps": 30}}
        - Standard section shortcuts: DH360_HI_RUNTIME_TARGET_FPS -> {"runtime": {"target_fps": 30}}

    Args:
        prefix: Environment variable prefix.

    Returns:
        Dictionary of extracted configuration settings.
    """
    env_config: dict[str, Any] = {}

    known_sections = {
        "quality",
        "runtime",
        "session",
        "recognition",
        "worker",
        "identity",
        "target_selector",
        "presence",
        "enrollment",
        "api",
    }

    for key, val in os.environ.items():
        if not key.startswith(prefix):
            continue

        raw_key = key[len(prefix) :].lower()
        if not raw_key:
            continue

        parsed_val = (
            val
            if raw_key
            in {
                "api__operator_token",
                "api_operator_token",
                "api__operator_scopes",
                "api_operator_scopes",
            }
            else _parse_env_val(val)
        )

        # 1. Double underscore separator (e.g. RUNTIME__TARGET_FPS)
        if "__" in raw_key:
            parts = raw_key.split("__", 1)
            sec, sub = parts[0], parts[1]
            if sec not in env_config or not isinstance(env_config[sec], dict):
                env_config[sec] = {}
            env_config[sec][sub] = parsed_val
            continue

        # 2. Known section prefix match (e.g. RUNTIME_TARGET_FPS)
        matched_sec = False
        for sec in known_sections:
            if raw_key.startswith(f"{sec}_"):
                sub = raw_key[len(sec) + 1 :]
                if sec not in env_config or not isinstance(env_config[sec], dict):
                    env_config[sec] = {}
                env_config[sec][sub] = parsed_val
                matched_sec = True
                break

        if not matched_sec:
            # 3. Top-level key
            env_config[raw_key] = parsed_val

    return env_config


def validate_config(config: HumanInteractionConfig) -> None:
    """Validate cross-field boundaries, numeric ranges, and invariants of configuration.

    Args:
        config: HumanInteractionConfig to validate.

    Raises:
        ValidationAppError: If any configuration parameter violates domain invariants.
    """
    errors: list[str] = []

    # Persistence
    if not str(config.db_path).strip():
        errors.append("db_path must not be empty")

    # Detector
    w, h = config.detector_input_size
    if w <= 0 or h <= 0:
        errors.append(f"detector_input_size dimensions must be positive, got ({w}, {h})")
    if not (0.0 <= config.detector_score_threshold <= 1.0):
        errors.append(
            f"detector_score_threshold must be in [0.0, 1.0], got {config.detector_score_threshold}"
        )
    if not (0.0 <= config.detector_nms_threshold <= 1.0):
        errors.append(
            f"detector_nms_threshold must be in [0.0, 1.0], got {config.detector_nms_threshold}"
        )
    if config.detector_top_k <= 0:
        errors.append(f"detector_top_k must be > 0, got {config.detector_top_k}")

    # Tracker
    if not (0.0 <= config.tracker_iou_threshold <= 1.0):
        errors.append(
            f"tracker_iou_threshold must be in [0.0, 1.0], got {config.tracker_iou_threshold}"
        )
    if config.tracker_max_missed_frames < 1:
        errors.append(
            f"tracker_max_missed_frames must be >= 1, got {config.tracker_max_missed_frames}"
        )
    if config.tracker_min_hits < 1:
        errors.append(f"tracker_min_hits must be >= 1, got {config.tracker_min_hits}")
    if not (0.0 <= config.tracker_smooth_factor <= 1.0):
        errors.append(
            f"tracker_smooth_factor must be in [0.0, 1.0], got {config.tracker_smooth_factor}"
        )

    # Aligner
    aw, ah = config.aligner_output_size
    if aw <= 0 or ah <= 0:
        errors.append(f"aligner_output_size dimensions must be positive, got ({aw}, {ah})")

    # Runtime
    if config.runtime.target_fps <= 0:
        errors.append(f"runtime.target_fps must be positive, got {config.runtime.target_fps}")
    if config.runtime.max_consecutive_frame_errors < 1:
        mcfe = config.runtime.max_consecutive_frame_errors
        errors.append(f"runtime.max_consecutive_frame_errors must be >= 1, got {mcfe}")

    # Quality Gate
    if not (0.0 <= config.quality.min_confidence <= 1.0):
        errors.append(
            f"quality.min_confidence must be in [0.0, 1.0], got {config.quality.min_confidence}"
        )
    if config.quality.min_face_width <= 0 or config.quality.min_face_height <= 0:
        qw, qh = config.quality.min_face_width, config.quality.min_face_height
        errors.append(f"quality.min_face_width/height must be positive, got ({qw}, {qh})")
    if config.quality.min_sharpness < 0.0:
        errors.append(f"quality.min_sharpness must be >= 0.0, got {config.quality.min_sharpness}")
    if config.quality.max_roll_angle_deg < 0.0:
        errors.append(
            f"quality.max_roll_angle_deg must be >= 0.0, got {config.quality.max_roll_angle_deg}"
        )
    if config.quality.max_yaw_asymmetry < 0.0:
        errors.append(
            f"quality.max_yaw_asymmetry must be >= 0.0, got {config.quality.max_yaw_asymmetry}"
        )

    # Recognition
    if not (0.0 <= config.recognition.similarity_threshold <= 1.0):
        st = config.recognition.similarity_threshold
        errors.append(f"recognition.similarity_threshold must be in [0.0, 1.0], got {st}")
    if not (0.0 <= config.recognition.min_similarity_margin <= 1.0):
        sm = config.recognition.min_similarity_margin
        errors.append(f"recognition.min_similarity_margin must be in [0.0, 1.0], got {sm}")

    # Worker
    if config.worker.max_queue_size < 1:
        errors.append(f"worker.max_queue_size must be >= 1, got {config.worker.max_queue_size}")
    if config.worker.max_results_queue_size < 1:
        mrq = config.worker.max_results_queue_size
        errors.append(f"worker.max_results_queue_size must be >= 1, got {mrq}")

    # Identity
    if config.identity.evidence_window_size < 1:
        ews = config.identity.evidence_window_size
        errors.append(f"identity.evidence_window_size must be >= 1, got {ews}")
    if not (
        1 <= config.identity.min_observations_to_decide <= config.identity.evidence_window_size
    ):
        obs = config.identity.min_observations_to_decide
        win = config.identity.evidence_window_size
        errors.append(f"identity.min_observations_to_decide must be between 1 and {win}, got {obs}")
    if not (0.0 <= config.identity.identification_consistency_ratio <= 1.0):
        cr = config.identity.identification_consistency_ratio
        errors.append(f"identity.identification_consistency_ratio must be in [0.0, 1.0], got {cr}")
    if config.identity.hysteresis_patience_observations < 1:
        hpo = config.identity.hysteresis_patience_observations
        errors.append(f"identity.hysteresis_patience_observations must be >= 1, got {hpo}")

    # Presence
    if config.presence.enter_patience_frames < 1:
        epf = config.presence.enter_patience_frames
        errors.append(f"presence.enter_patience_frames must be >= 1, got {epf}")
    if config.presence.leave_patience_frames < 1:
        lpf = config.presence.leave_patience_frames
        errors.append(f"presence.leave_patience_frames must be >= 1, got {lpf}")

    # Enrollment
    if config.enrollment.minimum_samples < 1:
        errors.append(
            f"enrollment.minimum_samples must be >= 1, got {config.enrollment.minimum_samples}"
        )
    if config.enrollment.target_samples < config.enrollment.minimum_samples:
        ts = config.enrollment.target_samples
        ms = config.enrollment.minimum_samples
        errors.append(
            f"enrollment.target_samples ({ts}) cannot be less than minimum_samples ({ms})"
        )
    if not (0.0 <= config.enrollment.duplicate_similarity_threshold <= 1.0):
        dst = config.enrollment.duplicate_similarity_threshold
        errors.append(f"enrollment.duplicate_similarity_threshold must be in [0.0, 1.0], got {dst}")

    if errors:
        msg = f"Configuration validation failed with {len(errors)} error(s): " + "; ".join(errors)
        raise ValidationAppError(message=msg, field="config", details={"errors": errors})


def load_config(
    preset: ConfigurationPreset | str | None = None,
    config_file: str | Path | None = None,
    env_prefix: str = "DH360_HI_",
    overrides: dict[str, Any] | None = None,
    validate: bool = True,
) -> HumanInteractionConfig:
    """Load, merge, and validate configuration following deterministic precedence.

    Precedence Model:
        1. Built-in defaults: HumanInteractionConfig()
        2. Selected preset (if specified, e.g. 'development', 'testing', 'production')
        3. Configuration file (TOML, if specified or via DH360_HI_CONFIG_FILE)
        4. Environment variables (DH360_HI_*)
        5. Programmatic / CLI overrides dictionary

    Args:
        preset: Optional preset name or enum.
        config_file: Optional path to a TOML configuration file.
        env_prefix: Prefix for environment variable matching (default 'DH360_HI_').
        overrides: Optional explicit dictionary of overrides.
        validate: Whether to run validate_config() after composition (default True).

    Returns:
        Fully composed, validated HumanInteractionConfig instance.

    Raises:
        ValidationAppError: If configuration fails validation or parsing.
    """
    # 1. Start with built-in defaults
    merged_dict = HumanInteractionConfig().to_dict()

    # 2. Check environment for preset override if not explicitly passed
    active_preset = preset or os.environ.get(f"{env_prefix}PRESET")
    if active_preset:
        try:
            preset_config = get_preset(active_preset)
            merged_dict = deep_merge(merged_dict, preset_config.to_dict())
        except ValueError as exc:
            raise ValidationAppError(
                message=str(exc), field="preset", details={"preset": str(active_preset)}
            ) from exc

    # 3. Check environment for config file path if not explicitly passed
    active_config_file = (
        config_file
        or os.environ.get(f"{env_prefix}CONFIG_FILE")
        or os.environ.get(f"{env_prefix}CONFIG_PATH")
    )
    if active_config_file:
        file_dict = load_toml_file(active_config_file)
        _reject_operator_token_source(file_dict, "TOML configuration")
        merged_dict = deep_merge(merged_dict, file_dict)

    # 4. Environment variable overrides
    env_dict = parse_env_vars(prefix=env_prefix)
    # Remove meta-keys like config_file / preset from env_dict before merging into config
    env_dict.pop("preset", None)
    env_dict.pop("config_file", None)
    env_dict.pop("config_path", None)

    # Fallback support for generic DH360_OPERATOR_TOKEN from Next.js / local dev
    if "api" not in env_dict:
        env_dict["api"] = {}
    if "operator_token" not in env_dict["api"]:
        fallback_token = (
            os.environ.get("DH360_OPERATOR_TOKEN")
            or os.environ.get("DH360_HI_OPERATOR_TOKEN")
        )
        if fallback_token:
            env_dict["api"]["operator_token"] = fallback_token

    # Default operator_scopes to all capabilities if token is set but scopes omitted
    active_token = env_dict["api"].get("operator_token")
    if active_token:
        has_scopes = "operator_scopes" in env_dict["api"] or (
            isinstance(merged_dict.get("api"), dict)
            and merged_dict["api"].get("operator_scopes")
        )
        if not has_scopes:
            from dh360_hi.api.security import OperatorScope
            env_dict["api"]["operator_scopes"] = tuple(s.value for s in OperatorScope)

    if env_dict:
        merged_dict = deep_merge(merged_dict, env_dict)

    # 5. Programmatic / CLI explicit overrides
    if overrides:
        _reject_operator_token_source(overrides, "programmatic overrides")
        merged_dict = deep_merge(merged_dict, overrides)

    # Construct final config
    try:
        config = HumanInteractionConfig.from_dict(merged_dict)
    except Exception as exc:
        msg = f"Failed to construct configuration from merged parameters: {exc}"
        raise ValidationAppError(message=msg, field="config", details={"error": str(exc)}) from exc

    # Validate
    if validate:
        validate_config(config)

    return config


def _reject_operator_token_source(config_data: dict[str, Any], source: str) -> None:
    """Require operator credentials to come from the environment, not serialized config."""
    api_data = config_data.get("api")
    if isinstance(api_data, dict) and "operator_token" in api_data:
        raise ValidationAppError(
            message=f"api.operator_token must be configured through the environment, not {source}",
            field="api.operator_token",
        )


def resolve_cli_db_path(
    args: Any | None = None,
    default_db_path: str = DEFAULT_DB_PATH,
) -> str | None:
    """Resolve database path for CLI execution with strict precedence:

    1. Explicit CLI --db-path (if provided)
    2. DH360_HI_DB_PATH environment variable (defer to environment loader)
    3. Explicit CLI --preset or DH360_HI_PRESET (defer to preset loader)
    4. Explicit CLI --config-file or DH360_HI_CONFIG_FILE (defer to config file loader)
    5. Fallback to default_db_path ("dh360_hi.db")
    """
    if args is not None and getattr(args, "db_path", None):
        return str(args.db_path)
    if os.environ.get("DH360_HI_DB_PATH"):
        return None
    if (args is not None and getattr(args, "preset", None)) or os.environ.get("DH360_HI_PRESET"):
        return None
    if (
        (args is not None and getattr(args, "config_file", None))
        or os.environ.get("DH360_HI_CONFIG_FILE")
        or os.environ.get("DH360_HI_CONFIG_PATH")
    ):
        return None
    return default_db_path

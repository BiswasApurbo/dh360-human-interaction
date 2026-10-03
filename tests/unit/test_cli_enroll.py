"""Unit tests for 'dh360-hi enroll' command."""

from __future__ import annotations

from pathlib import Path

import pytest

from dh360_hi.application.config import HumanInteractionConfig
from dh360_hi.application.factory import create_interaction_service
from dh360_hi.cli.main import main
from dh360_hi.contracts.dto import CreatePersonCommand

YUNET_PATH = "models/weights/face_detection_yunet_2023mar.onnx"
SFACE_PATH = "models/weights/face_recognition_sface_2021dec.onnx"


def _make_enroll_config_file(tmp_path: Path, db_file: str) -> str:
    cfg = tmp_path / "enroll_cfg.toml"
    cfg.write_text(f"""
    db_path = "{db_file}"
    auto_init_db = true
    detector_model_path = "{YUNET_PATH}"
    embedder_model_path = "{SFACE_PATH}"
    """)
    return str(cfg)


def test_cli_enroll_missing_detector_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify error on attempting to enroll when detector_model_path is not configured."""
    db_file = str(tmp_path / "enroll_no_model.db")
    target_id = "00000000-0000-0000-0000-000000000099"
    exit_code = main(["enroll", target_id, "--db-path", db_file, "--source", "synthetic"])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "detector_model_path" in captured.err


def test_cli_enroll_person_not_found(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify error on attempting to enroll non-existent person."""
    db_file = str(tmp_path / "enroll_missing.db")
    cfg_file = _make_enroll_config_file(tmp_path, db_file)
    target_id = "00000000-0000-0000-0000-000000000099"
    exit_code = main(["enroll", target_id, "--config-file", cfg_file, "--source", "synthetic"])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "does not exist" in captured.err


def test_cli_enroll_inactive_person(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify error on attempting to enroll inactive person."""
    db_file = str(tmp_path / "enroll_inactive.db")
    cfg_file = _make_enroll_config_file(tmp_path, db_file)
    config = HumanInteractionConfig(db_path=db_file)
    service = create_interaction_service(config=config, enable_perception=False)
    created = service.create_person(
        CreatePersonCommand(
            display_name="Inactive User", status="inactive", consent_status="granted"
        )
    )
    service.stop(timeout=1.0)

    exit_code = main(
        ["enroll", created.person_id, "--config-file", cfg_file, "--source", "synthetic"]
    )
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "not active" in captured.err


def test_cli_enroll_unconsented_person(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify error on attempting to enroll person without granted biometric consent."""
    db_file = str(tmp_path / "enroll_unconsented.db")
    cfg_file = _make_enroll_config_file(tmp_path, db_file)
    config = HumanInteractionConfig(db_path=db_file)
    service = create_interaction_service(config=config, enable_perception=False)
    created = service.create_person(
        CreatePersonCommand(
            display_name="No Consent User", status="active", consent_status="withdrawn"
        )
    )
    service.stop(timeout=1.0)

    exit_code = main(
        ["enroll", created.person_id, "--config-file", cfg_file, "--source", "synthetic"]
    )
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Biometric consent not granted" in captured.err

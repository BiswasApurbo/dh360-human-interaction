"""Unit tests for 'dh360-hi run' command."""

from __future__ import annotations

from pathlib import Path

import pytest

from dh360_hi.cli.main import main

YUNET_PATH = "models/weights/face_detection_yunet_2023mar.onnx"
SFACE_PATH = "models/weights/face_recognition_sface_2021dec.onnx"


def _make_run_config_file(tmp_path: Path, db_file: str) -> str:
    cfg = tmp_path / "run_cfg.toml"
    cfg.write_text(f"""
    db_path = "{db_file}"
    auto_init_db = true
    detector_model_path = "{YUNET_PATH}"
    embedder_model_path = "{SFACE_PATH}"
    """)
    return str(cfg)


def test_cli_run_missing_detector_path(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify error when detector_model_path is not configured."""
    db_file = str(tmp_path / "run_no_model.db")
    exit_code = main(["run", "--source", "synthetic", "--db-path", db_file])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "detector_model_path" in captured.err


def test_cli_run_duration_limit(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify 'dh360-hi run' starts and exits cleanly after duration limit."""
    db_file = str(tmp_path / "run_test.db")
    cfg_file = _make_run_config_file(tmp_path, db_file)
    exit_code = main(
        [
            "run",
            "--config-file",
            cfg_file,
            "--source",
            "synthetic",
            "--duration",
            "0.2",
        ]
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Digital Health 360 Human Interaction Service" in captured.out
    assert "Starting runtime frame pump" in captured.out
    assert "Final Runtime Execution Metrics" in captured.out
    assert "Service stopped successfully." in captured.out


def test_cli_run_json_events(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify 'dh360-hi run --json-events' flag is accepted."""
    db_file = str(tmp_path / "run_json.db")
    cfg_file = _make_run_config_file(tmp_path, db_file)
    exit_code = main(
        [
            "run",
            "--config-file",
            cfg_file,
            "--source",
            "synthetic",
            "--duration",
            "0.1",
            "--json-events",
        ]
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Starting runtime frame pump" in captured.out


def test_cli_run_missing_video_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify error when selecting video source without providing --video-file."""
    db_file = str(tmp_path / "run_video_missing.db")
    cfg_file = _make_run_config_file(tmp_path, db_file)
    exit_code = main(["run", "--config-file", cfg_file, "--source", "video"])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "--video-file is required" in captured.err

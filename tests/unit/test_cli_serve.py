"""Unit tests for 'dh360-hi serve' CLI command."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from dh360_hi.cli.main import main

YUNET_PATH = "models/weights/face_detection_yunet_2023mar.onnx"
SFACE_PATH = "models/weights/face_recognition_sface_2021dec.onnx"


def _make_serve_config_file(tmp_path: Path, db_file: str) -> str:
    cfg = tmp_path / "serve_cfg.toml"
    cfg.write_text(f"""
    db_path = "{db_file}"
    auto_init_db = true
    detector_model_path = "{YUNET_PATH}"
    embedder_model_path = "{SFACE_PATH}"
    """)
    return str(cfg)


def test_cli_serve_missing_detector_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify error when detector_model_path is not configured."""
    db_file = str(tmp_path / "serve_no_model.db")
    exit_code = main(["serve", "--source", "synthetic", "--db-path", db_file])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "detector_model_path" in captured.err


def test_cli_serve_wires_single_process_daemon(tmp_path: Path) -> None:
    """Verify serve wires perception, AuthorizationController, app, and uvicorn."""
    db_file = str(tmp_path / "serve_test.db")
    cfg_file = _make_serve_config_file(tmp_path, db_file)

    with (
        mock.patch("uvicorn.run") as mock_uvicorn_run,
        mock.patch("dh360_hi.cli.serve.create_interaction_service") as mock_create_svc,
        mock.patch("dh360_hi.cli.serve.create_app") as mock_create_app,
    ):
        mock_svc = mock.MagicMock()
        mock_runtime = mock.MagicMock()
        mock_sm = mock.MagicMock()
        mock_runtime.session_manager = mock_sm
        mock_svc.runtime = mock_runtime
        mock_create_svc.return_value = mock_svc

        mock_app = mock.MagicMock()
        mock_create_app.return_value = mock_app

        exit_code = main(
            [
                "serve",
                "--config-file",
                cfg_file,
                "--source",
                "synthetic",
                "--host",
                "0.0.0.0",
                "--port",
                "8080",
            ]
        )

        assert exit_code == 0
        mock_create_svc.assert_called_once()
        assert mock_create_svc.call_args[1]["enable_perception"] is True
        mock_svc.start.assert_called_once_with(async_thread=True)

        mock_create_app.assert_called_once()
        call_kwargs = mock_create_app.call_args[1]
        assert call_kwargs["service"] is mock_svc
        # An AuthorizationController must have been created and passed
        auth_ctrl = call_kwargs["authorization_controller"]
        assert auth_ctrl._session_manager is mock_sm

        mock_uvicorn_run.assert_called_once_with(
            mock_app, host="0.0.0.0", port=8080, log_level="warning"
        )
        mock_svc.stop.assert_called_once_with(timeout=3.0)

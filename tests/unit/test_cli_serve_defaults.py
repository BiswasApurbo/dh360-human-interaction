from __future__ import annotations

from pathlib import Path
from unittest import mock

from dh360_hi.cli.main import main


def test_cli_serve_defaults_to_project_models(tmp_path: Path) -> None:
    """Verify serve automatically uses project model weights when available."""
    db_file = str(tmp_path / "serve_defaults.db")

    with (
        mock.patch("uvicorn.run") as _mock_uvicorn_run,
        mock.patch("dh360_hi.cli.serve.create_interaction_service") as mock_create_svc,
        mock.patch("dh360_hi.cli.serve.create_app") as mock_create_app,
    ):
        mock_svc = mock.MagicMock()
        mock_runtime = mock.MagicMock()
        mock_sm = mock.MagicMock()
        mock_runtime.session_manager = mock_sm
        mock_svc.runtime = mock_runtime
        mock_create_svc.return_value = mock_svc
        mock_create_app.return_value = mock.MagicMock()

        # No --config-file or explicit model arguments provided
        exit_code = main(
            [
                "serve",
                "--source",
                "synthetic",
                "--db-path",
                db_file,
            ]
        )

        assert exit_code == 0
        mock_create_svc.assert_called_once()
        passed_config = mock_create_svc.call_args[1]["config"]
        det_path = "models/weights/face_detection_yunet_2023mar.onnx"
        emb_path = "models/weights/face_recognition_sface_2021dec.onnx"
        assert passed_config.detector_model_path == det_path
        assert passed_config.embedder_model_path == emb_path

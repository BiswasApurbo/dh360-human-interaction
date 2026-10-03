"""End-to-end integration tests for the dh360-hi CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dh360_hi.cli.main import main


def test_cli_end_to_end_operational_workflow(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Execute complete operator workflow across config, person, run, session, and health."""
    db_file = str(tmp_path / "e2e_clinic.db")
    config_file = tmp_path / "clinic_config.toml"
    config_file.write_text(f"""
    db_path = "{db_file}"
    auto_init_db = true
    detector_model_path = "models/weights/face_detection_yunet_2023mar.onnx"
    embedder_model_path = "models/weights/face_recognition_sface_2021dec.onnx"

    [runtime]
    target_fps = 40.0
    """)

    # 1. Validate configuration
    assert main(["config", "validate", "--config-file", str(config_file)]) == 0
    capsys.readouterr()  # Drain validate output

    # 2. Inspect health
    assert main(["health", "--config-file", str(config_file), "--json"]) == 0
    health_data = json.loads(capsys.readouterr().out)
    assert health_data["status"] in ("healthy", "degraded")

    # 3. Create a person profile
    assert (
        main(
            [
                "person",
                "create",
                "--config-file",
                str(config_file),
                "--display-name",
                "Nurse Joy",
                "--preferred-name",
                "Joy",
                "--status",
                "active",
                "--consent-status",
                "granted",
                "--json",
            ]
        )
        == 0
    )
    created_person = json.loads(capsys.readouterr().out)
    person_id = created_person["person_id"]

    # 4. Verify person via list
    assert main(["person", "list", "--config-file", str(config_file), "--json"]) == 0
    person_list = json.loads(capsys.readouterr().out)
    assert len(person_list) == 1
    assert person_list[0]["person_id"] == person_id

    # 5. Run runtime for a brief duration with synthetic source
    assert (
        main(
            [
                "run",
                "--config-file",
                str(config_file),
                "--source",
                "synthetic",
                "--duration",
                "0.2",
            ]
        )
        == 0
    )
    capsys.readouterr()  # Drain output

    # 6. Check session list (synthetic source without real faces creates 0 sessions)
    assert main(["session", "list", "--config-file", str(config_file), "--json"]) == 0
    sessions = json.loads(capsys.readouterr().out)
    assert isinstance(sessions, list)

    # 7. Check aggregate analytics
    assert main(["session", "analytics", "--config-file", str(config_file), "--json"]) == 0
    analytics = json.loads(capsys.readouterr().out)
    assert "total_sessions" in analytics

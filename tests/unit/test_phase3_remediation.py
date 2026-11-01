"""Tests for Phase 3 security remediation.

Verifies:
1. /health sanitization for public/unauthenticated requests (no db_path or private internal details).
2. /health privileged response for authenticated operators with INTERACTION_READ or SETTINGS_READ.
3. Unauthenticated rejection (401) of /runtime/state and /runtime/metrics.
4. Timestamp integrity: frames and persisted sessions use true Unix wall-clock epochs (year >= 2026).
5. Graceful fallback to synthetic source when webcam device cannot be opened.
"""

from __future__ import annotations

import os
import secrets
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from dh360_hi.api.app import create_app
from dh360_hi.application.service import HumanInteractionService
from dh360_hi.contracts.dto import InteractionSessionDTO, epoch_to_iso
from dh360_hi.backend.models import SessionRecord
from dh360_hi.vision.frame import Frame, SyntheticFrameSource, WebcamFrameSource


def _configure_operator(monkeypatch: pytest.MonkeyPatch, scopes: str = "interaction:read") -> str:
    token = f"test-token-{secrets.token_hex(16)}"
    monkeypatch.setenv("DH360_HI_API__OPERATOR_TOKEN", token)
    monkeypatch.setenv("DH360_HI_API__OPERATOR_SCOPES", scopes)
    monkeypatch.delenv("DH360_HI_CONFIG_FILE", raising=False)
    monkeypatch.delenv("DH360_HI_CONFIG_PATH", raising=False)
    return token


class TestPublicApiExposureRemediation:
    """Verify endpoint protection and health response sanitization."""

    def test_unauthenticated_health_is_sanitized_without_internal_paths(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Unauthenticated /health returns overall status without leaking internal host paths."""
        _configure_operator(monkeypatch, "interaction:read")
        app = create_app()

        with TestClient(app) as client:
            resp = client.get("/health")
            assert resp.status_code == 200
            data = resp.json()

            assert data["status"] in {"healthy", "degraded", "unhealthy"}
            assert "timestamp_iso" in data
            assert data["metrics"] is None
            assert "subsystems" in data

            # Verify no sensitive filesystem path or internal metrics leaked
            subsystems = data["subsystems"]
            for sub_name, sub in subsystems.items():
                assert "name" in sub
                assert "status" in sub
                assert "message" in sub
                # Details must be empty dict for unauthenticated callers
                assert sub.get("details") == {}, f"Subsystem {sub_name} leaked details to unauthenticated caller"

    def test_authenticated_health_returns_full_diagnostic_details(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Authenticated operator with interaction:read receives full subsystem details."""
        token = _configure_operator(monkeypatch, "interaction:read")
        app = create_app()

        with TestClient(app) as client:
            resp = client.get("/health", headers={"Authorization": f"Bearer {token}"})
            assert resp.status_code == 200
            data = resp.json()

            # Privileged caller gets database subsystem details including db_path
            db_sub = data["subsystems"].get("database")
            assert db_sub is not None
            assert "db_path" in db_sub.get("details", {})

    def test_unauthenticated_runtime_state_and_metrics_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Direct access to /runtime/state and /runtime/metrics requires operator auth."""
        token = _configure_operator(monkeypatch, "interaction:read")
        app = create_app()

        with TestClient(app) as client:
            # Unauthenticated -> 401
            assert client.get("/runtime/state").status_code == 401
            assert client.get("/runtime/metrics").status_code == 401

            # Invalid Bearer token -> 401
            bad_headers = {"Authorization": "Bearer invalid-token"}
            assert client.get("/runtime/state", headers=bad_headers).status_code == 401
            assert client.get("/runtime/metrics", headers=bad_headers).status_code == 401

            # Authenticated -> 200 / 503 (state returns 200)
            good_headers = {"Authorization": f"Bearer {token}"}
            state_resp = client.get("/runtime/state", headers=good_headers)
            assert state_resp.status_code == 200
            assert state_resp.json() in {"stopped", "running", "paused", "error"}


class TestTimestampIntegrity:
    """Verify timestamps represent Unix wall-clock time rather than monotonic offsets."""

    def test_synthetic_frame_source_uses_wall_clock_time(self) -> None:
        """SyntheticFrameSource produces frames with current epoch timestamp."""
        source = SyntheticFrameSource(fps=30.0, num_frames=5)
        source.open()
        frame = source.read()
        source.release()

        assert frame is not None
        # Must be modern Unix timestamp (>= 2026 era: ~1.7e9 seconds)
        assert frame.timestamp > 1_700_000_000.0
        assert abs(frame.timestamp - time.time()) < 5.0

    def test_session_dto_from_record_with_valid_timestamp(self) -> None:
        """SessionRecord with modern timestamp converts to accurate calendar year ISO."""
        now = time.time()
        record = SessionRecord(
            session_id="11111111-1111-4111-8111-111111111111",
            person_id=None,
            display_name=None,
            started_at=now,
            ended_at=now + 15.0,
            duration_seconds=15.0,
            final_state="completed",
            identity_was_known=False,
            identity_transitions_count=0,
        )

        dto = InteractionSessionDTO.from_record(record)
        assert dto.started_at_iso is not None
        # Check that year represents the current calendar era (2026)
        assert "2026" in dto.started_at_iso
        assert "1970" not in dto.started_at_iso


class TestWebcamServeFallback:
    """Verify serve daemon handles missing webcam hardware gracefully."""

    def test_webcam_open_failure_falls_back_to_synthetic(self) -> None:
        """When webcam cannot be opened, handle_serve falls back to SyntheticFrameSource."""
        from argparse import Namespace
        from dh360_hi.cli.serve import handle_serve

        args = Namespace(
            host="127.0.0.1",
            port=8000,
            source="webcam",
            camera_index=99,
            dialogue_mode="disabled",
            preset="testing",
            config_file=None,
            db_path=None,
            target_fps=None,
            detector_model_path=None,
            embedder_model_path=None,
            continuous_presence=True,
        )

        # Mock uvicorn.run to prevent blocking server run
        with patch("uvicorn.run") as mock_uvicorn_run, \
             patch("dh360_hi.vision.frame.WebcamFrameSource.open", side_effect=RuntimeError("Device busy")):
            # handle_serve should catch RuntimeError from webcam and fall back to synthetic
            exit_code = handle_serve(args)
            assert exit_code == 0
            assert mock_uvicorn_run.called

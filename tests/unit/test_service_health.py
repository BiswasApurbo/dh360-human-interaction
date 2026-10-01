"""Unit tests for service health evaluation and diagnostics."""

from __future__ import annotations

from unittest.mock import MagicMock

from dh360_hi.application.health import (
    ServiceHealthSnapshot,
    SubsystemHealth,
    SubsystemHealthStatus,
    evaluate_service_health,
)
from dh360_hi.application.service import HumanInteractionService
from dh360_hi.edge.runtime import RuntimeMetrics, RuntimeState


class TestServiceHealth:
    def test_health_dto_serialization(self) -> None:
        """Test SubsystemHealth and ServiceHealthSnapshot dictionary serialization."""
        sub = SubsystemHealth(
            name="runtime",
            status=SubsystemHealthStatus.HEALTHY,
            message="OK",
            details={"fps": 30.0},
        )
        snap = ServiceHealthSnapshot(
            status=SubsystemHealthStatus.HEALTHY,
            timestamp_iso="2026-10-01T12:00:00+00:00",
            subsystems={"runtime": sub},
            metrics=None,
        )
        d = snap.to_dict()
        assert d["status"] == "healthy"
        assert d["subsystems"]["runtime"]["status"] == "healthy"
        assert d["subsystems"]["runtime"]["details"]["fps"] == 30.0

    def test_evaluate_health_all_healthy(self) -> None:
        """Test evaluate_service_health when all components are active and healthy."""
        service = MagicMock(spec=HumanInteractionService)
        service.state = RuntimeState.RUNNING
        service.metrics = RuntimeMetrics(frames_processed=100, current_fps=30.0)

        db_mgr = MagicMock()
        db_mgr.get_connection.return_value.cursor.return_value.fetchone.return_value = (1,)
        db_mgr.db_path = ":memory:"
        service._db_manager = db_mgr

        worker = MagicMock()
        worker.queue_size = 0
        worker.is_alive = True
        service.persistence_worker = worker

        recognizer = MagicMock()
        recognizer.gallery.list_entries.return_value = ["p1", "p2"]
        service._recognizer = recognizer

        enrollment = MagicMock()
        enrollment.is_active = False
        service._enrollment_controller = enrollment

        snap = evaluate_service_health(service)
        assert snap.status == SubsystemHealthStatus.HEALTHY
        assert snap.subsystems["runtime"].status == SubsystemHealthStatus.HEALTHY
        assert snap.subsystems["database"].status == SubsystemHealthStatus.HEALTHY
        assert snap.subsystems["persistence_worker"].status == SubsystemHealthStatus.HEALTHY
        assert snap.subsystems["recognizer"].status == SubsystemHealthStatus.HEALTHY
        assert snap.subsystems["enrollment"].status == SubsystemHealthStatus.HEALTHY
        assert snap.metrics is not None
        assert snap.metrics.frames_processed == 100

    def test_evaluate_health_degraded_when_paused(self) -> None:
        """Test health status degrades when runtime is paused."""
        service = MagicMock(spec=HumanInteractionService)
        service.state = RuntimeState.PAUSED
        service.metrics = RuntimeMetrics()
        service._db_manager = None
        service.persistence_worker = None
        service._recognizer = None
        service._enrollment_controller = None

        snap = evaluate_service_health(service)
        assert snap.status == SubsystemHealthStatus.DEGRADED
        assert snap.subsystems["runtime"].status == SubsystemHealthStatus.DEGRADED
        assert snap.subsystems["database"].status == SubsystemHealthStatus.DISABLED

    def test_evaluate_health_unhealthy_on_exception(self) -> None:
        """Test health status becomes unhealthy when database connection throws."""
        service = MagicMock(spec=HumanInteractionService)
        service.state = RuntimeState.RUNNING
        service.metrics = RuntimeMetrics()

        db_mgr = MagicMock()
        db_mgr.get_connection.side_effect = RuntimeError("Disk I/O failure")
        service._db_manager = db_mgr
        service.persistence_worker = None
        service._recognizer = None
        service._enrollment_controller = None

        snap = evaluate_service_health(service)
        assert snap.status == SubsystemHealthStatus.UNHEALTHY
        assert snap.subsystems["database"].status == SubsystemHealthStatus.UNHEALTHY
        assert "Disk I/O failure" in (snap.subsystems["database"].message or "")

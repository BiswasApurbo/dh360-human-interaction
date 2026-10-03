"""Health evaluation and diagnostics for the Human Interaction Service and its subsystems."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from dh360_hi.backend.models import utc_now_iso
from dh360_hi.contracts.dto import RuntimeMetricsDTO
from dh360_hi.edge.runtime import RuntimeState

if TYPE_CHECKING:
    from dh360_hi.application.service import HumanInteractionService


class SubsystemHealthStatus(StrEnum):
    """Operational health status for a subsystem or the overall service."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    DISABLED = "disabled"


@dataclass(frozen=True)
class SubsystemHealth:
    """Health diagnostic for an individual subsystem."""

    name: str
    status: SubsystemHealthStatus
    message: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert subsystem health to a JSON-serializable dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
        }


@dataclass(frozen=True)
class ServiceHealthSnapshot:
    """Comprehensive health snapshot of the Human Interaction Service."""

    status: SubsystemHealthStatus
    timestamp_iso: str
    subsystems: dict[str, SubsystemHealth] = field(default_factory=dict)
    metrics: RuntimeMetricsDTO | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert service health snapshot to a JSON-serializable dictionary."""
        return {
            "status": self.status.value,
            "timestamp_iso": self.timestamp_iso,
            "subsystems": {k: v.to_dict() for k, v in self.subsystems.items()},
            "metrics": self.metrics.to_dict() if self.metrics is not None else None,
        }


def evaluate_service_health(service: HumanInteractionService) -> ServiceHealthSnapshot:
    """Inspect the service and all composed subsystems defensively without raising exceptions."""
    subsystems: dict[str, SubsystemHealth] = {}
    metrics_dto = None

    # 1. Edge Perception Runtime
    try:
        if service.runtime is not None:
            r_state = service.state
            r_metrics = service.metrics
            metrics_dto = RuntimeMetricsDTO.from_metrics(r_metrics)
            if r_state == RuntimeState.RUNNING:
                subsystems["runtime"] = SubsystemHealth(
                    name="runtime",
                    status=SubsystemHealthStatus.HEALTHY,
                    message="Perception runtime active",
                    details={"state": r_state.value, "fps": r_metrics.current_fps},
                )
            elif r_state == RuntimeState.PAUSED:
                subsystems["runtime"] = SubsystemHealth(
                    name="runtime",
                    status=SubsystemHealthStatus.DEGRADED,
                    message="Perception runtime paused",
                    details={"state": r_state.value},
                )
            elif r_state == RuntimeState.ERROR:
                subsystems["runtime"] = SubsystemHealth(
                    name="runtime",
                    status=SubsystemHealthStatus.UNHEALTHY,
                    message=r_metrics.last_error or "Runtime encountered an error",
                    details={"state": r_state.value, "error": r_metrics.last_error},
                )
            else:
                subsystems["runtime"] = SubsystemHealth(
                    name="runtime",
                    status=SubsystemHealthStatus.HEALTHY,
                    message=f"Runtime state: {r_state.value}",
                    details={"state": r_state.value},
                )
        else:
            subsystems["runtime"] = SubsystemHealth(
                name="runtime",
                status=SubsystemHealthStatus.DISABLED,
                message="Perception runtime not configured",
            )
    except Exception as exc:
        subsystems["runtime"] = SubsystemHealth(
            name="runtime",
            status=SubsystemHealthStatus.UNHEALTHY,
            message=str(exc),
        )

    # 2. Database Manager / SQLite Storage
    try:
        db_mgr = getattr(service, "_db_manager", None)
        if db_mgr is not None:
            # Perform lightweight read check
            conn = db_mgr.get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            subsystems["database"] = SubsystemHealth(
                name="database",
                status=SubsystemHealthStatus.HEALTHY,
                message="Database connection verified",
                details={"db_path": str(getattr(db_mgr, "db_path", ""))},
            )
        else:
            subsystems["database"] = SubsystemHealth(
                name="database",
                status=SubsystemHealthStatus.DISABLED,
                message="Database manager not configured",
            )
    except Exception as exc:
        subsystems["database"] = SubsystemHealth(
            name="database",
            status=SubsystemHealthStatus.UNHEALTHY,
            message=f"Database check failed: {exc}",
        )

    # 3. Session Persistence Worker
    try:
        worker = service.persistence_worker
        if worker is not None:
            queue_len = worker.queue_size
            is_alive = getattr(worker, "is_alive", getattr(worker, "is_running", False))
            subsystems["persistence_worker"] = SubsystemHealth(
                name="persistence_worker",
                status=SubsystemHealthStatus.HEALTHY,
                message="Session persistence worker active",
                details={"queue_size": queue_len, "is_alive": is_alive},
            )
        else:
            subsystems["persistence_worker"] = SubsystemHealth(
                name="persistence_worker",
                status=SubsystemHealthStatus.DISABLED,
                message="Session persistence worker not configured",
            )
    except Exception as exc:
        subsystems["persistence_worker"] = SubsystemHealth(
            name="persistence_worker",
            status=SubsystemHealthStatus.UNHEALTHY,
            message=f"Worker check failed: {exc}",
        )

    # 4. Face Recognizer & Gallery
    try:
        recognizer = getattr(service, "_recognizer", None)
        if recognizer is not None and hasattr(recognizer, "gallery"):
            entry_count = len(recognizer.gallery.list_entries())
            subsystems["recognizer"] = SubsystemHealth(
                name="recognizer",
                status=SubsystemHealthStatus.HEALTHY,
                message=f"Recognition gallery active with {entry_count} identities",
                details={"enrolled_identities": entry_count},
            )
        else:
            subsystems["recognizer"] = SubsystemHealth(
                name="recognizer",
                status=SubsystemHealthStatus.DISABLED,
                message="Recognizer or gallery not configured",
            )
    except Exception as exc:
        subsystems["recognizer"] = SubsystemHealth(
            name="recognizer",
            status=SubsystemHealthStatus.UNHEALTHY,
            message=f"Recognizer check failed: {exc}",
        )

    # 5. Face Enrollment Controller
    try:
        enrollment_ctrl = getattr(service, "_enrollment_controller", None)
        if enrollment_ctrl is not None:
            is_active = enrollment_ctrl.is_active
            subsystems["enrollment"] = SubsystemHealth(
                name="enrollment",
                status=SubsystemHealthStatus.HEALTHY,
                message="Enrollment controller ready"
                if not is_active
                else "Enrollment session in progress",
                details={"is_active": is_active},
            )
        else:
            subsystems["enrollment"] = SubsystemHealth(
                name="enrollment",
                status=SubsystemHealthStatus.DISABLED,
                message="Face enrollment controller not configured",
            )
    except Exception as exc:
        subsystems["enrollment"] = SubsystemHealth(
            name="enrollment",
            status=SubsystemHealthStatus.UNHEALTHY,
            message=f"Enrollment check failed: {exc}",
        )

    # Overall Status computation
    statuses = [s.status for s in subsystems.values()]
    if any(st == SubsystemHealthStatus.UNHEALTHY for st in statuses):
        overall = SubsystemHealthStatus.UNHEALTHY
    elif any(st == SubsystemHealthStatus.DEGRADED for st in statuses):
        overall = SubsystemHealthStatus.DEGRADED
    else:
        overall = SubsystemHealthStatus.HEALTHY

    return ServiceHealthSnapshot(
        status=overall,
        timestamp_iso=utc_now_iso(),
        subsystems=subsystems,
        metrics=metrics_dto,
    )

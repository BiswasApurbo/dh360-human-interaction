"""Contracts and typed dataclasses for deterministic scenario replay."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dh360_hi.contracts.integration import BaseIntegrationEvent


@dataclass(frozen=True)
class ScenarioLandmarks:
    """Five-point 2D facial landmark coordinates."""

    right_eye: tuple[float, float]
    left_eye: tuple[float, float]
    nose_tip: tuple[float, float]
    right_mouth_corner: tuple[float, float]
    left_mouth_corner: tuple[float, float]

    def to_dict(self) -> dict[str, list[float]]:
        """Convert landmarks to a dictionary."""
        return {
            "right_eye": list(self.right_eye),
            "left_eye": list(self.left_eye),
            "nose_tip": list(self.nose_tip),
            "right_mouth_corner": list(self.right_mouth_corner),
            "left_mouth_corner": list(self.left_mouth_corner),
        }


@dataclass(frozen=True)
class ScenarioDetection:
    """Scripted detection input for a frame."""

    bbox: tuple[float, float, float, float]  # (x, y, w, h)
    confidence: float = 0.95
    landmarks: ScenarioLandmarks | None = None
    embedding_label: str | None = None  # Optional identifier for recognizer/embedding mapping

    def to_dict(self) -> dict[str, Any]:
        """Convert detection to a dictionary."""
        d: dict[str, Any] = {
            "bbox": list(self.bbox),
            "confidence": self.confidence,
        }
        if self.landmarks is not None:
            d["landmarks"] = self.landmarks.to_dict()
        if self.embedding_label is not None:
            d["embedding_label"] = self.embedding_label
        return d


@dataclass(frozen=True)
class ScenarioFrame:
    """Scripted single frame input."""

    frame_id: int
    timestamp: float
    detections: list[ScenarioDetection] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert frame to a dictionary."""
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "detections": [d.to_dict() for d in self.detections],
        }


@dataclass(frozen=True)
class ScenarioPerson:
    """Registered profile to preload into the repository & gallery before running."""

    person_id: str
    display_name: str
    preferred_name: str | None = None
    status: str = "active"
    consent_status: str = "granted"
    notes: str | None = None
    embedding_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert person specification to a dictionary."""
        return {
            "person_id": self.person_id,
            "display_name": self.display_name,
            "preferred_name": self.preferred_name,
            "status": self.status,
            "consent_status": self.consent_status,
            "notes": self.notes,
            "embedding_index": self.embedding_index,
        }


@dataclass(frozen=True)
class ExpectedIntegrationEvent:
    """Expected integration event specification in a golden scenario."""

    event_type: str
    session_id: str | None = None
    person_id: str | None = None
    display_name: str | None = None
    previous_person_id: str | None = None
    approx_frame_id: int | None = None
    min_duration_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert expected event to a dictionary."""
        d: dict[str, Any] = {"event_type": self.event_type}
        if self.session_id is not None:
            d["session_id"] = self.session_id
        if self.person_id is not None:
            d["person_id"] = self.person_id
        if self.display_name is not None:
            d["display_name"] = self.display_name
        if self.previous_person_id is not None:
            d["previous_person_id"] = self.previous_person_id
        if self.approx_frame_id is not None:
            d["approx_frame_id"] = self.approx_frame_id
        if self.min_duration_seconds is not None:
            d["min_duration_seconds"] = self.min_duration_seconds
        return d


@dataclass(frozen=True)
class ScenarioDefinition:
    """Complete declarative scenario definition."""

    scenario_name: str
    description: str
    config_overrides: dict[str, Any] = field(default_factory=dict)
    initial_persons: list[ScenarioPerson] = field(default_factory=list)
    frames: list[ScenarioFrame] = field(default_factory=list)
    expected_events: list[ExpectedIntegrationEvent] = field(default_factory=list)
    expected_persisted_sessions: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert scenario definition to a dictionary."""
        return {
            "scenario_name": self.scenario_name,
            "description": self.description,
            "config_overrides": dict(self.config_overrides),
            "initial_persons": [p.to_dict() for p in self.initial_persons],
            "frames": [f.to_dict() for f in self.frames],
            "expected_events": [e.to_dict() for e in self.expected_events],
            "expected_persisted_sessions": self.expected_persisted_sessions,
        }


@dataclass(frozen=True)
class ScenarioResult:
    """Execution output and telemetry of a replayed scenario."""

    scenario_name: str
    total_frames_processed: int
    collected_integration_events: list[BaseIntegrationEvent]
    persisted_sessions_count: int
    is_success: bool
    error_message: str | None = None

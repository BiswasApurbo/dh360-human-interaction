"""JSON loader and validator for declarative scenario definitions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dh360_hi.testing.contracts import (
    ExpectedIntegrationEvent,
    ScenarioDefinition,
    ScenarioDetection,
    ScenarioFrame,
    ScenarioLandmarks,
    ScenarioPerson,
)


def _parse_landmarks(raw: dict[str, Any] | None) -> ScenarioLandmarks | None:
    if raw is None:
        return None
    try:
        return ScenarioLandmarks(
            right_eye=(float(raw["right_eye"][0]), float(raw["right_eye"][1])),
            left_eye=(float(raw["left_eye"][0]), float(raw["left_eye"][1])),
            nose_tip=(float(raw["nose_tip"][0]), float(raw["nose_tip"][1])),
            right_mouth_corner=(
                float(raw["right_mouth_corner"][0]),
                float(raw["right_mouth_corner"][1]),
            ),
            left_mouth_corner=(
                float(raw["left_mouth_corner"][0]),
                float(raw["left_mouth_corner"][1]),
            ),
        )
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        msg = f"Invalid landmarks structure: {exc}"
        raise ValueError(msg) from exc


def _parse_detection(raw: dict[str, Any]) -> ScenarioDetection:
    if "bbox" not in raw:
        msg = "Detection must contain 'bbox' array of [x, y, w, h]"
        raise ValueError(msg)
    raw_bbox = raw["bbox"]
    if len(raw_bbox) != 4:
        msg = f"Detection 'bbox' must have exactly 4 numbers, got {raw_bbox}"
        raise ValueError(msg)

    bbox = (
        float(raw_bbox[0]),
        float(raw_bbox[1]),
        float(raw_bbox[2]),
        float(raw_bbox[3]),
    )
    confidence = float(raw.get("confidence", 0.95))
    landmarks = _parse_landmarks(raw.get("landmarks"))
    embedding_label = raw.get("embedding_label")

    return ScenarioDetection(
        bbox=bbox,
        confidence=confidence,
        landmarks=landmarks,
        embedding_label=embedding_label,
    )


def _parse_frame(raw: dict[str, Any]) -> ScenarioFrame:
    if "frame_id" not in raw or "timestamp" not in raw:
        msg = "Frame must contain 'frame_id' and 'timestamp'"
        raise ValueError(msg)

    frame_id = int(raw["frame_id"])
    timestamp = float(raw["timestamp"])
    detections_raw = raw.get("detections", [])
    detections = [_parse_detection(d) for d in detections_raw]

    return ScenarioFrame(
        frame_id=frame_id,
        timestamp=timestamp,
        detections=detections,
    )


def _parse_person(raw: dict[str, Any]) -> ScenarioPerson:
    if "person_id" not in raw or "display_name" not in raw:
        msg = "Initial person must contain 'person_id' and 'display_name'"
        raise ValueError(msg)

    return ScenarioPerson(
        person_id=str(raw["person_id"]),
        display_name=str(raw["display_name"]),
        preferred_name=raw.get("preferred_name"),
        status=str(raw.get("status", "active")),
        consent_status=str(raw.get("consent_status", "granted")),
        notes=raw.get("notes"),
        embedding_index=raw.get("embedding_index"),
    )


def _parse_expected_event(raw: dict[str, Any]) -> ExpectedIntegrationEvent:
    if "event_type" not in raw:
        msg = "Expected event must specify 'event_type'"
        raise ValueError(msg)

    min_dur = raw.get("min_duration_seconds")
    return ExpectedIntegrationEvent(
        event_type=str(raw["event_type"]),
        session_id=raw.get("session_id"),
        person_id=raw.get("person_id"),
        display_name=raw.get("display_name"),
        previous_person_id=raw.get("previous_person_id"),
        approx_frame_id=raw.get("approx_frame_id"),
        min_duration_seconds=float(min_dur) if min_dur is not None else None,
    )


def parse_scenario_dict(data: dict[str, Any]) -> ScenarioDefinition:
    """Parse a dictionary into a validated ScenarioDefinition."""
    if not isinstance(data, dict):
        msg = f"Scenario data must be a dict, got {type(data).__name__}"
        raise ValueError(msg)

    if "scenario_name" not in data or not str(data["scenario_name"]).strip():
        msg = "Scenario definition must contain a non-empty 'scenario_name'"
        raise ValueError(msg)

    scenario_name = str(data["scenario_name"]).strip()
    description = str(data.get("description", ""))
    config_overrides = dict(data.get("config_overrides", {}))

    initial_persons = [_parse_person(p) for p in data.get("initial_persons", [])]
    frames = [_parse_frame(f) for f in data.get("frames", [])]
    expected_events = [_parse_expected_event(e) for e in data.get("expected_events", [])]
    exp_sessions = data.get("expected_persisted_sessions")
    expected_persisted_sessions = int(exp_sessions) if exp_sessions is not None else None

    return ScenarioDefinition(
        scenario_name=scenario_name,
        description=description,
        config_overrides=config_overrides,
        initial_persons=initial_persons,
        frames=frames,
        expected_events=expected_events,
        expected_persisted_sessions=expected_persisted_sessions,
    )


def load_scenario_from_json(path_or_content: str | Path | dict[str, Any]) -> ScenarioDefinition:
    """Load and validate a ScenarioDefinition from a JSON file, string, or dictionary."""
    if isinstance(path_or_content, dict):
        return parse_scenario_dict(path_or_content)

    if isinstance(path_or_content, Path) or (
        isinstance(path_or_content, str) and not path_or_content.strip().startswith("{")
    ):
        file_path = Path(path_or_content)
        if not file_path.exists():
            msg = f"Scenario file not found: {file_path}"
            raise FileNotFoundError(msg)
        content = file_path.read_text(encoding="utf-8")
        data = json.loads(content)
        return parse_scenario_dict(data)

    data = json.loads(path_or_content)
    return parse_scenario_dict(data)


def save_scenario_to_json(scenario: ScenarioDefinition, target_path: str | Path) -> Path:
    """Serialize a ScenarioDefinition to a JSON file."""
    path = Path(target_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scenario.to_dict(), indent=2), encoding="utf-8")
    return path

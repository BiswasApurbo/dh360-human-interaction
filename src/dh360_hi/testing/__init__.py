"""Deterministic scenario replay and interaction validation harness package."""

from __future__ import annotations

from dh360_hi.testing.assertions import (
    assert_persisted_session_matches,
    assert_scenario_events,
)
from dh360_hi.testing.contracts import (
    ExpectedIntegrationEvent,
    ScenarioDefinition,
    ScenarioDetection,
    ScenarioFrame,
    ScenarioLandmarks,
    ScenarioPerson,
    ScenarioResult,
)
from dh360_hi.testing.event_collector import ScenarioEventCollector
from dh360_hi.testing.loader import (
    load_scenario_from_json,
    parse_scenario_dict,
    save_scenario_to_json,
)
from dh360_hi.testing.runner import ScenarioRunner, build_scenario_config
from dh360_hi.testing.scripted_components import (
    DirectEmbeddingWorker,
    ScriptedFaceDetector,
    ScriptedFaceEmbedder,
    ScriptedFrameSource,
    make_deterministic_embedding,
)

__all__ = [
    "DirectEmbeddingWorker",
    "ExpectedIntegrationEvent",
    "ScenarioDefinition",
    "ScenarioDetection",
    "ScenarioEventCollector",
    "ScenarioFrame",
    "ScenarioLandmarks",
    "ScenarioPerson",
    "ScenarioResult",
    "ScenarioRunner",
    "ScriptedFaceDetector",
    "ScriptedFaceEmbedder",
    "ScriptedFrameSource",
    "assert_persisted_session_matches",
    "assert_scenario_events",
    "build_scenario_config",
    "load_scenario_from_json",
    "make_deterministic_embedding",
    "parse_scenario_dict",
    "save_scenario_to_json",
]

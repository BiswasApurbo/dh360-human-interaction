"""Unit tests for testing harness components.

Tests contracts, loader, scripted components, collector, assertions.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from dh360_hi.contracts.dto import InteractionSessionDTO
from dh360_hi.contracts.integration import (
    BaseIntegrationEvent,
    PersonIdentifiedIntegrationEvent,
    SessionEndedIntegrationEvent,
    SessionStartedIntegrationEvent,
)
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
)
from dh360_hi.testing.event_collector import ScenarioEventCollector
from dh360_hi.testing.loader import (
    load_scenario_from_json,
    parse_scenario_dict,
    save_scenario_to_json,
)
from dh360_hi.testing.scripted_components import (
    DirectEmbeddingWorker,
    ScriptedFaceDetector,
    ScriptedFaceEmbedder,
    ScriptedFrameSource,
)
from dh360_hi.vision.embedding_worker import EmbeddingTask
from dh360_hi.vision.recognition import FaceRecognizer


class TestScenarioContracts:
    """Test serialization and conversion of scenario dataclasses."""

    def test_landmarks_to_dict(self) -> None:
        lm = ScenarioLandmarks(
            right_eye=(10.0, 20.0),
            left_eye=(30.0, 20.0),
            nose_tip=(20.0, 30.0),
            right_mouth_corner=(15.0, 40.0),
            left_mouth_corner=(25.0, 40.0),
        )
        d = lm.to_dict()
        assert d["right_eye"] == [10.0, 20.0]
        assert d["nose_tip"] == [20.0, 30.0]

    def test_detection_to_dict(self) -> None:
        det = ScenarioDetection(
            bbox=(10.0, 20.0, 30.0, 40.0),
            confidence=0.9,
            embedding_label="alice",
        )
        d = det.to_dict()
        assert d["bbox"] == [10.0, 20.0, 30.0, 40.0]
        assert d["confidence"] == 0.9
        assert d["embedding_label"] == "alice"

    def test_frame_and_person_to_dict(self) -> None:
        frame = ScenarioFrame(frame_id=1, timestamp=10.0, detections=[])
        assert frame.to_dict() == {"frame_id": 1, "timestamp": 10.0, "detections": []}

        person = ScenarioPerson(
            person_id="p1",
            display_name="Person One",
            embedding_index=5,
        )
        pd = person.to_dict()
        assert pd["person_id"] == "p1"
        assert pd["embedding_index"] == 5


class TestScenarioLoader:
    """Test loading and validating scenario JSON definitions."""

    def test_parse_valid_scenario_dict(self) -> None:
        data = {
            "scenario_name": "test_scenario",
            "description": "Test description",
            "config_overrides": {"tracker_min_hits": 2},
            "initial_persons": [
                {
                    "person_id": "p1",
                    "display_name": "P1",
                }
            ],
            "frames": [
                {
                    "frame_id": 1,
                    "timestamp": 1.0,
                    "detections": [
                        {
                            "bbox": [10.0, 20.0, 30.0, 40.0],
                            "confidence": 0.95,
                        }
                    ],
                }
            ],
            "expected_events": [
                {
                    "event_type": "session_started",
                    "person_id": "p1",
                }
            ],
            "expected_persisted_sessions": 1,
        }

        scenario = parse_scenario_dict(data)
        assert scenario.scenario_name == "test_scenario"
        assert len(scenario.frames) == 1
        assert len(scenario.initial_persons) == 1
        assert len(scenario.expected_events) == 1
        assert scenario.expected_persisted_sessions == 1

    def test_save_and_load_scenario_roundtrip(self) -> None:
        scenario = ScenarioDefinition(
            scenario_name="roundtrip_test",
            description="Testing save and load roundtrip",
            config_overrides={"presence_enter_frames": 2},
            initial_persons=[ScenarioPerson(person_id="p-1", display_name="Test User")],
            frames=[ScenarioFrame(frame_id=1, timestamp=0.5, detections=[])],
            expected_events=[ExpectedIntegrationEvent(event_type="session_started")],
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "test_scen.json"
            save_scenario_to_json(scenario, file_path)
            loaded = load_scenario_from_json(file_path)

            assert loaded.scenario_name == "roundtrip_test"
            assert len(loaded.initial_persons) == 1
            assert loaded.initial_persons[0].display_name == "Test User"

    def test_load_scenario_validation_errors(self) -> None:
        with pytest.raises(ValueError, match="Scenario data must be a dict"):
            parse_scenario_dict("invalid")  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="non-empty 'scenario_name'"):
            parse_scenario_dict({"scenario_name": ""})

        with pytest.raises(ValueError, match="must contain 'bbox'"):
            parse_scenario_dict(
                {
                    "scenario_name": "s1",
                    "frames": [{"frame_id": 1, "timestamp": 1.0, "detections": [{}]}],
                }
            )

        with pytest.raises(ValueError, match="must have exactly 4 numbers"):
            parse_scenario_dict(
                {
                    "scenario_name": "s1",
                    "frames": [{"frame_id": 1, "timestamp": 1.0, "detections": [{"bbox": [1, 2]}]}],
                }
            )


class TestScriptedComponents:
    """Test scripted perception components."""

    def test_scripted_face_detector(self) -> None:
        frame_spec = ScenarioFrame(
            frame_id=5,
            timestamp=2.5,
            detections=[
                ScenarioDetection(
                    bbox=(10.0, 10.0, 50.0, 50.0),
                    confidence=0.9,
                    embedding_label="user_a",
                )
            ],
        )
        detector = ScriptedFaceDetector(frames=[frame_spec])

        res5 = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8), frame_id=5)
        assert len(res5.detections) == 1
        assert res5.detections[0].bbox == (10.0, 10.0, 50.0, 50.0)

        res6 = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8), frame_id=6)
        assert len(res6.detections) == 0

    def test_scripted_face_embedder(self) -> None:
        embedder = ScriptedFaceEmbedder()
        embedder.register_label("user_1", 1)
        embedder.register_label("user_2", 2)

        embedder.set_active_label("user_1")
        res1 = embedder.embed(np.zeros((112, 112, 3), dtype=np.uint8))
        assert res1.is_normalized
        assert res1.embedding[1] == 1.0
        assert res1.embedding[2] == 0.0

        embedder.set_active_label("user_2")
        res2 = embedder.embed(np.zeros((112, 112, 3), dtype=np.uint8))
        assert res2.embedding[2] == 1.0

    def test_scripted_frame_source(self) -> None:
        frames = [
            ScenarioFrame(
                frame_id=1,
                timestamp=1.0,
                detections=[ScenarioDetection(bbox=(10, 10, 40, 40))],
            ),
            ScenarioFrame(frame_id=2, timestamp=2.0, detections=[]),
        ]
        src = ScriptedFrameSource(frames)
        src.open()
        assert src.is_opened

        f1 = src.read()
        assert f1 is not None
        assert f1.frame_id == 1
        assert f1.timestamp == 1.0
        assert f1.image.shape == (320, 320, 3)

        f2 = src.read()
        assert f2 is not None
        assert f2.frame_id == 2

        f3 = src.read()
        assert f3 is None

        src.release()
        assert not src.is_opened

    def test_direct_embedding_worker(self) -> None:
        embedder = ScriptedFaceEmbedder()
        recognizer = FaceRecognizer()
        worker = DirectEmbeddingWorker(embedder=embedder, recognizer=recognizer)

        assert worker.is_running
        assert worker.pending_count == 0

        task = EmbeddingTask(
            track_id=1,
            frame_id=10,
            timestamp=5.0,
            image=np.zeros((112, 112, 3), dtype=np.uint8),
            bbox=(0, 0, 50, 50),
        )
        assert worker.submit(task) is True
        assert worker.pending_count == 1

        results = worker.poll_results()
        assert len(results) == 1
        assert results[0].track_id == 1
        assert results[0].frame_id == 10
        assert worker.pending_count == 0


class TestEventCollectorAndAssertions:
    """Test event collector and assertion helpers."""

    def test_collector_recording_and_filtering(self) -> None:
        collector = ScenarioEventCollector()
        ev1 = SessionStartedIntegrationEvent(session_id="s1", person_id="p1")
        ev2 = PersonIdentifiedIntegrationEvent(session_id="s1", person_id="p1")

        collector.on_integration_event(ev1)
        collector.on_integration_event(ev2)

        assert collector.get_event_types() == ["session_started", "person_identified"]
        assert collector.find_first_event("session_started") == ev1
        assert collector.find_events_by_type(PersonIdentifiedIntegrationEvent) == [ev2]

        collector.clear()
        assert len(collector.get_integration_events()) == 0

    def test_assert_scenario_events_success(self) -> None:
        collected: list[BaseIntegrationEvent] = [
            SessionStartedIntegrationEvent(session_id="s1", person_id="p1", display_name="Alice"),
            PersonIdentifiedIntegrationEvent(session_id="s1", person_id="p1", display_name="Alice"),
            SessionEndedIntegrationEvent(
                session_id="s1", final_person_id="p1", final_display_name="Alice"
            ),
        ]

        expected = [
            ExpectedIntegrationEvent(
                event_type="session_started", person_id="p1", display_name="Alice"
            ),
            ExpectedIntegrationEvent(event_type="person_identified", person_id="p1"),
            ExpectedIntegrationEvent(event_type="session_ended", person_id="p1"),
        ]

        # Should pass without error
        assert_scenario_events(collected, expected)

    def test_assert_scenario_events_mismatch_type(self) -> None:
        collected: list[BaseIntegrationEvent] = [
            SessionStartedIntegrationEvent(session_id="s1"),
        ]
        expected = [
            ExpectedIntegrationEvent(event_type="session_ended"),
        ]

        with pytest.raises(AssertionError, match="type mismatch"):
            assert_scenario_events(collected, expected)

    def test_assert_scenario_events_mismatch_count(self) -> None:
        collected: list[BaseIntegrationEvent] = [
            SessionStartedIntegrationEvent(session_id="s1"),
        ]
        expected = [
            ExpectedIntegrationEvent(event_type="session_started"),
            ExpectedIntegrationEvent(event_type="session_ended"),
        ]

        with pytest.raises(AssertionError, match="Event count mismatch"):
            assert_scenario_events(collected, expected)

    def test_assert_persisted_session_matches(self) -> None:
        session = InteractionSessionDTO(
            session_id="s1",
            person_id="p1",
            display_name="Alice",
            started_at_iso="2026-01-01T00:00:00Z",
            ended_at_iso="2026-01-01T00:00:10Z",
            duration_seconds=10.0,
            final_state="completed",
            identity_was_known=True,
            identity_transitions_count=1,
            created_at="2026-01-01T00:00:10Z",
        )

        assert_persisted_session_matches(session, expected_person_id="p1", min_duration=5.0)

        with pytest.raises(AssertionError, match="person_id mismatch"):
            assert_persisted_session_matches(session, expected_person_id="p2")

        with pytest.raises(AssertionError, match="duration_seconds mismatch"):
            assert_persisted_session_matches(session, min_duration=20.0)

"""End-to-end deterministic replay tests for all five golden scenarios."""

from __future__ import annotations

from pathlib import Path

from dh360_hi.contracts.integration import (
    PersonIdentifiedIntegrationEvent,
    SessionEndedIntegrationEvent,
    SessionStartedIntegrationEvent,
)
from dh360_hi.testing.loader import load_scenario_from_json
from dh360_hi.testing.runner import ScenarioRunner


class TestGoldenScenarios:
    """Suite executing the 5 official golden interaction scenarios."""

    def test_01_known_person_lifecycle(self, scenarios_dir: Path) -> None:
        """Scenario 1: Full lifecycle of a single enrolled person."""
        scenario_path = scenarios_dir / "01_known_person_lifecycle.json"
        scenario = load_scenario_from_json(scenario_path)

        runner = ScenarioRunner(auto_assert_events=True)
        result = runner.run(scenario)

        assert result.is_success, f"Scenario failed: {result.error_message}"
        assert result.total_frames_processed == 10
        assert result.persisted_sessions_count == 1

        # Verify exact integration event payload contents
        events = [
            e for e in result.collected_integration_events if e.event_type != "context_updated"
        ]
        assert len(events) == 3

        start_ev, id_ev, end_ev = events
        assert isinstance(start_ev, SessionStartedIntegrationEvent)
        assert start_ev.person_id == "00000000-0000-0000-0000-000000000001"
        assert start_ev.display_name == "Alice Smith"

        assert isinstance(id_ev, PersonIdentifiedIntegrationEvent)
        assert id_ev.person_id == "00000000-0000-0000-0000-000000000001"

        assert isinstance(end_ev, SessionEndedIntegrationEvent)
        assert end_ev.final_person_id == "00000000-0000-0000-0000-000000000001"
        assert end_ev.summary is not None
        assert end_ev.summary.identity_was_known is True

    def test_02_anonymous_session_lifecycle(self, scenarios_dir: Path) -> None:
        """Scenario 2: Unregistered person interaction lifecycle."""
        scenario_path = scenarios_dir / "02_anonymous_session_lifecycle.json"
        scenario = load_scenario_from_json(scenario_path)

        runner = ScenarioRunner(auto_assert_events=True)
        result = runner.run(scenario)

        assert result.is_success, f"Scenario failed: {result.error_message}"
        assert result.total_frames_processed == 10
        assert result.persisted_sessions_count == 1

        events = [
            e for e in result.collected_integration_events if e.event_type != "context_updated"
        ]
        assert len(events) == 2

        start_ev, end_ev = events
        assert isinstance(start_ev, SessionStartedIntegrationEvent)
        assert start_ev.person_id is None

        assert isinstance(end_ev, SessionEndedIntegrationEvent)
        assert end_ev.final_person_id is None
        assert end_ev.summary is not None
        assert end_ev.summary.identity_was_known is False

    def test_03_mid_session_identification(self, scenarios_dir: Path) -> None:
        """Scenario 3: Anonymous start with mid-session identification."""
        scenario_path = scenarios_dir / "03_mid_session_identification.json"
        scenario = load_scenario_from_json(scenario_path)

        runner = ScenarioRunner(auto_assert_events=True)
        result = runner.run(scenario)

        assert result.is_success, f"Scenario failed: {result.error_message}"
        assert result.total_frames_processed == 15
        assert result.persisted_sessions_count == 1

        events = [
            e for e in result.collected_integration_events if e.event_type != "context_updated"
        ]
        assert len(events) == 3

        start_ev, id_ev, end_ev = events
        assert isinstance(start_ev, SessionStartedIntegrationEvent)
        assert start_ev.person_id is None  # Initially anonymous

        assert isinstance(id_ev, PersonIdentifiedIntegrationEvent)
        assert id_ev.person_id == "00000000-0000-0000-0000-000000000002"
        assert id_ev.display_name == "Bob Jones"

        assert isinstance(end_ev, SessionEndedIntegrationEvent)
        assert end_ev.final_person_id == "00000000-0000-0000-0000-000000000002"
        assert end_ev.summary is not None
        assert end_ev.summary.person_id == "00000000-0000-0000-0000-000000000002"

    def test_04_target_switching_hysteresis(self, scenarios_dir: Path) -> None:
        """Scenario 4: Primary vs secondary target focus and handover."""
        scenario_path = scenarios_dir / "04_target_switching_hysteresis.json"
        scenario = load_scenario_from_json(scenario_path)

        runner = ScenarioRunner(auto_assert_events=True)
        result = runner.run(scenario)

        assert result.is_success, f"Scenario failed: {result.error_message}"
        assert result.total_frames_processed == 22
        assert result.persisted_sessions_count == 2

        events = [
            e for e in result.collected_integration_events if e.event_type != "context_updated"
        ]
        assert len(events) == 6

        # Alice's session
        alice_start, alice_id, alice_end = events[0], events[1], events[2]
        assert isinstance(alice_start, SessionStartedIntegrationEvent)
        assert alice_start.person_id == "00000000-0000-0000-0000-000000000001"
        assert isinstance(alice_id, PersonIdentifiedIntegrationEvent)
        assert isinstance(alice_end, SessionEndedIntegrationEvent)
        assert alice_end.final_person_id == "00000000-0000-0000-0000-000000000001"

        # Carol's promoted session
        carol_start, carol_id, carol_end = events[3], events[4], events[5]
        assert isinstance(carol_start, SessionStartedIntegrationEvent)
        assert carol_start.person_id == "00000000-0000-0000-0000-000000000003"
        assert isinstance(carol_id, PersonIdentifiedIntegrationEvent)
        assert isinstance(carol_end, SessionEndedIntegrationEvent)
        assert carol_end.final_person_id == "00000000-0000-0000-0000-000000000003"

    def test_05_debounce_glitch_tolerance(self, scenarios_dir: Path) -> None:
        """Scenario 5: 2-frame tracking dropout handled without session termination."""
        scenario_path = scenarios_dir / "05_debounce_glitch_tolerance.json"
        scenario = load_scenario_from_json(scenario_path)

        runner = ScenarioRunner(auto_assert_events=True)
        result = runner.run(scenario)

        assert result.is_success, f"Scenario failed: {result.error_message}"
        assert result.total_frames_processed == 15
        assert result.persisted_sessions_count == 1

        events = [
            e for e in result.collected_integration_events if e.event_type != "context_updated"
        ]
        assert len(events) == 3

        start_ev, id_ev, end_ev = events
        assert isinstance(start_ev, SessionStartedIntegrationEvent)
        assert isinstance(id_ev, PersonIdentifiedIntegrationEvent)
        assert isinstance(end_ev, SessionEndedIntegrationEvent)
        assert end_ev.final_person_id == "00000000-0000-0000-0000-000000000001"

    def test_deterministic_repeatability(self, scenarios_dir: Path) -> None:
        """Verify that executing the same scenario multiple times produces identical results."""
        scenario_path = scenarios_dir / "01_known_person_lifecycle.json"
        scenario = load_scenario_from_json(scenario_path)

        runner = ScenarioRunner(auto_assert_events=True)
        res1 = runner.run(scenario)
        res2 = runner.run(scenario)
        res3 = runner.run(scenario)

        assert res1.is_success and res2.is_success and res3.is_success

        types1 = [e.event_type for e in res1.collected_integration_events]
        types2 = [e.event_type for e in res2.collected_integration_events]
        types3 = [e.event_type for e in res3.collected_integration_events]

        assert types1 == types2 == types3

"""Unit tests for transport-neutral Application DTOs and command payloads."""

from __future__ import annotations

import dataclasses
import uuid

import pytest

from dh360_hi.backend.exceptions import ValidationError
from dh360_hi.backend.models import (
    ConsentStatus,
    PersonEngagementStats,
    PersonProfile,
    PersonStatus,
    SessionAggregateAnalytics,
    SessionRecord,
)
from dh360_hi.contracts.dto import (
    CreatePersonCommand,
    EnrollmentProgressDTO,
    EnrollmentResultDTO,
    InteractionContextDTO,
    InteractionSessionDTO,
    InteractionSummaryDTO,
    PersonEngagementStatsDTO,
    PersonProfileDTO,
    RuntimeMetricsDTO,
    SessionAggregateAnalyticsDTO,
    StepResultDTO,
    UpdatePersonCommand,
    epoch_to_iso,
)
from dh360_hi.contracts.integration import SessionStartedIntegrationEvent
from dh360_hi.contracts.interaction import (
    InteractionContext,
    InteractionState,
    InteractionSummary,
)
from dh360_hi.edge.runtime import RuntimeMetrics
from dh360_hi.vision.enrollment import (
    EnrollmentMode,
    EnrollmentProgress,
    EnrollmentResult,
    EnrollmentState,
    SampleRejectionReason,
)


class TestEpochToIso:
    def test_valid_epoch(self) -> None:
        """Test timestamp conversion to ISO 8601 UTC string."""
        iso_str = epoch_to_iso(1700000000.0)
        assert iso_str is not None
        assert "+00:00" in iso_str or "Z" in iso_str

    def test_none_or_negative_epoch(self) -> None:
        """Test None or invalid epoch handling."""
        assert epoch_to_iso(None) is None
        assert epoch_to_iso(0.0) is None
        assert epoch_to_iso(-10.0) is None


class TestPersonDTOsAndCommands:
    def test_person_profile_dto_from_profile(self) -> None:
        """Test constructing PersonProfileDTO from PersonProfile domain model."""
        pid = str(uuid.uuid4())
        profile = PersonProfile(
            person_id=pid,
            display_name="Alice Smith",
            preferred_name="Alice",
            status=PersonStatus.ACTIVE,
            consent_status=ConsentStatus.GRANTED,
            consent_timestamp="2026-10-01T12:00:00+00:00",
            notes="Confidential internal notes",
            created_at="2026-10-01T12:00:00+00:00",
            updated_at="2026-10-01T12:00:00+00:00",
            embedding_count=5,
        )
        dto = PersonProfileDTO.from_profile(profile)

        assert dto.person_id == pid
        assert dto.display_name == "Alice Smith"
        assert dto.preferred_name == "Alice"
        assert dto.status == "active"
        assert dto.consent_status == "granted"
        # Privacy verification: notes and embedding_count are excluded from PersonProfileDTO
        assert not hasattr(dto, "notes")
        assert not hasattr(dto, "embedding_count")

        d = dto.to_dict()
        assert d["person_id"] == pid
        assert "notes" not in d

        # Frozen immutability check
        with pytest.raises(dataclasses.FrozenInstanceError):
            dto.display_name = "Bob"  # type: ignore[misc]

    def test_create_person_command_validation(self) -> None:
        """Test CreatePersonCommand validation logic."""
        cmd = CreatePersonCommand(display_name="Charlie")
        assert cmd.display_name == "Charlie"
        assert cmd.to_dict()["display_name"] == "Charlie"

        with pytest.raises(ValidationError):
            CreatePersonCommand(display_name="   ")

        with pytest.raises(ValidationError):
            CreatePersonCommand(display_name="Charlie", person_id="invalid-uuid")

    def test_update_person_command_validation(self) -> None:
        """Test UpdatePersonCommand validation logic."""
        cmd = UpdatePersonCommand(display_name="Dave", preferred_name="D")
        assert cmd.display_name == "Dave"
        assert cmd.to_dict()["display_name"] == "Dave"

        with pytest.raises(ValidationError):
            UpdatePersonCommand(display_name="   ")


class TestInteractionContextAndSummaryDTOs:
    def test_interaction_context_dto_from_context(self) -> None:
        """Test constructing InteractionContextDTO from InteractionContext."""
        pid = str(uuid.uuid4())
        profile = PersonProfile(
            person_id=pid,
            display_name="Alice Smith",
            preferred_name="Alice",
            status=PersonStatus.ACTIVE,
            consent_status=ConsentStatus.GRANTED,
            consent_timestamp="2026-10-01T12:00:00+00:00",
            notes=None,
            created_at="2026-10-01T12:00:00+00:00",
            updated_at="2026-10-01T12:00:00+00:00",
        )
        ctx = InteractionContext(
            session_id="sess-123",
            person_id=pid,
            display_name="Alice Smith",
            person_profile=profile,
            state=InteractionState.ENGAGED,
            started_at=1700000000.0,
            last_updated_at=1700000010.0,
            identity_known=True,
            last_known_person_id=None,
            identity_transitions_count=1,
        )
        dto = InteractionContextDTO.from_context(ctx)

        assert dto.session_id == "sess-123"
        assert dto.person_id == pid
        assert dto.state == "engaged"
        assert dto.started_at_iso is not None
        assert dto.identity_known is True
        assert dto.person_profile is not None
        assert dto.person_profile.display_name == "Alice Smith"

        data = dto.to_dict()
        assert data["session_id"] == "sess-123"
        assert data["person_profile"]["person_id"] == pid

    def test_interaction_summary_dto_from_summary(self) -> None:
        """Test constructing InteractionSummaryDTO from InteractionSummary."""
        summary = InteractionSummary(
            session_id="sess-456",
            person_id="p-1",
            display_name="Bob",
            person_profile=None,
            started_at=1700000000.0,
            ended_at=1700000020.0,
            duration_seconds=20.0,
            final_state=InteractionState.FAREWELL,
            identity_was_known=True,
            identity_transitions_count=2,
        )
        dto = InteractionSummaryDTO.from_summary(summary)

        assert dto.session_id == "sess-456"
        assert dto.duration_seconds == 20.0
        assert dto.final_state == "farewell"
        assert dto.started_at_iso is not None
        assert dto.ended_at_iso is not None

        d = dto.to_dict()
        assert d["duration_seconds"] == 20.0


class TestSessionAndAnalyticsDTOs:
    def test_session_dto_from_record(self) -> None:
        """Test constructing InteractionSessionDTO from SessionRecord."""
        sid = str(uuid.uuid4())
        pid = str(uuid.uuid4())
        rec = SessionRecord(
            session_id=sid,
            person_id=pid,
            display_name="Carol",
            started_at=1700000000.0,
            ended_at=1700000030.0,
            duration_seconds=30.0,
            final_state="farewell",
            identity_was_known=True,
            identity_transitions_count=1,
            created_at="2026-10-01T12:00:00+00:00",
        )
        dto = InteractionSessionDTO.from_record(rec)
        assert dto.session_id == sid
        assert dto.duration_seconds == 30.0
        assert dto.final_state == "farewell"
        assert dto.to_dict()["session_id"] == sid

    def test_analytics_and_engagement_dtos(self) -> None:
        """Test SessionAggregateAnalyticsDTO and PersonEngagementStatsDTO."""
        analytics = SessionAggregateAnalytics(
            total_sessions=10,
            total_duration_seconds=100.0,
            average_duration_seconds=10.0,
            min_duration_seconds=5.0,
            max_duration_seconds=15.0,
            identified_sessions_count=8,
            anonymous_sessions_count=2,
            earliest_session_started_at=1700000000.0,
            latest_session_ended_at=1700000100.0,
        )
        dto = SessionAggregateAnalyticsDTO.from_analytics(analytics)
        assert dto.total_sessions == 10
        assert dto.identified_sessions_count == 8
        assert dto.earliest_session_started_at_iso is not None
        assert dto.to_dict()["total_sessions"] == 10

        pid = str(uuid.uuid4())
        stats = PersonEngagementStats(
            person_id=pid,
            display_name="Dave",
            total_sessions=5,
            total_duration_seconds=50.0,
            average_duration_seconds=10.0,
            first_interaction_at=1700000000.0,
            last_interaction_at=1700000050.0,
        )
        stats_dto = PersonEngagementStatsDTO.from_stats(stats)
        assert stats_dto.person_id == pid
        assert stats_dto.total_sessions == 5
        assert stats_dto.first_interaction_at_iso is not None
        assert stats_dto.to_dict()["person_id"] == pid


class TestEnrollmentDTOs:
    def test_enrollment_progress_dto_from_progress(self) -> None:
        """Test constructing EnrollmentProgressDTO from EnrollmentProgress."""
        prog = EnrollmentProgress(
            session_id="esess-1",
            enrollment_batch_id="batch-1",
            person_id="p-1",
            display_name="Eve",
            mode=EnrollmentMode.ADD_SAMPLES,
            state=EnrollmentState.COLLECTING,
            samples_collected=3,
            target_samples=5,
            minimum_samples=3,
            total_attempts=6,
            last_rejection_reason=SampleRejectionReason.NO_FACE,
            last_rejection_details="No bounding box",
            rejection_counts={"no_face": 3},
            is_viable=True,
            is_complete=False,
        )
        dto = EnrollmentProgressDTO.from_progress(prog)
        assert dto.session_id == "esess-1"
        assert dto.mode == "add_samples"
        assert dto.last_rejection_reason == "no_face"
        assert dto.is_viable is True
        assert dto.to_dict()["samples_collected"] == 3

    def test_enrollment_result_dto_from_result(self) -> None:
        """Test constructing EnrollmentResultDTO from EnrollmentResult."""
        res = EnrollmentResult(
            session_id="esess-1",
            enrollment_batch_id="batch-1",
            person_id="p-1",
            display_name="Eve",
            mode=EnrollmentMode.ADD_SAMPLES,
            is_success=True,
            state=EnrollmentState.COMPLETED,
            samples_enrolled=5,
            persisted_embedding_ids=["emb-1", "emb-2", "emb-3", "emb-4", "emb-5"],
            total_attempts=8,
            error_message=None,
            rejection_counts={},
        )
        dto = EnrollmentResultDTO.from_result(res)
        assert dto.is_success is True
        assert dto.persisted_embedding_count == 5
        # Privacy: list of raw embedding IDs or vector blobs is not leaked directly
        assert dto.to_dict()["persisted_embedding_count"] == 5


class TestRuntimeMetricsAndStepResultDTOs:
    def test_runtime_metrics_and_step_result_dtos(self) -> None:
        """Test RuntimeMetricsDTO and StepResultDTO serialization."""
        metrics = RuntimeMetrics(frames_processed=100, current_fps=29.9, average_latency_ms=12.5)
        dto = RuntimeMetricsDTO.from_metrics(metrics)
        assert dto.frames_processed == 100
        assert dto.current_fps == 29.9

        ctx_dto = InteractionContextDTO(session_id="sess-1", state="engaged")
        ev = SessionStartedIntegrationEvent(session_id="sess-1")
        step_res = StepResultDTO(context=ctx_dto, events=[ev])

        assert step_res.context.session_id == "sess-1"
        assert len(step_res.events) == 1
        d = step_res.to_dict()
        assert d["context"]["session_id"] == "sess-1"
        assert d["events"][0]["session_id"] == "sess-1"

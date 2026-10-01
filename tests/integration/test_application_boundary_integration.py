"""End-to-end integration tests for the Application Integration Boundary and DTO contracts."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from dh360_hi.application.config import HumanInteractionConfig
from dh360_hi.application.factory import create_interaction_service
from dh360_hi.application.health import SubsystemHealthStatus
from dh360_hi.backend.models import (
    ConsentStatus,
    PersonStatus,
    SessionQueryFilter,
)
from dh360_hi.contracts.dto import (
    CreatePersonCommand,
    EnrollmentProgressDTO,
    EnrollmentResultDTO,
    InteractionContextDTO,
    InteractionSessionDTO,
    PersonEngagementStatsDTO,
    PersonProfileDTO,
    SessionAggregateAnalyticsDTO,
    StepResultDTO,
    UpdatePersonCommand,
)
from dh360_hi.contracts.integration import (
    BaseIntegrationEvent,
)
from dh360_hi.vision.aligner import AlignmentResult
from dh360_hi.vision.detector import (
    BaseFaceDetector,
    DetectionResult,
    FaceDetection,
    FaceLandmarks,
)
from dh360_hi.vision.embedder import BaseFaceEmbedder, EmbeddingResult
from dh360_hi.vision.enrollment import EnrollmentConfig
from dh360_hi.vision.frame import Frame, SyntheticFrameSource
from dh360_hi.vision.quality_gate import FaceQualityGate, FaceQualityThresholds


class DummyDetector(BaseFaceDetector):
    def __init__(self, detections: list[FaceDetection] | None = None) -> None:
        self._detections = detections or []
        self._input_size = (320, 320)
        self._confidence_threshold = 0.6

    @property
    def input_size(self) -> tuple[int, int]:
        return self._input_size

    @input_size.setter
    def input_size(self, size: tuple[int, int]) -> None:
        self._input_size = size

    @property
    def confidence_threshold(self) -> float:
        return self._confidence_threshold

    @confidence_threshold.setter
    def confidence_threshold(self, threshold: float) -> None:
        self._confidence_threshold = threshold

    def detect(self, frame: Frame | np.ndarray, frame_id: int = 0) -> DetectionResult:
        f_id = frame.frame_id if isinstance(frame, Frame) else frame_id
        return DetectionResult(
            detections=list(self._detections),
            frame_id=f_id,
            latency_ms=1.0,
            input_size=self._input_size,
        )

    def set_detections(self, detections: list[FaceDetection]) -> None:
        self._detections = list(detections)


class DummyEmbedder(BaseFaceEmbedder):
    def __init__(self, _embedding_vector: np.ndarray | None = None) -> None:
        self._count = 0

    @property
    def dimension(self) -> int:
        return 128

    @property
    def input_size(self) -> tuple[int, int]:
        return (112, 112)

    @property
    def model_name(self) -> str:
        return "SFace"

    @property
    def model_version(self) -> str:
        return "2021dec"

    def embed(self, _aligned_face: np.ndarray | AlignmentResult) -> EmbeddingResult:
        self._count += 1
        v = np.zeros(128, dtype=np.float32)
        v[self._count % 128] = 1.0
        return EmbeddingResult(
            embedding=v,
            model_name=self.model_name,
            model_version=self.model_version,
            dimension=self.dimension,
            is_normalized=True,
            latency_ms=1.0,
        )


@pytest.fixture
def temp_db_path() -> Path:
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir) / "test_boundary.db"


class TestApplicationBoundaryIntegration:
    def test_full_application_lifecycle_and_boundary(self, temp_db_path: Path) -> None:
        """Verify full interaction lifecycle through the transport-neutral service boundary."""
        quality_thresh = FaceQualityThresholds(min_sharpness=0.0)
        enroll_cfg = EnrollmentConfig(
            min_sample_interval_seconds=0.0,
            target_samples=3,
            minimum_samples=2,
        )
        cfg = HumanInteractionConfig(
            db_path=temp_db_path,
            auto_init_db=True,
            model_name="SFace",
            model_version="2021dec",
            quality=quality_thresh,
            enrollment=enroll_cfg,
        )

        detector = DummyDetector()
        embedder = DummyEmbedder()
        qg = FaceQualityGate(quality_thresh)

        service = create_interaction_service(
            config=cfg,
            detector=detector,
            embedder=embedder,
            quality_gate=qg,
            frame_source=SyntheticFrameSource(),
        )

        try:
            # 1. Person Creation Command
            create_cmd = CreatePersonCommand(
                display_name="Dr. Jane Doe",
                preferred_name="Jane",
                status=PersonStatus.ACTIVE,
                consent_status=ConsentStatus.GRANTED,
                notes="Primary medical practitioner",
            )
            person_dto = service.create_person(create_cmd)
            assert isinstance(person_dto, PersonProfileDTO)
            assert person_dto.display_name == "Dr. Jane Doe"
            assert person_dto.preferred_name == "Jane"
            assert person_dto.status == "active"
            assert person_dto.consent_status == "granted"
            # Invariant: notes are excluded from DTO
            assert "notes" not in person_dto.to_dict()

            # 2. Person Query
            fetched_person = service.get_person(person_dto.person_id)
            assert fetched_person is not None
            assert fetched_person.person_id == person_dto.person_id
            assert fetched_person.display_name == "Dr. Jane Doe"

            persons_list = service.list_persons(status=PersonStatus.ACTIVE)
            assert len(persons_list) == 1
            assert persons_list[0].person_id == person_dto.person_id

            # 3. Person Update Command
            update_cmd = UpdatePersonCommand(preferred_name="Janie")
            updated_person = service.update_person(person_dto.person_id, update_cmd)
            assert updated_person.preferred_name == "Janie"

            # 4. Face Enrollment Workflow via Facade
            progress = service.start_enrollment(
                person_id=person_dto.person_id,
                mode="add_samples",
            )
            assert isinstance(progress, EnrollmentProgressDTO)
            assert progress.person_id == person_dto.person_id

            # Feed a valid face detection for enrollment
            dummy_face = FaceDetection(
                bbox=(50.0, 50.0, 150.0, 150.0),
                confidence=0.99,
                landmarks=FaceLandmarks(
                    right_eye=(70.0, 80.0),
                    left_eye=(130.0, 80.0),
                    nose_tip=(100.0, 110.0),
                    right_mouth_corner=(80.0, 140.0),
                    left_mouth_corner=(120.0, 140.0),
                ),
            )
            detector.set_detections([dummy_face])

            frame_img = np.zeros((480, 640, 3), dtype=np.uint8)
            for _ in range(5):
                progress = service.process_enrollment_frame(frame_img)

            assert progress.is_viable is True

            # Commit enrollment
            commit_result = service.commit_enrollment()
            assert isinstance(commit_result, EnrollmentResultDTO)
            assert commit_result.is_success is True
            assert commit_result.persisted_embedding_count > 0

            # 5. Integration Event Listener Registration
            received_integration_events: list[BaseIntegrationEvent] = []
            listener_id = service.add_integration_event_listener(
                lambda ev: received_integration_events.append(ev)
            )

            # 6. Step Perception & Event Bridge
            # Step with face present to initiate presence and session
            step_res = service.step_once()
            assert isinstance(step_res, StepResultDTO)
            assert isinstance(step_res.context, InteractionContextDTO)

            # Advance steps to confirm presence debouncing and session start
            for _ in range(5):
                service.step_once()

            # Flush session persistence
            assert service.flush_session_persistence(timeout=2.0) is True

            # 7. Historical Session Query and Analytics via DTOs
            sessions = service.query_sessions(SessionQueryFilter())
            assert isinstance(sessions, list)
            for s in sessions:
                assert isinstance(s, InteractionSessionDTO)

            analytics = service.get_aggregate_analytics(SessionQueryFilter())
            assert isinstance(analytics, SessionAggregateAnalyticsDTO)

            stats = service.get_person_engagement_stats(person_dto.person_id)
            if stats is not None:
                assert isinstance(stats, PersonEngagementStatsDTO)
                assert stats.person_id == person_dto.person_id

            # 8. Health Diagnostics
            health = service.get_health_snapshot()
            assert health.status in (SubsystemHealthStatus.HEALTHY, SubsystemHealthStatus.DEGRADED)
            assert "runtime" in health.subsystems
            assert "database" in health.subsystems
            assert "persistence_worker" in health.subsystems
            assert "recognizer" in health.subsystems
            assert "enrollment" in health.subsystems
            assert health.subsystems["database"].status == SubsystemHealthStatus.HEALTHY

            # Clean unregistration
            assert service.remove_integration_event_listener(listener_id) is True

        finally:
            service.stop()

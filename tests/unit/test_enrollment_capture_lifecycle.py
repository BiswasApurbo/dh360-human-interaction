"""
Regression tests for the enrollment capture lifecycle in HumanInteractionService.
These tests verify that the webcam capture source is opened, released on errors, and
released after successful commit or cancellation.
"""

from unittest.mock import MagicMock

import pytest

from dh360_hi.application.service import HumanInteractionService, ServiceUnavailableError
from dh360_hi.contracts.dto import EnrollmentProgressDTO
from dh360_hi.vision.enrollment import (
    EnrollmentMode,
    EnrollmentResult,
    EnrollmentState,
)
from dh360_hi.vision.frame import Frame


# Minimal stub controller that mimics FaceEnrollmentController behavior
class StubController:
    def __init__(self):
        self.started = False
        self.committed = False
        self.cancelled = False
        self.processed = False

    def start_session(self, *_, **__):
        self.started = True
        # Return dummy progress DTO
        return EnrollmentProgressDTO(
            session_id="s-1",
            enrollment_batch_id="b-1",
            person_id="p-1",
            display_name="Test",
            mode="add_samples",
            state="collecting",
            samples_collected=0,
            target_samples=5,
            minimum_samples=3,
            total_attempts=0,
            is_viable=False,
            is_complete=False,
        )

    def process_frame(self, frame: Frame):
        assert isinstance(frame, Frame)
        self.processed = True
        # Return same dummy progress (could be mutated if needed)
        return EnrollmentProgressDTO(
            session_id="s-1",
            enrollment_batch_id="b-1",
            person_id="p-1",
            display_name="Test",
            mode="add_samples",
            state="collecting",
            samples_collected=1,
            target_samples=5,
            minimum_samples=3,
            total_attempts=1,
            is_viable=False,
            is_complete=False,
        )

    def commit_session(self, *_, **__):
        self.committed = True
        return EnrollmentResult(
            session_id="s-1",
            enrollment_batch_id="b-1",
            person_id="p-1",
            display_name="Test",
            mode=EnrollmentMode.ADD_SAMPLES,
            is_success=True,
            state=EnrollmentState.COMPLETED,
            samples_enrolled=1,
            persisted_embedding_ids=["emb-1"],
            total_attempts=1,
            error_message=None,
            rejection_counts={},
        )

    def cancel_session(self, *_, **__):
        self.cancelled = True
        return True


# Mock capture source that records method calls
class MockCapture:
    def __init__(self, read_result: Frame | None = None, raise_on_open: Exception | None = None):
        self.open_called = False
        self.read_called = False
        self.release_called = False
        self.read_result = read_result
        self.raise_on_open = raise_on_open

    def open(self) -> None:
        self.open_called = True
        if self.raise_on_open:
            raise self.raise_on_open

    def read(self) -> Frame | None:
        self.read_called = True
        return self.read_result

    def release(self) -> None:
        self.release_called = True

    @property
    def is_opened(self) -> bool:
        return self.open_called and not self.release_called


@pytest.fixture
def service() -> HumanInteractionService:
    """Construct a minimal service with mocked dependencies for lifecycle tests."""
    coordinator = MagicMock()
    coordinator.close = MagicMock()
    person_service = MagicMock()
    repository = MagicMock()
    recognizer = MagicMock()
    db_manager = MagicMock()
    event_bus = MagicMock()
    event_bus.subscribe.return_value = 1
    cfg = MagicMock()
    cfg.camera_index = 0
    return HumanInteractionService(
        coordinator=coordinator,
        person_service=person_service,
        repository=repository,
        recognizer=recognizer,
        db_manager=db_manager,
        event_bus=event_bus,
        config=cfg,
    )


def test_start_enrollment_opens_capture(service: HumanInteractionService):
    frame = Frame(image=MagicMock(shape=(480, 640, 3)), frame_id=0, timestamp=0.0)
    capture = MockCapture(read_result=frame)
    service._enrollment_controller = StubController()
    progress = service.start_enrollment(person_id="p-1", mode="add_samples", capture_source=capture)
    assert capture.open_called, "Capture open should be called on start_enrollment"
    assert progress.state == "collecting"


def test_step_enrollment_successful_keeps_capture_open(service: HumanInteractionService):
    frame = Frame(image=MagicMock(shape=(480, 640, 3)), frame_id=0, timestamp=0.0)
    capture = MockCapture(read_result=frame)
    service._enrollment_controller = StubController()
    service.start_enrollment(person_id="p-1", mode="add_samples", capture_source=capture)
    progress = service.step_enrollment()
    assert capture.read_called, "Capture read should be called"
    assert not capture.release_called, "Capture should remain open after successful step"
    assert progress.samples_collected == 1


def test_step_enrollment_failure_releases_capture(service: HumanInteractionService):
    capture = MockCapture(read_result=None)  # Simulate read failure
    service._enrollment_controller = StubController()
    service.start_enrollment(person_id="p-1", mode="add_samples", capture_source=capture)
    with pytest.raises(ServiceUnavailableError):
        service.step_enrollment()
    assert capture.release_called, "Capture must be released on step failure"


def test_commit_enrollment_releases_capture(service: HumanInteractionService):
    frame = Frame(image=MagicMock(shape=(480, 640, 3)), frame_id=0, timestamp=0.0)
    capture = MockCapture(read_result=frame)
    service._enrollment_controller = StubController()
    service.start_enrollment(person_id="p-1", mode="add_samples", capture_source=capture)
    service.step_enrollment()
    result = service.commit_enrollment()
    assert result.is_success
    assert capture.release_called, "Capture must be released after commit"


def test_cancel_enrollment_releases_capture(service: HumanInteractionService):
    capture = MockCapture(read_result=None)
    service._enrollment_controller = StubController()
    service.start_enrollment(person_id="p-1", mode="add_samples", capture_source=capture)
    service.cancel_enrollment(reason="test")
    assert capture.release_called, "Capture must be released after cancel"


class FailingCommitController(StubController):
    """Controller whose commit_session always raises."""

    def commit_session(self, *_, **__):
        self.committed = False
        raise RuntimeError("Simulated DB failure")


def test_commit_enrollment_failure_releases_capture(service: HumanInteractionService):
    """If commit_session raises, the capture must still be released."""
    capture = MockCapture(read_result=None)
    service._enrollment_controller = FailingCommitController()
    service.start_enrollment(person_id="p-1", mode="add_samples", capture_source=capture)
    with pytest.raises(ServiceUnavailableError):
        service.commit_enrollment()
    assert capture.release_called, "Capture must be released even when commit fails"


def test_service_stop_releases_capture(service: HumanInteractionService):
    """Stopping the service must release any active enrollment capture."""
    capture = MockCapture(read_result=None)
    service._enrollment_controller = StubController()
    service.start_enrollment(person_id="p-1", mode="add_samples", capture_source=capture)
    assert not capture.release_called
    service.stop(timeout=0.1)
    assert capture.release_called, "Capture must be released on service stop"

"""Scenario runner executing deterministic multi-frame interaction simulations."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from dh360_hi.application.config import HumanInteractionConfig
from dh360_hi.application.factory import create_interaction_service
from dh360_hi.backend.database import DatabaseManager
from dh360_hi.backend.gallery_loader import load_gallery_from_repository
from dh360_hi.backend.models import ConsentStatus, PersonStatus
from dh360_hi.backend.person_service import PersonService
from dh360_hi.backend.repository import SQLitePersonRepository
from dh360_hi.backend.session_repository import (
    SessionPersistenceWorker,
    SQLiteSessionRepository,
)
from dh360_hi.edge.session_manager import SessionConfig
from dh360_hi.testing.assertions import assert_scenario_events
from dh360_hi.testing.contracts import (
    ScenarioDefinition,
    ScenarioResult,
)
from dh360_hi.testing.event_collector import ScenarioEventCollector
from dh360_hi.testing.loader import load_scenario_from_json
from dh360_hi.testing.scripted_components import (
    DirectEmbeddingWorker,
    ScriptedFaceDetector,
    ScriptedFaceEmbedder,
    ScriptedFrameSource,
    make_deterministic_embedding,
)
from dh360_hi.vision.identity_state import IdentityStateConfig
from dh360_hi.vision.presence_debouncer import PresenceDebounceConfig
from dh360_hi.vision.quality_gate import FaceQualityThresholds
from dh360_hi.vision.recognition import FaceRecognizer, RecognitionConfig

logger = logging.getLogger(__name__)


def build_scenario_config(
    overrides: dict[str, Any],
    db_path: str | Path = ":memory:",
) -> HumanInteractionConfig:
    """Build a HumanInteractionConfig instance applying scenario overrides."""
    enter_frames = overrides.get("presence_enter_frames", 3)
    leave_frames = overrides.get("presence_exit_frames", 5)
    presence_cfg = PresenceDebounceConfig(
        enter_patience_frames=enter_frames,
        leave_patience_frames=leave_frames,
    )

    req_presence = overrides.get("require_confirmed_presence", True)
    session_cfg = SessionConfig(require_confirmed_presence=req_presence)

    min_obs = overrides.get("identity_min_observations", 2)
    reverification = overrides.get("reverification_interval_frames", 3)
    identity_cfg = IdentityStateConfig(
        min_observations_to_decide=min_obs,
        reverification_interval_frames=reverification,
    )

    min_sharpness = overrides.get("min_sharpness", 40.0)
    quality_cfg = FaceQualityThresholds(min_sharpness=min_sharpness)

    rec_cfg = RecognitionConfig(similarity_threshold=0.60)

    tracker_min_hits = overrides.get("tracker_min_hits", 1)
    tracker_max_missed = overrides.get("tracker_max_missed_frames", 15)
    tracker_iou = overrides.get("tracker_iou_threshold", 0.3)

    return HumanInteractionConfig(
        db_path=db_path,
        auto_init_db=True,
        tracker_min_hits=tracker_min_hits,
        tracker_max_missed_frames=tracker_max_missed,
        tracker_iou_threshold=tracker_iou,
        presence=presence_cfg,
        session=session_cfg,
        identity=identity_cfg,
        quality=quality_cfg,
        recognition=rec_cfg,
    )


class ScenarioRunner:
    """Deterministic scenario execution engine for HumanInteractionService."""

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        auto_assert_events: bool = True,
    ) -> None:
        self._db_path = db_path
        self._auto_assert_events = auto_assert_events

    def run(
        self,
        scenario_or_path: ScenarioDefinition | str | Path | dict[str, Any],
    ) -> ScenarioResult:
        """Execute a deterministic scenario from start to finish.

        Args:
            scenario_or_path: ScenarioDefinition or path to JSON scenario file.

        Returns:
            ScenarioResult containing collected events, persisted count, and status.
        """
        if isinstance(scenario_or_path, ScenarioDefinition):
            scenario = scenario_or_path
        else:
            scenario = load_scenario_from_json(scenario_or_path)

        # 1. Prepare configuration and persistence layer
        cfg = build_scenario_config(scenario.config_overrides, db_path=self._db_path)
        db = DatabaseManager(cfg.db_path, auto_init=True)
        person_repo = SQLitePersonRepository(db)
        person_svc = PersonService(person_repo)
        session_repo = SQLiteSessionRepository(db)
        pers_worker = SessionPersistenceWorker(session_repo)

        # 2. Pre-populate initial registered persons and gallery embeddings
        embedder = ScriptedFaceEmbedder(
            dimension=128,
            model_name=cfg.model_name,
            model_version=cfg.model_version,
        )
        for idx, person_spec in enumerate(scenario.initial_persons, start=1):
            emb_idx = (
                person_spec.embedding_index if person_spec.embedding_index is not None else idx
            )
            emb_vec = make_deterministic_embedding(emb_idx, dimension=128)

            status = (
                PersonStatus(person_spec.status)
                if hasattr(PersonStatus, person_spec.status.upper())
                else PersonStatus.ACTIVE
            )
            consent = (
                ConsentStatus(person_spec.consent_status)
                if hasattr(ConsentStatus, person_spec.consent_status.upper())
                else ConsentStatus.GRANTED
            )

            person_svc.create_person(
                person_id=person_spec.person_id,
                display_name=person_spec.display_name,
                preferred_name=person_spec.preferred_name,
                status=status,
                consent_status=consent,
                notes=person_spec.notes,
            )
            person_svc.add_person_embedding(
                person_id=person_spec.person_id,
                embedding=emb_vec,
                model_name=cfg.model_name,
                model_version=cfg.model_version,
            )
            embedder.register_label(person_spec.person_id, emb_idx)
            embedder.register_label(person_spec.display_name, emb_idx)

        # 3. Initialize Recognition Gallery from Database
        gallery = load_gallery_from_repository(
            repository=person_repo,
            model_name=cfg.model_name,
            model_version=cfg.model_version,
        )
        recognizer = FaceRecognizer(
            gallery=gallery,
            config=cfg.recognition,
            model_name=cfg.model_name,
            model_version=cfg.model_version,
        )

        # 4. Construct Scripted Vision Components
        detector = ScriptedFaceDetector(
            frames=scenario.frames,
            input_size=cfg.detector_input_size,
        )
        frame_source = ScriptedFrameSource(frames=scenario.frames)
        direct_worker = DirectEmbeddingWorker(
            embedder=embedder,
            recognizer=recognizer,
            detector=detector,
        )

        # 5. Build HumanInteractionService using standard DI composition root
        service = create_interaction_service(
            config=cfg,
            frame_source=frame_source,
            detector=detector,
            embedder=embedder,
            recognizer=recognizer,
            worker=direct_worker,
            db_manager=db,
            repository=person_repo,
            person_service=person_svc,
            session_repository=session_repo,
            persistence_worker=pers_worker,
        )

        # 6. Attach Event Collector
        collector = ScenarioEventCollector()
        service.add_integration_event_listener(collector.on_integration_event)

        # 7. Step through all frames deterministically
        frames_processed = 0
        error_msg: str | None = None
        try:
            for _ in scenario.frames:
                service.step_once()
                frames_processed += 1
        except Exception as exc:
            logger.exception("Error during scenario execution")
            error_msg = str(exc)

        # 8. Flush session persistence worker
        service.flush_session_persistence(timeout=2.0)

        # 9. Query results
        collected_events = collector.get_integration_events()
        persisted_sessions = service.list_sessions(limit=100)
        persisted_count = len(persisted_sessions)

        # 10. Assert expectations if configured
        if error_msg is None and self._auto_assert_events and scenario.expected_events:
            try:
                assert_scenario_events(collected_events, scenario.expected_events)
                if (
                    scenario.expected_persisted_sessions is not None
                    and persisted_count != scenario.expected_persisted_sessions
                ):
                    msg = (
                        f"Persisted session count mismatch: expected "
                        f"{scenario.expected_persisted_sessions}, got {persisted_count}"
                    )
                    raise AssertionError(msg)
            except AssertionError as exc:
                error_msg = str(exc)

        # 11. Clean shutdown
        try:
            service.stop(timeout=1.0)
        except Exception:
            logger.exception("Error stopping service during scenario cleanup")

        is_success = error_msg is None
        return ScenarioResult(
            scenario_name=scenario.scenario_name,
            total_frames_processed=frames_processed,
            collected_integration_events=collected_events,
            persisted_sessions_count=persisted_count,
            is_success=is_success,
            error_message=error_msg,
        )

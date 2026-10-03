"""Composition root and factory for constructing the Human Interaction subsystem."""

from __future__ import annotations

import logging

from dh360_hi.application.config import HumanInteractionConfig
from dh360_hi.application.coordinator import InteractionCoordinator
from dh360_hi.application.service import HumanInteractionService
from dh360_hi.backend.database import DatabaseManager
from dh360_hi.backend.gallery_loader import load_gallery_from_repository
from dh360_hi.backend.person_service import BasePersonService, PersonService
from dh360_hi.backend.repository import BasePersonRepository, SQLitePersonRepository
from dh360_hi.backend.session_repository import (
    BaseSessionRepository,
    SessionPersistenceWorker,
    SQLiteSessionRepository,
)
from dh360_hi.edge.event_bus import EventBus
from dh360_hi.edge.runtime import EdgeInteractionRuntime
from dh360_hi.edge.session_manager import SessionManager
from dh360_hi.vision.aligner import BaseFaceAligner, FaceAligner
from dh360_hi.vision.detector import BaseFaceDetector, YuNetFaceDetector
from dh360_hi.vision.embedder import BaseFaceEmbedder, SFaceEmbedder
from dh360_hi.vision.embedding_worker import (
    BaseEmbeddingWorker,
    BoundedEmbeddingWorker,
)
from dh360_hi.vision.enrollment import FaceEnrollmentController
from dh360_hi.vision.frame import FrameSource, SyntheticFrameSource
from dh360_hi.vision.identity_state import IdentityStateManager
from dh360_hi.vision.presence_debouncer import PresenceDebouncer
from dh360_hi.vision.quality_gate import BaseFaceQualityGate, FaceQualityGate
from dh360_hi.vision.recognition import BaseFaceRecognizer, FaceRecognizer
from dh360_hi.vision.target_selector import BaseTargetSelector, InteractionTargetSelector
from dh360_hi.vision.tracker import BaseFaceTracker, FaceTracker
from dh360_hi.vision.two_rate_pipeline import TwoRatePerceptionPipeline

logger = logging.getLogger(__name__)


def _try_build_enrollment_controller(
    cfg: HumanInteractionConfig,
    person_service: BasePersonService,
    detector: BaseFaceDetector | None = None,
    embedder: BaseFaceEmbedder | None = None,
    aligner: BaseFaceAligner | None = None,
    quality_gate: BaseFaceQualityGate | None = None,
) -> FaceEnrollmentController | None:
    """Best-effort construction of a FaceEnrollmentController for on-demand capture.

    Decouples enrollment capability from the continuous perception runtime: the API
    control plane constructs an enrollment pipeline even when perception is disabled,
    so operator-driven camera enrollment can run on demand. Returns None when the
    required detector/embedder cannot be resolved, letting the service degrade
    gracefully (enrollment endpoints then report 503).
    """
    if embedder is not None:
        emb: BaseFaceEmbedder | None = embedder
    elif cfg.embedder_model_path is not None:
        emb = SFaceEmbedder(
            model_path=cfg.embedder_model_path,
            model_name=cfg.model_name,
            model_version=cfg.model_version,
        )
    else:
        emb = None

    if detector is not None:
        det: BaseFaceDetector | None = detector
    elif cfg.detector_model_path is not None:
        det = YuNetFaceDetector(
            model_path=cfg.detector_model_path,
            input_size=cfg.detector_input_size,
            score_threshold=cfg.detector_score_threshold,
            nms_threshold=cfg.detector_nms_threshold,
            top_k=cfg.detector_top_k,
        )
    else:
        det = None

    if emb is None or det is None:
        return None

    al = aligner or FaceAligner(
        output_size=cfg.aligner_output_size,
        fallback_to_bbox=cfg.aligner_fallback_to_bbox,
    )
    qg = quality_gate or FaceQualityGate(cfg.quality)
    return FaceEnrollmentController(
        person_service=person_service,
        detector=det,
        aligner=al,
        embedder=emb,
        quality_gate=qg,
        config=cfg.enrollment,
    )


def create_interaction_service(
    config: HumanInteractionConfig | None = None,
    frame_source: FrameSource | None = None,
    detector: BaseFaceDetector | None = None,
    embedder: BaseFaceEmbedder | None = None,
    tracker: BaseFaceTracker | None = None,
    quality_gate: BaseFaceQualityGate | None = None,
    aligner: BaseFaceAligner | None = None,
    recognizer: BaseFaceRecognizer | None = None,
    worker: BaseEmbeddingWorker | None = None,
    identity_manager: IdentityStateManager | None = None,
    target_selector: BaseTargetSelector | None = None,
    presence_debouncer: PresenceDebouncer | None = None,
    event_bus: EventBus | None = None,
    session_manager: SessionManager | None = None,
    coordinator: InteractionCoordinator | None = None,
    db_manager: DatabaseManager | None = None,
    repository: BasePersonRepository | None = None,
    person_service: BasePersonService | None = None,
    session_repository: BaseSessionRepository | None = None,
    persistence_worker: SessionPersistenceWorker | None = None,
    pipeline: TwoRatePerceptionPipeline | None = None,
    enable_perception: bool = True,
) -> HumanInteractionService:
    """Composition root: instantiate, wire, and return a complete HumanInteractionService.

    Builds the dependency graph across database storage, face perception,
    edge runtime, session management, and application context coordination.

    Args:
        config: Subsystem configuration (defaults to standard defaults).
        frame_source: Video, webcam, or synthetic frame source (defaults to SyntheticFrameSource).
        detector: Face detector instance (instantiated from config if model path provided).
        embedder: Face embedder instance (instantiated from config if model path provided).
        tracker: Face tracker instance (defaults to FaceTracker).
        quality_gate: Face quality gate instance (defaults to FaceQualityGate).
        aligner: Face crop aligner instance (defaults to FaceAligner).
        recognizer: Face recognizer instance (defaults to FaceRecognizer).
        worker: Background embedding worker instance (defaults to BoundedEmbeddingWorker).
        identity_manager: Identity state manager instance (defaults to IdentityStateManager).
        target_selector: Target selector instance (defaults to InteractionTargetSelector).
        presence_debouncer: Presence debouncer instance (defaults to PresenceDebouncer).
        event_bus: In-process event bus instance (defaults to fresh EventBus).
        session_manager: Session manager instance (defaults to SessionManager).
        coordinator: Context coordinator instance (defaults to InteractionCoordinator).
        db_manager: Database manager instance (defaults to DatabaseManager).
        repository: Person repository instance (defaults to SQLitePersonRepository).
        person_service: Person service instance (defaults to PersonService).
        session_repository: Session repository instance (defaults to SQLiteSessionRepository).
        persistence_worker: Session persistence worker instance (defaults to
            SessionPersistenceWorker).
        pipeline: Fully pre-constructed TwoRatePerceptionPipeline override, if provided.
        enable_perception: Whether to construct and wire the edge perception pipeline and runtime
            (defaults to True). If False, only storage, repository, gallery, coordinator, and
            session components are initialized for offline operations.

    Returns:
        Configured and wired HumanInteractionService instance.

    Raises:
        ValueError: If enable_perception is True and neither a detector instance nor a valid
            detector_model_path is provided when pipeline is not overridden.
    """
    cfg = config or HumanInteractionConfig()

    # 1. Database & Persistence Layer
    db = db_manager or DatabaseManager(cfg.db_path, auto_init=cfg.auto_init_db)
    repo = repository or SQLitePersonRepository(db)
    svc = person_service or PersonService(repo)
    session_repo = session_repository or SQLiteSessionRepository(db)
    pers_worker = persistence_worker or SessionPersistenceWorker(session_repo)
    pers_worker.start()

    # 2. Recognition Gallery & Recognizer
    gallery = load_gallery_from_repository(
        repository=repo,
        model_name=cfg.model_name,
        model_version=cfg.model_version,
        require_active=cfg.require_active_for_gallery,
        require_consent=cfg.require_consent_for_gallery,
    )
    rec = recognizer or FaceRecognizer(
        gallery=gallery,
        config=cfg.recognition,
        model_name=cfg.model_name,
        model_version=cfg.model_version,
    )

    # 3. Event Bus & Application Context Coordinator
    bus = event_bus or EventBus()
    coord = coordinator or InteractionCoordinator(
        event_bus=bus,
        person_service=svc,
        session_repository=session_repo,
        persistence_worker=pers_worker,
    )

    if not enable_perception:
        enrollment_ctrl = _try_build_enrollment_controller(
            cfg=cfg,
            person_service=svc,
            detector=detector,
            embedder=embedder,
        )
        return HumanInteractionService(
            runtime=None,
            coordinator=coord,
            person_service=svc,
            repository=repo,
            recognizer=rec,
            db_manager=db,
            event_bus=bus,
            session_repository=session_repo,
            persistence_worker=pers_worker,
            enrollment_controller=enrollment_ctrl,
            config=cfg,
        )

    # 4. Vision Perception Pipeline
    if pipeline is not None:
        perc_pipeline = pipeline
        enrollment_ctrl = None
    else:
        # Resolve detector
        if detector is not None:
            det = detector
        elif cfg.detector_model_path is not None:
            det = YuNetFaceDetector(
                model_path=cfg.detector_model_path,
                input_size=cfg.detector_input_size,
                score_threshold=cfg.detector_score_threshold,
                nms_threshold=cfg.detector_nms_threshold,
                top_k=cfg.detector_top_k,
            )
        else:
            msg = (
                "Cannot construct perception pipeline: either a 'detector' instance or "
                "'config.detector_model_path' must be provided."
            )
            raise ValueError(msg)

        # Resolve embedder
        emb = embedder
        if emb is None and cfg.embedder_model_path is not None:
            emb = SFaceEmbedder(
                model_path=cfg.embedder_model_path,
                model_name=cfg.model_name,
                model_version=cfg.model_version,
            )

        trk = tracker or FaceTracker(
            iou_threshold=cfg.tracker_iou_threshold,
            max_missed_frames=cfg.tracker_max_missed_frames,
            min_hits=cfg.tracker_min_hits,
            max_center_distance=cfg.tracker_max_center_distance,
            smooth_factor=cfg.tracker_smooth_factor,
        )
        qg = quality_gate or FaceQualityGate(cfg.quality)
        al = aligner or FaceAligner(
            output_size=cfg.aligner_output_size,
            fallback_to_bbox=cfg.aligner_fallback_to_bbox,
        )
        id_mgr = identity_manager or IdentityStateManager(cfg.identity)
        ts = target_selector or InteractionTargetSelector(cfg.target_selector)
        pd = presence_debouncer or PresenceDebouncer(cfg.presence)

        if worker is not None:
            emb_worker = worker
        elif emb is not None:
            emb_worker = BoundedEmbeddingWorker(
                aligner=al,
                embedder=emb,
                recognizer=rec,
                config=cfg.worker,
            )
        else:
            emb_worker = None

        perc_pipeline = TwoRatePerceptionPipeline(
            detector=det,
            embedder=emb,
            tracker=trk,
            quality_gate=qg,
            aligner=al,
            recognizer=rec,
            worker=emb_worker,
            identity_manager=id_mgr,
            target_selector=ts,
            presence_debouncer=pd,
        )

        enrollment_ctrl = None
        if emb is not None:
            enrollment_ctrl = FaceEnrollmentController(
                person_service=svc,
                detector=det,
                aligner=al,
                embedder=emb,
                quality_gate=qg,
                config=cfg.enrollment,
            )

    # 5. Frame Source & Session Supervision
    src = frame_source or SyntheticFrameSource()
    sm = session_manager or SessionManager(event_bus=bus, config=cfg.session)

    # 6. Edge Interaction Runtime
    rt = EdgeInteractionRuntime(
        frame_source=src,
        pipeline=perc_pipeline,
        session_manager=sm,
        event_bus=bus,
        config=cfg.runtime,
    )

    # 7. Human Interaction Application Service
    return HumanInteractionService(
        runtime=rt,
        coordinator=coord,
        person_service=svc,
        repository=repo,
        recognizer=rec,
        db_manager=db,
        event_bus=bus,
        session_repository=session_repo,
        persistence_worker=pers_worker,
        enrollment_controller=enrollment_ctrl,
        config=cfg,
    )

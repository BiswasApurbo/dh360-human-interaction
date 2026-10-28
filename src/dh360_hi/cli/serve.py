"""CLI handler for running unified live perception, authorization, and API server."""

from __future__ import annotations

import argparse
import logging
import os
from typing import Any

from dh360_hi.api.app import create_app
from dh360_hi.application.dialogue_coordinator import DialogueCoordinator
from dh360_hi.application.factory import create_interaction_service
from dh360_hi.application.loader import load_config, resolve_cli_db_path
from dh360_hi.backend.knowledge_service import KnowledgeService
from dh360_hi.cli.run import _build_frame_source
from dh360_hi.contracts.dialogue import MockLLMProvider
from dh360_hi.contracts.errors import ValidationAppError
from dh360_hi.edge.authorization_controller import AuthorizationController
from dh360_hi.integrations.eliva import (
    ElivaGeminiAdapter,
    ElivaGoogleTTSAdapter,
    ElivaSonioxAdapter,
)
from dh360_hi.integrations.eliva.adapters import build_kbcon_system_instruction

logger = logging.getLogger(__name__)


def register_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the 'serve' subcommand and its options."""
    parser = subparsers.add_parser(
        "serve",
        help="Run live edge perception, authorization controller, and API server in one process",
        description="Unified daemon serving API and live perception runtime simultaneously.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind socket host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Bind socket port (default: 8000)",
    )
    parser.add_argument(
        "--source",
        default="synthetic",
        choices=["synthetic", "webcam", "video"],
        help="Frame capture source (default: synthetic)",
    )
    parser.add_argument(
        "--camera-index", type=int, default=0, help="Webcam device index (default: 0)"
    )
    parser.add_argument("--video-file", help="Path to video file (when --source video)")
    parser.add_argument("--video-loop", action="store_true", help="Loop video stream continuously")
    parser.add_argument("--preset", help="Configuration preset (development, testing, production)")
    parser.add_argument("--config-file", help="Path to TOML configuration file")
    parser.add_argument("--db-path", help="Override SQLite database path (default: dh360_hi.db)")
    parser.add_argument("--target-fps", type=float, help="Override target frame rate ceiling")
    parser.add_argument("--detector-model-path", help="Path to YuNet ONNX face detection model")
    parser.add_argument("--embedder-model-path", help="Path to SFace ONNX face embedding model")
    parser.add_argument(
        "--dialogue-mode",
        choices=["auto", "mock", "live", "disabled"],
        default="auto",
        help="Dialogue mode: auto (live/mock fallback), mock, live, or disabled (default: auto)",
    )
    parser.add_argument(
        "--mock-dialogue",
        action="store_true",
        help="Run dialogue in offline mock mode (alias for --dialogue-mode mock)",
    )
    parser.add_argument(
        "--continuous-presence",
        dest="continuous_presence",
        action="store_true",
        default=True,
        help="Require continuous face presence to maintain authorization (default: True)",
    )
    parser.add_argument(
        "--allow-persistent-auth",
        dest="continuous_presence",
        action="store_false",
        help="Allow authorization to persist across temporary face absences",
    )


def handle_serve(args: argparse.Namespace) -> int:
    """Execute unified daemon running live perception and HTTP API."""
    from pathlib import Path

    import uvicorn

    overrides: dict[str, Any] = {}
    cli_db = resolve_cli_db_path(args)
    if cli_db is not None:
        overrides["db_path"] = cli_db
    if getattr(args, "target_fps", None):
        if "runtime" not in overrides:
            overrides["runtime"] = {}
        overrides["runtime"]["target_fps"] = args.target_fps
    if getattr(args, "camera_index", None) is not None:
        overrides["camera_index"] = args.camera_index
    if getattr(args, "detector_model_path", None):
        overrides["detector_model_path"] = args.detector_model_path
    if getattr(args, "embedder_model_path", None):
        overrides["embedder_model_path"] = args.embedder_model_path

    # Default to project's existing models/weights assets if present and not overridden
    default_yunet = Path("models/weights/face_detection_yunet_2023mar.onnx")
    default_sface = Path("models/weights/face_recognition_sface_2021dec.onnx")
    if default_yunet.is_file():
        overrides.setdefault("detector_model_path", str(default_yunet))
    if default_sface.is_file():
        overrides.setdefault("embedder_model_path", str(default_sface))

    config = load_config(
        preset=getattr(args, "preset", None),
        config_file=getattr(args, "config_file", None),
        overrides=overrides if overrides else None,
        validate=True,
    )

    if not config.detector_model_path:
        raise ValidationAppError(
            message="Cannot start serve: 'detector_model_path' is not configured.",
            field="detector_model_path",
        )

    frame_source = _build_frame_source(args)
    service = create_interaction_service(
        config=config, frame_source=frame_source, enable_perception=True
    )

    # Start live perception pipeline
    service.start(async_thread=True)

    # Instantiate single AuthorizationController bound to live session manager
    assert service.runtime is not None
    assert service.runtime.session_manager is not None
    timeout_sec = getattr(config.runtime, "session_timeout_seconds", 30.0)
    auth_ctrl = AuthorizationController(
        session_manager=service.runtime.session_manager,
        timeout_seconds=timeout_sec,
        require_continuous_presence=getattr(args, "continuous_presence", True),
    )

    # Dialogue subsystem composition
    raw_mode = (
        "mock" if getattr(args, "mock_dialogue", False) else getattr(args, "dialogue_mode", "auto")
    )
    dialogue_coord: DialogueCoordinator | None = None
    effective_mode = raw_mode

    if raw_mode != "disabled":
        knowledge_svc = KnowledgeService(person_service=service.person_service)
        has_gemini_key = bool(os.getenv("GEMINI_API_KEY", "").strip())

        if raw_mode == "mock" or (raw_mode == "auto" and not has_gemini_key):
            llm_provider = MockLLMProvider()
            stt_provider = None
            tts_provider = None
            effective_mode = "mock"
        else:
            # Live Eliva adapters (lazy SDK imports; no hardware accessed on init)
            try:
                from dh360_hi.integrations.eliva import gemini, gtts, soniox

                gemini_key = getattr(gemini, "GEMINI_API_KEY", "") or None
                soniox_key = getattr(soniox, "SONIOX_API_KEY", "") or None
                gtts_json = getattr(gtts, "SERVICE_ACCOUNT_JSON", "") or None
            except Exception:
                gemini_key = None
                soniox_key = None
                gtts_json = None

            llm_provider = ElivaGeminiAdapter(
                api_key=gemini_key or os.getenv("GEMINI_API_KEY"),
                system_instruction=build_kbcon_system_instruction(
                    institution_type=knowledge_svc.institution_type,
                    institution_name=knowledge_svc.institution_name,
                ),
            )
            stt_provider = ElivaSonioxAdapter(
                api_key=soniox_key or os.getenv("SONIOX_API_KEY")
            )
            tts_provider = ElivaGoogleTTSAdapter(
                credentials_json=gtts_json or os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON"),
                credentials_file=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
            )
            effective_mode = "live"

        dialogue_coord = DialogueCoordinator(
            event_bus=service.event_bus,
            knowledge_service=knowledge_svc,
            authorization_controller=auth_ctrl,
            llm_provider=llm_provider,
            stt_provider=stt_provider,
            tts_provider=tts_provider,
        )

    app = create_app(
        service=service,
        authorization_controller=auth_ctrl,
        dialogue_coordinator=dialogue_coord,
    )

    host = getattr(args, "host", "127.0.0.1")
    port = getattr(args, "port", 8000)

    try:
        print("=== Digital Health 360 Unified Live Service ===")
        print(f"  API Endpoint : http://{host}:{port}")
        print(f"  Frame Source : {args.source}")
        print(f"  Target FPS   : {config.runtime.target_fps}")
        print(f"  DB Path      : {config.db_path}")
        if dialogue_coord is not None:
            print(f"  Dialogue     : {effective_mode.upper()} (endpoint: /dialogue/turn)")
        else:
            print("  Dialogue     : DISABLED")
        print("Starting unified API & perception server...")

        uvicorn.run(app, host=host, port=port, log_level="warning")
        return 0
    finally:
        # Graceful shutdown of service, dialogue coordinator, and controller
        if dialogue_coord is not None:
            dialogue_coord.close()
        auth_ctrl.close()
        service.stop(timeout=3.0)

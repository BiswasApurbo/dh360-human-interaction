"""CLI handler for running unified live perception, authorization, and API server."""

from __future__ import annotations

import argparse
import logging
from typing import Any

from dh360_hi.api.app import create_app
from dh360_hi.application.factory import create_interaction_service
from dh360_hi.application.loader import load_config
from dh360_hi.cli.run import _build_frame_source
from dh360_hi.contracts.errors import ValidationAppError
from dh360_hi.edge.authorization_controller import AuthorizationController

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
    parser.add_argument("--db-path", help="Override SQLite database path")
    parser.add_argument("--target-fps", type=float, help="Override target frame rate ceiling")


def handle_serve(args: argparse.Namespace) -> int:
    """Execute unified daemon running live perception and HTTP API."""
    import uvicorn

    overrides: dict[str, Any] = {}
    if getattr(args, "db_path", None):
        overrides["db_path"] = args.db_path
    if getattr(args, "target_fps", None):
        if "runtime" not in overrides:
            overrides["runtime"] = {}
        overrides["runtime"]["target_fps"] = args.target_fps

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
    )

    app = create_app(service=service, authorization_controller=auth_ctrl)

    host = getattr(args, "host", "127.0.0.1")
    port = getattr(args, "port", 8000)

    try:
        print("=== Digital Health 360 Unified Live Service ===")
        print(f"  API Endpoint : http://{host}:{port}")
        print(f"  Frame Source : {args.source}")
        print(f"  Target FPS   : {config.runtime.target_fps}")
        print(f"  DB Path      : {config.db_path}")
        print("Starting unified API & perception server...")

        uvicorn.run(app, host=host, port=port, log_level="warning")
        return 0
    finally:
        # Graceful shutdown of service and controller
        auth_ctrl.close()
        service.stop(timeout=3.0)

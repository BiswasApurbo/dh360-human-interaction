"""CLI handler for running the continuous edge interaction service."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from typing import Any

from dh360_hi.application.factory import create_interaction_service
from dh360_hi.application.loader import load_config, resolve_cli_db_path
from dh360_hi.cli.formatting import format_json, format_key_value
from dh360_hi.contracts.errors import ValidationAppError
from dh360_hi.contracts.integration import BaseIntegrationEvent
from dh360_hi.edge.runtime import RuntimeState
from dh360_hi.vision.frame import (
    FrameSource,
    SyntheticFrameSource,
    VideoFileFrameSource,
    WebcamFrameSource,
)

logger = logging.getLogger(__name__)


def register_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the 'run' subcommand and its options."""
    parser = subparsers.add_parser(
        "run",
        help="Launch the continuous edge perception runtime and interaction service",
        description="Starts the edge interaction runtime frame pump and integration event logger.",
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
    parser.add_argument(
        "--duration", type=float, help="Run for a fixed duration in seconds then exit cleanly"
    )
    parser.add_argument(
        "--json-events", action="store_true", help="Output integration events as JSON lines"
    )


def _build_frame_source(args: argparse.Namespace) -> FrameSource:
    """Instantiate appropriate FrameSource based on CLI args."""
    source_type = args.source
    if source_type == "synthetic":
        return SyntheticFrameSource(fps=30.0, num_frames=1000000)
    if source_type == "video":
        if not getattr(args, "video_file", None):
            raise ValidationAppError(
                message="--video-file is required when --source video is selected",
                field="video_file",
            )
        return VideoFileFrameSource(
            file_path=args.video_file, loop=getattr(args, "video_loop", False)
        )
    return WebcamFrameSource(camera_index=getattr(args, "camera_index", 0))


def handle_run(args: argparse.Namespace) -> int:
    """Execute continuous runtime loop with graceful signal handling."""
    overrides: dict[str, Any] = {}
    cli_db = resolve_cli_db_path(args)
    if cli_db is not None:
        overrides["db_path"] = cli_db
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
            message="Cannot start runtime: 'detector_model_path' is not configured.",
            field="detector_model_path",
        )

    frame_source = _build_frame_source(args)
    service = create_interaction_service(
        config=config, frame_source=frame_source, enable_perception=True
    )

    stop_event = threading.Event()

    def _sig_handler(signum: int, _frame: Any) -> None:
        sig_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
        sys.stderr.write(f"\nReceived {sig_name}; initiating graceful shutdown...\n")
        stop_event.set()

    # Register signals on main thread
    old_sigint = signal.signal(signal.SIGINT, _sig_handler)
    old_sigterm = signal.signal(signal.SIGTERM, _sig_handler)

    def _on_integration_event(event: BaseIntegrationEvent) -> None:
        if getattr(args, "json_events", False):
            print(format_json(event.to_dict()))
        else:
            session_id = getattr(event, "session_id", "-")
            details_str = f"session_id={session_id}"
            person_id = getattr(event, "person_id", None)
            if person_id:
                details_str += f", person_id={person_id}"
            ts = event.timestamp_iso[:19]
            ev_type = event.event_type.upper().ljust(25)
            print(f"[{ts}] EVENT: {ev_type} | {details_str}")

    service.add_integration_event_listener(_on_integration_event)

    try:
        print("=== Digital Health 360 Human Interaction Service ===")
        print(f"  Frame Source : {args.source}")
        print(f"  Target FPS   : {config.runtime.target_fps}")
        print(f"  DB Path      : {config.db_path}")
        print("Starting runtime frame pump (Press Ctrl+C to stop)...")

        service.start(async_thread=True)
        start_time = time.monotonic()
        duration = getattr(args, "duration", None)

        while not stop_event.is_set():
            time.sleep(0.2)

            if duration is not None and (time.monotonic() - start_time) >= duration:
                print(f"Reached configured run duration ({duration}s); stopping.")
                break

            if service.state in (RuntimeState.STOPPED, RuntimeState.ERROR):
                break

        return 0

    finally:
        # Restore signals
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)

        print("Stopping HumanInteractionService and releasing resources...")
        service.stop(timeout=3.0)

        metrics = service.get_runtime_metrics()
        pairs = [
            ("Frames Processed", metrics.frames_processed),
            ("Observed Rolling FPS", f"{metrics.current_fps:.1f}"),
            ("Average Latency (ms)", f"{metrics.average_latency_ms:.2f}"),
            ("Total Uptime (s)", f"{metrics.uptime_seconds:.2f}"),
        ]
        print()
        print(format_key_value(pairs, title="Final Runtime Execution Metrics"))
        print("Service stopped successfully.")

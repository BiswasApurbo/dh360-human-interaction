#!/usr/bin/env python3
"""M2.8 Edge Interaction Runtime Benchmark.

Measures:
1. End-to-end edge interaction runtime throughput (FPS).
2. Frame processing and session dispatch latency percentiles.
3. Event bus publication overhead and metrics tracking accuracy.

DISCLAIMER:
Development benchmark — MacBook M4 Air; not representative of Raspberry Pi performance.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

# Add src and project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.download_models import download_all_resources  # noqa: E402
from scripts.generate_test_video import generate_face_animation_video  # noqa: E402

from dh360_hi.contracts.events import BaseEvent  # noqa: E402
from dh360_hi.edge.event_bus import EventBus  # noqa: E402
from dh360_hi.edge.runtime import EdgeInteractionRuntime, EdgeRuntimeConfig  # noqa: E402
from dh360_hi.edge.session_manager import SessionConfig, SessionManager  # noqa: E402
from dh360_hi.vision.aligner import FaceAligner  # noqa: E402
from dh360_hi.vision.benchmark import PipelineBenchmark  # noqa: E402
from dh360_hi.vision.detector import YuNetFaceDetector  # noqa: E402
from dh360_hi.vision.embedder import SFaceEmbedder  # noqa: E402
from dh360_hi.vision.embedding_worker import EmbeddingWorkerConfig  # noqa: E402
from dh360_hi.vision.frame import VideoFileFrameSource  # noqa: E402
from dh360_hi.vision.identity_state import IdentityStateManager  # noqa: E402
from dh360_hi.vision.presence_debouncer import PresenceDebouncer  # noqa: E402
from dh360_hi.vision.quality_gate import FaceQualityGate  # noqa: E402
from dh360_hi.vision.recognition import FaceRecognizer, GalleryEntry, InMemoryGallery  # noqa: E402
from dh360_hi.vision.target_selector import InteractionTargetSelector  # noqa: E402
from dh360_hi.vision.tracker import FaceTracker  # noqa: E402
from dh360_hi.vision.two_rate_pipeline import TwoRatePerceptionPipeline  # noqa: E402

MODELS_DIR = PROJECT_ROOT / "models" / "weights"
YUNET_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_PATH = MODELS_DIR / "face_recognition_sface_2021dec.onnx"
FIXTURE_PATH = PROJECT_ROOT / "tests" / "fixtures" / "sample_face.jpg"
VIDEO_PATH = PROJECT_ROOT / "data" / "face_recognition_benchmark.mp4"

MAC_DISCLAIMER = (
    "Development benchmark — MacBook M4 Air; not representative of Raspberry Pi performance."
)


def run_runtime_benchmark(
    num_frames: int = 150,
    device_label: str = "MacBook M4 Air",
) -> int:
    """Run benchmark across full edge interaction runtime."""
    download_all_resources()

    if not VIDEO_PATH.exists():
        VIDEO_PATH.parent.mkdir(parents=True, exist_ok=True)
        generate_face_animation_video(str(FIXTURE_PATH), str(VIDEO_PATH), num_frames=120)

    # 1. Initialize vision components
    detector = YuNetFaceDetector(model_path=str(YUNET_PATH))
    tracker = FaceTracker()
    quality_gate = FaceQualityGate()
    aligner = FaceAligner()
    embedder = SFaceEmbedder(model_path=str(SFACE_PATH))

    # Pre-populate single enrolled identity
    fixture_img = cv2.imread(str(FIXTURE_PATH))
    det_res = detector.detect(fixture_img)
    gallery_entries: list[GalleryEntry] = []
    if det_res.detections:
        align_res = aligner.align(fixture_img, det_res.detections[0])
        if align_res.is_success:
            emb_res = embedder.embed(align_res.aligned_face)
            entry = GalleryEntry(
                person_id="p-001",
                display_name="Benchmark Subject",
                embeddings=[emb_res.embedding],
                model_name="SFace",
                model_version="2021dec",
            )
            gallery_entries.append(entry)
    gallery = InMemoryGallery(entries=gallery_entries)
    recognizer = FaceRecognizer(gallery=gallery)

    identity_manager = IdentityStateManager()
    target_selector = InteractionTargetSelector()
    presence_debouncer = PresenceDebouncer()

    pipeline = TwoRatePerceptionPipeline(
        detector=detector,
        tracker=tracker,
        quality_gate=quality_gate,
        aligner=aligner,
        embedder=embedder,
        recognizer=recognizer,
        identity_manager=identity_manager,
        target_selector=target_selector,
        presence_debouncer=presence_debouncer,
        worker_config=EmbeddingWorkerConfig(max_queue_size=1),
    )

    # 2. Initialize edge session and event bus
    bus = EventBus()
    events_published: list[BaseEvent] = []
    bus.subscribe(BaseEvent, events_published.append)

    session_manager = SessionManager(
        event_bus=bus,
        config=SessionConfig(require_confirmed_presence=True),
    )

    frame_source = VideoFileFrameSource(file_path=VIDEO_PATH, loop=True)
    config = EdgeRuntimeConfig(target_fps=1000.0)  # Unthrottled for max throughput profiling

    runtime = EdgeInteractionRuntime(
        frame_source=frame_source,
        pipeline=pipeline,
        session_manager=session_manager,
        event_bus=bus,
        config=config,
    )

    bench = PipelineBenchmark()
    bench.start()

    print("\n" + "=" * 78)
    print("M2.8 Edge Interaction Runtime & Continuous Frame Pump Benchmark")
    print(f"Platform: {device_label}")
    print(f"Notice:   {MAC_DISCLAIMER}")
    print("=" * 78)

    # Step through benchmark frames
    for i in range(num_frames):
        t0 = time.perf_counter()
        frame_res, emitted_events = runtime.step_once()
        t1 = time.perf_counter()
        frame_duration_ms = (t1 - t0) * 1000.0
        bench.record_frame(
            total_latency_ms=frame_duration_ms,
            detections_count=len(frame_res.detections),
        )

        if (i + 1) % 30 == 0 or (i + 1) == num_frames:
            fps = bench.rolling_fps
            print(
                f"  Frame {i + 1:3d}/{num_frames}: "
                f"Latency = {frame_duration_ms:5.2f} ms | Rolling FPS = {fps:5.1f}"
            )

    runtime.stop()

    print("\n" + bench.format_summary(device_label=device_label))
    print("-" * 78)
    print(f"Events Dispatched to Bus: {len(events_published)}")
    print(f"Runtime Telemetry FPS:     {runtime.metrics.current_fps:.2f} FPS")
    print(f"Runtime Telemetry Frames:  {runtime.metrics.frames_processed}")
    print(f"Runtime Telemetry Latency: {runtime.metrics.average_latency_ms:.2f} ms")
    print("-" * 78)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M2.8 Edge Interaction Runtime Benchmark")
    parser.add_argument("--frames", type=int, default=100, help="Number of benchmark frames")
    args = parser.parse_args()
    sys.exit(run_runtime_benchmark(num_frames=args.frames))

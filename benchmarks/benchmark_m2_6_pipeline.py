#!/usr/bin/env python3
"""M2.6 Two-Rate Vision Perception Pipeline Benchmark.

Measures:
1. Fast-rate perception loop throughput (FPS) and latency percentiles.
2. Asynchronous embedding worker inference latency and queue depth.
3. Target selection and presence debouncing overhead.

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


def run_two_rate_benchmark(
    num_frames: int = 150,
    device_label: str = "MacBook M4 Air",
) -> int:
    """Run benchmark across two-rate perception pipeline."""
    download_all_resources()

    generate_face_animation_video(
        face_image_path=FIXTURE_PATH,
        output_path=VIDEO_PATH,
        target_width=640,
        target_height=480,
        fps=30.0,
        num_frames=num_frames,
    )

    detector = YuNetFaceDetector(model_path=YUNET_PATH, input_size=(640, 480))
    tracker = FaceTracker()
    quality_gate = FaceQualityGate()
    aligner = FaceAligner()
    embedder = SFaceEmbedder(model_path=SFACE_PATH)

    # Pre-register fixture identity into gallery
    fixture_img = cv2.imread(str(FIXTURE_PATH))
    det_res = detector.detect(fixture_img)
    gallery_entries: list[GalleryEntry] = []
    if det_res.detections:
        align_res = aligner.align(fixture_img, det_res.detections[0])
        if align_res.is_success:
            emb_res = embedder.embed(align_res.aligned_face)
            entry = GalleryEntry(
                person_id="subject-001",
                display_name="Dr. Alex Mercer",
                embeddings=[emb_res.embedding],
                model_name="SFace",
                model_version="2021dec",
            )
            gallery_entries.append(entry)

    recognizer = FaceRecognizer(gallery=InMemoryGallery(gallery_entries))

    pipeline = TwoRatePerceptionPipeline(
        detector=detector,
        embedder=embedder,
        tracker=tracker,
        quality_gate=quality_gate,
        aligner=aligner,
        recognizer=recognizer,
        identity_manager=IdentityStateManager(),
        target_selector=InteractionTargetSelector(),
        presence_debouncer=PresenceDebouncer(),
        worker_config=EmbeddingWorkerConfig(max_queue_size=4),
    )

    bench = PipelineBenchmark()

    frame_source = VideoFileFrameSource(VIDEO_PATH, loop=False)
    frame_source.open()
    pipeline.start()
    bench.start()

    worker_latencies_ms: list[float] = []
    queue_latencies_ms: list[float] = []
    queue_depths: list[int] = []

    print("=" * 70)
    print(f"Starting Two-Rate Perception Benchmark on {device_label}")
    print(f"Input Video: {VIDEO_PATH} ({num_frames} frames)")
    print("=" * 70)

    try:
        frame_idx = 0
        while frame_idx < num_frames:
            frame = frame_source.read()
            if frame is None:
                break

            t0 = time.perf_counter()
            result = pipeline.process_frame(frame)
            frame_duration_ms = (time.perf_counter() - t0) * 1000.0

            bench.record_frame(
                total_latency_ms=frame_duration_ms,
                detections_count=len(result.detections),
            )

            queue_depths.append(result.pending_worker_tasks)
            for rec in result.completed_recognitions:
                worker_latencies_ms.append(rec.inference_latency_ms)
                queue_latencies_ms.append(rec.queue_latency_ms)

            frame_idx += 1
            # Simulate real camera interval (~30 FPS: ~33ms)
            time.sleep(0.01)

    finally:
        pipeline.stop(timeout=2.0)
        frame_source.release()

    print("\n" + bench.format_summary(device_label=device_label))
    print("\nAsynchronous Embedding Worker Metrics:")
    if worker_latencies_ms:
        import numpy as np

        print(f"  • Completed Recognitions : {len(worker_latencies_ms)}")
        print(f"  • Worker Inference Mean  : {np.mean(worker_latencies_ms):.2f} ms")
        print(f"  • Worker Inference P95   : {np.percentile(worker_latencies_ms, 95):.2f} ms")
        print(f"  • Queue Wait Time Mean   : {np.mean(queue_latencies_ms):.2f} ms")
        print(f"  • Max Queue Depth Seen   : {max(queue_depths) if queue_depths else 0}")
    else:
        print("  • No asynchronous recognition tasks triggered.")

    print(f"\nNOTE: {MAC_DISCLAIMER}\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="M2.6 Two-Rate Perception Pipeline Benchmark")
    parser.add_argument("--frames", type=int, default=150, help="Number of frames to benchmark")
    parser.add_argument(
        "--device",
        type=str,
        default="MacBook M4 Air",
        help="Device label for benchmark reporting",
    )
    args = parser.parse_args()
    return run_two_rate_benchmark(num_frames=args.frames, device_label=args.device)


if __name__ == "__main__":
    sys.exit(main())

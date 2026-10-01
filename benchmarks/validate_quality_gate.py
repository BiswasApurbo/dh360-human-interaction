#!/usr/bin/env python3
"""Quality Gate Validation and Pipeline Benchmark.

Exercises the FaceQualityGate across detection and tracking streams to evaluate
quality metrics (sharpness, pose frontality, bounding box scale, confidence) and
acceptance/rejection distributions.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

# Add src and project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_test_video import generate_face_animation_video  # noqa: E402

from dh360_hi.vision.benchmark import PipelineBenchmark, StageTimer  # noqa: E402
from dh360_hi.vision.detector import YuNetFaceDetector  # noqa: E402
from dh360_hi.vision.frame import FrameSource, VideoFileFrameSource  # noqa: E402
from dh360_hi.vision.quality_gate import (  # noqa: E402
    FaceQualityGate,
    FaceQualityThresholds,
)
from dh360_hi.vision.tracker import FaceTracker  # noqa: E402

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "weights" / "face_detection_yunet_2023mar.onnx"
DEFAULT_FIXTURE_PATH = PROJECT_ROOT / "tests" / "fixtures" / "sample_face.jpg"
DEFAULT_ANIMATED_VIDEO = PROJECT_ROOT / "data" / "face_quality_video.mp4"
MAC_DISCLAIMER = (
    "Development benchmark — MacBook M4 Air; not representative of Raspberry Pi performance."
)


def run_quality_gate_benchmark(
    source: FrameSource,
    detector: YuNetFaceDetector,
    tracker: FaceTracker,
    quality_gate: FaceQualityGate,
    max_frames: int = 100,
    device_label: str = "MacBook M4 Air",
) -> int:
    """Run full detection -> tracking -> quality gate pipeline benchmark."""
    bench = PipelineBenchmark()
    total_detections = 0
    total_evaluated = 0
    total_acceptable = 0
    rejection_counter: Counter[str] = Counter()
    sharpness_scores: list[float] = []

    print("\n================================================================================")
    print("DH360 HUMAN INTERACTION — VISION DETECTION + TRACKING + QUALITY GATE")
    print("================================================================================")

    with source:
        input_resolution = source.resolution
        bench.start()

        for frame in source:
            if bench.frame_count >= max_frames:
                break

            stage_times: dict[str, float] = {}

            # 1. Detection Stage
            with StageTimer() as det_timer:
                det_result = detector.detect(frame)
            stage_times["face_detection"] = det_timer.elapsed_ms

            # 2. Tracking Stage
            with StageTimer() as track_timer:
                track_result = tracker.update(det_result, frame_id=frame.frame_id)
            stage_times["face_tracking"] = track_timer.elapsed_ms

            # 3. Quality Gate Stage
            gate_latency = 0.0
            with StageTimer() as gate_timer:
                for track in track_result.tracks:
                    total_evaluated += 1
                    q_res = quality_gate.evaluate(frame, track)
                    sharpness_scores.append(q_res.metrics.sharpness)
                    if q_res.is_acceptable:
                        total_acceptable += 1
                    else:
                        for reason in q_res.rejection_reasons:
                            rejection_counter[reason.value] += 1
            gate_latency = gate_timer.elapsed_ms
            stage_times["quality_gate"] = gate_latency

            total_detections += len(det_result.detections)
            total_frame_latency = det_timer.elapsed_ms + track_timer.elapsed_ms + gate_latency
            confidences = [d.confidence for d in det_result.detections]

            bench.record_frame(
                total_latency_ms=total_frame_latency,
                stage_latencies_ms=stage_times,
                detections_count=len(det_result.detections),
                confidences=confidences,
            )

    det_stats = bench.get_stage_stats("face_detection")
    track_stats = bench.get_stage_stats("face_tracking")
    gate_stats = bench.get_stage_stats("quality_gate")

    accept_rate = (total_acceptable / max(total_evaluated, 1)) * 100.0
    notes = (
        f"Input Resolution:          {input_resolution[0]}x{input_resolution[1]}\n"
        f"Faces Evaluated:           {total_evaluated}\n"
        f"Quality Accepted:          {total_acceptable} ({accept_rate:.1f}%)\n"
        f"Quality Rejected:          {total_evaluated - total_acceptable}"
    )

    report = bench.format_summary(device_label=device_label, notes=notes)
    print("\n" + report)

    print("\n--- Quality Gate Summary ---")
    print(f"  • Faces Evaluated by Gate:   {total_evaluated}")
    print(f"  • Quality Passed (Accepted): {total_acceptable} ({accept_rate:.1f}%)")
    print(f"  • Quality Failed (Rejected): {total_evaluated - total_acceptable}")
    if rejection_counter:
        print("  • Rejection Reasons Breakdown:")
        for reason, count in rejection_counter.most_common():
            print(f"      - {reason:<20}: {count}")
    if sharpness_scores:
        import numpy as np

        arr = np.array(sharpness_scores)
        print(f"  • Sharpness (Mean/p50):      {np.mean(arr):.1f} / {np.percentile(arr, 50):.1f}")
        print(f"  • Sharpness (Min/Max):       {np.min(arr):.1f} / {np.max(arr):.1f}")

    print("\n--- Latency Breakdown (p50) ---")
    print(f"  • Face Detection:            {det_stats.p50_ms:.2f} ms")
    print(f"  • Face Tracking:             {track_stats.p50_ms:.2f} ms")
    print(f"  • Quality Gate Evaluation:   {gate_stats.p50_ms:.2f} ms")
    print(f"  • Overall Pipeline FPS:      {bench.average_fps:.2f} FPS")
    print(f"  • Performance Note:          {MAC_DISCLAIMER}\n")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Face Quality Gate validation benchmark")
    parser.add_argument("--frames", type=int, default=100, help="Number of frames (default: 100)")
    parser.add_argument(
        "--min-sharpness",
        type=float,
        default=50.0,
        help="Minimum Laplacian sharpness threshold (default: 50.0)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.80,
        help="Minimum detector confidence threshold (default: 0.80)",
    )
    args = parser.parse_args()

    # Ensure model and fixture exist
    if not DEFAULT_MODEL_PATH.exists() or not DEFAULT_FIXTURE_PATH.exists():
        from scripts.download_models import download_all_resources

        download_all_resources()

    generate_face_animation_video(
        face_image_path=DEFAULT_FIXTURE_PATH,
        output_path=DEFAULT_ANIMATED_VIDEO,
        target_width=640,
        target_height=480,
        fps=30.0,
        num_frames=args.frames,
    )

    source = VideoFileFrameSource(DEFAULT_ANIMATED_VIDEO)
    detector = YuNetFaceDetector(model_path=DEFAULT_MODEL_PATH, input_size=(640, 480))
    tracker = FaceTracker()
    thresholds = FaceQualityThresholds(
        min_sharpness=args.min_sharpness,
        min_confidence=args.min_confidence,
    )
    quality_gate = FaceQualityGate(thresholds=thresholds)

    return run_quality_gate_benchmark(
        source=source,
        detector=detector,
        tracker=tracker,
        quality_gate=quality_gate,
        max_frames=args.frames,
    )


if __name__ == "__main__":
    sys.exit(main())

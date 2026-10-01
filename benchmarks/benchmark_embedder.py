#!/usr/bin/env python3
"""Face Alignment and Embedding Model Benchmark Utility.

Measures preprocessing/alignment latency, embedding inference latency, total latency
distributions (p50, p95, p99), throughput (FPS), and vector dimensionality across
candidate pretrained face embedding models.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src and project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.download_models import download_all_resources  # noqa: E402
from scripts.generate_test_video import generate_face_animation_video  # noqa: E402

from dh360_hi.vision.aligner import FaceAligner  # noqa: E402
from dh360_hi.vision.benchmark import PipelineBenchmark, StageTimer  # noqa: E402
from dh360_hi.vision.detector import YuNetFaceDetector  # noqa: E402
from dh360_hi.vision.embedder import SFaceEmbedder  # noqa: E402
from dh360_hi.vision.frame import VideoFileFrameSource  # noqa: E402

MODELS_DIR = PROJECT_ROOT / "models" / "weights"
YUNET_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_FP32_PATH = MODELS_DIR / "face_recognition_sface_2021dec.onnx"
SFACE_INT8_PATH = MODELS_DIR / "face_recognition_sface_2021dec_int8.onnx"
FIXTURE_PATH = PROJECT_ROOT / "tests" / "fixtures" / "sample_face.jpg"
ANIMATED_VIDEO_PATH = PROJECT_ROOT / "data" / "face_embedding_benchmark.mp4"

MAC_DISCLAIMER = (
    "Development benchmark — MacBook M4 Air; not representative of Raspberry Pi performance."
)


def evaluate_embedder_candidate(
    model_name: str,
    embedder: SFaceEmbedder,
    aligner: FaceAligner,
    detector: YuNetFaceDetector,
    video_path: Path,
    num_frames: int = 100,
    device_label: str = "MacBook M4 Air",
) -> dict[str, float | str | int]:
    """Benchmark a single embedding model candidate."""
    bench = PipelineBenchmark()
    source = VideoFileFrameSource(video_path)
    file_size_bytes = embedder.model_path.stat().st_size
    file_size_mb = file_size_bytes / (1024 * 1024)

    total_faces_embedded = 0

    print("\n================================================================================")
    print(f"BENCHMARKING EMBEDDING CANDIDATE: {model_name.upper()}")
    print("================================================================================")
    print(f"  • Model File:            {embedder.model_path.name}")
    print(f"  • Model File Size:       {file_size_mb:.2f} MB ({file_size_bytes:,} bytes)")
    print(f"  • Embedding Dimension:   {embedder.dimension}-D")
    print(f"  • Canonical Input Size:  {embedder.input_size[0]}x{embedder.input_size[1]}")

    with source:
        bench.start()
        for frame in source:
            if bench.frame_count >= num_frames:
                break

            stage_times: dict[str, float] = {}

            # 1. Detection
            with StageTimer() as det_timer:
                det_result = detector.detect(frame)
            stage_times["detection"] = det_timer.elapsed_ms

            # 2. Alignment & Embedding for detected faces
            align_ms = 0.0
            embed_ms = 0.0

            for det in det_result.detections:
                with StageTimer() as a_timer:
                    align_res = aligner.align(frame, det)
                align_ms += a_timer.elapsed_ms

                with StageTimer() as e_timer:
                    emb_res = embedder.embed(align_res)
                embed_ms += e_timer.elapsed_ms

                if emb_res.is_normalized:
                    total_faces_embedded += 1

            stage_times["face_alignment"] = align_ms
            stage_times["embedding_inference"] = embed_ms

            total_latency = det_timer.elapsed_ms + align_ms + embed_ms
            bench.record_frame(
                total_latency_ms=total_latency,
                stage_latencies_ms=stage_times,
                detections_count=len(det_result.detections),
            )

    det_stats = bench.get_stage_stats("detection")
    align_stats = bench.get_stage_stats("face_alignment")
    embed_stats = bench.get_stage_stats("embedding_inference")

    notes = (
        f"Embedding Model:           {model_name}\n"
        f"Model File Size:           {file_size_mb:.2f} MB\n"
        f"Embedding Dimension:       {embedder.dimension}-D\n"
        f"Total Faces Embedded:      {total_faces_embedded}"
    )

    report = bench.format_summary(device_label=device_label, notes=notes)
    print("\n" + report)

    print("\n--- Embedding Latency Breakdown ---")
    print(f"  • Alignment Stage (p50):     {align_stats.p50_ms:.2f} ms")
    print(f"  • Embedding Inference (p50): {embed_stats.p50_ms:.2f} ms")
    print(f"  • Alignment + Embedding p50: {align_stats.p50_ms + embed_stats.p50_ms:.2f} ms")
    print(f"  • Alignment + Embedding p95: {align_stats.p95_ms + embed_stats.p95_ms:.2f} ms")
    print(f"  • Alignment + Embedding p99: {align_stats.p99_ms + embed_stats.p99_ms:.2f} ms")
    print(f"  • Pipeline Throughput:       {bench.average_fps:.2f} FPS")
    print(f"  • Performance Note:          {MAC_DISCLAIMER}\n")

    return {
        "model_name": model_name,
        "file_size_mb": file_size_mb,
        "dimension": embedder.dimension,
        "align_p50_ms": align_stats.p50_ms,
        "align_mean_ms": align_stats.mean_ms,
        "embed_p50_ms": embed_stats.p50_ms,
        "embed_mean_ms": embed_stats.mean_ms,
        "embed_p95_ms": embed_stats.p95_ms,
        "embed_p99_ms": embed_stats.p99_ms,
        "fps": bench.average_fps,
        "det_p50_ms": det_stats.p50_ms,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Face Embedding Candidates")
    parser.add_argument(
        "--model",
        choices=["sface", "sface_int8", "all"],
        default="all",
        help="Model candidate to benchmark (default: all)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=100,
        help="Number of benchmark frames (default: 100)",
    )
    args = parser.parse_args()

    # Ensure required models and fixtures exist
    download_all_resources()

    generate_face_animation_video(
        face_image_path=FIXTURE_PATH,
        output_path=ANIMATED_VIDEO_PATH,
        target_width=640,
        target_height=480,
        fps=30.0,
        num_frames=args.frames,
    )

    detector = YuNetFaceDetector(model_path=YUNET_PATH, input_size=(640, 480))
    aligner = FaceAligner(output_size=(112, 112))

    candidates: list[tuple[str, Path]] = []
    if args.model in ("sface", "all"):
        candidates.append(("SFace FP32", SFACE_FP32_PATH))
    if args.model in ("sface_int8", "all"):
        candidates.append(("SFace INT8", SFACE_INT8_PATH))

    results = []
    for label, path in candidates:
        embedder = SFaceEmbedder(model_path=path, model_name=label)
        res = evaluate_embedder_candidate(
            model_name=label,
            embedder=embedder,
            aligner=aligner,
            detector=detector,
            video_path=ANIMATED_VIDEO_PATH,
            num_frames=args.frames,
        )
        results.append(res)

    print("\n================================================================================")
    print("EMBEDDING MODEL CANDIDATE COMPARISON SUMMARY")
    print("================================================================================")
    header = (
        f"{'Model':<14} | {'Size (MB)':<10} | {'Dim':<5} | "
        f"{'Align p50':<10} | {'Infer p50':<10} | {'Infer p95':<10} | {'Pipeline FPS':<12}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        row = (
            f"{r['model_name']:<14} | "
            f"{r['file_size_mb']:<10.2f} | "
            f"{r['dimension']:<5} | "
            f"{r['align_p50_ms']:<10.2f} | "
            f"{r['embed_p50_ms']:<10.2f} | "
            f"{r['embed_p95_ms']:<10.2f} | "
            f"{r['fps']:<12.2f}"
        )
        print(row)
    print("================================================================================")
    print(f"Disclaimer: {MAC_DISCLAIMER}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Unit tests for benchmarking and metric calculations."""

import time

from dh360_hi.vision.benchmark import PipelineBenchmark, StageTimer

MAC_DISCLAIMER = (
    "Development benchmark — MacBook M4 Air; not representative of Raspberry Pi performance."
)


def test_stage_timer() -> None:
    """Verify StageTimer correctly measures duration."""
    with StageTimer() as timer:
        time.sleep(0.01)

    assert timer.elapsed_ms >= 5.0  # At least 5ms given sleep(0.01)


def test_pipeline_benchmark_calculations() -> None:
    """Verify statistical calculations (p50, p95, p99, average FPS, rolling FPS)."""
    bench = PipelineBenchmark(window_size=5)
    bench.start()

    latencies = [10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0]
    for lat in latencies:
        bench.record_frame(
            total_latency_ms=lat,
            stage_latencies_ms={"detector": lat * 0.8, "transport": lat * 0.2},
        )

    assert bench.frame_count == 10

    overall = bench.get_overall_stats()
    assert overall.count == 10
    assert overall.min_ms == 10.0
    assert overall.max_ms == 28.0
    assert overall.p50_ms == 19.0
    assert overall.p95_ms > 24.0
    assert overall.p99_ms > 26.0

    det_stats = bench.get_stage_stats("detector")
    assert det_stats.count == 10
    assert round(det_stats.min_ms, 1) == 8.0
    assert round(det_stats.max_ms, 1) == 22.4


def test_empty_benchmark_stats() -> None:
    """Verify stats handling when no frames have been recorded."""
    bench = PipelineBenchmark()
    stats = bench.get_overall_stats()
    assert stats.count == 0
    assert stats.mean_ms == 0.0
    assert stats.p50_ms == 0.0
    assert stats.average_fps == 0.0
    assert bench.average_fps == 0.0
    assert bench.rolling_fps == 0.0


def test_format_summary_contains_mac_disclaimer() -> None:
    """Verify benchmark summary report includes the required platform disclaimer."""
    bench = PipelineBenchmark()
    bench.start()
    bench.record_frame(total_latency_ms=15.0, stage_latencies_ms={"face_detection": 15.0})

    summary = bench.format_summary(device_label="MacBook M4 Air")
    assert MAC_DISCLAIMER in summary
    assert "face_detection" in summary
    assert "Frames Processed:" in summary
